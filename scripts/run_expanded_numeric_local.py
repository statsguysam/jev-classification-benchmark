#!/usr/bin/env python3
"""Plan or run two pinned open-model families on frozen numerical datasets.

Default is a local plan. --allow-download authorizes public snapshot caching;
--execute authorizes serial inference. Classification delegates to the frozen
runner and likelihood scorer. Only chat rendering changes; cache/authentication
controls force pinned local snapshots and never use an implicit HF token.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from jevbench import providers
from jevbench.metrics import evaluate
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import _identity, digest, environment, run_model, save_json
from jevbench.types import Prediction
from run_tabular_classical import validate_native
from run_tabular_local import exclusive_device_lock
from tabular_data import load_native_prepared

PRESETS = ROOT / "configs/expanded_numeric_models.json"
DATASETS = ("breast_cancer", "wine")
MODEL_KEYS = ("smollm2", "granite")
FROZEN_CORE_SHA = "d5547ccde4224e182653315d81c0a631c789bbe294ad7bcf95ef0cddd25ce608"
RENDERER_VERSION = "fixed-system-no-thinking-v1"
SYSTEM_MESSAGE = "You are a helpful assistant."
PUBLIC_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "chat_template.jinja", "*.safetensors", "*.safetensors.index.json",
                "vocab.json", "merges.txt", "added_tokens.json"]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_presets():
    data = json.loads(PRESETS.read_text())
    require(data["renderer_version"] == RENDERER_VERSION and data["system_message"] == SYSTEM_MESSAGE
            and data["core_sha256"] == FROZEN_CORE_SHA == environment()["source_sha256"], "Frozen core or renderer differs")
    require(set(data["models"]) == set(MODEL_KEYS), "Unexpected model families")
    for preset in data["models"].values():
        require(preset["provider"] == "hf" and re.fullmatch("[0-9a-f]{40}", preset["revision"])
                and preset["dtype"] == "float16" and preset["max_context_tokens"] == 8192
                and preset["thinking"] is False and isinstance(preset["chat_template"], str), "Unexpected model preset")
    return data


def model_config(key, device, presets=None):
    require(key in MODEL_KEYS and device in ("mps", "cuda"), "Only explicit MPS/CUDA devices and approved model families are permitted")
    data = presets or load_presets()
    preset = data["models"][key]
    return {"provider": "hf", "model": preset["model"], "revision": preset["revision"],
        "device": device, "dtype": "float16", "max_context_tokens": 8192,
        "renderer": {"version": RENDERER_VERSION, "model_key": key, "system_message": SYSTEM_MESSAGE,
            "thinking": False, "chat_template_sha256": text_sha(preset["chat_template"]),
            "tokenizer_files_sha256": preset["tokenizer_files_sha256"],
            "tokenizer_fingerprint_sha256": digest(preset["tokenizer_files_sha256"]),
            "helper_sha256": file_sha(__file__), "preset_file_sha256": file_sha(PRESETS),
            "core_sha256": FROZEN_CORE_SHA}}


def messages(prompt):
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}]


def render_chat(preset, prompt):
    # Pure template rendering enables source audits without model/tokenizer loads.
    from transformers.utils.chat_template_utils import _compile_jinja_template
    return _compile_jinja_template(preset["chat_template"]).render(messages=messages(prompt),
        add_generation_prompt=True, thinking=False, **preset["tokenizer_special_tokens"])


def render_tokens(tokenizer, preset, prompt):
    require(tokenizer.chat_template == preset["chat_template"], "Loaded chat template differs from pinned template")
    rendered = tokenizer.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True, thinking=False)
    require(rendered == render_chat(preset, prompt), "Loaded tokenizer chat rendering differs from audited rendering")
    ids = tokenizer.apply_chat_template(messages(prompt), tokenize=True, add_generation_prompt=True, thinking=False)
    require(isinstance(ids, list) and ids and all(type(token) is int and token >= 0 for token in ids), "Invalid rendered token IDs")
    return ids, {"renderer_version": RENDERER_VERSION, "rendered_chat_prompt_sha256": text_sha(rendered),
                 "rendered_input_ids_sha256": digest(ids), "rendered_input_tokens": len(ids),
                 "chat_template_sha256": text_sha(preset["chat_template"]),
                 "tokenizer_fingerprint_sha256": digest(preset["tokenizer_files_sha256"])}


def cached_snapshot(preset, cache):
    return Path(cache) / ("models--" + preset["model"].replace("/", "--")) / "snapshots" / preset["revision"]


def verify_tokenizer_assets(preset, cache):
    snapshot = cached_snapshot(preset, cache)
    for name, expected in preset["tokenizer_files_sha256"].items():
        path = snapshot / name
        require(path.is_file() and file_sha(path) == expected, "Missing or changed pinned tokenizer asset: " + name)
    config = json.loads((snapshot / "config.json").read_text())
    require(config["model_type"] == preset["model_type"] and config["max_position_embeddings"] == preset["model_context_limit"],
            "Cached architecture/context differs")
    return snapshot


def download_public(preset, cache):
    from huggingface_hub import snapshot_download
    snapshot_download(preset["model"], revision=preset["revision"], cache_dir=cache, token=False,
                      allow_patterns=PUBLIC_FILES, max_workers=2)
    verify_tokenizer_assets(preset, cache)


class FixedChatClassifier(providers.HuggingFaceClassifier):
    """Frozen likelihood scoring with a deterministic, non-thinking chat wrapper."""
    def __init__(self, config, preset, cache):
        super().__init__(config)
        self.preset, self.cache = preset, Path(cache)

    def _load(self):
        if self._model is not None:
            return
        verify_tokenizer_assets(self.preset, self.cache)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        original_tokenizer, original_model = AutoTokenizer.from_pretrained, AutoModelForCausalLM.from_pretrained

        def local(loader):
            def wrapped(*args, **kwargs):
                kwargs.update(token=False, cache_dir=str(self.cache), local_files_only=True)
                return loader(*args, **kwargs)
            return wrapped

        # The inherited loader still selects architecture, precision, revision,
        # device and eval mode; these patches only enforce cache/auth controls.
        with patch.object(AutoTokenizer, "from_pretrained", side_effect=local(original_tokenizer)), \
             patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=local(original_model)):
            super()._load()
        require(self.metadata["resolved_revision"] == self.config["revision"], "Loaded revision differs from immutable pin")
        require(self._model.config.model_type == self.preset["model_type"] and
                self._model.dtype == self._torch.float16, "Unexpected loaded architecture or precision")

    def _prompt_ids(self, tokenizer, prompt):
        require(tokenizer is self._tokenizer, "Unexpected tokenizer instance")
        ids, metadata = render_tokens(tokenizer, self.preset, prompt)
        self.metadata.update(metadata)
        return ids

    def predict(self, row, labels, prompt):
        # Do not catch CUDA/MPS OOM or change device/dtype. Core scoring is intact.
        with patch.object(providers, "prompt_token_ids", side_effect=self._prompt_ids):
            return super().predict(row, labels, prompt)


def preflight(provider, jobs):
    provider._load()
    checks = []
    for dataset, shots in jobs:
        examples = select_examples(dataset.train, dataset.labels, shots, 42)
        candidate_lengths = [len(providers.response_token_ids(provider._tokenizer, i)) for i in range(len(dataset.labels))]
        lengths = []
        for row in dataset.test:
            ids, _ = render_tokens(provider._tokenizer, provider.preset, build_prompt(row, dataset.labels, examples))
            length = len(ids) + max(candidate_lengths)
            require(length <= min(8192, provider._model.config.max_position_embeddings),
                    f"Full prompt plus label exceeds context limit in {dataset.name}; no truncation")
            lengths.append(length)
        checks.append({"dataset": dataset.name, "shots_per_class": shots, "n_test": len(dataset.test),
                       "max_prompt_plus_label_tokens": max(lengths), "candidate_token_lengths": candidate_lengths,
                       "test_ids_sha256": digest([row.id for row in dataset.test])})
    return checks


def audit_source_run(path, dataset, expected_key, shots):
    """Read-only source verification for subsequent Jev review; no model loading."""
    path = Path(path)
    record = json.loads(path.read_text())
    data = load_presets()
    require(expected_key in MODEL_KEYS and dataset.name in DATASETS and shots in (0, 4), "Unexpected source condition")
    config = model_config(expected_key, record["config"]["device"], data)
    examples = select_examples(dataset.train, dataset.labels, shots, 42)
    identity = _identity(dataset, "zero_shot" if shots == 0 else "few_shot", {**config, "shots_per_class": shots}, 42, examples)
    identity["bootstrap_samples"] = record["bootstrap_samples"]
    require(record["bootstrap_samples"] >= 100 and all(record.get(k) == v for k, v in identity.items()), "Expanded source identity differs")
    expected_run = f"{dataset.name}__{config['model'].split('/')[-1]}__{digest(identity)[:12]}"
    require(record["status"] == "complete" and record["run_id"] == path.parent.name == expected_run, "Incomplete or unexpected source run")
    require(record["dataset_manifest"] == dataset.manifest and record["environment"]["source_sha256"] == FROZEN_CORE_SHA,
            "Source prepared/core identity differs")
    manifest = {"dataset": dataset.name, "labels": dataset.labels, "manifest_sha256": digest(dataset.manifest),
        "rows": [{"id": row.id, "label": row.label, "text_sha256": text_sha(row.text)} for row in dataset.test]}
    require(json.loads(path.with_name("test_manifest.json").read_text()) == manifest, "Source test manifest differs")
    predictions = [Prediction(**json.loads(line)) for line in path.with_name("predictions.jsonl").read_text().splitlines()]
    require([p.row_id for p in predictions] == [row.id for row in dataset.test], "Source prediction coverage/order differs")
    preset = data["models"][expected_key]
    for row, prediction in zip(dataset.test, predictions):
        expected = {"requested_model": config["model"], "requested_revision": config["revision"],
            "resolved_revision": config["revision"], "device": config["device"], "dtype": "float16",
            "probability_kind": "label_sequence_likelihood_normalized", "scoring": "sum_logp_numeric_id_plus_eos",
            "renderer_version": RENDERER_VERSION, "chat_template_sha256": text_sha(preset["chat_template"]),
            "tokenizer_fingerprint_sha256": digest(preset["tokenizer_files_sha256"]),
            "rendered_chat_prompt_sha256": text_sha(render_chat(preset, build_prompt(row, dataset.labels, examples)))}
        require(all(prediction.metadata.get(k) == v for k, v in expected.items()), "Source renderer/model provenance differs")
        require(re.fullmatch("[0-9a-f]{64}", prediction.metadata.get("rendered_input_ids_sha256", ""))
                and type(prediction.metadata.get("rendered_input_tokens")) is int
                and 0 < prediction.metadata["rendered_input_tokens"] < 8192, "Invalid source token identity/length")
        if prediction.error is None:
            require(type(prediction.label) is int and 0 <= prediction.label < len(dataset.labels)
                    and prediction.probabilities is not None and len(prediction.probabilities) == len(dataset.labels)
                    and prediction.label == max(range(len(dataset.labels)), key=prediction.probabilities.__getitem__)
                    and prediction.input_tokens == prediction.metadata["rendered_input_tokens"]
                    and prediction.output_tokens == 0 and prediction.metadata["context_forward_passes"] == len(dataset.labels),
                    "Source classification contract differs")
    measured = evaluate(dataset.test, predictions, len(dataset.labels))
    require(all(record["metrics"].get(k) == v for k, v in measured.items()), "Source metrics differ from saved predictions")
    return record, predictions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("mps", "cuda"), default="mps")
    p.add_argument("--model-keys", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    p.add_argument("--shots", nargs="+", type=int, choices=(0, 4), default=[0, 4])
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--data-root", type=Path, default=ROOT / "data/tabular-full")
    p.add_argument("--output", type=Path, default=ROOT / "results/numeric_expansion/local")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/hf")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="Load/check selected models and full prompt lengths; no predictions")
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.bootstrap_samples >= 100, "At least 100 bootstrap samples required")
    require(all(len(values) == len(set(values)) for values in (args.model_keys, args.shots, args.datasets)), "Duplicate condition arguments")
    data = load_presets()
    jobs = []
    for name in args.datasets:
        dataset, native = load_native_prepared(args.data_root / name)
        validate_native(dataset, native)
        require(dataset.name == name and all(f["kind"] == "numeric" for f in dataset.manifest["tabular"]["features"]),
                "Only the frozen strictly numerical datasets are permitted")
        jobs.extend((dataset, shots) for shots in args.shots)
    plan = {"status": "plan", "new_hosted_calls": 0, "adapter_training": False,
        "device": args.device, "dtype": "float16", "network_downloads_authorized": args.allow_download,
        "models": {key: model_config(key, args.device, data) for key in args.model_keys},
        "jobs": [{"dataset": d.name, "shots_per_class": s, "n_test": len(d.test), "manifest_sha256": digest(d.manifest),
                  "training_example_ids": [r.id for r in select_examples(d.train, d.labels, s, 42)]} for d, s in jobs],
        "prediction_rows": sum(len(d.test) for d, _ in jobs) * len(args.model_keys)}
    print(json.dumps(plan, indent=2), flush=True)
    if args.allow_download:
        for key in args.model_keys:
            download_public(data["models"][key], args.cache_dir)
    if not args.execute and not args.preflight_only:
        return 0
    import torch
    require((args.device == "mps" and torch.backends.mps.is_available()) or
            (args.device == "cuda" and torch.cuda.is_available()), "Requested device unavailable; no fallback")
    with exclusive_device_lock(ROOT / "artifacts/tabular-neural.lock"):
        for key in args.model_keys:
            config = plan["models"][key]
            provider = FixedChatClassifier(config, data["models"][key], args.cache_dir)
            try:
                checks = preflight(provider, jobs)
                save_json(args.output / "plans" / f"{key}-{args.device}-preflight.json", {**plan, "status": "preflight_complete", "checks": checks})
                print(json.dumps({"model_key": key, "preflight": checks}), flush=True)
                if args.execute:
                    for dataset, shots in jobs:
                        with patch.object(providers, "build_provider", return_value=provider):
                            record = run_model(dataset, config, args.output, shots=shots, seed=42,
                                allow_paid=False, max_requests=len(dataset.test), bootstrap_samples=args.bootstrap_samples)
                        path = args.output / record["run_id"] / "run.json"
                        audit_source_run(path, dataset, key, shots)
                        print(json.dumps({"run_id": record["run_id"], "accuracy": record["metrics"]["accuracy"],
                                          "n_failures": record["metrics"]["n_failures"]}), flush=True)
            finally:
                provider._model = provider._tokenizer = provider._torch = None
                del provider
                gc.collect()
                if args.device == "mps":
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
