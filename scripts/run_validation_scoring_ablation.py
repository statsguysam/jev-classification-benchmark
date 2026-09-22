#!/usr/bin/env python3
"""Frozen validation-only ID versus ID+EOS diagnostic; public cached MPS only.

Default plans without loading tokenizers/models. --freeze binds all IDs, code,
configs and rendered tokens before inference. --execute requires that protocol.
Both scores use the same forward passes. No generation, training or API calls.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_expanded_numeric_local as frozen
from jevbench.data import normalized_text
from jevbench.metrics import evaluate
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import digest, save_json
from jevbench.types import Prediction
from tabular_data import load_native_prepared
from run_tabular_classical import validate_native

CONFIG = ROOT / "configs/review_value_scoring_ablation.json"
OUTPUT = ROOT / "results/review_value/validation_scoring"
HELPER_SHA = "b0c07e82e03b79040f13bd7810baf92bcd3a30fe8aa3b48d0704baedea8801a3"
PRESET_SHA = "ed176a77f40fd8cd842672193333559a8221b3ac9dd52259d275c54da4cf6421"
require, file_sha = frozen.require, frozen.file_sha


def load_inputs():
    require(file_sha(frozen.__file__) == HELPER_SHA and file_sha(frozen.PRESETS) == PRESET_SHA,
            "Frozen renderer/preset source differs")
    presets = frozen.load_presets()
    config = json.loads(CONFIG.read_text())
    expected = {"schema_version": 1, "seed": 42, "model_keys": ["smollm2", "granite"],
        "shots_per_class": [0, 4], "device": "mps", "dtype": "float16", "max_context_tokens": 8192,
        "arms": ["label_only", "label_plus_eos"], "free_generation": False, "adapters": False,
        "downloads": False, "hosted_calls": False}
    require(all(config.get(k) == v for k, v in expected.items()) and
            set(config["datasets"]) == {"breast_cancer", "wine"}, "Diagnostic recipe differs")
    datasets, samples, examples = {}, {}, {}
    for name, count in (("breast_cancer", 8), ("wine", 6)):
        dataset, native = load_native_prepared(ROOT / "data/tabular-full" / name)
        validate_native(dataset, native)
        spec = config["datasets"][name]
        require(digest(dataset.manifest) == spec["manifest_sha256"] and spec["validation_per_class"] == count,
                "Frozen diagnostic dataset differs")
        selected = select_examples(dataset.validation, dataset.labels, count, 42)
        demos = select_examples(dataset.train, dataset.labels, 4, 42)
        require([r.id for r in selected] == spec["validation_ids"] and
                [r.id for r in demos] == spec["training_example_ids_k4"], "Frozen diagnostic IDs differ")
        require(not {r.id for r in selected} & {r.id for r in dataset.train + dataset.test}, "Diagnostic IDs leak across splits")
        require(not {normalized_text(r.text) for r in selected} &
                {normalized_text(r.text) for r in dataset.train + dataset.test}, "Diagnostic content leaks across splits")
        datasets[name], samples[name], examples[name] = dataset, selected, demos
    return config, presets, datasets, samples, examples


def condition_key(model_key, dataset, shots, row_id):
    return digest([model_key, dataset, shots, row_id])


def build_protocol(config, presets, datasets, samples, examples):
    """Tokenizer-only offline preflight; model weights are not constructed."""
    from transformers import AutoTokenizer
    contexts, models = [], {}
    for key in config["model_keys"]:
        preset = presets["models"][key]
        frozen.verify_tokenizer_assets(preset, ROOT / "data/hf")
        tokenizer = AutoTokenizer.from_pretrained(preset["model"], revision=preset["revision"],
            cache_dir=ROOT / "data/hf", local_files_only=True, token=False, trust_remote_code=False)
        models[key] = frozen.model_config(key, "mps", presets)
        for name, dataset in datasets.items():
            candidates = [frozen.providers.response_token_ids(tokenizer, i) for i in range(len(dataset.labels))]
            require(all(len(ids) == 2 and ids[-1] == tokenizer.eos_token_id for ids in candidates),
                    "Diagnostic requires one numeric-ID token plus the pinned EOS")
            for shots in config["shots_per_class"]:
                demos = examples[name] if shots else []
                for row in samples[name]:
                    ids, metadata = frozen.render_tokens(tokenizer, preset, build_prompt(row, dataset.labels, demos))
                    require(len(ids) + 2 <= 8192, "Diagnostic context overflow; no truncation")
                    contexts.append({"key": condition_key(key, name, shots, row.id), "model_key": key,
                        "dataset": name, "shots_per_class": shots, "row_id": row.id,
                        "candidate_token_ids": candidates, **metadata})
        del tokenizer
    source_paths = [Path(__file__), frozen.PRESETS, Path(frozen.__file__), CONFIG,
                    ROOT / "scripts/tabular_data.py", ROOT / "scripts/run_tabular_classical.py"]
    source_paths.extend(sorted((ROOT / "src/jevbench").glob("*.py")))
    return {"schema_version": 1, "config": config, "models": models, "contexts": contexts,
        "source_sha256": {p.relative_to(ROOT).as_posix(): file_sha(p) for p in source_paths},
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "huggingface-hub", "numpy")},
        "frozen_core_sha256": frozen.FROZEN_CORE_SHA,
        "validation_content_sha256": {name: digest([asdict(r) for r in samples[name]]) for name in datasets},
        "all_validation_ids_sha256": {name: digest([r.id for r in ds.validation]) for name, ds in datasets.items()},
        "test_predictions_read_or_changed": False, "n_distinct_validation_rows": sum(map(len, samples.values())),
        "n_model_contexts": len(contexts), "n_paired_decisions": 2 * len(contexts),
        "free_generation": "explicitly omitted from this bounded diagnostic",
        "purpose": "Explain scoring changes on development examples; no test rescore, method selection or gate fitting."}


def normalize(scores):
    require(scores and all(math.isfinite(v) for v in scores), "Nonfinite diagnostic likelihood")
    peak = max(scores)
    weights = [math.exp(v - peak) for v in scores]
    total = sum(weights)
    return [v / total for v in weights]


def score_context(provider, context_ids, candidates):
    """Exactly one forward pass per class; no row or hidden truth is accessed."""
    torch = provider._torch
    token_scores, label_scores, eos_scores, joint_scores = [], [], [], []
    with torch.inference_mode():
        for candidate in candidates:
            require(len(candidate) == 2, "Diagnostic candidate must be ID then EOS")
            ids = torch.tensor([context_ids + candidate], dtype=torch.long, device=provider._device)
            logits = provider._model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            log_probs = torch.log_softmax(logits[0, len(context_ids) - 1:-1].float(), dim=-1)
            scored = log_probs.gather(1, ids[0, len(context_ids):].unsqueeze(1)).squeeze(1)
            require(scored.numel() == 2 and bool(torch.isfinite(scored).all()), "Nonfinite token contribution")
            label_scores.append(scored[0].item()); eos_scores.append(scored[1].item())
            joint_scores.append(scored.sum().item())
            token_scores.append(scored.tolist())
    label_probs, joint_probs = normalize(label_scores), normalize(joint_scores)
    return {"per_candidate_token_logp": token_scores, "label_logp": label_scores,
        "conditional_eos_logp": eos_scores, "joint_logp": joint_scores,
        "label_only": {"label": max(range(len(candidates)), key=label_probs.__getitem__), "probabilities": label_probs},
        "label_plus_eos": {"label": max(range(len(candidates)), key=joint_probs.__getitem__), "probabilities": joint_probs},
        "context_forward_passes": len(candidates)}


def audit_prediction(prediction, context, protocol_sha):
    require(prediction["context"] == context and prediction["protocol_sha256"] == protocol_sha,
            "Saved diagnostic context/protocol differs")
    count = len(context["candidate_token_ids"])
    for field in ("label_logp", "conditional_eos_logp", "joint_logp", "per_candidate_token_logp"):
        require(len(prediction[field]) == count, "Diagnostic class count differs")
    require(prediction["context_forward_passes"] == count, "Diagnostic forward-pass count differs")
    for i in range(count):
        token = prediction["per_candidate_token_logp"][i]
        label, eos, joint = (prediction[field][i] for field in ("label_logp", "conditional_eos_logp", "joint_logp"))
        require(len(token) == 2 and token == [label, eos] and all(math.isfinite(v) and v <= 0 for v in (label, eos, joint)),
                "Malformed token contribution")
        require(math.isclose(label + eos, joint, abs_tol=2e-5, rel_tol=1e-6), "Label/EOS contributions do not reconstruct joint score")
    for arm, field in (("label_only", "label_logp"), ("label_plus_eos", "joint_logp")):
        expected = normalize(prediction[field])
        require(prediction[arm] == {"label": max(range(count), key=expected.__getitem__), "probabilities": expected},
                "Diagnostic score normalization/argmax differs")


def read_checkpoint(path, protocol, protocol_sha):
    if not path.exists():
        return []
    raw = path.read_text()
    require(not raw or raw.endswith("\n"), "Partial diagnostic checkpoint; inspect, do not silently truncate")
    values = [json.loads(line) for line in raw.splitlines()]
    require(len(values) <= len(protocol["contexts"]), "Too many diagnostic rows")
    for value, context in zip(values, protocol["contexts"]):
        audit_prediction(value, context, protocol_sha)
    return values


def summarize(values, datasets, samples):
    summaries = []
    for key in ("smollm2", "granite"):
        for name, dataset in datasets.items():
            for shots in (0, 4):
                rows = [v for v in values if (v["context"]["model_key"], v["context"]["dataset"], v["context"]["shots_per_class"]) == (key, name, shots)]
                require([v["context"]["row_id"] for v in rows] == [r.id for r in samples[name]], "Incomplete diagnostic condition")
                metrics, counts = {}, {}
                for arm in ("label_only", "label_plus_eos"):
                    pred = [Prediction(v["context"]["row_id"], v[arm]["label"], v[arm]["probabilities"]) for v in rows]
                    metrics[arm] = evaluate(samples[name], pred, len(dataset.labels))
                    counts[arm] = [sum(p.label == i for p in pred) for i in range(len(dataset.labels))]
                changed = sum(v["label_only"]["label"] != v["label_plus_eos"]["label"] for v in rows)
                fixed = sum(v["label_only"]["label"] == r.label and v["label_plus_eos"]["label"] != r.label for v, r in zip(rows, samples[name]))
                harmed = sum(v["label_only"]["label"] != r.label and v["label_plus_eos"]["label"] == r.label for v, r in zip(rows, samples[name]))
                summaries.append({"model_key": key, "dataset": name, "shots_per_class": shots,
                    "n_validation": len(rows), "metrics": metrics, "predicted_class_counts": counts,
                    "class_decisions_changed": changed, "label_only_corrected_joint_errors": fixed,
                    "label_only_harmed_joint_correct": harmed})
    return summaries


def write_report(output, protocol, protocol_sha, values, datasets, samples):
    summary = {"status": "complete", "protocol_sha256": protocol_sha, "n_contexts": len(values),
               "prediction_sha256": file_sha(output / "predictions.jsonl"), "conditions": summarize(values, datasets, samples),
               "test_results_changed": False, "method_selected": False, "confidence_gate_fitted": False}
    save_json(output / "SUMMARY.json", summary)
    lines = ["# Validation scoring diagnostic: numeric ID versus ID + EOS", "",
        f"Completed {len(values)} model contexts on 34 distinct validation rows. Both arms use the same forward pass per candidate. The recipe, row IDs, demonstration IDs, code, tokenizers and rendered contexts were frozen before inference. No test predictions were read or changed, no adapter was trained, and no API or download was used.", "",
        "| Dataset | Model | Examples/class | Validation n | ID+EOS accuracy | ID-only accuracy | Class changes | Fixed / harmed by ID-only |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary["conditions"]:
        lines.append(f"| {item['dataset']} | {item['model_key']} | {item['shots_per_class']} | {item['n_validation']} | {item['metrics']['label_plus_eos']['accuracy']:.1%} | {item['metrics']['label_only']['accuracy']:.1%} | {item['class_decisions_changed']} | {item['label_only_corrected_joint_errors']} / {item['label_only_harmed_joint_correct']} |")
    lines.extend(["", "These small balanced samples were selected only from the existing validation split using the unchanged selector and seed42 (8/class Breast Cancer, 6/class Wine). The original training-only four-per-class examples are unchanged. Zero-shot has no examples. Model families are SmolLM2 and Granite, each with the exact frozen fixed-system, thinking-disabled chat renderer, cached pinned revision, float16 MPS and 8,192-token ceiling.", "",
        "Each record saves the numeric-ID log probability, conditional EOS log probability, joint log probability and both normalized vectors. The joint score is the frozen core's float32 sum; small floating-point differences between its rounded sum and adding recorded Python floats are audited within declared tolerance. ID-only omits the EOS contribution and is therefore a distinct output-scoring event. No architecture comparison is isolated by this within-context diagnostic.", "",
        "This is exploratory development evidence after earlier test behavior was observed. No confidence gate or winning method is selected, and no historical score is replaced. The 34 validation labels are an added diagnostic label budget; repeating them across models/shots does not make them independent test examples. The balanced samples change natural class prevalence, so their accuracies are not directly comparable to previous test accuracy. Confirm any recipe on new held-out data before deployment or general claims. Greedy free generation was explicitly omitted before execution and remains a separately planned ablation."])
    (output / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config, presets, datasets, samples, examples = load_inputs()
    print(json.dumps({"status": "plan", "validation_rows": sum(map(len, samples.values())), "contexts": 136,
                      "paired_decisions": 272, "device": "mps", "free_generation": False}), flush=True)
    if not args.freeze and not args.execute:
        return
    protocol = build_protocol(config, presets, datasets, samples, examples)
    path = args.output / "protocol.json"
    if args.freeze and not path.exists():
        args.output.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
    require(path.is_file() and json.loads(path.read_text()) == protocol, "Protocol absent or changed; refuse inference")
    protocol_sha = file_sha(path)
    print(json.dumps({"status": "protocol_verified", "protocol_sha256": protocol_sha, "contexts": len(protocol["contexts"])}), flush=True)
    if not args.execute:
        return
    import torch
    require(torch.backends.mps.is_available(), "MPS unavailable; no device fallback")
    torch.manual_seed(42)
    with frozen.exclusive_device_lock(ROOT / "artifacts/tabular-neural.lock"):
        prediction_path = args.output / "predictions.jsonl"
        saved = read_checkpoint(prediction_path, protocol, protocol_sha)
        run_path = args.output / "run.json"
        run = {"status": "running", "protocol_sha256": protocol_sha, "device": "mps", "dtype": "float16",
               "torch_version": torch.__version__, "n_completed": len(saved), "n_expected": len(protocol["contexts"])}
        if run_path.exists():
            require(json.loads(run_path.read_text())["protocol_sha256"] == protocol_sha, "Run protocol differs")
        save_json(run_path, run)
        try:
            for key in config["model_keys"]:
                pending = [c for c in protocol["contexts"][len(saved):] if c["model_key"] == key]
                if not pending:
                    continue
                provider = frozen.FixedChatClassifier(protocol["models"][key], presets["models"][key], ROOT / "data/hf")
                try:
                    provider._load()
                    for context in pending:
                        name, shots = context["dataset"], context["shots_per_class"]
                        row = next(r for r in samples[name] if r.id == context["row_id"])
                        ids, metadata = frozen.render_tokens(provider._tokenizer, provider.preset,
                            build_prompt(row, datasets[name].labels, examples[name] if shots else []))
                        require(all(context.get(k) == v for k, v in metadata.items()), "Rendered context differs from preflight")
                        value = {"context": context, "protocol_sha256": protocol_sha,
                                 **score_context(provider, ids, context["candidate_token_ids"])}
                        audit_prediction(value, context, protocol_sha)
                        with prediction_path.open("a") as stream:
                            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
                            stream.flush(); os.fsync(stream.fileno())
                        saved.append(value)
                        print(json.dumps({"completed": len(saved), "total": len(protocol["contexts"]),
                                          "model_key": key, "dataset": name, "shots": shots}), flush=True)
                finally:
                    provider._model = provider._tokenizer = provider._torch = None
                    del provider
                    gc.collect(); torch.mps.empty_cache()
            require(len(saved) == len(protocol["contexts"]), "Incomplete diagnostic")
            write_report(args.output, protocol, protocol_sha, saved, datasets, samples)
            run.update(status="complete", n_completed=len(saved), prediction_sha256=file_sha(prediction_path))
            save_json(run_path, run)
        except BaseException as exc:
            run.update(status="interrupted", n_completed=len(saved), error_type=type(exc).__name__)
            save_json(run_path, run)
            raise


if __name__ == "__main__":
    main()
