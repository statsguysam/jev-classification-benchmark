"""Verify immutable text source predictions without model loads or network calls.

Original Colab manifest hashes remain unequal. Separately recorded full-content
and matched-training proofs establish which frozen rows can be paired. Source
freezing is optional, per condition, exclusive, and never overwrites a file.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from audit_cross_environment import audit, read_run, recompute_prepared
from compare_combined_pilot import read_predictions
from jevbench.data import load_prepared
from jevbench.metrics import evaluate
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import digest, environment
from jevbench.types import Prediction

DATASETS = ("sst2", "trec")
SHOTS = (0, 4)
SEED = 42
AREA = ROOT / "results/text_extension"
LOCAL = AREA / "local"
SOURCES_DIR = AREA / "sources"
CONFIG = ROOT / "configs/text_extension_sources.json"
FROZEN_CORE_SHA = "d5547ccde4224e182653315d81c0a631c789bbe294ad7bcf95ef0cddd25ce608"
FILE_NAMES = ("run.json", "predictions.jsonl", "test_manifest.json")
MODELS = {
    "qwen_small": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "provider": "hf", "revision": "7ae557604adf67be50417f59c2c2f167def9a775"},
    "qwen_main": {"model": "Qwen/Qwen3-4B-Instruct-2507", "provider": "hf", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554"},
    "luna": {"model": "gpt-5.6-luna", "provider": "openai"},
    "astra": {"model": "gpt-6-astra", "provider": "openai"},
    "smollm2": {"model": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "provider": "hf", "revision": "31b70e2e869a7173562077fd711b654946d38674", "new_local": True},
    "granite": {"model": "ibm-granite/granite-3.3-2b-instruct", "provider": "hf", "revision": "707f574c62054322f6b5b04b6d075f0a8f05e0f0", "new_local": True},
}
DIRECT = {"model": "typesafe/jev-1.13", "provider": "jev"}


class SourceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SourceError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def condition_key(dataset, model_key, shots):
    require(dataset in DATASETS and model_key in {*MODELS, "jev"} and type(shots) is int and shots in SHOTS,
            "Unsupported text source condition")
    return f"{dataset}__{model_key}__k{shots}"


def inside_root(relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Source paths must be repository relative")
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT.resolve()), "Source path escapes repository")
    return path


def load_config():
    config = json.loads(CONFIG.read_text())
    require(config.get("schema_version") == 1 and config.get("seed") == SEED and
            config.get("core_sha256") == FROZEN_CORE_SHA == environment()["source_sha256"],
            "Frozen text registry/core identity differs")
    require(config.get("models") == MODELS and config.get("direct_model") == DIRECT, "Frozen text model pins differ")
    expected = {condition_key(d, m, s) for d in DATASETS for m in ("qwen_small", "qwen_main", "luna", "astra", "jev") for s in SHOTS}
    require(set(config.get("datasets", {})) == set(DATASETS) and set(config.get("historical_sources", {})) == expected,
            "Frozen text registry inventory differs")
    return config


def load_dataset(name):
    require(name in DATASETS, "Only frozen SST2/TREC datasets are permitted")
    pinned = load_config()["datasets"][name]
    folder = inside_root(pinned["prepared_path"])
    require(set(pinned["hashes"]) == {"manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"},
            "All prepared source file hashes are required")
    require(all(file_sha(folder / filename) == expected for filename, expected in pinned["hashes"].items()),
            "Frozen prepared text artifact changed")
    dataset = load_prepared(folder)
    require(dataset.name == name and dataset.labels == pinned["labels"] and
            digest(dataset.manifest) == pinned["manifest_sha256"] and
            dataset.manifest["prepared_content_sha256"] == pinned["prepared_content_sha256"] and
            {split: len(getattr(dataset, split)) for split in ("train", "validation", "test")} == pinned["sizes"],
            "Frozen prepared text identity differs")
    return dataset


def historical_source(dataset, model_key, shots):
    key = condition_key(dataset, model_key, shots)
    require(model_key == "jev" or not MODELS[model_key].get("new_local"), "New local sources are not historical")
    return load_config()["historical_sources"][key]


def verify_artifacts(source):
    require(set(source) == {"model", "provider", "path", "hashes"} and set(source["hashes"]) == set(FILE_NAMES),
            "Unexpected frozen source artifact schema")
    folder = inside_root(source["path"])
    require(all(file_sha(folder / name) == expected for name, expected in source["hashes"].items()),
            "Frozen source artifact changed")
    return folder


def discover_local_source(dataset, model_key, shots):
    candidates = []
    for path in sorted(LOCAL.glob("*/run.json")):
        record = json.loads(path.read_text())
        config = record.get("config", {})
        if (record.get("dataset"), config.get("model"), config.get("shots_per_class"), record.get("status")) == (
                dataset, MODELS[model_key]["model"], shots, "complete"):
            candidates.append(path)
    require(len(candidates) <= 1, "Ambiguous completed local source; never select by test score")
    if not candidates:
        return None
    path = candidates[0]
    require(all(path.with_name(name).is_file() for name in FILE_NAMES), "Completed source lacks final artifacts")
    return {"model": MODELS[model_key]["model"], "provider": "hf", "path": path.parent.relative_to(ROOT).as_posix(),
            "hashes": {name: file_sha(path.with_name(name)) for name in FILE_NAMES}}


def validate_source(dataset, model_key, shots, source):
    condition_key(dataset.name, model_key, shots)
    canonical = load_dataset(dataset.name)
    require(asdict(dataset) == asdict(canonical), "Caller dataset differs from frozen text corpus")
    spec = DIRECT if model_key == "jev" else MODELS[model_key]
    require(source["model"] == spec["model"] and source["provider"] == spec["provider"], "Source model/provider differs")
    directory = verify_artifacts(source)
    if spec.get("new_local"):
        require(directory.parent == LOCAL.resolve(), "Local source must use canonical text result directory")
    else:
        require(source == historical_source(dataset.name, model_key, shots), "Historical source pins differ")
    record, test = read_run(directory)
    examples = select_examples(dataset.train, dataset.labels, shots, SEED)
    expected = {"run_id": directory.name, "status": "complete", "dataset": dataset.name, "labels": dataset.labels,
                "seed": SEED, "method": "zero_shot" if shots == 0 else "few_shot",
                "test_ids_sha256": digest([row.id for row in dataset.test]),
                "training_example_ids": [row.id for row in examples], "implementation_sha256": FROZEN_CORE_SHA,
                "prompt_template_sha256": digest(build_prompt.__code__.co_consts.__repr__())}
    require(all(record.get(key) == value for key, value in expected.items()), "Source dataset/training/core identity differs")
    config = record["config"]
    require(config.get("provider") == spec["provider"] and config.get("model") == spec["model"] and
            config.get("shots_per_class") == shots and not config.get("adapter_path") and
            record.get("environment", {}).get("source_sha256") == FROZEN_CORE_SHA,
            "Source model, method, or label budget differs")
    if "revision" in spec:
        require(config.get("revision") == spec["revision"], "Pinned source revision differs")
    prepared_path = inside_root(load_config()["datasets"][dataset.name]["prepared_path"])
    recompute_prepared(prepared_path, record, test)
    # Do not rewrite Colab hashes to the local manifest. Bind the independently
    # audited equivalence proof to a frozen same-shot canonical reference run.
    reference = historical_source(dataset.name, "qwen_small", shots)
    reference_dir = verify_artifacts(reference)
    proof = audit(directory, reference_dir, prepared_path)
    require(proof["eligible_for_separately_labeled_matched_pairing"] and proof["raw_content_recomputed"],
            "Source full-content/matched-training audit failed")
    # Paths describe evidence locations, not content identity. Keep the frozen
    # proof portable between private checkouts without changing any source hash.
    proof["run_a"]["path"] = directory.relative_to(ROOT).as_posix()
    proof["run_b"]["path"] = reference_dir.relative_to(ROOT).as_posix()
    for side in ("a", "b"):
        proof[f"recomputed_content_{side}"]["prepared_directory"] = prepared_path.relative_to(ROOT).as_posix()
    # Class-count dictionaries contain integer keys in the audit API. Normalize
    # once so an exclusive JSON freeze compares identically when read back.
    proof = json.loads(json.dumps(proof, sort_keys=True, allow_nan=False))
    labels, measured_labels = read_predictions(directory, record, test)
    predictions = [Prediction(**json.loads(line)) for line in (directory / "predictions.jsonl").read_bytes().splitlines() if line.strip()]
    require(measured_labels["prediction_order_matches_test_order"] and
            [prediction.row_id for prediction in predictions] == [row.id for row in dataset.test],
            "Source prediction order differs from frozen test rows")
    for prediction in predictions:
        require(isinstance(prediction.metadata, dict), "Malformed source prediction metadata")
        if prediction.error is not None:
            require(isinstance(prediction.error, str) and bool(prediction.error) and prediction.label is None and
                    prediction.probabilities is None, "Malformed source failure")
            continue
        require(type(prediction.label) is int and 0 <= prediction.label < len(dataset.labels), "Invalid source class ID")
        values = prediction.probabilities
        if values is not None:
            require(isinstance(values, list) and len(values) == len(dataset.labels) and
                    all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in values) and
                    abs(sum(values) - 1) < 1e-6 and prediction.label == max(range(len(values)), key=values.__getitem__),
                    "Invalid source probability vector or class decision")
        if spec["provider"] in {"hf", "jev"}:
            require(values is not None, "Successful source row lacks class probability evidence")
    measured = evaluate(dataset.test, predictions, len(dataset.labels))
    require(all(record["metrics"].get(key) == value for key, value in measured.items()), "Source metrics differ from raw predictions")
    if spec.get("new_local"):
        helper = importlib.import_module("run_text_extension_local")
        checked_record, checked_predictions = helper.audit_source_run(directory / "run.json", dataset, model_key, shots)
        require(checked_record == record and [asdict(value) for value in checked_predictions] == [asdict(value) for value in predictions],
                "Local source renderer audit differs")
    return record, predictions, examples, proof


def source_manifest(dataset, model_key, shots, source, record, proof):
    return {"schema_version": 1, "dataset": dataset.name, "model_key": model_key, "shots_per_class": shots,
            "source": source, "source_model_config": record["config"], "source_config_sha256": digest(record["config"]),
            "dataset_manifest_sha256": digest(dataset.manifest), "original_source_manifest_sha256": record["manifest_sha256"],
            "test_ids_sha256": record["test_ids_sha256"], "training_example_ids": record["training_example_ids"],
            "implementation_sha256": record["implementation_sha256"], "registry_sha256": file_sha(CONFIG),
            "source_validator_sha256": file_sha(__file__), "content_audit": proof}


def _load(dataset, model_key, shots, freeze):
    slug = condition_key(dataset.name, model_key, shots)
    path = SOURCES_DIR / f"{slug}.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    if saved is not None:
        source = saved["source"]
    elif model_key != "jev" and MODELS[model_key].get("new_local"):
        source = discover_local_source(dataset.name, model_key, shots)
    else:
        source = historical_source(dataset.name, model_key, shots)
    if source is None:
        return None
    record, predictions, examples, proof = validate_source(dataset, model_key, shots, source)
    manifest = source_manifest(dataset, model_key, shots, source, record, proof)
    if saved is not None:
        require(saved == manifest, "Frozen text source manifest changed")
    elif freeze:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            file.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
    return {"dataset": dataset, "model_key": model_key, "shots": shots, "source": source, "record": record,
            "predictions": predictions, "examples": examples, "content_audit": proof,
            "manifest_path": path.relative_to(ROOT).as_posix(), "manifest_sha256": file_sha(path) if path.exists() else None}


def load_source(dataset, model_key, shots, freeze=False):
    require(model_key in MODELS, "Unsupported LLM source model")
    return _load(dataset, model_key, shots, freeze)


def load_direct(dataset, shots, freeze=False):
    return _load(dataset, "jev", shots, freeze)
