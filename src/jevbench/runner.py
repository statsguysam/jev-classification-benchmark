import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from .metrics import evaluate, stratified_bootstrap
from .prompts import build_prompt, select_examples
from .types import Prediction, PreparedDataset


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temp.replace(path)


def environment() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "huggingface-hub", "pyarrow", "torch", "transformers", "peft", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    source = Path(__file__).parent
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "source_sha256": digest({str(p.relative_to(source)): p.read_text() for p in sorted(source.glob("*.py"))})}


def cap_text(dataset: PreparedDataset, max_chars: int | None):
    from .data import _split_audit, normalized_text, validate_prepared

    if max_chars is not None and max_chars < 1:
        raise ValueError("max_chars must be positive")
    dataset.manifest["text_policy"] = {"kind": "character_prefix", "max_chars": max_chars, "applies_to": "all splits and all model families"}
    dataset.manifest["splits_before_text_transform"] = {
        split: _split_audit(getattr(dataset, split)) for split in ("train", "validation", "test")
    }
    for split in ("train", "validation", "test"):
        rows = getattr(dataset, split)
        dataset.manifest.setdefault("truncated_rows", {})[split] = sum(len(r.text) > max_chars for r in rows) if max_chars else 0
        if max_chars:
            setattr(dataset, split, [replace(row, text=row.text[:max_chars]) for row in rows])
    # The prefix transform can create new exact overlaps; held-out data wins.
    test_texts = {normalized_text(r.text) for r in dataset.test}
    removed_val = [r.id for r in dataset.validation if normalized_text(r.text) in test_texts]
    dataset.validation = [r for r in dataset.validation if normalized_text(r.text) not in test_texts]
    heldout = test_texts | {normalized_text(r.text) for r in dataset.validation}
    removed_train = [r.id for r in dataset.train if normalized_text(r.text) in heldout]
    dataset.train = [r for r in dataset.train if normalized_text(r.text) not in heldout]
    dataset.manifest["post_transform_overlap_removed"] = {"train": len(removed_train), "validation": len(removed_val)}
    dataset.manifest["post_transform_overlap_removed_ids"] = {"train": removed_train, "validation": removed_val}
    dataset.manifest["splits"] = {
        split: _split_audit(getattr(dataset, split)) for split in ("train", "validation", "test")
    }
    validate_prepared(dataset)
    dataset.manifest["prepared_content_sha256"] = digest({split: [asdict(r) for r in getattr(dataset, split)] for split in ("train", "validation", "test")})
    return dataset


def _identity(dataset, method, config, seed, examples):
    test_ids = [row.id for row in dataset.test]
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Test row IDs must be unique")
    return {"dataset": dataset.name, "labels": dataset.labels, "manifest_sha256": digest(dataset.manifest),
            "test_ids_sha256": digest(test_ids), "method": method, "config": config,
            "test_records_sha256": digest([asdict(row) for row in dataset.test]),
            "test_manifest_sha256": digest(_test_manifest(dataset)),
            "training_records_sha256": digest([asdict(row) for row in examples]),
            "validation_records_sha256": digest([]),
            "seed": seed, "training_example_ids": [r.id for r in examples],
            "prompt_template_sha256": digest(build_prompt.__code__.co_consts.__repr__()),
            "implementation_sha256": environment()["source_sha256"]}


def _test_manifest(dataset):
    return {
        "dataset": dataset.name,
        "labels": dataset.labels,
        "manifest_sha256": digest(dataset.manifest),
        "rows": [
            {"id": row.id, "label": row.label,
             "text_sha256": hashlib.sha256(row.text.encode()).hexdigest()}
            for row in dataset.test
        ],
    }


def _validate_run_identity(dataset, record, identity):
    if any(record.get(key) != value for key, value in identity.items()):
        raise ValueError("Cached run identity differs from the requested experiment")
    if record.get("dataset_manifest") != dataset.manifest:
        raise ValueError("Cached run dataset manifest differs from the prepared dataset")


def _score_predictions(dataset, predictions, *, seed, samples):
    scored = evaluate(dataset.test, predictions, len(dataset.labels))
    by_id = {prediction.row_id: prediction for prediction in predictions}
    scored["bootstrap"] = stratified_bootstrap(
        [row.label for row in dataset.test],
        [by_id[row.id].label if by_id[row.id].label is not None and not by_id[row.id].error else -1
         for row in dataset.test],
        n_classes=len(dataset.labels), samples=samples, seed=seed,
    )
    return scored


def _resume_completed_run(dataset, predictions, run_dir, record, identity):
    """Verify cached artifacts; repair only a missing, derivable test manifest."""
    _validate_run_identity(dataset, record, identity)
    measured = _score_predictions(
        dataset, predictions, seed=identity["seed"], samples=identity["bootstrap_samples"]
    )
    if record.get("metrics") != measured:
        raise ValueError("Completed run metrics disagree with cached predictions")
    expected = _test_manifest(dataset)
    path = run_dir / "test_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("Cached test manifest differs from the prepared test rows")
    else:
        save_json(path, expected)
    return record


def finish_run(dataset, predictions, run_dir, record, bootstrap_samples=1000):
    scored = _score_predictions(dataset, predictions, seed=record["seed"], samples=bootstrap_samples)
    manifest_path = run_dir / "test_manifest.json"
    expected_manifest = _test_manifest(dataset)
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != expected_manifest:
        raise ValueError("Cached test manifest differs from the prepared test rows")
    save_json(manifest_path, expected_manifest)
    completed = {**record, "status": "complete",
                 "completed_at": datetime.now(timezone.utc).isoformat(), "metrics": scored}
    # Completion is the final commit marker, after every required artifact exists.
    save_json(run_dir / "run.json", completed)
    record.update(completed)
    return record


def run_classical(dataset, model, output_root, *, seed=42, train_per_class=None, bootstrap_samples=1000):
    from .classical import fit_classical
    examples = select_examples(dataset.train, dataset.labels, train_per_class, seed) if train_per_class is not None else dataset.train
    ident = _identity(dataset, "classical", {"model": model, "train_per_class": train_per_class}, seed, examples)
    if train_per_class is None:
        ident["validation_records_sha256"] = digest([asdict(row) for row in dataset.validation])
    ident["bootstrap_samples"] = bootstrap_samples
    run_dir = Path(output_root) / f"{dataset.name}__{model}__{digest(ident)[:12]}"
    existing = run_dir / "run.json"
    if existing.exists() and json.loads(existing.read_text()).get("status") == "complete":
        record = json.loads(existing.read_text())
        predictions = [Prediction(**json.loads(line))
                       for line in run_dir.joinpath("predictions.jsonl").read_text().splitlines()]
        return _resume_completed_run(dataset, predictions, run_dir, record, ident)
    record = {**ident, "run_id": run_dir.name, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "environment": environment(), "dataset_manifest": dataset.manifest}
    save_json(existing, record)
    start = time.perf_counter()
    predictions, training = fit_classical(model, dataset, seed=seed, train_per_class=train_per_class)
    record["training"] = training
    record["wall_time_s"] = time.perf_counter() - start
    run_dir.joinpath("predictions.jsonl").write_text("".join(json.dumps(asdict(p)) + "\n" for p in predictions))
    return finish_run(dataset, predictions, run_dir, record, bootstrap_samples)


def run_model(dataset, config, output_root, *, shots=0, seed=42, allow_paid=False, max_requests=1000, bootstrap_samples=1000):
    from .providers import build_provider
    config = dict(config)
    if any(k.lower() in {"api_key", "token", "authorization"} for k in config):
        raise ValueError("Use api_key_env, never put credentials in a model config")
    paid = config["provider"] not in {"hf", "huggingface", "local_hf", "local"} and not config.get("local", False)
    if paid and not allow_paid:
        raise ValueError("Hosted calls require explicit --allow-paid and --max-requests; no requests sent")
    if config["provider"] in {"hf", "huggingface"}:
        revision = config.get("revision")
        if config.get("adapter_path"):
            training = json.loads((Path(config["adapter_path"]) / "jevbench_training.json").read_text())
            revision = training["resolved_revision"]
        if not revision or len(revision) != 40:
            from huggingface_hub import HfApi
            revision = HfApi().model_info(config["model"], revision=revision).sha
        config["revision"] = revision
    examples = select_examples(dataset.train, dataset.labels, shots, seed)
    ident = _identity(dataset, "zero_shot" if shots == 0 else "few_shot", {**config, "shots_per_class": shots}, seed, examples)
    ident["bootstrap_samples"] = bootstrap_samples
    if config.get("adapter_path"):
        ident["method"] = "lora"
        if shots:
            raise ValueError("The LoRA track evaluates without in-context demonstrations")
        metadata_path = Path(config["adapter_path"]) / "jevbench_training.json"
        training = json.loads(metadata_path.read_text())
        if training["dataset"] != dataset.name or training["labels"] != dataset.labels or training["seed"] != seed:
            raise ValueError("Adapter dataset, label order and seed must match evaluation")
        train_ids = set(training["training_row_ids"])
        if not train_ids <= {r.id for r in dataset.train} or train_ids & {r.id for r in dataset.test}:
            raise ValueError("Adapter training IDs must belong to the frozen training split")
        expected = select_examples(dataset.train, dataset.labels, training["train_per_class"], seed) if training["train_per_class"] is not None else dataset.train
        if train_ids != {r.id for r in expected}:
            raise ValueError("Adapter training IDs differ from the declared label budget")
        expected_prompt_hash = hashlib.sha256(json.dumps([build_prompt(r, dataset.labels) for r in expected], ensure_ascii=False).encode()).hexdigest()
        if training["prompt_sha256"] != expected_prompt_hash:
            raise ValueError("Adapter training text or prompt differs from the evaluation protocol")
        ident["adapter_training"] = training
        ident["training_example_ids"] = training["training_row_ids"]
        ident["training_records_sha256"] = digest([asdict(row) for row in expected])
        adapter_dir = Path(config["adapter_path"])
        ident["adapter_files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(adapter_dir.glob("adapter*")) if p.is_file()}
    model_slug = str(config["model"]).split("/")[-1].replace(":", "-")
    run_dir = Path(output_root) / f"{dataset.name}__{model_slug}__{digest(ident)[:12]}"
    old_record_path = run_dir / "run.json"
    record = json.loads(old_record_path.read_text()) if old_record_path.exists() else None
    if record is not None:
        _validate_run_identity(dataset, record, ident)
    path = run_dir / "predictions.jsonl"
    previous = []
    needs_separator = False
    if path.exists():
        raw = path.read_text()
        needs_separator = bool(raw) and not raw.endswith("\n")
        lines = raw.splitlines()
        for index, line in enumerate(lines):
            try:
                saved = json.loads(line)
            except json.JSONDecodeError:
                if (index != len(lines) - 1 or raw.endswith("\n")
                        or (record and record.get("status") == "complete")):
                    raise
                # A terminated process may leave only the final JSONL row incomplete.
                path.write_text("\n".join(lines[:index]) + ("\n" if index else ""))
                needs_separator = False
                break
            previous.append(Prediction(**saved))
    complete = {p.row_id for p in previous}
    if len(complete) != len(previous) or not complete <= {row.id for row in dataset.test}:
        raise ValueError("Cached predictions have duplicate or unexpected test IDs")
    if config["provider"] in {"hf", "huggingface"} and any(p.metadata.get("resolved_revision") not in {None, config["revision"]} for p in previous):
        raise ValueError("Cached predictions use a different model revision")
    remaining = [row for row in dataset.test if row.id not in complete]
    if len(remaining) > max_requests:
        raise ValueError(f"Run needs {len(remaining)} requests; cap is {max_requests}. Prepare a smaller shared test set or raise the cap explicitly.")
    if record is not None:
        if record.get("status") == "complete":
            return _resume_completed_run(dataset, previous, run_dir, record, ident)
    else:
        record = {**ident, "run_id": run_dir.name, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "environment": environment(), "dataset_manifest": dataset.manifest,
              "execution": {"concurrency": 1, "automatic_retries": 0, "max_new_requests": max_requests, "paid_authorized": allow_paid}}
    record["status"] = "running"
    record.setdefault("execution_sessions", []).append({"started_at": datetime.now(timezone.utc).isoformat(), "cached_rows": len(previous), "new_rows_requested": len(remaining), "environment": environment()})
    save_json(run_dir / "run.json", record)
    if remaining:
        provider = build_provider(config)
        with path.open("a") as file:
            if needs_separator:
                file.write("\n")
                file.flush()
            for i, row in enumerate(remaining):
                prediction = provider.predict(row, dataset.labels, build_prompt(row, dataset.labels, examples))
                file.write(json.dumps(asdict(prediction)) + "\n")
                file.flush()
                previous.append(prediction)
                if (i+1) % 25 == 0:
                    print(f"{dataset.name}: {i+1}/{len(remaining)} new predictions", flush=True)
                # Authentication/outage/config errors should not drain the entire run budget.
                if len(previous) >= 3 and all(p.error for p in previous[-3:]):
                    record["status"] = "stopped_after_three_consecutive_errors"
                    record["n_predictions"] = len(previous)
                    save_json(run_dir / "run.json", record)
                    raise RuntimeError("Stopped after three consecutive provider errors; inspect predictions.jsonl")
    return finish_run(dataset, previous, run_dir, record, bootstrap_samples)


def _sha256_identity(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _comparison_run(directory):
    """Read a complete current run and verify its evidence before pairing it."""
    from .types import Row

    directory = Path(directory)
    record = json.loads((directory / "run.json").read_text())
    if not isinstance(record, dict) or record.get("status") != "complete":
        raise ValueError("Paired comparisons require complete run records")
    required = {
        "run_id", "dataset", "labels", "dataset_manifest", "manifest_sha256",
        "test_ids_sha256", "test_records_sha256", "test_manifest_sha256",
        "training_example_ids", "training_records_sha256", "validation_records_sha256",
        "seed", "bootstrap_samples", "metrics",
    }
    if not required <= record.keys():
        raise ValueError("Run lacks current content identity; use the frozen study runtime for historical comparisons")
    if record["run_id"] != directory.name:
        raise ValueError("Run ID differs from its artifact directory")
    if any(not _sha256_identity(record[name]) for name in required if name.endswith("_sha256")):
        raise ValueError("Invalid run content identity")
    if (type(record["seed"]) is not int or type(record["bootstrap_samples"]) is not int
            or record["bootstrap_samples"] < 100):
        raise ValueError("Invalid run resampling protocol")
    train_ids = record["training_example_ids"]
    if (not isinstance(train_ids, list) or any(not isinstance(value, str) for value in train_ids)
            or len(set(train_ids)) != len(train_ids)):
        raise ValueError("Invalid or duplicate training example IDs")
    training = record.get("training", record.get("adapter_training", {}))
    if not isinstance(training, dict):
        raise ValueError("Invalid saved training metadata")
    validation_count = training.get("validation_rows", 0)
    if type(validation_count) is not int or validation_count < 0:
        raise ValueError("Invalid saved validation-label count")
    if ((not train_ids) != (record["training_records_sha256"] == digest([]))
            or (validation_count == 0) != (record["validation_records_sha256"] == digest([]))):
        raise ValueError("Training or validation budget differs from its empty-content identity")

    manifest = json.loads((directory / "test_manifest.json").read_text())
    if not isinstance(manifest, dict) or digest(manifest) != record["test_manifest_sha256"]:
        raise ValueError("Test manifest differs from the saved content identity")
    labels = record["labels"]
    if (not isinstance(labels, list) or len(labels) < 2
            or any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels)):
        raise ValueError("Invalid saved class labels")
    if (manifest.get("dataset") != record["dataset"] or manifest.get("labels") != labels
            or manifest.get("manifest_sha256") != record["manifest_sha256"]
            or digest(record["dataset_manifest"]) != record["manifest_sha256"]):
        raise ValueError("Test manifest and run dataset identity differ")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Test manifest must contain held-out rows")
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"id", "label", "text_sha256"}
                or not isinstance(row["id"], str) or not row["id"]
                or type(row["label"]) is not int or not 0 <= row["label"] < len(labels)
                or not _sha256_identity(row["text_sha256"])):
            raise ValueError("Invalid test manifest row")
    if digest([row["id"] for row in rows]) != record["test_ids_sha256"]:
        raise ValueError("Test manifest IDs differ from the run identity")
    dataset = PreparedDataset(record["dataset"], labels, [], [],
                              [Row(row["id"], "", row["label"]) for row in rows],
                              record["dataset_manifest"])
    predictions = [Prediction(**json.loads(line))
                   for line in (directory / "predictions.jsonl").read_text().splitlines()]
    # Scoring checks the complete prediction ID set and duplicates before any dict projection.
    measured = _score_predictions(dataset, predictions, seed=record["seed"], samples=record["bootstrap_samples"])
    if record["metrics"] != measured:
        raise ValueError("Completed run metrics disagree with cached predictions")
    return manifest, record, predictions, validation_count


def compare_runs(a_dir, b_dir, *, samples=2000, allow_unequal_training=False):
    runs = [_comparison_run(directory) for directory in (a_dir, b_dir)]
    a, b = [run[0] for run in runs]
    records = [run[1] for run in runs]
    if a != b or records[0]["test_records_sha256"] != records[1]["test_records_sha256"]:
        raise ValueError("Paired comparisons require exactly the same dataset manifest and ordered test rows")
    equal_training = (
        records[0]["seed"] == records[1]["seed"]
        and set(records[0]["training_example_ids"]) == set(records[1]["training_example_ids"])
        and records[0]["training_records_sha256"] == records[1]["training_records_sha256"]
        and records[0]["validation_records_sha256"] == records[1]["validation_records_sha256"]
        and runs[0][3] == runs[1][3]
    )
    if not equal_training and not allow_unequal_training:
        raise ValueError("Unequal seed, training content or validation use: use --allow-unequal-training for a separately labeled descriptive comparison")
    predictions = []
    for _, _, saved, _ in runs:
        values = {prediction.row_id: prediction for prediction in saved}
        predictions.append([values[row["id"]].label if values[row["id"]].label is not None
                            and not values[row["id"]].error else -1 for row in a["rows"]])
    result = stratified_bootstrap([row["label"] for row in a["rows"]], *predictions,
                                 n_classes=len(a["labels"]), samples=samples)
    result.update({"run_a": records[0]["run_id"], "run_b": records[1]["run_id"],
                   "equal_training_ids_and_seed": equal_training,
                   "interpretation": "Paired test-item contrast; equality includes training and validation content, not just IDs. Check output protocols before causal claims"})
    return result
