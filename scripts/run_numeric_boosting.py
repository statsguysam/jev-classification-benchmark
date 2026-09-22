#!/usr/bin/env python3
"""Exploratory, numerical-only XGBoost/LightGBM supplement to frozen tabular runs.

No API calls, no test-based selection, and no validation fitting. Configuration
is persisted before any fits. Existing test outcomes were already seen, so this
is not a preregistered or independent confirmatory experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import _identity, digest, environment, save_json
from jevbench.types import Prediction, PreparedDataset
from run_tabular_classical import feature_matrix, feature_schema, validate_native
from summarize_tabular import group_bootstrap
from tabular_data import load_native_prepared

MODELS = ("xgboost", "lightgbm")
STUDY_STATUS = "exploratory_supplement_after_prior_test_results_were_viewed"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    """Preserve non-finite library defaults as strings in strict JSON metadata."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if (config.get("study_status") != STUDY_STATUS or config.get("seed") != 42
            or config.get("datasets") != ["breast_cancer", "wine"]
            or config.get("train_per_class") != [4, None]
            or set(config.get("models", {})) != set(MODELS)
            or config.get("bootstrap_samples", 0) < 100):
        raise ValueError("Supplement configuration must retain its numeric scope and two label budgets")
    for name in MODELS:
        if importlib.metadata.version(name) != config["versions"][name]:
            raise ValueError(f"Install pinned {name}=={config['versions'][name]}")
    return config


def numeric_schema(dataset: PreparedDataset, native: dict) -> list[dict]:
    validate_native(dataset, native)
    features = feature_schema(dataset)
    if any(feature["kind"] != "numeric" for feature in features):
        raise ValueError("This supplement accepts numerical columns only; categorical/text data are excluded")
    return features


def model_parameters(model: str, config: dict, n_classes: int, seed: int) -> dict:
    if model not in MODELS:
        raise ValueError(f"Unknown numeric boosting model: {model}")
    params = dict(config["models"][model], random_state=seed)
    if model == "xgboost":
        params.update(objective="binary:logistic" if n_classes == 2 else "multi:softprob",
                      eval_metric="logloss" if n_classes == 2 else "mlogloss")
        if n_classes > 2:
            params["num_class"] = n_classes
    else:
        params["objective"] = "binary" if n_classes == 2 else "multiclass"
    return params


def make_pipeline(model: str, params: dict) -> Pipeline:
    if model == "xgboost":
        from xgboost import XGBClassifier
        estimator = XGBClassifier(**params)
    elif model == "lightgbm":
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown numeric boosting model: {model}")
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("estimator", estimator)])


def fit_numeric(model: str, config: dict, dataset: PreparedDataset, native: dict,
                *, train_per_class: int | None = 4, seed: int = 42):
    if train_per_class is not None and train_per_class < 1:
        raise ValueError("train_per_class must be positive or None")
    features = numeric_schema(dataset, native)
    selected = (dataset.train if train_per_class is None else
                select_examples(dataset.train, dataset.labels, train_per_class, seed))
    if {row.label for row in selected} != set(range(len(dataset.labels))):
        raise ValueError("Selected training rows must cover every class")
    by_id = {record["id"]: record for record in native["train"]}
    training_records = [by_id[row.id] for row in selected]
    x_train = feature_matrix(training_records, features).astype(float)
    parameters = model_parameters(model, config, len(dataset.labels), seed)
    pipeline = make_pipeline(model, parameters)
    fit_start = time.perf_counter()
    # No eval_set, callbacks, early stopping, validation data or test labels.
    pipeline.fit(x_train, np.asarray([row.label for row in selected]))
    fit_s = time.perf_counter() - fit_start
    estimator = pipeline.named_steps["estimator"]
    inference_start = time.perf_counter()
    x_test = feature_matrix(native["test"], features).astype(float)
    probabilities = pipeline.predict_proba(x_test)
    classes = list(estimator.classes_)
    probabilities = probabilities[:, [classes.index(i) for i in range(len(dataset.labels))]]
    chosen = np.argmax(probabilities, axis=1)
    inference_s = time.perf_counter() - inference_start
    predictions = [Prediction(row_id=row.id, label=int(chosen[i]),
        probabilities=probabilities[i].astype(float).tolist(), latency_s=inference_s / len(dataset.test),
        metadata={"probability_kind": "native_uncalibrated", "input_representation": "native_numeric_columns",
                  "decision_rule": "argmax in frozen class order; lowest class index on ties",
                  "latency_measurement": "amortized batch preprocessing and predict_proba"})
        for i, row in enumerate(dataset.test)]
    resolved = (json.loads(estimator.get_booster().save_config()) if model == "xgboost" else
                estimator.booster_.params)
    ids = [row.id for row in selected]
    metadata = {
        "model": model, "seed": seed, "library_version": importlib.metadata.version(model),
        "input_representation": "native_numeric_columns", "training_rows": len(selected),
        "train_per_class": train_per_class, "training_row_ids": ids,
        "training_row_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "training_native_records_sha256": digest(training_records), "validation_rows": 0,
        "test_rows": len(dataset.test), "selection_metric": "fixed_before_supplement_execution",
        "candidate_budget": 1, "selected_parameters": parameters,
        "all_estimator_parameters": json_safe(estimator.get_params(deep=False)),
        "resolved_booster_parameters": json_safe(resolved),
        "selected_validation_macro_f1": None, "refit_with_validation": False,
        "early_stopping": False, "test_used_for_selection": False,
        "fit_and_selection_s": fit_s, "test_inference_s": inference_s,
        "probability_kind": "native_uncalibrated", "probability_calibration": "none",
        "features": {"feature_names": [value["name"] for value in features],
                     "fit_split": "selected training rows only", "scaling": "none",
                     "numeric_training_medians": pipeline.named_steps["impute"].statistics_.tolist(),
                     "missing_values": "training median; all-missing training column becomes zero",
                     "serialized_text_used_as_feature": False},
        "latency_measurement": "amortized batch; not comparable to individual hosted request latency",
    }
    return predictions, metadata


def source_hashes() -> dict:
    names = ["scripts/run_numeric_boosting.py", "scripts/run_tabular_classical.py",
             "scripts/tabular_data.py", "scripts/summarize_tabular.py"]
    return {name: sha(ROOT / name) for name in names}


def run_numeric(dataset, native, model, config, output, *, train_per_class, protocol):
    numeric_schema(dataset, native)
    examples = (dataset.train if train_per_class is None else
                select_examples(dataset.train, dataset.labels, train_per_class, config["seed"]))
    run_config = {"model": model, "train_per_class": train_per_class,
        "input_representation": "native_numeric_columns", "study_status": STUDY_STATUS,
        "protocol_sha256": digest(protocol), "experiment_config_sha256": digest(config),
        "native_records_sha256": digest(native), "feature_schema_sha256": digest(feature_schema(dataset)),
        "source_sha256": source_hashes(), "library_versions": config["versions"],
        "parameters": model_parameters(model, config, len(dataset.labels), config["seed"])}
    identity = _identity(dataset, "classical_numeric_supplement", run_config, config["seed"], examples)
    identity["bootstrap_samples"] = config["bootstrap_samples"]
    run_dir = Path(output) / f"{dataset.name}__{model}__{digest(identity)[:12]}"
    record_path = run_dir / "run.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous.get("status") == "complete":
            for name, expected in previous["artifacts_sha256"].items():
                if sha(run_dir / name) != expected:
                    raise ValueError(f"Saved supplementary artifact changed: {name}")
            return previous
        raise ValueError("Incomplete run exists; preserve it and choose another output directory")
    record = {**identity, "run_id": run_dir.name, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "dataset_manifest": dataset.manifest,
        "environment": environment(), "study_status": STUDY_STATUS,
        "comparison_track": ("matched demonstration labels" if train_per_class is not None else
                             "full training reference; more labels than Jev few-shot"),
        "config_saved_before_first_fit": True}
    record["environment"]["packages"].update(config["versions"])
    save_json(record_path, record)
    started = time.perf_counter()
    predictions, training = fit_numeric(model, config, dataset, native,
                                        train_per_class=train_per_class, seed=config["seed"])
    record.update(training=training, wall_time_s=time.perf_counter() - started)
    (run_dir / "predictions.jsonl").write_text("".join(json.dumps(asdict(p)) + "\n" for p in predictions))
    test_manifest = {"dataset": dataset.name, "labels": dataset.labels,
        "manifest_sha256": digest(dataset.manifest), "rows": [{"id": row.id, "label": row.label,
        "text_sha256": hashlib.sha256(row.text.encode()).hexdigest()} for row in dataset.test]}
    save_json(run_dir / "test_manifest.json", test_manifest)
    metrics = evaluate(dataset.test, predictions, len(dataset.labels))
    metrics["bootstrap"] = group_bootstrap([row.label for row in dataset.test],
        [p.label for p in predictions], [row["text_sha256"] for row in test_manifest["rows"]],
        n_classes=len(dataset.labels), samples=config["bootstrap_samples"], seed=config["seed"])
    record.update(metrics=metrics, status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                  artifacts_sha256={name: sha(run_dir / name) for name in ("predictions.jsonl", "test_manifest.json")})
    save_json(record_path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/numeric_boosting.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/tabular-full")
    parser.add_argument("--output", type=Path, default=ROOT / "results/numeric_decisions/boosting")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    loaded = [load_native_prepared(args.data_root / name) for name in config["datasets"]]
    for dataset, native in loaded:
        numeric_schema(dataset, native)
        select_examples(dataset.train, dataset.labels, 4, config["seed"])
    plan = {"config": config, "config_file_sha256": sha(args.config), "source_sha256": source_hashes(),
        "prepared_manifest_sha256": {dataset.name: digest(dataset.manifest) for dataset, _ in loaded},
        "study_status": STUDY_STATUS, "hosted_model_calls": 0,
        "validation_labels_used_for_fitting_or_selection": 0,
        "prior_results_already_viewed": True}
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if {k: v for k, v in protocol.items() if k != "saved_before_first_fit_at"} != plan:
            raise ValueError("Existing supplement protocol differs; retain it and use a new output directory")
    else:
        protocol = {**plan, "saved_before_first_fit_at": datetime.now(timezone.utc).isoformat()}
        save_json(protocol_path, protocol)
    summary = {"study_status": STUDY_STATUS, "protocol_sha256": digest(protocol), "runs": []}
    for dataset, native in loaded:
        for budget in config["train_per_class"]:
            for model in MODELS:
                record = run_numeric(dataset, native, model, config, args.output,
                                     train_per_class=budget, protocol=protocol)
                metrics = record["metrics"]
                summary["runs"].append({"run_id": record["run_id"], "dataset": dataset.name,
                    "model": model, "train_per_class": budget, "training_rows": record["training"]["training_rows"],
                    "n_test": metrics["n_test"], "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"], "n_failures": metrics["n_failures"],
                    "probability_coverage": metrics["probability_coverage"], "log_loss": metrics["log_loss"],
                    "brier_sum": metrics["brier_sum"], "bootstrap": metrics["bootstrap"],
                    "comparison_track": record["comparison_track"], "run_json_sha256": sha(args.output / record["run_id"] / "run.json")})
                save_json(args.output / "summary.json", summary)
                print(json.dumps({k: v for k, v in summary["runs"][-1].items() if k != "bootstrap"}), flush=True)
    summary["status"] = "complete"
    save_json(args.output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
