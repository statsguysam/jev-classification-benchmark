#!/usr/bin/env python3
"""Frozen-text extension: four classical models, shared sparse TF-IDF, two budgets.

Fixed configurations are persisted before fitting. Earlier text results were
already viewed; this is exploratory, not a preregistered confirmatory study.
No hosted calls and no validation data enter training or model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from jevbench.classical import _make_vectorizer
from jevbench.data import load_prepared, normalized_text, validate_prepared
from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import _identity, digest, environment, save_json
from jevbench.types import Prediction
from summarize_tabular import group_bootstrap

MODELS = ("xgboost", "lightgbm", "logistic_regression", "random_forest")
STUDY_STATUS = "exploratory_extension_after_prior_text_test_results_were_viewed"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def load_config(path):
    config = json.loads(Path(path).read_text())
    if (config.get("study_status") != STUDY_STATUS or config.get("seed") != 42
            or config.get("datasets") != ["sst2", "trec"]
            or config.get("train_per_class") != [4, None]
            or set(config.get("models", {})) != set(MODELS)
            or config.get("bootstrap_samples", 0) < 100):
        raise ValueError("Retain the frozen text scope, four models and two label budgets")
    for package, expected in config["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Install pinned {package}=={expected}")
    return config


def model_parameters(model, config, n_classes, seed):
    if model not in MODELS:
        raise ValueError(f"Unknown text model: {model}")
    params = dict(config["models"][model], random_state=seed)
    if model == "xgboost":
        params.update(objective="binary:logistic" if n_classes == 2 else "multi:softprob",
                      eval_metric="logloss" if n_classes == 2 else "mlogloss")
        if n_classes > 2:
            params["num_class"] = n_classes
    elif model == "lightgbm":
        params["objective"] = "binary" if n_classes == 2 else "multiclass"
    return params


def make_estimator(model, params):
    if model == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if model == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    return {"logistic_regression": LogisticRegression, "random_forest": RandomForestClassifier}[model](**params)


def array_digest(array):
    value = np.ascontiguousarray(array)
    return digest({"dtype": str(value.dtype), "shape": list(value.shape),
                   "bytes_sha256": hashlib.sha256(value.tobytes()).hexdigest()})


def sparse_digest(matrix):
    if not sparse.issparse(matrix):
        raise ValueError("TF-IDF must remain sparse")
    value = matrix.tocsr(copy=True)
    value.sort_indices()
    return digest({"shape": list(value.shape), "data": array_digest(value.data),
                   "indices": array_digest(value.indices), "indptr": array_digest(value.indptr)})


def feature_provenance(vectorizer, x_train, selected):
    parts = {}
    for name, fitted in vectorizer.transformer_list:
        ordered = sorted(fitted.vocabulary_.items(), key=lambda item: item[1])
        parts[name] = {"parameters": json_safe(fitted.get_params(deep=False)),
                       "feature_count": len(ordered), "vocabulary_sha256": digest(json_safe(ordered)),
                       "idf_sha256": array_digest(fitted.idf_)}
    return {"fit_split": "selected training rows only", "input_representation": "word_char_tfidf",
            "validation_rows_seen": 0, "test_rows_seen_during_fit": 0,
            "matrix_format": x_train.format, "matrix_dtype": str(x_train.dtype),
            "matrix_shape": list(x_train.shape), "nonzero_count": x_train.nnz,
            "matrix_sha256": sparse_digest(x_train), "dense_conversion": False,
            "training_text_sha256": digest([row.text for row in selected]),
            "training_records_sha256": digest([asdict(row) for row in selected]),
            "transformers": parts}


def fit_text(model, config, dataset, *, train_per_class=4, seed=42):
    validate_prepared(dataset)
    if train_per_class is not None and train_per_class < 1:
        raise ValueError("train_per_class must be positive or None")
    selected = (dataset.train if train_per_class is None else
                select_examples(dataset.train, dataset.labels, train_per_class, seed))
    if {row.label for row in selected} != set(range(len(dataset.labels))):
        raise ValueError("Training subset must contain every class")
    params = model_parameters(model, config, len(dataset.labels), seed)
    vectorizer, estimator = _make_vectorizer(), make_estimator(model, params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        x_train = vectorizer.fit_transform([row.text for row in selected]).tocsr()
        features = feature_provenance(vectorizer, x_train, selected)
        feature_fit_s = time.perf_counter() - start
        start = time.perf_counter()
        # No eval_set, early stopping, validation examples or held-out labels.
        estimator.fit(x_train, np.asarray([row.label for row in selected]))
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        x_test = vectorizer.transform([row.text for row in dataset.test]).tocsr()
        probabilities = estimator.predict_proba(x_test)
        classes = list(estimator.classes_)
        probabilities = probabilities[:, [classes.index(i) for i in range(len(dataset.labels))]]
        if (probabilities.shape != (len(dataset.test), len(dataset.labels))
                or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)):
            raise ValueError("Estimator returned invalid class probabilities")
        chosen = np.argmax(probabilities, axis=1)
        inference_s = time.perf_counter() - start
    warning_counts = Counter((w.category.__name__, str(w.message)) for w in caught)
    resolved = (json.loads(estimator.get_booster().save_config()) if model == "xgboost" else
                json_safe(estimator.booster_.params) if model == "lightgbm" else None)
    ids = [row.id for row in selected]
    metadata = {"model": model, "seed": seed, "input_representation": "word_char_tfidf",
        "training_rows": len(selected), "train_per_class": train_per_class,
        "training_row_ids": ids, "training_row_ids_sha256": digest(ids),
        "training_class_counts": dict(sorted(Counter(str(row.label) for row in selected).items())),
        "validation_rows": 0, "test_rows": len(dataset.test), "selection_metric": "fixed_before_extension_execution",
        "candidate_budget": 1, "selected_parameters": params,
        "all_estimator_parameters": json_safe(estimator.get_params(deep=False)),
        "resolved_booster_parameters": resolved, "selected_validation_macro_f1": None,
        "refit_with_validation": False, "early_stopping": False, "test_used_for_selection": False,
        "feature_fit_s": feature_fit_s, "model_fit_s": fit_s,
        "fit_and_selection_s": feature_fit_s + fit_s, "test_inference_s": inference_s,
        "probability_kind": "native_uncalibrated", "probability_calibration": "none",
        "features": features, "test_matrix_sha256": sparse_digest(x_test),
        "library_versions": {package: importlib.metadata.version(package) for package in config["versions"]},
        "runtime_warnings": [{"category": category, "message": message, "count": count}
                             for (category, message), count in sorted(warning_counts.items())],
        "sparse_absence_semantics": ("XGBoost treats unstored sparse entries as missing; present TF-IDF entries carry their numerical weights"
                                     if model == "xgboost" else "Unstored sparse TF-IDF entries represent zero"),
        "latency_measurement": "amortized batch transform and predict_proba; not individual hosted request latency"}
    predictions = [Prediction(row_id=row.id, label=int(chosen[i]),
        probabilities=probabilities[i].astype(float).tolist(), latency_s=inference_s / len(dataset.test),
        metadata={"probability_kind": "native_uncalibrated", "input_representation": "word_char_tfidf",
                  "decision_rule": "argmax in frozen class order; lowest class index on ties",
                  "latency_measurement": "amortized batch TF-IDF transform and predict_proba"})
        for i, row in enumerate(dataset.test)]
    return predictions, metadata


def source_hashes():
    names = ["scripts/run_text_classical_extension.py", "scripts/summarize_tabular.py"]
    names += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "src/jevbench").glob("*.py"))]
    return {name: sha(ROOT / name) for name in names}


def run_text(dataset, model, config, output, *, train_per_class, protocol):
    selected = (dataset.train if train_per_class is None else
                select_examples(dataset.train, dataset.labels, train_per_class, config["seed"]))
    run_config = {"model": model, "train_per_class": train_per_class,
        "input_representation": "word_char_tfidf", "study_status": STUDY_STATUS,
        "protocol_sha256": digest(protocol), "experiment_config_sha256": digest(config),
        "source_sha256": source_hashes(), "library_versions": config["versions"],
        "parameters": model_parameters(model, config, len(dataset.labels), config["seed"])}
    identity = _identity(dataset, "classical_text_extension", run_config, config["seed"], selected)
    identity["bootstrap_samples"] = config["bootstrap_samples"]
    run_dir = Path(output) / f"{dataset.name}__{model}__{digest(identity)[:12]}"
    path = run_dir / "run.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("status") != "complete":
            raise ValueError("Incomplete run exists; preserve it and choose another output directory")
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("Saved run identity differs")
        for name, expected in previous["artifacts_sha256"].items():
            if sha(run_dir / name) != expected:
                raise ValueError(f"Saved extension artifact changed: {name}")
        return previous
    record = {**identity, "run_id": run_dir.name, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "dataset_manifest": dataset.manifest,
        "environment": environment(), "study_status": STUDY_STATUS,
        "comparison_track": ("matched demonstration labels" if train_per_class is not None else
                             "full prepared training reference; more labels than Jev few-shot"),
        "config_saved_before_first_fit": True}
    record["environment"]["packages"].update(config["versions"])
    save_json(path, record)
    test_manifest = {"dataset": dataset.name, "labels": dataset.labels,
        "manifest_sha256": digest(dataset.manifest), "rows": [{"id": row.id, "label": row.label,
        "text_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
        "normalized_text_sha256": hashlib.sha256(normalized_text(row.text).encode()).hexdigest()}
        for row in dataset.test]}
    save_json(run_dir / "test_manifest.json", test_manifest)
    start = time.perf_counter()
    try:
        predictions, training = fit_text(model, config, dataset,
                                          train_per_class=train_per_class, seed=config["seed"])
    except Exception as exc:
        # Preserve the entire failed condition and its rows; never silently omit it.
        predictions = [Prediction(row_id=row.id, label=None,
                       error=f"classical_condition_failed:{type(exc).__name__}") for row in dataset.test]
        (run_dir / "predictions.jsonl").write_text("".join(json.dumps(asdict(p)) + "\n" for p in predictions))
        record.update(status="failed", wall_time_s=time.perf_counter() - start,
                      error={"type": type(exc).__name__, "message": str(exc)},
                      failed_rows=[row.id for row in dataset.test])
        save_json(path, record)
        raise
    record.update(training=training, wall_time_s=time.perf_counter() - start)
    (run_dir / "predictions.jsonl").write_text("".join(json.dumps(asdict(p)) + "\n" for p in predictions))
    metrics = evaluate(dataset.test, predictions, len(dataset.labels))
    metrics["bootstrap"] = group_bootstrap([row.label for row in dataset.test],
        [p.label for p in predictions], [row["normalized_text_sha256"] for row in test_manifest["rows"]],
        n_classes=len(dataset.labels), samples=config["bootstrap_samples"], seed=config["seed"])
    record.update(metrics=metrics, status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                  artifacts_sha256={name: sha(run_dir / name) for name in ("predictions.jsonl", "test_manifest.json")})
    save_json(path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/text_classical_extension.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "results/text_extension/classical")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    datasets = [load_prepared(args.data_root / name) for name in config["datasets"]]
    for dataset in datasets:
        select_examples(dataset.train, dataset.labels, 4, config["seed"])
        if len(dataset.test) != 200:
            raise ValueError("Extension must use all 200 frozen pilot test rows")
    plan = {"config": config, "config_file_sha256": sha(args.config), "source_sha256": source_hashes(),
        "prepared_manifest_sha256": {dataset.name: digest(dataset.manifest) for dataset in datasets},
        "prepared_files_sha256": {dataset.name: {name: sha(args.data_root / dataset.name / name)
            for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")} for dataset in datasets},
        "study_status": STUDY_STATUS, "hosted_model_calls": 0,
        "validation_labels_used_for_fitting_or_selection": 0, "prior_results_already_viewed": True}
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if {k: v for k, v in protocol.items() if k != "saved_before_first_fit_at"} != plan:
            raise ValueError("Existing extension protocol differs; retain it and use a new output directory")
    else:
        protocol = {**plan, "saved_before_first_fit_at": datetime.now(timezone.utc).isoformat()}
        save_json(protocol_path, protocol)
    summary = {"status": "running", "study_status": STUDY_STATUS,
               "protocol_sha256": digest(protocol), "planned_conditions": 16, "runs": []}
    save_json(args.output / "summary.json", summary)
    for dataset in datasets:
        for budget in config["train_per_class"]:
            for model in MODELS:
                record = run_text(dataset, model, config, args.output, train_per_class=budget, protocol=protocol)
                metrics = record["metrics"]
                summary["runs"].append({"run_id": record["run_id"], "dataset": dataset.name,
                    "status": record["status"], "model": model, "train_per_class": budget,
                    "training_rows": record["training"]["training_rows"], "n_test": metrics["n_test"],
                    "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"], "n_failures": metrics["n_failures"],
                    "probability_coverage": metrics["probability_coverage"], "log_loss": metrics["log_loss"],
                    "brier_sum": metrics["brier_sum"], "bootstrap": metrics["bootstrap"],
                    "comparison_track": record["comparison_track"], "runtime_warnings": record["training"]["runtime_warnings"],
                    "run_json_sha256": sha(args.output / record["run_id"] / "run.json")})
                save_json(args.output / "summary.json", summary)
                print(json.dumps({k: v for k, v in summary["runs"][-1].items()
                                  if k not in {"bootstrap", "runtime_warnings"}}), flush=True)
    summary.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
    save_json(args.output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
