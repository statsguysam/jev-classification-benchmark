#!/usr/bin/env python3
"""Benchmark native tabular features on the same rows used by serialized LLMs.

All five estimators use fixed hyperparameters in both label-budget tracks. The
matched track uses exactly the frozen demonstration selector's training rows;
the full track uses only the prepared training split. Validation labels are not
used. Numeric/categorical preprocessing is fitted inside a training-only sklearn
Pipeline. The frozen text benchmark implementation remains unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from jevbench.data import validate_prepared
from jevbench.prompts import select_examples
from jevbench.runner import _identity, digest, environment, finish_run, save_json
from jevbench.types import Prediction, PreparedDataset

MODELS = ("majority", "logistic_regression", "rbf_svc", "random_forest", "hist_gradient_boosting")
SPLITS = ("train", "validation", "test")
MISSING_CATEGORY = "__JEVBENCH_MISSING__"


def feature_schema(dataset: PreparedDataset) -> list[dict]:
    features = dataset.manifest.get("tabular", {}).get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Prepared data must declare native tabular features")
    names = [value.get("name") for value in features]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Feature names must be nonempty and unique")
    if any(value.get("kind") not in {"numeric", "categorical"} for value in features):
        raise ValueError("Every feature must be numeric or categorical")
    return features


def validate_native(dataset: PreparedDataset, native: dict[str, list[dict]]) -> None:
    """Reject missing, reordered, mislabeled, or nonnative sidecar records.

    The file loader additionally checks native-file hashes and reserializes each
    record to prove its identity with the frozen LLM input.
    """
    validate_prepared(dataset)
    features = feature_schema(dataset)
    names = [value["name"] for value in features]
    if set(native) != set(SPLITS):
        raise ValueError("Native records must contain exactly the three prepared splits")
    for split in SPLITS:
        rows = getattr(dataset, split)
        values = native[split]
        if len(rows) != len(values):
            raise ValueError(f"Native/serialized row count differs in {split}")
        for row, value in zip(rows, values):
            if set(value) != {"id", "label", "features"} or value["id"] != row.id or value["label"] != row.label:
                raise ValueError(f"Native/serialized row identity differs in {split}")
            if not isinstance(value["features"], dict) or set(value["features"]) != set(names):
                raise ValueError("Native feature columns differ from the declared schema")
            for feature in features:
                cell = value["features"][feature["name"]]
                if cell is None:
                    continue
                if feature["kind"] == "numeric":
                    if type(cell) not in (int, float) or not np.isfinite(cell):
                        raise ValueError("Numeric features must be finite native numbers or null")
                elif not isinstance(cell, str) or cell == MISSING_CATEGORY:
                    raise ValueError("Categorical features must be strings excluding the reserved missing marker")


def feature_matrix(records: list[dict], features: list[dict]) -> np.ndarray:
    """Use schema order, never text serialization or mapping insertion order."""
    return np.asarray([
        [np.nan if record["features"][feature["name"]] is None else record["features"][feature["name"]]
         for feature in features]
        for record in records
    ], dtype=object)


def model_parameters(model: str, seed: int) -> dict:
    if model == "majority":
        return {"strategy": "most_frequent"}
    if model == "logistic_regression":
        return {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", "random_state": seed}
    if model == "rbf_svc":
        # No cross-validation-based Platt calibration on the tiny label budget.
        return {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": False,
                "random_state": seed}
    if model == "random_forest":
        return {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1,
                "max_features": "sqrt", "class_weight": None, "n_jobs": 1, "random_state": seed}
    if model == "hist_gradient_boosting":
        return {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 15,
                "min_samples_leaf": 2, "l2_regularization": 1.0,
                "early_stopping": False, "random_state": seed}
    raise ValueError(f"Unknown native model {model!r}; choose from {MODELS}")


def make_pipeline(model: str, features: list[dict], seed: int) -> Pipeline:
    numeric = [i for i, value in enumerate(features) if value["kind"] == "numeric"]
    categorical = [i for i, value in enumerate(features) if value["kind"] == "categorical"]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY,
                                     keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    params = model_parameters(model, seed)
    estimator = {"majority": DummyClassifier, "logistic_regression": LogisticRegression,
                 "rbf_svc": SVC, "random_forest": RandomForestClassifier,
                 "hist_gradient_boosting": HistGradientBoostingClassifier}[model](**params)
    return Pipeline([
        ("features", ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)),
        ("estimator", estimator),
    ])


def preprocessing_provenance(pipeline: Pipeline, features: list[dict]) -> dict:
    preprocessing = pipeline.named_steps["features"]
    fitted = {"fit_split": "selected training rows only", "source_representation": "native_columns",
              "feature_names": [value["name"] for value in features],
              "numeric_columns": [value["name"] for value in features if value["kind"] == "numeric"],
              "categorical_columns": [value["name"] for value in features if value["kind"] == "categorical"],
              "numeric_missing": "training median; all-missing training column becomes zero",
              "categorical_missing": MISSING_CATEGORY, "unknown_categories": "all-zero one-hot encoding",
              "transformed_feature_count": len(preprocessing.get_feature_names_out()),
              "serialized_text_used_as_feature": False}
    if "numeric" in preprocessing.named_transformers_:
        numeric = preprocessing.named_transformers_["numeric"]
        fitted["numeric_training_medians"] = numeric.named_steps["impute"].statistics_.tolist()
        fitted["numeric_training_means_after_imputation"] = numeric.named_steps["scale"].mean_.tolist()
        fitted["numeric_training_scales"] = numeric.named_steps["scale"].scale_.tolist()
    if "categorical" in preprocessing.named_transformers_:
        fitted["training_categories"] = [values.tolist() for values in
            preprocessing.named_transformers_["categorical"].named_steps["encode"].categories_]
    return fitted


def fit_native_classical(model: str, dataset: PreparedDataset, native: dict[str, list[dict]],
                         *, seed: int = 42, train_per_class: int | None = 4
                         ) -> tuple[list[Prediction], dict[str, Any]]:
    if train_per_class is not None and train_per_class < 1:
        raise ValueError("train_per_class must be positive or None")
    validate_native(dataset, native)
    features = feature_schema(dataset)
    selected = (select_examples(dataset.train, dataset.labels, train_per_class, seed)
                if train_per_class is not None else dataset.train)
    train_by_id = {value["id"]: value for value in native["train"]}
    training_records = [train_by_id[row.id] for row in selected]
    if {row.label for row in selected} != set(range(len(dataset.labels))):
        raise ValueError("Selected training rows must cover every class")
    # Neither validation features nor validation labels enter this pipeline.
    x_train = feature_matrix(training_records, features)
    pipeline = make_pipeline(model, features, seed)
    fit_start = time.perf_counter()
    pipeline.fit(x_train, [row.label for row in selected])
    fit_s = time.perf_counter() - fit_start
    preprocessing = preprocessing_provenance(pipeline, features)
    inference_start = time.perf_counter()
    x_test = feature_matrix(native["test"], features)
    chosen = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None
    inference_s = time.perf_counter() - inference_start
    classes = list(pipeline.classes_)
    if probabilities is not None:
        probabilities = probabilities[:, [classes.index(label) for label in range(len(dataset.labels))]]
    probability_kind = "none" if probabilities is None else "native_uncalibrated"
    predictions = [Prediction(
        row_id=row.id, label=int(chosen[i]),
        probabilities=None if probabilities is None else probabilities[i].astype(float).tolist(),
        latency_s=inference_s / len(dataset.test),
        metadata={"probability_kind": probability_kind, "input_representation": "native_columns",
                  "latency_measurement": "amortized batch inference including native feature preprocessing"},
    ) for i, row in enumerate(dataset.test)]
    ids = [row.id for row in selected]
    metadata = {
        "model": model, "seed": seed, "sklearn_version": sklearn_version,
        "input_representation": "native_columns", "training_rows": len(selected),
        "train_per_class": train_per_class, "training_row_ids": ids,
        "training_row_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "training_native_records_sha256": digest(training_records),
        "validation_rows": 0, "test_rows": len(dataset.test),
        "selection_metric": "fixed_a_priori", "candidate_budget": 1,
        "selected_parameters": model_parameters(model, seed), "selected_validation_macro_f1": None,
        "hyperparameter_trials": [{"parameters": model_parameters(model, seed),
                                   "validation_macro_f1": None, "fit_s": fit_s}],
        "fit_and_selection_s": fit_s, "test_inference_s": inference_s,
        "refit_with_validation": False, "probability_kind": probability_kind,
        "probability_calibration": "none", "features": preprocessing,
        "latency_measurement": "amortized batch; not individual request latency",
    }
    return predictions, metadata


def run_native_classical(dataset: PreparedDataset, native: dict[str, list[dict]], model: str,
                         output: Path, *, seed: int = 42, train_per_class: int | None = 4,
                         bootstrap_samples: int = 1000) -> dict:
    validate_native(dataset, native)
    examples = (select_examples(dataset.train, dataset.labels, train_per_class, seed)
                if train_per_class is not None else dataset.train)
    loader_path = Path(__file__).with_name("tabular_data.py")
    config = {"model": model, "train_per_class": train_per_class,
              "input_representation": "native_columns", "selection": "fixed_parameters_no_validation_labels",
              "native_records_sha256": digest(native), "feature_schema_sha256": digest(feature_schema(dataset)),
              "tabular_classical_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "tabular_data_sha256": hashlib.sha256(loader_path.read_bytes()).hexdigest()}
    identity = _identity(dataset, "classical_tabular", config, seed, examples)
    identity["bootstrap_samples"] = bootstrap_samples
    run_dir = Path(output) / f"{dataset.name}__native_{model}__{digest(identity)[:12]}"
    path = run_dir / "run.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("status") == "complete":
            return previous
    record = {**identity, "run_id": run_dir.name, "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "environment": environment(), "dataset_manifest": dataset.manifest}
    save_json(path, record)
    started = time.perf_counter()
    predictions, training = fit_native_classical(model, dataset, native, seed=seed,
                                                 train_per_class=train_per_class)
    record.update(training=training, wall_time_s=time.perf_counter() - started)
    run_dir.joinpath("predictions.jsonl").write_text("".join(json.dumps(asdict(value)) + "\n" for value in predictions))
    return finish_run(dataset, predictions, run_dir, record, bootstrap_samples)


def main(argv: list[str] | None = None) -> int:
    from tabular_data import load_native_prepared

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tabular-full"))
    parser.add_argument("--datasets", nargs="+", default=["titanic", "breast_cancer", "wine"])
    parser.add_argument("--output", type=Path, default=Path("results/tabular/classical"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=4)
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.train_per_class < 1 or args.bootstrap_samples < 100:
        parser.error("--train-per-class must be positive and --bootstrap-samples at least 100")
    loaded = [load_native_prepared(args.data_root / name) for name in dict.fromkeys(args.datasets)]
    budgets = [args.train_per_class] + ([None] if args.include_full else [])
    for dataset, native in loaded:
        validate_native(dataset, native)
        select_examples(dataset.train, dataset.labels, args.train_per_class, args.seed)
    summary = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
               "input_representation": "native_columns", "seed": args.seed,
               "validation_labels_used": 0, "hyperparameters": "fixed before fitting",
               "budgets_per_class": budgets, "runs": []}
    summary_path = args.output / "classical_matrix_summary.json"
    save_json(summary_path, summary)
    for dataset, native in loaded:
        for budget in budgets:
            for model in dict.fromkeys(args.models):
                record = run_native_classical(dataset, native, model, args.output, seed=args.seed,
                    train_per_class=budget, bootstrap_samples=args.bootstrap_samples)
                summary["runs"].append({"run_id": record["run_id"], "dataset": dataset.name,
                    "model": model, "train_per_class": budget, "training_rows": record["training"]["training_rows"],
                    "n_test": record["metrics"]["n_test"], "accuracy": record["metrics"]["accuracy"],
                    "macro_f1": record["metrics"]["macro_f1"], "status": record["status"]})
                save_json(summary_path, summary)
                print(json.dumps(summary["runs"][-1]), flush=True)
    summary.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
    save_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
