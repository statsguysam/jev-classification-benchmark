"""Read-only consistency audit of the numerical boosting supplement; never fit.

This verifies recorded provenance against source/configuration, native records,
and saved predictions. Hash consistency is reproducibility evidence, not a
cryptographic attestation of the machine that originally trained a model.
"""
from __future__ import annotations

from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import _identity, digest, environment
from jevbench.types import Prediction
from run_numeric_boosting import (ROOT, STUDY_STATUS, json_safe, make_pipeline,
                                  model_parameters, numeric_schema, sha)
from summarize_tabular import group_bootstrap


def require(condition, message):
    if not condition:
        raise ValueError("Numeric boosting audit: " + message)


def read(path):
    return json.loads(Path(path).read_text())


def _resolved_parameters(training, parameters, model, n_classes, n_features, version):
    resolved = training["resolved_booster_parameters"]
    if model == "lightgbm":
        expected = dict(training["all_estimator_parameters"])
        for name in ("class_weight", "importance_type"):
            expected.pop(name)
        expected["num_iterations"] = expected.pop("n_estimators")
        expected["num_threads"] = expected.pop("n_jobs")
        expected["metric"] = [parameters["objective"]]
        if n_classes > 2:
            expected["num_class"] = n_classes
        require(resolved == expected, "resolved LightGBM parameters differ")
        return
    require(resolved["version"] == [int(x) for x in version.split(".")], "resolved XGBoost version differs")
    learner = resolved["learner"]
    generic = learner["generic_param"]
    require(generic["device"] == "cpu" and int(generic["n_jobs"]) == parameters["n_jobs"]
            and int(generic["random_state"]) == parameters["random_state"], "resolved XGBoost execution settings differ")
    require(learner["objective"]["name"] == parameters["objective"]
            and learner["learner_train_param"]["objective"] == parameters["objective"]
            and learner["metrics"] == [{"name": parameters["eval_metric"]}], "resolved XGBoost objective differs")
    dimensions = learner["learner_model_param"]
    require(int(dimensions["num_feature"]) == n_features and
            int(dimensions["num_class"]) == (0 if n_classes == 2 else n_classes), "resolved XGBoost dimensions differ")
    booster = learner["gradient_booster"]
    require(booster["name"] == "gbtree" and booster["gbtree_train_param"]["tree_method"] == "hist",
            "resolved XGBoost algorithm differs")
    expected_trees = parameters["n_estimators"] * (1 if n_classes == 2 else n_classes)
    require(int(booster["gbtree_model_param"]["num_trees"]) == expected_trees,
            "resolved XGBoost boosting rounds differ")
    for name in ("max_depth", "learning_rate", "min_child_weight", "reg_lambda", "reg_alpha", "subsample", "colsample_bytree"):
        require(math.isclose(float(booster["tree_train_param"][name]), parameters[name], rel_tol=1e-6, abs_tol=1e-9),
                f"resolved XGBoost {name} differs")


def audit_boosting_run(path, dataset, native, *, root=ROOT):
    """Return None after verifying one completed run, or raise ValueError.

    Estimator constructors are used only to check the complete default parameter
    set of the pinned library version. No fit/predict method or network is used.
    """
    path, root = Path(path), Path(root)
    record, protocol, summary = read(path), read(path.parent.parent / "protocol.json"), read(path.parent.parent / "summary.json")
    config_path = root / "configs/numeric_boosting.json"
    config = read(config_path)
    features = numeric_schema(dataset, native)
    model, budget = record["config"]["model"], record["config"]["train_per_class"]
    require(model in ("xgboost", "lightgbm") and budget in (4, None), "unexpected condition")
    require(config["datasets"] == ["breast_cancer", "wine"] and dataset.name in config["datasets"]
            and config["seed"] == 42 and config["train_per_class"] == [4, None]
            and config["bootstrap_samples"] == 2000 and config["study_status"] == STUDY_STATUS, "protocol scope differs")
    expected_sources = {name: sha(root / name) for name in (
        "scripts/run_numeric_boosting.py", "scripts/run_tabular_classical.py",
        "scripts/tabular_data.py", "scripts/summarize_tabular.py")}
    require(protocol["config"] == config and protocol["config_file_sha256"] == sha(config_path)
            and protocol["source_sha256"] == expected_sources, "configuration or source hashes differ")
    require(protocol["study_status"] == STUDY_STATUS and protocol["hosted_model_calls"] == 0
            and protocol["validation_labels_used_for_fitting_or_selection"] == 0
            and protocol["prior_results_already_viewed"] is True, "study declarations differ")
    require(set(protocol["prepared_manifest_sha256"]) == set(config["datasets"])
            and protocol["prepared_manifest_sha256"][dataset.name] == digest(dataset.manifest), "prepared manifest hash differs")
    require(datetime.fromisoformat(protocol["saved_before_first_fit_at"]) <= datetime.fromisoformat(record["started_at"]),
            "protocol was saved after the run started")
    selected = dataset.train if budget is None else select_examples(dataset.train, dataset.labels, budget, 42)
    parameters = model_parameters(model, config, len(dataset.labels), 42)
    expected_config = {"model": model, "train_per_class": budget, "input_representation": "native_numeric_columns",
        "study_status": STUDY_STATUS, "protocol_sha256": digest(protocol), "experiment_config_sha256": digest(config),
        "native_records_sha256": digest(native), "feature_schema_sha256": digest(features),
        "source_sha256": expected_sources, "library_versions": config["versions"], "parameters": parameters}
    identity = _identity(dataset, "classical_numeric_supplement", expected_config, 42, selected)
    identity["bootstrap_samples"] = config["bootstrap_samples"]
    require(all(record.get(k) == value for k, value in identity.items()), "run/source/native identity differs")
    expected_run_id = f"{dataset.name}__{model}__{digest(identity)[:12]}"
    require(record["run_id"] == path.parent.name == expected_run_id and record["status"] == "complete", "run identity/status differs")
    require(record["dataset_manifest"] == dataset.manifest and record["study_status"] == STUDY_STATUS
            and record["config_saved_before_first_fit"] is True, "run declarations differ")
    require(record["environment"]["source_sha256"] == environment()["source_sha256"], "core source hash differs")
    for library, version in config["versions"].items():
        require(record["environment"]["packages"][library] == version
                and importlib.metadata.version(library) == version, "pinned library version differs")
    training = record["training"]
    require(training["model"] == model and training["seed"] == 42 and training["library_version"] == config["versions"][model],
            "training model/version differs")
    ids = [row.id for row in selected]
    lookup = {row["id"]: row for row in native["train"]}
    selected_native = [lookup[row_id] for row_id in ids]
    import hashlib
    require(training["training_row_ids"] == ids and training["training_rows"] == len(ids)
            and training["train_per_class"] == budget
            and training["training_row_ids_sha256"] == hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            and training["training_native_records_sha256"] == digest(selected_native), "selected training identities differ")
    medians = []
    for feature in features:
        values = [row["features"][feature["name"]] for row in selected_native if row["features"][feature["name"]] is not None]
        medians.append(float(np.median(values)) if values else 0.0)
    expected_preprocessing = {"feature_names": [f["name"] for f in features], "fit_split": "selected training rows only",
        "scaling": "none", "numeric_training_medians": medians,
        "missing_values": "training median; all-missing training column becomes zero", "serialized_text_used_as_feature": False}
    require(training["features"] == expected_preprocessing, "preprocessing differs from selected training rows")
    expected_declarations = {"input_representation": "native_numeric_columns", "validation_rows": 0,
        "test_rows": len(dataset.test), "selection_metric": "fixed_before_supplement_execution", "candidate_budget": 1,
        "selected_validation_macro_f1": None, "refit_with_validation": False, "early_stopping": False,
        "test_used_for_selection": False, "probability_kind": "native_uncalibrated", "probability_calibration": "none"}
    require(all(training.get(k) == value for k, value in expected_declarations.items()), "fitting/selection declarations differ")
    require(training["selected_parameters"] == parameters, "declared parameters differ")
    estimator = make_pipeline(model, parameters).named_steps["estimator"]
    require(training["all_estimator_parameters"] == json_safe(estimator.get_params(deep=False)), "complete constructor parameters differ")
    _resolved_parameters(training, parameters, model, len(dataset.labels), len(features), config["versions"][model])
    require(set(record["artifacts_sha256"]) == {"predictions.jsonl", "test_manifest.json"}
            and all(sha(path.with_name(name)) == expected for name, expected in record["artifacts_sha256"].items()),
            "prediction/test artifact hash differs")
    expected_manifest = {"dataset": dataset.name, "labels": dataset.labels, "manifest_sha256": digest(dataset.manifest),
        "rows": [{"id": row.id, "label": row.label, "text_sha256": hashlib.sha256(row.text.encode()).hexdigest()} for row in dataset.test]}
    require(read(path.with_name("test_manifest.json")) == expected_manifest, "test manifest differs")
    predictions = [Prediction(**json.loads(line)) for line in path.with_name("predictions.jsonl").read_text().splitlines()]
    require([p.row_id for p in predictions] == [row.id for row in dataset.test], "prediction order/coverage differs")
    for prediction in predictions:
        require(type(prediction.label) is int and prediction.error is None and prediction.probabilities is not None
                and prediction.label == int(np.argmax(prediction.probabilities)), "class/probability contract differs")
        require(prediction.metadata.get("probability_kind") == "native_uncalibrated"
                and prediction.metadata.get("input_representation") == "native_numeric_columns", "prediction provenance differs")
    metrics = evaluate(dataset.test, predictions, len(dataset.labels))
    metrics["bootstrap"] = group_bootstrap([row.label for row in dataset.test], [p.label for p in predictions],
        [row["text_sha256"] for row in expected_manifest["rows"]], n_classes=len(dataset.labels), samples=2000, seed=42)
    require(record["metrics"] == metrics, "saved metrics or intervals differ from predictions")
    matches = [r for r in summary["runs"] if r["run_id"] == record["run_id"]]
    require(len(matches) == 1 and matches[0]["run_json_sha256"] == sha(path)
            and summary["protocol_sha256"] == digest(protocol) and summary["study_status"] == STUDY_STATUS,
            "saved summary run/protocol hash differs")
    for name in ("n_test", "accuracy", "macro_f1", "balanced_accuracy", "n_failures", "probability_coverage", "log_loss", "brier_sum", "bootstrap"):
        require(matches[0][name] == metrics[name], f"summary metric differs: {name}")
