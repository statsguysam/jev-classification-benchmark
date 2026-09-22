import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import digest
from jevbench.types import PreparedDataset, Row


@pytest.fixture
def native_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import run_tabular_classical
    return run_tabular_classical


@pytest.fixture
def native_fixture():
    features = [
        {"name": "amount", "kind": "numeric"},
        {"name": "empty_in_train", "kind": "numeric"},
        {"name": "city", "kind": "categorical"},
    ]
    native, rows = {}, {}
    for split, n_rows, offset in (("train", 12, 0), ("validation", 3, 1000), ("test", 3, 5000)):
        native[split], rows[split] = [], []
        for label in range(2):
            for index in range(n_rows):
                row_id = f"{split}-{label}-{index}"
                value = {"id": row_id, "label": label, "features": {
                    "amount": None if split == "train" and index == 0 else offset + 20 * label + index,
                    "empty_in_train": None if split == "train" else offset + index,
                    "city": (None if index == 1 else ["North", "South"][label]) if split == "train" else f"{split}-only",
                }}
                native[split].append(value)
                # Production loader verifies exact feature serialization. These
                # fixture strings deliberately contain no usable feature values.
                rows[split].append(Row(row_id, f"serialized data placeholder {row_id}", label))
    dataset = PreparedDataset("fixture", ["no", "yes"], rows["train"], rows["validation"], rows["test"],
                              {"tabular": {"features": features}})
    return dataset, native


def test_pipeline_fits_imputation_scaling_and_categories_on_train_only(native_module, native_fixture, monkeypatch):
    dataset, native = native_fixture
    original = native_module.make_pipeline
    fitted = []

    def capture(*args):
        pipeline = original(*args)
        fitted.append(pipeline)
        return pipeline

    monkeypatch.setattr(native_module, "make_pipeline", capture)
    _, metadata = native_module.fit_native_classical("logistic_regression", dataset, native, train_per_class=None)
    preprocessing = fitted[0].named_steps["features"]
    numeric = preprocessing.named_transformers_["numeric"]
    values = [record["features"]["amount"] for record in native["train"] if record["features"]["amount"] is not None]
    assert numeric.named_steps["impute"].statistics_.tolist() == [float(np.median(values)), 0.0]
    assert numeric.named_steps["scale"].mean_[0] < 100
    assert numeric.named_steps["scale"].mean_[1] == 0
    categories = preprocessing.named_transformers_["categorical"].named_steps["encode"].categories_[0].tolist()
    assert categories == ["North", "South", native_module.MISSING_CATEGORY]
    assert "validation-only" not in categories and "test-only" not in categories
    assert metadata["features"]["serialized_text_used_as_feature"] is False
    assert metadata["validation_rows"] == 0
    assert metadata["refit_with_validation"] is False


@pytest.mark.parametrize("model", ["majority", "logistic_regression", "rbf_svc", "random_forest", "hist_gradient_boosting"])
def test_exact_shared_label_budget_and_probability_contract(native_module, native_fixture, model):
    dataset, native = native_fixture
    predictions, metadata = native_module.fit_native_classical(model, dataset, native, seed=37, train_per_class=4)
    selected = select_examples(dataset.train, dataset.labels, 4, 37)
    ids = [row.id for row in selected]
    native_by_id = {value["id"]: value for value in native["train"]}
    assert metadata["training_row_ids"] == ids
    assert metadata["training_row_ids_sha256"] == hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    assert metadata["training_native_records_sha256"] == digest([native_by_id[row_id] for row_id in ids])
    assert metadata["training_rows"] == 8
    assert metadata["validation_rows"] == 0
    assert metadata["candidate_budget"] == 1
    assert metadata["selected_validation_macro_f1"] is None
    assert [value.row_id for value in predictions] == [row.id for row in dataset.test]
    metrics = evaluate(dataset.test, predictions, len(dataset.labels))
    assert metrics["n_failures"] == 0
    assert metrics["probability_coverage"] == (0 if model == "rbf_svc" else 1)
    assert metadata["probability_kind"] == ("none" if model == "rbf_svc" else "native_uncalibrated")


@pytest.mark.parametrize("budget", [4, None])
def test_validation_labels_and_serialized_text_do_not_affect_training(native_module, native_fixture, budget):
    dataset, native = native_fixture
    before, before_metadata = native_module.fit_native_classical("logistic_regression", dataset, native, train_per_class=budget)
    altered_native = copy.deepcopy(native)
    for record in altered_native["validation"]:
        record["label"] = 1 - record["label"]
        record["features"]["amount"] = 1e30
        record["features"]["city"] = "must-never-be-fitted"
    altered_dataset = replace(dataset,
        train=[replace(row, text=f"misleading label=other {row.id}") for row in dataset.train],
        validation=[replace(row, label=1-row.label) for row in dataset.validation])
    after, after_metadata = native_module.fit_native_classical("logistic_regression", altered_dataset,
                                                               altered_native, train_per_class=budget)
    assert [(value.label, value.probabilities) for value in before] == [(value.label, value.probabilities) for value in after]
    assert before_metadata["features"] == after_metadata["features"]
    assert before_metadata["training_native_records_sha256"] == after_metadata["training_native_records_sha256"]


@pytest.mark.parametrize("mutation", ["id", "label", "order", "extra_column", "numeric_string", "infinity", "category_number", "category_marker"])
def test_invalid_native_inputs_are_rejected(native_module, native_fixture, mutation):
    dataset, native = native_fixture
    broken = copy.deepcopy(native)
    value = broken["train"][0]
    if mutation == "id":
        value["id"] = "wrong-row"
    elif mutation == "label":
        value["label"] = 1-value["label"]
    elif mutation == "order":
        broken["train"].reverse()
    elif mutation == "extra_column":
        value["features"]["target_leak"] = value["label"]
    elif mutation == "numeric_string":
        value["features"]["amount"] = "123"
    elif mutation == "infinity":
        value["features"]["amount"] = float("inf")
    elif mutation == "category_number":
        value["features"]["city"] = 3
    elif mutation == "category_marker":
        value["features"]["city"] = native_module.MISSING_CATEGORY
    with pytest.raises(ValueError):
        native_module.fit_native_classical("logistic_regression", dataset, broken)


def test_feature_order_follows_schema_and_all_numeric_input_works(native_module, native_fixture):
    dataset, native = native_fixture
    altered = copy.deepcopy(native)
    altered_dataset = copy.deepcopy(dataset)
    altered_dataset.manifest["tabular"]["features"] = altered_dataset.manifest["tabular"]["features"][:2]
    for records in altered.values():
        for record in records:
            features = record["features"]
            record["features"] = {"empty_in_train": features["empty_in_train"], "amount": features["amount"]}
    predictions, metadata = native_module.fit_native_classical("logistic_regression", altered_dataset, altered)
    assert metadata["features"]["feature_names"] == ["amount", "empty_in_train"]
    assert "training_categories" not in metadata["features"]
    assert all(value.probabilities is not None for value in predictions)


def test_native_run_retains_shared_test_manifest_and_frozen_core(native_module, native_fixture, tmp_path):
    dataset, native = native_fixture
    record = native_module.run_native_classical(dataset, native, "majority", tmp_path, bootstrap_samples=100)
    run_dir = tmp_path / record["run_id"]
    manifest = json.loads((run_dir / "test_manifest.json").read_text())
    assert manifest == {"dataset": dataset.name, "labels": dataset.labels,
        "manifest_sha256": digest(dataset.manifest), "rows": [{"id": row.id, "label": row.label,
        "text_sha256": hashlib.sha256(row.text.encode()).hexdigest()} for row in dataset.test]}
    assert record["implementation_sha256"] == native_module.environment()["source_sha256"]
    assert record["method"] == "classical_tabular"
    assert record["config"]["input_representation"] == "native_columns"
    assert record["config"]["tabular_classical_sha256"] == hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest()
    assert record["metrics"]["bootstrap"]["samples"] == 100
    assert record["status"] == "complete"
    assert native_module.run_native_classical(dataset, native, "majority", tmp_path, bootstrap_samples=100) == record
