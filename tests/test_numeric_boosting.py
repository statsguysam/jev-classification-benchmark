"""Run native boosting checks in a fresh interpreter, isolated from torch tests.

The native-library tests execute below in a child pytest process. This avoids
mixing macOS OpenMP runtimes used by PyTorch and the boosting wheels in one
interpreter while preserving normal assertion failures and complete coverage.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

if os.environ.get("JEVBENCH_ISOLATED_NUMERIC_TESTS") != "1":
    def test_native_boosting_in_isolated_interpreter():
        if any(importlib.util.find_spec(name) is None for name in ("xgboost", "lightgbm")):
            pytest.skip("Optional pinned XGBoost/LightGBM packages are not installed")
        env = dict(os.environ, JEVBENCH_ISOLATED_NUMERIC_TESTS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        env.pop("PYTEST_ADDOPTS", None)
        result = subprocess.run([sys.executable, "-m", "pytest", str(Path(__file__).resolve()), "-q"],
            cwd=Path(__file__).parents[1], env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=120)
        assert result.returncode == 0, result.stdout
else:
    import copy
    from dataclasses import replace
    import json
    from pathlib import Path

    import numpy as np
    import pytest

    from jevbench.metrics import evaluate
    from jevbench.prompts import select_examples
    from jevbench.types import PreparedDataset, Row


    @pytest.fixture
    def boosting(monkeypatch):
        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
        import run_numeric_boosting
        return run_numeric_boosting


    @pytest.fixture
    def config():
        return json.loads((Path(__file__).parents[1] / "configs/numeric_boosting.json").read_text())


    @pytest.fixture(params=[2, 3])
    def numeric_fixture(request):
        n_classes = request.param
        native, rows = {}, {}
        features = [{"name": "value", "kind": "numeric"}, {"name": "empty", "kind": "numeric"}]
        for split, n, offset in (("train", 7, 0), ("validation", 2, 10000), ("test", 3, 20000)):
            native[split], rows[split] = [], []
            for label in range(n_classes):
                for i in range(n):
                    row_id = f"{split}-{label}-{i}"
                    native[split].append({"id": row_id, "label": label,
                        "features": {"value": None if i == 0 and split == "train" else offset + label * 10 + i,
                                     "empty": None if split == "train" else offset}})
                    rows[split].append(Row(row_id, f"unusable serialized placeholder {row_id}", label))
        dataset = PreparedDataset("fixture", [f"class {i}" for i in range(n_classes)],
            rows["train"], rows["validation"], rows["test"], {"tabular": {"features": features}})
        return dataset, native


    @pytest.mark.parametrize("model", ["xgboost", "lightgbm"])
    @pytest.mark.parametrize("budget", [4, None])
    def test_real_boosters_train_only_and_shared_ids(boosting, config, numeric_fixture, model, budget):
        dataset, native = numeric_fixture
        predictions, training = boosting.fit_numeric(model, config, dataset, native, train_per_class=budget)
        selected = dataset.train if budget is None else select_examples(dataset.train, dataset.labels, 4, 42)
        ids = [row.id for row in selected]
        lookup = {row["id"]: row for row in native["train"]}
        values = [lookup[row_id]["features"]["value"] for row_id in ids]
        median = np.median([v for v in values if v is not None])
        assert training["training_row_ids"] == ids
        assert training["training_native_records_sha256"] == boosting.digest([lookup[row_id] for row_id in ids])
        assert training["features"]["numeric_training_medians"] == [median, 0.0]
        assert training["validation_rows"] == 0 and training["early_stopping"] is False
        assert training["candidate_budget"] == 1 and training["selected_validation_macro_f1"] is None
        assert [p.row_id for p in predictions] == [row.id for row in dataset.test]
        assert all(p.label == int(np.argmax(p.probabilities)) for p in predictions)
        metrics = evaluate(dataset.test, predictions, len(dataset.labels))
        assert metrics["n_failures"] == 0 and metrics["probability_coverage"] == 1
        assert training["resolved_booster_parameters"]


    @pytest.mark.parametrize("model", ["xgboost", "lightgbm"])
    def test_validation_and_test_labels_do_not_change_fit_or_predictions(boosting, config, numeric_fixture, model):
        dataset, native = numeric_fixture
        before, before_training = boosting.fit_numeric(model, config, dataset, native)
        altered_native = copy.deepcopy(native)
        for split in ("validation", "test"):
            for record in altered_native[split]:
                record["label"] = (record["label"] + 1) % len(dataset.labels)
                if split == "validation":
                    record["features"] = {"value": 1e30, "empty": -1e30}
        altered = replace(dataset,
            train=[replace(row, text=f"arbitrary changed serialization {row.id}") for row in dataset.train],
            validation=[replace(row, label=(row.label + 1) % len(dataset.labels)) for row in dataset.validation],
            test=[replace(row, label=(row.label + 1) % len(dataset.labels)) for row in dataset.test])
        after, after_training = boosting.fit_numeric(model, config, altered, altered_native)
        assert [(p.label, p.probabilities) for p in before] == [(p.label, p.probabilities) for p in after]
        assert before_training["features"] == after_training["features"]
        assert before_training["resolved_booster_parameters"] == after_training["resolved_booster_parameters"]


    def test_rejects_categorical_and_sidecar_mismatch(boosting, config, numeric_fixture):
        dataset, native = numeric_fixture
        altered = copy.deepcopy(dataset)
        altered.manifest["tabular"]["features"][0]["kind"] = "categorical"
        altered_native = copy.deepcopy(native)
        for records in altered_native.values():
            for row in records:
                if row["features"]["value"] is not None:
                    row["features"]["value"] = str(row["features"]["value"])
        with pytest.raises(ValueError, match="numerical columns only"):
            boosting.fit_numeric("xgboost", config, altered, altered_native)
        native["train"][0]["id"] = "wrong"
        with pytest.raises(ValueError, match="identity differs"):
            boosting.fit_numeric("lightgbm", config, dataset, native)
