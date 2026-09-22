"""Sparse text models isolated from torch's macOS OpenMP runtime."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

if os.environ.get("JEVBENCH_ISOLATED_TEXT_CLASSICAL") != "1":
    def test_text_classical_in_isolated_interpreter():
        if any(importlib.util.find_spec(name) is None for name in ("xgboost", "lightgbm")):
            pytest.skip("Optional pinned boosting packages are not installed")
        env = dict(os.environ, JEVBENCH_ISOLATED_TEXT_CLASSICAL="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        env.pop("PYTEST_ADDOPTS", None)
        result = subprocess.run([sys.executable, "-m", "pytest", str(Path(__file__).resolve()), "-q"],
            cwd=Path(__file__).parents[1], env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, result.stdout
else:
    from dataclasses import replace
    import json

    import numpy as np
    from scipy import sparse

    from jevbench.prompts import select_examples
    from jevbench.runner import digest
    from jevbench.types import PreparedDataset, Row

    @pytest.fixture
    def module(monkeypatch):
        monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
        import run_text_classical_extension
        return run_text_classical_extension

    @pytest.fixture
    def config():
        cfg = json.loads((Path(__file__).parents[1] / "configs/text_classical_extension.json").read_text())
        cfg["bootstrap_samples"] = 100
        return cfg

    @pytest.fixture(params=[2, 6])
    def dataset(request):
        labels = ["animal", "fruit", "vehicle", "mineral", "planet", "instrument"][:request.param]
        rows = {}
        for split, count in (("train", 7), ("validation", 2), ("test", 3)):
            rows[split] = [Row(f"{split}-{label}-{i}", f"{split}sentinel {labels[label]} context sample {i}", label)
                           for label in range(len(labels)) for i in range(count)]
        return PreparedDataset("fixture", labels, rows["train"], rows["validation"], rows["test"], {})

    @pytest.mark.parametrize("model", ["xgboost", "lightgbm", "logistic_regression", "random_forest"])
    @pytest.mark.parametrize("budget", [4, None])
    def test_sparse_same_examples_and_training_only_vocabulary(module, config, dataset, model, budget, monkeypatch):
        original = module._make_vectorizer
        recorded = []
        def vectorizer():
            value = original()
            recorded.append(value)
            return value
        monkeypatch.setattr(module, "_make_vectorizer", vectorizer)
        def reject_dense(*args, **kwargs):
            raise AssertionError("Sparse feature matrix was densified")
        monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_dense)
        monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_dense)
        predictions, training = module.fit_text(model, config, dataset, train_per_class=budget)
        selected = dataset.train if budget is None else select_examples(dataset.train, dataset.labels, 4, 42)
        assert training["training_row_ids"] == [row.id for row in selected]
        assert training["training_rows"] == len(dataset.labels) * (7 if budget is None else 4)
        assert training["training_row_ids_sha256"] == digest([row.id for row in selected])
        assert training["features"]["training_records_sha256"] == digest([module.asdict(row) for row in selected])
        assert training["validation_rows"] == 0 and training["candidate_budget"] == 1
        assert training["test_used_for_selection"] is False and training["early_stopping"] is False
        assert training["features"]["matrix_format"] == "csr"
        word = dict(recorded[0].transformer_list)["word"]
        assert "trainsentinel" in word.vocabulary_
        assert "validationsentinel" not in word.vocabulary_
        assert "testsentinel" not in word.vocabulary_
        assert [p.row_id for p in predictions] == [row.id for row in dataset.test]
        assert all(p.label == np.argmax(p.probabilities) for p in predictions)
        assert all(np.isfinite(p.probabilities).all() and np.isclose(sum(p.probabilities), 1) for p in predictions)
        assert not any(w["category"] == "ConvergenceWarning" for w in training["runtime_warnings"])

    @pytest.mark.parametrize("model", ["xgboost", "lightgbm", "logistic_regression", "random_forest"])
    def test_validation_features_and_test_labels_do_not_influence_fit(module, config, dataset, model):
        before, before_meta = module.fit_text(model, config, dataset)
        changed = replace(dataset,
            validation=[replace(row, text=f"unseenvalidationtoken {row.id}", label=(row.label + 1) % len(dataset.labels))
                        for row in dataset.validation],
            test=[replace(row, label=(row.label + 1) % len(dataset.labels)) for row in dataset.test])
        after, after_meta = module.fit_text(model, config, changed)
        assert [(p.label, p.probabilities) for p in before] == [(p.label, p.probabilities) for p in after]
        assert before_meta["features"] == after_meta["features"]
        assert before_meta["selected_parameters"] == after_meta["selected_parameters"]

    def test_checkpoint_reuse_rejects_corruption(module, config, dataset, tmp_path):
        record = module.run_text(dataset, "logistic_regression", config, tmp_path,
                                 train_per_class=4, protocol={"test": True})
        again = module.run_text(dataset, "logistic_regression", config, tmp_path,
                                train_per_class=4, protocol={"test": True})
        assert again == record
        saved = tmp_path / record["run_id"] / "predictions.jsonl"
        saved.write_text(saved.read_text() + "{}\n")
        with pytest.raises(ValueError, match="artifact changed"):
            module.run_text(dataset, "logistic_regression", config, tmp_path,
                            train_per_class=4, protocol={"test": True})

    def test_failed_fit_preserves_every_failed_row(module, config, dataset, tmp_path, monkeypatch):
        def fail(*args, **kwargs):
            raise RuntimeError("fixture estimator failure")
        monkeypatch.setattr(module, "fit_text", fail)
        with pytest.raises(RuntimeError, match="fixture estimator failure"):
            module.run_text(dataset, "random_forest", config, tmp_path,
                            train_per_class=4, protocol={"test": True})
        path = next(tmp_path.glob("*/run.json"))
        record = json.loads(path.read_text())
        predictions = [json.loads(line) for line in path.with_name("predictions.jsonl").read_text().splitlines()]
        assert record["status"] == "failed"
        assert record["failed_rows"] == [row.id for row in dataset.test]
        assert all(p["label"] is None and p["error"] for p in predictions)
        assert [p["row_id"] for p in predictions] == record["failed_rows"]

    def test_rejects_bad_budget_and_split_leakage(module, config, dataset):
        with pytest.raises(ValueError, match="positive"):
            module.fit_text("logistic_regression", config, dataset, train_per_class=0)
        changed = replace(dataset, test=[replace(dataset.test[0], text=dataset.train[0].text)] + dataset.test[1:])
        with pytest.raises(ValueError, match="leakage"):
            module.fit_text("logistic_regression", config, changed)
