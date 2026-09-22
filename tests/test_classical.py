from dataclasses import replace

import numpy as np
import pytest

import jevbench.classical as classical
from jevbench.prompts import select_examples
from jevbench.types import PreparedDataset, Row


@pytest.fixture
def dataset():
    return PreparedDataset(
        "fixture", ["negative", "positive"],
        [Row(f"train-{label}-{i}", f"{'dreadful bad sad' if label == 0 else 'wonderful good happy'} trainitem{i}", label)
         for label in range(2) for i in range(12)],
        [Row(f"val-{label}-{i}", f"{'dreadful bad' if label == 0 else 'wonderful good'} valonlytoken valitem{i}", label)
         for label in range(2) for i in range(3)],
        [Row(f"test-{label}-{i}", f"{'dreadful sad' if label == 0 else 'wonderful happy'} testonlytoken testitem{i}", label)
         for label in range(2) for i in range(3)],
    )


def test_features_never_fit_holdout_text(monkeypatch, dataset):
    real_factory = classical._make_vectorizer
    vectorizer = real_factory()
    monkeypatch.setattr(classical, "_make_vectorizer", lambda: vectorizer)
    predictions, metadata = classical.fit_classical("logistic_regression", dataset)
    vocabulary = vectorizer.transformer_list[0][1].vocabulary_
    assert "testonlytoken" not in vocabulary
    assert "valonlytoken" not in vocabulary
    assert metadata["candidate_budget"] == 3
    assert metadata["refit_with_validation"] is False
    assert {row.id for row in dataset.train} == set(metadata["training_row_ids"])
    assert [p.row_id for p in predictions] == [row.id for row in dataset.test]


@pytest.mark.parametrize("model", classical.CLASSICAL_MODELS)
def test_probability_contract_and_shared_label_budget(model, dataset):
    predictions, metadata = classical.fit_classical(model, dataset, seed=37, train_per_class=4)
    assert metadata["training_row_ids"] == [row.id for row in select_examples(dataset.train, dataset.labels, 4, 37)]
    assert metadata["candidate_budget"] == 1
    assert metadata["validation_rows"] == 0
    assert metadata["selected_validation_macro_f1"] is None
    assert metadata["selection_metric"] == "fixed_a_priori"
    if model == "linear_svc":
        assert all(prediction.probabilities is None for prediction in predictions)
        assert metadata["probability_kind"] == "none"
    else:
        for prediction in predictions:
            assert len(prediction.probabilities) == 2
            assert np.isclose(sum(prediction.probabilities), 1)
            assert all(0 <= p <= 1 for p in prediction.probabilities)
    assert all(prediction.error is None for prediction in predictions)


def test_equal_label_track_does_not_select_on_validation_labels(dataset):
    original, metadata = classical.fit_classical("logreg", dataset, seed=20, train_per_class=2)
    altered = replace(dataset, validation=[replace(row, label=1-row.label) for row in dataset.validation])
    rerun, rerun_metadata = classical.fit_classical("logreg", altered, seed=20, train_per_class=2)
    assert [p.label for p in original] == [p.label for p in rerun]
    assert [p.probabilities for p in original] == [p.probabilities for p in rerun]
    assert metadata["selected_parameters"] == rerun_metadata["selected_parameters"] == {"C": 1.0}
