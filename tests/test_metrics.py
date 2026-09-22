import pytest
from jevbench.metrics import evaluate, stratified_bootstrap
from jevbench.types import Row, Prediction


def test_failures_are_counted_and_not_renormalized_away():
    rows = [Row("a", "", 0), Row("b", "", 1)]
    result = evaluate(rows, [Prediction("a", 0, [.9, .1]), Prediction("b", None, error="timeout")], 2)
    assert result["accuracy"] == .5
    assert result["n_failures"] == 1
    assert result["probability_coverage"] == .5
    assert result["macro_f1"] == .5
    assert result["confusion_matrix"][1][-1] == 1


def test_duplicate_or_missing_ids_rejected():
    with pytest.raises(ValueError):
        evaluate([Row("a", "", 0)], [Prediction("b", 0)], 2)


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        evaluate([Row("a", "", 0)], [Prediction("a", 0, [.8, .8])], 2)


def test_no_probabilities_no_calibration_claim():
    result = evaluate([Row("a", "", 0), Row("b", "", 1)], [Prediction("a", 0), Prediction("b", 1)], 2)
    assert result["probability_coverage"] == 0
    assert "log_loss" not in result


def test_paired_identical_models_have_zero_difference():
    result = stratified_bootstrap([0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], n_classes=2, samples=100)
    assert result["metrics"]["macro_f1"]["ci95"] == [0, 0]
