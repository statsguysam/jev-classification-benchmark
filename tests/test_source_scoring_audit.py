from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_source_scoring as audit
from jevbench.types import Row, Prediction, PreparedDataset


def data():
    return PreparedDataset("sst2", ["negative", "positive"],
        [Row(f"train-{i}", f"train item {i}", i % 2) for i in range(20)],
        [Row(f"val-{i}", f"validation item {i}", i % 2) for i in range(20)],
        [Row(f"test-{i}", f"test item {i}", i % 2) for i in range(4)], {})


def prediction(row, probs=None, *, error=None):
    return Prediction(row.id, None if error else max(range(2), key=probs.__getitem__), probs, error=error,
        metadata={"scoring": "sum_logp_numeric_id_plus_eos", "probability_kind": "label_sequence_likelihood_normalized",
                  "device": "fixture", "dtype": "float16", "candidate_tokens_scored": 4})


def test_constant_label_can_have_informative_margin_without_gate_selection():
    dataset = data()
    values = [[.9, .1], [.6, .4], [.8, .2], [.55, .45]]
    result = audit.profile(dataset, [prediction(row, p) for row, p in zip(dataset.test, values)])
    assert result["constant_class"] and result["predicted_class_counts"] == [4, 0]
    assert result["accuracy"] == .5 and result["n_failures"] == 0
    assert result["descriptive_error_auroc_negative_margin"] == 1
    assert result["confidence_gate"]["constant_label_does_not_imply_constant_confidence"]
    assert result["confidence_gate"]["prospective_threshold_validated"] is False
    assert result["confidence_gate"]["threshold_selected_from_test"] is False


def test_equal_confidence_ties_and_failure_denominator_are_explicit():
    dataset = data()
    predictions = [prediction(row, [.5, .5]) for row in dataset.test[:3]]
    predictions.append(prediction(dataset.test[3], error="synthetic failure"))
    result = audit.profile(dataset, predictions)
    assert result["n_failures"] == 1 and result["n_probability_rows"] == 3 and result["accuracy"] == .5
    assert result["n_top_probability_ties"] == 3 and result["descriptive_error_auroc_negative_margin"] == .5
    assert result["confidence_gate"]["finite_scores_available"] is False
    assert result["confidence_gate"]["margin_ranking_has_variation"] is False


@pytest.mark.parametrize("change", ["nan", "argmax", "order", "protocol", "unnormalized"])
def test_invalid_or_misaligned_scores_fail_closed(change):
    dataset = data()
    predictions = [prediction(row, [.8, .2]) for row in dataset.test]
    if change == "nan": predictions[0].probabilities = [float("nan"), 0]
    if change == "argmax": predictions[0].label = 1
    if change == "order": predictions.reverse()
    if change == "protocol": predictions[0].metadata["scoring"] = "free_generation"
    if change == "unnormalized": predictions[0].probabilities = [.8, .3]
    with pytest.raises(ValueError): audit.profile(dataset, predictions)


def test_validation_sample_and_demonstrations_are_separate_and_reproducible():
    dataset = data()
    plan = audit.validation_inventory(dataset)
    assert len(plan["proposed_diagnostic_sample_ids"]) == 16
    assert set(plan["proposed_diagnostic_sample_ids"]) <= {r.id for r in dataset.validation}
    assert set(plan["fixed_four_per_class_demonstration_ids"]) <= {r.id for r in dataset.train}
    assert not set(plan["proposed_diagnostic_sample_ids"]) & set(plan["fixed_four_per_class_demonstration_ids"])
    assert audit.validation_inventory(dataset) == plan
    dataset.validation[0] = Row("other-id", dataset.test[0].text.upper(), 0)
    with pytest.raises(ValueError, match="crosses splits"): audit.validation_inventory(dataset)
