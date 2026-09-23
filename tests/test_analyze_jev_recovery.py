from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_jev_recovery as recovery
from jevbench.types import Prediction, Row


def prediction(row_id, label=None, error=None):
    probabilities = None if error is not None else ([0.9, 0.1] if label == 0 else [0.1, 0.9])
    return asdict(Prediction(row_id, label, probabilities, error=error))


@pytest.fixture
def sample():
    rows = [Row("a", "first numerical row", 0), Row("b", "second numerical row", 1)]
    original = {"a": prediction("a", 0), "b": prediction("b", error="network_error: timeout")}
    identities = {row.id: {"condition_id": "original_run_id", "row_id": row.id,
        "prompt_sha256": "a" * 64, "choices_sha256": "b" * 64, "source_sha256": "c" * 64,
        "original_prediction_sha256": recovery.digest(original[row.id])} for row in rows}
    return rows, ["no", "yes"], original, identities


def attempt(identities, row_id="b", index=1, label=1, error=None, stage="retry"):
    return {"attempt_index": index, "identity": deepcopy(identities[row_id]),
            "prediction": prediction(row_id, label if error is None else None, error), "stage": stage}


def test_earliest_success_wins_even_when_a_later_result_is_correct(sample):
    rows, labels, original, identities = sample
    attempts = [attempt(identities, label=0), attempt(identities, index=2, label=1)]
    untouched = deepcopy((original, identities, attempts))
    report = recovery.analyze_condition(rows, labels, original, identities, attempts)
    assert report["rows"][1]["selected_attempt_index"] == 1
    assert report["rows"][1]["recovered_label"] == 0
    assert report["first_attempt"]["metrics"]["accuracy"] == 0.5
    assert report["recovered"]["metrics"]["accuracy"] == 0.5
    assert report["first_attempt"]["metrics"]["n_failures"] == 1
    assert report["recovered"]["metrics"]["n_failures"] == 0
    assert report["post_success_attempts_ignored"] == 1 and report["retry_policy_deviation"]
    assert (original, identities, attempts) == untouched


def test_failures_remain_visible_when_later_retry_succeeds(sample):
    rows, labels, original, identities = sample
    attempts = [attempt(identities, error="http_error: status=402"), attempt(identities, index=2)]
    report = recovery.analyze_condition(rows, labels, original, identities, attempts)
    assert report["original_snapshot"]["outcome_counts"] == {"success": 1, "transport": 1}
    assert report["first_attempt"]["outcome_counts"] == {"success": 1, "transport": 1}
    assert report["recovered"]["metrics"]["accuracy"] == 1
    assert report["paired_first_to_recovered"]["corrected"] == 1
    assert report["paired_first_to_recovered"]["harmed"] == 0
    assert report["rows"][1]["n_service_attempts"] == 3
    assert len(report["rows"][1]["attempt_prediction_sha256"]) == 3
    assert report["all_service_attempt_outcome_counts"] == {"success": 2, "transport": 1, "billing": 1}


def test_original_success_is_never_replaced(sample):
    rows, labels, original, identities = sample
    report = recovery.analyze_condition(rows, labels, original, identities, [attempt(identities, "a", label=1)])
    assert report["rows"][0]["selected_attempt_index"] == 0
    assert report["rows"][0]["recovered_label"] == 0
    assert report["retry_policy_deviation"]


def test_pending_rows_are_null_until_a_first_call_exists(sample):
    rows, labels, original, identities = sample
    original.pop("b")
    identities["b"]["original_prediction_sha256"] = None
    pending = recovery.analyze_condition(rows, labels, original, identities, [])
    assert pending["first_attempt"]["metrics"] is None and pending["recovered"]["metrics"] is None
    assert pending["recovered"]["n_not_attempted"] == 1
    completed = recovery.analyze_condition(rows, labels, original, identities,
                                           [attempt(identities, stage="first_attempt")])
    assert completed["original_snapshot"]["metrics"] is None
    assert completed["first_attempt"]["metrics"]["accuracy"] == 1
    assert completed["first_attempt"] == completed["recovered"]


def test_upstream_skip_requires_separate_dependent_first_call(sample):
    rows, labels, original, identities = sample
    original["b"] = prediction("b", error="source_proposal_failed: Jev review not called")
    identities["b"]["original_prediction_sha256"] = recovery.digest(original["b"])
    no_call = recovery.analyze_condition(rows, labels, original, identities, [])
    assert no_call["original_snapshot"]["outcome_counts"]["upstream_not_called"] == 1
    assert no_call["first_attempt"]["metrics"] is None
    assert not no_call["original_snapshot_same_request_comparable"]
    with pytest.raises(ValueError, match="Attempt stage"):
        recovery.analyze_condition(rows, labels, original, identities, [attempt(identities)])
    completed = recovery.analyze_condition(rows, labels, original, identities,
        [attempt(identities, stage="dependent_first_call")],
        original_stages={"a": "attempted", "b": "upstream_not_called"})
    assert completed["first_attempt"]["metrics"]["accuracy"] == 1
    assert completed["original_snapshot"]["metrics"]["accuracy"] == 0.5
    assert completed["rows"][1]["n_service_attempts"] == 1


@pytest.mark.parametrize("field", sorted(recovery.IDENTITY_FIELDS))
def test_request_or_source_changes_are_rejected(sample, field):
    rows, labels, original, identities = sample
    record = attempt(identities)
    record["identity"][field] = "wrong" if field in {"condition_id", "row_id"} else "d" * 64
    with pytest.raises(ValueError, match="identity|Unknown attempt row"):
        recovery.analyze_condition(rows, labels, original, identities, [record])


@pytest.mark.parametrize("indices", [[2], [1, 1], [1, 3], [2, 1]])
def test_attempt_order_cannot_be_sorted_or_deduplicated_after_the_fact(sample, indices):
    rows, labels, original, identities = sample
    with pytest.raises(ValueError, match="append-ordered"):
        recovery.analyze_condition(rows, labels, original, identities,
                                  [attempt(identities, index=i, error="network_error: timed out") for i in indices])


@pytest.mark.parametrize("mutation", ["nonfinite", "sum", "argmax", "label", "error_with_label", "missing_probs"])
def test_malformed_predictions_cannot_be_reclassified_as_success(sample, mutation):
    rows, labels, original, identities = sample
    record = attempt(identities)
    if mutation == "nonfinite": record["prediction"]["probabilities"] = [float("nan"), 1]
    if mutation == "sum": record["prediction"]["probabilities"] = [0.1, 0.5]
    if mutation == "argmax": record["prediction"]["probabilities"] = [0.9, 0.1]
    if mutation == "label": record["prediction"]["label"] = 2
    if mutation == "error_with_label": record["prediction"]["error"] = "network_error: timeout"
    if mutation == "missing_probs": record["prediction"]["probabilities"] = None
    with pytest.raises(ValueError):
        recovery.analyze_condition(rows, labels, original, identities, [record])


def test_original_digest_and_explicit_stages_must_match(sample):
    rows, labels, original, identities = sample
    identities["b"]["original_prediction_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="Original prediction digest"):
        recovery.analyze_condition(rows, labels, original, identities, [])
    identities["b"]["original_prediction_sha256"] = recovery.digest(original["b"])
    with pytest.raises(ValueError, match="Original stage"):
        recovery.analyze_condition(rows, labels, original, identities, [],
                                  original_stages={"a": "attempted", "b": "not_attempted"})


def test_available_original_review_prompt_must_match_retry(sample):
    rows, labels, original, identities = sample
    original["b"]["metadata"]["review_prompt_sha256"] = "d" * 64
    identities["b"]["original_prediction_sha256"] = recovery.digest(original["b"])
    with pytest.raises(ValueError, match="Original review prompt"):
        recovery.analyze_condition(rows, labels, original, identities, [attempt(identities)])


def test_absent_truth_class_has_no_balanced_accuracy_claim(sample):
    rows, labels, original, identities = sample
    report = recovery.analyze_condition(rows[:1], labels, {"a": original["a"]}, {"a": identities["a"]}, [])
    assert report["recovered"]["metrics"]["balanced_accuracy"] is None
    assert report["recovered"]["metrics"]["accuracy"] == 1


@pytest.mark.parametrize("probabilities", [[0.5, 0.5], [0.5000004, 0.4999996], [0.5000005, 0.4999995]])
def test_jev_accepts_nonfirst_ties_and_frozen_provider_decision_tolerance(probabilities):
    value = asdict(Prediction("synthetic-row", 1, probabilities))
    unchanged = deepcopy(value)
    # This is the comparison used by the frozen JevClassifier, not argmax.
    assert not probabilities[1] < max(probabilities) - 1e-6
    assert recovery.validate_prediction(value, "synthetic-row", 2, require_probabilities=True)
    assert value == unchanged


def test_jev_rejects_a_selected_probability_beyond_provider_tolerance():
    value = asdict(Prediction("synthetic-row", 1, [0.500001, 0.499999]))
    with pytest.raises(ValueError, match="decision tolerance"):
        recovery.validate_prediction(value, "synthetic-row", 2, require_probabilities=True)


def test_optional_probability_branch_keeps_exact_maximum_semantics():
    tied = asdict(Prediction("synthetic-row", 1, [0.5, 0.5]))
    assert recovery.validate_prediction(tied, "synthetic-row", 2, require_probabilities=False)
    near = asdict(Prediction("synthetic-row", 1, [0.5000004, 0.4999996]))
    with pytest.raises(ValueError, match="probability maximum"):
        recovery.validate_prediction(near, "synthetic-row", 2, require_probabilities=False)


def test_jev_decision_tolerance_does_not_relax_probability_normalization():
    invalid = asdict(Prediction("synthetic-row", 1, [0.5000015, 0.5000015]))
    with pytest.raises(ValueError, match="probability vector"):
        recovery.validate_prediction(invalid, "synthetic-row", 2, require_probabilities=True)


def test_accepted_nearmax_retry_is_the_first_success_without_relabeling(sample):
    rows, labels, original, identities = sample
    record = attempt(identities, label=1)
    record["prediction"]["probabilities"] = [0.5000004, 0.4999996]
    unchanged = deepcopy(record)
    report = recovery.analyze_condition(rows, labels, original, identities, [record])
    assert report["rows"][1]["selected_attempt_index"] == 1
    assert report["rows"][1]["recovered_label"] == 1
    assert report["recovered"]["metrics"]["n_failures"] == 0
    assert record == unchanged
