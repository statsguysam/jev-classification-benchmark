"""Sensitivity adds constant labels only; cached evidence and ranks stay fixed."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import analyze_review_sensitivity as sensitivity
from jevbench.types import Prediction


def p(row, label, probabilities, error=None):
    return Prediction(row, label, probabilities, error=error,
                      metadata={"probability_kind": "label_sequence_likelihood_normalized"})


def test_constant_labels_with_varying_scores_are_added():
    values = [p("a", 0, [.9, .1]), p("b", 0, [.55, .45])]
    gate = sensitivity.eligibility({"config": {"provider": "hf"}}, values, [0, 0])
    assert gate["eligible"] and not gate["primary_eligible"]
    assert gate["constant_source_predictions"] and gate["n_distinct_source_labels"] == 1
    assert gate["primary_exclusion_reasons"] == ["constant_or_missing_source_labels"]
    assert sensitivity.primary.rank_rows(values, ["a", "b"])[0] == [1, 0]


def test_sensitivity_does_not_admit_hosted_missing_probabilities_or_source_failures():
    values = [p("a", 0, [.9, .1]), p("b", 0, [.55, .45])]
    assert not sensitivity.eligibility({"config": {"provider": "openai"}}, values, [0, 0])["eligible"]
    values[1].probabilities = None
    assert not sensitivity.eligibility({"config": {"provider": "hf"}}, values, [0, 0])["eligible"]
    values[1] = p("b", None, None, "unknown failure")
    gate = sensitivity.eligibility({"config": {"provider": "hf"}}, values, [0, -1])
    assert not gate["eligible"] and not gate["constant_source_predictions"]
    assert "source_failures_prevent_complete_probability_ranking" in gate["exclusion_reasons"]


def test_primary_variable_label_condition_remains_in_both_analyses():
    values = [p("a", 0, [.9, .1]), p("b", 1, [.3, .7])]
    gate = sensitivity.eligibility({"config": {"provider": "hf"}}, values, [0, 1])
    assert gate["eligible"] and gate["primary_eligible"] and not gate["constant_source_predictions"]


def test_constant_score_ties_remain_eligible_and_use_fixed_row_hash():
    values = [p("a", 0, [.7, .3]), p("b", 0, [.7, .3])]
    assert sensitivity.eligibility({"config": {"provider": "hf"}}, values, [0, 0])["eligible"]
    rank, _ = sensitivity.primary.rank_rows(values, ["a", "b"])
    assert rank == sorted(range(2), key=lambda i: sensitivity.hashlib.sha256(["a", "b"][i].encode()).hexdigest())


def test_real_appendix_counts_endpoints_and_primary_artifacts_unchanged(monkeypatch):
    import socket
    def no_connect(*args, **kwargs):
        raise AssertionError("Sensitivity attempted network access")
    monkeypatch.setattr(socket.socket, "connect", no_connect)
    saved = sensitivity.primary.source.read(sensitivity.ROOT / "results/review_value/ANALYSIS.json")
    before = {path: sensitivity.file_sha(path) for path in sensitivity.protected_paths(sensitivity.ROOT, saved)}
    report = sensitivity.collect()
    assert report["counters"] == {"completed_review_conditions_audited": 24, "expected_review_conditions": 24,
        "primary_eligible_conditions": 5, "sensitivity_eligible_conditions": 16,
        "added_constant_label_conditions": 11, "added_constant_conditions_with_varying_max_probability": 11,
        "excluded_complete_conditions": 8, "unscored_incomplete_conditions": 0,
        "scored_curve_points": 96, "partial_condition_metrics": 0}
    assert report["added_condition_accuracy_vs_random_patterns"] == {
        "above_at_all_intermediate_rates": 2, "below_at_all_intermediate_rates": 4,
        "ties_at_all_intermediate_rates": 2, "mixed": 3}
    assert before == {path: sensitivity.file_sha(path) for path in before}
    primary_by_run = {c["review_run_id"]: c for c in saved["conditions"]}
    for c in report["conditions"]:
        original = primary_by_run[c["review_run_id"]]
        points = c["curve"]["points"]
        assert [point["requested_review_percent"] for point in points] == [0, 10, 25, 50, 75, 100]
        assert points[0]["metrics"] == original["metrics"]["never_review"]
        assert points[-1]["metrics"] == original["metrics"]["always_review"]
        assert all(point["required_inferences"]["review_api_requests"] == point["selected_rows"] for point in points)
        if c["eligibility"]["primary_eligible"]:
            assert c["curve"] == original["selective_review"]
            assert c["analysis_membership"] == "primary_and_sensitivity"
        else:
            assert c["analysis_membership"] == "sensitivity_only_constant_labels"
            assert c["eligibility"]["constant_source_predictions"]
    assert all(c["curve"] is None for c in report["excluded_complete_conditions"])
    assert all(c["metrics"] is None for c in report["unscored_incomplete_conditions"])
