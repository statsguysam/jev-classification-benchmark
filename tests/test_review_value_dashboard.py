"""Aggregate-only publication checks; no audits, model calls or network needed."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_review_value_dashboard as dashboard


@pytest.fixture
def reports():
    return (json.loads((ROOT / "results/review_value/ANALYSIS.json").read_text()),
            json.loads((ROOT / "results/review_controls/COMPARISON.json").read_text()))


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def test_real_export_has_exact_complete_inventory_and_full_coverage_curves(reports):
    report, controls = reports
    output = dashboard.project(report, controls)
    assert output["complete_review_conditions"] == len(output["conditions"]) == 17
    assert output["expected_review_conditions"] == 24
    assert len(output["pending_conditions"]) == 7
    keys = [(r["dataset"], r["source_model"], r["shots_per_class"]) for r in output["conditions"]]
    assert len(set(keys)) == 17
    gates = [r for r in output["conditions"] if r["curve"] is not None]
    assert len(gates) == 4
    assert all(c["selective_review_eligibility"]["eligible"] == (c["curve"] is not None) for c in output["conditions"])
    for gate in gates:
        assert [p["requested_review_percent"] for p in gate["curve"]] == [0, 10, 25, 50, 75, 100]
        assert gate["curve"][0]["metrics"] == gate["metrics"]["never_review"]
        assert gate["curve"][-1]["metrics"] == gate["metrics"]["always_review"]
        assert all(p["required_inferences"]["review_api_requests"] == p["selected_rows"] for p in gate["curve"])
    for exported, original in zip(output["conditions"], report["conditions"]):
        for arm in ("never_review", "always_review", "direct_jev"):
            for metric in ("micro_accuracy", "balanced_accuracy", "macro_f1", "n_rows", "n_failures"):
                assert exported["metrics"][arm][metric] == original["metrics"][arm][metric]
    assert output["identical_prompts"] == {"same_label_identical_prompt_rows": 1260,
        "both_valid_rows": 1255, "valid_output_disagreements": 41, "rows_with_either_review_failure": 5}
    # Wine Astra's inherited source failure must not become a review call.
    skipped = next(c for c in output["conditions"] if c["dataset"] == "wine" and c["source_model"] == "gpt-6-astra" and c["shots_per_class"] == 4)
    assert skipped["required_inferences"]["always_review"]["review_api_requests"] == 35
    assert skipped["metrics"]["always_review"]["n_failures"] == 1


def test_all_twelve_pending_controls_keep_null_metrics(reports):
    output = dashboard.project(*reports)
    assert len(output["controls"]["runs"]) == 12
    assert len({(r["dataset"], r["arm"]) for r in output["controls"]["runs"]}) == 12
    assert all(r["status"] == "pending" and r["metrics"] is None for r in output["controls"]["runs"])
    assert output["controls"]["planned_primary_requests"] == 1650
    assert output["controls"]["planned_repeats"] == 64


def test_export_never_contains_current_raw_identifiers_or_prompt_fields(reports):
    output = dashboard.project(*reports)
    forbidden = {"row_id", "row_ids", "test_row_ids", "selected_row_ids", "ranked_row_ids", "prompt",
        "source_max_probabilities", "source_run_id", "review_run_id", "request_id", "source_path",
        "config", "api_key", "artifacts", "files_sha256", "disagreement_row_ids", "prompt_hashes"}
    assert not any(forbidden & node.keys() for node in walk(output))
    blob = json.dumps(output)
    assert "source:00" not in blob and "Cached proposal class ID" not in blob


def test_nested_allowlist_omits_unexpected_sensitive_fields(reports):
    report, controlled = copy.deepcopy(reports)
    sentinel = "PRIVATE_PROMPT_ROW_SENTINEL"
    report["private_metadata"] = sentinel
    condition = report["conditions"][0]
    condition.update(prompt=sentinel, row_ids=[sentinel])
    condition["metrics"]["never_review"].update(prompt=sentinel, row_ids=[sentinel])
    condition["review_minus_base"].update(prompt=sentinel, row_id=sentinel)
    condition["selective_review_eligibility"]["raw_rows"] = [sentinel]
    point = next(c for c in report["conditions"] if c["selective_review"])["selective_review"]["points"][0]
    point.update(prompt=sentinel, selected_row_ids=[sentinel])
    point["metrics"].update(prompt=sentinel, row_id=sentinel)
    point["random_matched_rate"].update(prompt=sentinel, selected_row_ids=[sentinel])
    point["transitions_from_never_review"]["row_ids"] = [sentinel]
    point["required_inferences"]["request_ids"] = [sentinel]
    controlled["runs"][0].update(status="complete", metrics={"accuracy": .5, "balanced_accuracy": .5,
        "macro_f1": .5, "n_test": 114, "n_failures": 0, "prompt": sentinel, "row_ids": [sentinel]})
    output = dashboard.project(report, controlled)
    assert sentinel not in json.dumps(output)
    assert output["controls"]["runs"][0]["metrics"]["accuracy"] == .5


def test_projection_does_not_mutate_reports(reports):
    before = copy.deepcopy(reports)
    dashboard.project(*reports)
    assert reports == before
