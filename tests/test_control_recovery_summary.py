"""Synthetic full-condition recovery tests; no files, models, or paid calls."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import summarize_control_failed_retries as summary
from test_review_controls_summary import study


def recovery(study, requested=(0,), successful=True):
    plan, datasets, originals = copy.deepcopy(study)
    items, attempts = [], []
    for index in requested:
        request, original = plan["requests"][index], originals[index]
        original.label = original.probabilities = None
        original.error = "network_error: fixture"
        digest = summary.runner.preparation.prediction_sha256(asdict(original))
        item = {"failure_id": f"failure-{index}", "control_request_id": request["request_id"],
            "control_arm": request["arm"], "dataset": request["dataset"], "original_prediction_sha256": digest}
        items.append(item)
        for attempt_index in range(1, 2 if successful else 3):
            # Deliberately choose the incorrect but valid label for request 0;
            # response validity, never ground truth, defines a recovery.
            prediction = asdict(original)
            if successful:
                prediction.update(label=1, probabilities=[.1, .9], error=None)
            prediction["metadata"]["retry"] = {"failure_id": item["failure_id"]}
            prediction["metadata"]["post_control"] = {"control_request_id": request["request_id"],
                "dataset": request["dataset"], "arm": request["arm"], "original_control_prediction_sha256": digest}
            attempts.append({"attempt_index": attempt_index, "stage": "retry", "prediction": prediction,
                "identity": {"row_id": original.row_id, "prompt_sha256": request["prompt_sha256"],
                    "original_prediction_sha256": summary.analysis.digest(asdict(original))}})
    audit = {"record": {"status": "complete", "no_op": not items}, "plan": {"inventory": {"failures": items}},
        "attempts": attempts, "budget": {**summary.runner.ZERO, "reservations": len(attempts)}, "artifact_sha256": {"run.json": "fixture"}}
    return plan, datasets, originals, audit


def test_full_denominators_and_wrong_valid_recovery_retained(study):
    args = recovery(study)
    before = copy.deepcopy(args)
    result = summary.assemble(*args, samples=100)
    assert args == before
    assert len(result["runs"]) == 12 and len(result["comparisons"]) == 8
    row = result["runs"][0]
    assert row["n_rows"] == row["first_attempt"]["metrics"]["n_rows"] == row["recovered"]["metrics"]["n_rows"] == 16
    assert row["first_attempt"]["metrics"]["n_failures"] == 1 and row["recovered"]["metrics"]["n_failures"] == 0
    assert row["first_attempt"]["metrics"]["accuracy"] == row["recovered"]["metrics"]["accuracy"] == 15/16
    assert row["resolved_failures"] == 1
    assert row["paired_first_to_recovered"]["metrics"]["accuracy"]["estimate"] == 0
    assert result["recovery_counts"]["primary"]["resolved_failures"] == 1
    assert result["first_attempt_serving_repeat_diagnostic"]["counts"]["both_valid"] == 64
    assert "latency_p50_s" not in json.dumps(result)


def test_correct_recovery_updates_secondary_paired_contrast_only(study):
    args = recovery(study, requested=(1,))
    result = summary.assemble(*args, samples=100)
    row = result["runs"][0]
    assert row["first_attempt"]["metrics"]["accuracy"] == 15/16
    assert row["recovered"]["metrics"]["accuracy"] == 1
    assert row["transitions_first_to_recovered"]["wrong_to_correct"] == 1
    paired = result["comparisons"][0]
    assert paired["first_attempt"]["paired_group_bootstrap"]["metrics"]["accuracy"]["estimate"] == 7/16
    assert paired["recovered"]["paired_group_bootstrap"]["metrics"]["accuracy"]["estimate"] == .5
    assert paired["recovered"]["paired_group_bootstrap"]["n_groups"] == 8


def test_two_failures_stay_incorrect_with_full_denominator(study):
    result = summary.assemble(*recovery(study, successful=False), samples=100)
    row = result["runs"][0]
    assert row["recovered"]["metrics"]["n_failures"] == 1
    assert row["recovered"]["metrics"]["accuracy"] == 15/16
    assert result["new_call_outcomes"] == {"n_calls": 2, "counts": {"transport": 2}}
    assert result["budget"]["reservations"] == 2


def test_repeat_recovery_does_not_replace_primary_repeat_agreement(study):
    result = summary.assemble(*recovery(study, requested=(48,)), samples=100)
    repeated = result["first_attempt_serving_repeat_diagnostic"]
    assert repeated["counts"]["repeat_failed_only"] == 1
    assert repeated["counts"]["both_valid"] == 63
    assert result["recovery_counts"]["repeat"]["resolved_failures"] == 1
    assert result["recovery_counts"]["primary"]["resolved_failures"] == 0


def test_zero_failure_noop_references_identical_primary(study):
    result = summary.assemble(*recovery(study, requested=()), samples=100)
    assert result["no_op"] is True
    assert all(row["first_attempt"] == row["recovered"] for row in result["runs"] + result["comparisons"])
    assert result["new_call_outcomes"]["n_calls"] == 0
    assert "No-op" in summary.findings(result)


@pytest.mark.parametrize("mutate,match", [
    (lambda a: a[3]["record"].update(status="paused"), "not complete"),
    (lambda a: a[3]["plan"]["inventory"]["failures"].clear(), "every original"),
    (lambda a: a[3]["attempts"].clear(), "omitted an original"),
    (lambda a: a[3]["attempts"][0]["identity"].update(prompt_sha256="wrong"), "original request"),
    (lambda a: a[3]["attempts"][0]["prediction"]["metadata"]["post_control"].update(arm="shuffled"), "lineage differs"),
])
def test_malformed_or_partial_evidence_rejected(study, mutate, match):
    args = recovery(study)
    mutate(args)
    with pytest.raises(ValueError, match=match):
        summary.assemble(*args, samples=100)


def test_later_truth_matching_response_cannot_replace_first_valid(study):
    args = recovery(study)
    later = copy.deepcopy(args[3]["attempts"][0])
    later["attempt_index"] = 2
    later["prediction"].update(label=0, probabilities=[.9,.1])
    args[3]["attempts"].append(later)
    with pytest.raises(ValueError, match="after a valid success"):
        summary.assemble(*args, samples=100)


def test_optional_report_withholds_partial_and_rejects_orphans(tmp_path, monkeypatch):
    monkeypatch.setattr(summary.runner, "AREA", tmp_path)
    assert summary.optional_collect() is None
    (tmp_path / "run.json").write_text('{"status":"paused"}')
    assert summary.optional_collect() is None
    (tmp_path / "run.json").unlink()
    (tmp_path / "attempts.jsonl").write_text("")
    with pytest.raises(ValueError, match="Orphan"):
        summary.optional_collect()


def test_collector_calls_both_original_and_retry_audits_and_detects_race(study, monkeypatch, tmp_path):
    plan, datasets, originals, audited = recovery(study)
    count = [0]
    def audit():
        count[0] += 1
        value = copy.deepcopy(audited)
        if count[0] == 2:
            value["artifact_sha256"]["run.json"] = "changed"
        return value
    monkeypatch.setattr(summary.runner, "audit_run", audit)
    monkeypatch.setattr(summary.primary.preparation, "load_frozen_plan", lambda _: plan)
    monkeypatch.setattr(summary.primary, "load_execution", lambda _: (originals, {"status":"complete", "artifact_sha256":{}}))
    monkeypatch.setattr(summary.primary.preparation, "load_jobs", lambda: {name:{"dataset":d} for name,d in datasets.items()})
    monkeypatch.setattr(summary.primary, "OUTPUT", tmp_path)
    (tmp_path / "COMPARISON.json").write_text('{}')
    monkeypatch.setattr(summary.primary, "collect", lambda: {})
    monkeypatch.setattr(summary, "file_sha", lambda _: "fixture")
    with pytest.raises(ValueError, match="changed during analysis"):
        summary.collect(samples=100)
    assert count[0] == 2


def test_stale_linked_primary_report_fails_closed(study, monkeypatch, tmp_path):
    monkeypatch.setattr(summary.runner, "audit_run", lambda: recovery(study)[3])
    monkeypatch.setattr(summary.primary, "OUTPUT", tmp_path)
    (tmp_path / "COMPARISON.json").write_text('{"status":"pending"}')
    monkeypatch.setattr(summary.primary, "collect", lambda: {"status":"complete"})
    with pytest.raises(ValueError, match="Saved primary control report is stale"):
        summary.collect(samples=100)
