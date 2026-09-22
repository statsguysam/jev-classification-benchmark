"""Synthetic complete aggregates test safety and publication gates, not accuracy."""
import copy
from decimal import Decimal
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_expanded_numeric_dashboard as dashboard


def bootstrap(n_test, *, paired=False):
    return {"method": "paired unstratified group percentile bootstrap" if paired else "unstratified group percentile bootstrap",
        "samples": 2000, "seed": 42, "n_rows": n_test, "n_groups": n_test,
        "metrics": {name: {"estimate": 0.0 if paired else .5, "ci95": [-.1, .1] if paired else [.4, .6]}
                    for name in ("accuracy", "macro_f1")}}


@pytest.fixture
def complete_report():
    rows, by_key = [], {}
    for dataset, model, arm, budget in dashboard.scientific.expected_conditions():
        descriptor = dashboard.DATASETS[dataset]
        n_test, n_classes = descriptor["n_test"], descriptor["n_classes"]
        key = dataset, model, arm, budget
        row = dashboard.scientific.placeholder(key)
        row.update(status="complete", arm=arm, run_id=f"run-{len(rows)}", n_test=n_test, n_classes=n_classes,
            train_labels=descriptor["full_training_labels"] if budget is None else budget*n_classes,
            accuracy=.5, macro_f1=.5, balanced_accuracy=.5, n_failures=0, failure_rate=0., probability_coverage=1.,
            log_loss=.7, brier_sum=.5, group_bootstrap=bootstrap(n_test),
            source_path="/private/PRIVATE_SENTINEL/run.json", config={"token": "PRIVATE_SENTINEL"},
            raw_rows=["PRIVATE_SENTINEL"], prompt="PRIVATE_SENTINEL")
        if arm == "review":
            row["review_transitions"] = {name: 0 for name in dashboard.TRANSITION_COUNTS}
            row["review_transitions"].update(both_correct=n_test//2, both_wrong=n_test//2,
                                              net_correct_change=0, accuracy_delta_pp=0.)
        rows.append(row); by_key[key] = row
    for key, row in by_key.items():
        if key[2] == "review":
            row["source_run_id"] = by_key[key[0], key[1], "base", key[3]]["run_id"]
    comparisons = []
    for kind, a, b, equal in dashboard.scientific.comparison_specs():
        item = {"kind": kind, "dataset": a[0], "a": by_key[a]["run_id"], "b": by_key[b]["run_id"],
            "status": "complete", "train_per_class_a": a[3], "train_per_class_b": b[3],
            "equal_new_label_budget": equal, "paired_bootstrap": bootstrap(by_key[a]["n_test"], paired=True),
            "prompt": "PRIVATE_SENTINEL"}
        if kind == "review_minus_source":
            item["transitions"] = copy.deepcopy(by_key[a]["review_transitions"])
        comparisons.append(item)
    condition_costs = []
    for dataset in dashboard.DATASETS:
        for key, model in dashboard.MODEL_KEYS.items():
            for shots in (0, 4):
                if model in dashboard.scientific.REUSED_REVIEW_MODELS and shots == 4:
                    continue
                condition_costs.append({"dataset": dataset, "model_key": key, "shots_per_class": shots,
                    "model_requests": 1, "unknown_cost_requests": 0, "known_reported_api_usd": ".01",
                    "conservative_usd": ".1", "request_id": "PRIVATE_SENTINEL"})
    condition_costs[-1].update(unknown_cost_requests=1, known_reported_api_usd="0")
    costs = {"status": "complete", "complete_checkpoint_conditions": 20, "complete_summary_conditions": 20,
        "expected_new_review_conditions": 20, "reservations_without_result": 0,
        "results_without_saved_prediction": 0, "partial_checkpoint_files": 0,
        "new_model_requests": 20, "reported_cost_requests": 19, "unknown_cost_requests": 1,
        "known_reported_api_usd": ".19", "review_conservative_usd": "2", "prior_conservative_usd": "19",
        "cumulative_conservative_usd": "21", "cumulative_authorized_usd": "25", "new_llm_generation_calls": 0,
        "source_failure_rows_without_jev_call": 0, "reused_old_review_conditions": 4,
        "condition_costs": condition_costs, "ledger_id": "PRIVATE_SENTINEL"}
    return {"status": "complete", "complete_runs": 68, "expected_runs": 68, "complete_comparisons": 72,
        "expected_comparisons": 72, "bootstrap_samples": 2000, "seed": 42, "runs": rows,
        "comparisons": comparisons, "costs": costs, "source_sha256": "a"*64, "protocol_sha256": "b"*64,
        "private_metadata": "PRIVATE_SENTINEL"}


def test_allowlist_omits_private_fields_and_preserves_complete_filters_costs(complete_report):
    result = dashboard.sanitize(complete_report)
    assert len(result["runs"]) == 68 and len(result["comparisons"]) == 72
    assert "PRIVATE_SENTINEL" not in json.dumps(result)
    forbidden = {"prompt", "config", "source_path", "raw_rows", "request_id", "ledger_id"}
    def visit(value):
        if isinstance(value, dict):
            assert not set(value) & forbidden
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(result)
    assert {item["task"] for item in result["datasets"]} == {"binary", "multiclass"}
    assert {row["jev_mode"] for row in result["runs"]} == {"none", "alone", "review"}
    full = [row for row in result["runs"] if row["label_budget"] == "full_training"]
    assert len(full) == 8 and all(row["shots_per_class"] is None for row in full)
    assert result["costs"]["unknown_cost_calls"] == 1
    assert Decimal(result["costs"]["known_reported_api_usd"]) == Decimal(".19")
    assert result["costs"]["local_compute_usd"] is None


def test_percentile_endpoints_are_preserved_even_when_point_estimate_is_outside(complete_report):
    complete_report["runs"][0]["group_bootstrap"]["metrics"]["accuracy"]["ci95"] = [.6, .7]
    complete_report["comparisons"][0]["paired_bootstrap"]["metrics"]["accuracy"]["ci95"] = [.1, .2]
    result = dashboard.sanitize(complete_report)
    assert result["runs"][0]["accuracy"] == .5 and result["runs"][0]["accuracy_ci95"] == [.6, .7]
    assert result["comparisons"][0]["accuracy_delta"] == 0 and result["comparisons"][0]["accuracy_delta_ci95"] == [.1, .2]


@pytest.mark.parametrize("change,match", [
    ("duplicate", "duplicate run"), ("nan", "Nonfinite"), ("ci", "Reversed"),
    ("cost", "USD totals"), ("partial_cost", "incomplete"),
    ("budget", "label budgets"), ("harm", "decomposition"), ("source", "Review source")])
def test_invalid_aggregates_fail_closed(complete_report, change, match):
    report = complete_report
    review = next(row for row in report["runs"] if row["arm"] == "review")
    if change == "duplicate": report["runs"][-1] = copy.deepcopy(report["runs"][0])
    elif change == "nan": report["runs"][0]["accuracy"] = float("nan")
    elif change == "ci": report["runs"][0]["group_bootstrap"]["metrics"]["accuracy"]["ci95"] = [.8, .2]
    elif change == "cost": report["costs"]["cumulative_conservative_usd"] = "0"
    elif change == "partial_cost": report["costs"]["status"] = "in_progress"
    elif change == "budget":
        next(row for row in report["comparisons"] if row["kind"] == "few_minus_zero_descriptive")["equal_new_label_budget"] = True
    elif change == "harm": review["review_transitions"]["correct_to_failure"] = 1
    else: review["source_run_id"] = "wrong-source"
    with pytest.raises(ValueError, match=match): dashboard.sanitize(report)


def test_partial_or_stale_report_cannot_replace_existing_asset(tmp_path, monkeypatch, complete_report):
    source = tmp_path / "results/numeric_expansion/COMPARISON.json"
    source.parent.mkdir(parents=True)
    output = tmp_path / "dashboard/dist/numeric-data.json"
    output.parent.mkdir(parents=True); output.write_text("existing")
    partial = copy.deepcopy(complete_report); partial["complete_runs"] = 67
    source.write_text(json.dumps(partial))
    monkeypatch.setattr(dashboard.scientific, "collect", lambda *a, **k: pytest.fail("Partial report should fail before audit"))
    with pytest.raises(ValueError, match="68 conditions"): dashboard.build(tmp_path)
    assert output.read_text() == "existing"
    source.write_text(json.dumps(complete_report))
    different = copy.deepcopy(complete_report); different["runs"][0]["accuracy"] = .4
    monkeypatch.setattr(dashboard.scientific, "collect", lambda *a, **k: different)
    with pytest.raises(ValueError, match="independently revalidated"): dashboard.build(tmp_path)
    assert output.read_text() == "existing"
    monkeypatch.setattr(dashboard.scientific, "collect", lambda *a, **k: complete_report)
    dashboard.build(tmp_path)
    saved = json.loads(output.read_text())
    assert saved["completion"]["complete_runs"] == 68
    assert len(saved["provenance"]["source_report_sha256"]) == 64
    assert "PRIVATE_SENTINEL" not in output.read_text()


def paused_report(complete_report):
    report = copy.deepcopy(complete_report)
    rename = {}
    for index, row in enumerate(report["runs"]):
        key = dashboard.scientific.row_key(row)
        dataset, model, arm, shots = key
        if arm != "review" or model not in {dashboard.MODEL_KEYS["smollm2"], dashboard.MODEL_KEYS["granite"]}:
            continue
        if dataset == "breast_cancer" and model == dashboard.MODEL_KEYS["smollm2"] and shots == 0:
            continue
        pending = dashboard.scientific.placeholder(key)
        pending["source_run_id"] = row["source_run_id"]
        if dataset == "breast_cancer" and model == dashboard.MODEL_KEYS["smollm2"]:
            pending.update(status="stopped_after_three_consecutive_errors", run_id=row["run_id"])
        rename[row["run_id"]] = pending["run_id"]
        report["runs"][index] = pending
    for comparison in report["comparisons"]:
        incomplete = comparison["a"] in rename or comparison["b"] in rename
        comparison["a"] = rename.get(comparison["a"], comparison["a"])
        comparison["b"] = rename.get(comparison["b"], comparison["b"])
        if incomplete:
            comparison.update(status="pending", paired_bootstrap=None)
            comparison.pop("transitions", None)
    report.update(status="in_progress_or_incomplete", complete_runs=61,
                  complete_comparisons=sum(row["status"] == "complete" for row in report["comparisons"]))
    report["costs"].update(status="halted", complete_checkpoint_conditions=13, complete_summary_conditions=13,
                            reported_cost_requests=13, unknown_cost_requests=7, known_reported_api_usd=".13")
    for item in report["costs"]["condition_costs"][-7:]:
        item.update(unknown_cost_requests=1, known_reported_api_usd="0")
    return report


def test_explicit_partial_mode_retains_all_rows_null_scores_links_and_actual_costs(complete_report):
    report = paused_report(complete_report)
    with pytest.raises(ValueError, match="68 conditions"):
        dashboard.sanitize(report)
    result = dashboard.sanitize(report, allow_incomplete=True)
    assert result["completion"] == {"status": "in_progress_or_incomplete", "complete_runs": 61,
        "expected_runs": 68, "complete_comparisons": 54, "expected_comparisons": 72}
    assert len(result["runs"]) == 68 and len(result["comparisons"]) == 72
    by_id = {row["run_id"]: row for row in result["runs"]}
    pending = [row for row in result["runs"] if row["status"] != "complete"]
    assert len(pending) == 7
    assert sum(row["status"] == "stopped_after_three_consecutive_errors" for row in pending) == 1
    assert "breast_cancer__granite__k0__jev-review" in by_id
    null_fields = ("accuracy", "macro_f1", "accuracy_ci95", "macro_f1_ci95", "balanced_accuracy", "n_failures", "failure_rate",
                   "n_test", "train_labels", "probability_coverage", "log_loss", "brier_sum", "review_transitions",
                   "ci_method", "ci_samples", "ci_n_groups")
    for row in pending:
        assert all(row[name] is None for name in null_fields)
        base = by_id[row["source_run_id"]]
        assert base["status"] == "complete" and base["model"] == row["source_model"]
        assert row["train_per_class"] == base["train_per_class"] and row["dataset"] == base["dataset"]
    for comparison in result["comparisons"]:
        assert comparison["a"] in by_id and comparison["b"] in by_id
        if comparison["status"] == "pending":
            assert all(comparison[name] is None for name in ("accuracy_delta", "accuracy_delta_ci95", "macro_f1_delta",
                "macro_f1_delta_ci95", "transitions", "ci_method", "ci_samples", "ci_n_groups"))
    assert result["costs"]["status"] == "halted" and result["costs"]["unknown_cost_calls"] == 7
    assert result["costs"]["new_review_calls"] == 20 and result["costs"]["complete_review_conditions"] == 13
    assert Decimal(result["costs"]["known_reported_api_usd"]) == Decimal(".13")
    assert "Incomplete operational snapshot" in result["limitations"][0]
    assert "PRIVATE_SENTINEL" not in json.dumps(result)


def test_partial_mode_still_refuses_partial_scores_and_incorrect_links(complete_report):
    for field, value, match in (("accuracy", .9, "partial score"), ("n_failures", 3, "partial score"),
                                ("source_run_id", "wrong-source", "Review source")):
        report = paused_report(complete_report)
        pending = next(row for row in report["runs"] if row["status"] != "complete")
        pending[field] = value
        with pytest.raises(ValueError, match=match):
            dashboard.sanitize(report, allow_incomplete=True)
    assert dashboard.sanitize(complete_report) == dashboard.sanitize(complete_report, allow_incomplete=True)


def test_partial_build_requires_explicit_flag_and_still_reaudits(tmp_path, monkeypatch, complete_report):
    report = paused_report(complete_report)
    source = tmp_path / "results/numeric_expansion/COMPARISON.json"
    source.parent.mkdir(parents=True); source.write_text(json.dumps(report))
    calls = []
    def audit(*args, **kwargs):
        calls.append(kwargs)
        return report
    monkeypatch.setattr(dashboard.scientific, "collect", audit)
    with pytest.raises(ValueError, match="68 conditions"):
        dashboard.build(tmp_path)
    assert not calls and not (tmp_path / "dashboard/dist/numeric-data.json").exists()
    result = dashboard.build(tmp_path, allow_incomplete=True)
    assert calls == [{"samples": 2000}]
    assert json.loads((tmp_path / "dashboard/dist/numeric-data.json").read_text()) == result


def test_in_progress_cost_snapshot_keeps_outstanding_attempt_counts(complete_report):
    report = paused_report(complete_report)
    report["costs"].update(status="in_progress", reservations_without_result=1, partial_checkpoint_files=1)
    next(row for row in report["runs"] if row["status"] == "stopped_after_three_consecutive_errors")["status"] = "running"
    result = dashboard.sanitize(report, allow_incomplete=True)
    assert result["costs"]["status"] == "in_progress"
    assert result["costs"]["reservations_without_result"] == result["costs"]["partial_checkpoint_files"] == 1
    assert result["costs"]["unknown_cost_calls"] == 7 and result["costs"]["new_review_calls"] == 20
    assert next(row for row in result["runs"] if row["status"] == "running")["accuracy"] is None
