"""Regression coverage for the shared numeric/text dashboard projection."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(params=[
    ("build_expanded_numeric_dashboard", "numeric_expansion", "numeric-data.json"),
    ("build_text_extension_dashboard", "text_extension", "text-data.json"),
])
def study(request):
    module, report_directory, asset_name = request.param
    exporter = importlib.import_module(module)
    report_path = Path("results") / report_directory / "COMPARISON.json"
    report = json.loads((ROOT / report_path).read_text())
    return exporter, report, report_path, asset_name


def encoded(payload):
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_shared_projection_preserves_saved_dashboard_payload(study):
    exporter, report, _, asset_name = study
    saved = json.loads((ROOT / "dashboard/dist" / asset_name).read_text())
    saved.pop("provenance")
    assert encoded(exporter.sanitize(report)) == encoded(saved)


def pause_one_review(exporter, report, status):
    """Represent one audited, unscored checkpoint with its dependent contrasts."""
    report = copy.deepcopy(report)
    row = next(row for row in report["runs"] if row["arm"] == "review")
    key = exporter.scientific.row_key(row)
    replacement = exporter.scientific.placeholder(key)
    replacement.update(status=status, run_id=row["run_id"], source_run_id=row["source_run_id"])
    report["runs"][report["runs"].index(row)] = replacement
    for contrast in report["comparisons"]:
        if row["run_id"] in (contrast["a"], contrast["b"]):
            contrast.update(status="pending", paired_bootstrap=None)
            contrast.pop("transitions", None)
    report.update(
        status="in_progress_or_incomplete",
        complete_runs=report["complete_runs"] - 1,
        complete_comparisons=sum(row["status"] == "complete" for row in report["comparisons"]),
    )
    report["costs"].update(
        status="halted",
        complete_summary_conditions=report["costs"]["complete_summary_conditions"] - 1,
        complete_checkpoint_conditions=report["costs"]["complete_checkpoint_conditions"] - 1,
    )
    return report, row["run_id"]


def test_explicit_partial_projection_keeps_unscored_conditions_and_links(study):
    exporter, report, _, _ = study
    report, run_id = pause_one_review(exporter, report, "stopped_after_three_consecutive_errors")
    with pytest.raises(ValueError, match="all 68 conditions"):
        exporter.sanitize(report)
    result = exporter.sanitize(report, allow_incomplete=True)
    pending = next(row for row in result["runs"] if row["run_id"] == run_id)
    assert pending["accuracy"] is None
    assert pending["n_failures"] is None
    assert pending["review_transitions"] is None
    source = next(row for row in result["runs"] if row["run_id"] == pending["source_run_id"])
    assert source["status"] == "complete"
    affected = [row for row in result["comparisons"] if run_id in (row["a"], row["b"])]
    assert affected
    assert all(row["status"] == "pending" and row["accuracy_delta"] is None for row in affected)


def test_billing_pause_remains_specific_to_the_text_study(study):
    exporter, report, _, asset_name = study
    report, _ = pause_one_review(exporter, report, "stopped_after_billing_error")
    if asset_name == "numeric-data.json":
        with pytest.raises(ValueError, match="Unexpected run status"):
            exporter.sanitize(report, allow_incomplete=True)
    else:
        result = exporter.sanitize(report, allow_incomplete=True)
        assert any(row["status"] == "stopped_after_billing_error" for row in result["runs"])


def test_shared_projection_preserves_each_studys_cost_scope(study):
    exporter, report, _, _ = study
    report["costs"]["reused_old_review_conditions"] += 1
    with pytest.raises(ValueError, match="Unexpected cost scope"):
        exporter.sanitize(report)


def test_new_exports_record_both_exporter_sources(study, tmp_path, monkeypatch):
    exporter, report, report_path, _ = study
    source = tmp_path / report_path
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(report))
    monkeypatch.setattr(exporter.scientific, "collect", lambda *args, **kwargs: copy.deepcopy(report))
    result = exporter.build(tmp_path, tmp_path / "export.json")
    helper = ROOT / "scripts/dashboard_aggregate.py"
    assert result["provenance"]["exporter_sha256"] == hashlib.sha256(Path(exporter.__file__).read_bytes()).hexdigest()
    assert result["provenance"]["exporter_helper_sha256"] == hashlib.sha256(helper.read_bytes()).hexdigest()
