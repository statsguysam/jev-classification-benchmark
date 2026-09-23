"""Synthetic figure inputs only: these values are not measured study results."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_review_outcomes as plot


def fixture_report(datasets):
    runs, pairs = [], []
    for dataset in datasets:
        n = plot.DATASETS[dataset][2]
        for model, _ in plot.MODELS:
            for shots in (0, 4):
                ident = f"SYNTHETIC-{dataset}-{model}-{shots}"
                common = {"dataset": dataset, "train_per_class": shots, "status": "complete", "n_test": n,
                          "source_path": f"synthetic/{ident}/run.json", "artifact_sha256": {}}
                source = {**common, "run_id": ident, "model": model, "arm": "base", "n_failures": 1, "accuracy": 13 / n}
                review = {**common, "run_id": ident + "-review", "model": model + "+jev_review", "arm": "review",
                          "source_run_id": ident, "n_failures": 3, "accuracy": 14 / n}
                runs.extend((source, review))
                pairs.append({"kind": "review_minus_source", "dataset": dataset, "a": review["run_id"], "b": ident,
                    "status": "complete", "transitions": {"wrong_to_correct": 4, "correct_to_wrong": 3,
                        "both_correct": 10, "both_wrong": n - 17, "correct_to_wrong_label": 2, "correct_to_failure": 1,
                        "source_failure_rows": 1, "review_stage_failure_rows": 2, "net_correct_change": 1, "accuracy_delta_pp": 100 / n},
                    "paired_bootstrap": {"metrics": {"accuracy": {"estimate": 1 / n, "ci95": [-2 / n, 4 / n]}}}})
        for shots in (0, 4):
            runs.append({"dataset": dataset, "run_id": f"SYNTHETIC-{dataset}-direct-{shots}", "arm": "direct", "status": "complete"})
        for i in range(8):
            runs.append({"dataset": dataset, "run_id": f"SYNTHETIC-{dataset}-classical-{i}", "arm": "classical", "status": "complete"})
    other = [{"kind": "synthetic_other_contrast", "a": f"SYNTHETIC-{i}", "b": "SYNTHETIC-other", "status": "complete"} for i in range(48)]
    return {"status": "complete", "expected_runs": 68, "complete_runs": 68, "datasets": list(datasets),
            "expected_comparisons": 72, "complete_comparisons": 72, "bootstrap_samples": 100,
            "runs": runs, "comparisons": pairs + other}


@pytest.fixture
def reports():
    return [fixture_report(("breast_cancer", "wine")), fixture_report(("sst2", "trec"))]


def test_every_model_shot_and_dataset_present_with_full_denominator(reports):
    panels = plot.prepare(reports)
    assert list(panels) == list(plot.DATASETS)
    assert sum(map(len, panels.values())) == 48
    for dataset, rows in panels.items():
        assert {(r["model"], r["shots_per_class"]) for r in rows} == {(m, s) for m, _ in plot.MODELS for s in (0, 4)}
        for row in rows:
            assert row["n_test"] == plot.DATASETS[dataset][2]
            assert row["corrected"] == 4 and row["harmed_wrong_label"] == 2 and row["harmed_failure"] == 1
            assert row["review_failures_including_upstream_skips"] == 3
            assert row["net_accuracy_pp"] == pytest.approx(100 / row["n_test"])


@pytest.mark.parametrize("mutation", ["pending_count", "pending_row", "duplicates", "missing_source", "wrong_source", "denominator",
                                     "transition_sum", "harm_partition", "failure_damage", "accuracy", "interval", "failure_total"])
def test_invalid_or_incomplete_results_cannot_be_drawn(reports, mutation):
    report = reports[1]
    source, review, pair = report["runs"][0], report["runs"][1], report["comparisons"][0]
    if mutation == "pending_count": report["complete_runs"] = 67
    elif mutation == "pending_row": review["status"] = "pending"
    elif mutation == "duplicates": report["runs"][-1]["run_id"] = source["run_id"]
    elif mutation == "missing_source": source["model"] = "unplanned-model"
    elif mutation == "wrong_source": review["source_run_id"] = "different-source"
    elif mutation == "denominator": review["n_test"] -= 1
    elif mutation == "transition_sum": pair["transitions"]["both_wrong"] -= 1
    elif mutation == "harm_partition": pair["transitions"]["correct_to_wrong_label"] += 1
    elif mutation == "failure_damage": pair["transitions"]["review_stage_failure_rows"] = 0
    elif mutation == "accuracy": review["accuracy"] = 14 / (review["n_test"] - review["n_failures"])
    elif mutation == "interval": pair["paired_bootstrap"]["metrics"]["accuracy"]["ci95"] = [float("nan"), .5]
    elif mutation == "failure_total": review["n_failures"] = 0
    with pytest.raises(ValueError):
        plot.prepare(reports)


def save_reports(root, reports):
    for (name, _), report in zip(plot.SOURCES, reports):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))


def test_partial_text_stops_before_importing_or_reading_either_collector(tmp_path, monkeypatch, reports):
    reports[1]["complete_runs"] = 67
    save_reports(tmp_path, reports)
    monkeypatch.setattr(plot.importlib, "import_module", lambda _: pytest.fail("Partial study must not invoke a collector"))
    with pytest.raises(ValueError, match="68/68"):
        plot.load_audited(tmp_path)
    assert not (tmp_path / "figures").exists()


def test_saved_scores_must_equal_fresh_raw_audit(tmp_path, monkeypatch, reports):
    save_reports(tmp_path, reports)
    module = "summarize_expanded_numeric"
    script = tmp_path / "scripts" / f"{module}.py"
    script.parent.mkdir(); script.write_text("# synthetic collector\n")
    seen = []
    def fresh(*, root, samples):
        seen.append((root, samples))
        altered = copy.deepcopy(reports[0]); altered["status"] = "different"
        return altered
    fake = SimpleNamespace(__file__=str(plot.ROOT / "scripts" / f"{module}.py"), collect=fresh)
    monkeypatch.setattr(plot.importlib, "import_module", lambda _: fake)
    with pytest.raises(ValueError, match="fresh raw-data audit"):
        plot.load_audited(tmp_path)
    assert seen == [(tmp_path, 100)]


def test_evidence_change_after_audit_is_rejected(tmp_path):
    source = tmp_path / "original.json"
    source.write_text("original synthetic content")
    pins = {"original.json": plot.file_sha(source)}
    plot.verify_pins(pins, tmp_path)
    source.write_text("changed synthetic content")
    with pytest.raises(ValueError, match="evidence changed"):
        plot.verify_pins(pins, tmp_path)


@pytest.mark.parametrize("corrected,net", [(2, -1), (3, 0)])
def test_zero_and_negative_net_conditions_are_not_dropped(reports, corrected, net):
    pair = reports[0]["comparisons"][0]
    pair["transitions"].update(wrong_to_correct=corrected, both_wrong=101-corrected, net_correct_change=net, accuracy_delta_pp=100 * net / 114)
    pair["paired_bootstrap"]["metrics"]["accuracy"]["estimate"] = net / 114
    reports[0]["runs"][1]["accuracy"] = (10 + corrected) / 114
    panels = plot.prepare(reports)
    assert panels["breast_cancer"][0]["net_accuracy_pp"] == pytest.approx(100 * net / 114)
    assert len(panels["breast_cancer"]) == 12
