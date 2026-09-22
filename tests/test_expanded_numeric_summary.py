import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/"scripts"))
import summarize_expanded_numeric as summary
from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import digest
from jevbench.types import Prediction, PreparedDataset, Row


def test_declared_inventory_has_68_unique_conditions_and_72_contrasts():
    keys = summary.expected_conditions()
    assert len(keys) == len(set(keys)) == 68
    assert {arm: sum(k[2] == arm for k in keys) for arm in ("base", "review", "direct", "classical")} == {
        "base": 24, "review": 24, "direct": 4, "classical": 16}
    comparisons = summary.comparison_specs()
    assert len(comparisons) == 72
    assert sum(kind == "review_minus_source" for kind, *_ in comparisons) == 24
    assert sum(kind == "review_minus_jev_alone" for kind, *_ in comparisons) == 24
    assert sum(not equal for *_, equal in comparisons) == 24
    assert all(a in keys and b in keys for _, a, b, _ in comparisons)


def test_all_missing_conditions_are_visible_and_unscored(tmp_path):
    report = summary.assemble({}, {}, samples=100)
    assert report["complete_runs"] == report["complete_comparisons"] == 0
    assert report["status"] == "in_progress_or_incomplete"
    assert len(report["runs"]) == 68
    assert all(row["accuracy"] is None and row["group_bootstrap"] is None for row in report["runs"])
    assert all(item["paired_bootstrap"] is None for item in report["comparisons"])
    summary.write(report, tmp_path)
    output = tmp_path/"results/numeric_expansion"
    assert (output/"COMPARISON.csv").read_text().count("\n") == 69
    text = (output/"FINDINGS.md").read_text()
    assert "0/68" in text and "pending" in text
    assert "arbitrary class meaning" in text and "not supervised learnability" in text


def rows_and_payloads(shots=4):
    model = "gpt-6-astra"
    keys = [("breast_cancer", model, arm, shots) for arm in ("base", "review")]
    keys.append(("breast_cancer", summary.JEV, "direct", shots))
    rows, payloads = {}, {}
    predicted = ([0, 1, -1, 0], [-1, 1, -1, 1], [0, 1, 0, 1])
    for key, values in zip(keys, predicted):
        row = summary.placeholder(key)
        row.update(status="complete", run_id=key[2], n_test=4, train_labels=8 if shots else 0,
                   accuracy=sum(a == b for a, b in zip([0, 1, 0, 1], values))/4, macro_f1=.5)
        payloads[key] = {"y": [0, 1, 0, 1], "predictions": list(values), "groups": ["a", "b", "c", "d"],
            "n_classes": 2, "test_row_ids": ["a", "b", "c", "d"],
            "training_example_ids": [f"train-{i}" for i in range(8)] if shots else [], "manifest_sha256": "same"}
        rows[key] = row
    rows[keys[1]]["source_run_id"] = "base"
    return rows, payloads


@pytest.mark.parametrize("shots", [0, 4])
def test_review_deltas_keep_source_and_reviewer_failures_and_match_shot_budget(shots):
    rows, payloads = rows_and_payloads(shots)
    report = summary.assemble(rows, payloads, samples=100)
    done = [item for item in report["comparisons"] if item["status"] == "complete"]
    assert len(done) == 2
    contrast = next(item for item in done if item["kind"] == "review_minus_source")
    counts = contrast["transitions"]
    assert counts["wrong_to_correct"] == 1
    assert counts["correct_to_wrong"] == counts["correct_to_failure"] == 1
    assert counts["correct_to_wrong_label"] == 0
    assert counts["source_failure_rows"] == 1 and counts["review_stage_failure_rows"] == 1
    assert counts["net_correct_change"] == 0
    assert contrast["paired_bootstrap"]["metrics"]["accuracy"]["estimate"] == 0
    jev = next(item for item in done if item["kind"] == "review_minus_jev_alone")
    assert jev["train_per_class_a"] == jev["train_per_class_b"] == shots
    assert jev["paired_bootstrap"]["metrics"]["accuracy"]["estimate"] == -.5


def test_mismatched_heldout_rows_or_matched_training_ids_are_rejected():
    for field, change, match in (("test_row_ids", ["wrong"]*4, "held-out"),
                                ("training_example_ids", ["extra"], "training IDs")):
        rows, payloads = rows_and_payloads()
        payloads["breast_cancer", "gpt-6-astra", "review", 4][field] = change
        with pytest.raises(ValueError, match=match):
            summary.assemble(rows, payloads, samples=100)


def test_wrong_or_unavailable_source_link_is_rejected():
    rows, payloads = rows_and_payloads()
    rows["breast_cancer", "gpt-6-astra", "review", 4]["source_run_id"] = "another-source"
    with pytest.raises(ValueError, match="unavailable/different source"):
        summary.assemble(rows, payloads, samples=100)


def test_incomplete_review_does_not_receive_transitions_or_scores():
    rows, payloads = rows_and_payloads()
    key = "breast_cancer", "gpt-6-astra", "review", 4
    rows[key] = summary.placeholder(key)
    rows[key].update(status="running", run_id="partial-review")
    report = summary.assemble(rows, payloads, samples=100)
    row = next(row for row in report["runs"] if row["run_id"] == "partial-review")
    assert row["accuracy"] is None and "review_transitions" not in row
    assert report["complete_review_runs"] == 0


def test_few_minus_zero_is_explicitly_unequal_labels():
    rows, payloads = rows_and_payloads(0)
    few_rows, few_payloads = rows_and_payloads(4)
    # Real run IDs differ by shot count.
    for row in rows.values():
        row["run_id"] += "-zero"
        if "source_run_id" in row:
            row["source_run_id"] += "-zero"
    rows.update(few_rows); payloads.update(few_payloads)
    report = summary.assemble(rows, payloads, samples=100)
    descriptions = [item for item in report["comparisons"] if item["kind"] == "few_minus_zero_descriptive" and item["status"] == "complete"]
    assert len(descriptions) == 2
    assert all(item["equal_new_label_budget"] is False and item["train_per_class_a"] == 4 and item["train_per_class_b"] == 0 for item in descriptions)


@pytest.fixture
def artifact(tmp_path):
    dataset = PreparedDataset("breast_cancer", ["a", "b"],
        [Row(f"train-{i}", f"x={i}", i % 2) for i in range(8)], [],
        [Row("test-0", "x=20", 0), Row("test-1", "x=21", 1)], {"fixture": True})
    predictions = [Prediction("test-0", 0, [.8, .2]), Prediction("test-1", None, error="source_proposal_failed: Jev review not called")]
    path = tmp_path/"run/run.json"; path.parent.mkdir()
    record = {"status": "complete", "dataset": dataset.name, "labels": dataset.labels, "seed": 42,
        "manifest_sha256": digest(dataset.manifest), "dataset_manifest": dataset.manifest,
        "method": "cached_label_jev_review", "training_example_ids": [r.id for r in select_examples(dataset.train, dataset.labels, 4, 42)],
        "run_id": "fixture", "config": {}, "metrics": evaluate(dataset.test, predictions, 2)}
    path.write_text(json.dumps(record))
    path.with_name("predictions.jsonl").write_text("".join(json.dumps(asdict(p))+"\n" for p in predictions))
    path.with_name("test_manifest.json").write_text(json.dumps({"dataset": dataset.name, "labels": dataset.labels,
        "manifest_sha256": digest(dataset.manifest), "rows": [{"id": row.id, "label": row.label,
        "text_sha256": hashlib.sha256(row.text.encode()).hexdigest()} for row in dataset.test]}))
    return path, dataset, record, tmp_path


def audit(artifact):
    path, dataset, _, root = artifact
    return summary.audit_predictions(path, dataset, model="gpt-6-astra+jev_review", budget=4, display="Astra → Jev", root=root, samples=100)


def test_raw_metric_audit_counts_failure_and_probability_coverage(artifact):
    path, _, _, _ = artifact
    before = {file.name: file.read_bytes() for file in path.parent.iterdir()}
    row, payload = audit(artifact)
    assert row["accuracy"] == .5 and row["n_failures"] == 1
    assert row["probability_coverage"] == .5 and payload["predictions"] == [0, -1]
    assert before == {file.name: file.read_bytes() for file in path.parent.iterdir()}


@pytest.mark.parametrize("mutation,match", [("metric", "Saved metric"), ("training", "Training/example"), ("test", "held-out"), ("incomplete", "Incomplete")])
def test_stale_or_misaligned_artifacts_fail_closed(artifact, mutation, match):
    path, _, record, _ = artifact
    if mutation == "metric":
        record["metrics"]["accuracy"] = 1.
    elif mutation == "training":
        record["training_example_ids"].reverse()
    elif mutation == "incomplete":
        record["status"] = "running"
    else:
        test = summary.read(path.with_name("test_manifest.json")); test["rows"].reverse()
        path.with_name("test_manifest.json").write_text(json.dumps(test))
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=match):
        audit(artifact)


def test_historical_artifact_hashes_must_match_pinned_report(artifact):
    path, _, _, root = artifact
    row = {"source_path": path.relative_to(root).as_posix(), "artifact_sha256": {
        name: summary.file_sha(path.with_name(name)) for name in ("run.json", "predictions.jsonl", "test_manifest.json")}}
    assert summary.verify_historical_artifacts(row, root) == path
    path.with_name("predictions.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Historical artifact changed"):
        summary.verify_historical_artifacts(row, root)


def test_source_report_pin_is_checked_before_collecting_results(tmp_path):
    (tmp_path/"results").mkdir()
    (tmp_path/"results/TABULAR_COMPARISON.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned source"):
        summary.collect(tmp_path, samples=100)
