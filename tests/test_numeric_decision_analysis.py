import sys
from dataclasses import asdict
from pathlib import Path
import json

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_numeric_decisions as analysis
from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import digest
from jevbench.types import Prediction, PreparedDataset, Row


def test_review_corrections_keep_failures_in_denominator():
    counts = analysis.transitions([0, 1, 0, 1, 1], [1, 1, 0, -1, 0], [0, 0, 0, -1, 0])
    assert counts == {"wrong_to_correct": 1, "correct_to_wrong": 1, "both_correct": 1,
                      "both_wrong": 2, "changed_predictions": 2, "net_correct_change": 0,
                      "accuracy_delta_pp": 0, "correct_to_failure": 0, "correct_to_wrong_label": 1,
                      "source_failure_rows": 1, "review_stage_failure_rows": 0}
    with pytest.raises(ValueError, match="Unaligned"):
        analysis.transitions([0, 1], [0], [0, 1])


def test_failures_are_separated_from_incorrect_bounded_choices():
    counts = analysis.transitions([0, 1, 1], [0, 1, -1], [-1, 0, -1])
    assert counts["correct_to_wrong"] == 2
    assert counts["correct_to_wrong_label"] == counts["correct_to_failure"] == 1
    assert counts["source_failure_rows"] == counts["review_stage_failure_rows"] == 1


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    dataset = PreparedDataset("numeric_test", ["a", "b"],
        [Row(f"train-{i}", f"x={i}", i % 2) for i in range(8)], [],
        [Row("test-0", "x=20", 0), Row("test-1", "x=21", 1)], {"fixture": True})
    predictions = [Prediction("test-0", 0, [.8, .2]), Prediction("test-1", 1, [.1, .9])]
    directory = tmp_path / "run"
    directory.mkdir()
    record = {"status": "complete", "dataset": dataset.name, "labels": dataset.labels,
        "seed": 42, "manifest_sha256": digest(dataset.manifest), "method": "classical_numeric_supplement",
        "training_example_ids": [r.id for r in select_examples(dataset.train, dataset.labels, 4, 42)],
        "run_id": "test", "metrics": evaluate(dataset.test, predictions, 2)}
    (directory / "run.json").write_text(json.dumps(record))
    (directory / "predictions.jsonl").write_text("".join(json.dumps(asdict(p)) + "\n" for p in predictions))
    (directory / "test_manifest.json").write_text(json.dumps({"dataset": dataset.name, "labels": dataset.labels,
        "manifest_sha256": digest(dataset.manifest), "rows": [{"id": r.id, "label": r.label,
        "text_sha256": analysis.hashlib.sha256(r.text.encode()).hexdigest()} for r in dataset.test]}))
    return directory / "run.json", dataset, record


def check(artifact):
    path, dataset, _ = artifact
    return analysis.audit_supplement(path, dataset, model="xgboost", train_per_class=4, display="XGBoost")


def test_audit_recomputes_saved_scores_and_checks_training_rows(artifact):
    row, payload = check(artifact)
    assert row["accuracy"] == 1 and payload["predictions"] == [0, 1]
    path, _, record = artifact
    record["training_example_ids"][0] = "test-0"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="training IDs"):
        check(artifact)


def test_audit_rejects_stale_scores(artifact):
    path, _, record = artifact
    record["metrics"]["accuracy"] = .5
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Saved metric disagrees: accuracy"):
        check(artifact)


def test_audit_rejects_reordered_or_changed_test_rows(artifact):
    path, _, _ = artifact
    manifest = analysis.read(path.with_name("test_manifest.json"))
    manifest["rows"].reverse()
    path.with_name("test_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="test manifest differs"):
        check(artifact)
