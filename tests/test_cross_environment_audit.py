import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("cross_environment_audit", Path(__file__).parents[1] / "scripts/audit_cross_environment.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write_run(path, record, test):
    path.mkdir(exist_ok=True)
    record["manifest_sha256"] = audit.digest(record["dataset_manifest"])
    record["test_ids_sha256"] = audit.digest([r["id"] for r in test["rows"]])
    test["manifest_sha256"] = record["manifest_sha256"]
    (path / "run.json").write_text(json.dumps(record))
    (path / "test_manifest.json").write_text(json.dumps(test))


@pytest.fixture
def fixture(tmp_path):
    content = {
        "train": [{"id": f"train{i}", "text": f"training {i}", "label": i % 2} for i in range(4)],
        "validation": [{"id": f"val{i}", "text": f"development {i}", "label": i} for i in range(2)],
        "test": [{"id": f"test{i}", "text": f"heldout {i}", "label": i} for i in range(2)],
    }
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    hashes, splits = {}, {}
    for split, rows in content.items():
        raw = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
        (prepared / f"{split}.jsonl").write_bytes(raw)
        hashes[f"{split}.jsonl"] = hashlib.sha256(raw).hexdigest()
        splits[split] = {"rows": len(rows), "class_counts": {str(i): sum(r["label"] == i for r in rows) for i in range(2)},
                         "row_ids_sha256": audit.digest([r["id"] for r in rows]), "duplicate_text_rows": 0,
                         "conflicting_label_text_groups": 0}
    manifest = {"name": "fixture", "labels": ["zero", "one"], "schema_version": 1, "seed": 42,
                "source": {"revision": "a" * 40}, "files_sha256": hashes, "splits": splits,
                "prepared_content_sha256": audit.digest(content),
                "post_transform_overlap_removed": {"train": 0, "validation": 0},
                "text_policy": {"kind": "character_prefix", "max_chars": 2000}}
    record = {"dataset": "fixture", "labels": ["zero", "one"], "status": "complete",
              "method": "few_shot", "config": {"shots_per_class": 1}, "seed": 42,
              "run_id": "run-a", "training_example_ids": ["train0", "train1"],
              "dataset_manifest": manifest, "metrics": {"n_test": 2}, "environment": {"python": "3.12.0"}}
    test = {"dataset": "fixture", "labels": ["zero", "one"], "rows": [
        {"id": r["id"], "label": r["label"], "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest()}
        for r in content["test"]]}
    record_b, test_b = copy.deepcopy(record), copy.deepcopy(test)
    record_b["run_id"] = "run-b"
    record_b["environment"]["python"] = "3.13.0"
    record_b["dataset_manifest"]["splits_before_text_transform"] = copy.deepcopy(splits)
    record_b["dataset_manifest"]["post_transform_overlap_removed_ids"] = {"train": [], "validation": []}
    a, b = tmp_path / "run-a", tmp_path / "run-b"
    _write_run(a, record, test)
    _write_run(b, record_b, test_b)
    return a, b, prepared, record, test, record_b, test_b


def test_metadata_only_compatibility_preserves_unequal_hashes_and_original_files(fixture):
    a, b, prepared, *_ = fixture
    original = {p: p.read_bytes() for directory in (a, b, prepared) for p in directory.iterdir()}
    report = audit.audit(a, b, prepared)
    assert report["eligible_for_separately_labeled_matched_pairing"]
    assert report["raw_content_recomputed"]
    assert not report["manifest_hashes_are_equal"]
    assert report["run_a"]["manifest_sha256"] != report["run_b"]["manifest_sha256"]
    assert report["run_a"]["prepared_content_sha256"] == report["run_b"]["prepared_content_sha256"]
    assert set(report["manifest_difference_paths"]) == {"splits_before_text_transform", "post_transform_overlap_removed_ids"}
    assert all(not item["present_a"] and item["present_b"] and "value_b" in item for item in report["manifest_differences"])
    assert all(path.read_bytes() == raw for path, raw in original.items())


@pytest.mark.parametrize("change", ["test_label", "test_order", "test_text_hash", "training_ids", "training_order", "seed", "source_revision", "unapproved_metadata"])
def test_real_semantic_or_training_change_never_passes(fixture, change):
    a, b, _, _, _, record_b, test_b = fixture
    if change == "test_label":
        test_b["rows"][0]["label"] = 1
    elif change == "test_order":
        test_b["rows"].reverse()
    elif change == "test_text_hash":
        test_b["rows"][0]["text_sha256"] = "c" * 64
    elif change == "training_ids":
        record_b["training_example_ids"] = ["train2", "train3"]
    elif change == "training_order":
        record_b["training_example_ids"].reverse()
    elif change == "seed":
        record_b["seed"] = 87
    elif change == "unapproved_metadata":
        record_b["dataset_manifest"]["unapproved"] = "claim only"
    else:
        record_b["dataset_manifest"]["source"]["revision"] = "b" * 40
    _write_run(b, record_b, test_b)
    assert not audit.audit(a, b)["eligible_for_separately_labeled_matched_pairing"]


def test_equal_bogus_content_claim_fails_when_raw_files_are_supplied(fixture):
    a, b, prepared, record_a, test_a, record_b, test_b = fixture
    for path, record, test in [(a, record_a, test_a), (b, record_b, test_b)]:
        record["dataset_manifest"]["prepared_content_sha256"] = "d" * 64
        _write_run(path, record, test)
    assert audit.audit(a, b)["eligible_for_separately_labeled_matched_pairing"]  # hash-claim equality alone
    with pytest.raises(audit.AuditError, match="Recomputed full prepared-content"):
        audit.audit(a, b, prepared)


def test_forged_split_file_hash_fails_when_raw_files_are_supplied(fixture):
    a, b, prepared, record_a, test_a, record_b, test_b = fixture
    for path, record, test in [(a, record_a, test_a), (b, record_b, test_b)]:
        record["dataset_manifest"]["files_sha256"]["train.jsonl"] = "e" * 64
        _write_run(path, record, test)
    with pytest.raises(audit.AuditError, match="Supplied prepared file hashes"):
        audit.audit(a, b, prepared)


def test_manifest_self_inconsistency_fails_closed(fixture):
    a, b, _, _, _, record_b, _ = fixture
    record_b["manifest_sha256"] = "f" * 64
    (b / "run.json").write_text(json.dumps(record_b))
    with pytest.raises(audit.AuditError, match="own embedded manifest"):
        audit.audit(a, b)


def test_prepared_output_path_cannot_overwrite_original(fixture):
    a, b, prepared, *_ = fixture
    original = (prepared / "train.jsonl").read_bytes()
    with pytest.raises(audit.AuditError, match="must not overwrite"):
        audit.main([str(a), str(b), "--prepared-data", str(prepared), "--output", str(prepared / "train.jsonl")])
    assert (prepared / "train.jsonl").read_bytes() == original


@pytest.fixture
def prediction_fixture(fixture, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import compare_combined_pilot as combined
    a, b, prepared, record_a, test_a, record_b, test_b = fixture
    for path, record, test in [(a, record_a, test_a), (b, record_b, test_b)]:
        record["metrics"].update({"accuracy": 1.0, "macro_f1": 1.0, "n_failures": 0})
        _write_run(path, record, test)
        (path / "predictions.jsonl").write_text("".join(json.dumps({"row_id": row["id"], "label": row["label"], "error": None}) + "\n" for row in test["rows"]))
    return combined, a, b, prepared, record_b, test_b


def test_separate_comparison_keeps_unequal_original_manifests(prediction_fixture):
    combined, a, b, prepared, *_ = prediction_fixture
    result = combined.compare_pair(a, b, prepared, 100, 42)
    assert result["metrics"]["macro_f1"]["estimate"] == 0
    assert result["content_audit"]["raw_content_recomputed"]
    assert not result["strict_compare_runs_manifest_requirement_satisfied"]
    assert result["content_audit"]["run_a"]["manifest_sha256"] != result["content_audit"]["run_b"]["manifest_sha256"]


def test_separate_comparison_rejects_duplicate_prediction_ids(prediction_fixture):
    combined, a, b, prepared, *_ = prediction_fixture
    path = b / "predictions.jsonl"
    raw = path.read_text()
    path.write_text(raw + raw.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="no duplicates"):
        combined.compare_pair(a, b, prepared, 100, 42)


def test_separate_comparison_rejects_inconsistent_saved_metric(prediction_fixture):
    combined, a, b, prepared, record_b, test_b = prediction_fixture
    record_b["metrics"]["accuracy"] = 0.5
    _write_run(b, record_b, test_b)
    with pytest.raises(ValueError, match="Recomputed accuracy"):
        combined.compare_pair(a, b, prepared, 100, 42)
