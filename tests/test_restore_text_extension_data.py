"""Exact byte/metadata checks for offline restoration; no network or models."""
import fcntl
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import restore_text_extension_data as helper


@pytest.fixture
def reproduction(tmp_path):
    root = tmp_path / "repository"
    candidate = tmp_path / "candidate"
    registry = {"schema_version": 1, "datasets": {}, "historical_sources": {}}
    originals = {}
    for name in helper.DATASETS:
        folder = candidate / name
        folder.mkdir(parents=True)
        manifest = {"name": name, "splits": {"train": {"count": 2}, "validation": {"count": 1}, "test": {"count": 1}},
                    "source": "pinned-public-synthetic", "labels": ["negative", "positive"]}
        originals[name] = helper.encode_manifest(manifest)
        hashes = {"manifest.json": helper.sha(originals[name])}
        for filename in helper.ROW_FILES:
            raw = (json.dumps({"id": name + filename, "text": "unchanged", "label": 0}) + "\n").encode()
            (folder / filename).write_bytes(raw)
            hashes[filename] = helper.sha(raw)
        generated = {**manifest, "post_transform_overlap_removed_ids": {"train": [], "validation": []},
                     "splits_before_text_transform": manifest["splits"]}
        (folder / "manifest.json").write_bytes(helper.encode_manifest(generated))
        run_path = root / "results/historical" / name / "run.json"
        run_path.parent.mkdir(parents=True)
        raw_run = helper.encode_manifest({"dataset_manifest": manifest})
        run_path.write_bytes(raw_run)
        registry["datasets"][name] = {"hashes": hashes}
        registry["historical_sources"][f"{name}__qwen_small__k0"] = {
            "path": run_path.parent.relative_to(root).as_posix(), "hashes": {"run.json": helper.sha(raw_run)}}
    (root / "configs").mkdir()
    (root / "configs/text_extension_sources.json").write_text(json.dumps(registry))
    return root, candidate, originals


def tree_bytes(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_restore_exact_manifest_bytes_preserves_rows_evidence_and_is_idempotent(reproduction):
    root, candidate, originals = reproduction
    evidence = tree_bytes(root)
    before = tree_bytes(candidate)
    result = helper.restore(candidate, repository_root=root)
    assert result["restored"] == ["sst2", "trec"] and result["verified_jsonl_files"] == 6
    for name, original in originals.items():
        assert (candidate / name / "manifest.json").read_bytes() == original
        for filename in helper.ROW_FILES:
            assert (candidate / name / filename).read_bytes() == before[f"{name}/{filename}"]
    assert tree_bytes(root) == evidence
    assert not list(candidate.rglob("*.tmp"))
    mtimes = {p: p.stat().st_mtime_ns for p in candidate.rglob("*")}
    again = helper.restore(candidate, repository_root=root)
    assert again["restored"] == [] and again["already_frozen"] == ["sst2", "trec"]
    assert {p: p.stat().st_mtime_ns for p in mtimes} == mtimes


@pytest.mark.parametrize("change", ["row", "overlap", "split", "extra", "missing", "source", "encoded_pin", "numeric_type"])
def test_both_datasets_validated_before_any_write(reproduction, change, monkeypatch):
    root, candidate, _ = reproduction
    manifest_path = candidate / "trec/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "row":
        with (candidate / "trec/test.jsonl").open("ab") as stream: stream.write(b" ")
    elif change == "source":
        with (root / "results/historical/trec/run.json").open("ab") as stream: stream.write(b" ")
    elif change == "encoded_pin":
        registry_path = root / "configs/text_extension_sources.json"
        registry = json.loads(registry_path.read_text())
        registry["datasets"]["trec"]["hashes"]["manifest.json"] = "0" * 64
        registry_path.write_text(json.dumps(registry))
    else:
        if change == "overlap": manifest["post_transform_overlap_removed_ids"]["train"] = ["removed-row"]
        if change == "split": manifest["splits_before_text_transform"]["test"]["count"] = 2
        if change == "extra": manifest["unapproved"] = True
        if change == "missing": del manifest["splits_before_text_transform"]
        if change == "numeric_type": manifest["splits"]["test"]["count"] = 1.0
        manifest_path.write_bytes(helper.encode_manifest(manifest))
    before = tree_bytes(candidate)
    monkeypatch.setattr(helper.tempfile, "mkstemp", lambda **_: pytest.fail("Wrote before validating both datasets"))
    with pytest.raises(ValueError): helper.restore(candidate, repository_root=root)
    assert tree_bytes(candidate) == before


def test_candidate_symlink_is_rejected_without_writes(reproduction):
    root, candidate, _ = reproduction
    path = candidate / "trec/train.jsonl"
    raw = path.read_bytes(); path.unlink()
    outside = candidate.parent / "outside.jsonl"; outside.write_bytes(raw); path.symlink_to(outside)
    before = tree_bytes(candidate)
    with pytest.raises(ValueError, match="row bytes"): helper.restore(candidate, repository_root=root)
    assert tree_bytes(candidate) == before and outside.read_bytes() == raw


def test_directory_lock_prevents_second_writer(reproduction):
    root, candidate, _ = reproduction
    before = tree_bytes(candidate)
    fd = os.open(candidate, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): helper.restore(candidate, repository_root=root)
    finally:
        os.close(fd)
    assert tree_bytes(candidate) == before


def test_staging_failure_keeps_both_original_candidates(reproduction, monkeypatch):
    root, candidate, _ = reproduction
    before = tree_bytes(candidate)
    actual = helper.tempfile.mkstemp
    calls = 0
    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2: raise OSError("simulated staging failure")
        return actual(**kwargs)
    monkeypatch.setattr(helper.tempfile, "mkstemp", fail_second)
    with pytest.raises(OSError, match="staging failure"): helper.restore(candidate, repository_root=root)
    assert tree_bytes(candidate) == before and not list(candidate.rglob("*.tmp"))
