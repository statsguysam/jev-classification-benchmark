"""Boundary checks for the historical audit launcher; no model or network calls."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("frozen_runtime", ROOT / "scripts/reproduce_frozen_study.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def test_archive_contains_exact_original_package_without_raw_results_or_bridge():
    manifest, payloads = runtime.verify_snapshot()
    assert manifest["commit"] == runtime.COMMIT
    assert len(payloads) == 216
    assert "tests/test_frozen_suite.py" not in payloads
    assert not any(name.startswith(("data/", "results/", "tests/current/")) for name in payloads)
    assert "src/jevbench/runner.py" in payloads and "tests/test_runner.py" in payloads


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a\\escape", "a//b", "C:/escape", "a/./b"])
def test_snapshot_path_rejects_traversal_and_alternate_separators(name):
    with pytest.raises(ValueError, match="Unsafe snapshot path"):
        runtime.safe_relative(name)


def test_tampered_archive_is_rejected_before_extraction(tmp_path):
    archive = tmp_path / "snapshot.zip"
    archive.write_bytes(runtime.SNAPSHOT.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive checksum"):
        runtime.verify_snapshot(archive=archive)


def test_tampered_manifest_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(runtime.MANIFEST.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest checksum"):
        runtime.verify_snapshot(manifest_path=manifest)


def test_evidence_copy_does_not_share_writable_inodes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "prediction.json").write_text("original")
    runtime.copy_regular_tree(source, target)
    (target / "prediction.json").write_text("changed")
    assert (source / "prediction.json").read_text() == "original"


def test_evidence_copy_rejects_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to(runtime.MANIFEST)
    with pytest.raises(ValueError, match="symlink"):
        runtime.copy_regular_tree(source, tmp_path / "target")


def test_child_environment_selects_archived_imports_and_omits_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/package")
    environment = runtime.child_environment(tmp_path)
    assert "OPENAI_API_KEY" not in environment and "HF_TOKEN" not in environment
    assert environment["PYTHONPATH"].split(runtime.os.pathsep) == [str(tmp_path / "src"), str(tmp_path / "scripts")]
    assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"


def test_prepare_refuses_existing_dataset_before_running_a_child(monkeypatch, tmp_path):
    destination = tmp_path / "data"
    (destination / "pilot/sst2").mkdir(parents=True)
    monkeypatch.setattr(runtime, "run_python", lambda *a, **k: pytest.fail("must not run a child"))
    with pytest.raises(ValueError, match="replace an existing dataset"):
        runtime.prepare_data(tmp_path / "checkout", destination)


def test_prepared_native_bytes_must_match_original_record(monkeypatch, tmp_path):
    runs = []
    for name in ("titanic", "breast_cancer", "wine"):
        folder = tmp_path / "data/tabular-full" / name
        folder.mkdir(parents=True)
        (folder / "test.jsonl").write_bytes(b"serialized")
        (folder / "native.jsonl").write_bytes(b"native")
        manifest = {"files_sha256": {"test.jsonl": runtime.sha(b"serialized")},
                    "tabular": {"native_files_sha256": {"native.jsonl": runtime.sha(b"native")}}}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        record = json.dumps({"dataset_manifest": manifest}).encode()
        relative = f"results/{name}/run.json"
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(record)
        runs.append({"dataset": name, "source_path": relative,
                     "artifact_sha256": {"run.json": runtime.sha(record)}})
    report = json.dumps({"runs": runs}).encode()
    (tmp_path / "results/TABULAR_COMPARISON.json").write_bytes(report)
    monkeypatch.setattr(runtime, "TABULAR_REPORT_SHA256", runtime.sha(report))
    runtime.verify_tabular_preparation(tmp_path)
    (tmp_path / "data/tabular-full/wine/native.jsonl").write_bytes(b"changed native features")
    with pytest.raises(ValueError, match="Prepared tabular file differs: wine/native.jsonl"):
        runtime.verify_tabular_preparation(tmp_path)
