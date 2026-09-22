import hashlib
import importlib.util
import json
import struct
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("package_adapters", Path(__file__).parents[1]/"scripts/package_adapters.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def adapter(tmp_path, monkeypatch, model="Qwen/Qwen3-4B-Instruct-2507", name="adapter"):
    folder = tmp_path/name
    folder.mkdir()
    metadata = {"model_id": model, "resolved_revision": packager.SUPPORTED[model], "dataset": "trec",
                "test_accessed": False, "r": 2, "train_per_class": 4, "seed": 42,
                "training_rows": 24, "prompt_sha256": "a"*64}
    config = {"base_model_name_or_path": model, "r": 2, "peft_type": "LORA", "task_type": "CAUSAL_LM"}
    (folder/"jevbench_training.json").write_text(json.dumps(metadata))
    (folder/"adapter_config.json").write_text(json.dumps(config))
    header = {"__metadata__": {"format": "pt"}}
    for kind, shape, offsets in [("A", [2, 4], [0, 32]), ("B", [4, 2], [32, 64])]:
        header[f"base_model.model.model.layers.0.self_attn.q_proj.lora_{kind}.weight"] = {
            "dtype": "F32", "shape": shape, "data_offsets": offsets}
    raw = json.dumps(header).encode()
    (folder/"adapter_model.safetensors").write_bytes(struct.pack("<Q", len(raw))+raw+bytes(64))
    upstream = tmp_path/"upstream"/model.split("/")[-1]
    upstream.mkdir(parents=True, exist_ok=True)
    fixture_license = b"Unit test fixture license, not a distribution artifact."
    (upstream/"LICENSE").write_bytes(fixture_license)
    monkeypatch.setattr(packager, "LICENSE_SHA256", hashlib.sha256(fixture_license).hexdigest())
    return folder


def test_both_model_families_have_complete_hashed_provenance(tmp_path, monkeypatch):
    small = adapter(tmp_path, monkeypatch, "Qwen/Qwen2.5-0.5B-Instruct", "small")
    main = adapter(tmp_path, monkeypatch, name="main")
    output = packager.package([small, main], tmp_path/"release.zip", upstream_root=tmp_path/"upstream")
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert len(manifest["adapters"]) == 2
        assert set(archive.namelist()) == set(manifest["files_sha256"]) | {"MANIFEST.json"}
        for name, expected in manifest["files_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        assert "upstream/Qwen3-4B-Instruct-2507/LICENSE" in archive.namelist()
        assert "main/DOWNSTREAM_NOTICE.txt" in archive.namelist()
        assert not any(name.endswith("model.safetensors") and not name.endswith("adapter_model.safetensors") for name in archive.namelist())


@pytest.mark.parametrize("unexpected", [".env", "model-00001-of-00003.safetensors", "train.jsonl"])
def test_unexpected_files_refused_before_output(tmp_path, monkeypatch, unexpected):
    folder = adapter(tmp_path, monkeypatch)
    (folder/unexpected).write_text("excluded")
    output = tmp_path/"release.zip"
    with pytest.raises(ValueError, match="non-allowlisted"):
        packager.package([folder], output, upstream_root=tmp_path/"upstream")
    assert not output.exists()


def test_renamed_base_weights_are_rejected(tmp_path, monkeypatch):
    folder = adapter(tmp_path, monkeypatch)
    header = {"model.embed_tokens.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
    raw = json.dumps(header).encode()
    (folder/"adapter_model.safetensors").write_bytes(struct.pack("<Q", len(raw))+raw+bytes(16))
    with pytest.raises(ValueError, match="Only Qwen LoRA"):
        packager.package([folder], tmp_path/"release.zip", upstream_root=tmp_path/"upstream")


def test_wrong_revision_or_license_cannot_be_packaged(tmp_path, monkeypatch):
    folder = adapter(tmp_path, monkeypatch)
    metadata = json.loads((folder/"jevbench_training.json").read_text())
    metadata["resolved_revision"] = "b"*40
    (folder/"jevbench_training.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not allowlisted"):
        packager.package([folder], tmp_path/"release.zip", upstream_root=tmp_path/"upstream")


def test_symlink_and_credential_fields_fail_closed(tmp_path, monkeypatch):
    folder = adapter(tmp_path, monkeypatch)
    (folder/"README.md").symlink_to(folder/"jevbench_training.json")
    with pytest.raises(ValueError, match="Symlinks"):
        packager.package([folder], tmp_path/"release.zip", upstream_root=tmp_path/"upstream")
    (folder/"README.md").unlink()
    metadata = json.loads((folder/"jevbench_training.json").read_text())
    metadata["api_key"] = "dummy-key"
    (folder/"jevbench_training.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Sensitive field"):
        packager.package([folder], tmp_path/"release.zip", upstream_root=tmp_path/"upstream")


@pytest.mark.parametrize("dataset,training_rows", [("titanic", 8), ("breast_cancer", 8), ("wine", 12)])
def test_training_only_tabular_adapters_keep_licenses_and_dataset_provenance(tmp_path, monkeypatch, dataset, training_rows):
    folder = adapter(tmp_path, monkeypatch, name=dataset)
    metadata = json.loads((folder/"jevbench_training.json").read_text())
    metadata.update(dataset=dataset, training_rows=training_rows)
    (folder/"jevbench_training.json").write_text(json.dumps(metadata))
    output = packager.package([folder], tmp_path/"release.zip", upstream_root=tmp_path/"upstream")
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["adapters"][dataset]["dataset"] == dataset
        assert manifest["adapters"][dataset]["training_rows"] == training_rows
        assert "upstream/Qwen3-4B-Instruct-2507/LICENSE" in archive.namelist()
        assert "serialized tabular" in archive.read("NOTICE.txt").decode()
        for name, expected in manifest["files_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected


@pytest.mark.parametrize("dataset,test_accessed", [("unknown", False), ("titanic", True),
    ("breast_cancer", None), ("wine", "false"), ("wine", 0)])
def test_unknown_or_not_explicitly_training_only_adapters_refused(tmp_path, monkeypatch, dataset, test_accessed):
    folder = adapter(tmp_path, monkeypatch)
    metadata = json.loads((folder/"jevbench_training.json").read_text())
    metadata.update(dataset=dataset, test_accessed=test_accessed)
    (folder/"jevbench_training.json").write_text(json.dumps(metadata))
    output = tmp_path/"release.zip"
    with pytest.raises(ValueError, match="allowlisted text/tabular training-only"):
        packager.package([folder], output, upstream_root=tmp_path/"upstream")
    assert not output.exists()
