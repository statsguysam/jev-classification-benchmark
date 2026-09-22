import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import import_expanded_numeric_colab as helper


def test_archive_rejects_unsafe_extra_and_duplicate_members():
    for names in (["../escape.json"], [helper.ARCHIVE_PREFIX + "/run/token.txt"],
                  [helper.ARCHIVE_PREFIX + "/run/run.json"] * 2):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            for name in names:
                archive.writestr(name, "{}")
        with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError):
            helper.members(archive)


def make_archive(path):
    with zipfile.ZipFile(path, "w") as archive:
        for dataset in helper.DATASETS:
            for shots in (0, 4):
                run = f"{dataset}-smollm2-k{shots}"
                prefix = helper.ARCHIVE_PREFIX + "/" + run + "/"
                record = {"run_id": run, "dataset": dataset,
                          "config": {"shots_per_class": shots, "renderer": {"model_key": "smollm2"}}}
                archive.writestr(prefix + "run.json", json.dumps(record))
                archive.writestr(prefix + "test_manifest.json", "{}")
                archive.writestr(prefix + "predictions.jsonl", "{}\n")


def test_import_requires_complete_audits_then_refuses_overwrite(tmp_path, monkeypatch):
    archive = tmp_path / "results.zip"
    make_archive(archive)
    calls = []

    def audit(path, dataset, key, shots):
        calls.append((dataset.name, key, shots))
        return json.loads(path.read_text()), [None]

    monkeypatch.setattr(helper, "audit_source_run", audit)
    output = tmp_path / "local"
    result = helper.import_archive(archive, output=output, model_keys=["smollm2"])
    assert result["status"] == "verified" and len(calls) == 4 and not output.exists()
    result = helper.import_archive(archive, output=output, model_keys=["smollm2"], execute=True)
    assert result["status"] == "imported"
    member = output / "breast_cancer-smollm2-k0/predictions.jsonl"
    member.write_text("changed")
    with pytest.raises(ValueError, match="refusing overwrite"):
        helper.import_archive(archive, output=output, model_keys=["smollm2"], execute=True)
    assert member.read_text() == "changed"
    with pytest.raises(ValueError, match="lacks required"):
        helper.import_archive(archive, output=tmp_path / "second")
