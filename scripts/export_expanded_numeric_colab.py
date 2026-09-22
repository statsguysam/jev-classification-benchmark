"""Allowlisted source and public numeric data bundle; no credentials or weights."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_expanded_numeric_local import load_presets, file_sha, FROZEN_CORE_SHA
from tabular_data import load_native_prepared

PREFIX = "jev-numeric-expansion"


def export(root=ROOT):
    load_presets()
    names = ["pyproject.toml", "configs/expanded_numeric_models.json", "configs/tabular_datasets.json",
        "scripts/run_expanded_numeric_local.py", "scripts/export_expanded_numeric_colab.py",
        "scripts/run_tabular_local.py", "scripts/run_tabular_classical.py", "scripts/tabular_data.py",
        "tests/test_expanded_numeric_local.py"]
    names.extend(p.relative_to(root).as_posix() for p in (root / "src/jevbench").glob("*.py"))
    manifests = {}
    for dataset in ("breast_cancer", "wine"):
        folder = root / "data/tabular-full" / dataset
        load_native_prepared(folder)
        manifests[dataset] = file_sha(folder / "manifest.json")
        for member in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl",
                       "native/train.jsonl", "native/validation.jsonl", "native/test.jsonl"):
            names.append((folder / member).relative_to(root).as_posix())
    contents = {}
    for name in sorted(set(names)):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("External or symlinked source is forbidden")
        contents[name] = path.read_bytes()
    manifest = {"schema_version": 1, "core_sha256": FROZEN_CORE_SHA,
        "files_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
        "dataset_manifest_sha256": manifests,
        "runtime": {"transformers": "4.57.6", "huggingface-hub": "0.36.2", "device": "cuda", "dtype": "float16"},
        "data_notice": "Public Breast Cancer Wisconsin Diagnostic and Wine numeric features and labels, from UCI under CC BY 4.0. Attribution/source hashes are preserved in the dataset manifests and config. No credentials, hosted outputs, weights, or private records are included."}
    contents["bundle_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    target = root / "artifacts/jev-numeric-expansion-colab.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            info = zipfile.ZipInfo(f"{PREFIX}/{name}")
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    print(json.dumps({"path": str(target), "files": len(contents), "bytes": target.stat().st_size,
                      "sha256": file_sha(target), "helper_sha256": file_sha(root / "scripts/run_expanded_numeric_local.py")}))
    return target


if __name__ == "__main__":
    export()
