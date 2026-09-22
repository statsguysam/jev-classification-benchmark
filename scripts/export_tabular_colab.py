"""Bundle frozen tabular splits and allowlisted local-model code for Colab.

Includes public, de-identified feature rows and their labels, with attribution.
Excludes raw Titanic source records, secrets, results, base weights and adapters.
"""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "jev-tabular-benchmark"


def export(root=ROOT):
    from tabular_data import load_native_prepared
    names = ["pyproject.toml", "configs/models.json", "configs/tabular_datasets.json",
             "scripts/tabular_data.py", "scripts/run_tabular_local.py", "scripts/export_tabular_colab.py"]
    names.extend(p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py"))
    names.extend(p.relative_to(root).as_posix() for p in (root / "tests").glob("test_tabular_data.py"))
    names.extend(p.relative_to(root).as_posix() for p in (root / "tests").glob("test_tabular_local.py"))
    for dataset in ("titanic", "breast_cancer", "wine"):
        folder = root / "data/tabular-full" / dataset
        load_native_prepared(folder)
        for path in folder.rglob("*"):
            if path.is_file():
                if path.suffix not in {".json", ".jsonl"} or path.is_symlink():
                    raise ValueError("Unexpected prepared data member")
                names.append(path.relative_to(root).as_posix())
    contents = {}
    for name in sorted(set(names)):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unexpected external source path")
        contents[name] = path.read_bytes()
    manifest = {"schema_version": 1, "files_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
                "data_notice": "Public benchmark feature projections and labels only. Titanic from Vanderbilt hbiostat with attribution; Breast Cancer/Wine from UCI under CC BY 4.0. Attribution and source hashes are in configs/tabular_datasets.json and each dataset manifest. No raw names, tickets, addresses, source identifiers or outcomes-as-features are included."}
    contents["bundle_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    target = root / "artifacts/jev-tabular-colab.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            info = zipfile.ZipInfo(f"{PREFIX}/{name}")
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    print(json.dumps({"path": str(target), "files": len(contents), "bytes": target.stat().st_size,
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
    return target


if __name__ == "__main__":
    export()
