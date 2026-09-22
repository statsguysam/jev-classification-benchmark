"""Audit an allowlisted Colab result ZIP before importing immutable evidence.

Create the archive in Colab with archive members rooted at
results/numeric_expansion/local. Use --model-keys smollm2 for an early complete
four-condition model batch, or omit it to require all eight conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from jevbench.runner import save_json
from run_expanded_numeric_local import MODEL_KEYS, DATASETS, audit_source_run, file_sha, require
from tabular_data import load_native_prepared

ARCHIVE_PREFIX = "results/numeric_expansion/local"
RUN_FILES = {"run.json", "predictions.jsonl", "test_manifest.json"}


def members(archive):
    """Validate every member before any bytes are extracted."""
    prefix = PurePosixPath(ARCHIVE_PREFIX)
    result = {}
    total = 0
    for item in archive.infolist():
        name = PurePosixPath(item.filename)
        require(not name.is_absolute() and ".." not in name.parts and "\\" not in item.filename,
                "Unsafe archive path")
        require(not stat.S_ISLNK(item.external_attr >> 16), "Archive symlinks are forbidden")
        if item.is_dir():
            continue
        require(name.is_relative_to(prefix), "Unexpected archive root")
        relative = name.relative_to(prefix)
        require(len(relative.parts) == 2 and
                ((relative.parts[0] == "plans" and relative.name.endswith("-preflight.json")) or
                 (relative.parts[0] != "plans" and relative.name in RUN_FILES)), "Unexpected evidence file")
        require(relative.as_posix() not in result, "Duplicate archive member")
        require(item.file_size <= 20_000_000, "Evidence member exceeds size limit")
        total += item.file_size
        require(total <= 100_000_000, "Evidence archive exceeds size limit")
        result[relative.as_posix()] = item
    require(result, "Empty evidence archive")
    return result


def import_archive(path, *, output=None, model_keys=MODEL_KEYS, data_root=None, execute=False):
    output = Path(output or ROOT / ARCHIVE_PREFIX)
    data_root = Path(data_root or ROOT / "data/tabular-full")
    require(set(model_keys) <= set(MODEL_KEYS) and len(model_keys) == len(set(model_keys)) and model_keys,
            "Unexpected model selection")
    path = Path(path)
    archive_sha = file_sha(path)
    expected = {(dataset, key, shots) for dataset in DATASETS for key in model_keys for shots in (0, 4)}
    report = {"schema_version": 1, "archive_sha256": archive_sha, "status": "verified",
              "conditions": [], "files_sha256": {}}
    with tempfile.TemporaryDirectory(prefix="jev-numeric-import-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            inventory = members(archive)
            for name, info in inventory.items():
                content = archive.read(info)
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                report["files_sha256"][name] = hashlib.sha256(content).hexdigest()
        seen = set()
        datasets = {name: load_native_prepared(data_root / name)[0] for name in DATASETS}
        for folder in sorted(stage.iterdir()):
            if folder.name == "plans":
                continue
            require(folder.is_dir() and {p.name for p in folder.iterdir()} == RUN_FILES,
                    "Incomplete evidence run")
            record = json.loads((folder / "run.json").read_text())
            key = record["config"]["renderer"]["model_key"]
            condition = (record["dataset"], key, record["config"]["shots_per_class"])
            require(condition in expected and condition not in seen, "Unexpected or repeated source condition")
            audited, predictions = audit_source_run(folder / "run.json", datasets[condition[0]], key, condition[2])
            require(audited == record, "Audited run differs from archive record")
            seen.add(condition)
            report["conditions"].append({"dataset": condition[0], "model_key": key,
                "shots_per_class": condition[2], "run_id": record["run_id"], "n_predictions": len(predictions)})
        require(seen == expected, "Archive lacks required complete conditions")
        # Check all conflicts before writing even one file.
        for name, expected_sha in report["files_sha256"].items():
            target = output / name
            require(not target.is_symlink() and all(not parent.is_symlink() for parent in target.parents),
                    "Destination symlinks are forbidden")
            require(not target.exists() or (target.is_file() and file_sha(target) == expected_sha),
                    "Existing evidence differs; refusing overwrite")
        if execute:
            for name in report["files_sha256"]:
                target = output / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(stage / name, target)
            report["status"] = "imported"
            save_json(output.parent / "imports" / f"{archive_sha}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--model-keys", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Import after successful verification; default verifies only")
    args = parser.parse_args()
    print(json.dumps(import_archive(args.archive, output=args.output, model_keys=args.model_keys, execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
