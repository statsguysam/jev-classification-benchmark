#!/usr/bin/env python3
"""Restore historical manifest bytes after exact, metadata-only text reproduction.

No downloads, source/result edits, or row transformations occur. Both datasets
must pass before any file is written. A directory lock serializes this helper;
each manifest replacement uses an exclusive temporary file and atomic rename.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("sst2", "trec")
ROW_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")
ADDED_FIELDS = {"post_transform_overlap_removed_ids", "splits_before_text_transform"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode_manifest(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def validate_all(candidate_root, repository_root):
    """Return exact replacement bytes only for the single permitted difference."""
    registry = json.loads((repository_root / "configs/text_extension_sources.json").read_text())
    require(registry.get("schema_version") == 1 and set(registry["datasets"]) == set(DATASETS),
            "Unexpected text registry schema/datasets")
    plans = []
    for name in DATASETS:
        folder = candidate_root / name
        require(folder.is_dir() and not folder.is_symlink(), f"Unsafe or missing candidate folder: {name}")
        pinned = registry["datasets"][name]["hashes"]
        require(set(pinned) == {"manifest.json", *ROW_FILES}, "Incomplete prepared artifact pins")
        for filename in ROW_FILES:
            path = folder / filename
            require(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == pinned[filename],
                    f"Frozen row bytes differ: {name}/{filename}")
        source = registry["historical_sources"][f"{name}__qwen_small__k0"]
        relative = Path(source["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe historical source path")
        run_path = repository_root / relative / "run.json"
        require(run_path.resolve().is_relative_to(repository_root.resolve()) and not run_path.is_symlink(),
                "Unsafe historical source artifact")
        raw_run = run_path.read_bytes()
        require(sha(raw_run) == source["hashes"]["run.json"], "Historical source run hash differs")
        original = json.loads(raw_run)["dataset_manifest"]
        require(isinstance(original, dict) and not ADDED_FIELDS.intersection(original),
                "Historical manifest has unexpected transformation metadata")
        restored = encode_manifest(original)
        require(sha(restored) == pinned["manifest.json"], "Encoded historical manifest does not match its exact pin")
        path = folder / "manifest.json"
        require(path.is_file() and not path.is_symlink(), f"Unsafe or missing candidate manifest: {name}")
        current = path.read_bytes()
        if sha(current) != pinned["manifest.json"]:
            generated = json.loads(current)
            expected = {**original, "post_transform_overlap_removed_ids": {"train": [], "validation": []},
                        "splits_before_text_transform": original["splits"]}
            require(encode_manifest(generated) == encode_manifest(expected),
                    f"Candidate manifest is not the permitted metadata-only reproduction: {name}")
        plans.append({"dataset": name, "path": path, "before": current, "after": restored,
                      "changed": current != restored})
    return plans


def restore(candidate_root=None, *, repository_root=ROOT):
    repository_root = Path(repository_root)
    candidate_root = Path(candidate_root or repository_root / "data/pilot")
    require(candidate_root.is_dir() and not candidate_root.is_symlink(), "Unsafe or missing candidate root")
    directory_fd = os.open(candidate_root, os.O_RDONLY)
    temporary_paths = []
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plans = validate_all(candidate_root, repository_root)
        # Stage both only after all six row files, both manifests and both
        # historical source records have been independently verified.
        for plan in plans:
            if not plan["changed"]:
                continue
            fd, filename = tempfile.mkstemp(prefix=".manifest-restore-", suffix=".tmp", dir=plan["path"].parent)
            temporary = Path(filename)
            temporary_paths.append(temporary)
            with os.fdopen(fd, "wb") as stream:
                stream.write(plan["after"])
                stream.flush()
                os.fsync(stream.fileno())
            plan["temporary"] = temporary
        # Reject observed concurrent changes before replacing either manifest.
        require(validate_all(candidate_root, repository_root) == plans_without_temporary(plans),
                "Candidate evidence changed while staging restoration")
        for plan in plans:
            if plan["changed"]:
                os.replace(plan["temporary"], plan["path"])
                folder_fd = os.open(plan["path"].parent, os.O_RDONLY)
                try:
                    os.fsync(folder_fd)
                finally:
                    os.close(folder_fd)
        require(not any(plan["changed"] for plan in validate_all(candidate_root, repository_root)),
                "Restored manifest verification failed")
        return {"status": "verified", "restored": [plan["dataset"] for plan in plans if plan["changed"]],
                "already_frozen": [plan["dataset"] for plan in plans if not plan["changed"]],
                "verified_jsonl_files": 6,
                "manifest_sha256": {plan["dataset"]: sha(plan["after"]) for plan in plans}}
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        os.close(directory_fd)


def plans_without_temporary(plans):
    return [{key: value for key, value in plan.items() if key != "temporary"} for plan in plans]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=ROOT / "data/pilot")
    args = parser.parse_args(argv)
    print(json.dumps(restore(args.candidate_root), indent=2))


if __name__ == "__main__":
    main()
