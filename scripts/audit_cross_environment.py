#!/usr/bin/env python3
"""Audit two immutable run directories without changing either manifest.

Different hashes are never treated as equal. This tool distinguishes verified
content/training compatibility from supplemental manifest-metadata differences.
It does not modify or bypass jevbench.compare_runs and does not compute scores.

Example:
    python scripts/audit_cross_environment.py RUN_A RUN_B \
        --prepared-data data/pilot/trec --output results/comparisons/content-audit.json

Without --prepared-data the audit checks internally consistent saved hashes and
row evidence. With it, content/file hashes and selected training labels are also
recomputed from the supplied frozen corpus files. Neither mode authenticates
artifacts against deliberate forgery by their producer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ALLOWED_METADATA_KEYS = {"prepared_path", "splits_before_text_transform", "post_transform_overlap_removed_ids"}
SPLITS = ("train", "validation", "test")
HEX256 = re.compile(r"[0-9a-f]{64}")


class AuditError(ValueError):
    pass


def digest(value) -> str:
    # Exact serialization used by runner.digest; do not substitute compact JSON.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"Cannot read valid JSON at {path}: {type(exc).__name__}") from None


def require(condition: bool, message: str):
    if not condition:
        raise AuditError(message)


def valid_hash(value) -> bool:
    return isinstance(value, str) and HEX256.fullmatch(value) is not None


def training_metadata(record: dict) -> dict:
    return record.get("training", record.get("adapter_training", {}))


def validate_supplemental_metadata(manifest: dict):
    if "prepared_path" in manifest:
        require(isinstance(manifest["prepared_path"], str), "prepared_path must be a string")
    if "post_transform_overlap_removed_ids" in manifest:
        removed = manifest["post_transform_overlap_removed_ids"]
        require(isinstance(removed, dict) and set(removed) == {"train", "validation"}, "Invalid post-transform removal audit")
        for split, ids in removed.items():
            require(isinstance(ids, list) and all(isinstance(i, str) for i in ids) and len(ids) == len(set(ids)), "Removal audit must contain unique IDs")
            require(len(ids) == manifest["post_transform_overlap_removed"][split], "Removal ID audit disagrees with recorded count")
    if "splits_before_text_transform" in manifest:
        before = manifest["splits_before_text_transform"]
        require(isinstance(before, dict) and set(before) == set(SPLITS), "Invalid pre-transform split audit")
        for split in SPLITS:
            previous, current = before[split], manifest["splits"][split]
            removed = manifest["post_transform_overlap_removed"].get(split, 0)
            require(previous["rows"] - removed == current["rows"], "Pre-transform row count disagrees with post-transform audit")
            require(sum(previous["class_counts"].values()) == previous["rows"], "Pre-transform class counts are inconsistent")
            require(all(previous["class_counts"].get(label, 0) >= count for label, count in current["class_counts"].items()), "Pre-transform class counts cannot increase after filtering")
            require(valid_hash(previous["row_ids_sha256"]), "Pre-transform row ID hash is missing")
            if removed == 0:
                require(previous["row_ids_sha256"] == current["row_ids_sha256"], "Row IDs changed despite zero post-transform removals")


def read_run(directory: Path) -> tuple[dict, dict]:
    record = load_json(directory / "run.json")
    test = load_json(directory / "test_manifest.json")
    require(record.get("status") == "complete", f"Run at {directory} is not complete")
    manifest = record.get("dataset_manifest")
    require(isinstance(manifest, dict), "Run lacks embedded dataset manifest")
    require(record.get("manifest_sha256") == digest(manifest), "Run manifest hash does not match its own embedded manifest")
    require(test.get("manifest_sha256") == record["manifest_sha256"], "Test manifest refers to a different dataset manifest")
    require(record.get("dataset") == test.get("dataset") == manifest.get("name"), "Dataset names disagree within one run")
    labels = record.get("labels")
    require(isinstance(labels, list) and len(labels) >= 2 and len(set(labels)) == len(labels), "Invalid label map")
    require(labels == test.get("labels") == manifest.get("labels"), "Label maps disagree within one run")
    rows = test.get("rows")
    require(isinstance(rows, list) and bool(rows), "Test row evidence is missing")
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"id", "label", "text_sha256"}, "Unexpected test row schema")
        require(isinstance(row["id"], str) and type(row["label"]) is int and row["label"] in range(len(labels)) and valid_hash(row["text_sha256"]), "Invalid test ID/label/text digest")
    ids = [row["id"] for row in rows]
    require(len(ids) == len(set(ids)), "Duplicate held-out row IDs")
    require(digest(ids) == record.get("test_ids_sha256"), "Ordered test ID digest disagrees with run")
    require(len(rows) == record.get("metrics", {}).get("n_test") == manifest["splits"]["test"]["rows"], "Held-out counts disagree")
    require(valid_hash(manifest.get("prepared_content_sha256")), "Full prepared-content SHA-256 is required")
    require(set(manifest.get("files_sha256", {})) == {f"{split}.jsonl" for split in SPLITS}, "All prepared split file hashes are required")
    require(all(valid_hash(value) for value in manifest["files_sha256"].values()), "Invalid prepared split file digest")
    selected = record.get("training_example_ids")
    require(isinstance(selected, list) and all(isinstance(i, str) for i in selected) and len(selected) == len(set(selected)), "Invalid selected training IDs")
    require(set(selected).isdisjoint(ids), "Training IDs overlap test IDs")
    require(type(record.get("seed")) is int, "Training/selection seed is required")
    metadata = training_metadata(record)
    if "training_row_ids" in metadata:
        require(metadata["training_row_ids"] == selected, "Training metadata disagrees with declared training IDs")
    require(type(metadata.get("validation_rows", 0)) is int and metadata.get("validation_rows", 0) >= 0, "Invalid validation-label use count")
    if record.get("method") == "zero_shot":
        require(not selected, "Zero-shot run includes training examples")
    if record.get("method") == "few_shot":
        k = record["config"].get("shots_per_class")
        require(type(k) is int and k > 0 and len(selected) == k * len(labels), "Few-shot label count disagrees with declared budget")
    for key in ("train_per_class",):
        k = metadata.get(key, record.get("config", {}).get(key))
        if k is not None:
            require(type(k) is int and k > 0 and len(selected) == k * len(labels), "Selected training count disagrees with class budget")
    validate_supplemental_metadata(manifest)
    return record, test


def leaf_differences(a, b, prefix="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            location = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                result.append(location)
            else:
                result.extend(leaf_differences(a[key], b[key], location))
        return result
    return [] if a == b else [prefix]


def difference_evidence(a: dict, b: dict, paths: list[str]) -> list[dict]:
    result = []
    for path in paths:
        item = {"path": path, "allowlisted_supplemental_metadata": path.split(".", 1)[0] in ALLOWED_METADATA_KEYS}
        for side, manifest in (("a", a), ("b", b)):
            value, present = manifest, True
            for key in path.split("."):
                if not isinstance(value, dict) or key not in value:
                    present = False
                    break
                value = value[key]
            item[f"present_{side}"] = present
            if present:
                item[f"value_{side}"] = value
        result.append(item)
    return result


def run_evidence(directory: Path, record: dict, test: dict) -> dict:
    manifest = record["dataset_manifest"]
    return {
        "path": str(directory), "id": record["run_id"],
        "manifest_sha256": record["manifest_sha256"],
        "python": record.get("environment", {}).get("python"),
        "implementation_sha256": record.get("implementation_sha256"),
        "prompt_template_sha256": record.get("prompt_template_sha256"),
        "prepared_content_sha256": manifest["prepared_content_sha256"],
        "prepared_files_sha256": manifest["files_sha256"],
        "labels": record["labels"], "test_rows": len(test["rows"]),
        "ordered_test_ids_labels_text_hashes_sha256": digest(test["rows"]),
        "training_rows": len(record["training_example_ids"]),
        "ordered_training_ids_sha256": digest(record["training_example_ids"]),
        "training_selection_seed": record["seed"],
        "validation_rows_used": training_metadata(record).get("validation_rows", 0),
    }


def recompute_prepared(path: Path, record: dict, test: dict) -> dict:
    content, file_hashes = {}, {}
    for split in SPLITS:
        filename = f"{split}.jsonl"
        raw = (path / filename).read_bytes()
        file_hashes[filename] = hashlib.sha256(raw).hexdigest()
        # bytes.splitlines avoids treating Unicode line separators inside JSON strings as records.
        content[split] = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require(file_hashes == record["dataset_manifest"]["files_sha256"], "Supplied prepared file hashes differ from run evidence")
    require(digest(content) == record["dataset_manifest"]["prepared_content_sha256"], "Recomputed full prepared-content hash differs")
    expected_test = [{"id": row["id"], "label": row["label"], "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()} for row in content["test"]]
    require(expected_test == test["rows"], "Recomputed held-out content differs from test manifest")
    training_by_id = {row["id"]: row for row in content["train"]}
    require(len(training_by_id) == len(content["train"]), "Duplicate prepared training IDs")
    require(set(record["training_example_ids"]) <= set(training_by_id), "Selected IDs do not belong to supplied training split")
    selected_labels = Counter(training_by_id[ident]["label"] for ident in record["training_example_ids"])
    k = record.get("config", {}).get("shots_per_class") if record["method"] == "few_shot" else training_metadata(record).get("train_per_class", record.get("config", {}).get("train_per_class"))
    if k is not None and k > 0:
        require(selected_labels == {label: k for label in range(len(record["labels"]))}, "Recomputed training labels violate the declared per-class budget")
    return {"verified": True, "prepared_directory": str(path), "full_content_sha256": digest(content), "file_hashes": file_hashes, "selected_training_class_counts": dict(sorted(selected_labels.items()))}


def audit(run_a: Path, run_b: Path, prepared_data: Path | None = None) -> dict:
    a, test_a = read_run(run_a)
    b, test_b = read_run(run_b)
    manifest_a, manifest_b = a["dataset_manifest"], b["dataset_manifest"]
    checks = {
        "same_dataset": a["dataset"] == b["dataset"],
        "same_label_map_and_order": a["labels"] == b["labels"],
        "same_ordered_test_ids_labels_text_hashes": test_a["rows"] == test_b["rows"],
        "same_full_prepared_content_hash": manifest_a["prepared_content_sha256"] == manifest_b["prepared_content_sha256"],
        "same_prepared_file_hashes": manifest_a["files_sha256"] == manifest_b["files_sha256"],
        "same_ordered_training_ids": a["training_example_ids"] == b["training_example_ids"],
        "same_training_selection_seed": a["seed"] == b["seed"],
        "same_validation_label_use_count": training_metadata(a).get("validation_rows", 0) == training_metadata(b).get("validation_rows", 0),
    }
    differences = leaf_differences(manifest_a, manifest_b)
    forbidden = [location for location in differences if location.split(".", 1)[0] not in ALLOWED_METADATA_KEYS]
    checks["manifest_differences_only_allowlisted_supplemental_metadata"] = not forbidden
    result = {
        "audit_type": "cross_environment_content_and_matched_training_audit", "schema_version": 1,
        "run_a": run_evidence(run_a, a, test_a),
        "run_b": run_evidence(run_b, b, test_b),
        "manifest_hashes_are_equal": a["manifest_sha256"] == b["manifest_sha256"],
        "checks": checks, "manifest_difference_paths": differences, "forbidden_manifest_difference_paths": forbidden,
        "manifest_differences": difference_evidence(manifest_a, manifest_b, differences),
        "allowed_supplemental_metadata_keys": sorted(ALLOWED_METADATA_KEYS),
        "raw_content_recomputed": False,
        "eligible_for_separately_labeled_matched_pairing": all(checks.values()),
        "interpretation": "Content/training compatibility is separate from manifest identity. Original hashes remain unchanged. This audit does not bypass compare_runs, establish equivalent models/output protocols, authenticate producer artifacts, or prove GPU numerical reproducibility.",
        "python_version_note": "Compatibility is based on realized row order/content and training IDs, never inferred from equal seed values across Python versions.",
    }
    if prepared_data is not None:
        result["recomputed_content_a"] = recompute_prepared(prepared_data, a, test_a)
        result["recomputed_content_b"] = recompute_prepared(prepared_data, b, test_b)
        result["raw_content_recomputed"] = True
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit(args.run_a, args.run_b, args.prepared_data)
    except (AuditError, KeyError, TypeError, OSError, ValueError) as exc:
        result = {"audit_type": "cross_environment_content_and_matched_training_audit", "eligible_for_separately_labeled_matched_pairing": False, "error": str(exc), "original_artifacts_modified": False}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        # Refuse to write inside either original run directory.
        output = args.output.resolve()
        protected = [args.run_a, args.run_b] + ([args.prepared_data] if args.prepared_data is not None else [])
        require(not any(output.is_relative_to(directory.resolve()) for directory in protected), "Audit output must not overwrite anything in an original run or prepared-data directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["eligible_for_separately_labeled_matched_pairing"] else 1


if __name__ == "__main__":
    sys.exit(main())
