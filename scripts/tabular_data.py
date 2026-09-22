#!/usr/bin/env python3
"""Pinned tabular sources, target-free serialization, grouped splits and native rows.

This extension deliberately leaves the published jevbench core unchanged. Native
rows and serialized Row objects share IDs, labels and order, and are verified
against each other on load. Nothing is fitted while preparing the data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.resources
import io
import json
import math
import sys
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevbench.data import (SPLITS, _split_audit, load_prepared, normalized_text,
                          save_prepared, stratified_subset, validate_prepared)
from jevbench.runner import digest
from jevbench.types import PreparedDataset, Row

CONFIG_PATH = ROOT / "configs" / "tabular_datasets.json"
SERIALIZATION = {
    "version": "name-value-v1", "feature_order": "declared features list",
    "number_format": ".12g (12 significant decimal digits; no data-dependent rounding)",
    "missing_value": "NA", "categorical_format": "JSON-quoted UTF-8 strings",
    "separator": "; ", "prefix": "fixed task description followed by newline",
    "fitted_preprocessing": "none", "text_truncation": "none",
    "target_or_row_id_in_features": False,
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def specs() -> dict[str, dict]:
    return json.loads(CONFIG_PATH.read_text())


def feature_schema(spec: dict) -> list[dict]:
    if "features" in spec:
        features = spec["features"]
    else:
        features = [{"name": value.replace(" ", "_"), "source_name": value,
                     "kind": "numeric", "unit": "original source scale (units not supplied)",
                     "description": value} for value in spec["feature_names"]]
    names = [f["name"] for f in features]
    if len(set(names)) != len(names) or not features:
        raise ValueError("Feature names must be unique and nonempty")
    excluded = set(spec.get("excluded_columns", [])) | {spec["target_column"]}
    if any(f["source_name"] in excluded or f["name"] in excluded for f in features):
        raise ValueError("Target or excluded source column included in feature schema")
    return features


def _format_value(value: Any, feature: dict) -> str:
    if value is None:
        return SERIALIZATION["missing_value"]
    if feature["kind"] == "numeric":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Non-finite or non-numeric feature {feature['name']}")
        return format(value, ".12g")
    if feature["kind"] != "categorical" or not isinstance(value, str):
        raise ValueError(f"Invalid categorical feature {feature['name']}")
    if "values" in feature and value not in feature["values"]:
        raise ValueError(f"Unknown category for {feature['name']}")
    return json.dumps(value, ensure_ascii=False)


def serialize_features(values: dict, features: list[dict], task_description: str) -> str:
    """Render only declared features; reject extra fields instead of silently leaking them."""
    if set(values) != {f["name"] for f in features}:
        raise ValueError("Native feature names differ from declared feature schema")
    return task_description + "\n" + "; ".join(
        f"{f['name']}={_format_value(values[f['name']], f)}" for f in features)


def _verified_source(path: Path, spec: dict) -> bytes:
    content = path.read_bytes()
    if sha256(content) != spec["sha256"]:
        raise ValueError(f"Pinned source hash mismatch for {spec['filename']}")
    return content


def load_source(name: str, cache_dir: Path) -> tuple[dict, list[dict], dict]:
    """Return config, target-separated native records, and source provenance."""
    import sklearn
    from sklearn import datasets
    spec = specs()[name]
    features = feature_schema(spec)
    raw_records = []
    if spec["source"] == "vanderbilt_csv":
        path = Path(cache_dir) / spec["filename"]
        if not path.exists():
            with urllib.request.urlopen(spec["url"], timeout=60) as response:
                content = response.read()
            if sha256(content) != spec["sha256"]:
                raise ValueError("Downloaded Titanic source differs from pinned hash")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        content = _verified_source(path, spec)
        for row in csv.DictReader(io.StringIO(content.decode("utf-8"))):
            values = {}
            for feature in features:
                value = row[feature["source_name"]]
                if value in {"", "NA", "?"}:
                    value = None
                elif feature["kind"] == "numeric":
                    value = float(value)
                elif feature["name"] == "embarked":
                    value = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[value]
                values[feature["name"]] = value
            raw_records.append({"features": values, "label": int(row[spec["target_column"]])})
    elif spec["source"] == "sklearn_bundled":
        resource = importlib.resources.files("sklearn.datasets.data").joinpath(spec["filename"])
        content = resource.read_bytes()
        if sha256(content) != spec["sha256"]:
            raise ValueError("Installed scikit-learn data differs from pinned source hash")
        dataset = getattr(datasets, spec["loader"])()
        if list(dataset.feature_names) != spec["feature_names"]:
            raise ValueError("Installed sklearn feature order differs from pinned specification")
        expected_target_names = {"breast_cancer": ["malignant", "benign"],
                                 "wine": ["class_0", "class_1", "class_2"]}[name]
        if list(dataset.target_names) != expected_target_names:
            raise ValueError("Installed sklearn label order differs from pinned specification")
        raw_records = [{"features": {f["name"]: float(value) for f, value in zip(features, values)},
                        "label": int(label)} for values, label in zip(dataset.data, dataset.target)]
    else:
        raise ValueError("Unknown tabular source")
    if len(raw_records) != spec["rows"]:
        raise ValueError("Source row count differs from pinned specification")
    records = []
    for index, record in enumerate(raw_records):
        if record["label"] not in range(len(spec["labels"])):
            raise ValueError("Invalid target label")
        text = serialize_features(record["features"], features, spec["task_description"])
        # All current source numbers survive fixed serialization exactly; reject
        # a future changed source if this promise ceases to hold.
        for feature in features:
            value = record["features"][feature["name"]]
            if feature["kind"] == "numeric" and value is not None and float(_format_value(value, feature)) != value:
                raise ValueError("Numeric serialization would lose source precision")
        identity = sha256(json_bytes(record))[:16]
        records.append({"id": f"{name}:source:{index:05d}:{identity}", **record})
    source = {**spec, "observed_sha256": sha256(content), "sklearn_version": sklearn.__version__,
              "selected_records_sha256": sha256(json_bytes(records))}
    return spec, records, source


def grouped_splits(rows: list[Row], seed: int) -> dict[str, list[Row]]:
    """Assign complete feature-identical groups to one of five stratified folds.

    Grouping does not use labels; stratification does. It retains contradictory
    observations together rather than silently selecting a favorable target.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    ordered = sorted(rows, key=lambda r: r.id)
    groups = [sha256(normalized_text(row.text).encode()) for row in ordered]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = [indices for _, indices in splitter.split(np.zeros(len(rows)),
             [row.label for row in ordered], groups)]
    assignments = {"test": set(folds[0]), "validation": set(folds[1]),
                   "train": set().union(*[set(x) for x in folds[2:]])}
    return {split: [row for index, row in enumerate(ordered) if index in assignments[split]]
            for split in SPLITS}


def prepare_from_native(name: str, spec: dict, records: list[dict], source: dict, *, seed=42,
                        test_limit: int | None = None) -> tuple[PreparedDataset, dict[str, list[dict]]]:
    features = feature_schema(spec)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Source IDs are not unique")
    rows = [Row(r["id"], serialize_features(r["features"], features, spec["task_description"]), r["label"])
            for r in records]
    split_rows = grouped_splits(rows, seed)
    full_audit = {split: _split_audit(values) for split, values in split_rows.items()}
    all_test = split_rows["test"]
    split_rows["test"] = stratified_subset(all_test, test_limit, seed)
    selected_ids = {r.id for r in split_rows["test"]}
    by_id = {r["id"]: r for r in records}
    native = {split: [by_id[row.id] for row in values] for split, values in split_rows.items()}
    manifest = {"schema_version": 1, "name": name, "labels": spec["labels"], "seed": seed,
        "source": source, "splits": {split: _split_audit(values) for split, values in split_rows.items()},
        "limits": {"train": None, "validation": None, "test": test_limit},
        "text_policy": {"kind": "none", "max_chars": None, "applies_to": "all serialized model arms"},
        "prepared_content_sha256": digest({split: [asdict(r) for r in values] for split, values in split_rows.items()}),
        "tabular": {"schema_version": 1, "features": features, "serialization": SERIALIZATION,
            "task_description": spec["task_description"], "source_rows": len(records),
            "config_sha256": sha256(json_bytes(spec)),
            "preparation_implementation_sha256": sha256(Path(__file__).read_bytes()),
            "split_method": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed): fold0=test; fold1=validation; remaining folds=train",
            "nominal_split_fractions": {"train": .6, "validation": .2, "test": .2},
            "group_definition": "SHA256 of normalized serialized features, excluding target and row ID",
            "duplicate_policy": "Retain all natural duplicates and conflicting-label groups in one split; no exact-feature overlap across splits",
            "full_splits_before_test_subset": full_audit,
            "test_subset_policy": "Frozen core deterministic proportional stratified subset; excluded rows remain held out and are never reassigned",
            "excluded_test_row_ids": [r.id for r in all_test if r.id not in selected_ids],
            "native_records_sha256": {split: sha256(json_bytes(values)) for split, values in native.items()},
            "max_serialized_characters": max(len(row.text) for row in rows),
            "numeric_roundtrip_exact": True,
            "leakage_limitations": "Exact feature grouping does not prevent related-family similarity or pretraining memorization of these public datasets. Wine cultivar IDs are arbitrary; zero-shot interpretation is limited."}}
    dataset = PreparedDataset(name, spec["labels"], split_rows["train"], split_rows["validation"], split_rows["test"], manifest)
    validate_prepared(dataset)
    return dataset, native


def save_native_prepared(dataset: PreparedDataset, native: dict[str, list[dict]], path: Path) -> None:
    path = Path(path)
    (path / "native").mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in SPLITS:
        content = b"".join(json_bytes(record) + b"\n" for record in native[split])
        relative = f"native/{split}.jsonl"
        (path / relative).write_bytes(content)
        hashes[relative] = sha256(content)
    dataset.manifest["tabular"]["native_files_sha256"] = hashes
    save_prepared(dataset, path)
    load_native_prepared(path)


def load_native_prepared(path: Path) -> tuple[PreparedDataset, dict[str, list[dict]]]:
    """Check native file/content hashes, IDs/order, labels, schema and rendered text."""
    path = Path(path)
    dataset = load_prepared(path)
    metadata = dataset.manifest["tabular"]
    if metadata["schema_version"] != 1 or metadata["serialization"] != SERIALIZATION:
        raise ValueError("Unsupported tabular serialization protocol")
    native = {}
    for split in SPLITS:
        relative = f"native/{split}.jsonl"
        content = (path / relative).read_bytes()
        if sha256(content) != metadata["native_files_sha256"][relative]:
            raise ValueError(f"Native data hash mismatch: {relative}")
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
        if sha256(json_bytes(records)) != metadata["native_records_sha256"][split]:
            raise ValueError(f"Native record content hash mismatch: {split}")
        rows = getattr(dataset, split)
        if len(records) != len(rows):
            raise ValueError(f"Native/serialized row count differs: {split}")
        for record, row in zip(records, rows):
            if set(record) != {"id", "label", "features"} or record["id"] != row.id or record["label"] != row.label:
                raise ValueError(f"Native/serialized identity mismatch: {split}")
            if serialize_features(record["features"], metadata["features"], metadata["task_description"]) != row.text:
                raise ValueError(f"Native features differ from serialized input: {row.id}")
        native[split] = records
    expected = digest({split: [asdict(row) for row in getattr(dataset, split)] for split in SPLITS})
    if expected != dataset.manifest["prepared_content_sha256"]:
        raise ValueError("Prepared tabular content hash mismatch")
    return dataset, native


def prepare_tabular(name: str, destination: Path, cache_dir: Path, *, seed=42, test_limit=None):
    destination = Path(destination)
    if (destination / "manifest.json").exists():
        dataset, native = load_native_prepared(destination)
        if (dataset.name != name or dataset.manifest["seed"] != seed
                or dataset.manifest["limits"]["test"] != test_limit
                or dataset.manifest["tabular"]["config_sha256"] != sha256(json_bytes(specs()[name]))
                or dataset.manifest["tabular"]["preparation_implementation_sha256"] != sha256(Path(__file__).read_bytes())):
            raise ValueError("Existing prepared directory has a different protocol; use a new output directory")
        return dataset, native
    spec, records, source = load_source(name, cache_dir)
    dataset, native = prepare_from_native(name, spec, records, source, seed=seed, test_limit=test_limit)
    save_native_prepared(dataset, native, destination)
    return dataset, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(specs()), default=list(specs()))
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "tabular")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "downloads" / "tabular")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Optional shared pilot subset; remaining full-test rows stay held out")
    args = parser.parse_args()
    for name in args.datasets:
        dataset, _ = prepare_tabular(name, args.output / name, args.cache, seed=args.seed, test_limit=args.test_limit)
        print(json.dumps({"dataset": name, "path": str(args.output / name),
            "splits": {split: len(getattr(dataset, split)) for split in SPLITS},
            "max_serialized_characters": dataset.manifest["tabular"]["max_serialized_characters"],
            "prepared_content_sha256": dataset.manifest["prepared_content_sha256"]}))


if __name__ == "__main__":
    main()
