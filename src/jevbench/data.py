"""Pinned, script-free data loading and auditable split preparation.

Duplicate policy: preserve every original test row, including natural duplicates.
Remove training rows matching any final-test text before drawing validation. Then
remove remaining training rows matching validation text. Duplicates within a split
are retained and counted; Unicode NFKC + whitespace collapse + casefold defines
exact normalized text equality. This does not detect paraphrases/contamination.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .types import PreparedDataset, Row

SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "datasets.json"


def dataset_specs() -> dict[str, dict[str, Any]]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _rank(row: Row, seed: int) -> str:
    return _sha256(f"{seed}:{row.id}".encode("utf-8"))


def stratified_subset(rows: list[Row], limit: int | None, seed: int = 42) -> list[Row]:
    """Deterministic, proportional stratification, retaining every present class.

    Selection is stable under input reordering. Limit must cover every class; a
    requested limit larger than available rows returns all rows.
    """
    if limit is None or limit >= len(rows):
        return list(rows)
    groups: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        groups[row.label].append(row)
    if limit < len(groups) or limit <= 0:
        raise ValueError(f"limit={limit} must be at least the number of classes ({len(groups)})")
    labels = sorted(groups)
    quotas = {label: 1 for label in labels}
    targets = {label: limit * len(groups[label]) / len(rows) for label in labels}
    for _ in range(limit - len(labels)):
        eligible = [label for label in labels if quotas[label] < len(groups[label])]
        chosen = max(eligible, key=lambda label: (targets[label] - quotas[label], -label))
        quotas[chosen] += 1
    chosen_ids = {
        row.id
        for label in labels
        for row in sorted(groups[label], key=lambda row: (_rank(row, seed), row.id))[:quotas[label]]
    }
    return sorted((row for row in rows if row.id in chosen_ids), key=lambda row: (_rank(row, seed), row.id))


def select_train_rows(rows: list[Row], train_per_class: int | None, seed: int = 42) -> list[Row]:
    """Shared exact per-class label budget for classical, prompting and LoRA arms."""
    if train_per_class is None:
        return list(rows)
    if train_per_class < 1:
        raise ValueError("train_per_class must be positive")
    from .prompts import select_examples
    present_labels = sorted({row.label for row in rows})
    if not present_labels or present_labels != list(range(len(present_labels))):
        raise ValueError("Training labels must be contiguous and nonempty")
    return select_examples(rows, [str(label) for label in present_labels], train_per_class, seed)


def _rows_from_records(name: str, source_split: str, records: Iterable[dict], spec: dict,
                       labels: list[str]) -> list[Row]:
    rows = []
    label_lookup = {label: i for i, label in enumerate(labels)}
    for index, record in enumerate(records):
        text = record[spec["text_column"]]
        if not isinstance(text, str) or not normalized_text(text):
            raise ValueError(f"Empty/non-text input at {source_split}:{index}")
        raw_label = record[spec["label_column"]]
        label = label_lookup[raw_label] if isinstance(raw_label, str) else int(raw_label)
        if label not in range(len(labels)):
            raise ValueError(f"Invalid or hidden label {label} at {source_split}:{index}")
        identity = _sha256(_json_bytes({"text": text, "label": label}))[:16]
        rows.append(Row(f"{name}:{source_split}:{index}:{identity}", text, label))
    return rows


def _source_files(spec: dict, cache_dir: Path) -> dict[str, Path]:
    revision = spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full immutable 40-character source revision is required")
    if spec["source"] == "huggingface_parquet":
        from huggingface_hub import hf_hub_download
        return {split: Path(hf_hub_download(
            repo_id=spec["repo_id"], repo_type="dataset", filename=filename,
            revision=revision, cache_dir=str(cache_dir / "huggingface"),
        )) for split, filename in spec["files"].items()}
    if spec["source"] == "github_csv":
        result = {}
        for split, filename in spec["files"].items():
            path = cache_dir / "github" / spec["repo_id"] / revision / filename
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://raw.githubusercontent.com/{spec['repo_id']}/{revision}/{filename}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read()
                temp = path.with_suffix(path.suffix + ".tmp")
                temp.write_bytes(content)
                temp.replace(path)
            result[split] = path
        return result
    raise ValueError(f"Unknown source {spec['source']!r}")


def _read_sources(spec: dict, paths: dict[str, Path], name: str) -> tuple[list[str], list[Row], list[Row]]:
    labels = spec.get("labels") or json.loads(paths["labels"].read_text(encoding="utf-8"))
    result = {}
    for split in ("train", "test"):
        if spec["source"] == "huggingface_parquet":
            import pyarrow.parquet as pq
            records = pq.read_table(paths[split], columns=[spec["text_column"], spec["label_column"]]).to_pylist()
        else:
            with paths[split].open(encoding="utf-8", newline="") as stream:
                records = list(csv.DictReader(stream))
        result[split] = _rows_from_records(name, spec["source_splits"][split], records, spec, labels)
    return labels, result["train"], result["test"]


def _overlap_filter(rows: list[Row], holdout: list[Row]) -> tuple[list[Row], list[str]]:
    held_texts = {normalized_text(row.text) for row in holdout}
    kept, removed = [], []
    for row in rows:
        if normalized_text(row.text) in held_texts:
            removed.append(row.id)
        else:
            kept.append(row)
    return kept, removed


def _split_audit(rows: list[Row]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[normalized_text(row.text)].append(row.label)
    return {
        "rows": len(rows), "class_counts": {str(label): count for label, count in sorted(Counter(row.label for row in rows).items())},
        "duplicate_text_rows": sum(len(values) - 1 for values in groups.values()),
        "conflicting_label_text_groups": sum(len(set(values)) > 1 for values in groups.values()),
        "row_ids_sha256": _sha256(_json_bytes([row.id for row in rows])),
    }


def prepare_from_rows(name: str, labels: list[str], source_train: list[Row], source_test: list[Row],
                      *, seed: int = 42, validation_fraction: float = 0.1,
                      train_limit: int | None = None, validation_limit: int | None = None,
                      test_limit: int | None = None, manifest: dict | None = None) -> PreparedDataset:
    """Pure preparation entry point, also useful for offline fixtures."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    train, removed_test = _overlap_filter(source_train, source_test)
    validation_size = max(len(labels), math.ceil(len(train) * validation_fraction))
    if validation_size >= len(train):
        raise ValueError("Not enough rows for disjoint train and validation splits")
    validation = stratified_subset(train, validation_size, seed)
    validation_ids = {row.id for row in validation}
    train = [row for row in train if row.id not in validation_ids]
    train, removed_validation = _overlap_filter(train, validation)
    prepared = PreparedDataset(
        name, labels, stratified_subset(train, train_limit, seed),
        stratified_subset(validation, validation_limit, seed),
        stratified_subset(source_test, test_limit, seed), dict(manifest or {}),
    )
    prepared.manifest.update({
        "schema_version": SCHEMA_VERSION, "name": name, "labels": labels, "seed": seed,
        "validation_fraction": validation_fraction,
        "limits": {"train": train_limit, "validation": validation_limit, "test": test_limit},
        "source_counts": {"train": len(source_train), "test": len(source_test)},
        "duplicate_policy": {
            "normalization": "Unicode NFKC, casefold, collapse whitespace; no punctuation or HTML changes",
            "heldout_priority": "test then validation then train",
            "within_split": "retain all natural duplicates, including final test; report counts",
            "before_subsampling": True,
            "train_rows_removed_for_test_overlap": removed_test,
            "train_rows_removed_for_validation_overlap": removed_validation,
            "limitation": "Exact normalized text filtering does not detect semantic/phrase overlap or pretraining contamination",
        },
        "splits": {split: _split_audit(getattr(prepared, split)) for split in SPLITS},
    })
    validate_prepared(prepared)
    return prepared


def validate_prepared(dataset: PreparedDataset) -> None:
    if len(dataset.labels) < 2 or len(set(dataset.labels)) != len(dataset.labels):
        raise ValueError("Need at least two unique label names")
    all_ids: set[str] = set()
    text_sets = {}
    for split in SPLITS:
        rows = getattr(dataset, split)
        if not rows:
            raise ValueError(f"{split} split is empty")
        if {row.label for row in rows} != set(range(len(dataset.labels))):
            raise ValueError(f"{split} split must contain every class; use a larger limit")
        for row in rows:
            if row.id in all_ids:
                raise ValueError(f"Duplicate row id: {row.id}")
            if not normalized_text(row.text):
                raise ValueError(f"Empty text: {row.id}")
            all_ids.add(row.id)
        text_sets[split] = {normalized_text(row.text) for row in rows}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if text_sets[left] & text_sets[right]:
            raise ValueError(f"Normalized text leakage between {left} and {right}")


def prepared_path(name: str, cache_dir: Path, seed: int = 42, train_limit: int | None = None,
                  test_limit: int | None = None, validation_limit: int | None = None) -> Path:
    spec = dataset_specs()[name]
    config = {"schema_version": SCHEMA_VERSION, "spec": spec, "seed": seed,
              "train_limit": train_limit, "test_limit": test_limit, "validation_limit": validation_limit}
    return Path(cache_dir) / "prepared" / f"{name}-{_sha256(_json_bytes(config))[:16]}"


def prepare_dataset(name: str, cache_dir: Path, seed: int = 42, train_limit: int | None = None,
                    test_limit: int | None = None, validation_limit: int | None = None) -> PreparedDataset:
    specs = dataset_specs()
    if name not in specs:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(specs)}")
    destination = prepared_path(name, cache_dir, seed, train_limit, test_limit, validation_limit)
    if (destination / "manifest.json").exists():
        return load_prepared(destination)
    spec = specs[name]
    paths = _source_files(spec, Path(cache_dir) / "downloads")
    labels, train, test = _read_sources(spec, paths, name)
    source = dict(spec)
    source["files_sha256"] = {split: _sha256(path.read_bytes()) for split, path in paths.items()}
    dataset = prepare_from_rows(name, labels, train, test, seed=seed,
                                validation_fraction=spec["validation_fraction"], train_limit=train_limit,
                                validation_limit=validation_limit, test_limit=test_limit,
                                manifest={"source": source, "prepared_path": str(destination.resolve())})
    save_prepared(dataset, destination)
    return dataset


def save_prepared(dataset: PreparedDataset, path: Path) -> None:
    """Persist JSONL data and content hashes. Dataset text remains in ignored cache."""
    validate_prepared(dataset)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in SPLITS:
        data = b"".join(_json_bytes(asdict(row)) + b"\n" for row in getattr(dataset, split))
        hashes[f"{split}.jsonl"] = _sha256(data)
        tmp = path / f".{split}.{os.getpid()}.tmp"
        tmp.write_bytes(data)
        tmp.replace(path / f"{split}.jsonl")
    dataset.manifest.update({"schema_version": SCHEMA_VERSION, "name": dataset.name,
                             "labels": dataset.labels, "files_sha256": hashes})
    tmp = path / f".manifest.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(dataset.manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path / "manifest.json")


def load_prepared(path: Path) -> PreparedDataset:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported prepared dataset schema")
    rows = {}
    for split in SPLITS:
        filename = f"{split}.jsonl"
        content = (path / filename).read_bytes()
        if _sha256(content) != manifest["files_sha256"][filename]:
            raise ValueError(f"Prepared dataset hash mismatch: {filename}")
        rows[split] = [Row(**json.loads(line)) for line in content.splitlines() if line.strip()]
    result = PreparedDataset(manifest["name"], manifest["labels"], rows["train"], rows["validation"], rows["test"], manifest)
    validate_prepared(result)
    return result
