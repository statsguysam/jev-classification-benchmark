"""Meaningful checks for target separation, group isolation and native parity."""
import copy
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tabular_data", ROOT / "scripts" / "tabular_data.py")
tabular = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tabular)


def fixture_spec():
    return {"labels": ["negative", "positive"], "target_column": "outcome",
            "excluded_columns": ["outcome", "identity"], "task_description": "Fixture classification; Missing=NA.",
            "features": [{"name": "value", "source_name": "value", "kind": "numeric"},
                         {"name": "category", "source_name": "category", "kind": "categorical"}]}


def fixture_records():
    records = [{"id": f"fixture:{index:03d}", "label": index % 2,
                "features": {"value": float(index), "category": "ordinary"}} for index in range(60)]
    # Exact feature duplicates include both agreeing and conflicting targets.
    for index in (3, 7, 11):
        records.append({"id": f"fixture:dup:{index}", "label": (index + 1) % 2,
                        "features": copy.deepcopy(records[index]["features"])})
    return records


def test_serializer_preserves_order_missing_types_and_escaping():
    spec = fixture_spec()
    values = {"category": 'A; value=100\n"quoted"', "value": None}
    rendered = tabular.serialize_features(values, spec["features"], spec["task_description"])
    assert rendered.endswith('value=NA; category="A; value=100\\n\\"quoted\\""')
    assert tabular.serialize_features({"value": 1.25e-5, "category": "NA"}, spec["features"], "Task") == 'Task\nvalue=1.25e-05; category="NA"'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.25"])
def test_serializer_rejects_invalid_numeric_values(value):
    spec = fixture_spec()
    with pytest.raises(ValueError, match="Non-finite or non-numeric"):
        tabular.serialize_features({"value": value, "category": "x"}, spec["features"], "Task")


def test_feature_allowlist_rejects_targets_and_unexpected_fields():
    spec = fixture_spec()
    with pytest.raises(ValueError, match="differ"):
        tabular.serialize_features({"value": 2., "category": "x", "outcome": "positive"}, spec["features"], "Task")
    spec["features"][0]["source_name"] = "outcome"
    with pytest.raises(ValueError, match="Target or excluded"):
        tabular.feature_schema(spec)
    titanic = tabular.specs()["titanic"]
    assert {f["source_name"] for f in tabular.feature_schema(titanic)} == {"pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"}
    assert {"survived", "name", "ticket", "boat", "body", "home.dest"} <= set(titanic["excluded_columns"])


def test_grouped_split_retains_all_records_and_keeps_duplicates_together():
    spec, records = fixture_spec(), fixture_records()
    dataset, native = tabular.prepare_from_native("fixture", spec, records, {}, seed=42)
    inverse = {}
    for split in tabular.SPLITS:
        for row in getattr(dataset, split):
            key = tabular.normalized_text(row.text)
            assert inverse.setdefault(key, split) == split
        assert {row.label for row in getattr(dataset, split)} == {0, 1}
        assert [record["id"] for record in native[split]] == [row.id for row in getattr(dataset, split)]
    assert sum(len(getattr(dataset, split)) for split in tabular.SPLITS) == len(records)
    repeated, _ = tabular.prepare_from_native("fixture", spec, list(reversed(records)), {}, seed=42)
    assert dataset.manifest["prepared_content_sha256"] == repeated.manifest["prepared_content_sha256"]
    assert any(dataset.manifest["splits"][split]["conflicting_label_text_groups"] for split in tabular.SPLITS)


def test_test_subset_is_audited_and_never_reassigned():
    spec, records = fixture_spec(), fixture_records()
    full, _ = tabular.prepare_from_native("fixture", spec, records, {}, seed=42)
    subset, _ = tabular.prepare_from_native("fixture", spec, records, {}, seed=42, test_limit=4)
    assert len(subset.test) == 4
    assert subset.train == full.train and subset.validation == full.validation
    removed = set(subset.manifest["tabular"]["excluded_test_row_ids"])
    assert removed == {row.id for row in full.test} - {row.id for row in subset.test}
    assert not removed & {row.id for row in subset.train + subset.validation}


def test_frozen_load_prepared_and_native_roundtrip(tmp_path):
    dataset, native = tabular.prepare_from_native("fixture", fixture_spec(), fixture_records(), {}, seed=42)
    tabular.save_native_prepared(dataset, native, tmp_path)
    loaded, loaded_native = tabular.load_native_prepared(tmp_path)
    assert [asdict(row) for row in loaded.test] == [asdict(row) for row in dataset.test]
    assert loaded_native == native
    assert tabular.load_prepared(tmp_path).manifest == loaded.manifest


def test_native_file_tampering_is_rejected(tmp_path):
    dataset, native = tabular.prepare_from_native("fixture", fixture_spec(), fixture_records(), {}, seed=42)
    tabular.save_native_prepared(dataset, native, tmp_path)
    with (tmp_path / "native" / "test.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="Native data hash mismatch"):
        tabular.load_native_prepared(tmp_path)


def test_native_label_mismatch_rejected_even_when_native_hashes_updated(tmp_path):
    dataset, native = tabular.prepare_from_native("fixture", fixture_spec(), fixture_records(), {}, seed=42)
    tabular.save_native_prepared(dataset, native, tmp_path)
    native["test"][0]["label"] = 1 - native["test"][0]["label"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    payload = b"".join(tabular.json_bytes(r) + b"\n" for r in native["test"])
    (tmp_path / "native" / "test.jsonl").write_bytes(payload)
    manifest["tabular"]["native_files_sha256"]["native/test.jsonl"] = tabular.sha256(payload)
    manifest["tabular"]["native_records_sha256"]["test"] = tabular.sha256(tabular.json_bytes(native["test"]))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Native/serialized identity mismatch"):
        tabular.load_native_prepared(tmp_path)


@pytest.mark.parametrize("name,rows,features,labels", [("breast_cancer", 569, 30, ["malignant", "benign"]), ("wine", 178, 13, ["cultivar 1", "cultivar 2", "cultivar 3"])])
def test_real_bundled_sources_are_pinned_and_target_free(name, rows, features, labels, tmp_path):
    spec, records, source = tabular.load_source(name, tmp_path)
    assert len(records) == rows and len(records[0]["features"]) == features
    assert spec["labels"] == labels
    assert source["observed_sha256"] == spec["sha256"]
    assert spec["target_column"] not in records[0]["features"]
    prepared, native = tabular.prepare_from_native(name, spec, records, source)
    tabular.save_native_prepared(prepared, native, tmp_path / name)
    assert prepared.manifest["tabular"]["numeric_roundtrip_exact"]
    assert prepared.manifest["tabular"]["max_serialized_characters"] < 1000


def test_titanic_missing_values_and_excluded_outcome_columns(monkeypatch, tmp_path):
    spec = copy.deepcopy(tabular.specs()["titanic"])
    csv_content = b'pclass,survived,sex,age,sibsp,parch,fare,embarked,boat,body,name,ticket,home.dest,cabin\n1,1,female,29,0,0,211.3375,S,boat-leak,body-leak,name-leak,ticket-leak,home-leak,cabin-leak\n3,0,male,,0,0,,,boat-leak,body-leak,name-leak,ticket-leak,home-leak,cabin-leak\n'
    spec["rows"] = 2
    spec["sha256"] = hashlib.sha256(csv_content).hexdigest()
    (tmp_path / spec["filename"]).write_bytes(csv_content)
    monkeypatch.setattr(tabular, "specs", lambda: {"titanic": spec})
    _, records, _ = tabular.load_source("titanic", tmp_path)
    assert records[1]["features"]["age_years"] is None
    assert records[1]["features"]["fare_gbp"] is None
    assert records[1]["features"]["embarked"] is None
    text = tabular.serialize_features(records[0]["features"], spec["features"], spec["task_description"])
    assert "-leak" not in text and "survived=" not in text
    assert 'embarked="Southampton"' in text
    (tmp_path / spec["filename"]).write_bytes(csv_content + b"\n")
    with pytest.raises(ValueError, match="Pinned source hash mismatch"):
        tabular.load_source("titanic", tmp_path)
