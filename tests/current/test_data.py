import json
from collections import Counter

import pytest

from jevbench.data import (dataset_specs, load_prepared, normalized_text, prepare_from_rows,
                           save_prepared, select_train_rows, stratified_subset)
from jevbench.prompts import select_examples
from jevbench.types import Row


def source_rows():
    train = [Row(f"train-{label}-{i}", f"training class {label} sample {i}", label)
             for label in range(2) for i in range(40)]
    test = [Row(f"test-{label}-{i}", f"testing class {label} sample {i}", label)
            for label in range(2) for i in range(10)]
    return train, test


def test_normalized_leakage_removes_training_and_preserves_duplicate_test():
    train, test = source_rows()
    test.extend([Row("dup-test-1", " ＨＥＬＬＯ   World ", 0), Row("dup-test-2", "hello world", 0)])
    train.append(Row("leak", "Hello\nWorld", 0))
    result = prepare_from_rows("fixture", ["negative", "positive"], train, test, seed=17)
    assert len(result.test) == len(test)
    assert "leak" not in {row.id for row in result.train + result.validation}
    assert result.manifest["duplicate_policy"]["train_rows_removed_for_test_overlap"] == ["leak"]
    assert result.manifest["splits"]["test"]["duplicate_text_rows"] == 1
    text_sets = [{normalized_text(row.text) for row in split}
                 for split in [result.train, result.validation, result.test]]
    assert not text_sets[0] & text_sets[1]
    assert not text_sets[0] & text_sets[2]
    assert not text_sets[1] & text_sets[2]


def test_validation_duplicate_cannot_leak_back_to_train():
    train, test = source_rows()
    train.extend(Row(f"repeated-{i}", "same training duplicate", i % 2) for i in range(20))
    result = prepare_from_rows("fixture", ["zero", "one"], train, test, seed=42)
    assert "same training duplicate" in {row.text for row in result.validation}
    assert "same training duplicate" not in {row.text for row in result.train}
    assert result.manifest["duplicate_policy"]["train_rows_removed_for_validation_overlap"]


def test_seeded_subsets_are_proportional_and_reproducible():
    rows = [Row(str(i), f"text {i}", int(i >= 80)) for i in range(100)]
    a = stratified_subset(rows, 20, seed=42)
    assert Counter(row.label for row in a) == {0: 16, 1: 4}
    assert a == stratified_subset(list(reversed(rows)), 20, seed=42)
    assert a != stratified_subset(rows, 20, seed=43)
    with pytest.raises(ValueError, match="number of classes"):
        stratified_subset(rows, 1)


def test_label_budget_matches_prompt_demonstrations():
    train, _ = source_rows()
    assert select_train_rows(train, 4, 91) == select_examples(train, ["zero", "one"], 4, 91)
    assert Counter(row.label for row in select_train_rows(train, 4, 91)) == {0: 4, 1: 4}
    with pytest.raises(ValueError, match="fewer than"):
        select_train_rows(train, 100, 91)


def test_roundtrip_checks_dataset_integrity(tmp_path):
    train, test = source_rows()
    prepared = prepare_from_rows("fixture", ["zero", "one"], train, test, seed=42,
                                 train_limit=30, validation_limit=6, test_limit=8)
    save_prepared(prepared, tmp_path)
    loaded = load_prepared(tmp_path)
    assert loaded == prepared
    assert len(loaded.train) == 30 and len(loaded.test) == 8
    with (tmp_path / "test.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prepared(tmp_path)


def test_registry_pins_every_source_and_sst2_uses_labeled_holdout():
    specs = dataset_specs()
    assert {"sst2", "imdb", "ag_news", "trec", "banking77"} <= specs.keys()
    for spec in specs.values():
        assert len(spec["revision"]) == 40
        assert set(spec["revision"]) <= set("0123456789abcdef")
    assert specs["sst2"]["source_splits"]["test"] == "validation"
    assert specs["trec"]["label_column"] == "coarse_label"


def test_text_cap_audits_new_unicode_overlap_without_removing_test_rows():
    from jevbench.runner import cap_text
    from jevbench.types import PreparedDataset

    dataset = PreparedDataset("fixture", ["zero", "one"],
        [Row("a", "ＡＢＣＤdifferent source", 0), Row("b", "train zero", 0), Row("c", "label one train", 1)],
        [Row("d", "abcdvalidation suffix", 0), Row("e", "val zero", 0), Row("f", "positive val", 1)],
        [Row("g", "abcdtest suffix", 0), Row("h", "negative test", 1)])
    result = cap_text(dataset, 4)
    assert [row.id for row in result.test] == ["g", "h"]
    assert result.manifest["post_transform_overlap_removed"] == {"train": 1, "validation": 1}
    assert result.manifest["post_transform_overlap_removed_ids"] == {"train": ["a"], "validation": ["d"]}
    assert result.manifest["splits"]["train"]["rows"] == 2
    assert result.manifest["splits_before_text_transform"]["train"]["rows"] == 3
