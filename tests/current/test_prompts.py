import pytest
from jevbench.prompts import build_prompt, select_examples
from jevbench.types import Row


def test_sampling_balanced_seeded_and_shared():
    rows = [Row(str(i), "example", i%2) for i in range(20)]
    a = select_examples(rows, ["a", "b"], 4, 42)
    assert a == select_examples(list(reversed(rows)), ["a", "b"], 4, 42)
    assert len(a) == 8
    assert sum(r.label == 0 for r in a) == 4
    assert {r.id for r in select_examples(rows, ["a", "b"], 1, 42)} <= {r.id for r in a}


def test_prompt_does_not_include_gold_test_label():
    assert build_prompt(Row("x", "sample", 0), ["a", "b"]) == build_prompt(Row("x", "sample", 1), ["a", "b"])


def test_prevent_test_demonstration_overlap():
    row = Row("x", "sample", 0)
    with pytest.raises(ValueError):
        build_prompt(row, ["a", "b"], [row])
