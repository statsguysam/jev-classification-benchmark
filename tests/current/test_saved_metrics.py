"""Check maintained scoring against the original 136 study conditions."""
import json
from pathlib import Path

import pytest

from jevbench.metrics import evaluate
from jevbench.types import Prediction, Row


ROOT = Path(__file__).resolve().parents[2]


def test_current_metrics_reproduce_saved_numeric_and_text_scores():
    checked = set()
    for name in ("numeric_expansion", "text_extension"):
        summary = json.loads((ROOT / "results" / name / "COMPARISON.json").read_text())
        for item in summary["runs"]:
            path = ROOT / item["source_path"]
            if path in checked:
                continue
            checked.add(path)
            record = json.loads(path.read_text())
            manifest = json.loads(path.with_name("test_manifest.json").read_text())
            rows = [Row(row["id"], "", row["label"]) for row in manifest["rows"]]
            predictions = [Prediction(**json.loads(line))
                           for line in path.with_name("predictions.jsonl").read_text().splitlines()]
            measured = evaluate(rows, predictions, len(record["labels"]))
            for metric, value in measured.items():
                expected = record["metrics"][metric]
                if isinstance(value, float):
                    assert value == pytest.approx(expected, abs=1e-12), (path, metric)
                else:
                    assert value == expected, (path, metric)
    assert len(checked) == 136
