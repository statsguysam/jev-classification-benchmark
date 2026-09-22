import json
from dataclasses import asdict
import pytest
from jevbench.runner import run_model, compare_runs
from jevbench.types import Row, PreparedDataset, Prediction


def fixture_dataset():
    return PreparedDataset("toy", ["negative", "positive"], [Row("tr0", "bad", 0), Row("tr1", "good", 1)], [], [Row("te0", "bad sample", 0), Row("te1", "good sample", 1)], {})


def test_paid_access_is_blocked_before_building_provider(tmp_path, monkeypatch):
    monkeypatch.setattr("jevbench.providers.build_provider", lambda _: pytest.fail("provider must not be built"))
    with pytest.raises(ValueError, match="allow-paid"):
        run_model(fixture_dataset(), {"provider": "jev", "model": "jev-1.13.0"}, tmp_path)


def test_request_cap_prevents_partial_unintended_run(tmp_path, monkeypatch):
    monkeypatch.setattr("jevbench.providers.build_provider", lambda _: pytest.fail("provider must not be built"))
    with pytest.raises(ValueError, match="cap is 1"):
        run_model(fixture_dataset(), {"provider": "openai_compatible", "model": "toy", "local": True}, tmp_path, max_requests=1)


def test_completed_run_resume_has_no_additional_requests(tmp_path, monkeypatch):
    calls = []
    class Fake:
        def predict(self, row, labels, prompt):
            calls.append(row.id)
            return Prediction(row.id, 0)
    monkeypatch.setattr("jevbench.providers.build_provider", lambda _: Fake())
    config = {"provider": "openai_compatible", "model": "toy", "local": True}
    first = run_model(fixture_dataset(), config, tmp_path, bootstrap_samples=100)
    second = run_model(fixture_dataset(), config, tmp_path, bootstrap_samples=100)
    assert first == second
    assert calls == ["te0", "te1"]


def test_compare_refuses_unequal_training_without_explicit_flag(tmp_path):
    manifest = {"dataset": "toy", "labels": ["a", "b"], "rows": [{"id": "t0", "label": 0}, {"id": "t1", "label": 1}]}
    for name, ids in (("a", ["train0"]), ("b", ["train0", "train1"])):
        path = tmp_path / name
        path.mkdir()
        (path / "test_manifest.json").write_text(json.dumps(manifest))
        (path / "run.json").write_text(json.dumps({"run_id": name, "seed": 42, "training_example_ids": ids}))
        (path / "predictions.jsonl").write_text("\n".join(json.dumps(asdict(Prediction("t"+str(i), i))) for i in range(2)))
    with pytest.raises(ValueError, match="Unequal"):
        compare_runs(tmp_path / "a", tmp_path / "b", samples=100)
    result = compare_runs(tmp_path / "a", tmp_path / "b", samples=100, allow_unequal_training=True)
    assert result["equal_training_ids_and_seed"] is False
