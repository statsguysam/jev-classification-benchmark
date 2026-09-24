import json
import pytest
from jevbench.runner import run_classical, run_model, compare_runs
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


@pytest.fixture
def completed_model(tmp_path, monkeypatch):
    class Fake:
        def predict(self, row, labels, prompt):
            return Prediction(row.id, 0)

    monkeypatch.setattr("jevbench.providers.build_provider", lambda _: Fake())

    def run(*, dataset=None, model="toy", shots=1):
        config = {"provider": "openai_compatible", "model": model, "local": True}
        record = run_model(dataset or fixture_dataset(), config, tmp_path,
                           shots=shots, bootstrap_samples=100)
        return tmp_path / record["run_id"]

    return run


def test_compare_refuses_unequal_training_without_explicit_flag(completed_model):
    a = completed_model(shots=0)
    b = completed_model(shots=1)
    with pytest.raises(ValueError, match="Unequal"):
        compare_runs(a, b, samples=100)
    result = compare_runs(a, b, samples=100, allow_unequal_training=True)
    assert result["equal_training_ids_and_seed"] is False


def test_compare_accepts_verified_runs_with_matching_training_content(completed_model):
    a = completed_model(model="source-a")
    b = completed_model(model="source-b")
    result = compare_runs(a, b, samples=100)
    assert result["equal_training_ids_and_seed"] is True
    assert result["metrics"]["accuracy"]["estimate"] == 0
    assert result["run_a"] == a.name and result["run_b"] == b.name


@pytest.mark.parametrize("corruption, message", [
    ("duplicate_prediction", "Predictions must match"),
    ("extra_prediction", "Predictions must match"),
    ("missing_prediction", "Predictions must match"),
    ("running", "complete run records"),
    ("score", "metrics disagree"),
    ("bootstrap", "metrics disagree"),
    ("manifest", "saved content identity"),
    ("dataset_identity", "dataset identity differ"),
    ("legacy_identity", "frozen study runtime"),
])
def test_compare_rejects_invalid_evidence_without_mutation(completed_model, corruption, message):
    a = completed_model(model="source-a")
    b = completed_model(model="source-b")
    if corruption.endswith("prediction"):
        path = a / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if corruption == "missing_prediction":
            rows.pop()
        else:
            row = {**rows[0], "label": 1}
            if corruption == "extra_prediction":
                row["row_id"] = "unexpected-test-id"
            rows.append(row)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif corruption == "manifest":
        path = a / "test_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["rows"][0]["text_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
    else:
        path = a / "run.json"
        record = json.loads(path.read_text())
        if corruption == "running":
            record["status"] = "running"
        elif corruption == "score":
            record["metrics"]["accuracy"] = 0.123
        elif corruption == "bootstrap":
            record["metrics"]["bootstrap"]["metrics"]["accuracy"]["ci95"] = [0.123, 0.456]
        elif corruption == "dataset_identity":
            record["dataset_manifest"] = {"changed": True}
        else:
            del record["test_manifest_sha256"]
        path.write_text(json.dumps(record))
    before = {path: path.read_bytes() for folder in (a, b) for path in folder.iterdir()}
    with pytest.raises(ValueError, match=message):
        compare_runs(a, b, samples=100, allow_unequal_training=True)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_training_ids_alone_do_not_establish_equal_training(completed_model):
    dataset = fixture_dataset()
    a = completed_model(dataset=dataset)
    original = dataset.train[0]
    dataset.train[0] = Row(original.id, "changed training input", original.label)
    b = completed_model(dataset=dataset)
    assert a != b
    with pytest.raises(ValueError, match="training content"):
        compare_runs(a, b, samples=100)
    result = compare_runs(a, b, samples=100, allow_unequal_training=True)
    assert result["equal_training_ids_and_seed"] is False


def test_validation_contents_are_part_of_equal_training(tmp_path):
    dataset = fixture_dataset()
    dataset.validation = [Row("v0", "bad validation", 0), Row("v1", "good validation", 1)]
    first = run_classical(dataset, "logistic_regression", tmp_path, bootstrap_samples=100)
    dataset.validation[0] = Row("v0", "different bad validation", 0)
    second = run_classical(dataset, "logistic_regression", tmp_path, bootstrap_samples=100)
    a, b = tmp_path / first["run_id"], tmp_path / second["run_id"]
    assert a != b
    with pytest.raises(ValueError, match="validation use"):
        compare_runs(a, b, samples=100)
    result = compare_runs(a, b, samples=100, allow_unequal_training=True)
    assert result["equal_training_ids_and_seed"] is False
