from dataclasses import asdict
import json
from unittest.mock import patch

import pytest

from jevbench import runner
from jevbench.metrics import evaluate
from jevbench.providers import build_provider
from jevbench.types import Prediction, PreparedDataset, Row


@pytest.fixture
def dataset():
    return PreparedDataset(
        "toy", ["no", "yes"],
        [Row("train-a", "first training example", 0), Row("train-b", "second training example", 1)],
        [], [Row("a", "first test example", 0), Row("b", "second test example", 1)], {},
    )


CONFIG = {"provider": "openai_compatible", "model": "toy", "local": True}


@pytest.fixture
def fake_provider(monkeypatch):
    calls = []

    class Fake:
        def predict(self, row, labels, prompt):
            calls.append(row.id)
            return Prediction(row.id, 0)

    monkeypatch.setattr("jevbench.providers.build_provider", lambda _: Fake())
    return calls


@pytest.mark.parametrize("failed_file", ["test_manifest.json", "run.json"])
def test_interrupted_finalization_resumes_without_inference(
    tmp_path, monkeypatch, dataset, fake_provider, failed_file
):
    save = runner.save_json

    def interrupt(path, content):
        if path.name == failed_file and (
            failed_file == "test_manifest.json" or content.get("status") == "complete"
        ):
            raise OSError("simulated interruption")
        save(path, content)

    monkeypatch.setattr(runner, "save_json", interrupt)
    with pytest.raises(OSError, match="interruption"):
        runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
    folder = next(tmp_path.iterdir())
    assert json.loads((folder / "run.json").read_text())["status"] == "running"
    predictions_before = (folder / "predictions.jsonl").read_bytes()
    assert fake_provider == ["a", "b"]

    monkeypatch.setattr(runner, "save_json", save)
    result = runner.run_model(dataset, CONFIG, tmp_path, max_requests=0, bootstrap_samples=100)
    assert result["status"] == "complete"
    assert fake_provider == ["a", "b"]
    assert (folder / "test_manifest.json").is_file()
    assert (folder / "predictions.jsonl").read_bytes() == predictions_before


@pytest.mark.parametrize("method", ["model", "classical"])
def test_completed_resume_repairs_missing_manifest_without_new_work(
    tmp_path, monkeypatch, dataset, fake_provider, method
):
    fits = []

    def fit(*args, **kwargs):
        fits.append(1)
        return [Prediction("a", 0), Prediction("b", 0)], {}

    monkeypatch.setattr("jevbench.classical.fit_classical", fit)

    def run():
        if method == "model":
            return runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
        return runner.run_classical(dataset, "majority", tmp_path, bootstrap_samples=100)

    first = run()
    folder = tmp_path / first["run_id"]
    expected = (folder / "test_manifest.json").read_bytes()
    record_before = (folder / "run.json").read_bytes()
    (folder / "test_manifest.json").unlink()
    assert run() == first
    assert (folder / "test_manifest.json").read_bytes() == expected
    assert (folder / "run.json").read_bytes() == record_before
    assert len(fits) == (1 if method == "classical" else 0)
    assert len(fake_provider) == (2 if method == "model" else 0)


@pytest.mark.parametrize("corruption", ["manifest", "metrics", "bootstrap", "identity", "predictions"])
def test_completed_resume_rejects_corruption_without_overwriting_or_requesting(
    tmp_path, dataset, fake_provider, corruption
):
    result = runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
    folder = tmp_path / result["run_id"]
    if corruption == "manifest":
        path = folder / "test_manifest.json"
        content = json.loads(path.read_text())
        content["rows"][0]["text_sha256"] = "different"
        path.write_text(json.dumps(content))
    elif corruption in {"metrics", "bootstrap", "identity"}:
        path = folder / "run.json"
        content = json.loads(path.read_text())
        if corruption == "metrics":
            content["metrics"]["accuracy"] = 0.123
        elif corruption == "bootstrap":
            content["metrics"]["bootstrap"]["metrics"]["accuracy"]["ci95"] = [0.123, 0.456]
        else:
            content["seed"] = 99
        path.write_text(json.dumps(content))
    else:
        path = folder / "predictions.jsonl"
        path.write_text(path.read_text().rsplit("\n", 2)[0] + '\n{"broken":')
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    with pytest.raises((ValueError, TypeError)):
        runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert fake_provider == ["a", "b"]


@pytest.mark.parametrize("changed_input", ["test", "train"])
def test_input_content_changes_cannot_reuse_predictions_under_the_same_ids(
    tmp_path, dataset, fake_provider, changed_input
):
    first = runner.run_model(dataset, CONFIG, tmp_path, shots=1, bootstrap_samples=100)
    folder = tmp_path / first["run_id"]
    (folder / "test_manifest.json").unlink()
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    rows = getattr(dataset, changed_input)
    rows[0] = Row(rows[0].id, "changed input under the same ID", rows[0].label)
    with pytest.raises(ValueError, match="cap is 0"):
        runner.run_model(dataset, CONFIG, tmp_path, shots=1, max_requests=0, bootstrap_samples=100)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert fake_provider == ["a", "b"]


@pytest.mark.parametrize("last_line", [
    '{"row_id": "b", "label": 0, "unexpected_field": true}\n',
    '{"row_id": "b", "label": 0, "unexpected_field": true}',
    '{"row_id":\n',
])
def test_incomplete_run_preserves_and_rejects_corrupt_complete_records(
    tmp_path, monkeypatch, dataset, fake_provider, last_line
):
    save = runner.save_json

    def interrupt(path, content):
        if path.name == "test_manifest.json":
            raise OSError("interrupted")
        save(path, content)

    monkeypatch.setattr(runner, "save_json", interrupt)
    with pytest.raises(OSError):
        runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
    monkeypatch.setattr(runner, "save_json", save)
    folder = next(tmp_path.iterdir())
    predictions = folder / "predictions.jsonl"
    predictions.write_text(predictions.read_text().splitlines()[0] + "\n" + last_line)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    with pytest.raises((ValueError, TypeError)):
        runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert fake_provider == ["a", "b"]


def test_only_unterminated_final_json_can_be_recovered(tmp_path, monkeypatch, dataset, fake_provider):
    save = runner.save_json

    def interrupt(path, content):
        if path.name == "test_manifest.json":
            raise OSError("interrupted")
        save(path, content)

    monkeypatch.setattr(runner, "save_json", interrupt)
    with pytest.raises(OSError):
        runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
    monkeypatch.setattr(runner, "save_json", save)
    folder = next(tmp_path.iterdir())
    predictions = folder / "predictions.jsonl"
    first_line = predictions.read_text().splitlines()[0]
    predictions.write_text(first_line + '\n{"row_id":')
    result = runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
    assert result["status"] == "complete"
    assert predictions.read_text().splitlines()[0] == first_line
    assert fake_provider == ["a", "b", "b"]


def test_complete_json_without_newline_is_reused_before_appending(
    tmp_path, monkeypatch, dataset, fake_provider
):
    save = runner.save_json

    def interrupt(path, content):
        if path.name == "test_manifest.json":
            raise OSError("interrupted")
        save(path, content)

    monkeypatch.setattr(runner, "save_json", interrupt)
    with pytest.raises(OSError):
        runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
    monkeypatch.setattr(runner, "save_json", save)
    folder = next(tmp_path.iterdir())
    path = folder / "predictions.jsonl"
    first_line = path.read_text().splitlines()[0]
    path.write_text(first_line)
    result = runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
    assert result["status"] == "complete"
    lines = path.read_text().splitlines()
    assert lines[0] == first_line
    assert [json.loads(line)["row_id"] for line in lines] == ["a", "b"]
    assert fake_provider == ["a", "b", "b"]
    assert runner.run_model(dataset, CONFIG, tmp_path, max_requests=0, bootstrap_samples=100) == result
    assert fake_provider == ["a", "b", "b"]


@pytest.mark.parametrize("train_per_class", [None, 1])
def test_classical_cache_pins_validation_content_only_when_used(
    tmp_path, monkeypatch, dataset, train_per_class
):
    fits = []

    def fit(*args, **kwargs):
        fits.append(1)
        return [Prediction(row.id, 0) for row in dataset.test], {}

    monkeypatch.setattr("jevbench.classical.fit_classical", fit)
    dataset.validation = [Row("validation-a", "validation example", 0)]
    first = runner.run_classical(dataset, "majority", tmp_path,
                                 train_per_class=train_per_class, bootstrap_samples=100)
    first_folder = tmp_path / first["run_id"]
    original = {path.name: path.read_bytes() for path in first_folder.iterdir()}
    dataset.validation = [Row("validation-a", "changed validation example", 1)]
    second = runner.run_classical(dataset, "majority", tmp_path,
                                  train_per_class=train_per_class, bootstrap_samples=100)
    assert {path.name: path.read_bytes() for path in first_folder.iterdir()} == original
    if train_per_class is None:
        assert first["run_id"] != second["run_id"]
        assert second["validation_records_sha256"] == runner.digest([asdict(dataset.validation[0])])
        assert len(fits) == 2
    else:
        assert first == second
        assert second["validation_records_sha256"] == runner.digest([])
        assert len(fits) == 1


def test_duplicate_test_ids_are_rejected_before_inference(tmp_path, dataset, fake_provider):
    dataset.test.append(dataset.test[0])
    with pytest.raises(ValueError, match="Test row IDs must be unique"):
        runner.run_model(dataset, CONFIG, tmp_path, bootstrap_samples=100)
    assert fake_provider == []
    assert list(tmp_path.iterdir()) == []


def test_duplicate_test_ids_are_rejected_by_standalone_metrics(dataset):
    with pytest.raises(ValueError, match="Test row IDs must be unique"):
        evaluate(dataset.test + [dataset.test[0]], [Prediction("a", 0), Prediction("b", 0)], 2)


@pytest.mark.parametrize("provider,response", [
    ("openai", {"choices": [None]}),
    ("openai", {"choices": "invalid"}),
    ("openai", {"choices": [{"finish_reason": "stop", "message": None}]}),
    ("anthropic", {"stop_reason": "end_turn", "content": [None]}),
    ("anthropic", {"stop_reason": "end_turn", "content": "invalid"}),
    ("gemini", {"candidates": [None]}),
    ("gemini", {"candidates": [{"finishReason": "STOP", "content": None}]}),
    ("gemini", {"candidates": [{"finishReason": "STOP", "content": {"parts": [None]}}]}),
    ("jev", {"answers": None}),
    ("jev", {"answers": {"classification": None}}),
])
def test_malformed_nested_response_is_recorded_with_usage(provider, response):
    response.update({"model": "snapshot", "modelVersion": "snapshot"})
    response["usage"] = {"prompt_tokens": 10, "completion_tokens": 1,
                         "input_tokens": 10, "output_tokens": 1}
    response["usageMetadata"] = {"promptTokenCount": 10, "candidatesTokenCount": 1}
    classifier = build_provider({"provider": provider, "model": "mock", "api_key_env": None})
    with patch("jevbench.providers._post_json", return_value=response) as post:
        result = classifier.predict(Row("a", "input", 0), ["no", "yes"], "prompt")
    assert post.call_count == 1
    assert result.label is None and result.error.startswith("invalid_output:")
    assert (result.input_tokens, result.output_tokens) == (10, 1)
    assert result.metadata["resolved_model"] == "snapshot"
    assert "invalid" not in result.metadata.values()


def test_malformed_response_is_checkpointed_and_not_retried_on_resume(tmp_path, dataset):
    responses = [
        {"usage": {"prompt_tokens": 10, "completion_tokens": 1}, "choices": [None]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "1"}}]},
    ]
    config = {**CONFIG, "api_key_env": None}
    with patch("jevbench.providers._post_json", side_effect=responses) as post:
        first = runner.run_model(dataset, config, tmp_path, bootstrap_samples=100)
        second = runner.run_model(dataset, config, tmp_path, max_requests=0, bootstrap_samples=100)
    assert post.call_count == 2
    assert first == second
    assert first["metrics"]["n_failures"] == 1
    rows = (tmp_path / first["run_id"] / "predictions.jsonl").read_text().splitlines()
    assert len(rows) == 2
    assert json.loads(rows[0])["input_tokens"] == 10
