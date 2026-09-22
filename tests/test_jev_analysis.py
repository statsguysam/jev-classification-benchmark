import json
import hashlib
from pathlib import Path

import pytest

from jevbench.metrics import evaluate
from jevbench.types import Prediction, Row


def jev_config():
    return {"provider": "jev", "model": "typesafe/jev-1.13", "shots_per_class": 4,
            "base_url": "https://openrouter.ai/api/v1", "budget_guard": {
                "dataset": "sst2", "seed": 42, "shots_per_class": 4, "route": "OpenRouter",
                "endpoint": "https://openrouter.ai/api/v1/systemone", "native_protocol": "frozen_JevClassifier_Choice",
                "response_model_allowlist": ["typesafe/jev-1.13", "typesafe/jev-1.13-20260917"],
                "wrapper_sha256": hashlib.sha256((Path(__file__).parents[1] / "scripts/run_openrouter_jev.py").read_bytes()).hexdigest()}}


def jev_metadata():
    return {"probability_kind": "jev_choice_distribution", "requested_model": "typesafe/jev-1.13",
            "resolved_model": "typesafe/jev-1.13-20260917", "openrouter": {
                "provider": "TypeSafe", "endpoint": "https://openrouter.ai/api/v1/systemone"}}


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import compare_combined_pilot
    import summarize_jev_pilot
    return compare_combined_pilot, summarize_jev_pilot


@pytest.fixture
def probability_run(tmp_path):
    rows = [Row("a", "", 0), Row("b", "", 1), Row("c", "", 1)]
    predictions = [
        {"row_id": "a", "label": 0, "probabilities": [0.8, 0.2], "metadata": jev_metadata()},
        {"row_id": "b", "label": 1, "probabilities": [0.1, 0.9], "metadata": jev_metadata()},
        {"row_id": "c", "label": None, "probabilities": None, "error": "network_error"},
    ]
    test = {"rows": [{"id": row.id, "label": row.label} for row in rows]}
    record = {"run_id": "fixture", "dataset": "sst2", "seed": 42, "config": jev_config(),
              "method": "few_shot", "labels": ["negative", "positive"],
              "metrics": evaluate(rows, [Prediction(**value) for value in predictions], 2)}
    (tmp_path / "predictions.jsonl").write_text("".join(json.dumps(value) + "\n" for value in predictions))
    return tmp_path, record, test, predictions


def test_probability_report_keeps_conditional_coverage_and_failures(modules, probability_run):
    _, summary = modules
    directory, record, test, _ = probability_run
    measured, proof = summary.recompute(directory, record, test)
    assert measured["n_failures"] == 1
    assert measured["probability_coverage"] == 2 / 3
    assert measured["accuracy"] == 2 / 3
    assert measured["n_probability_rows"] == 2
    assert proof["recorded_probability_kinds"] == ["jev_choice_distribution"]
    assert proof["failure_breakdown"] == {"network_error": 1}
    assert proof["failed_rows_without_retained_probabilities"] == 1


def test_argmax_rejection_is_reported_even_when_valid_rows_have_zero_disagreements(modules, probability_run):
    _, summary = modules
    directory, record, test, predictions = probability_run
    error = "invalid_output: Jev choice disagrees with maximum probability"
    predictions[2]["error"] = error
    (directory / "predictions.jsonl").write_text("".join(json.dumps(value) + "\n" for value in predictions))
    measured, proof = summary.recompute(directory, record, test)
    assert proof["failure_breakdown"] == {error: 1}
    assert measured["chosen_label_argmax_disagreement_rows"] == 0
    assert measured["n_failures"] == proof["failed_rows_without_retained_probabilities"] == 1
    assert measured["n_probability_rows"] == 2


def test_probability_report_rejects_forged_nll(modules, probability_run):
    _, summary = modules
    directory, record, test, _ = probability_run
    record["metrics"]["log_loss"] = 0.0
    with pytest.raises(ValueError, match="metric log_loss differs"):
        summary.recompute(directory, record, test)


def test_probability_report_rejects_invalid_distribution(modules, probability_run):
    _, summary = modules
    directory, record, test, predictions = probability_run
    predictions[0]["probabilities"] = [0.9, 0.9]
    (directory / "predictions.jsonl").write_text("".join(json.dumps(value) + "\n" for value in predictions))
    with pytest.raises(ValueError, match="Invalid class probability"):
        summary.recompute(directory, record, test)


def test_probability_report_rejects_wrong_native_provenance(modules, probability_run):
    _, summary = modules
    directory, record, test, predictions = probability_run
    predictions[0]["metadata"]["probability_kind"] = "generated_self_report"
    (directory / "predictions.jsonl").write_text("".join(json.dumps(value) + "\n" for value in predictions))
    with pytest.raises(ValueError, match="native probability provenance"):
        summary.recompute(directory, record, test)


def test_jev_contrasts_are_exactly_three_matched_few_shot_pairs(modules):
    compare, _ = modules
    specs = [row for row in compare.contrast_specs() if row[1] == "jev"]
    assert len(specs) == 3
    assert {row[3] for row in specs} == {"nb_pilot", "astra", "qwen4b"}
    assert all(row[2] == "few_shot" and row[4] in ("few_shot", "classical") for row in specs)
    assert len(compare.contrast_specs()) == 11


def test_jev_discovery_never_selects_partial_or_ambiguous_runs(modules, tmp_path):
    compare, _ = modules
    record = {"dataset": "sst2", "config": jev_config(),
              "method": "few_shot", "seed": 42, "status": "running"}
    directory = tmp_path / "jev" / "one"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps(record))
    selected, reason = compare.completed_candidate(tmp_path, "sst2", "jev", "few_shot", 42)
    assert selected is None and "incomplete" in reason
    record["status"] = "complete"
    (directory / "run.json").write_text(json.dumps(record))
    duplicate = tmp_path / "jev" / "two"
    duplicate.mkdir()
    (duplicate / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Ambiguous completed runs"):
        compare.completed_candidate(tmp_path, "sst2", "jev", "few_shot", 42)


@pytest.mark.parametrize("change", ["base_url", "guard_missing", "wrapper_sha", "allowlist", "resolved_model", "provider", "endpoint", "row_route_missing"])
def test_fixed_jev_report_rejects_unverified_route_provenance(modules, probability_run, change):
    _, summary = modules
    directory, record, test, predictions = probability_run
    if change == "base_url":
        record["config"]["base_url"] = "https://example.com/v1"
    elif change == "guard_missing":
        record["config"].pop("budget_guard")
    elif change == "wrapper_sha":
        record["config"]["budget_guard"]["wrapper_sha256"] = "a" * 64
    elif change == "allowlist":
        record["config"]["budget_guard"]["response_model_allowlist"].append("unverified-model")
    elif change == "resolved_model":
        predictions[0]["metadata"]["resolved_model"] = "unverified-model"
    elif change == "provider":
        predictions[0]["metadata"]["openrouter"]["provider"] = "OtherProvider"
    elif change == "endpoint":
        predictions[0]["metadata"]["openrouter"]["endpoint"] = "https://example.com/v1/systemone"
    else:
        predictions[0]["metadata"].pop("openrouter")
    (directory / "predictions.jsonl").write_text("".join(json.dumps(value) + "\n" for value in predictions))
    with pytest.raises(ValueError):
        summary.recompute(directory, record, test)
