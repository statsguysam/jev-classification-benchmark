import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jevbench import providers
from jevbench.types import PreparedDataset, Row

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("openrouter_jev", ROOT / "scripts/run_openrouter_jev.py")
jev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jev)
CONFIG = {"provider": "jev", "model": jev.MODEL, "base_url": jev.BASE_URL,
          "api_key_env": "OPENROUTER_API_KEY", "timeout": 120}


@pytest.fixture(autouse=True)
def dummy_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "DUMMY_TEST_ONLY_NOT_A_REAL_KEY")


def response(cost="0.000019992"):
    return {"id": "gen-dec-123-test", "model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe",
            "answers": {"classification": {"type": "choice", "choice": "1",
                        "probabilities": {"0": 0.25, "1": 0.75}, "confidence": 0.75}},
            "usage": {"input_tokens": 476, "output_tokens": 70, "cost": cost}}


def wrapped(tmp_path, cap="2.50", quota=None):
    ledger = jev.budget.Ledger(tmp_path / "ledger.jsonl", cap, initialize=True)
    return jev.GuardedOpenRouterJev(CONFIG, ledger, {}, quota), ledger


def predict(classifier):
    return classifier.predict(Row("test-id", "UNUSED_TEXT", 999), ["negative", "positive"], "shared prompt")


def test_exact_payload_reservation_before_http_native_output_and_cost(tmp_path):
    classifier, ledger = wrapped(tmp_path)

    def fake_post(url, body, headers, timeout):
        assert ledger.snapshot()["reservations"] == 1
        assert ledger.snapshot()["charged_or_reserved_usd"] == "0.002688000"
        assert url == jev.ENDPOINT
        assert body == {"model": jev.MODEL, "state": "shared prompt", "questions": {"classification": {
            "type": "choice", "instructions": "Choose the numeric class id for the final text in the state, following its classification task and examples.",
            "criteria": {"0": "negative", "1": "positive"}}}}
        assert timeout == 120
        return response()

    with patch.object(jev, "post_json_exact_cost", fake_post):
        result = predict(classifier)
    assert result.label == 1 and result.probabilities == [0.25, 0.75]
    assert result.input_tokens == 476 and result.output_tokens == 70
    assert result.metadata["resolved_model"] == "typesafe/jev-1.13-20260917"
    assert result.metadata["probability_kind"] == "jev_choice_distribution"
    assert result.metadata["openrouter"]["reported_cost_usd"] == "0.000019992"
    assert result.metadata["openrouter"]["provider"] == "TypeSafe"
    assert result.metadata["budget"]["reserved_upper_bound_usd"] == "0.002688000"
    assert ledger.snapshot()["settlements"] == 0
    assert "DUMMY_TEST" not in ledger.path.read_text()
    assert "shared prompt" not in ledger.path.read_text()


@pytest.mark.parametrize("cost, expected", [(0, "0"), (None, None), ("0.00000000000000001234567890", "1.234567890E-17")])
def test_cost_zero_missing_and_precision_retained(tmp_path, cost, expected):
    classifier, ledger = wrapped(tmp_path)
    with patch.object(jev, "post_json_exact_cost", return_value=response(cost)):
        result = predict(classifier)
    assert result.metadata["openrouter"]["reported_cost_usd"] == expected
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.002688000"


def test_transport_keeps_wire_cost_precision_and_float_probabilities():
    raw = json.dumps(response()).replace('"0.000019992"', '0.00001999212345678901234567890').encode()
    connection = MagicMock()
    connection.__enter__.return_value.read.return_value = raw
    opener = MagicMock()
    opener.open.return_value = connection
    with patch.object(jev.urllib.request, "build_opener", return_value=opener) as build:
        parsed = jev.post_json_exact_cost(jev.ENDPOINT, {}, {}, 120)
    assert parsed["usage"]["cost"] == "0.00001999212345678901234567890"
    assert type(parsed["answers"]["classification"]["probabilities"]["0"]) is float
    assert isinstance(build.call_args.args[0], providers._NoRedirect)
    assert opener.open.call_count == 1


def test_malformed_answer_keeps_billable_metadata_and_reservation(tmp_path):
    classifier, ledger = wrapped(tmp_path)
    malformed = response()
    malformed["answers"]["classification"]["choice"] = "explanation"
    with patch.object(jev, "post_json_exact_cost", return_value=malformed):
        result = predict(classifier)
    assert result.error and result.label is None
    assert result.input_tokens == 476
    assert result.metadata["openrouter"]["reported_cost_usd"] == "0.000019992"
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.002688000"


def test_timeout_no_retry_keeps_reservation_on_resume(tmp_path):
    classifier, ledger = wrapped(tmp_path)
    with patch.object(jev, "post_json_exact_cost", side_effect=providers.ProviderError("network_error: timed out")) as post:
        result = predict(classifier)
    assert result.error and post.call_count == 1
    resumed = jev.budget.Ledger(ledger.path, "2.50")
    assert resumed.snapshot()["reservations"] == 1
    assert resumed.snapshot()["charged_or_reserved_usd"] == "0.002688000"


def test_insufficient_budget_prevents_http(tmp_path):
    classifier, ledger = wrapped(tmp_path, cap="0.002687")
    with patch.object(jev, "post_json_exact_cost") as post:
        with pytest.raises(jev.budget.BudgetStop):
            predict(classifier)
    post.assert_not_called()
    assert ledger.snapshot()["reservations"] == 0


@pytest.mark.parametrize("changed", [{"model": "typesafe/other"}, {"model": {}},
    {"provider": "Unexpected"}, {"usage": {"input_tokens": 64001, "output_tokens": 0}},
    {"usage": {"input_tokens": 1, "output_tokens": 0, "cost": "0.002689"}},
    {"usage": {"cost": "NaN"}}])
def test_route_usage_or_cost_violation_durably_halts_ledger(tmp_path, changed):
    classifier, ledger = wrapped(tmp_path)
    with patch.object(jev, "post_json_exact_cost", return_value={**response(), **changed}):
        with pytest.raises(jev.budget.GuardError, match="ledger halted"):
            predict(classifier)
    resumed = jev.budget.Ledger(ledger.path, "2.50")
    assert resumed.snapshot()["halted"]
    with pytest.raises(jev.budget.GuardError, match="halted"):
        resumed.reserve(jev.RESERVE_NANO, {})


@pytest.mark.parametrize("changed", [{"base_url": "https://evil.example"}, {"model": "jev-1.13"},
    {"api_key_env": "OTHER_KEY"}, {"api_key": "secret"}, {"temperature": 0}, {"timeout": float("nan")}])
def test_config_rejects_unverified_route_or_fields(changed):
    with pytest.raises(jev.budget.GuardError):
        jev.validate_config({**CONFIG, **changed})


def test_payload_change_is_rejected_before_reservation(tmp_path):
    classifier, ledger = wrapped(tmp_path)

    def wrong_predict(row, labels, prompt):
        return providers._post_json(jev.ENDPOINT, {"model": jev.MODEL, "state": prompt, "questions": {}}, {}, 120)

    classifier.inner.predict = wrong_predict
    with patch.object(jev, "post_json_exact_cost") as post:
        with pytest.raises(jev.budget.GuardError, match="frozen native Choice"):
            predict(classifier)
    post.assert_not_called()
    assert ledger.snapshot()["reservations"] == 0


def test_first_row_checkpoint_resumes_same_run_without_repeat(tmp_path, monkeypatch):
    dataset = PreparedDataset("toy", ["negative", "positive"], [Row("tr0", "bad", 0), Row("tr1", "good", 1)],
                              [], [Row("te0", "bad sample", 0), Row("te1", "good sample", 1)], {})
    monkeypatch.setattr("jevbench.data.load_prepared", lambda _: dataset)
    monkeypatch.setattr(jev, "VERIFIED_ON", datetime.now(timezone.utc).date().isoformat())
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"jev_openrouter": CONFIG}))
    args = ["--config", str(config), "--data", str(tmp_path / "unused"), "--shots", "0",
            "--output", str(tmp_path / "runs"), "--ledger", str(tmp_path / "ledger.jsonl"),
            "--bootstrap-samples", "100", "--execute"]
    with patch.object(jev, "post_json_exact_cost", return_value=response()) as post:
        first = jev.main(args + ["--init-ledger", "--stop-after-new-requests", "1"])
        assert first["status"] == "paused_after_requested_new_requests"
        run_dirs = list((tmp_path / "runs").iterdir())
        assert len(run_dirs) == 1
        assert len((run_dirs[0] / "predictions.jsonl").read_text().splitlines()) == 1
        final = jev.main(args)
        assert final["completed"] == [run_dirs[0].name]
        assert post.call_count == 2
        again = jev.main(args)
        assert again["completed"] == final["completed"] and post.call_count == 2
    predictions = [json.loads(line) for line in (run_dirs[0] / "predictions.jsonl").read_text().splitlines()]
    assert [p["row_id"] for p in predictions] == ["te0", "te1"]
    assert final["budget"]["reservations"] == 2


def test_frozen_ledger_hash_and_matrix_reservation():
    assert hashlib.sha256(jev._BUDGET_PATH.read_bytes()).hexdigest() == jev.FROZEN_BUDGET_SHA256
    assert jev.budget.usd_string(800 * jev.RESERVE_NANO) == "2.150400000"
