import importlib.util
import json
import subprocess
import sys
import warnings
import os
import select
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock

import pytest

from jevbench import providers
from jevbench.types import Row

SPEC = importlib.util.spec_from_file_location("budgeted_hosted", Path(__file__).parents[1]/"scripts/run_budgeted_hosted.py")
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


def test_ledger_concurrency_and_resume_never_exceeds_cap(tmp_path):
    path = tmp_path/"ledger.jsonl"
    budget.Ledger(path, "0.0000017", initialize=True)

    def reserve(_):
        ledger = budget.Ledger(path, "0.0000017")
        try:
            ledger.reserve(100, {})
            return 1
        except budget.BudgetStop:
            return 0

    with ThreadPoolExecutor(max_workers=12) as pool:
        assert sum(pool.map(reserve, range(50))) == 17
    resumed = budget.Ledger(path, "0.0000017", initialize=True)
    assert resumed.snapshot()["charged_or_reserved_usd"] == "0.000001700"
    assert resumed.snapshot()["reservations"] == 17
    with pytest.raises(budget.BudgetStop):
        resumed.reserve(1, {})
    with pytest.raises(budget.GuardError, match="immutable"):
        budget.Ledger(path, "10", initialize=True)


def test_crash_reservation_is_not_refunded(tmp_path):
    path = tmp_path/"ledger.jsonl"
    ledger = budget.Ledger(path, "1", initialize=True)
    ledger.reserve(budget.usd_nano("0.8"), {"request": "about to dispatch"})
    resumed = budget.Ledger(path, "1")
    with pytest.raises(budget.BudgetStop):
        resumed.reserve(budget.usd_nano("0.3"), {})
    assert resumed.snapshot()["charged_or_reserved_usd"] == "0.800000000"


def test_valid_suffix_deletion_is_detected_by_durable_anchor(tmp_path):
    path = tmp_path/"ledger.jsonl"
    ledger = budget.Ledger(path, "1", initialize=True)
    header = path.read_text()
    ledger.reserve(100, {})
    path.write_text(header)
    with pytest.raises(budget.GuardError, match="anchor"):
        budget.Ledger(path, "1")


def test_missing_ledger_cannot_be_reinitialized_with_old_identity(tmp_path):
    path = tmp_path/"ledger.jsonl"
    budget.Ledger(path, "1", initialize=True)
    path.unlink()
    with pytest.raises(budget.GuardError, match="Missing or corrupt"):
        budget.Ledger(path, "1", initialize=True)


def test_settlement_is_atomic_exactly_once(tmp_path):
    ledger = budget.Ledger(tmp_path/"ledger.jsonl", "1", initialize=True)
    ident = ledger.reserve(budget.usd_nano("0.8"), {})
    ledger.settle(ident, budget.usd_nano("0.2"), {"complete_usage": True})
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.200000000"
    with pytest.raises(budget.GuardError, match="already settled"):
        ledger.settle(ident, 0, {})
    ledger.reserve(budget.usd_nano("0.8"), {})
    with pytest.raises(budget.BudgetStop):
        ledger.reserve(1, {})


def test_overrun_persistently_halts_future_requests(tmp_path):
    path = tmp_path/"ledger.jsonl"
    ledger = budget.Ledger(path, "1", initialize=True)
    ident = ledger.reserve(100, {})
    with pytest.raises(budget.GuardError, match="exceeds"):
        ledger.settle(ident, 101, {})
    resumed = budget.Ledger(path, "1")
    assert resumed.snapshot()["halted"]
    with pytest.raises(budget.GuardError, match="halted"):
        resumed.reserve(1, {})


def _openai(tmp_path):
    config = budget.validate_config({"provider": "openai", "model": "gpt-5.6-luna", "max_output_tokens": 32})
    ledger = budget.Ledger(tmp_path/"ledger.jsonl", "1", initialize=True)
    wrapped = budget.BudgetedProvider(providers.build_provider(config), config, budget.PRICES[config["model"]], ledger, {})
    return wrapped, ledger


def _response(text="1", finish="stop", usage=True):
    result = {"model": "gpt-5.6-luna", "choices": [{"finish_reason": finish, "message": {"content": text}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 100, "completion_tokens": 10}
    return result


def test_reservation_precedes_http_and_valid_usage_settles(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    wrapped, ledger = _openai(tmp_path)

    def fake_post(url, body, headers, timeout):
        assert ledger.snapshot()["reservations"] == 1
        assert ledger.snapshot()["settlements"] == 0
        assert body["service_tier"] == "default"
        return _response()

    with patch.object(providers, "_post_json", fake_post):
        prediction = wrapped.predict(Row("id", "unused", 0), ["a", "b"], "prompt")
    assert prediction.label == 1
    assert ledger.snapshot()["settlements"] == 1
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.000037000"
    assert "not-a-real-key" not in ledger.path.read_text()


@pytest.mark.parametrize("response", [_response(text="bad answer"), _response(finish="length"), _response(usage=False)])
def test_invalid_or_unknown_usage_keeps_full_reservation(tmp_path, monkeypatch, response):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    wrapped, ledger = _openai(tmp_path)
    expected, _ = budget.reservation(wrapped.config, wrapped.price, "prompt")
    with patch.object(providers, "_post_json", return_value=response):
        wrapped.predict(Row("id", "unused", 0), ["a", "b"], "prompt")
    assert ledger.snapshot()["settlements"] == 0
    assert ledger.snapshot()["charged_or_reserved_usd"] == budget.usd_string(expected)


def test_timeout_keeps_reservation_on_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    wrapped, ledger = _openai(tmp_path)
    with patch.object(providers, "_post_json", side_effect=providers.ProviderError("network_error: timed out")):
        prediction = wrapped.predict(Row("id", "unused", 0), ["a", "b"], "prompt")
    assert prediction.error
    resumed = budget.Ledger(ledger.path, "1")
    assert resumed.snapshot()["reservations"] == 1
    assert resumed.snapshot()["settlements"] == 0


def test_budget_stop_makes_no_network_call(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-real-key")
    config = budget.validate_config({"provider": "jev", "model": "jev-1.13.0"})
    ledger = budget.Ledger(tmp_path/"ledger.jsonl", "0.001", initialize=True)
    wrapped = budget.BudgetedProvider(providers.build_provider(config), config, budget.PRICES[config["model"]], ledger, {})
    with patch.object(providers, "_post_json") as post:
        with pytest.raises(budget.BudgetStop):
            wrapped.predict(Row("id", "unused", 0), ["a", "b"], "prompt")
    post.assert_not_called()
    assert ledger.snapshot()["reservations"] == 0


def test_unexpected_output_usage_halts_across_process_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    wrapped, ledger = _openai(tmp_path)
    response = _response()
    response["usage"]["completion_tokens"] = 100
    with patch.object(providers, "_post_json", return_value=response):
        with pytest.raises(budget.GuardError, match="exceeded"):
            wrapped.predict(Row("id", "unused", 0), ["a", "b"], "prompt")
    resumed = budget.Ledger(ledger.path, "1")
    with pytest.raises(budget.GuardError, match="halted"):
        resumed.reserve(1, {})


@pytest.mark.parametrize("extra", [{"base_url": "https://evil.example/v1"}, {"api_key": "secret"},
                                 {"service_tier": "priority"}, {"tools": []}, {"api_key_env": None},
                                 {"token_limit_parameter": "max_tokens"}])
def test_unsafe_config_fails_closed(extra):
    with pytest.raises(budget.GuardError):
        budget.validate_config({"provider": "openai", "model": "gpt-5.6-luna", "max_output_tokens": 32, **extra})


def test_jev_full_context_and_openai_byte_bound():
    c = {"provider": "jev", "model": "jev-1.13.0"}
    assert budget.reservation(c, budget.PRICES[c["model"]], "short")[0] == budget.usd_nano("0.002688")
    c = {"provider": "openai", "model": "gpt-5.6-luna", "max_output_tokens": 32}
    _, detail = budget.reservation(c, budget.PRICES[c["model"]], "é")
    assert detail["input_token_bound"] == 2050
    with pytest.raises(budget.GuardError, match="272k"):
        budget.reservation(c, budget.PRICES[c["model"]], "x"*272000)


def test_stale_or_unverified_pricing_rejected():
    c = {"provider": "openai", "model": "gpt-5.6-luna", "max_output_tokens": 32}
    declared = {c["model"]: {**budget.PRICES[c["model"]], "verified_on": "2000-01-01"}}
    with pytest.raises(budget.GuardError, match="today"):
        budget.select_price(c, declared)


def test_separate_processes_share_one_reservation_cap(tmp_path):
    path = tmp_path/"ledger.jsonl"
    budget.Ledger(path, "0.0000007", initialize=True)
    code = '''import runpy,sys
api=runpy.run_path(sys.argv[1])
ledger=api['Ledger'](sys.argv[2], '0.0000007')
count=0
for _ in range(10):
 try:
  ledger.reserve(100,{})
  count+=1
 except api['BudgetStop']:
  pass
print(count)
'''
    processes = [subprocess.Popen([sys.executable, "-c", code, str(SPEC.origin), str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    outputs = [process.communicate(timeout=20) for process in processes]
    assert all(process.returncode == 0 for process in processes), outputs
    assert sum(int(stdout) for stdout, _ in outputs) == 7
    assert budget.Ledger(path, "0.0000007").snapshot()["charged_or_reserved_usd"] == "0.000000700"


def test_getpass_will_not_fallback_to_echoed_input(monkeypatch):
    tty = MagicMock()
    tty.__enter__.return_value.isatty.return_value = True
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def unsafe_getpass(*args, **kwargs):
        warnings.warn("Cannot disable echo", budget.getpass.GetPassWarning)
        raise AssertionError("The echo fallback must never be reached")

    with patch("builtins.open", return_value=tty), patch.object(budget.getpass, "getpass", unsafe_getpass):
        with pytest.raises(budget.getpass.GetPassWarning):
            budget.prompt_credentials([{"api_key_env": "OPENAI_API_KEY"}])
    assert "OPENAI_API_KEY" not in budget.os.environ


def test_unpriced_model_variant_cannot_receive_settlement(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    wrapped, ledger = _openai(tmp_path)
    response = _response()
    response["model"] = "gpt-5.6-luna-unpriced-variant"
    with patch.object(providers, "_post_json", return_value=response):
        wrapped.predict(Row("id", "unused", 0), ["a", "b"], "prompt")
    assert ledger.snapshot()["settlements"] == 0


@pytest.mark.skipif(sys.platform == "win32", reason="Driver requires a POSIX controlling terminal")
def test_real_controlling_pty_key_entry_is_hidden():
    import pty
    import termios

    secret = b"DUMMY_TEST_KEY_7f1bd9"
    code = "import runpy,os,sys; api=runpy.run_path(sys.argv[1]); api['prompt_credentials']([{'api_key_env':'JEVBENCH_PTY_TEST_KEY'}]); print('KEY_LENGTH='+str(len(os.environ['JEVBENCH_PTY_TEST_KEY'])))"
    pid, terminal = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, "-c", code, str(SPEC.origin)])
    transcript = bytearray()
    sent, reaped, status = False, False, None
    deadline = time.monotonic()+10
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([terminal], [], [], 0.1)
            if ready:
                try:
                    data = os.read(terminal, 4096)
                except OSError:
                    data = b""
                transcript.extend(data)
                if not sent and b"(hidden; memory only): " in transcript:
                    # Check the real kernel terminal setting before sending the
                    # fake key. Merely suppressing Python output is insufficient.
                    assert not termios.tcgetattr(terminal)[3] & termios.ECHO
                    os.write(terminal, secret+b"\n")
                    sent = True
            ended, status = os.waitpid(pid, os.WNOHANG)
            if ended:
                reaped = True
                break
        assert reaped, "PTY child did not finish"
        if b"Operation not permitted: '/dev/tty'" in transcript:
            pytest.skip("Execution sandbox denies /dev/tty; run with local terminal access for the PTY integration check")
        assert os.waitstatus_to_exitcode(status) == 0, transcript.decode(errors="replace")
        assert sent
        assert secret not in transcript
        assert f"KEY_LENGTH={len(secret)}".encode() in transcript
    finally:
        os.close(terminal)
        if not reaped:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
