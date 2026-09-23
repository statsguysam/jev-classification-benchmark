"""No model or authenticated calls: scoped re-verification and evidence checks."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_jev_completion as completion

NOW = datetime(2026, 9, 23, 12, tzinfo=timezone.utc)


def receipt(at=NOW):
    return {"verified_at": at.isoformat(), "verified_on": "2026-09-23",
        "endpoint": completion.route.ENDPOINT, "input_usd_per_million": "0.042",
        "output_usd_per_million": "0", "reserved_input_tokens": 64000,
        "retained_reservation_usd": "0.002688000",
        "historical_route_price_declaration": deepcopy(completion.route.PRICE),
        "model_catalog": {"name": "TypeSafe | typesafe/jev-1.13-20260917",
            "model_id": "typesafe/jev-1.13", "provider_name": "TypeSafe", "context_length": 32000,
            "status": 0, "pricing": {"prompt": "0.000000042", "completion": "0", "discount": 0}},
        "sources": {u: "a" * 64 for u in (completion.probe.CATALOG_URL,
            completion.probe.ROUTE_DOCS, completion.probe.MODEL_DOCS)}}


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setattr(completion, "utc_now", lambda: NOW)
    monkeypatch.setattr(completion, "verify_sources", lambda *args: None)
    target = SimpleNamespace(__file__=completion.historical.numeric.__file__)
    return completion.VerifiedTransport(target, completion.sha(target.__file__), execute=True)


def test_fresh_evidence_and_unchanged_old_price_are_required():
    assert completion.validate_verification(receipt(), NOW)["verified_on"] == "2026-09-23"
    assert completion.route.PRICE["verified_on"] == "2026-09-22"
    with pytest.raises(completion.budget.GuardError, match="stale"):
        completion.validate_verification(receipt(NOW-timedelta(seconds=3601)), NOW)
    with pytest.raises(completion.budget.GuardError, match="UTC date"):
        completion.validate_verification(receipt(), NOW+timedelta(days=1))


@pytest.mark.parametrize("field,value", [("pricing", {"prompt": "0.000000043", "completion": "0"}),
    ("name", "TypeSafe | typesafe/jev-1.14"), ("provider_name", "Other"), ("context_length", 64000)])
def test_changed_route_price_or_snapshot_fails(field, value):
    value_receipt = receipt()
    value_receipt["model_catalog"][field] = value
    with pytest.raises(completion.budget.GuardError):
        completion.validate_verification(value_receipt, NOW)


def test_refreshes_before_next_request_not_on_every_request(transport, monkeypatch):
    calls = []
    monkeypatch.setattr(completion.probe, "verify_public", lambda: calls.append(1) or receipt(completion.utc_now()))
    transport.check()
    transport.check()
    assert len(calls) == 1
    monkeypatch.setattr(completion, "utc_now", lambda: NOW+timedelta(seconds=3601))
    transport.check()
    assert len(calls) == 2
    monkeypatch.setattr(completion, "utc_now", lambda: NOW+timedelta(days=1))
    with pytest.raises(completion.budget.GuardError, match="date changed"):
        transport.check()
    assert len(calls) == 2


def test_failed_refresh_cannot_reserve_or_clear_old_evidence(transport, monkeypatch):
    transport.receipts = [receipt(NOW-timedelta(hours=2))]
    def fail(): raise completion.budget.GuardError("catalog changed")
    monkeypatch.setattr(completion.probe, "verify_public", fail)
    with pytest.raises(completion.budget.GuardError, match="catalog changed"):
        transport.check()
    assert transport.receipts == [receipt(NOW-timedelta(hours=2))]


def test_source_hash_check_is_kept_on_every_request(transport, monkeypatch):
    calls = []
    monkeypatch.setattr(completion, "verify_sources", lambda *args: calls.append(args))
    transport.receipts = [receipt()]
    transport.check(); transport.check()
    assert len(calls) == 2


@pytest.mark.parametrize("arguments", [["--exe"], ["--execute=true"], ["--prompt-api-key"],
    ["--freeze-sources"], ["--init-ledger"], ["--execute", "sk-not-a-real-key"]])
def test_no_abbreviated_execution_or_dry_write_or_key_arguments(arguments):
    with pytest.raises(completion.budget.GuardError):
        completion.validate_arguments("numeric", arguments)


def test_dry_run_is_offline_read_only_and_restores_original_verifier(monkeypatch):
    previous = completion.base.verify_transport
    calls = []
    def main(args):
        assert completion.base.verify_transport is not previous
        completion.base.verify_transport()
        calls.append(args)
        return "planned"
    target = SimpleNamespace(__file__=completion.historical.numeric.__file__, main=main)
    monkeypatch.setattr(completion.importlib, "import_module", lambda name: target)
    monkeypatch.setattr(completion, "verify_sources", lambda *args: None)
    monkeypatch.setattr(completion.probe, "verify_public", lambda: pytest.fail("unexpected public GET"))
    monkeypatch.setattr(completion, "execute_session", lambda *args: pytest.fail("unexpected write/credential execution"))
    assert completion.main(["numeric", "--datasets", "wine", "--shots", "0"]) == "planned"
    assert calls == [["--datasets", "wine", "--shots", "0"]]
    assert completion.base.verify_transport is previous


def test_original_source_checks_reject_selected_code_change(monkeypatch):
    monkeypatch.setattr(completion.historical, "verify_frozen_helpers", lambda: None)
    target = SimpleNamespace(__file__=completion.historical.numeric.__file__)
    with pytest.raises(completion.budget.GuardError, match="Selected producer changed"):
        completion.verify_sources(target, "0" * 64, completion.sha(completion.__file__))


def snap(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def test_resume_allows_only_append_prefix_and_declared_mode(tmp_path):
    name = "results/numeric_expansion/review/a/predictions.jsonl"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    original = b'{"error":"old failure"}\n'
    appended = original+b'{"label":0}\n'
    path.write_bytes(appended)
    before, after = {name: snap(original)}, {name: snap(appended)}
    assert completion.audit_preservation(before, after, "numeric", tmp_path) == [name]
    with pytest.raises(completion.budget.GuardError, match="Protected evidence"):
        completion.audit_preservation(before, after, "text", tmp_path)
    path.write_bytes(b'{"label":1}\n'+appended)
    with pytest.raises(completion.budget.GuardError, match="prefix changed"):
        completion.audit_preservation(before, {name: snap(path.read_bytes())}, "numeric", tmp_path)


def test_original_completed_sources_cannot_change(tmp_path):
    name = "results/numeric_expansion/local/a/run.json"
    with pytest.raises(completion.budget.GuardError, match="Protected evidence"):
        completion.audit_preservation({name:snap(b"original")}, {name:snap(b"changed")}, "numeric", tmp_path)


def test_execution_receipt_retains_failures_and_cap_without_exposing_credentials(tmp_path, monkeypatch, transport):
    area = tmp_path / "sessions"
    monkeypatch.setattr(completion, "AREA", area)
    monkeypatch.setattr(completion, "snapshot", lambda: {})
    transport.receipts = [receipt()]
    original_cap = completion.historical.numeric.STAGE_CAP
    original_price = deepcopy(completion.route.PRICE)
    def run(args):
        assert args == ["--execute", "--stop-after-new-requests", "1"]
        assert completion.historical.numeric.STAGE_CAP == original_cap
        assert completion.route.PRICE == original_price
        return "checkpoint"
    target = SimpleNamespace(__file__=completion.historical.numeric.__file__, main=run)
    assert completion.execute_session("numeric", ["--execute", "--stop-after-new-requests", "1"], target, transport) == "checkpoint"
    sessions = [p for p in area.iterdir() if p.is_dir()]
    assert len(sessions) == 1
    start = json.loads((sessions[0]/"start.json").read_text())
    final = json.loads((sessions[0]/"completion.json").read_text())
    assert start["automatic_retries_added"] == 0 and start["original_failed_predictions_retained"] is True
    assert final["historical_evidence_preserved"] is True
    assert len(final["public_verification_sha256"]) == 1
    assert "total_credits" not in json.dumps(start) and "total_usage" not in json.dumps(final)


def test_failed_original_execution_still_writes_receipt(tmp_path, monkeypatch, transport):
    monkeypatch.setattr(completion, "AREA", tmp_path/"sessions")
    monkeypatch.setattr(completion, "snapshot", lambda: {})
    transport.receipts = [receipt()]
    def run(args): raise completion.budget.GuardError("original halt")
    target = SimpleNamespace(__file__=completion.historical.numeric.__file__, main=run)
    with pytest.raises(completion.budget.GuardError, match="original halt"):
        completion.execute_session("numeric", ["--execute"], target, transport)
    final_path = next((tmp_path/"sessions").glob("*/completion.json"))
    final = json.loads(final_path.read_text())
    assert final["status"] == "raised" and final["exception_type"] == "GuardError"
