"""One-call/retained-cost and isolation checks; all transport is synthetic."""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probe_jev_health as probe
from jevbench.providers import ProviderError


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Network is prohibited in probe tests")
    monkeypatch.setattr("socket.socket.connect", fail)


def prepared(tmp_path):
    area = tmp_path / "probe"
    area.mkdir()
    verification = {"verified_at": probe.budget.utc_now()}
    receipt = {"checked_at": probe.budget.utc_now(), "positive_available_credit": True}
    probe.write_exclusive(area / "verification.json", verification)
    return area, verification, receipt


def wire_result():
    return {"model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe", "id": "probe-fixture-id",
            "answers": {"classification": {"choice": "0", "probabilities": {"0": 0.98, "1": 0.02}}},
            "usage": {"input_tokens": 300, "output_tokens": 30, "cost": "0.0000126"}}


@pytest.mark.parametrize("failure", [False, True])
def test_one_request_retains_full_reservation_and_cannot_retry(tmp_path, monkeypatch, failure):
    area, verification, receipt = prepared(tmp_path)
    calls = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-secret-never-persist")
    def post(url, body, headers, timeout):
        calls.append(body)
        assert url == probe.route.ENDPOINT and body["state"] == probe.PROMPT
        assert body["model"] == probe.CONFIG["model"]
        assert headers["Authorization"] == "Bearer synthetic-secret-never-persist"
        if failure:
            raise ProviderError("http_error: status=402; no automatic retry")
        return wire_result()
    monkeypatch.setattr(probe.route, "post_json_exact_cost", post)
    result = probe.perform_one(area, verification, receipt, {})
    assert len(calls) == 1
    assert result["status"] == ("failed" if failure else "responded_expected_choice")
    assert result["budget"]["charged_or_reserved_usd"] == probe.CAP
    assert result["budget"]["reservations"] == 1 and result["budget"]["settlements"] == 0
    assert all("synthetic-secret" not in path.read_text() for path in area.iterdir() if path.is_file())
    with pytest.raises(probe.budget.GuardError, match="never retry"):
        probe.perform_one(area, verification, receipt, {})
    assert len(calls) == 1


def test_raised_after_reservation_is_durable_and_not_retryable(tmp_path, monkeypatch):
    area, verification, receipt = prepared(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    def post(*args):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(probe.route, "post_json_exact_cost", post)
    with pytest.raises(RuntimeError):
        probe.perform_one(area, verification, receipt, {})
    record = json.loads((area / "run.json").read_text())
    assert record["status"] == "stopped_or_raised_no_retry"
    assert record["budget"]["charged_or_reserved_usd"] == probe.CAP
    assert not (area / "prediction.json").exists()
    with pytest.raises(probe.budget.GuardError, match="never retry"):
        probe.perform_one(area, verification, receipt, {})


def test_dry_run_does_not_access_network_credentials_or_write(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "verify_envelope", lambda root: {})
    monkeypatch.setattr(probe, "hidden_key", lambda: pytest.fail("Credentials accessed in dry run"))
    assert probe.main([])["combined_envelope_usd"] == "24.960515700"
    assert list(tmp_path.iterdir()) == []


def test_controls_prior_attempt_other_allocation_and_stale_receipt_rejected(tmp_path):
    controls = tmp_path / "results/review_controls/execution"
    controls.mkdir(parents=True)
    with pytest.raises(probe.budget.GuardError, match="Controls execution"):
        probe.ensure_unused(tmp_path)
    controls.rmdir()
    area = tmp_path / probe.RELATIVE_AREA
    area.mkdir(parents=True)
    (area / "verification.json").write_text("{}")
    with pytest.raises(probe.budget.GuardError, match="already has evidence"):
        probe.ensure_unused(tmp_path)
    with pytest.raises(probe.budget.GuardError, match="Fresh verification"):
        probe.fresh("2026-09-22T00:00:00+00:00")


def catalog():
    return {"data": {"id": probe.route.MODEL, "architecture": {"modality": "text->decisions"},
        "endpoints": [{"name": "TypeSafe | typesafe/jev-1.13-20260917", "model_id": probe.route.MODEL,
            "provider_name": "TypeSafe", "context_length": 32000, "status": 0,
            "pricing": {"prompt": "0.000000042", "completion": "0", "discount": 0}}]}}


def test_public_verification_keeps_both_fresh_and_historical_dates(monkeypatch):
    values = {probe.CATALOG_URL: json.dumps(catalog()).encode(),
              probe.ROUTE_DOCS: b"/api/v1/systemone typesafe/jev-1.13",
              probe.MODEL_DOCS: b"64k 32k jev-1.13.0"}
    monkeypatch.setattr(probe, "public_get", values.__getitem__)
    receipt = probe.verify_public()
    assert receipt["verified_on"] == "2026-09-23"
    assert receipt["historical_route_price_declaration"]["verified_on"] == "2026-09-22"
    assert set(receipt["sources"]) == set(values)


@pytest.mark.parametrize("field,value", [("name", "new-snapshot"), ("context_length", 64000),
                                         ("provider_name", "Other"), ("status", 1),
                                         ("pricing", {"prompt": "0.0001", "completion": "0"})])
def test_changed_catalog_fails_closed(field, value):
    payload = catalog()
    payload["data"]["endpoints"][0][field] = value
    with pytest.raises(probe.budget.GuardError):
        probe.validate_catalog(payload)


def test_historical_mutation_blocks_before_http(tmp_path, monkeypatch):
    area, verification, receipt = prepared(tmp_path)
    protected = tmp_path / "old-evidence"
    protected.write_text("original")
    pins = {str(protected): probe.sha(protected)}
    protected.write_text("changed")
    monkeypatch.setattr(probe.route, "post_json_exact_cost", lambda *args: pytest.fail("HTTP must not run"))
    with pytest.raises(probe.budget.GuardError, match="Historical evidence changed"):
        probe.perform_one(area, verification, receipt, pins)
    record = json.loads((area / "run.json").read_text())
    assert record["budget"]["reservations"] == 0


def test_real_frozen_envelope_and_hashes_are_valid(monkeypatch):
    # This offline test audits hashes/envelopes on future CI days as well.
    # The executable's fixed-date guard remains unchanged.
    monkeypatch.setattr(probe, "DATE", probe.datetime.now(probe.timezone.utc).date().isoformat())
    pins = probe.verify_envelope(probe.ROOT)
    assert len(pins) >= 15
    assert all(probe.sha(path) == value for path, value in pins.items())
