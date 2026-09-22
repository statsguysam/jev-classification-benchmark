"""Money, request-identity and restart invariants without network/model calls."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_review_controls as controls
from jevbench.types import Prediction


def request(index=0, arm="actual"):
    return {"request_id": f"request-{index}", "case_id": "case-1", "dataset": "fixture",
            "row_id": "test-row", "arm": arm, "proposal_value": 0,
            "donor_row_id": "test-row", "execution_index": index,
            "prompt": "fixture", "prompt_sha256": controls.hashlib.sha256(b"fixture").hexdigest(),
            "choices": ["negative", "positive"]}


def setup(tmp_path, requests=None, cap="0.01"):
    plan = {"requests": requests or [request()]}
    inner = controls.budget.Ledger(tmp_path / "budget.jsonl", cap, initialize=True)
    identity = {"ledger_id": inner.identity["ledger_id"], "producer_pins": {"runner": "sha"},
                "protocol_sha256": "protocol-sha", "prior_snapshot": {"allocation_usd": cap}}
    ledger = controls.ControlLedger(inner, plan, identity, verify=lambda: None)
    ledger.remember_head()
    return plan, identity, ledger


def save_response(tmp_path, plan, identity, ledger, *, error=None, unknown=False, append=True):
    item = plan["requests"][0]
    ledger.active_request = item
    ident = ledger.reserve(controls.route.RESERVE_NANO, controls.expected_reservation(item, identity))
    details = {"outcome": "prediction_error" if error else "returned",
        "usage": {"input_tokens": None if unknown else 100, "output_tokens": None if unknown else 0},
        "route": "OpenRouter", "request_id": None if error else "request-native-1",
        "provider": None if error else "TypeSafe", "resolved_model": None if error else controls.route.MODEL,
        "reported_cost_usd": None if unknown or error else "0.0000042", "reservation_released": False,
        "usage_exceeded_reservation_assumptions": False,
        "guard_reasons": {"reported_usage_or_cost_overrun": False, "unexpected_route_or_snapshot": False,
                          "invalid_reported_cost": False}}
    ledger.result(ident, details)
    prediction = Prediction(item["row_id"], None if error else 0,
        probabilities=None if error else [0.8, 0.2], error=error,
        input_tokens=details["usage"]["input_tokens"], output_tokens=details["usage"]["output_tokens"],
        metadata={"resolved_model": details["resolved_model"], "probability_kind": "jev_choice_distribution",
            "openrouter": {"id": details["request_id"], "provider": details["provider"],
                "reported_cost_usd": details["reported_cost_usd"], "endpoint": controls.route.ENDPOINT},
            "budget": {"ledger_id": ledger.identity["ledger_id"], "reservation_id": ident,
                "reserved_upper_bound_usd": controls.route.PRICE["per_request_reserved_usd"], "reservation_released": False},
            "control": {key: item[key] for key in ("request_id", "case_id", "dataset", "arm", "proposal_value",
                "donor_row_id", "execution_index", "prompt_sha256")}})
    controls.annotate_settlement(prediction, ledger)
    if append:
        (tmp_path / "predictions.jsonl").write_text(json.dumps(asdict(prediction)) + "\n")
        ledger.prediction_saved(prediction)
    return prediction


def test_verified_success_settles_and_reaudits(tmp_path):
    plan, identity, ledger = setup(tmp_path)
    saved = save_response(tmp_path, plan, identity, ledger)
    predictions = controls.audit_saved(plan, identity, ledger, tmp_path / "predictions.jsonl")
    assert predictions == [saved]
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.000005250"
    assert ledger.pending_id is None


@pytest.mark.parametrize("error,unknown", [("http_error: status=402; no automatic retry", True), (None, True)])
def test_failed_or_unknown_usage_retains_reservation(tmp_path, error, unknown):
    plan, identity, ledger = setup(tmp_path)
    save_response(tmp_path, plan, identity, ledger, error=error, unknown=unknown)
    controls.audit_saved(plan, identity, ledger, tmp_path / "predictions.jsonl")
    assert ledger.snapshot()["settlements"] == 0
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.002688000"


def test_response_without_durable_prediction_is_not_retried(tmp_path):
    plan, identity, ledger = setup(tmp_path)
    save_response(tmp_path, plan, identity, ledger, append=False)
    with pytest.raises(controls.budget.GuardError, match="Orphan"):
        controls.audit_saved(plan, identity, ledger, tmp_path / "predictions.jsonl")
    with pytest.raises(controls.budget.GuardError, match="unresolved"):
        ledger.reserve(controls.route.RESERVE_NANO, controls.expected_reservation(plan["requests"][0], identity))


@pytest.mark.parametrize("field,value", [("arm", "shuffled"), ("proposal_value", 1), ("row_id", "another-row"),
                                       ("execution_index", 1), ("prompt_sha256", "changed")])
def test_saved_control_identity_tamper_is_rejected(tmp_path, field, value):
    plan, identity, ledger = setup(tmp_path)
    saved = save_response(tmp_path, plan, identity, ledger)
    if field == "row_id": saved.row_id = value
    else: saved.metadata["control"][field] = value
    (tmp_path / "predictions.jsonl").write_text(json.dumps(asdict(saved)) + "\n")
    with pytest.raises(controls.budget.GuardError):
        controls.audit_saved(plan, identity, ledger, tmp_path / "predictions.jsonl")


def test_duplicate_and_out_of_order_requests_fail_before_reserve(tmp_path):
    plan, identity, ledger = setup(tmp_path, [request(), request(1, "no_proposal")])
    ledger.active_request = plan["requests"][1]
    with pytest.raises(controls.budget.GuardError, match="out-of-order"):
        ledger.reserve(controls.route.RESERVE_NANO, controls.expected_reservation(ledger.active_request, identity))
    assert ledger.snapshot()["reservations"] == 0
    save_response(tmp_path, plan, identity, ledger)
    ledger.active_request = plan["requests"][0]
    with pytest.raises(controls.budget.GuardError, match="out-of-order"):
        ledger.reserve(controls.route.RESERVE_NANO, controls.expected_reservation(ledger.active_request, identity))
    assert ledger.snapshot()["reservations"] == 1


def test_incomplete_prior_studies_stop_before_credentials_or_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(controls, "OUTPUT", tmp_path / "never-created")
    monkeypatch.setattr(controls.budget, "prompt_credentials", lambda *_: pytest.fail("Credential read before readiness"))
    with pytest.raises(controls.budget.GuardError, match="Incomplete"):
        controls.execute({}, {"ready_for_new_controls": False, "reason": "Incomplete"}, prompt_key=True, init_ledger=True)
    assert not controls.OUTPUT.exists()


def test_budget_stop_has_no_second_reservation(tmp_path):
    plan, identity, ledger = setup(tmp_path, [request(), request(1)], cap="0.004")
    save_response(tmp_path, plan, identity, ledger, unknown=True)
    ledger.active_request = plan["requests"][1]
    with pytest.raises(controls.budget.BudgetStop):
        ledger.reserve(controls.route.RESERVE_NANO, controls.expected_reservation(ledger.active_request, identity))
    assert ledger.snapshot()["reservations"] == 1


def test_changed_prior_ledger_stops_before_next_call(tmp_path):
    plan, identity, ledger = setup(tmp_path)
    def changed(): raise controls.budget.GuardError("Earlier ledger changed")
    ledger.verify = changed
    ledger.active_request = plan["requests"][0]
    with pytest.raises(controls.budget.GuardError, match="Earlier ledger changed"):
        ledger.reserve(controls.route.RESERVE_NANO, controls.expected_reservation(ledger.active_request, identity))
    assert ledger.snapshot()["reservations"] == 0


def test_partial_line_is_preserved_and_rejected(tmp_path):
    plan, identity, ledger = setup(tmp_path)
    path = tmp_path / "predictions.jsonl"; path.write_text('{"row_id":')
    with pytest.raises(controls.budget.GuardError, match="Partial prediction"):
        controls.audit_saved(plan, identity, ledger, path)
    assert path.read_text() == '{"row_id":'


def test_failure_cannot_use_source_label():
    item = request()
    p = Prediction(item["row_id"], 0, error="failure", metadata={"control": {key: item[key] for key in
        ("request_id", "case_id", "dataset", "arm", "proposal_value", "donor_row_id", "execution_index", "prompt_sha256")}})
    with pytest.raises(controls.budget.GuardError, match="fallback"):
        controls.validate_prediction(p, item)

@pytest.mark.parametrize('field,value', [('latency_s', float('nan')), ('latency_s', -1), ('label', 1), ('probabilities', [float('inf'), 0])])
def test_invalid_numeric_response_is_rejected(tmp_path, field, value):
    plan, identity, ledger = setup(tmp_path)
    prediction = save_response(tmp_path, plan, identity, ledger)
    setattr(prediction, field, value)
    with pytest.raises(controls.budget.GuardError):
        controls.validate_prediction(prediction, plan['requests'][0])


def test_failed_response_cannot_keep_distribution(tmp_path):
    plan, identity, ledger = setup(tmp_path)
    prediction = save_response(tmp_path, plan, identity, ledger, error='fixture error')
    prediction.probabilities = [0.8, 0.2]
    with pytest.raises(controls.budget.GuardError, match='fallback'):
        controls.validate_prediction(prediction, plan['requests'][0])
