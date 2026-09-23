"""Synthetic money guards only: never read credentials or dispatch HTTP."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_failed_retries as retry
from jevbench.types import Prediction


def item(fid, provider="jev", kind="paid_failure", depends=None):
    return {"failure_id": fid, "run_id": "run-" + fid, "row_id": fid, "provider": provider,
            "model": retry.route.MODEL if provider == "jev" else "gpt-6-astra", "kind": kind,
            "depends_on_failure_id": depends}


def builder(plan, entry, phase, index, source=None):
    config = {"provider": entry["provider"], "model": entry["model"], "api_key_env": "UNUSED_TEST_KEY",
              "max_output_tokens": 1024}
    request = {"config": config, "row_id": entry["row_id"], "prompt": "fixed synthetic prompt", "labels": ["A", "B"],
               "prompt_sha256": "a" * 64, "endpoint": retry.route.ENDPOINT, "request_body": {"synthetic": True}}
    identity = {"condition_id": entry["run_id"], "row_id": entry["row_id"], "prompt_sha256": "a" * 64,
                "choices_sha256": "b" * 64, "source_sha256": "c" * 64, "original_prediction_sha256": "d" * 64}
    meta = {"failure_id": entry["failure_id"], "phase": phase, "request_sha256": "e" * 64,
            "dependency": retry.analysis.digest(source) if source else None}
    return {"item": entry, "request": request, "attempt_index": index, "identity": identity,
            "stage": "dependent_first_call" if source is not None and index == 1 else "retry", "metadata": meta}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    plan = {"inventory": {"failures": [item("a"), item("b")]}, "allocation_usd": "0.030000000"}
    price = json.loads(retry.PRICES.read_text())["gpt-6-astra"]
    price["verification_kind"] = "operator_declared"
    monkeypatch.setattr(retry, "build_action", builder)
    monkeypatch.setattr(retry, "price_for", lambda request: retry.route.PRICE if request["config"]["provider"] == "jev" else price)
    monkeypatch.setattr(retry, "request_amount", lambda request: retry.route.RESERVE_NANO if request["config"]["provider"] == "jev" else 20_000_000)
    monkeypatch.setattr(retry.budget, "validate_config", lambda config: config)
    inner = retry.budget.Ledger(tmp_path / "budget.jsonl", plan["allocation_usd"], initialize=True)
    ledger = retry.RetryLedger(inner, plan, lambda: None)
    path = tmp_path / "attempts.jsonl"
    retry.audit_saved(plan, ledger, path)
    return plan, ledger, path


def record_for(action, error=None, label=0):
    pred = asdict(Prediction(action["item"]["row_id"], None if error else label,
                None if error or action["item"]["provider"] == "openai" else ([.9, .1] if label == 0 else [.1, .9]),
                input_tokens=None if error else 100, output_tokens=None if error else 1, error=error))
    pred["metadata"] = {"retry": action["metadata"], "resolved_model": None if error else action["item"]["model"]}
    return {"attempt_index": action["attempt_index"], "identity": action["identity"], "stage": action["stage"], "prediction": pred}


def dispatch(plan, ledger, path, *, error=None, label=0, save=True):
    action, _ = retry.walk_policy(plan, ledger.saved)
    ledger.active = action
    ident = ledger.reserve(retry.request_amount(action["request"]), retry.expected_reservation(action, plan))
    record = record_for(action, error, label)
    pred = record["prediction"]
    details = {"outcome": "prediction_error" if error else "returned",
               "usage": {"input_tokens": pred["input_tokens"], "output_tokens": pred["output_tokens"]},
               "resolved_model": pred["metadata"]["resolved_model"], "usage_exceeded_reservation_assumptions": False}
    pred["metadata"]["budget"] = {"ledger_id": ledger.identity["ledger_id"], "reservation_id": ident,
                "reserved_upper_bound_usd": retry.budget.usd_string(retry.request_amount(action["request"]))}
    if action["item"]["provider"] == "jev":
        extra = {"id": None if error else "synthetic-id", "provider": None if error else "TypeSafe",
                 "reported_cost_usd": None if error else "0.000001"}
        pred["metadata"]["openrouter"] = extra
        pred["metadata"]["budget"]["reservation_released"] = False
        details.update(route="OpenRouter", provider=extra["provider"], request_id=extra["id"],
                       reported_cost_usd=extra["reported_cost_usd"], reservation_released=False)
    ledger.result(ident, details)
    expected = retry.settlement(details, action)
    if expected is not None:
        ledger.settle(ident, *expected)
        pred["metadata"]["budget"]["settled_conservative_usd"] = retry.budget.usd_string(expected[0])
    if save:
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n"); stream.flush(); os.fsync(stream.fileno())
        retry.audit_saved(plan, ledger, path)
    return record


def test_full_jev_reservations_are_retained_even_on_success(setup):
    plan, ledger, path = setup
    dispatch(plan, ledger, path)
    dispatch(plan, ledger, path)
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.005376000"
    assert ledger.snapshot()["settlements"] == 0
    assert retry.walk_policy(plan, ledger.saved)[0] is None


def test_failure_only_two_rounds_stop_on_any_valid_success():
    plan = {"inventory": {"failures": [item("a"), item("b"), item("dependent", kind="upstream_skip", depends="a")]}}
    records = []
    expected = [("a", "paid_round_1"), ("b", "paid_round_1"), ("a", "paid_round_2"),
                ("dependent", "dependent_round_1"), ("dependent", "dependent_round_2")]
    for index, pair in enumerate(expected):
        action, _ = retry.walk_policy(plan, records, builder)
        assert (action["item"]["failure_id"], action["metadata"]["phase"]) == pair
        record = record_for(action, error="network_error: test" if index in [0, 3] else None, label=0)
        records.append(record)
        if index == 3:
            assert action["metadata"]["dependency"] == retry.analysis.digest(records[2]["prediction"])
    assert retry.walk_policy(plan, records, builder)[0] is None
    with pytest.raises(retry.budget.GuardError, match="exceed"):
        retry.walk_policy(plan, records + [records[-1]], builder)


def test_unrecovered_upstream_never_dispatches_dependent():
    plan = {"inventory": {"failures": [item("a"), item("dependent", kind="upstream_skip", depends="a")]}}
    records = []
    for _ in range(2):
        action, _ = retry.walk_policy(plan, records, builder)
        assert action["item"]["failure_id"] == "a"
        records.append(record_for(action, error="network_error: test"))
    assert retry.walk_policy(plan, records, builder)[0] is None
    assert retry.final_counts(plan, records)["upstream_unresolved_not_called"] == 1


def test_wrong_but_valid_original_retry_is_not_repeated(setup):
    plan, ledger, path = setup
    dispatch(plan, ledger, path, label=1)
    action, _ = retry.walk_policy(plan, ledger.saved)
    assert action["item"]["failure_id"] == "b"


def test_pending_request_blocks_next_reserve_and_resume(setup):
    plan, ledger, path = setup
    dispatch(plan, ledger, path, save=False)
    with pytest.raises(retry.budget.GuardError, match="durable"):
        ledger.reserve(1, {})
    with pytest.raises(retry.budget.GuardError, match="Orphan"):
        retry.audit_saved(plan, ledger, path)
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.002688000"


def test_unknown_or_duplicate_result_cannot_be_recorded(setup):
    _, ledger, _ = setup
    with pytest.raises(retry.budget.GuardError, match="unrelated"):
        ledger.result("unknown", {})


def test_partial_attempt_append_is_not_silently_discarded(setup):
    plan, ledger, path = setup
    path.write_text('{"attempt_index":')
    with pytest.raises(retry.budget.GuardError, match="Partial attempt"):
        retry.audit_saved(plan, ledger, path)
    assert path.read_text() == '{"attempt_index":'


def test_jev_settlement_is_forbidden(setup):
    plan, ledger, path = setup
    record = dispatch(plan, ledger, path, save=False)
    with pytest.raises(retry.budget.GuardError, match="Jev reservations stay retained"):
        ledger.settle(record["prediction"]["metadata"]["budget"]["reservation_id"], 1, {})
    assert ledger.snapshot()["settlements"] == 0


def test_only_exact_complete_new_openai_usage_can_settle(setup):
    plan, ledger, path = setup
    plan["inventory"]["failures"] = [item("astra", provider="openai")]
    dispatch(plan, ledger, path)
    assert ledger.snapshot()["settlements"] == 1
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.001300000"
    assert retry.walk_policy(plan, ledger.saved)[0] is None


@pytest.mark.parametrize("change", ["failed", "missing_usage", "wrong_model", "overrun"])
def test_untrusted_openai_usage_cannot_release_money(setup, change):
    plan, ledger, _ = setup
    entry = item("astra", provider="openai")
    action = builder(plan, entry, "paid_round_1", 1)
    details = {"outcome": "returned", "usage": {"input_tokens": 100, "output_tokens": 1},
               "resolved_model": "gpt-6-astra", "usage_exceeded_reservation_assumptions": False}
    if change == "failed": details["outcome"] = "prediction_error"
    if change == "missing_usage": details["usage"]["output_tokens"] = None
    if change == "wrong_model": details["resolved_model"] = "other-model"
    if change == "overrun": details["usage_exceeded_reservation_assumptions"] = True
    assert retry.settlement(details, action) is None


def test_tampered_request_is_rejected_before_reservation(setup):
    plan, ledger, _ = setup
    ledger.active, _ = retry.walk_policy(plan, [])
    details = retry.expected_reservation(ledger.active, plan)
    details["prompt_sha256"] = "f" * 64
    with pytest.raises(retry.budget.GuardError, match="Request differs"):
        ledger.reserve(retry.route.RESERVE_NANO, details)
    assert ledger.snapshot()["reservations"] == 0


def test_each_http_reservation_rechecks_prior_evidence(setup):
    plan, ledger, _ = setup
    ledger.active, _ = retry.walk_policy(plan, [])
    ledger.verify = lambda: retry.require(False, "prior changed")
    with pytest.raises(retry.budget.GuardError, match="prior changed"):
        ledger.reserve(retry.route.RESERVE_NANO, retry.expected_reservation(ledger.active, plan))
    assert ledger.snapshot()["reservations"] == 0


def test_saved_billing_error_requires_explicit_resume_independent_of_status(setup):
    plan, ledger, path = setup
    dispatch(plan, ledger, path, error="http_error: status=402; no automatic retry")
    assert retry.halt_reason(ledger.saved) == "halted_billing"
    ledger.active, _ = retry.walk_policy(plan, ledger.saved)
    with pytest.raises(retry.budget.GuardError, match="service-error tail"):
        ledger.reserve(retry.route.RESERVE_NANO, retry.expected_reservation(ledger.active, plan))
    ledger.allow_halt_tail_once = True
    dispatch(plan, ledger, path)
    assert not ledger.allow_halt_tail_once


def test_three_failure_tail_is_derived_from_attempts():
    records = [{"prediction": {"error": "network_error: test"}} for _ in range(3)]
    assert retry.halt_reason(records) == "halted_errors"
    records[-1]["prediction"]["error"] = None
    assert retry.halt_reason(records) is None


def test_missing_known_health_ledger_cannot_release_its_allowance(tmp_path, monkeypatch):
    monkeypatch.setattr(retry, "HEALTH", tmp_path / "absent-budget.jsonl")
    monkeypatch.setattr(retry, "CONTROL_AREA", tmp_path / "controls")
    with pytest.raises(retry.budget.GuardError, match="health ledger"):
        retry.prior_state()


def test_dry_run_never_reads_credentials_or_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(retry, "AREA", tmp_path / "unused")
    monkeypatch.setattr(retry, "prior_state", lambda: {"ready": False, "numeric_complete": 67,
                                                     "text_complete": 44, "prior_conservative_usd": "22.00"})
    monkeypatch.setattr(retry.budget, "prompt_credentials", lambda *_: pytest.fail("read credentials"))
    monkeypatch.setattr(retry.funding, "check_credit", lambda *_: pytest.fail("made network request"))
    monkeypatch.setattr(retry, "make_plan", lambda *_: pytest.fail("froze incomplete plan"))
    assert retry.main([])["ready"] is False
    assert not list(tmp_path.iterdir())


def test_mutation_flags_need_explicit_execution():
    for flag in ("--init-ledger", "--prompt-api-key", "--resume-after-errors"):
        with pytest.raises(retry.budget.GuardError, match="require --execute"):
            retry.main([flag])


@pytest.mark.parametrize("remaining,expected", [("1.00", "0.050752000"), ("0.01", "0.010000000")])
def test_allocation_is_minimum_of_remaining_and_two_full_rounds(setup, monkeypatch, remaining, expected):
    _, _, _ = setup
    entries = [item("jev"), item("astra", provider="openai"), item("dependent", kind="upstream_skip", depends="astra")]
    for entry in entries:
        entry["request_recipe"] = "standard_prompt"
    inventory = {"failures": entries, "pending_first_pass": {"numeric_pending_rows": 0, "text_pending_rows": 0}}
    monkeypatch.setattr(retry.preparation, "build_inventory", lambda: inventory)
    monkeypatch.setattr(retry.preparation, "reconstruct_request", lambda entry: builder({}, entry, "paid_round_1", 1)["request"])
    monkeypatch.setattr(retry, "pins", lambda: {"synthetic": "a" * 64})
    result = retry.make_plan({"ready": True, "remaining_usd": remaining})
    assert result["worst_case_usd"] == "0.050752000"
    assert result["allocation_usd"] == expected


def test_finite_policy_does_not_mark_budget_exhaustion_complete(setup, tmp_path):
    plan, _, path = setup
    plan["allocation_usd"] = "0.003000000"
    ledger = retry.RetryLedger(retry.budget.Ledger(tmp_path / "small.jsonl", "0.003", initialize=True), plan, lambda: None)
    retry.audit_saved(plan, ledger, path)
    dispatch(plan, ledger, path)
    ledger.active, _ = retry.walk_policy(plan, ledger.saved)
    with pytest.raises(retry.budget.BudgetStop):
        ledger.reserve(retry.route.RESERVE_NANO, retry.expected_reservation(ledger.active, plan))
    assert ledger.snapshot()["reservations"] == 1 and ledger.pending is None
    assert retry.walk_policy(plan, ledger.saved)[0] is not None


def test_final_audit_requires_complete_and_preserves_raw_files(setup, monkeypatch):
    plan, ledger, path = setup
    dispatch(plan, ledger, path)
    dispatch(plan, ledger, path)
    area = path.parent
    monkeypatch.setattr(retry, "AREA", area)
    monkeypatch.setattr(retry, "check_plan", lambda plan, read_only=False: None)
    (area / "plan.json").write_text(json.dumps(plan))
    record = {"status": "complete", "plan_sha256": retry.file_sha(area / "plan.json"),
        "ledger_id": ledger.identity["ledger_id"], "counts": retry.final_counts(plan, ledger.saved), "budget": ledger.snapshot(),
        "protected_artifacts_sha256": {name: retry.file_sha(area / name) for name in
            ("plan.json", "attempts.jsonl", "budget.jsonl", "budget.jsonl.lock")}}
    (area / "run.json").write_text(json.dumps(record))
    before = {p.name: p.read_bytes() for p in area.iterdir()}
    audited = retry.audit_run()
    assert audited["record"]["counts"]["recovered"] == 2
    assert all((area / name).read_bytes() == raw for name, raw in before.items())
    record["status"] = "interrupted"
    (area / "run.json").write_text(json.dumps(record))
    with pytest.raises(retry.budget.GuardError, match="not completed"):
        retry.audit_run()


@pytest.fixture
def pinned_prior(tmp_path, monkeypatch):
    root = tmp_path
    previous = root / "results/old-budget.jsonl"
    health = root / "results/health-budget.jsonl"
    old = retry.budget.Ledger(previous, "1.00", initialize=True)
    old.reserve(retry.budget.usd_nano("0.50"), {"synthetic": True})
    retry.budget.Ledger(health, "0.002688", initialize=True)
    monkeypatch.setattr(retry, "ROOT", root)
    monkeypatch.setattr(retry, "AREA", root / "results/retries")
    monkeypatch.setattr(retry, "HEALTH", health)
    monkeypatch.setattr(retry, "CONTROL_AREA", root / "results/controls")
    monkeypatch.setattr(retry, "canonical_prior_paths", lambda: {"results/old-budget.jsonl", "results/health-budget.jsonl"})
    monkeypatch.setattr(retry, "pins", lambda: {"synthetic": "a" * 64})
    evidence = {p.relative_to(root).as_posix(): retry.file_sha(p) for p in
                (previous, previous.with_name(previous.name + ".lock"), health, health.with_name(health.name + ".lock"))}
    entry = {**item("a"), "artifact_sha256": {}}
    inventory = {"failures": [entry]}
    plan = {"producer_pins": retry.pins(), "inventory": inventory, "inventory_sha256": retry.budget.sha(inventory),
            "prior": {"ready": True, "numeric_complete": 68, "text_complete": 68, "authorized_usd": "25.00",
                      "prior_conservative_usd": "0.502688000", "remaining_usd": "24.497312000", "files_sha256": evidence},
            "allocation_usd": "0.005376000", "worst_case_usd": "0.005376000",
            "request_specs": {"a": {"reservation_nano": retry.route.RESERVE_NANO}}}
    return plan, root


def test_check_plan_recomputes_previous_charge_plus_full_health_allowance(pinned_prior):
    plan, _ = pinned_prior
    retry.check_plan(plan)
    plan["prior"].update(prior_conservative_usd="0.500000000", remaining_usd="24.500000000")
    with pytest.raises(retry.budget.GuardError, match="reconstruct"):
        retry.check_plan(plan)


def test_live_dispatch_rejects_new_ledger_but_readonly_audit_allows_later_allocation(pinned_prior):
    plan, root = pinned_prior
    retry.budget.Ledger(root / "results/later-budget.jsonl", "0.01", initialize=True)
    with pytest.raises(retry.budget.GuardError, match="ledger inventory changed"):
        retry.check_plan(plan)
    retry.check_plan(plan, read_only=True)


def test_declared_allocation_cannot_exceed_frozen_worst_case(pinned_prior):
    plan, _ = pinned_prior
    plan["allocation_usd"] = "0.006000000"
    with pytest.raises(retry.budget.GuardError, match="fixed minimum"):
        retry.check_plan(plan)


def test_changed_original_budget_anchor_is_rejected(pinned_prior):
    plan, root = pinned_prior
    path = root / "results/old-budget.jsonl.lock"
    path.write_text(path.read_text() + " ")
    with pytest.raises(retry.budget.GuardError, match="previous budget changed"):
        retry.check_plan(plan)
