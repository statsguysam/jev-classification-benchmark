"""Synthetic post-control execution/accounting tests. No API calls."""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"), str(Path(__file__).resolve().parent)]
import run_control_failed_retries as post
from test_run_failed_retries import builder, item, record_for


def control_item(fid):
    return {**item(fid), "request_recipe": "control_frozen_request", "control_request_id": "control-" + fid,
            "control_arm": "actual", "dataset": "wine", "original_prediction_sha256": "f" * 64,
            "artifact_sha256": {}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    plan = {"inventory": {"failures": [control_item("a"), control_item("b")]}, "allocation_usd": "0.030000000"}
    monkeypatch.setattr(post.base, "build_action", builder)
    monkeypatch.setattr(post.base, "price_for", lambda request: post.route.PRICE)
    monkeypatch.setattr(post.base, "request_amount", lambda request: post.route.RESERVE_NANO)
    inner = post.budget.Ledger(tmp_path / "budget.jsonl", plan["allocation_usd"], initialize=True)
    ledger = post.ControlRetryLedger(inner, plan, lambda: None)
    path = tmp_path / "attempts.jsonl"
    post.audit_saved(plan, ledger, path)
    return plan, ledger, path


def dispatch(plan, ledger, path, *, error=None, label=0, durable=True):
    action, _ = post.base.walk_policy(plan, ledger.saved)
    ledger.active = action
    ident = ledger.reserve(post.route.RESERVE_NANO, post.base.expected_reservation(action, plan))
    record = record_for(action, error, label)
    pred = record["prediction"]
    wire = {"id": None if error else "synthetic-id", "provider": None if error else "TypeSafe",
            "reported_cost_usd": None if error else "0.000001"}
    pred["metadata"].update(openrouter=wire, post_control=post.control_metadata(action), budget={
        "ledger_id": ledger.identity["ledger_id"], "reservation_id": ident,
        "reserved_upper_bound_usd": "0.002688000", "reservation_released": False})
    ledger.result(ident, {"outcome": "prediction_error" if error else "returned",
        "usage": {"input_tokens": pred["input_tokens"], "output_tokens": pred["output_tokens"]},
        "resolved_model": pred["metadata"]["resolved_model"], "usage_exceeded_reservation_assumptions": False,
        "route": "OpenRouter", "request_id": wire["id"], "provider": wire["provider"],
        "reported_cost_usd": wire["reported_cost_usd"], "reservation_released": False})
    if durable:
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n"); stream.flush(); os.fsync(stream.fileno())
        post.audit_saved(plan, ledger, path)
    return record


def test_success_retains_full_reserve_and_is_not_repeated(setup):
    plan, ledger, path = setup
    dispatch(plan, ledger, path, label=1)
    dispatch(plan, ledger, path, label=0)
    assert post.base.walk_policy(plan, ledger.saved)[0] is None
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.005376000"
    assert ledger.snapshot()["settlements"] == 0


def test_control_recovery_allows_only_two_failed_rounds(setup):
    plan, ledger, path = setup
    dispatch(plan, ledger, path, error="network_error: test")
    dispatch(plan, ledger, path)
    dispatch(plan, ledger, path, error="network_error: test")
    assert post.base.walk_policy(plan, ledger.saved)[0] is None
    counts = post.base.final_counts(plan, ledger.saved)
    assert counts["new_calls"] == 3 and counts["unrecovered"] == 1
    assert ledger.snapshot()["charged_or_reserved_usd"] == "0.008064000"


def test_post_control_settlement_is_always_forbidden(setup):
    _, ledger, _ = setup
    with pytest.raises(post.budget.GuardError, match="remain retained"):
        ledger.settle("arbitrary", 0, {})


def test_orphan_control_retry_blocks_resume_without_another_call(setup):
    plan, ledger, path = setup
    dispatch(plan, ledger, path, durable=False)
    with pytest.raises(post.budget.GuardError, match="Orphan"):
        post.audit_saved(plan, ledger, path)
    assert ledger.snapshot()["reservations"] == 1
    with pytest.raises(post.budget.GuardError, match="durable"):
        ledger.reserve(1, {})


def test_control_arm_lineage_is_verified_against_frozen_failure_item(setup):
    plan, ledger, path = setup
    dispatch(plan, ledger, path)
    record = json.loads(path.read_text())
    record["prediction"]["metadata"]["post_control"]["arm"] = "shuffled"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(post.budget.GuardError, match="lineage"):
        post.audit_saved(plan, ledger, path)


def test_no_failure_noop_uses_no_credentials_network_or_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "AREA", tmp_path)
    monkeypatch.setattr(post, "check_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(post.funding, "check_credit", lambda *_: pytest.fail("Network call during no-op"))
    monkeypatch.setattr(post.budget, "prompt_credentials", lambda *_: pytest.fail("Credential prompt during no-op"))
    plan = {"study": "post-control-failure-recovery-v1", "inventory": {"failures": []}, "allocation_usd": "0.000000000"}
    record = post.execute(plan, initialize=True, prompt_key=True)
    assert record["status"] == "complete" and record["no_op"]
    assert record["budget"] == post.ZERO
    assert not (tmp_path / "budget.jsonl").exists() and not (tmp_path / "budget.jsonl.lock").exists()
    audited = post.audit_run()
    assert audited["attempts"] == [] and audited["record"]["counts"]["new_calls"] == 0
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert post.execute(plan, prompt_key=True) == record
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_noop_cannot_hide_paid_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "AREA", tmp_path)
    monkeypatch.setattr(post, "check_plan", lambda *args, **kwargs: None)
    plan = {"study": "post-control-failure-recovery-v1", "inventory": {"failures": []}, "allocation_usd": "0.000000000"}
    post.execute(plan)
    (tmp_path / "budget.jsonl").write_text("unexpected paid evidence")
    with pytest.raises(post.budget.GuardError, match="No-op unexpectedly"):
        post.audit_run()


@pytest.mark.parametrize("remaining,expected", [("1.00", "0.010752000"), ("0.006", "0.006000000")])
def test_plan_filters_historical_failures_and_caps_two_rounds(setup, monkeypatch, remaining, expected):
    _, _, _ = setup
    paid_old = {**item("old"), "request_recipe": "standard_prompt"}
    failures = [paid_old, control_item("a"), control_item("b")]
    monkeypatch.setattr(post, "snapshot_previous", lambda ready: {"remaining_usd": remaining})
    monkeypatch.setattr(post, "producer_pins", lambda: {"synthetic": "a" * 64})
    monkeypatch.setattr(post.preparation, "build_inventory", lambda: {"failures": failures})
    monkeypatch.setattr(post.preparation, "reconstruct_request", lambda entry: {
        **builder({}, entry, "paid_round_1", 1)["request"], "is_new_dependent_request": False})
    plan = post.make_plan({"ready": True, "failed_control_requests": 2})
    assert len(plan["inventory"]["failures"]) == 2
    assert plan["worst_case_usd"] == "0.010752000" and plan["allocation_usd"] == expected


def test_no_control_outcome_is_read_before_both_predecessors_complete(tmp_path, monkeypatch):
    hist, ctl = tmp_path / "hist", tmp_path / "ctl"
    hist.mkdir(); ctl.mkdir()
    (hist / "run.json").write_text('{"status":"complete"}')
    (ctl / "run.json").write_text('{"status":"running"}')
    monkeypatch.setattr(post.base, "AREA", hist)
    monkeypatch.setattr(post.controls, "OUTPUT", ctl)
    monkeypatch.setattr(post.base, "audit_run", lambda: pytest.fail("Read outcomes while primary run is incomplete"))
    assert post.readiness()["ready"] is False


def test_dry_run_no_mutations_or_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "AREA", tmp_path / "unused")
    monkeypatch.setattr(post, "readiness", lambda: {"ready": False, "controls_status": "not_started"})
    monkeypatch.setattr(post, "make_plan", lambda _: pytest.fail("Built plan without complete controls"))
    monkeypatch.setattr(post.funding, "check_credit", lambda _: pytest.fail("Queried private account"))
    assert post.main([])["ready"] is False
    assert not list(tmp_path.iterdir())


def test_original_historical_runner_remains_frozen():
    assert post.file_sha(post.base.__file__) == post.BASE_SHA


@pytest.fixture
def pinned_envelope(tmp_path, monkeypatch):
    paths = {name: tmp_path / "results" / f"{name}-budget.jsonl"
             for name in ("old", "health", "historical_retry", "controls")}
    ledgers = {name: post.budget.Ledger(path, "1.00", initialize=True) for name, path in paths.items()}
    ledgers["old"].reserve(post.budget.usd_nano("0.5"), {"synthetic": True})
    ledgers["historical_retry"].reserve(post.route.RESERVE_NANO, {"synthetic": True})
    control_id = ledgers["controls"].reserve(post.route.RESERVE_NANO, {"synthetic": True})
    ledgers["controls"].settle(control_id, post.budget.usd_nano("0.0001"), {"synthetic": True})
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for name in ("protocol.json", "manifest.json", "requests.jsonl"):
        (prepared / name).write_text("synthetic immutable payload\n")
    monkeypatch.setattr(post, "ROOT", tmp_path)
    monkeypatch.setattr(post, "AREA", tmp_path / "results/control_retries")
    monkeypatch.setattr(post.base, "HEALTH", paths["health"])
    monkeypatch.setattr(post.controls, "PREPARED", prepared)
    monkeypatch.setattr(post, "prior_ledger_paths", lambda: {p.relative_to(tmp_path).as_posix() for p in paths.values()})
    monkeypatch.setattr(post, "producer_pins", lambda: {"synthetic": "a" * 64})
    ready = {"ready": True, "historical_retries_status": "complete", "controls_status": "complete",
             "control_requests": 1714, "failed_control_requests": 1,
             "historical_retry_artifacts": {}, "control_artifacts": {}}
    prior = post.snapshot_previous(ready)
    inventory = {"failures": [control_item("a")]}
    plan = {"producer_pins": post.producer_pins(), "inventory": inventory,
            "inventory_sha256": post.budget.sha(inventory), "readiness": ready, "prior": prior,
            "allocation_usd": "0.005376000", "worst_case_usd": "0.005376000",
            "request_specs": {"a": {"reservation_nano": post.route.RESERVE_NANO}}}
    return plan, tmp_path


def test_prior_envelope_counts_actual_ledgers_and_full_empty_health_allowance(pinned_envelope):
    plan, _ = pinned_envelope
    assert plan["prior"]["prior_conservative_usd"] == "0.505476000"
    assert plan["prior"]["remaining_usd"] == "24.494524000"
    post.check_plan(plan)
    plan["prior"].update(prior_conservative_usd="0.502788000", remaining_usd="24.497212000")
    with pytest.raises(post.budget.GuardError, match="does not reconstruct"):
        post.check_plan(plan)


@pytest.mark.parametrize("filename", ["prepared/requests.jsonl", "results/health-budget.jsonl.lock"])
def test_primary_payload_and_prior_budget_anchor_are_immutable(pinned_envelope, filename):
    plan, root = pinned_envelope
    path = root / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(post.budget.GuardError, match="evidence changed"):
        post.check_plan(plan)


def test_missing_previous_ledger_cannot_release_authorization(pinned_envelope):
    plan, root = pinned_envelope
    (root / "results/health-budget.jsonl").unlink()
    with pytest.raises(post.budget.GuardError, match="missing"):
        post.check_plan(plan)


def test_post_control_allocation_cannot_exceed_two_attempt_policy(pinned_envelope):
    plan, _ = pinned_envelope
    plan["allocation_usd"] = "0.01"
    with pytest.raises(post.budget.GuardError, match="does not reconstruct"):
        post.check_plan(plan)
