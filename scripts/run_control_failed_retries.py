"""Separate failure-only recovery AFTER the complete proposal-control study.

Dry-run is offline/read-only. Two fixed failure-only rounds; every new Jev call
retains its full reservation. Original control and repeat predictions/metrics
remain unchanged. No inference is performed at import or by the final auditor.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_failed_retries as base
import run_review_controls as controls
import summarize_review_controls as control_report
from jevbench.types import Row

preparation, completion, funding = base.preparation, base.completion, base.funding
route, budget, require = base.route, base.budget, base.require
AREA = ROOT / "results/completion_20260923/control_retries"
BASE_SHA = "f855b3299aede41f94ee8e73f9ce5454166944e69b66e145f303503df6bc5181"
ZERO = {"ledger_id": None, "budget_usd": "0.000000000", "charged_or_reserved_usd": "0.000000000",
        "remaining_reservation_usd": "0.000000000", "reservations": 0, "settlements": 0, "events": 0, "halted": False}
file_sha, atomic_json = base.file_sha, base.atomic_json


def producer_pins():
    require(file_sha(base.__file__) == BASE_SHA, "Frozen historical retry implementation changed")
    paths = [Path(__file__), Path(controls.__file__), Path(control_report.__file__), Path(controls.preparation.__file__)]
    return {**base.pins(), **{p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths}}


def prior_ledger_paths():
    return base.canonical_prior_paths() | {(base.AREA / "budget.jsonl").relative_to(ROOT).as_posix(),
                                          controls.LEDGER.relative_to(ROOT).as_posix()}


def readiness():
    """No partial control outcomes are inspected or scored."""
    historical_path, control_path = base.AREA / "run.json", controls.OUTPUT / "run.json"
    historical_status = json.loads(historical_path.read_text()).get("status") if historical_path.exists() else "not_started"
    control_status = json.loads(control_path.read_text()).get("status") if control_path.exists() else "not_started"
    if historical_status != "complete" or control_status != "complete":
        return {"ready": False, "historical_retries_status": historical_status, "controls_status": control_status}
    producer_pins()
    historical = base.audit_run()
    prepared = controls.preparation.load_frozen_plan(controls.PREPARED)
    predictions, observed = control_report.load_execution(prepared)
    require(observed["status"] == "complete" and len(predictions) == len(prepared["requests"]) == 1714,
            "All 1714 primary and repeat control requests must be complete and audited")
    return {"ready": True, "historical_retries_status": "complete", "controls_status": "complete",
            "control_requests": 1714, "failed_control_requests": sum(bool(p.error) for p in predictions),
            "historical_retry_artifacts": {str((base.AREA / name).relative_to(ROOT)): value
                                           for name, value in historical["artifact_sha256"].items()},
            "control_artifacts": {str((controls.OUTPUT / name).relative_to(ROOT)): value
                                   for name, value in observed["artifact_sha256"].items()}}


def snapshot_previous(ready):
    inventory = preparation.ledger_inventory(ROOT)
    observed = {entry["path"] for entry in inventory.values() if not (ROOT / entry["path"]).is_relative_to(AREA)}
    require(observed == prior_ledger_paths(), "Previous ledger inventory is incomplete or unexpected")
    protected, total = {}, 0
    for entry in inventory.values():
        path = ROOT / entry["path"]
        if path.is_relative_to(AREA):
            continue
        header = preparation.read_predictions(path)[0]
        state = budget.Ledger(path, budget.usd_string(header["budget_nano"])).snapshot()
        require(not state["halted"], "A previous ledger is halted")
        total += route.RESERVE_NANO if path == base.HEALTH else budget.usd_nano(state["charged_or_reserved_usd"])
        protected.update(entry["pins"])
    require(total <= base.TOTAL_NANO, "Previous conservative spending exceeds $25")
    protected.update(ready["historical_retry_artifacts"])
    protected.update(ready["control_artifacts"])
    for name in ("protocol.json", "manifest.json", "requests.jsonl"):
        path = controls.PREPARED / name
        protected[path.relative_to(ROOT).as_posix()] = file_sha(path)
    return {"prior_conservative_usd": budget.usd_string(total), "remaining_usd": budget.usd_string(base.TOTAL_NANO - total),
            "authorized_usd": "25.00", "ledger_paths": sorted(observed), "files_sha256": protected}


def make_plan(ready):
    require(ready["ready"], "Complete and audit historical retries and all controls before allocating recovery")
    prior = snapshot_previous(ready)
    all_failures = preparation.build_inventory()
    failures = [item for item in all_failures["failures"] if item["request_recipe"] == "control_frozen_request"]
    require(len(failures) == ready["failed_control_requests"], "Control failure inventory does not cover the completed control audit")
    specs = {}
    for item in failures:
        require(item["kind"] == "paid_failure" and item["provider"] == "jev" and item["model"] == route.MODEL,
                "Only failed, previously paid control Jev requests may enter this phase")
        request = preparation.reconstruct_request(item)
        require(not request["is_new_dependent_request"] and base.request_amount(request) == route.RESERVE_NANO,
                "Control retry request or reserve differs")
        specs[item["failure_id"]] = {"request_sha256": base.request_sha(request), "reservation_nano": route.RESERVE_NANO}
    inventory = {"schema_version": 1, "failures": failures,
                 "scope": "Only failures from the finalized 1714-request control experiment; historical failures are excluded",
                 "source_inventory_sha256": budget.sha(all_failures)}
    worst = 2 * len(failures) * route.RESERVE_NANO
    allocation = min(worst, budget.usd_nano(prior["remaining_usd"]))
    require(not failures or allocation >= route.RESERVE_NANO, "Remaining authorization cannot reserve a control retry")
    return {"schema_version": 1, "study": "post-control-failure-recovery-v1", "readiness": ready,
        "producer_pins": producer_pins(), "prior": prior, "inventory": inventory,
        "inventory_sha256": budget.sha(inventory), "request_specs": specs,
        "allocation_usd": budget.usd_string(allocation), "worst_case_usd": budget.usd_string(worst),
        "policy": {"phases": ["paid_round_1", "paid_round_2"], "max_additional_attempts_per_request": 2,
                   "stop_on_first_valid_success": True, "settlements_allowed": False,
                   "original_control_and_repeat_results_preserved": True}}


def check_plan(plan, *, read_only=False):
    require(plan["producer_pins"] == producer_pins() and plan["inventory_sha256"] == budget.sha(plan["inventory"]),
            "Post-control producer or frozen inventory changed")
    ready, prior = plan["readiness"], plan["prior"]
    require(ready["ready"] and ready["historical_retries_status"] == ready["controls_status"] == "complete"
            and ready["control_requests"] == 1714 and prior["authorized_usd"] == "25.00", "Incomplete preceding studies or authorization")
    require(set(prior["ledger_paths"]) == prior_ledger_paths(), "Canonical previous ledger inventory differs")
    current = {p.relative_to(ROOT).as_posix() for p in (ROOT / "results").rglob("*budget*.jsonl") if not p.is_relative_to(AREA)}
    require(set(prior["ledger_paths"]) <= current if read_only else set(prior["ledger_paths"]) == current,
            "A previous allocation is missing or another allocation appeared")
    protected = dict(prior["files_sha256"])
    for item in plan["inventory"]["failures"]:
        require(item["provider"] == "jev" and item["kind"] == "paid_failure" and item["request_recipe"] == "control_frozen_request",
                "Retry inventory includes an out-of-scope request")
        protected.update(item["artifact_sha256"])
    require(all(not (ROOT / name).is_symlink() and file_sha(ROOT / name) == value for name, value in protected.items()),
            "Original control, payload, retry, or budget evidence changed")
    total = 0
    for name in prior["ledger_paths"]:
        path = ROOT / name
        header = preparation.read_predictions(path)[0]
        state = budget.Ledger(path, budget.usd_string(header["budget_nano"])).snapshot()
        require(not state["halted"], "A protected prior ledger is halted")
        total += route.RESERVE_NANO if path == base.HEALTH else budget.usd_nano(state["charged_or_reserved_usd"])
    worst = 2 * len(plan["inventory"]["failures"]) * route.RESERVE_NANO
    allocation = budget.usd_nano(plan["allocation_usd"])
    require(total == budget.usd_nano(prior["prior_conservative_usd"]) and total <= base.TOTAL_NANO
            and budget.usd_nano(prior["remaining_usd"]) == base.TOTAL_NANO - total
            and worst == budget.usd_nano(plan["worst_case_usd"])
            and allocation == min(worst, base.TOTAL_NANO - total), "Post-control allocation does not reconstruct within $25")
    require(len(plan["inventory"]["failures"]) == ready["failed_control_requests"]
            and all(spec["reservation_nano"] == route.RESERVE_NANO for spec in plan["request_specs"].values()),
            "Failure count or full-reservation policy differs")


def control_metadata(action):
    item = action["item"]
    return {"control_request_id": item["control_request_id"], "dataset": item["dataset"], "arm": item["control_arm"],
            "original_control_prediction_sha256": item["original_prediction_sha256"]}


class ControlRetryLedger(base.RetryLedger):
    def settle(self, *args):
        raise budget.GuardError("Every post-control Jev reservation must remain retained")


def audit_saved(plan, ledger, path):
    records = base.audit_saved(plan, ledger, path)
    _, actions = base.walk_policy(plan, records)
    require(all(record["prediction"]["metadata"].get("post_control") == control_metadata(action)
                for record, action in zip(records, actions)), "Control-recovery request/arm lineage differs")
    require(ledger.snapshot()["settlements"] == 0, "A post-control reservation was released")
    return records


class LiveVerifier:
    def __init__(self, plan, session, transport, receipt):
        self.plan, self.session, self.transport, self.receipt = plan, session, transport, receipt
        self.transport.area = session
        self.credit_count = 0

    def check(self):
        check_plan(self.plan)
        require(json.loads((AREA / "plan.json").read_text()) == self.plan, "Frozen control retry plan changed")
        self.transport.check()
        if (datetime.now(timezone.utc) - datetime.fromisoformat(self.receipt["checked_at"])).total_seconds() > 45:
            self.receipt = funding.check_credit(os.environ.get("OPENROUTER_API_KEY", ""))
            self.credit_count += 1
            completion.probe.write_exclusive(self.session / f"funding-{self.credit_count:03d}.json", funding.public_receipt(self.receipt))
        funding.verify_receipt(self.receipt)
        require(Decimal(self.receipt["available_credit_usd"]) >= Decimal(route.PRICE["per_request_reserved_usd"]),
                "Available Jev credit cannot fund the next full reservation")


def protect_final(record):
    names = ["plan.json", "attempts.jsonl"] + ([] if record["no_op"] else ["budget.jsonl", "budget.jsonl.lock"])
    record["protected_artifacts_sha256"] = {name: file_sha(AREA / name) for name in names}


def execute(plan, *, initialize=False, prompt_key=False, stop_after=None, resume_after_errors=False):
    check_plan(plan)
    plan_path, run_path, attempts_path = (AREA / name for name in ("plan.json", "run.json", "attempts.jsonl"))
    existing = json.loads(run_path.read_text()) if run_path.exists() else None
    if existing:
        require(not initialize and json.loads(plan_path.read_text()) == plan, "Do not replace or reinitialize post-control evidence")
        if existing.get("status") == "complete":
            return audit_run()["record"]
        require(existing["status"] not in {"halted_billing", "halted_errors"} or resume_after_errors, "Explicit fresh-funded resume required after service halt")
    else:
        require(not any((AREA / name).exists() for name in ("plan.json", "attempts.jsonl", "budget.jsonl", "budget.jsonl.lock")),
                "Unknown/partial post-control evidence requires reconciliation")
    if not plan["inventory"]["failures"]:
        require(existing is None, "Incomplete no-op evidence requires reconciliation")
        atomic_json(plan_path, plan)
        with attempts_path.open("x") as stream:
            stream.flush(); os.fsync(stream.fileno())
        record = {"schema_version": 1, "study": plan["study"], "status": "complete", "no_op": True,
                  "plan_sha256": file_sha(plan_path), "budget": dict(ZERO), "counts": base.final_counts(plan, []),
                  "completed_at": budget.utc_now(), "note": "No failed original control requests; zero new calls and no budget ledger initialized."}
        protect_final(record); atomic_json(run_path, record)
        return record
    require(existing is not None or initialize, "Explicit --init-ledger required once for a new paid recovery phase")
    config = {"provider": "jev", "model": route.MODEL, "base_url": "https://openrouter.ai/api/v1",
              "api_key_env": "OPENROUTER_API_KEY", "timeout": 120}
    if prompt_key:
        budget.prompt_credentials([config])
    receipt = funding.check_credit(os.environ.get("OPENROUTER_API_KEY", ""))
    require(Decimal(receipt["available_credit_usd"]) >= Decimal(route.PRICE["per_request_reserved_usd"]), "Insufficient available Jev credit")
    transport = completion.VerifiedTransport(sys.modules[__name__], file_sha(__file__), execute=True)
    transport.check()
    if not existing:
        atomic_json(plan_path, plan)
    inner = budget.Ledger(AREA / "budget.jsonl", plan["allocation_usd"], initialize=initialize)
    record = existing or {"schema_version": 1, "study": plan["study"], "no_op": False,
                           "ledger_id": inner.identity["ledger_id"], "plan_sha256": file_sha(plan_path), "execution_sessions": []}
    require(record["ledger_id"] == inner.identity["ledger_id"] and record["plan_sha256"] == file_sha(plan_path), "Post-control execution identity differs")
    session = AREA / "sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12])
    session.mkdir(parents=True, exist_ok=False)
    for index, proof in enumerate(transport.receipts, 1):
        completion.probe.write_exclusive(session / f"public-verification-{index:03d}.json", proof)
    completion.probe.write_exclusive(session / "funding-initial.json", funding.public_receipt(receipt))
    verifier = LiveVerifier(plan, session, transport, receipt)
    ledger = ControlRetryLedger(inner, plan, verifier.check, stop_after=stop_after)
    records = audit_saved(plan, ledger, attempts_path)
    require(base.halt_reason(records) is None or resume_after_errors, "Audited service-error tail requires explicit resume")
    ledger.allow_halt_tail_once = bool(resume_after_errors)
    record["execution_sessions"].append({"started_at": budget.utc_now(), "saved_attempts": len(records),
        "path": session.relative_to(ROOT).as_posix(), "producer_pins": producer_pins(), "resume_after_errors": resume_after_errors})
    record["status"] = "running"; atomic_json(run_path, record)
    try:
        with attempts_path.open("a") as stream:
            while True:
                action, _ = base.walk_policy(plan, records)
                if action is None: break
                ledger.active = action
                request = action["request"]
                provider = route.GuardedOpenRouterJev(request["config"], ledger, base.provenance(action, plan))
                prediction = provider.predict(Row(request["row_id"], "", -1), request["labels"], request["prompt"])
                prediction.metadata.update(retry=action["metadata"], post_control=control_metadata(action))
                saved = {"attempt_index": action["attempt_index"], "identity": action["identity"],
                         "stage": action["stage"], "prediction": asdict(prediction)}
                require(ledger.pending is not None, "Provider returned without a reserved request")
                base.analysis.validate_prediction(saved["prediction"], request["row_id"], len(request["labels"]), require_probabilities=True)
                stream.write(json.dumps(saved, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
                records = audit_saved(plan, ledger, attempts_path)
                print(json.dumps({"saved_attempts": len(records), "failed": bool(prediction.error), "budget": ledger.snapshot()}), flush=True)
                if base.halt_reason(records) and base.walk_policy(plan, records)[0] is not None:
                    record["status"] = base.halt_reason(records); break
        if record["status"] == "running":
            require(base.walk_policy(plan, records)[0] is None, "Post-control recovery policy is incomplete")
            record["status"] = "complete"
    except route.StopRequested:
        record["status"] = "paused"
    except budget.BudgetStop:
        record["status"] = "budget_exhausted"; raise
    except BaseException:
        record["status"] = "interrupted"; raise
    finally:
        record.update(budget=ledger.snapshot(), counts=base.final_counts(plan, records), finished_at=budget.utc_now())
        if record["status"] == "complete":
            audit_saved(plan, ledger, attempts_path); protect_final(record)
        atomic_json(run_path, record)
    return record


def audit_run(area=None):
    area = AREA if area is None else Path(area)
    require(area.resolve() == AREA.resolve() and not area.is_symlink(), "Audit the canonical post-control recovery area")
    basic = ("run.json", "plan.json", "attempts.jsonl")
    require(all((area / name).is_file() and not (area / name).is_symlink() for name in basic), "Incomplete final post-control evidence")
    record, plan = (json.loads((area / name).read_text()) for name in ("run.json", "plan.json"))
    require(record["status"] == "complete", "Post-control recovery has not completed")
    check_plan(plan, read_only=True)
    no_op = not plan["inventory"]["failures"]
    require(record["no_op"] == no_op, "No-op classification differs from inventory")
    names = basic + (() if no_op else ("budget.jsonl", "budget.jsonl.lock"))
    require(all((area / name).is_file() and not (area / name).is_symlink() for name in names), "Missing or unsafe budget artifact")
    before = {name: file_sha(area / name) for name in names}
    require(record["plan_sha256"] == before["plan.json"] and record["protected_artifacts_sha256"] ==
            {name: value for name, value in before.items() if name != "run.json"}, "Post-control final artifacts changed")
    if no_op:
        require(not (area / "attempts.jsonl").read_bytes() and not (area / "budget.jsonl").exists()
                and not (area / "budget.jsonl.lock").exists(), "No-op unexpectedly contains paid evidence")
        attempts, state = [], dict(ZERO)
    else:
        ledger = ControlRetryLedger(budget.Ledger(area / "budget.jsonl", plan["allocation_usd"]), plan, lambda: None)
        attempts = audit_saved(plan, ledger, area / "attempts.jsonl")
        require(base.walk_policy(plan, attempts)[0] is None, "Saved post-control policy is incomplete")
        state = ledger.snapshot()
        require(record["ledger_id"] == state["ledger_id"], "Post-control ledger identity changed")
    require(record["budget"] == state and record["counts"] == base.final_counts(plan, attempts)
            and all(file_sha(area / name) == before[name] for name in names), "Post-control final accounting differs")
    return {"plan": plan, "record": record, "attempts": attempts, "budget": state, "artifact_sha256": before}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("execute", "init-ledger", "prompt-api-key", "resume-after-errors"):
        parser.add_argument("--" + name, action="store_true")
    parser.add_argument("--stop-after-new-requests", type=int)
    args = parser.parse_args(argv)
    require(args.stop_after_new_requests is None or args.stop_after_new_requests > 0, "Invalid checkpoint count")
    require(args.execute or not (args.init_ledger or args.prompt_api_key or args.resume_after_errors), "Mutation/credential flags require --execute")
    ready = readiness()
    if not args.execute:
        plan = make_plan(ready) if ready["ready"] else None
        result = {"mode": "dry_run_no_writes", "ready": ready["ready"], "controls_status": ready["controls_status"],
                  "failures": len(plan["inventory"]["failures"]) if plan else None,
                  "allocation_usd": plan["allocation_usd"] if plan else None}
        print(json.dumps(result, indent=2)); return result
    require(ready["ready"], "Finish historical retries and all controls first")
    AREA.mkdir(parents=True, exist_ok=True)
    require(not AREA.is_symlink(), "Post-control retry directory must be canonical")
    with ExitStack() as stack:
        paths = (base.historical.numeric.LEDGER.parent / "review-execution.lock", base.historical.LEDGER.parent / "review-execution.lock",
                 base.AREA / "execution.lock", controls.OUTPUT / "execution.lock", AREA / "execution.lock")
        for path in paths:
            require(not path.is_symlink(), "Execution lock must be canonical")
            lock = stack.enter_context(path.open("a+"))
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise budget.GuardError("Another original/control/recovery execution is active") from None
        plan_path = AREA / "plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else make_plan(readiness())
        return execute(plan, initialize=args.init_ledger, prompt_key=args.prompt_api_key,
                       stop_after=args.stop_after_new_requests, resume_after_errors=args.resume_after_errors)


if __name__ == "__main__":
    main()
