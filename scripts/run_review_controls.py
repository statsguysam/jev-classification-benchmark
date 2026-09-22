"""Guarded execution of the frozen, matched proposal-value experiment.

Planning is read-only. Paid controls wait for completion of the existing numeric
and text matrices; their final conservative ledgers are pinned before the
remaining allocation inside $25 is initialized. No historical reserve is
released and no model request is automatically retried.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import prepare_review_controls as preparation
import run_text_jev_review as historical
import summarize_expanded_numeric as numeric_report
import summarize_text_extension as text_report
from jevbench.types import Prediction, Row

route, budget = historical.route, historical.budget
AREA = ROOT / "results/review_controls"
PREPARED = AREA / "prepared_full"
OUTPUT = AREA / "execution"
LEDGER = OUTPUT / "budget.jsonl"
TOTAL = Decimal("25.00")
TEXT_WRAPPER_SHA = "bdc3d1cac8e52791359023b57abac8833d898af6ff7b0240512e337baa3ba43e"


def require(condition, message):
    if not condition:
        raise budget.GuardError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def producer_pins():
    require(file_sha(historical.__file__) == TEXT_WRAPPER_SHA,
            "Previously audited settlement/accounting helper changed")
    historical.verify_frozen_helpers()
    return {"runner": file_sha(__file__), "preparer": file_sha(preparation.__file__),
            "text_accounting_helper": TEXT_WRAPPER_SHA,
            "route": file_sha(route.__file__), "ledger": file_sha(route._BUDGET_PATH)}


def prior_paths():
    paths = [ROOT / name for name in historical.numeric.PRIOR]
    paths += [historical.numeric.LEDGER, historical.LEDGER]
    return [item for path in paths for item in (path, path.with_name(path.name + ".lock"))]


def protected_hashes():
    return {path.relative_to(ROOT).as_posix(): file_sha(path) for path in prior_paths()}


def check_prior(snapshot):
    require(snapshot["authorized_usd"] == "25.00", "Authorization differs")
    amount, allocation = Decimal(snapshot["prior_conservative_usd"]), Decimal(snapshot["allocation_usd"])
    require(amount >= 0 and allocation > 0 and amount + allocation <= TOTAL,
            "Combined allocations exceed the existing authorization")
    require(set(snapshot["files_sha256"]) == {p.relative_to(ROOT).as_posix() for p in prior_paths()},
            "Protected prior-ledger inventory differs")
    require(all(not (ROOT / name).is_symlink() and file_sha(ROOT / name) == digest
                for name, digest in snapshot["files_sha256"].items()),
            "An earlier ledger changed; no additional control request is allowed")


def readiness():
    """Reaudit previous evidence; never treat a partial matrix as completed."""
    producer_pins()
    historical.verify_envelope()
    numeric = numeric_report.collect(samples=100)
    text = text_report.collect(samples=100)
    ready = numeric["complete_runs"] == 68 and text["complete_runs"] == 68
    result = {"numeric_complete": numeric["complete_runs"], "numeric_planned": 68,
              "text_complete": text["complete_runs"], "text_planned": 68,
              "ready_for_new_controls": ready,
              "prior_conservative_usd": text["costs"]["cumulative_conservative_usd"],
              "authorized_usd": "25.00"}
    if ready:
        amount = Decimal(result["prior_conservative_usd"])
        remaining = TOTAL - amount
        require(remaining >= Decimal(route.PRICE["per_request_reserved_usd"]),
                "Remaining authorization cannot reserve another request")
        result["allocation_usd"] = budget.usd_string(budget.usd_nano(str(remaining)))
        result["files_sha256"] = protected_hashes()
        check_prior(result)
    else:
        result.update(allocation_usd=None,
            reason="Finish and audit the already planned numeric and text reviews before allocating the remaining budget to controls")
    return result


def provenance(request, identity):
    return {"control_runner_sha256": identity["producer_pins"]["runner"],
            "control_protocol_sha256": identity["protocol_sha256"],
            "control_request_id": request["request_id"], "execution_index": request["execution_index"],
            "dataset": request["dataset"], "arm": request["arm"],
            "ledger_id": identity["ledger_id"], "stage_cap_usd": identity["prior_snapshot"]["allocation_usd"],
            "authorized_usd": "25.00", "settlement_version": historical.SETTLEMENT_VERSION}


def expected_reservation(request, identity):
    return {"row_id": request["row_id"], "provider": "jev", "model": route.MODEL,
            "route": "OpenRouter", "endpoint": route.ENDPOINT,
            "prompt_sha256": request["prompt_sha256"], "pricing": route.PRICE,
            **provenance(request, identity)}


def validate_prediction(prediction, request):
    require(prediction.row_id == request["row_id"], "Prediction row differs")
    require(type(prediction.latency_s) in (int, float) and math.isfinite(prediction.latency_s)
            and prediction.latency_s >= 0, "Invalid response latency")
    meta = prediction.metadata.get("control", {})
    require(meta == {key: request[key] for key in ("request_id", "case_id", "dataset", "arm",
                "proposal_value", "donor_row_id", "execution_index", "prompt_sha256")},
            "Prediction control identity differs")
    if prediction.error:
        require(isinstance(prediction.error, str) and prediction.label is None
                and prediction.probabilities is None,
                "A failed control cannot silently use a fallback label")
    else:
        k = len(request["choices"])
        require(type(prediction.label) is int and 0 <= prediction.label < k,
                "Control result is outside the allowed categories")
        probabilities = prediction.probabilities
        require(isinstance(probabilities, list) and len(probabilities) == k and
                all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                and math.isclose(sum(probabilities), 1, abs_tol=1e-6), "Invalid Choice distribution")
        require(prediction.label == max(range(k), key=lambda i: probabilities[i]),
                "Choice label differs from its distribution argmax")


def audit_saved(plan, identity, ledger, path):
    """Strict prefix and global ledger audit, including calls without predictions."""
    path = Path(path)
    raw = path.read_text() if path.exists() else ""
    require(not raw or raw.endswith("\n"), "Partial prediction append; preserve it for reconciliation")
    predictions = [Prediction(**json.loads(line)) for line in raw.splitlines()]
    require(len(predictions) <= len(plan["requests"]), "Too many saved control predictions")
    events, _ = ledger.inner._read()
    reserves = [e for e in events if e["type"] == "reserve"]
    results = [e for e in events if e["type"] == "result"]
    require(len(reserves) == len(results) == len(predictions),
            "Orphan reservation, result or prediction: do not retry or reset")
    require(len({e["reservation_id"] for e in reserves}) == len(reserves), "Duplicate reservation")
    require([e["reservation_id"] for e in reserves] == [e["reservation_id"] for e in results],
            "Result order differs from requests")
    for request, prediction, reserve in zip(plan["requests"], predictions, reserves):
        validate_prediction(prediction, request)
        require(reserve["reserved_nano"] == route.RESERVE_NANO and
                reserve["details"] == expected_reservation(request, identity) and
                prediction.metadata.get("budget", {}).get("reservation_id") == reserve["reservation_id"],
                "Saved request/proposal/ledger provenance differs")
    historical.validate_accounting(ledger, predictions)
    require(not ledger.snapshot()["halted"], "Control ledger has a routing or usage violation")
    ledger.remember_head(); ledger.pending_id = None
    return predictions


class ControlLedger(historical.TextLedger):
    def __init__(self, inner, plan, identity, *, stop_after=None, verify=None):
        super().__init__(inner, stop_after=stop_after)
        self.plan, self.run_identity, self.active_request = plan, identity, None
        self.verify = verify or self.verify_current

    def verify_current(self):
        historical.verify_transport()
        require(producer_pins() == self.run_identity["producer_pins"], "Control producer changed")
        check_prior(self.run_identity["prior_snapshot"])
        require(file_sha(PREPARED / "protocol.json") == self.run_identity["protocol_sha256"] and
                file_sha(PREPARED / "requests.jsonl") == self.run_identity["requests_sha256"],
                "Frozen request bundle changed")

    def reserve(self, amount, details):
        self.verify(); self.check_head()
        require(self.pending_id is None and self.active_request is not None,
                "Previous call is unresolved or no control is active")
        events, _ = self.inner._read()
        reserves = [event for event in events if event["type"] == "reserve"]
        require(len(reserves) < len(self.plan["requests"]) and
                self.active_request == self.plan["requests"][len(reserves)] and
                amount == route.RESERVE_NANO and
                details == expected_reservation(self.active_request, self.run_identity),
                "Unexpected, duplicated or out-of-order control request")
        results = [e for e in events if e["type"] == "result"]
        require(not (len(results) >= 3 and all(e["details"].get("outcome") in
                {"prediction_error", "raised"} for e in results[-3:])), "Three errors halt controls")
        if self.stop_after is not None and self.new_requests >= self.stop_after:
            raise route.StopRequested("Requested checkpoint reached")
        ident = self.inner.reserve(amount, details)
        self.pending_id = ident; self.remember_head(); self.new_requests += 1
        return ident


def annotate_settlement(prediction, ledger):
    ident = prediction.metadata["budget"]["reservation_id"]
    settled = [e for e in ledger.inner._read()[0] if e["type"] == "settle" and e["reservation_id"] == ident]
    if settled:
        amount = settled[0]["charged_nano"]
        prediction.metadata["budget"].update(settled_conservative_usd=budget.usd_string(amount),
            reservation_released=amount < route.RESERVE_NANO, settlement_version=historical.SETTLEMENT_VERSION)


def execute(plan, prior, *, init_ledger=False, prompt_key=False, stop_after=None):
    require(prior["ready_for_new_controls"], prior.get("reason", "Prior studies incomplete"))
    require(plan["protocol"]["study_role"] == "primary_full_fixed_holdouts" and
            plan["protocol"]["request_count"] == 1714, "Only the full frozen study is enabled for paid execution")
    historical.verify_transport()
    config = route.validate_config(json.loads((ROOT / "configs/jev_openrouter.json").read_text())["jev_openrouter"])
    if prompt_key:
        budget.prompt_credentials([config])
    require(bool(os.environ.get(config["api_key_env"])), "Missing OpenRouter API credential")
    receipt = historical.public_receipt(historical.check_credit(os.environ[config["api_key_env"]]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    existing = json.loads((OUTPUT / "run.json").read_text()) if (OUTPUT / "run.json").exists() else None
    if existing:
        require(not init_ledger and existing["status"] not in {"halted_billing", "halted_errors"},
                "A halted run needs reconciliation; a completed ledger must never be reinitialized")
        require(existing["prior_snapshot"] == prior, "Prior evidence/allocation differs from this execution")
    else:
        require(init_ledger, "Initialize the allocation explicitly once, after funding and prior completion")
        require(not LEDGER.exists() and not LEDGER.with_name(LEDGER.name + ".lock").exists()
                and not (OUTPUT / "predictions.jsonl").exists(), "Existing/orphan evidence must not be reset")
    inner = budget.Ledger(LEDGER, prior["allocation_usd"], initialize=init_ledger)
    identity = {"schema_version": 1, "study": "proposal-value-controls-v1",
                "producer_pins": producer_pins(), "protocol_sha256": file_sha(PREPARED / "protocol.json"),
                "requests_sha256": file_sha(PREPARED / "requests.jsonl"),
                "ledger_id": inner.identity["ledger_id"], "prior_snapshot": prior,
                "request_count": len(plan["requests"]), "route": route.ENDPOINT,
                "price": route.PRICE, "settlement_version": historical.SETTLEMENT_VERSION}
    if existing:
        require(all(existing.get(key) == value for key, value in identity.items()), "Execution identity changed")
    ledger = ControlLedger(inner, plan, identity, stop_after=stop_after)
    predictions = audit_saved(plan, identity, ledger, OUTPUT / "predictions.jsonl")
    require(not any(p.error == historical.BILLING_ERROR for p in predictions) and
            not (len(predictions) >= 3 and all(p.error for p in predictions[-3:])),
            "Saved errors halt controls; changing the status cannot clear a halt")
    record = existing or {**identity, "started_at": budget.utc_now(), "execution_sessions": []}
    record["execution_sessions"].append({"started_at": budget.utc_now(), "saved_requests": len(predictions), "funding_check": receipt})
    record.update(status="running", saved_requests=len(predictions))
    atomic_json(OUTPUT / "run.json", record)
    try:
        with (OUTPUT / "predictions.jsonl").open("a") as stream:
            for request in plan["requests"][len(predictions):]:
                ledger.active_request = request
                provider = route.GuardedOpenRouterJev(config, ledger, provenance(request, identity))
                # Ground truth is intentionally absent from the transport row.
                prediction = provider.predict(Row(request["row_id"], "", -1), request["choices"], request["prompt"])
                prediction.metadata["control"] = {key: request[key] for key in ("request_id", "case_id", "dataset", "arm",
                    "proposal_value", "donor_row_id", "execution_index", "prompt_sha256")}
                annotate_settlement(prediction, ledger)
                stream.write(json.dumps(asdict(prediction), allow_nan=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
                predictions.append(prediction); ledger.prediction_saved(prediction)
                validate_prediction(prediction, request)
                record["saved_requests"] = len(predictions)
                if prediction.error == historical.BILLING_ERROR or (len(predictions) >= 3 and all(p.error for p in predictions[-3:])):
                    record["status"] = "halted_billing" if prediction.error == historical.BILLING_ERROR else "halted_errors"
                    atomic_json(OUTPUT / "run.json", record)
                    raise budget.GuardError("Billing or consecutive errors halted controls; no automatic retry")
                if len(predictions) % 25 == 0:
                    atomic_json(OUTPUT / "run.json", record)
                    print(json.dumps({"saved_requests": len(predictions), "planned": len(plan["requests"]), "budget": ledger.snapshot()}), flush=True)
    except (route.StopRequested, budget.BudgetStop) as exc:
        record.update(status="paused_request_limit" if isinstance(exc, route.StopRequested) else "paused_budget",
                      saved_requests=len(predictions))
        atomic_json(OUTPUT / "run.json", record)
        audit_saved(plan, identity, ledger, OUTPUT / "predictions.jsonl")
        return {"status": record["status"], "saved_requests": len(predictions), "budget": ledger.snapshot()}
    audit_saved(plan, identity, ledger, OUTPUT / "predictions.jsonl")
    record.update(status="complete", completed_at=budget.utc_now(), saved_requests=len(predictions))
    atomic_json(OUTPUT / "run.json", record)
    return {"status": "complete", "saved_requests": len(predictions), "budget": ledger.snapshot()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--init-ledger", action="store_true")
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--stop-after-new-requests", type=int)
    args = parser.parse_args(argv)
    require(args.stop_after_new_requests is None or args.stop_after_new_requests > 0, "Invalid request limit")
    require(args.execute or not (args.init_ledger or args.prompt_api_key), "Credential/initialization flags require execution")
    plan = preparation.load_frozen_plan(PREPARED)
    if not args.execute:
        prior = readiness()
        print(json.dumps({"mode": "dry_run_no_writes", "control_requests": len(plan["requests"]), **prior}, indent=2))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Existing workers use these same locks. Old allocations cannot change while
    # their remaining budget is being assigned to the new experiment.
    with ExitStack() as stack:
        for path in (historical.numeric.LEDGER.parent / "review-execution.lock",
                     historical.LEDGER.parent / "review-execution.lock", OUTPUT / "execution.lock"):
            lock = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise budget.GuardError("An earlier study or control worker is active") from None
        prior = readiness()
        result = execute(plan, prior, init_ledger=args.init_ledger,
                         prompt_key=args.prompt_api_key, stop_after=args.stop_after_new_requests)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (budget.GuardError, budget.BudgetStop) as exc:
        raise SystemExit(f"STOP: {exc}") from None
