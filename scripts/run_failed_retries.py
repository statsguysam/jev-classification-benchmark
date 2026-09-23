"""Explicit, finite recovery of failed calls; dry-run is offline and read-only.

Both original matrices must finish before allocation. Original paid failures get
two failure-only rounds, followed by two dependent-review rounds. Successful
responses are never repeated. Every new Jev reservation is retained in full.
Only the frozen OpenAI budget driver's verified successful-usage settlement may
release part of a NEW Astra reservation. Historical files are never rewritten.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import analyze_jev_recovery as analysis
import prepare_failed_retries as preparation
import run_jev_completion as completion
import recover_expanded_numeric_billing as funding
import summarize_expanded_numeric as numeric_report
import summarize_text_extension as text_report
from jevbench import providers
from jevbench.types import Prediction, Row

historical, route, budget = completion.historical, completion.route, completion.budget
AREA = ROOT / "results/completion_20260923/retries"
CONTROL_AREA = ROOT / "results/review_controls/execution"
PRICES = ROOT / "configs/completion_prices_20260923.json"
HEALTH = ROOT / "results/health_checks/jev-20260923/budget.jsonl"
TOTAL_NANO = budget.usd_nano("25.00")
PHASES = (("paid_round_1", "paid_failure"), ("paid_round_2", "paid_failure"),
          ("dependent_round_1", "upstream_skip"), ("dependent_round_2", "upstream_skip"))
require = historical.require


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def pins():
    paths = [Path(__file__), Path(preparation.__file__), Path(analysis.__file__), Path(completion.__file__),
             Path(completion.probe.__file__), Path(funding.__file__), Path(route.__file__),
             route._BUDGET_PATH, PRICES]
    historical.verify_frozen_helpers()
    return {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths}


def canonical_prior_paths():
    return set(historical.numeric.PRIOR) | {historical.numeric.LEDGER.relative_to(ROOT).as_posix(),
        historical.LEDGER.relative_to(ROOT).as_posix(), HEALTH.relative_to(ROOT).as_posix()}


def prior_state():
    require(not CONTROL_AREA.exists(), "Controls execution already exists; retry allocation must precede controls")
    require(HEALTH.is_file() and HEALTH.with_name(HEALTH.name + ".lock").is_file()
            and not HEALTH.is_symlink() and not HEALTH.with_name(HEALTH.name + ".lock").is_symlink(),
            "Known health ledger and anchor must remain present")
    historical.verify_envelope()
    numeric, text = numeric_report.collect(samples=100), text_report.collect(samples=100)
    ready = numeric["complete_runs"] == 68 and text["complete_runs"] == 68
    inventory = preparation.ledger_inventory(ROOT)
    canonical = canonical_prior_paths()
    observed = {entry["path"] for entry in inventory.values() if not (ROOT / entry["path"]).is_relative_to(AREA)}
    require(observed <= canonical and (not ready or observed == canonical), "Unexpected or missing first-pass budget ledger")
    files, total = {}, 0
    for entry in inventory.values():
        path = ROOT / entry["path"]
        if path.is_relative_to(AREA):
            continue
        events = preparation.read_predictions(path)
        cap = budget.usd_string(events[0]["budget_nano"])
        state = budget.Ledger(path, cap).snapshot()
        require(not state["halted"], "A previous budget ledger is halted")
        amount = budget.usd_nano(state["charged_or_reserved_usd"])
        if path == HEALTH:
            require(state["budget_usd"] == "0.002688000" and state["reservations"] <= 1
                    and state["settlements"] == 0, "Health accounting differs")
            amount = route.RESERVE_NANO
        total += amount
        files.update(entry["pins"])
    require(total <= TOTAL_NANO, "Previous spend exceeds aggregate authorization")
    return {"ready": ready, "numeric_complete": numeric["complete_runs"], "text_complete": text["complete_runs"],
            "authorized_usd": "25.00", "prior_conservative_usd": budget.usd_string(total),
            "remaining_usd": budget.usd_string(TOTAL_NANO - total), "files_sha256": files}


def price_for(request):
    if request["config"]["provider"] == "jev":
        return route.PRICE
    config = budget.validate_config(request["config"])
    require(config["provider"] == "openai" and config["model"] == "gpt-6-astra", "Unpriced retry provider/model")
    price = dict(json.loads(PRICES.read_text())[config["model"]])
    expected = {"provider": "openai", "input_usd_per_million": "10", "output_usd_per_million": "50",
        "input_multiplier": "1.25", "mode": "utf8_bound", "max_input_tokens": 272000,
        "max_output_tokens": 128000, "verified_on": "2026-09-23",
        "source": "https://developers.openai.com/api/docs/models/gpt-6-astra",
        "declaration": "I verified the maximum standard request rates and context limits at the cited official source today"}
    require(price == expected, "Recorded Astra price declaration differs")
    return {**price, "verification_kind": "operator_declared"}


def request_amount(request):
    if request["config"]["provider"] == "jev":
        route.validate_config(request["config"])
        return route.RESERVE_NANO
    return budget.reservation(budget.validate_config(request["config"]), price_for(request), request["prompt"])[0]


def request_sha(request):
    return budget.sha({key: request[key] for key in ("row_id", "prompt_sha256", "labels", "config", "endpoint", "request_body")}
                      | {"dependency": request.get("dependency")})


def make_plan(prior):
    require(prior["ready"], "Finish both 68-condition first-pass matrices before retries")
    inventory = preparation.build_inventory()
    require(inventory["pending_first_pass"]["numeric_pending_rows"] == 0
            and inventory["pending_first_pass"]["text_pending_rows"] == 0, "First-pass rows remain pending")
    require(not any(item["request_recipe"] == "control_frozen_request" for item in inventory["failures"]),
            "Post-control retries require a separate authorized phase")
    specs, worst = {}, 0
    for item in inventory["failures"]:
        require(item["kind"] in {"paid_failure", "upstream_skip"}, "Unknown failure kind")
        if item["kind"] == "paid_failure":
            request = preparation.reconstruct_request(item)
            amount = request_amount(request)
            specs[item["failure_id"]] = {"request_sha256": request_sha(request), "reservation_nano": amount}
        else:
            amount = route.RESERVE_NANO
            specs[item["failure_id"]] = {"request_sha256": None, "reservation_nano": amount}
        worst += 2 * amount
    remaining = budget.usd_nano(prior["remaining_usd"])
    allocation = min(remaining, worst)
    require(worst > 0 and allocation > 0, "No funded retry allocation is needed or available")
    return {"schema_version": 1, "study": "failed-call-recovery-v1", "producer_pins": pins(),
        "prior": prior, "inventory": inventory, "inventory_sha256": budget.sha(inventory), "request_specs": specs,
        "allocation_usd": budget.usd_string(allocation), "worst_case_usd": budget.usd_string(worst),
        "policy": {"phases": [phase for phase, _ in PHASES], "max_additional_attempts_per_failure": 2,
                   "max_dependent_calls": 2, "stop_on_first_valid_success": True,
                   "Jev_settlements": False, "historical_reservations_released": False}}


def check_plan(plan, *, read_only=False):
    require(read_only or not CONTROL_AREA.exists(), "Controls allocation appeared; no retry request allowed")
    require(plan["producer_pins"] == pins(), "Retry producer, helper or pricing file changed")
    require(plan["inventory_sha256"] == budget.sha(plan["inventory"]), "Failure inventory changed")
    prior, allocation = plan["prior"], budget.usd_nano(plan["allocation_usd"])
    amount = budget.usd_nano(prior["prior_conservative_usd"])
    require(prior["ready"] and prior["numeric_complete"] == prior["text_complete"] == 68
            and prior["authorized_usd"] == "25.00" and 0 < allocation
            and amount + allocation <= TOTAL_NANO, "Retry authorization/envelope differs")
    worst = sum(2 * plan["request_specs"][item["failure_id"]]["reservation_nano"] for item in plan["inventory"]["failures"])
    require(worst == budget.usd_nano(plan["worst_case_usd"])
            and budget.usd_nano(prior["remaining_usd"]) == TOTAL_NANO - amount
            and allocation == min(TOTAL_NANO - amount, worst), "Retry allocation is not the fixed minimum of remaining and worst-case budget")
    current_paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / "results").rglob("*budget*.jsonl")
                     if not p.is_relative_to(AREA)}
    expected_paths = {name for name in prior["files_sha256"] if not name.endswith(".lock")}
    require(expected_paths == canonical_prior_paths(), "Pinned prior ledger inventory is not the canonical first-pass set")
    require(expected_paths <= current_paths if read_only else current_paths == expected_paths, "Earlier ledger inventory changed")
    protected = dict(prior["files_sha256"])
    for item in plan["inventory"]["failures"]:
        protected.update(item["artifact_sha256"])
    require(all(not (ROOT / name).is_symlink() and file_sha(ROOT / name) == value for name, value in protected.items()),
            "Original failure evidence or previous budget changed")
    observed_total = 0
    for name in expected_paths:
        path = ROOT / name
        header = preparation.read_predictions(path)[0]
        state = budget.Ledger(path, budget.usd_string(header["budget_nano"])).snapshot()
        require(not state["halted"], "A pinned prior ledger is halted")
        observed_total += route.RESERVE_NANO if path == HEALTH else budget.usd_nano(state["charged_or_reserved_usd"])
    require(observed_total == amount, "Pinned previous budgets do not reconstruct the declared prior spend")


def build_action(plan, item, phase, index, source=None):
    request = preparation.reconstruct_request(item, recovered_source_prediction=source)
    if item["kind"] == "paid_failure":
        require(request_sha(request) == plan["request_specs"][item["failure_id"]]["request_sha256"], "Original request changed")
    original = preparation.original_prediction(item, ROOT)
    identity = {"condition_id": item["run_id"] + ("::dependent-source-recovery-v1" if source is not None else ""),
        "row_id": item["row_id"], "prompt_sha256": request["prompt_sha256"],
        "choices_sha256": analysis.digest(request["labels"]),
        "source_sha256": analysis.digest({"original_source": item["source_condition"],
                                         "dependency": request.get("dependency"), "files": item["artifact_sha256"]}),
        "original_prediction_sha256": analysis.digest(original)}
    stage = "dependent_first_call" if source is not None and index == 1 else "retry"
    meta = {"failure_id": item["failure_id"], "phase": phase, "request_sha256": request_sha(request),
            "inventory_sha256": plan["inventory_sha256"], "producer_sha256": plan["producer_pins"][Path(__file__).relative_to(ROOT).as_posix()],
            "dependency": request.get("dependency")}
    return {"item": item, "request": request, "attempt_index": index, "identity": identity, "stage": stage, "metadata": meta}


def walk_policy(plan, records, builder=None):
    """Replay the deterministic policy; return exactly the next permitted call."""
    builder = builder or build_action
    successes, counts, cursor, actions = {}, {}, 0, []
    for phase, kind in PHASES:
        for item in plan["inventory"]["failures"]:
            fid = item["failure_id"]
            if item["kind"] != kind or fid in successes:
                continue
            source = successes.get(item.get("depends_on_failure_id")) if kind == "upstream_skip" else None
            if kind == "upstream_skip" and source is None:
                continue
            action = builder(plan, item, phase, counts.get(fid, 0) + 1, source)
            if cursor == len(records):
                return action, actions
            record = records[cursor]
            require(set(record) == {"attempt_index", "identity", "prediction", "stage"}
                    and all(record[key] == action[key] for key in ("attempt_index", "identity", "stage")),
                    "Saved attempt is not the next deterministic policy action")
            prediction = record["prediction"]
            require(prediction.get("metadata", {}).get("retry") == action["metadata"], "Saved request/dependency provenance differs")
            valid = analysis.validate_prediction(prediction, item["row_id"], len(action["request"]["labels"]),
                                                 require_probabilities=item["provider"] == "jev")
            if valid:
                successes[fid] = prediction
            counts[fid] = action["attempt_index"]
            actions.append(action); cursor += 1
    require(cursor == len(records), "Saved attempts exceed the finite recovery policy")
    return None, actions


def provenance(action, plan):
    return {"recovery": action["metadata"], "request_identity": action["identity"],
            "attempt_index": action["attempt_index"], "recovery_allocation_usd": plan["allocation_usd"]}


def expected_reservation(action, plan):
    request = action["request"]
    common = {"row_id": request["row_id"], "provider": request["config"]["provider"], "model": request["config"]["model"],
              "prompt_sha256": request["prompt_sha256"], "pricing": price_for(request), **provenance(action, plan)}
    if common["provider"] == "jev":
        return {**common, "route": "OpenRouter", "endpoint": route.ENDPOINT}
    return {**common, "bound": budget.reservation(budget.validate_config(request["config"]), common["pricing"], request["prompt"])[1]}


def settlement(details, action):
    config, price = action["request"]["config"], price_for(action["request"])
    if config["provider"] != "openai" or details.get("outcome") != "returned" or details.get("usage_exceeded_reservation_assumptions"):
        return None
    usage, model = details.get("usage", {}), details.get("resolved_model")
    if not (isinstance(model, str) and (model == config["model"] or re.fullmatch(re.escape(config["model"]) + r"-\d{4}-\d{2}-\d{2}", model))
            and all(type(usage.get(k)) is int and usage[k] > 0 for k in ("input_tokens", "output_tokens"))):
        return None
    amount = budget.usd_nano((Decimal(usage["input_tokens"]) * Decimal(price["input_usd_per_million"]) * Decimal(price["input_multiplier"])
                           + Decimal(usage["output_tokens"]) * Decimal(price["output_usd_per_million"])) / Decimal(1_000_000))
    evidence = {"usage": usage, "pricing_sha256": budget.sha(price), "model": model,
                "basis": "complete successful OpenAI usage; maximum input/cache-write rate; no cache discount"}
    return amount, evidence


class RetryLedger:
    def __init__(self, inner, plan, verify, *, stop_after=None):
        self.inner, self.identity, self.plan, self.verify = inner, inner.identity, plan, verify
        self.stop_after, self.new_requests = stop_after, 0
        self.pending, self.active, self.saved, self.head = None, None, [], None
        self.allow_halt_tail_once = False

    def remember(self):
        self.head = self.inner._read()[0][-1]["event_sha256"]

    def check_head(self):
        require(self.head is not None and self.inner._read()[0][-1]["event_sha256"] == self.head, "Unaudited or changed retry ledger")

    def reserve(self, amount, details):
        self.verify(); self.check_head()
        require(self.pending is None and self.active is not None, "Previous call lacks a durable prediction or no active request")
        require(halt_reason(self.saved) is None or self.allow_halt_tail_once,
                "Audited service-error tail requires explicit resume and fresh funding")
        expected, _ = walk_policy(self.plan, self.saved)
        require(expected == self.active and details == expected_reservation(self.active, self.plan)
                and amount == request_amount(self.active["request"]), "Request differs from the next frozen recovery action")
        if self.stop_after is not None and self.new_requests >= self.stop_after:
            raise route.StopRequested("Requested recovery checkpoint reached")
        ident = self.inner.reserve(amount, details)
        self.allow_halt_tail_once = False
        self.pending = ident; self.new_requests += 1; self.remember()
        return ident

    def result(self, ident, details):
        self.check_head()
        require(ident == self.pending and not any(e["type"] == "result" and e["reservation_id"] == ident for e in self.inner._read()[0]),
                "Duplicate or unrelated recovery result")
        self.inner.result(ident, details); self.remember()

    def settle(self, ident, amount, evidence):
        self.check_head()
        results = [e["details"] for e in self.inner._read()[0] if e["type"] == "result" and e["reservation_id"] == ident]
        require(ident == self.pending and len(results) == 1 and settlement(results[0], self.active) == (amount, evidence),
                "Only exact verified NEW OpenAI usage can settle; Jev reservations stay retained")
        self.inner.settle(ident, amount, evidence); self.remember()

    def snapshot(self):
        return self.inner.snapshot()


def audit_saved(plan, ledger, path, builder=None):
    raw = path.read_bytes() if path.exists() else b""
    require(not raw or raw.endswith(b"\n"), "Partial attempt append; do not truncate or retry")
    records = [json.loads(line) for line in raw.splitlines()]
    _, actions = walk_policy(plan, records, builder)
    events, _ = ledger.inner._read()
    reserves = [e for e in events if e["type"] == "reserve"]
    results = [e for e in events if e["type"] == "result"]
    require(len(records) == len(reserves) == len(results)
            and [e["reservation_id"] for e in reserves] == [e["reservation_id"] for e in results],
            "Orphan request/result/prediction; preserve it and reconcile before resume")
    for record, action, reserve, result in zip(records, actions, reserves, results):
        prediction, details = record["prediction"], result["details"]
        metadata = prediction["metadata"].get("budget", {})
        require(reserve["details"] == expected_reservation(action, plan)
                and reserve["reserved_nano"] == request_amount(action["request"])
                and metadata.get("ledger_id") == ledger.identity["ledger_id"]
                and metadata.get("reservation_id") == reserve["reservation_id"]
                and metadata.get("reserved_upper_bound_usd") == budget.usd_string(reserve["reserved_nano"]),
                "Attempt reservation or budget ownership differs")
        require(details.get("outcome") == ("prediction_error" if prediction["error"] else "returned")
                and details.get("usage") == {"input_tokens": prediction["input_tokens"], "output_tokens": prediction["output_tokens"]}
                and details.get("resolved_model") == prediction["metadata"].get("resolved_model")
                and details.get("usage_exceeded_reservation_assumptions") is False, "Saved result/usage differs from prediction")
        for key in ("input_tokens", "output_tokens"):
            value = prediction[key]
            require(value is None or type(value) is int and value >= 0, "Malformed saved usage")
        settled = [e for e in events if e["type"] == "settle" and e["reservation_id"] == reserve["reservation_id"]]
        expected = settlement(details, action)
        if action["item"]["provider"] == "jev":
            require(not settled and details.get("reservation_released") is False
                    and metadata.get("reservation_released") is False
                    and details.get("reported_cost_usd") == prediction["metadata"]["openrouter"]["reported_cost_usd"]
                    and details.get("request_id") == prediction["metadata"]["openrouter"]["id"]
                    and details.get("provider") == prediction["metadata"]["openrouter"]["provider"]
                    and details.get("route") == "OpenRouter", "Jev accounting or full-reserve policy differs")
            cost = route.parse_cost(details.get("reported_cost_usd"))
            require((cost is None or Decimal(cost) <= Decimal(route.PRICE["per_request_reserved_usd"]))
                    and (prediction["input_tokens"] is None or prediction["input_tokens"] <= route.PRICE["reserve_input_tokens"]),
                    "Saved Jev usage exceeds its reservation assumptions")
            require(details.get("provider") in (None, "TypeSafe")
                    and details.get("resolved_model") in (None, *route.ALLOWED_RESPONSE_MODELS), "Saved Jev route/model differs")
            if not prediction["error"]:
                require(details.get("provider") == "TypeSafe" and details.get("resolved_model") in route.ALLOWED_RESPONSE_MODELS,
                        "Successful Jev result lacks an approved route/model")
        elif expected is None:
            require(not settled and "settled_conservative_usd" not in metadata, "Unjustified OpenAI settlement")
        else:
            require(len(settled) == 1 and (settled[0]["charged_nano"], settled[0]["evidence"]) == expected
                    and metadata.get("settled_conservative_usd") == budget.usd_string(expected[0]), "OpenAI settlement differs from exact usage evidence")
        if action["item"]["provider"] == "openai":
            bound = expected_reservation(action, plan)["bound"]
            require((prediction["input_tokens"] is None or prediction["input_tokens"] <= bound["input_token_bound"])
                    and (prediction["output_tokens"] is None or prediction["output_tokens"] <= bound["output_token_cap"]),
                    "Saved OpenAI usage exceeds the priced request bounds")
    require(not ledger.snapshot()["halted"], "Retry ledger halted after a usage/route violation")
    ledger.saved, ledger.pending = records, None
    ledger.remember()
    return records


class LiveVerifier:
    def __init__(self, plan, session):
        self.plan, self.session, self.receipt, self.credit_count = plan, session, None, 0
        self.transport = completion.VerifiedTransport(sys.modules[__name__], file_sha(__file__), execute=True)
        self.transport.area = session

    def check(self):
        check_plan(self.plan)
        require(file_sha(AREA / "plan.json") == hashlib.sha256((json.dumps(self.plan, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest(),
                "Frozen retry plan changed")
        self.transport.check()
        # Verify the custom declaration's date on every reservation, including a
        # Jev call that shares this mixed-provider allocation.
        for item in self.plan["inventory"]["failures"]:
            if item["provider"] == "openai":
                request = preparation.reconstruct_request(item)
                current = budget.select_price(budget.validate_config(request["config"]), json.loads(PRICES.read_text()))
                require(current == price_for(request), "Fresh Astra pricing differs from the frozen declaration")
        if self.receipt is None or (datetime.now(timezone.utc) - datetime.fromisoformat(self.receipt["checked_at"])).total_seconds() > 45:
            self.receipt = funding.check_credit(os.environ.get("OPENROUTER_API_KEY", ""))
            require(Decimal(self.receipt["available_credit_usd"]) >= Decimal(route.PRICE["per_request_reserved_usd"]), "Available credit cannot fund one full Jev reservation")
            self.credit_count += 1
            completion.probe.write_exclusive(self.session / f"funding-{self.credit_count:03d}.json", funding.public_receipt(self.receipt))
        funding.verify_receipt(self.receipt)


def final_counts(plan, records):
    by_failure = {}
    for record in records:
        fid = record["prediction"]["metadata"]["retry"]["failure_id"]
        by_failure.setdefault(fid, []).append(record)
    return {"planned_failures": len(plan["inventory"]["failures"]), "new_calls": len(records),
        "recovered": sum(any(not r["prediction"]["error"] for r in values) for values in by_failure.values()),
        "unrecovered": sum(not any(not r["prediction"]["error"] for r in by_failure.get(item["failure_id"], []))
                           for item in plan["inventory"]["failures"]),
        "upstream_unresolved_not_called": sum(item["kind"] == "upstream_skip" and item["failure_id"] not in by_failure
                                             for item in plan["inventory"]["failures"])}


def halt_reason(records):
    if records and records[-1]["prediction"]["error"] and "status=402" in records[-1]["prediction"]["error"]:
        return "halted_billing"
    if len(records) >= 3 and all(r["prediction"]["error"] for r in records[-3:]):
        return "halted_errors"
    return None


def execute(plan, *, initialize=False, prompt_key=False, stop_after=None, resume_after_errors=False):
    check_plan(plan)
    plan_path, run_path, attempts_path = (AREA / name for name in ("plan.json", "run.json", "attempts.jsonl"))
    existing = json.loads(run_path.read_text()) if run_path.exists() else None
    if existing:
        require(not initialize and json.loads(plan_path.read_text()) == plan, "Existing retry plan cannot be replaced or reset")
        if existing.get("status") == "complete":
            return audit_run()["record"]
        require(existing["status"] not in {"halted_billing", "halted_errors"} or resume_after_errors,
                "Explicit --resume-after-errors and fresh funding required after a service halt")
    else:
        require(initialize and not any((AREA / name).exists() for name in ("plan.json", "attempts.jsonl", "budget.jsonl", "budget.jsonl.lock")),
                "Initialize once; existing partial/orphan artifacts require reconciliation")
    configs = [preparation.reconstruct_request(item)["config"] for item in plan["inventory"]["failures"] if item["kind"] == "paid_failure"]
    if prompt_key:
        budget.prompt_credentials(configs)
    require(all(os.environ.get(config["api_key_env"]) for config in configs), "Missing provider credential")
    # Funding and public reads precede initialization. Only the safe assertion is saved.
    initial_credit = funding.check_credit(os.environ.get("OPENROUTER_API_KEY", ""))
    require(Decimal(initial_credit["available_credit_usd"]) >= Decimal(route.PRICE["per_request_reserved_usd"]), "Insufficient available Jev credit")
    transport = completion.VerifiedTransport(sys.modules[__name__], file_sha(__file__), execute=True)
    transport.check()
    if not existing:
        atomic_json(plan_path, plan)
    inner = budget.Ledger(AREA / "budget.jsonl", plan["allocation_usd"], initialize=initialize)
    record = existing or {"schema_version": 1, "study": plan["study"], "ledger_id": inner.identity["ledger_id"],
                           "plan_sha256": file_sha(plan_path), "execution_sessions": []}
    require(record["ledger_id"] == inner.identity["ledger_id"] and record["plan_sha256"] == file_sha(plan_path), "Retry execution identity differs")
    session = AREA / "sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12])
    session.mkdir(parents=True, exist_ok=False)
    verifier = LiveVerifier(plan, session)
    verifier.transport.receipts = transport.receipts
    for index, receipt in enumerate(transport.receipts, 1):
        completion.probe.write_exclusive(session / f"public-verification-{index:03d}.json", receipt)
    verifier.receipt = initial_credit
    completion.probe.write_exclusive(session / "funding-initial.json", funding.public_receipt(initial_credit))
    ledger = RetryLedger(inner, plan, verifier.check, stop_after=stop_after)
    records = audit_saved(plan, ledger, attempts_path)
    require(halt_reason(records) is None or resume_after_errors,
            "Audited attempts contain a service halt; explicit --resume-after-errors is required")
    ledger.allow_halt_tail_once = bool(resume_after_errors)
    record["execution_sessions"].append({"started_at": budget.utc_now(), "saved_attempts": len(records),
                                        "path": session.relative_to(ROOT).as_posix(), "producer_pins": pins(),
                                        "resume_after_errors": resume_after_errors})
    record["status"] = "running"
    atomic_json(run_path, record)
    try:
        with attempts_path.open("a") as stream:
            while True:
                action, _ = walk_policy(plan, records)
                if action is None:
                    break
                ledger.active = action
                request = action["request"]
                config = request["config"]
                if config["provider"] == "jev":
                    provider = route.GuardedOpenRouterJev(config, ledger, provenance(action, plan))
                else:
                    config = budget.validate_config(config)
                    provider = budget.BudgetedProvider(providers.build_provider(config), config, price_for(request), ledger, provenance(action, plan))
                prediction = provider.predict(Row(request["row_id"], "", -1), request["labels"], request["prompt"])
                prediction.metadata["retry"] = action["metadata"]
                saved = {"attempt_index": action["attempt_index"], "identity": action["identity"], "stage": action["stage"], "prediction": asdict(prediction)}
                require(ledger.pending is not None, "Provider returned without a reserved HTTP attempt")
                analysis.validate_prediction(saved["prediction"], request["row_id"], len(request["labels"]), require_probabilities=config["provider"] == "jev")
                stream.write(json.dumps(saved, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
                records = audit_saved(plan, ledger, attempts_path)
                print(json.dumps({"saved_attempts": len(records), "phase": action["metadata"]["phase"], "failed": bool(prediction.error), "budget": ledger.snapshot()}), flush=True)
                if halt_reason(records) and walk_policy(plan, records)[0] is not None:
                    record["status"] = halt_reason(records); break
        if record["status"] == "running":
            require(walk_policy(plan, records)[0] is None, "Finite policy is incomplete")
            record["status"] = "complete"
    except route.StopRequested:
        record["status"] = "paused"
    except budget.BudgetStop:
        record["status"] = "budget_exhausted"
        raise
    except BaseException:
        record["status"] = "interrupted"
        raise
    finally:
        record.update(budget=ledger.snapshot(), counts=final_counts(plan, records), finished_at=budget.utc_now())
        if record["status"] == "complete":
            audit_saved(plan, ledger, attempts_path)
            record["protected_artifacts_sha256"] = {name: file_sha(AREA / name) for name in
                ("plan.json", "attempts.jsonl", "budget.jsonl", "budget.jsonl.lock")}
        atomic_json(run_path, record)
    return record


def audit_run(area=None):
    """Read-only final evidence audit; usable after later controls are allocated.

    Returns plan, original run record, append-ordered attempts and audited budget.
    No public verification, credentials, tokenizer/model loads or API calls.
    """
    area = AREA if area is None else Path(area)
    require(area.resolve() == AREA.resolve() and not area.is_symlink(), "Retry audit requires the canonical evidence area")
    names = ("run.json", "plan.json", "attempts.jsonl", "budget.jsonl", "budget.jsonl.lock")
    require(all((area / name).is_file() and not (area / name).is_symlink() for name in names), "Finalized retry evidence is incomplete")
    before = {name: file_sha(area / name) for name in names}
    record, plan = (json.loads((area / name).read_text()) for name in ("run.json", "plan.json"))
    require(record.get("status") == "complete", "Finite recovery policy has not completed")
    check_plan(plan, read_only=True)
    require(record["plan_sha256"] == before["plan.json"] and record["protected_artifacts_sha256"] ==
            {name: before[name] for name in names if name != "run.json"}, "Finalized recovery files changed")
    ledger = RetryLedger(budget.Ledger(area / "budget.jsonl", plan["allocation_usd"]), plan, lambda: None)
    attempts = audit_saved(plan, ledger, area / "attempts.jsonl")
    require(walk_policy(plan, attempts)[0] is None and record["counts"] == final_counts(plan, attempts),
            "Final report does not match the completed finite policy")
    state = ledger.snapshot()
    require(record["ledger_id"] == state["ledger_id"] and record["budget"] == state
            and all(file_sha(area / name) == before[name] for name in names), "Final retry accounting changed")
    return {"plan": plan, "record": record, "attempts": attempts, "budget": state,
            "artifact_sha256": before}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--init-ledger", action="store_true")
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--resume-after-errors", action="store_true")
    parser.add_argument("--stop-after-new-requests", type=int)
    args = parser.parse_args(argv)
    require(args.stop_after_new_requests is None or args.stop_after_new_requests > 0, "Invalid checkpoint request count")
    require(args.execute or not (args.init_ledger or args.prompt_api_key or args.resume_after_errors), "Mutation/credential flags require --execute")
    prior = prior_state()
    if not args.execute:
        plan = make_plan(prior) if prior["ready"] else None
        result = {"mode": "dry_run_no_writes", "ready": prior["ready"], "numeric_complete": prior["numeric_complete"],
                  "text_complete": prior["text_complete"], "prior_conservative_usd": prior["prior_conservative_usd"],
                  "allocation_usd": plan["allocation_usd"] if plan else None,
                  "failures": plan["inventory"]["counts"] if plan else None}
        print(json.dumps(result, indent=2)); return result
    require(prior["ready"], "Finish both first-pass matrices before paid retries")
    AREA.mkdir(parents=True, exist_ok=True)
    require(not AREA.is_symlink(), "Retry directory must be canonical")
    with ExitStack() as stack:
        for path in (historical.numeric.LEDGER.parent / "review-execution.lock", historical.LEDGER.parent / "review-execution.lock", AREA / "execution.lock"):
            require(not path.is_symlink(), "Execution lock must be canonical")
            lock = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise budget.GuardError("A first-pass or retry execution is active") from None
        plan_path = AREA / "plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else make_plan(prior_state())
        return execute(plan, initialize=args.init_ledger, prompt_key=args.prompt_api_key,
                       stop_after=args.stop_after_new_requests, resume_after_errors=args.resume_after_errors)


if __name__ == "__main__":
    main()
