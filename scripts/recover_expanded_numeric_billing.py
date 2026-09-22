"""Resume one audited HTTP-402 halt after a fresh positive account-credit check.

Dry-run is read-only and never reads credentials. Execute preserves an immutable
halt snapshot and funding receipt before changing only mutable execution state.
The frozen producer performs inference; only its exact acknowledged billing-stop
triplet receives one reservation exemption. No failed prediction is retried.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_expanded_numeric_review as review
import audit_expanded_numeric_costs as costs
from jevbench.providers import _NoRedirect

CREDITS_URL = "https://openrouter.ai/api/v1/credits"
HALTED = "stopped_after_three_consecutive_errors"
BILLING_ERROR = "http_error: status=402; no automatic retry"
RECEIPT_MAX_AGE_SECONDS = 60


def require(condition, message):
    if not condition:
        raise review.budget.GuardError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def snapshot_ledger(ledger):
    with ledger.lock_path.open("r") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            events, _ = ledger._read()
            return events, {"ledger_sha256": review.file_sha(ledger.path),
                "ledger_lock_sha256": review.file_sha(ledger.lock_path),
                "ledger_head_sha256": events[-1]["event_sha256"], "event_count": len(events)}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class CheckpointView:
    """Read-only view for the frozen validator, with only halt status masked.

    This does not mutate the saved run. All schema, source, prompt, ordered-prefix
    and paid-call validations still run through the original frozen function.
    """
    def __init__(self, directory, record):
        self.directory, self.record = directory, record

    def __truediv__(self, name):
        path, record = self.directory / name, self.record

        class ReadPath:
            def exists(self):
                return path.exists()

            def read_text(self):
                return json.dumps({**record, "status": "running"}) if name == "run.json" else path.read_text()

        return ReadPath()


def billing_tail(events, predictions):
    require(len(predictions) >= 3 and all(p.error == BILLING_ERROR for p in predictions[-3:]),
            "Only an exact final triplet of HTTP 402 failures is eligible")
    results = [e for e in events if e["type"] == "result"]
    reserves = {e["reservation_id"]: e for e in events if e["type"] == "reserve"}
    require(len(results) >= 3, "Billing halt lacks durable result events")
    ids = [p.metadata["budget"]["reservation_id"] for p in predictions[-3:]]
    require(len(set(ids)) == 3 and [e["reservation_id"] for e in results[-3:]] == ids,
            "Billing failures are not the exact current global result tail")
    require(events[-1] == results[-1], "Unfinished reservation follows the billing halt")
    for event in results[-3:]:
        details = event["details"]
        require(event["reservation_id"] in reserves and details.get("outcome") == "prediction_error"
                and details.get("route") == "OpenRouter" and details.get("reservation_released") is False
                and not details.get("usage_exceeded_reservation_assumptions"),
                "Billing halt result or budget guard is inconsistent")
    return {"reservation_ids": ids, "result_event_sha256": [e["event_sha256"] for e in results[-3:]],
        "reservation_event_sha256": [reserves[i]["event_sha256"] for i in ids]}


def protected_hashes(job, directory):
    paths = {path for path in review.OUTPUT.rglob("*") if path.is_file()}
    paths.update(review.SOURCES_DIR.glob("*.json"))
    paths.update(review.ROOT / job["source"]["path"] / name for name in review.FILE_NAMES)
    paths.update({review.LEDGER, review.LEDGER.with_name(review.LEDGER.name + ".lock")})
    for relative in review.PRIOR:
        path = review.ROOT / relative
        paths.update({path, path.with_name(path.name + ".lock")})
    for path in paths:
        require(not path.is_symlink() and path.resolve().is_relative_to(review.ROOT.resolve()), "Protected evidence path must be canonical")
    return {path.relative_to(review.ROOT).as_posix(): review.file_sha(path) for path in sorted(paths)}


def audit_halt(dataset_name, model_key, shots):
    review.verify_transport()
    review.verify_prior()
    require(not review.is_reused(model_key, shots), "Prior review arms are immutable")
    slug = review.condition_key(dataset_name, model_key, shots) + "__jev-review"
    directory = review.OUTPUT / slug
    require(directory.is_dir() and not directory.is_symlink(), "Canonical halted checkpoint required")
    record_raw = (directory / "run.json").read_bytes()
    record = json.loads(record_raw)
    require(record.get("status") == HALTED and record.get("run_id") == slug, "Condition is not a halted canonical run")
    dataset = review.load_native_prepared(review.ROOT / "data/tabular-full" / dataset_name)[0]
    job = review.load_source(dataset, model_key, shots)
    require(job is not None and job["manifest_sha256"] is not None, "Individually frozen source required")
    config = review.route.validate_config(json.loads((review.ROOT / "configs/tabular_hosted.json").read_text())["jev_openrouter"])
    inner = review.budget.Ledger(review.LEDGER, review.STAGE_CAP)
    identity, guard = review.make_identity(job, config, inner.identity["ledger_id"], record["bootstrap_samples"])
    _, predictions = review.base.load_checkpoint(CheckpointView(directory, record), identity,
        dataset, job["source"], job["record"], job["predictions"], job["examples"])
    require(record.get("n_predictions") == len(predictions) and 3 <= len(predictions) < len(dataset.test),
            "Halt row count must be an incomplete exact prefix")
    ledger = review.AnchoredLedger(inner, root=review.ROOT)
    review.base.validate_reservations(ledger, dataset_name, guard["proposal_key"], predictions, identity)
    accounting = costs.collect_costs([])
    require(accounting["status"] == "halted" and all(accounting[name] == 0 for name in
            ("reservations_without_result", "results_without_saved_prediction", "partial_checkpoint_files")),
            "All existing requests and checkpoints must reconcile before recovery")
    events, anchor = snapshot_ledger(inner)
    require(accounting["ledger_sha256"] == anchor["ledger_sha256"], "Ledger changed during the evidence audit")
    tail = billing_tail(events, predictions)
    require(review.base.valid_proposal(job["predictions"][len(predictions)], len(dataset.labels)),
            "The next unseen row has no valid source proposal; investigate without bypassing the failure stop")
    remaining = sum(review.base.valid_proposal(p, len(dataset.labels)) for p in job["predictions"][len(predictions):])
    require(accounting["new_model_requests"] + remaining <= review.MAX_REQUESTS, "Recovery would exceed the fixed request limit")
    require(Decimal(accounting["review_conservative_usd"]) + Decimal(review.route.PRICE["per_request_reserved_usd"])*remaining <= Decimal(review.STAGE_CAP),
            "Recovery would exceed the original stage allocation")
    return {"job": job, "config": config, "inner": inner, "directory": directory, "record": record,
        "record_raw": record_raw, "prediction_raw": (directory / "predictions.jsonl").read_bytes(),
        "ledger_raw": inner.path.read_bytes(),
        "predictions": predictions, "guard": guard, "anchor": anchor, "tail": tail, "accounting": accounting,
        "protected": protected_hashes(job, directory), "remaining_calls": remaining}


def parse_credit_response(payload, checked_at):
    require(isinstance(payload, dict) and isinstance(payload.get("data"), dict), "Credit response schema is invalid")
    values = {}
    for name in ("total_credits", "total_usage"):
        value = payload["data"].get(name)
        require(type(value) in (int, float, str, Decimal), "Credit balance must be numeric")
        try:
            value = Decimal(str(value))
        except InvalidOperation:
            raise review.budget.GuardError("Credit balance is invalid") from None
        require(value.is_finite() and value >= 0, "Credit balance must be finite and nonnegative")
        values[name] = value
    available = values["total_credits"] - values["total_usage"]
    require(available > 0, "Account available credit is not positive; no evidence or status changed")
    return {"endpoint": CREDITS_URL, "method": "GET", "authenticated": True, "http_status": 200,
        "checked_at": checked_at, "total_credits_usd": str(values["total_credits"]),
        "total_usage_usd": str(values["total_usage"]), "available_credit_usd": str(available)}


def check_credit(api_key):
    require(isinstance(api_key, str) and bool(api_key.strip()), "Missing OpenRouter credential")
    request = urllib.request.Request(CREDITS_URL, headers={"Authorization": "Bearer " + api_key,
        "Accept": "application/json"}, method="GET")
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=15) as response:
            require(response.status == 200 and response.geturl() == CREDITS_URL, "Credit response route/status differs")
            raw = response.read(65537)
            require(len(raw) <= 65536, "Credit response exceeds the expected size")
            payload = json.loads(raw, parse_float=Decimal)
    except urllib.error.HTTPError as error:
        raise review.budget.GuardError(f"Credit check returned HTTP {error.code}; no changes made") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, UnicodeError):
        raise review.budget.GuardError("Credit check failed; no retry and no changes made") from None
    return parse_credit_response(payload, review.budget.utc_now())


def verify_receipt(receipt):
    require(receipt.get("endpoint") == CREDITS_URL and receipt.get("method") == "GET" and
            receipt.get("authenticated") is True and receipt.get("http_status") == 200, "Authenticated credit receipt required")
    parsed = parse_credit_response({"data": {"total_credits": receipt.get("total_credits_usd"),
        "total_usage": receipt.get("total_usage_usd")}}, receipt.get("checked_at"))
    require(parsed == receipt, "Credit receipt contains unexpected or inconsistent fields")
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(receipt["checked_at"])).total_seconds()
    except (ValueError, TypeError):
        raise review.budget.GuardError("Credit receipt timestamp is invalid") from None
    require(0 <= age <= RECEIPT_MAX_AGE_SECONDS, "Credit check is stale; check again before recovery")


def public_receipt(receipt):
    """Persist the funding assertion, never account-wide balances or usage."""
    verify_receipt(receipt)
    return {**{key: receipt[key] for key in ("endpoint", "method", "authenticated", "http_status", "checked_at")},
        "positive_available_credit": True}


class BillingRecoveryLedger(review.AnchoredLedger):
    """One exact billing-tail exemption; every later reservation uses frozen guard."""
    def __init__(self, inner, audit, receipt, *, stop_after=None):
        super().__init__(inner, root=review.ROOT, stop_after=stop_after)
        self.audit, self.receipt, self.exemption_used = audit, receipt, False

    def reserve(self, amount, details):
        if self.exemption_used:
            return super().reserve(amount, details)
        review.verify_transport()
        review.verify_prior(self.root)
        verify_receipt(self.receipt)
        events, anchor = snapshot_ledger(self.inner)
        require(anchor == self.audit["anchor"] and billing_tail(events, self.audit["predictions"]) == self.audit["tail"],
                "Acknowledged billing tail or durable ledger head changed")
        require(amount == review.route.RESERVE_NANO, "Unexpected Jev reservation")
        reservations = [event for event in events if event["type"] == "reserve"]
        if len(reservations) >= review.MAX_REQUESTS:
            raise review.budget.BudgetStop("Expansion is limited to 1500 new Jev calls")
        key = details.get("dataset"), details.get("proposal_key"), details.get("row_id")
        require(not any((e["details"].get("dataset"), e["details"].get("proposal_key"), e["details"].get("row_id")) == key for e in reservations),
                "This expanded review row was already reserved; no automatic retry")
        job = self.audit["job"]
        next_row = job["dataset"].test[len(self.audit["predictions"])]
        require(key == (job["dataset"].name, f"{job['model_key']}_k{job['shots']}", next_row.id),
                "Billing exemption applies only to the next unseen row of the halted condition")
        next_proposal = job["predictions"][len(self.audit["predictions"])]
        prompt = review.base.build_review_prompt(next_row, job["dataset"].labels, job["examples"], next_proposal.label)
        expected_request = {**self.audit["guard"], "row_id": next_row.id, "provider": "jev",
            "model": review.route.MODEL, "pricing": review.route.PRICE, "prompt_sha256": sha(prompt.encode())}
        require(all(details.get(field) == value for field, value in expected_request.items()),
                "Recovery request differs from frozen source/prompt/model/budget provenance")
        if self.stop_after is not None and self.new_requests >= self.stop_after:
            raise review.route.StopRequested("Requested checkpoint reached before another API call")
        ident = self.inner.reserve(amount, details)  # Original cap, chain and routing/usage halt checks.
        self.exemption_used = True
        self.new_requests += 1
        return ident


def write_exclusive(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def prepare_recovery(audit, receipt):
    verify_receipt(receipt)
    require(protected_hashes(audit["job"], audit["directory"]) == audit["protected"], "Evidence changed after the credit check")
    require(snapshot_ledger(audit["inner"])[1] == audit["anchor"], "Ledger changed after the credit check")
    helper_sha = review.file_sha(__file__)
    recovery_id = audit["directory"].name + "__" + sha(audit["record_raw"])[:16]
    destination = review.AREA / "billing_recovery" / recovery_id
    require(not destination.exists(), "Recovery record already exists; preserve it and investigate before another attempt")
    record = deepcopy(audit["record"])
    metadata = {"recovery_id": recovery_id, "helper_sha256": helper_sha,
        "manifest_path": (destination / "manifest.json").relative_to(review.ROOT).as_posix(),
        "halted_run_sha256": sha(audit["record_raw"]), "credit_receipt": public_receipt(receipt),
        "acknowledged_billing_tail": audit["tail"], "ledger_anchor_before": audit["anchor"]}
    record.setdefault("execution", {}).setdefault("billing_recoveries", []).append(metadata)
    record["status"] = "running"
    # This field is written only on a halt by the frozen producer; retaining it
    # would leave a stale 85-row count in a later complete 114-row run. The exact
    # count remains in the immutable snapshot and recovery manifest below.
    record.pop("n_predictions", None)
    after = json_bytes(record)
    manifest = {"schema_version": 1, **metadata, "created_at": review.budget.utc_now(),
        "reason": "Fresh authenticated positive account credit after exactly three HTTP 402 failures; failed rows remain failures and are never retried.",
        "before_run_sha256": sha(audit["record_raw"]), "cleared_run_sha256": sha(after),
        "protected_before_sha256": audit["protected"], "predictions_retained": len(audit["predictions"]),
        "maximum_remaining_calls_in_condition": audit["remaining_calls"],
        "stage_cap_usd": review.STAGE_CAP, "cumulative_cap_usd": review.TOTAL_CAP,
        "stop_behavior": "One next-row exemption only. The frozen producer still stops if the first resumed prediction fails; later three-error stops are unchanged."}
    destination.mkdir(parents=True, exist_ok=False)
    write_exclusive(destination / "halted-run.json", audit["record_raw"])
    write_exclusive(destination / "manifest.json", json_bytes(manifest))
    review.save_json(audit["directory"] / "run.json", record)
    require((audit["directory"] / "run.json").read_bytes() == after, "Unexpected checkpoint serialization")
    expected = dict(audit["protected"])
    expected[(audit["directory"] / "run.json").relative_to(review.ROOT).as_posix()] = sha(after)
    require(protected_hashes(audit["job"], audit["directory"]) == expected, "Recovery modified protected evidence")
    return destination, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=review.DATASETS, required=True)
    parser.add_argument("--model-key", choices=review.MODELS, required=True)
    parser.add_argument("--shots", choices=review.SHOTS, type=int, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--stop-after-new-requests", type=int)
    args = parser.parse_args(argv)
    require(args.stop_after_new_requests is None or args.stop_after_new_requests > 0, "Checkpoint limit must be positive")
    lock_path = review.LEDGER.with_name("review-execution.lock")
    require(lock_path.is_file() and not lock_path.is_symlink(), "Existing canonical execution lock required")
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise review.budget.GuardError("Another expanded review worker is running") from None
        audit = audit_halt(args.dataset, args.model_key, args.shots)
        plan = {"status": "eligible_billing_halt", "condition": audit["directory"].name,
            "preserved_predictions": len(audit["predictions"]), "remaining_calls": audit["remaining_calls"],
            "ledger_anchor": audit["anchor"], "acknowledged_tail": audit["tail"],
            "stage_cap_usd": review.STAGE_CAP, "cumulative_cap_usd": review.TOTAL_CAP,
            "execute": args.execute, "requires_fresh_positive_account_credit": True}
        print(json.dumps(plan, allow_nan=False), flush=True)
        if not args.execute:
            return plan
        if args.prompt_api_key:
            review.budget.prompt_credentials([audit["config"]])
        receipt = check_credit(os.environ.get(audit["config"]["api_key_env"]))
        destination, manifest = prepare_recovery(audit, receipt)
        ledger = BillingRecoveryLedger(audit["inner"], audit, receipt, stop_after=args.stop_after_new_requests)
        status, exception_type = "running", None
        try:
            record = review.run_review(audit["job"], audit["config"], ledger, bootstrap_samples=audit["record"]["bootstrap_samples"])
            status = record["status"]
        except review.route.StopRequested:
            status = "checkpoint"
        except BaseException as exc:
            status, exception_type = "stopped", type(exc).__name__
            raise
        finally:
            require((audit["directory"] / "predictions.jsonl").read_bytes().startswith(audit["prediction_raw"]),
                    "Original predictions were modified during resume")
            require(audit["inner"].path.read_bytes().startswith(audit["ledger_raw"]), "Historical ledger events were modified during resume")
            after_hashes = protected_hashes(audit["job"], audit["directory"])
            mutable = {(audit["directory"] / name).relative_to(review.ROOT).as_posix()
                for name in ("run.json", "predictions.jsonl")}
            mutable.update(path.relative_to(review.ROOT).as_posix() for path in (audit["inner"].path, audit["inner"].lock_path))
            require(all(after_hashes.get(path) == before for path, before in audit["protected"].items() if path not in mutable),
                    "Source, manifest, prior ledger, or another condition changed during recovery")
            _, anchor = snapshot_ledger(audit["inner"])
            completion = {"schema_version": 1, "status": status, "exception_type": exception_type,
                "completed_at": review.budget.utc_now(), "manifest_sha256": review.file_sha(destination / "manifest.json"),
                "new_calls": ledger.new_requests, "one_time_exemption_used": ledger.exemption_used,
                "run_sha256_after": review.file_sha(audit["directory"] / "run.json"),
                "predictions_sha256_after": review.file_sha(audit["directory"] / "predictions.jsonl"),
                "ledger_anchor_after": anchor, "protected_after_sha256": after_hashes}
            write_exclusive(destination / "completion.json", json_bytes(completion))
        print(json.dumps(completion, allow_nan=False), flush=True)
        return completion


if __name__ == "__main__":
    try:
        main()
    except (review.budget.GuardError, review.budget.BudgetStop) as exc:
        raise SystemExit(f"STOP: {exc}") from None
