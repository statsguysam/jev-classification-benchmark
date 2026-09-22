"""Review frozen SST-2/TREC class proposals with native Jev Choice.

Dry-run by default. This new experiment has a $1.60 allocation inside the
existing $25 cumulative ceiling, preserving room for every unfinished numeric
review. Historical reservations are never released. New successful requests
with complete, verified usage/cost can settle once at a conservative amount.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_numeric_jev_review as base
import run_expanded_numeric_review as numeric
import text_extension_sources as sources
from recover_expanded_numeric_billing import check_credit, public_receipt, CheckpointView
from jevbench.metrics import evaluate
from jevbench.prompts import build_prompt
from jevbench.runner import _identity, digest, environment, finish_run, save_json
from jevbench.types import Prediction

route, budget = base.jev_route, base.budget
AREA = ROOT / "results/text_extension"
OUTPUT, LEDGER = AREA / "review", AREA / "review-budget.jsonl"
STAGE_CAP, TOTAL_CAP = "1.60", "25.00"
FIXED_PRIOR = Decimal("19.325827700")
NUMERIC_RESERVED_ALLOWANCE = Decimal("4.032000000")
MAX_REQUESTS = 4800
SETTLEMENT_VERSION = "verified-success-max-reported-or-token-cost-times-1.25-v1"
REGISTRY = ROOT / "configs/text_extension_sources.json"
FROZEN_NUMERIC_SHA = "68a16db205cceb81b46916630c776827ccdb09e7df6aed60dd1db7413d4d7a88"
FROZEN_CREDIT_HELPER_SHA = "045abae3f02797e921d5eef4758e95e246922e39a03a69bddb65a0f1882fc416"
NUMERIC_PREFIX_BYTES = 4651252
NUMERIC_PREFIX_SHA = "9ae26c11d837440425730abefae9354f0bdd5d61284232c33877633e2a72f496"
NUMERIC_PREFIX_EVENTS = 2199
NUMERIC_PREFIX_HEAD = "4f74841b0029e51586ca3ff23eb774549b38b39795a76e757138ad68427d079c"
NUMERIC_LEDGER_ID = "c3b3c91e-8194-48e7-8101-64d96479da49"
HALTED = {"stopped_after_billing_error", "stopped_after_three_consecutive_errors"}
BILLING_ERROR = "http_error: status=402; no automatic retry"


def require(condition, message):
    if not condition:
        raise budget.GuardError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_envelope(root=ROOT):
    """Keep the earlier numeric allocation available without rewriting its ledger."""
    verify_frozen_helpers()
    numeric.verify_prior(root)
    path = root / "results/numeric_expansion/review-budget.jsonl"
    ledger = budget.Ledger(path, numeric.STAGE_CAP)
    snapshot = ledger.snapshot()
    require(not snapshot["halted"], "Earlier numeric ledger has an accounting/routing halt")
    events, _ = ledger._read()
    with path.open("rb") as stream:
        prefix = stream.read(NUMERIC_PREFIX_BYTES)
    require(len(prefix) == NUMERIC_PREFIX_BYTES and hashlib.sha256(prefix).hexdigest() == NUMERIC_PREFIX_SHA
            and ledger.identity["ledger_id"] == NUMERIC_LEDGER_ID and len(events) >= NUMERIC_PREFIX_EVENTS
            and events[NUMERIC_PREFIX_EVENTS - 1]["event_sha256"] == NUMERIC_PREFIX_HEAD,
            "Earlier numeric ledger prefix changed or was replaced")
    reserves = [e for e in events if e["type"] == "reserve"]
    require(len(reserves) <= numeric.MAX_REQUESTS and not any(e["type"] == "settle" for e in events),
            "Earlier numeric reservation policy changed")
    require(all(e["reserved_nano"] == route.RESERVE_NANO for e in reserves), "Earlier numeric reservation differs")
    require(Decimal(snapshot["charged_or_reserved_usd"]) <= NUMERIC_RESERVED_ALLOWANCE,
            "Earlier numerical study exceeds its reserved envelope")
    require(FIXED_PRIOR + NUMERIC_RESERVED_ALLOWANCE + Decimal(STAGE_CAP) <= Decimal(TOTAL_CAP),
            "Combined current and unfinished experiments exceed authorized spending")
    return snapshot


def verify_frozen_helpers():
    require(file_sha(numeric.__file__) == FROZEN_NUMERIC_SHA and
            file_sha(Path(__file__).with_name("recover_expanded_numeric_billing.py")) == FROZEN_CREDIT_HELPER_SHA,
            "Frozen imported numeric or credit helper changed")
    require(file_sha(base.__file__) == numeric.BASE_REVIEW_SHA and
            file_sha(route.__file__) == base.FROZEN_ROUTE_SHA and
            file_sha(route._BUDGET_PATH) == route.FROZEN_BUDGET_SHA256 and
            environment()["source_sha256"] == base.FROZEN_CORE_SHA, "Frozen text transport/core changed")


def verify_transport():
    verify_frozen_helpers()
    numeric.verify_transport()


def build_review_prompt(row, labels, examples, proposal_label):
    require(type(proposal_label) is int and 0 <= proposal_label < len(labels), "Invalid proposed class")
    prompt = build_prompt(row, labels, examples).removesuffix("\nClass ID:")
    return (prompt + "\nCached proposal from a separate model: class ID " + str(proposal_label) + "."
        "\nThe proposal may be incorrect. Classify the original item in Final text. "
        "Use that item's content and the labeled examples above to make the final decision. "
        "You may keep or override the proposal. Return only the numeric class ID.\nClass ID:")


def settlement_amount(details):
    """Only complete successful responses justify releasing new reservations."""
    usage = details.get("usage", {})
    if not isinstance(usage, dict):
        return None
    if (details.get("outcome") != "returned" or details.get("usage_exceeded_reservation_assumptions")
            or details.get("provider") != "TypeSafe" or details.get("resolved_model") not in route.ALLOWED_RESPONSE_MODELS
            or details.get("route") != "OpenRouter" or not details.get("request_id")
            or type(usage.get("input_tokens")) is not int or not 0 < usage["input_tokens"] <= route.PRICE["reserve_input_tokens"]
            or type(usage.get("output_tokens")) is not int or usage["output_tokens"] < 0):
        return None
    try:
        cost = route.parse_cost(details.get("reported_cost_usd"))
    except budget.GuardError:
        return None
    if cost is None or Decimal(cost) > Decimal(route.PRICE["per_request_reserved_usd"]):
        return None
    token_cost = Decimal(usage["input_tokens"]) * Decimal(route.PRICE["input_usd_per_million"]) / Decimal(1_000_000)
    conservative = max(Decimal(cost), token_cost) * Decimal("1.25")
    amount = int((conservative * Decimal(1_000_000_000)).to_integral_value(rounding=ROUND_CEILING))
    return min(amount, route.RESERVE_NANO)


def settlement_evidence(details):
    return {"settlement_version": SETTLEMENT_VERSION, "usage": details["usage"],
        "reported_cost_usd": details["reported_cost_usd"], "pricing_sha256": digest(route.PRICE),
        "multiplier": "1.25", "basis": "maximum of reported USD and uncached input-token price; capped at original reservation"}


class TextLedger:
    def __init__(self, inner, *, stop_after=None):
        self.inner, self.identity = inner, inner.identity
        self.stop_after, self.new_requests = stop_after, 0
        self.trusted_head, self.pending_id = None, None
        self.active_job, self.active_identity, self.active_index = None, None, None

    def remember_head(self):
        self.trusted_head = self.inner._read()[0][-1]["event_sha256"]

    def check_head(self):
        require(self.trusted_head is not None and self.inner._read()[0][-1]["event_sha256"] == self.trusted_head,
                "Unaudited or externally changed text ledger; full global audit required")

    def prediction_saved(self, prediction):
        """Called only after the durable prediction append; an interruption stays pending."""
        self.check_head()
        require(prediction.metadata.get("budget", {}).get("reservation_id") == self.pending_id,
                "Durable prediction does not close the pending request")
        validate_accounting(self, [prediction])
        self.pending_id = None

    def reserve(self, amount, details):
        verify_transport()
        verify_envelope()
        self.check_head()
        require(self.pending_id is None, "Previous request lacks a durable verified prediction")
        require(self.active_job is not None and self.active_identity is not None, "No audited active review condition")
        guard = self.active_identity["config"]["budget_guard"]
        require(all(details.get(key) == value for key, value in guard.items()), "Request differs from active condition identity")
        require(type(self.active_index) is int and 0 <= self.active_index < len(self.active_job["dataset"].test),
                "Request lacks the exact next source row")
        row = self.active_job["dataset"].test[self.active_index]
        proposal = self.active_job["predictions"][self.active_index]
        require(base.valid_proposal(proposal, len(self.active_job["dataset"].labels)), "Cannot review a failed proposal")
        prompt = build_review_prompt(row, self.active_job["dataset"].labels, self.active_job["examples"], proposal.label)
        require(details.get("row_id") == row.id and details.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest()
                and details.get("provider") == "jev" and details.get("model") == route.MODEL
                and details.get("pricing") == route.PRICE, "Request row/prompt differs from the frozen cached proposal")
        require(amount == route.RESERVE_NANO, "Unexpected request reservation")
        events, _ = self.inner._read()
        require(not any(json.loads(path.read_text()).get("status") in {
            "stopped_after_billing_error", "stopped_after_three_consecutive_errors"}
            for path in OUTPUT.glob("*/run.json")), "A text review checkpoint is halted; investigate before another request")
        results = [e for e in events if e["type"] == "result"]
        require(len(results) < 3 or not all(e["details"].get("outcome") in ("prediction_error", "raised") for e in results[-3:]),
                "Three consecutive errors halted text review")
        reserves = [e for e in events if e["type"] == "reserve"]
        require(len(reserves) < MAX_REQUESTS, "Text matrix request limit reached")
        key = tuple(details.get(k) for k in ("dataset", "proposal_key", "row_id"))
        require(not any(tuple(e["details"].get(k) for k in ("dataset", "proposal_key", "row_id")) == key for e in reserves),
                "Previously reserved text row cannot be retried")
        if self.stop_after is not None and self.new_requests >= self.stop_after:
            raise route.StopRequested("Requested checkpoint reached")
        ident = self.inner.reserve(amount, details)
        self.pending_id = ident
        self.remember_head()
        self.new_requests += 1
        return ident

    def result(self, reservation_id, details):
        self.check_head()
        require(reservation_id == self.pending_id and not any(e["type"] == "result" and
                e["reservation_id"] == reservation_id for e in self.inner._read()[0]),
                "Result is duplicate or does not belong to the pending request")
        self.inner.result(reservation_id, details)
        amount = settlement_amount(details)
        if amount is not None:
            self.inner.settle(reservation_id, amount, settlement_evidence(details))
        self.remember_head()

    def snapshot(self):
        return self.inner.snapshot()


def make_identity(job, config, ledger_id, bootstrap_samples):
    require(job["manifest_sha256"] and file_sha(ROOT / job["manifest_path"]) == job["manifest_sha256"],
            "Freeze the source before paid review")
    provenance = {"text_review_wrapper_sha256": file_sha(__file__),
        "text_source_validator_sha256": file_sha(sources.__file__), "source_registry_sha256": file_sha(REGISTRY),
        "wrapper_sha256": base.FROZEN_ROUTE_SHA, "ledger_driver_sha256": route.FROZEN_BUDGET_SHA256,
        "ledger_id": ledger_id, "dataset": job["dataset"].name,
        "proposal_key": f"{job['model_key']}_k{job['shots']}", "model_key": job["model_key"],
        "seed": 42, "shots_per_class": job["shots"], "source_shots_per_class": job["shots"],
        "stage_cap_usd": STAGE_CAP, "aggregate_authorized_usd": TOTAL_CAP,
        "fixed_prior_usd": str(FIXED_PRIOR), "numeric_reserved_allowance_usd": str(NUMERIC_RESERVED_ALLOWANCE),
        "settlement_version": SETTLEMENT_VERSION, "route": "OpenRouter", "endpoint": route.ENDPOINT,
        "price": route.PRICE, "source_run_id": job["record"]["run_id"],
        "source_files_sha256": job["source"]["hashes"], "source_manifest_path": job["manifest_path"],
        "source_manifest_sha256": job["manifest_sha256"], "review_prompt_version": "cached-label-text-review-v1"}
    identity = _identity(job["dataset"], "cached_label_jev_review", {**config, "shots_per_class": job["shots"],
        "source_model": job["source"]["model"], "source_run_id": job["record"]["run_id"],
        "budget_guard": provenance}, 42, job["examples"])
    identity.update(bootstrap_samples=bootstrap_samples, proposal=job["source"],
        method_label="cached-label proposal + Jev text review", new_proposal_generation=False,
        source_manifest_sha256=job["manifest_sha256"])
    return identity, provenance


def load_checkpoint(directory, identity, job, ledger, *, allow_halted=False):
    # The pinned validator audits exact prefixes, prompts, proposals, finite Choice
    # vectors and complete coverage. Only the domain wording of the prompt differs.
    record_path = directory / "run.json"
    original = json.loads(record_path.read_text()) if record_path.exists() else None
    view = CheckpointView(directory, original) if original and allow_halted and original.get("status") in HALTED else directory
    with patch.object(base, "build_review_prompt", build_review_prompt):
        record, saved = base.load_checkpoint(view, identity, job["dataset"], job["source"],
            job["record"], job["predictions"], job["examples"])
    base.validate_reservations(ledger, job["dataset"].name, identity["config"]["budget_guard"]["proposal_key"], saved, identity)
    for prediction, proposal in zip(saved, job["predictions"]):
        if not base.valid_proposal(proposal, len(job["dataset"].labels)):
            require("budget" not in prediction.metadata and prediction.metadata.get("jev_review_called") is False,
                    "A failed source proposal cannot incur a Jev request")
    if record and record.get("status") == "stopped_after_billing_error" and not allow_halted:
        raise budget.GuardError("Billing error halted this condition; funded audited recovery is required")
    if original:
        record = original
        require(record.get("status") in {"running", "complete", *HALTED} and record.get("run_id") == directory.name and
                record.get("dataset_manifest") == job["dataset"].manifest and
                record.get("environment", {}).get("source_sha256") == base.FROZEN_CORE_SHA, "Review record identity differs")
        if record["status"] == "complete":
            require(json.loads((directory / "test_manifest.json").read_text()) == base.expected_test_manifest(job["dataset"]),
                    "Complete review test manifest differs")
            require(all(record["metrics"].get(key) == value for key, value in
                    evaluate(job["dataset"].test, saved, len(job["dataset"].labels)).items()), "Saved review metrics differ")
    validate_accounting(ledger, saved)
    return record, saved


def validate_accounting(ledger, predictions):
    events, _ = ledger.inner._read()
    results_list = [e for e in events if e["type"] == "result"]
    settles_list = [e for e in events if e["type"] == "settle"]
    results = {e["reservation_id"]: e for e in results_list}
    settles = {e["reservation_id"]: e for e in settles_list}
    reserves = {e["reservation_id"]: e for e in events if e["type"] == "reserve"}
    require(len(results) == len(results_list) and len(settles) == len(settles_list), "Duplicate text accounting event")
    for prediction in predictions:
        meta = prediction.metadata.get("budget")
        if not meta:
            require(prediction.error == "source_proposal_failed: Jev review not called" and
                    prediction.metadata.get("jev_review_called") is False, "Missing paid-call accounting metadata")
            continue
        ident = meta["reservation_id"]
        require(ident in results and ident in reserves, "Saved prediction lacks result/reservation evidence")
        result = results[ident]["details"]
        route_meta = prediction.metadata.get("openrouter", {})
        require(all(value is None or (type(value) is int and value >= 0)
                    for value in (prediction.input_tokens, prediction.output_tokens)), "Malformed saved token usage")
        cost = route.parse_cost(route_meta.get("reported_cost_usd"))
        require((prediction.input_tokens is None or prediction.input_tokens <= route.PRICE["reserve_input_tokens"])
                and (cost is None or Decimal(cost) <= Decimal(route.PRICE["per_request_reserved_usd"])),
                "Saved response exceeds the original request reservation assumptions")
        require(reserves[ident]["reserved_nano"] == route.RESERVE_NANO and
                meta.get("ledger_id") == ledger.identity["ledger_id"] and
                meta.get("reserved_upper_bound_usd") == route.PRICE["per_request_reserved_usd"] and
                result.get("outcome") == ("prediction_error" if prediction.error else "returned") and
                result.get("usage") == {"input_tokens": prediction.input_tokens, "output_tokens": prediction.output_tokens} and
                result.get("route") == "OpenRouter" and route_meta.get("endpoint") == route.ENDPOINT and
                result.get("provider") == route_meta.get("provider") and
                result.get("request_id") == route_meta.get("id") and
                result.get("reported_cost_usd") == route_meta.get("reported_cost_usd") and
                result.get("resolved_model") == prediction.metadata.get("resolved_model") and
                result.get("reservation_released") is False and result.get("usage_exceeded_reservation_assumptions") is False and
                result.get("guard_reasons") == {"reported_usage_or_cost_overrun": False,
                    "unexpected_route_or_snapshot": False, "invalid_reported_cost": False},
                "Saved response usage/cost/route differs from ledger result")
        if not prediction.error:
            require(prediction.metadata.get("probability_kind") == "jev_choice_distribution" and
                    prediction.metadata.get("resolved_model") in route.ALLOWED_RESPONSE_MODELS and
                    route_meta.get("provider") == "TypeSafe", "Successful result lacks verified native Choice provenance")
        expected = settlement_amount(result)
        if expected is None:
            require(ident not in settles and meta.get("reservation_released") is False and
                    "settled_conservative_usd" not in meta and "settlement_version" not in meta,
                    "Failed/unknown usage reservation was released")
        else:
            require(ident in settles and settles[ident]["charged_nano"] == expected and
                settles[ident].get("evidence") == settlement_evidence(result) and
                events.index(reserves[ident]) < events.index(results[ident]) < events.index(settles[ident]) and
                meta.get("settled_conservative_usd") == budget.usd_string(expected) and
                meta.get("settlement_version") == SETTLEMENT_VERSION and
                meta.get("reservation_released") == (expected < route.RESERVE_NANO), "Settlement differs from verified response")


def review_context(directory, dataset=None):
    record = json.loads((directory / "run.json").read_text())
    guard = record.get("config", {}).get("budget_guard", {})
    name, model, shots = record.get("dataset"), guard.get("model_key"), guard.get("shots_per_class")
    require(name in sources.DATASETS and model in sources.MODELS and type(shots) is int and shots in sources.SHOTS,
            "Unknown text review condition")
    require(directory == OUTPUT / f"{name}__{model}__k{shots}__jev-review", "Noncanonical text review directory")
    dataset = sources.load_dataset(name) if dataset is None else dataset
    job = sources.load_source(dataset, model, shots)
    require(job is not None and job["manifest_sha256"], "Review lacks its frozen source")
    config = route.validate_config({k: v for k, v in record["config"].items()
        if k not in {"shots_per_class", "source_model", "source_run_id", "budget_guard"}})
    return record, job, config


def audit_global(ledger, *, allow_halted=False):
    """Read every condition, including unselected ones; any uncertain call fails closed."""
    verify_frozen_helpers()
    events, _ = ledger.inner._read()
    reserves = [e for e in events if e["type"] == "reserve"]
    results = [e for e in events if e["type"] == "result"]
    settlements = [e for e in events if e["type"] == "settle"]
    ids = {e["reservation_id"] for e in reserves}
    require(len(reserves) <= MAX_REQUESTS and len(ids) == len(reserves) and
            {e["reservation_id"] for e in results} == ids and len(results) == len(ids) and
            {e["reservation_id"] for e in settlements} <= ids, "Orphan or duplicate text reservation/result")
    paths = sorted(OUTPUT.glob("*/run.json"))
    allowed_files = {p.parent / name for p in paths for name in ("run.json", "predictions.jsonl", "test_manifest.json")}
    require(all(path in allowed_files for path in OUTPUT.rglob("*") if path.is_file()),
            "Orphan/partial file exists outside a canonical review checkpoint")
    saved_ids, records = [], []
    for path in paths:
        record, job, config = review_context(path.parent)
        if not allow_halted:
            require(record.get("status") not in HALTED, "A text review checkpoint is halted")
        identity, _ = make_identity(job, config, ledger.identity["ledger_id"], record["bootstrap_samples"])
        checked, predictions = load_checkpoint(path.parent, identity, job, ledger, allow_halted=allow_halted)
        if not allow_halted:
            require(not any(p.error == BILLING_ERROR for p in predictions), "A billing error halted text review globally")
        saved_ids.extend(p.metadata["budget"]["reservation_id"] for p in predictions if p.metadata.get("budget"))
        records.append((path.parent, checked, predictions))
    require(len(saved_ids) == len(set(saved_ids)) and set(saved_ids) == ids,
            "Orphan reservation/result/prediction anywhere in the text experiment")
    if not allow_halted:
        require(len(results) < 3 or not all(e["details"].get("outcome") in {"raised", "prediction_error"} for e in results[-3:]),
                "Three consecutive errors halted text review globally")
    ledger.remember_head()
    # An interrupted in-memory acknowledgment is recoverable only after all
    # ledger events and durable prediction rows have passed this full audit.
    ledger.pending_id = None
    return records


def audit_checkpoint(path, dataset):
    """Historical evidence audit, including halted/partial prefixes; no HTTP/date check."""
    directory = Path(path)
    if directory.name == "run.json":
        directory = directory.parent
    ledger = TextLedger(budget.Ledger(LEDGER, STAGE_CAP))
    for observed, record, predictions in audit_global(ledger, allow_halted=True):
        if observed.resolve() == directory.resolve():
            require(record["dataset_manifest"] == dataset.manifest, "Requested review uses a different dataset")
            return record, predictions
    raise budget.GuardError("Review checkpoint is absent from the audited experiment")


def audit_completed_review(path, dataset):
    record, predictions = audit_checkpoint(path, dataset)
    require(record["status"] == "complete", "Requested review is incomplete")
    return record, predictions


def run_review(job, config, ledger, receipt, bootstrap_samples=1000):
    audit_global(ledger)
    identity, provenance = make_identity(job, config, ledger.identity["ledger_id"], bootstrap_samples)
    dataset, shots, model_key = job["dataset"], job["shots"], job["model_key"]
    directory = OUTPUT / f"{dataset.name}__{model_key}__k{shots}__jev-review"
    record, predictions = load_checkpoint(directory, identity, job, ledger)
    if record and record["status"] == "complete":
        require(all(record["metrics"].get(k) == v for k, v in evaluate(dataset.test, predictions, len(dataset.labels)).items()),
                "Saved review metrics differ")
        return record
    record = record or {**identity, "run_id": directory.name, "started_at": budget.utc_now(),
        "environment": environment(), "dataset_manifest": dataset.manifest,
        "execution": {"concurrency": 1, "automatic_retries": 0, "max_total_review_requests": MAX_REQUESTS,
            "proposal_source": "cached_labels_only", "source_and_reviewer_shots_matched": True,
            "latency_scope": "Jev review only; source stage recorded separately"}}
    record["status"] = "running"
    record.setdefault("execution_sessions", []).append({"started_at": budget.utc_now(),
        "cached_rows": len(predictions), "funding_check": receipt})
    save_json(directory / "run.json", record)
    ledger.active_job, ledger.active_identity = job, identity
    provider = route.GuardedOpenRouterJev(config, ledger, provenance)
    with (directory / "predictions.jsonl").open("a") as stream:
        for row, proposal in zip(dataset.test[len(predictions):], job["predictions"][len(predictions):]):
            ledger.active_index = len(predictions)
            if base.valid_proposal(proposal, len(dataset.labels)):
                prompt = build_review_prompt(row, dataset.labels, job["examples"], proposal.label)
                prediction = provider.predict(row, dataset.labels, prompt)
                prediction.metadata["review_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                ident = prediction.metadata["budget"]["reservation_id"]
                events, _ = ledger.inner._read()
                settled = [e for e in events if e["type"] == "settle" and e["reservation_id"] == ident]
                if settled:
                    amount = settled[0]["charged_nano"]
                    prediction.metadata["budget"].update(settled_conservative_usd=budget.usd_string(amount),
                        reservation_released=amount < route.RESERVE_NANO, settlement_version=SETTLEMENT_VERSION)
            else:
                prediction = Prediction(row.id, None, error="source_proposal_failed: Jev review not called",
                    metadata={"probability_kind": "unavailable", "jev_review_called": False})
            prediction.metadata["proposal"] = base.proposal_metadata(job["source"], job["record"], proposal)
            stream.write(json.dumps(asdict(prediction), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            predictions.append(prediction)
            if prediction.metadata.get("budget"):
                ledger.prediction_saved(prediction)
            billing = prediction.error == "http_error: status=402; no automatic retry"
            if billing or (len(predictions) >= 3 and all(p.error for p in predictions[-3:])):
                record.update(status="stopped_after_billing_error" if billing else "stopped_after_three_consecutive_errors",
                    n_predictions=len(predictions))
                save_json(directory / "run.json", record)
                raise budget.GuardError("Billing error or three consecutive errors stopped text review; no automatic retry")
            if len(predictions) % 25 == 0:
                print(json.dumps({"condition": directory.name, "completed_rows": len(predictions), "test_rows": len(dataset.test)}), flush=True)
    completed = finish_run(dataset, predictions, directory, record, bootstrap_samples)
    audit_global(ledger)
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sources.DATASETS, default=list(sources.DATASETS))
    parser.add_argument("--model-keys", nargs="+", choices=list(sources.MODELS), default=list(sources.MODELS))
    parser.add_argument("--shots", nargs="+", type=int, choices=(0, 4), default=[0, 4])
    parser.add_argument("--freeze-sources", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--init-ledger", action="store_true")
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--stop-after-new-requests", type=int)
    args = parser.parse_args(argv)
    require(all(len(v) == len(set(v)) for v in (args.datasets, args.model_keys, args.shots)) and args.bootstrap_samples >= 100
        and (args.stop_after_new_requests is None or args.stop_after_new_requests > 0), "Duplicate condition or invalid limit")
    verify_transport()
    numeric_snapshot = verify_envelope()
    jobs, pending = [], []
    for name in args.datasets:
        dataset = sources.load_dataset(name)
        for model_key in args.model_keys:
            for shots in args.shots:
                job = sources.load_source(dataset, model_key, shots, freeze=args.freeze_sources)
                if job is None:
                    pending.append(f"{name}__{model_key}__k{shots}")
                else:
                    jobs.append(job)
    plan = {"mode": "execute" if args.execute else "dry_run", "source_conditions_ready": len(jobs),
        "pending_sources": pending, "maximum_review_calls": MAX_REQUESTS,
        "fresh_ready_requests": sum(sum(base.valid_proposal(p, len(j["dataset"].labels)) for p in j["predictions"]) for j in jobs),
        "new_stage_cap_usd": STAGE_CAP, "authorized_cumulative_usd": TOTAL_CAP,
        "fixed_prior_usd": str(FIXED_PRIOR), "numeric_current_usd": numeric_snapshot["charged_or_reserved_usd"],
        "numeric_reserved_allowance_usd": str(NUMERIC_RESERVED_ALLOWANCE),
        "maximum_combined_conservative_usd": str(FIXED_PRIOR + NUMERIC_RESERVED_ALLOWANCE + Decimal(STAGE_CAP)),
        "settlement_version": SETTLEMENT_VERSION,
        "note": "Completion depends on verified new-call usage and funded account credit. Unknown/failed calls retain full reservations. No guarantee all calls fit."}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return plan
    require(not pending and all(j["manifest_sha256"] for j in jobs), "Complete and freeze selected sources before execution")
    config = route.validate_config(json.loads((ROOT / "configs/jev_openrouter.json").read_text())["jev_openrouter"])
    AREA.mkdir(parents=True, exist_ok=True)
    with (AREA / "review-execution.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise budget.GuardError("Another text review worker is active") from None
        if args.prompt_api_key:
            budget.prompt_credentials([config])
        require(bool(os.environ.get(config["api_key_env"])), "Missing OpenRouter credential")
        # Provider-funding preflight precedes ledger initialization or any model
        # call, so creating a new study cannot evade the earlier billing failure.
        receipt = public_receipt(check_credit(os.environ[config["api_key_env"]]))
        ledger = TextLedger(budget.Ledger(LEDGER, STAGE_CAP, initialize=args.init_ledger), stop_after=args.stop_after_new_requests)
        audit_global(ledger)
        completed = []
        for job in jobs:
            try:
                record = run_review(job, config, ledger, receipt, args.bootstrap_samples)
            except route.StopRequested:
                print(json.dumps({"status": "checkpoint", "new_requests": ledger.new_requests, "budget": ledger.snapshot()}), flush=True)
                return
            completed.append(record["run_id"])
            print(json.dumps({"completed": record["run_id"], "accuracy": record["metrics"]["accuracy"],
                "budget": ledger.snapshot()}), flush=True)
        return {"completed": completed, "budget": ledger.snapshot()}


if __name__ == "__main__":
    try:
        main()
    except (budget.GuardError, budget.BudgetStop) as exc:
        raise SystemExit(f"STOP: {exc}") from None
