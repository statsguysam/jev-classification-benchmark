"""Read-only reconciliation of every reserved numeric Jev review request.

Includes partial checkpoints and reserved calls without a saved response. The
ledger's shared lock makes the event/hash snapshot consistent while preventing
new reservations/results from racing the checkpoint read. No model calls,
settlements, or edits are performed.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
from pathlib import Path

import run_numeric_jev_review as review
from jevbench.types import Prediction
from tabular_data import load_native_prepared


def require(condition, message):
    if not condition:
        raise ValueError(message)


def money(value):
    if isinstance(value, bool):
        raise ValueError("Boolean cost is invalid")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("Invalid recorded API cost") from None
    require(result.is_finite() and result >= 0, "Nonfinite or negative API cost")
    return result


def same_cost(a, b):
    return a is None and b is None or a is not None and b is not None and money(a) == money(b)


def reconcile_events(events, paid_predictions, skipped_source_rows, *, ledger_id,
                     expected_reservations, complete_conditions, complete_summary_conditions,
                     partial_checkpoint_files=0, halted_conditions=0, ledger_sha256="", lock_sha256=""):
    """Reconcile one stable event/checkpoint snapshot; all IDs are globally unique.

    expected_reservations maps allowed (dataset, proposal_key, row_id) tuples to
    exact request provenance and prompt hashes, derived only from pinned data.
    paid_predictions maps reservation IDs to saved Prediction objects.
    """
    require(not any(event["type"] == "settle" for event in events), "Review reservations must never be released")
    reservations, results, condition_rows = {}, {}, set()
    for event in events:
        kind = event["type"]
        if kind == "header":
            require(event.get("ledger_id") == ledger_id, "Review ledger identity differs")
            continue
        ident = event.get("reservation_id")
        if kind == "reserve":
            require(isinstance(ident, str) and ident and ident not in reservations, "Duplicate or invalid reservation ID")
            require(event["reserved_nano"] == review.jev_route.RESERVE_NANO, "Unexpected reservation amount")
            details = event["details"]
            key = details.get("dataset"), details.get("proposal_key"), details.get("row_id")
            require(key in expected_reservations and key not in condition_rows, "Unrecognized or repeated paid row")
            expected = expected_reservations[key]
            require(all(details.get(field) == value for field, value in expected.items()), "Reservation prompt/source provenance differs")
            condition_rows.add(key)
            reservations[ident] = event
        elif kind == "result":
            require(ident in reservations, "Result lacks an earlier reservation")
            require(ident not in results, "Multiple result events for one reserved call")
            details = event["details"]
            require(details.get("outcome") in {"returned", "prediction_error", "raised"}, "Unknown review result outcome")
            require(not details.get("usage_exceeded_reservation_assumptions", False), "Review ledger contains a routing/usage guard violation")
            if details.get("reported_cost_usd") is not None:
                require(money(details["reported_cost_usd"]) <= money(review.jev_route.PRICE["per_request_reserved_usd"]),
                        "Reported cost exceeds the bound; reconciliation is required")
            if details["outcome"] != "raised":
                require(details.get("route") == "OpenRouter" and details.get("reservation_released") is False,
                        "Result route or retained-reservation flag differs")
                if details["outcome"] == "returned":
                    require(details.get("provider") == "TypeSafe" and details.get("resolved_model") in review.jev_route.ALLOWED_RESPONSE_MODELS,
                            "Successful result has an unapproved provider/model")
                else:
                    require(details.get("provider") in {None, "TypeSafe"} and
                            details.get("resolved_model") in {None, *review.jev_route.ALLOWED_RESPONSE_MODELS},
                            "Failed result has an unapproved provider/model")
            results[ident] = event
        else:
            raise ValueError("Unexpected review ledger event type")
    require(len(reservations) <= review.MAX_REQUESTS, "Review request limit exceeded")
    require(set(paid_predictions) <= set(reservations), "Paid prediction has no reservation")
    for ident, prediction in paid_predictions.items():
        require(ident in results, "Saved paid prediction has no result evidence")
        reservation, result = reservations[ident], results[ident]["details"]
        require(result["outcome"] != "raised", "Raised request unexpectedly has a saved prediction")
        meta, charged = prediction.metadata, prediction.metadata.get("budget", {})
        route = meta.get("openrouter", {})
        require(charged.get("ledger_id") == ledger_id and charged.get("reservation_id") == ident and
                charged.get("reservation_released") is False and
                same_cost(charged.get("reserved_upper_bound_usd"), review.jev_route.PRICE["per_request_reserved_usd"]),
                "Prediction budget provenance differs")
        require(prediction.row_id == reservation["details"]["row_id"] and
                meta.get("review_prompt_sha256") == reservation["details"]["prompt_sha256"],
                "Paid prediction row or prompt differs")
        require(route.get("endpoint") == review.jev_route.ENDPOINT and
                route.get("provider") == result.get("provider") and route.get("id") == result.get("request_id") and
                meta.get("resolved_model") == result.get("resolved_model") and
                same_cost(route.get("reported_cost_usd"), result.get("reported_cost_usd")),
                "Prediction cost/provider/model/request differs from ledger result")
        require(result["outcome"] == ("prediction_error" if prediction.error else "returned"), "Prediction/result outcome differs")
        require(result.get("usage") == {"input_tokens": prediction.input_tokens, "output_tokens": prediction.output_tokens},
                "Prediction token usage differs from ledger result")
    missing_result = set(reservations) - set(results)
    missing_prediction = set(results) - set(paid_predictions)
    # Completed conditions must be fully reconciled even if other arms are still
    # running. Partial arms may legitimately contain a call in flight.
    for ident in missing_result | missing_prediction:
        detail = reservations[ident]["details"]
        require((detail["dataset"], detail["proposal_key"]) not in complete_conditions,
                "Completed condition has an unreconciled reserved request")
    known = {ident: money(event["details"]["reported_cost_usd"]) for ident, event in results.items()
             if event["details"].get("reported_cost_usd") is not None}
    reserved_nano = sum(event["reserved_nano"] for event in reservations.values())
    conservative = review.budget.usd_string(reserved_nano)
    require(money(conservative) <= money(review.STAGE_CAP), "Review stage cap exceeded")
    cumulative = money(review.PRIOR_TOTAL) + money(conservative)
    require(cumulative <= money(review.TOTAL_CAP), "Cumulative authorization exceeded")
    all_conditions = set(review.SOURCES)
    require(complete_summary_conditions <= complete_conditions, "Summary marks an incomplete review as complete")
    complete = (complete_conditions == complete_summary_conditions == all_conditions and not missing_result
                and not missing_prediction and not partial_checkpoint_files and not halted_conditions)
    return {
        "status": "complete" if complete else "halted" if halted_conditions else "in_progress",
        "new_model_requests": len(reservations), "reported_cost_requests": len(known),
        "unknown_cost_requests": len(reservations) - len(known),
        "known_reported_api_usd": str(sum(known.values(), Decimal(0))),
        "review_conservative_usd": conservative, "prior_conservative_usd": review.PRIOR_TOTAL,
        "cumulative_conservative_usd": str(cumulative), "cumulative_authorized_usd": review.TOTAL_CAP,
        "new_llm_generation_calls": 0, "ledger_sha256": ledger_sha256, "ledger_lock_sha256": lock_sha256,
        "unique_result_events": len(results), "checkpointed_paid_predictions": len(paid_predictions),
        "reservations_without_result": len(missing_result), "results_without_saved_prediction": len(missing_prediction),
        "source_failure_rows_without_jev_call": skipped_source_rows, "partial_checkpoint_files": partial_checkpoint_files,
        "complete_checkpoint_conditions": len(complete_conditions), "complete_summary_conditions": len(complete_summary_conditions),
        "note": "Review-only incremental expense. Unknown charges include reserved calls without reported cost. "
                "Source failures trigger no Jev call. Cached LLM proposal cost remains in the earlier study; local compute cost is unmeasured.",
    }


def collect_costs(rows):
    """Return current accounting without writing any ledger, run, or report."""
    review.verify_prior()
    if not review.LEDGER.exists():
        require(not list(review.OUTPUT.glob("*/predictions.jsonl")), "Review checkpoints exist without their ledger")
        return {"status": "not_executed", "new_model_requests": 0, "reported_cost_requests": 0,
                "unknown_cost_requests": 0, "known_reported_api_usd": "0", "review_conservative_usd": "0.000000000",
                "prior_conservative_usd": review.PRIOR_TOTAL, "cumulative_conservative_usd": review.PRIOR_TOTAL,
                "cumulative_authorized_usd": review.TOTAL_CAP, "new_llm_generation_calls": 0,
                "ledger_sha256": None, "note": "No review ledger exists; no new Jev review calls are recorded."}
    ledger = review.budget.Ledger(review.LEDGER, review.STAGE_CAP)
    summary_conditions = set()
    for row in rows:
        if "source_run_id" not in row:
            continue
        matches = [key for key, source in review.SOURCES.items() if key[0] == row["dataset"] and
                   Path(source["path"]).name == row["source_run_id"]]
        require(len(matches) == 1 and matches[0] not in summary_conditions, "Duplicate or unknown summarized review source")
        summary_conditions.add(matches[0])
    config = json.loads((review.ROOT / "configs/tabular_hosted.json").read_text())["jev_openrouter"]
    config = review.jev_route.validate_config(config)
    contexts, expected = {}, {}
    for dataset_name, proposal_key in review.SOURCES:
        dataset, _ = load_native_prepared(review.ROOT / "data/tabular-full" / dataset_name)
        source, source_record, proposals, examples = review.load_source(dataset, proposal_key)
        contexts[dataset_name, proposal_key] = dataset, source, source_record, proposals, examples
    # This shared lock protects the exact events and their durable anchor during
    # the checkpoint scan; the active worker can still finish writing a response
    # for a result already present in this snapshot.
    with ledger.lock_path.open("r") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            events, _ = ledger._read()
            ledger_hash = hashlib.sha256(ledger.path.read_bytes()).hexdigest()
            lock_hash = hashlib.sha256(ledger.lock_path.read_bytes()).hexdigest()
            paid, complete, partial_count, skipped, halted = {}, set(), 0, 0, 0
            seen_conditions = set()
            for path in sorted(review.OUTPUT.glob("*/run.json")):
                record = json.loads(path.read_text())
                key = record.get("dataset"), record.get("config", {}).get("budget_guard", {}).get("proposal_key")
                require(key in contexts and key not in seen_conditions, "Unexpected or duplicated review checkpoint")
                seen_conditions.add(key)
                dataset, source, source_record, proposals, examples = contexts[key]
                identity, guard = review.make_identity(dataset, key[1], source, source_record, examples, config,
                    ledger.identity["ledger_id"], record["bootstrap_samples"])
                require(all(record.get(field) == value for field, value in identity.items()), "Checkpoint execution provenance changed")
                require(record.get("status") in {"running", "complete", "stopped_after_three_consecutive_errors"}, "Unknown checkpoint status")
                require(record.get("run_id") == path.parent.name == f"{key[0]}__{key[1]}__jev-review", "Checkpoint directory/run identity differs")
                for row, proposal in zip(dataset.test, proposals):
                    if review.valid_proposal(proposal, len(dataset.labels)):
                        prompt = review.build_review_prompt(row, dataset.labels, examples, proposal.label)
                        expected[key[0], key[1], row.id] = {**guard, "row_id": row.id,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "model": review.jev_route.MODEL, "provider": "jev", "pricing": review.jev_route.PRICE}
                pred_path = path.with_name("predictions.jsonl")
                raw = pred_path.read_text() if pred_path.exists() else ""
                if raw and not raw.endswith("\n"):
                    require(record["status"] != "complete", "Complete run contains an incomplete prediction write")
                    partial_count += 1
                    raw = raw.rsplit("\n", 1)[0] + "\n" if "\n" in raw else ""
                predictions = [Prediction(**json.loads(line)) for line in raw.splitlines()]
                require([p.row_id for p in predictions] == [r.id for r in dataset.test[:len(predictions)]], "Checkpoint is not an aligned test prefix")
                if record["status"] == "complete":
                    require(len(predictions) == len(dataset.test), "Complete run is missing predictions")
                    complete.add(key)
                halted += record["status"] == "stopped_after_three_consecutive_errors"
                for row, prediction, proposal in zip(dataset.test, predictions, proposals):
                    require(prediction.metadata.get("proposal") == review.proposal_metadata(source, source_record, proposal),
                            "Saved source proposal metadata differs")
                    if not review.valid_proposal(proposal, len(dataset.labels)):
                        require(prediction.error == "source_proposal_failed: Jev review not called" and
                                prediction.label is None and prediction.probabilities is None and
                                prediction.metadata.get("jev_review_called") is False and "budget" not in prediction.metadata and
                                "openrouter" not in prediction.metadata, "Failed source proposal incurred a fallback call")
                        skipped += 1
                        continue
                    ident = prediction.metadata.get("budget", {}).get("reservation_id")
                    require(isinstance(ident, str) and ident and ident not in paid, "Missing or duplicate paid prediction reservation")
                    paid[ident] = prediction
            return reconcile_events(events, paid, skipped, ledger_id=ledger.identity["ledger_id"],
                expected_reservations=expected, complete_conditions=complete, complete_summary_conditions=summary_conditions,
                partial_checkpoint_files=partial_count, halted_conditions=halted, ledger_sha256=ledger_hash, lock_sha256=lock_hash)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
