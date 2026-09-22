"""Read-only reconciliation of the expanded numerical Jev review ledger.

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

import run_expanded_numeric_review as review
from jevbench.types import Prediction
from tabular_data import load_native_prepared

# The four previously measured Qwen3/Astra four-shot reviews are deliberately
# absent: their costs are already included in the immutable prior allowance.
CONDITIONS = {
    (dataset, f"{model_key}_k{shots}"): (model_key, shots)
    for dataset in review.DATASETS for model_key in review.MODELS for shots in review.SHOTS
    if not review.is_reused(model_key, shots)
}
RUN_IDS = {
    f"{dataset}__{model_key}__k{shots}__jev-review": (dataset, proposal_key)
    for (dataset, proposal_key), (model_key, shots) in CONDITIONS.items()
}


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
            require(event["reserved_nano"] == review.route.RESERVE_NANO, "Unexpected reservation amount")
            details = event["details"]
            key = details.get("dataset"), details.get("proposal_key"), details.get("row_id")
            require(key[:2] in CONDITIONS, "Old or unplanned review must not incur an expansion charge")
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
                require(money(details["reported_cost_usd"]) <= money(review.route.PRICE["per_request_reserved_usd"]),
                        "Reported cost exceeds the bound; reconciliation is required")
            if details["outcome"] != "raised":
                require(details.get("route") == "OpenRouter" and details.get("reservation_released") is False,
                        "Result route or retained-reservation flag differs")
                if details["outcome"] == "returned":
                    require(details.get("provider") == "TypeSafe" and details.get("resolved_model") in review.route.ALLOWED_RESPONSE_MODELS,
                            "Successful result has an unapproved provider/model")
                else:
                    require(details.get("provider") in {None, "TypeSafe"} and
                            details.get("resolved_model") in {None, *review.route.ALLOWED_RESPONSE_MODELS},
                            "Failed result has an unapproved provider/model")
            results[ident] = event
        else:
            raise ValueError("Unexpected review ledger event type")
    require(len(reservations) <= review.MAX_REQUESTS, "Review request limit exceeded")
    require(complete_conditions <= set(CONDITIONS), "An old or unplanned review appears as a new condition")
    require(set(paid_predictions) <= set(reservations), "Paid prediction has no reservation")
    for ident, prediction in paid_predictions.items():
        require(ident in results, "Saved paid prediction has no result evidence")
        reservation, result = reservations[ident], results[ident]["details"]
        require(result["outcome"] != "raised", "Raised request unexpectedly has a saved prediction")
        meta, charged = prediction.metadata, prediction.metadata.get("budget", {})
        route = meta.get("openrouter", {})
        require(charged.get("ledger_id") == ledger_id and charged.get("reservation_id") == ident and
                charged.get("reservation_released") is False and
                same_cost(charged.get("reserved_upper_bound_usd"), review.route.PRICE["per_request_reserved_usd"]),
                "Prediction budget provenance differs")
        require(prediction.row_id == reservation["details"]["row_id"] and
                meta.get("review_prompt_sha256") == reservation["details"]["prompt_sha256"],
                "Paid prediction row or prompt differs")
        require(route.get("endpoint") == review.route.ENDPOINT and
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
    all_conditions = set(CONDITIONS)
    require(complete_summary_conditions <= complete_conditions, "Summary marks an incomplete review as complete")
    complete = (complete_conditions == complete_summary_conditions == all_conditions and not missing_result
                and not missing_prediction and not partial_checkpoint_files and not halted_conditions)
    condition_costs = []
    for (dataset, proposal_key), (model_key, shots) in sorted(CONDITIONS.items()):
        ids = {ident for ident, event in reservations.items() if
               (event["details"]["dataset"], event["details"]["proposal_key"]) == (dataset, proposal_key)}
        observed = ids & set(known)
        condition_costs.append({"dataset": dataset, "model_key": model_key, "shots_per_class": shots,
            "model_requests": len(ids), "reported_cost_requests": len(observed),
            "unknown_cost_requests": len(ids) - len(observed),
            "known_reported_api_usd": str(sum((known[ident] for ident in observed), Decimal(0))),
            "conservative_usd": review.budget.usd_string(sum(reservations[ident]["reserved_nano"] for ident in ids)),
            "checkpointed_paid_predictions": len(ids & set(paid_predictions)),
            "reservations_without_result": len(ids & missing_result),
            "results_without_saved_prediction": len(ids & missing_prediction)})
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
        "expected_new_review_conditions": len(CONDITIONS), "maximum_new_model_requests": review.MAX_REQUESTS,
        "stage_authorized_usd": review.STAGE_CAP, "reused_old_review_conditions": 4,
        "condition_costs": condition_costs,
        "note": "Expansion-only incremental expense. Four reused reviews are excluded from these calls and included in prior accounting. "
                "Unknown charges include reserved calls without reported cost. Source failures trigger no Jev call. "
                "Cached hosted proposal costs remain in the earlier study; local compute cost is unmeasured.",
    }


def collect_costs(rows):
    """Reconcile every new call, including incomplete/paused run checkpoints."""
    review.verify_prior()
    require(len(CONDITIONS) == 20 and len(RUN_IDS) == 20, "Expanded condition catalog changed")
    if not review.LEDGER.exists():
        require(not list(review.OUTPUT.glob("*/run.json")) and not list(review.OUTPUT.glob("*/predictions.jsonl")),
                "Expanded checkpoints exist without their canonical ledger")
        return {"status": "not_executed", "new_model_requests": 0, "reported_cost_requests": 0,
            "unknown_cost_requests": 0, "known_reported_api_usd": "0", "review_conservative_usd": "0.000000000",
            "prior_conservative_usd": review.PRIOR_TOTAL, "cumulative_conservative_usd": review.PRIOR_TOTAL,
            "cumulative_authorized_usd": review.TOTAL_CAP, "new_llm_generation_calls": 0,
            "ledger_sha256": None, "expected_new_review_conditions": 20, "maximum_new_model_requests": 1500,
            "note": "No expanded review ledger exists; the four reused reviews remain in prior accounting."}
    ledger = review.budget.Ledger(review.LEDGER, review.STAGE_CAP)
    summary_conditions, summary_sources = set(), {}
    prefix = review.OUTPUT.relative_to(review.ROOT).as_posix() + "/"
    for row in rows:
        if row.get("status", "complete") != "complete" or "source_run_id" not in row:
            continue
        run_id, source_path = row.get("run_id"), row.get("source_path")
        if run_id not in RUN_IDS:
            require(not isinstance(source_path, str) or not source_path.startswith(prefix),
                    "Unknown completed run in expanded-review namespace")
            continue  # Includes the four old reviews and non-review arms.
        key = RUN_IDS[run_id]
        require(key not in summary_conditions and row.get("dataset") == key[0], "Duplicate or mismatched expanded summary condition")
        require(source_path == prefix + run_id + "/run.json", "Expanded summary source path differs")
        summary_conditions.add(key)
        summary_sources[key] = row["source_run_id"]
    config = review.route.validate_config(json.loads((review.ROOT / "configs/tabular_hosted.json").read_text())["jev_openrouter"])
    datasets = {name: load_native_prepared(review.ROOT / "data/tabular-full" / name)[0] for name in review.DATASETS}
    # One shared lock binds the ledger events, durable head, hashes, and all
    # checkpoints to the same reservation/result snapshot. A response may finish
    # its append outside the ledger lock; it can only reference an existing event.
    with ledger.lock_path.open("r") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            events, _ = ledger._read()
            ledger_hash = hashlib.sha256(ledger.path.read_bytes()).hexdigest()
            lock_hash = hashlib.sha256(ledger.lock_path.read_bytes()).hexdigest()
            paid, expected, complete, seen = {}, {}, set(), set()
            partial_count, skipped, halted = 0, 0, 0
            for path in sorted(review.OUTPUT.glob("*/run.json")):
                record = json.loads(path.read_text())
                guard_record = record.get("config", {}).get("budget_guard", {})
                key = record.get("dataset"), guard_record.get("proposal_key")
                require(key in CONDITIONS and key not in seen, "Unplanned, reused, or duplicate expanded checkpoint")
                seen.add(key)
                model_key, shots = CONDITIONS[key]
                dataset = datasets[key[0]]
                require(guard_record.get("model_key") == model_key and guard_record.get("shots_per_class") == shots and
                        guard_record.get("source_shots_per_class") == shots, "Source/reviewer shot condition differs")
                job = review.load_source(dataset, model_key, shots)
                require(job is not None and job["manifest_sha256"] is not None, "Paid checkpoint lacks an individually frozen source")
                identity, guard = review.make_identity(job, config, ledger.identity["ledger_id"], record["bootstrap_samples"])
                require(all(record.get(field) == value for field, value in identity.items()), "Expanded execution/source provenance changed")
                require(record.get("status") in {"running", "complete", "stopped_after_three_consecutive_errors"}, "Unknown expanded checkpoint status")
                expected_id = review.condition_key(key[0], model_key, shots) + "__jev-review"
                require(record.get("run_id") == path.parent.name == expected_id, "Expanded directory/run identity differs")
                source, original, proposals, examples = job["source"], job["record"], job["predictions"], job["examples"]
                if key in summary_sources:
                    require(summary_sources[key] == original["run_id"], "Expanded summary identifies a different source proposal")
                for row, proposal in zip(dataset.test, proposals):
                    if review.base.valid_proposal(proposal, len(dataset.labels)):
                        prompt = review.base.build_review_prompt(row, dataset.labels, examples, proposal.label)
                        expected[key[0], key[1], row.id] = {**guard, "row_id": row.id,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "model": review.route.MODEL, "provider": "jev", "pricing": review.route.PRICE}
                pred_path = path.with_name("predictions.jsonl")
                raw = pred_path.read_text() if pred_path.exists() else ""
                if raw and not raw.endswith("\n"):
                    require(record["status"] != "complete", "Complete run contains an incomplete prediction write")
                    partial_count += 1
                    raw = raw.rsplit("\n", 1)[0] + "\n" if "\n" in raw else ""
                predictions = [Prediction(**json.loads(line)) for line in raw.splitlines()]
                require([prediction.row_id for prediction in predictions] == [row.id for row in dataset.test[:len(predictions)]],
                        "Expanded checkpoint is not an ordered test prefix")
                if record["status"] == "complete":
                    require(len(predictions) == len(dataset.test), "Completed expanded checkpoint is missing rows")
                    complete.add(key)
                halted += record["status"] == "stopped_after_three_consecutive_errors"
                for row, prediction, proposal in zip(dataset.test, predictions, proposals):
                    require(isinstance(prediction.metadata, dict) and
                            prediction.metadata.get("proposal") == review.base.proposal_metadata(source, original, proposal),
                            "Expanded checkpoint source proposal differs")
                    if not review.base.valid_proposal(proposal, len(dataset.labels)):
                        require(prediction.error == "source_proposal_failed: Jev review not called" and
                                prediction.label is None and prediction.probabilities is None and
                                prediction.metadata.get("jev_review_called") is False and "budget" not in prediction.metadata and
                                "openrouter" not in prediction.metadata, "Failed source proposal incurred a fallback call")
                        skipped += 1
                        continue
                    ident = prediction.metadata.get("budget", {}).get("reservation_id")
                    require(isinstance(ident, str) and ident and ident not in paid, "Missing or duplicate expanded reservation identity")
                    paid[ident] = prediction
            return reconcile_events(events, paid, skipped, ledger_id=ledger.identity["ledger_id"],
                expected_reservations=expected, complete_conditions=complete, complete_summary_conditions=summary_conditions,
                partial_checkpoint_files=partial_count, halted_conditions=halted, ledger_sha256=ledger_hash, lock_sha256=lock_hash)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    summary_path = review.AREA / "COMPARISON.json"
    rows = json.loads(summary_path.read_text())["runs"] if summary_path.exists() else []
    print(json.dumps(collect_costs(rows), indent=2, allow_nan=False))
