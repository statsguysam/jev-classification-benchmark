"""Pure analysis of separately recorded recovery attempts; no I/O or inference.

The caller must audit its frozen plan, original files and append-only attempt
ledger before supplying these objects. This helper checks request identities,
original prediction digests and attempt order, then applies a fixed selection
rule without consulting ground truth: preserve an original success, otherwise
select the earliest valid success. Later successes never replace it.

Identity fields are exactly condition_id (= run_id), row_id, prompt_sha256,
choices_sha256, source_sha256, original_prediction_sha256. Direct requests may
use source_sha256=None. A previously unattempted row has original_prediction_
sha256=None. Records are {attempt_index: 1, identity: {...}, prediction: full
Prediction mapping, stage: 'retry'}. Indices start at one and increase per row.
The first new call for an originally unattempted row uses stage='first_attempt'.
After an explicit upstream_not_called outcome, a separately versioned dependent
review uses stage='dependent_first_call', with the new proposal's source hash.
Subsequent calls use stage='retry'. Raw historical outcomes remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevbench.metrics import evaluate
from jevbench.runner import digest
from jevbench.types import Prediction, Row

IDENTITY_FIELDS = {"condition_id", "row_id", "prompt_sha256", "choices_sha256",
                   "source_sha256", "original_prediction_sha256"}
ORIGINAL_STAGES = {"attempted", "not_attempted", "upstream_not_called"}
METRIC_FIELDS = ("accuracy", "macro_f1", "balanced_accuracy", "n_failures", "failure_rate",
                 "valid_response_accuracy", "probability_coverage", "n_probability_rows")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prediction_mapping(value):
    return asdict(value) if isinstance(value, Prediction) else dict(value)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def validate_prediction(value, row_id, n_classes, *, require_probabilities):
    require(isinstance(value, dict), "Prediction must be a mapping")
    try:
        prediction = Prediction(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("Malformed prediction fields") from error
    require(prediction.row_id == row_id, "Prediction row identity differs")
    if prediction.error is not None:
        require(isinstance(prediction.error, str) and bool(prediction.error)
                and prediction.label is None and prediction.probabilities is None,
                "Failed prediction must retain an error and no label/probabilities")
        return False
    require(type(prediction.label) is int and 0 <= prediction.label < n_classes, "Invalid successful class label")
    probabilities = prediction.probabilities
    require(probabilities is not None or not require_probabilities, "Successful response lacks required probabilities")
    if probabilities is not None:
        require(isinstance(probabilities, list) and len(probabilities) == n_classes
                and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in probabilities)
                and math.isclose(sum(probabilities), 1, rel_tol=0, abs_tol=1e-6), "Invalid successful probability vector")
        if require_probabilities:
            # Jev callers use this branch. Match the frozen JevClassifier rule
            # exactly, including ties and its absolute decision tolerance.
            # Keep the saved label/vector unchanged; this is an audit check.
            require(probabilities[prediction.label] >= max(probabilities) - 1e-6,
                    "Successful Jev label exceeds the provider probability decision tolerance")
        else:
            require(probabilities[prediction.label] == max(probabilities), "Successful label is not a probability maximum")
    return True


def failure_kind(prediction):
    if prediction is None:
        return "not_attempted"
    error = prediction.get("error")
    if error is None:
        return "success"
    if error.startswith("source_proposal_failed"):
        return "upstream_not_called"
    if "status=402" in error:
        return "billing"
    if error.startswith("network_error"):
        return "transport"
    if error.startswith("http_error"):
        return "http_service"
    if error.startswith("invalid_output"):
        return "invalid_output"
    return "other_failure"


def summarize_view(rows, labels, predictions):
    """Never score an unattempted row as an observed model failure or zero."""
    missing = sum(predictions[row.id] is None for row in rows)
    counts = {}
    for row in rows:
        kind = failure_kind(predictions[row.id])
        counts[kind] = counts.get(kind, 0) + 1
    metrics = None
    if not missing:
        measured = evaluate(rows, [Prediction(**predictions[row.id]) for row in rows], len(labels))
        metrics = {key: measured[key] for key in METRIC_FIELDS}
        metrics["n_rows"] = len(rows)
        if {row.label for row in rows} != set(range(len(labels))):
            metrics["balanced_accuracy"] = None
    return {"status": "complete" if not missing else "incomplete", "n_rows": len(rows),
        "n_observed": len(rows) - missing, "n_not_attempted": missing,
        "outcome_counts": counts, "metrics": metrics}


def analyze_condition(rows: list[Row], labels: list[str], original_predictions_by_row: dict,
                      expected_identities: dict, retry_attempts: list[dict], *,
                      original_stages: dict | None = None, require_probabilities: bool = True) -> dict:
    """Analyze one run; input objects are never modified.

    original_snapshot includes historical upstream skips. first_attempt uses the
    first actual service call (original or new); recovered uses the first valid
    success, falling back to the latest actual failure. No-service-call rows keep
    both metrics unavailable until the complete evaluation set has been tried.
    No spending or ledger audit is implied by this pure analysis.
    """
    require(rows and len({r.id for r in rows}) == len(rows), "Rows must be nonempty and unique")
    require(len(labels) >= 2 and len(set(labels)) == len(labels), "Declared classes must be unique")
    require(all(type(r.label) is int and 0 <= r.label < len(labels) for r in rows), "Invalid truth label")
    row_ids = {r.id for r in rows}
    require(set(expected_identities) == row_ids and set(original_predictions_by_row) <= row_ids,
            "Original/request rows differ from the evaluation set")
    if original_stages is not None:
        require(set(original_stages) == row_ids, "Explicit original stages must cover every row")
    originals, stages, valid_original = {}, {}, {}
    conditions = set()
    for row in rows:
        value = original_predictions_by_row.get(row.id)
        value = None if value is None else prediction_mapping(value)
        originals[row.id] = value
        inferred = ("not_attempted" if value is None else
                    "upstream_not_called" if failure_kind(value) == "upstream_not_called" else "attempted")
        stage = original_stages[row.id] if original_stages is not None else inferred
        require(stage in ORIGINAL_STAGES and stage == inferred, "Original stage contradicts the saved outcome")
        stages[row.id] = stage
        valid_original[row.id] = False if value is None else validate_prediction(
            value, row.id, len(labels), require_probabilities=require_probabilities)
        identity = expected_identities[row.id]
        require(isinstance(identity, dict) and set(identity) == IDENTITY_FIELDS, "Request identity fields differ")
        require(isinstance(identity["condition_id"], str) and bool(identity["condition_id"])
                and identity["row_id"] == row.id, "Request condition/row identity differs")
        conditions.add(identity["condition_id"])
        require(valid_sha(identity["prompt_sha256"]) and valid_sha(identity["choices_sha256"])
                and (identity["source_sha256"] is None or valid_sha(identity["source_sha256"])),
                "Request prompt/choices/source digest is invalid")
        require(identity["original_prediction_sha256"] == (None if value is None else digest(value)),
                "Original prediction digest differs")
        original_prompt = value.get("metadata", {}).get("review_prompt_sha256") if value else None
        if stage == "attempted" and original_prompt is not None:
            require(original_prompt == identity["prompt_sha256"], "Original review prompt differs from retry identity")
        if stage == "upstream_not_called":
            require(identity["source_sha256"] is not None, "Dependent review requires a versioned source digest")
    require(len(conditions) == 1, "Analyze exactly one condition/run at a time")
    per_row = {row.id: [] for row in rows}
    for record in retry_attempts:
        require(isinstance(record, dict) and set(record) == {"attempt_index", "identity", "prediction", "stage"},
                "Attempt record fields differ")
        identity = record["identity"]
        require(isinstance(identity, dict) and identity.get("row_id") in row_ids, "Unknown attempt row")
        row_id = identity["row_id"]
        require(identity == expected_identities[row_id], "Attempt prompt/source/original identity differs")
        index = record["attempt_index"]
        require(type(index) is int and index == len(per_row[row_id]) + 1, "Attempts must be append-ordered and consecutive per row")
        expected_stage = ("retry" if index > 1 or stages[row_id] == "attempted" else
                          "dependent_first_call" if stages[row_id] == "upstream_not_called" else "first_attempt")
        require(record["stage"] == expected_stage, "Attempt stage differs from its original call history")
        value = prediction_mapping(record["prediction"])
        successful = validate_prediction(value, row_id, len(labels), require_probabilities=require_probabilities)
        require(failure_kind(value) != "upstream_not_called", "A new service attempt cannot be an upstream skip")
        per_row[row_id].append({"index": index, "prediction": value, "success": successful, "stage": expected_stage})
    first, recovered, selections = {}, {}, []
    post_success_attempts = 0
    all_attempt_outcomes = {}
    for row in rows:
        attempts = []
        if stages[row.id] == "attempted":
            attempts.append({"index": 0, "prediction": originals[row.id], "success": valid_original[row.id], "stage": "original"})
        attempts.extend(per_row[row.id])
        for item in attempts:
            kind = failure_kind(item["prediction"])
            all_attempt_outcomes[kind] = all_attempt_outcomes.get(kind, 0) + 1
        first[row.id] = attempts[0]["prediction"] if attempts else None
        success_position = next((i for i, a in enumerate(attempts) if a["success"]), None)
        selected = attempts[success_position] if success_position is not None else attempts[-1] if attempts else None
        later = len(attempts) - success_position - 1 if success_position is not None else 0
        post_success_attempts += later
        recovered[row.id] = selected["prediction"] if selected else None
        selections.append({"row_id": row.id, "identity": dict(expected_identities[row.id]),
            "original_stage": stages[row.id], "original_outcome": failure_kind(originals[row.id]),
            "n_service_attempts": len(attempts), "n_new_calls": len(per_row[row.id]),
            "selected_attempt_index": selected["index"] if selected else None,
            "selected_stage": selected["stage"] if selected else None,
            "selected_prediction_sha256": digest(selected["prediction"]) if selected else None,
            "first_attempt_label": first[row.id]["label"] if first[row.id] else None,
            "recovered_label": recovered[row.id]["label"] if recovered[row.id] else None,
            "first_attempt_outcome": failure_kind(first[row.id]), "recovered_outcome": failure_kind(recovered[row.id]),
            "post_success_attempts_ignored": later,
            "attempt_prediction_sha256": [digest(a["prediction"]) for a in attempts],
            "attempt_outcomes": [{"attempt_index": a["index"], "stage": a["stage"],
                                  "outcome": failure_kind(a["prediction"])} for a in attempts]})
    first_view, recovered_view = summarize_view(rows, labels, first), summarize_view(rows, labels, recovered)
    paired = None
    if first_view["metrics"] is not None and recovered_view["metrics"] is not None:
        corrected = sum(first[r.id]["label"] != r.label and recovered[r.id]["label"] == r.label for r in rows)
        harmed = sum(first[r.id]["label"] == r.label and recovered[r.id]["label"] != r.label for r in rows)
        paired = {"n_rows": len(rows), "corrected": corrected, "harmed": harmed,
            "accuracy_delta": recovered_view["metrics"]["accuracy"] - first_view["metrics"]["accuracy"],
            "note": "Retry availability effect; successful first responses are preserved, so this is not evidence of better semantic decisions."}
    return {"schema_version": 1, "condition_id": next(iter(conditions)),
        "selection_rule": "preserve original success; otherwise earliest valid success by append-ordered attempt index",
        "original_snapshot": summarize_view(rows, labels, originals), "first_attempt": first_view,
        "recovered": recovered_view, "paired_first_to_recovered": paired,
        "n_new_calls": len(retry_attempts), "post_success_attempts_ignored": post_success_attempts,
        "all_service_attempt_outcome_counts": all_attempt_outcomes,
        "retry_policy_deviation": post_success_attempts > 0,
        "original_snapshot_same_request_comparable": not any(stage == "upstream_not_called" for stage in stages.values()),
        "latency_measurement_status": "not_reported_for_mixed_attempts", "cost_accounting_status": "caller_must_audit_ledger",
        "rows": selections}
