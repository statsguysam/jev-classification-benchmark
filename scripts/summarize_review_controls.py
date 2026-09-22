#!/usr/bin/env python3
"""Audit matched proposal controls offline; never score partial primary arms.

The frozen protocol and request bundle are rebuilt from their pinned sources.
Any available execution must pass the runner's complete prefix/ledger audit.
Only complete dataset/arm predictions receive metrics. Serving-repeat agreement
requires every declared repeat and its no-proposal reference to be available.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import prepare_review_controls as preparation
import run_review_controls as execution
from jevbench.metrics import evaluate
from jevbench.types import Prediction
from summarize_numeric_decisions import chosen, transitions
from summarize_tabular import group_bootstrap

PREPARED = ROOT / "results/review_controls/prepared_full"
OUTPUT = ROOT / "results/review_controls"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_complete(rows, predictions, classes):
    metrics = evaluate(rows, predictions, classes)
    missing = sorted(set(range(classes)) - {row.label for row in rows})
    metrics["balanced_accuracy_missing_classes"] = missing
    metrics["balanced_accuracy_status"] = "unavailable: missing declared true classes" if missing else "available"
    if missing:
        metrics["balanced_accuracy"] = None
    return metrics


def repeat_diagnostic(requests, saved, expected):
    repeats = [r for r in requests if r["arm"] == preparation.REPEAT_ARM]
    require(len(repeats) == expected, "Repeat inventory differs from frozen protocol")
    by_id = {r["request_id"]: r for r in requests}
    complete_pairs = sum(r["request_id"] in saved and r["repeat_of_request_id"] in saved for r in repeats)
    result = {"status": "disabled" if not expected else "complete" if complete_pairs == expected else "pending",
              "expected_pairs": expected, "available_complete_pairs": complete_pairs,
              "counts": None, "by_dataset": None,
              "interpretation": "Descriptive, prospectively selected identical-prompt pairs; not independent architecture trials or proof of deterministic serving. Failures remain separate from label disagreements."}
    if not expected or complete_pairs != expected:
        return result
    count_keys = ("both_valid", "valid_label_agreement", "valid_label_disagreement",
                  "reference_failed_only", "repeat_failed_only", "both_failed",
                  "resolved_model_mismatch_among_valid", "unknown_resolved_model_among_valid")
    totals = Counter({key: 0 for key in count_keys})
    per_dataset = {}
    for request in repeats:
        reference_request = by_id.get(request["repeat_of_request_id"])
        require(reference_request and reference_request["arm"] == "no_proposal" and
                all(reference_request[key] == request[key] for key in ("row_id", "case_id", "dataset", "prompt", "prompt_sha256", "choices", "proposal_value")),
                "Repeat is not an identical no-proposal request")
        a, b = saved[reference_request["request_id"]], saved[request["request_id"]]
        counts = per_dataset.setdefault(request["dataset"], Counter({key: 0 for key in count_keys}))
        valid_a = not a.error and a.label is not None
        valid_b = not b.error and b.label is not None
        if valid_a and valid_b:
            counts["both_valid"] += 1
            counts["valid_label_agreement" if a.label == b.label else "valid_label_disagreement"] += 1
            ma, mb = a.metadata.get("resolved_model"), b.metadata.get("resolved_model")
            if ma is None or mb is None:
                counts["unknown_resolved_model_among_valid"] += 1
            elif ma != mb:
                counts["resolved_model_mismatch_among_valid"] += 1
        else:
            counts["both_failed" if not valid_a and not valid_b else "reference_failed_only" if not valid_a else "repeat_failed_only"] += 1
    for counts in per_dataset.values():
        totals.update(counts)
    require(totals["both_valid"] + totals["reference_failed_only"] + totals["repeat_failed_only"] + totals["both_failed"] == expected,
            "Repeat failure decomposition does not cover all pairs")
    result.update(counts=dict(totals), by_dataset={key: dict(value) for key, value in per_dataset.items()},
        valid_pair_agreement=totals["valid_label_agreement"] / totals["both_valid"] if totals["both_valid"] else None,
        valid_agreement_fraction_of_all_pairs=totals["valid_label_agreement"] / expected)
    return result


def assemble(plan, datasets, predictions, *, samples=None):
    """Pure report assembly after provenance audit; useful for synthetic tests."""
    protocol, requests = plan["protocol"], plan["requests"]
    samples = protocol["bootstrap_samples"] if samples is None else samples
    require(samples >= 100, "Use at least 100 bootstrap samples")
    require(len(predictions) <= len(requests), "Too many predictions")
    saved = {}
    for request, prediction in zip(requests, predictions):
        execution.validate_prediction(prediction, request)
        require(request["request_id"] not in saved, "Duplicate saved control request")
        saved[request["request_id"]] = prediction
    by_arm, payloads = {}, {}
    for name, descriptor in protocol["datasets"].items():
        dataset = datasets[name]
        rows_by_id = {row.id: row for row in dataset.test}
        selected_ids = descriptor["selected_row_ids"]
        require(len(set(selected_ids)) == len(selected_ids) and set(selected_ids) <= set(rows_by_id), "Selected test IDs differ")
        rows = [rows_by_id[row_id] for row_id in selected_ids]
        require(len(dataset.labels) == descriptor["n_classes"], "Declared classes differ")
        for arm in protocol["primary_arms"]:
            wanted = [r for r in requests if r["dataset"] == name and r["arm"] == arm]
            request_for_row = {r["row_id"]: r for r in wanted}
            require(len(wanted) == len(rows) and set(request_for_row) == set(selected_ids), "Primary arm is missing or duplicates test rows")
            known = sum(request["request_id"] in saved for request in wanted)
            complete = known == len(wanted)
            item = {"dataset": name, "arm": arm, "status": "complete" if complete else "pending",
                    "expected_requests": len(wanted), "saved_requests": known,
                    "training_examples": len(descriptor["training_example_ids"]),
                    "metrics": None, "group_bootstrap": None, "failures": None}
            if complete:
                ordered = [saved[request_for_row[row.id]["request_id"]] for row in rows]
                values = chosen(ordered, len(dataset.labels))
                groups = [hashlib.sha256(row.text.encode()).hexdigest() for row in rows]
                item.update(metrics=score_complete(rows, ordered, len(dataset.labels)),
                    failures=[{"row_id": p.row_id, "error": p.error,
                               "reported_cost_usd": p.metadata.get("openrouter", {}).get("reported_cost_usd")}
                              for p, value in zip(ordered, values) if value == -1],
                    group_bootstrap=group_bootstrap([row.label for row in rows], values, groups,
                        n_classes=len(dataset.labels), samples=samples, seed=protocol["bootstrap_seed"]))
                payloads[name, arm] = {"y": [row.label for row in rows], "values": values, "groups": groups,
                                      "row_ids": selected_ids, "classes": len(dataset.labels)}
            by_arm[name, arm] = item
    comparisons = []
    for name in protocol["datasets"]:
        for kind, a, b in (("actual_minus_no_proposal", "actual", "no_proposal"),
                           ("actual_minus_shuffled", "actual", "shuffled")):
            item = {"dataset": name, "contrast": kind, "a": a, "b": b, "status": "pending",
                    "paired_group_bootstrap": None, "balanced_accuracy_delta": None,
                    "transitions_B_to_A": None,
                    "interpretation": "A minus B; exploratory unadjusted interval conditional on frozen cases, prompts, examples and served responses"}
            if (name, a) in payloads and (name, b) in payloads:
                pa, pb = payloads[name, a], payloads[name, b]
                require(all(pa[key] == pb[key] for key in ("y", "groups", "row_ids", "classes")), "Unpaired control contrast")
                ma, mb = by_arm[name, a]["metrics"], by_arm[name, b]["metrics"]
                item.update(status="complete", paired_group_bootstrap=group_bootstrap(pa["y"], pa["values"], pa["groups"],
                    pb["values"], n_classes=pa["classes"], samples=samples, seed=protocol["bootstrap_seed"]),
                    balanced_accuracy_delta=(ma["balanced_accuracy"] - mb["balanced_accuracy"]
                        if ma["balanced_accuracy"] is not None and mb["balanced_accuracy"] is not None else None),
                    transitions_B_to_A=transitions(pa["y"], pb["values"], pa["values"]))
            comparisons.append(item)
    return {"schema_version": 1, "study": protocol["study"], "study_role": protocol["study_role"],
        "status": "complete" if len(predictions) == len(requests) else "pending_no_execution" if not predictions else "incomplete_execution",
        "expected_requests": len(requests), "saved_requests": len(predictions),
        "expected_primary_arms": len(by_arm), "complete_primary_arms": sum(r["status"] == "complete" for r in by_arm.values()),
        "expected_primary_contrasts": len(comparisons), "complete_primary_contrasts": sum(r["status"] == "complete" for r in comparisons),
        "partial_arm_metrics_reported": False, "bootstrap_samples": samples, "bootstrap_seed": protocol["bootstrap_seed"],
        "runs": list(by_arm.values()), "comparisons": comparisons,
        "serving_repeat_diagnostic": repeat_diagnostic(requests, saved, protocol["repeat_request_count"]),
        "saved_failure_records": [{"request_id": request["request_id"], "dataset": request["dataset"],
            "arm": request["arm"], "row_id": p.row_id, "error": p.error,
            "reported_cost_usd": p.metadata.get("openrouter", {}).get("reported_cost_usd")}
            for request, p in zip(requests, predictions) if p.error],
        "limitations": ["Existing holdouts were already observed; this is exploratory, not a fresh confirmatory test.",
            "Only the proposal-slot value differs; no-proposal uses an explicit not-provided slot.",
            "One deterministic, label-blind proposal permutation preserves frequencies but can retain the same label.",
            "Failed responses count as incorrect; incomplete arms have no accuracy or F1.",
            "Group bootstrap resamples identical serialized-input groups; intervals exclude retraining and unseen serving variability.",
            "Serving-repeat metrics require all declared pairs and keep failed responses separate.",
            "No aggregate winner, causal architecture claim, deployable confidence threshold, or dollar savings is established."]}


def load_execution(plan, prepared=PREPARED):
    directory = execution.OUTPUT
    record_path = directory / "run.json"
    if not record_path.exists():
        orphans = [directory / name for name in ("predictions.jsonl", "budget.jsonl", "budget.jsonl.lock", "run.json.tmp")]
        require(not any(path.exists() for path in orphans), "Orphan execution evidence without run.json; do not treat it as unexecuted")
        return [], {"status": "not_started", "budget": None, "artifact_sha256": {}}
    record = read(record_path)
    prior = record["prior_snapshot"]
    execution.check_prior(prior)
    require(prior == execution.readiness(), "Saved prior allocation differs from audited completed studies")
    inner = execution.budget.Ledger(execution.LEDGER, prior["allocation_usd"])
    identity = {"schema_version": 1, "study": "proposal-value-controls-v1",
        "producer_pins": execution.producer_pins(), "protocol_sha256": file_sha(prepared / "protocol.json"),
        "requests_sha256": file_sha(prepared / "requests.jsonl"), "ledger_id": inner.identity["ledger_id"],
        "prior_snapshot": prior, "request_count": len(plan["requests"]), "route": execution.route.ENDPOINT,
        "price": execution.route.PRICE, "settlement_version": execution.historical.SETTLEMENT_VERSION}
    require(all(record.get(key) == value for key, value in identity.items()), "Control execution identity changed")
    ledger = execution.ControlLedger(inner, plan, identity)
    predictions = execution.audit_saved(plan, identity, ledger, directory / "predictions.jsonl")
    require(record.get("status") in {"running", "paused_request_limit", "paused_budget", "halted_billing", "halted_errors", "complete"},
            "Unknown execution status")
    require(record["status"] != "complete" or len(predictions) == len(plan["requests"]), "Complete execution is missing requests")
    return predictions, {"status": record["status"], "started_at": record.get("started_at"),
        "completed_at": record.get("completed_at"), "execution_sessions": record.get("execution_sessions", []),
        "budget": ledger.snapshot(), "artifact_sha256": {name: file_sha(directory / name)
            for name in ("run.json", "predictions.jsonl", "budget.jsonl", "budget.jsonl.lock") if (directory / name).is_file()},
        "resolved_models": sorted({p.metadata["resolved_model"] for p in predictions if p.metadata.get("resolved_model")})}


def collect(prepared=PREPARED):
    prepared = Path(prepared).resolve()
    require(prepared == execution.PREPARED.resolve(), "Summarize the canonical full frozen control study")
    plan = preparation.load_frozen_plan(prepared)
    require(plan["protocol"]["study_role"] == "primary_full_fixed_holdouts", "Operational pilot cannot replace the primary report")
    jobs = preparation.load_jobs()
    predictions, observed = load_execution(plan, prepared)
    report = assemble(plan, {name: job["dataset"] for name, job in jobs.items()}, predictions)
    report.update(execution=observed,
        protocol={"path": (prepared / "protocol.json").relative_to(ROOT).as_posix(),
            "files_sha256": {name: file_sha(prepared / name) for name in ("protocol.json", "requests.jsonl", "manifest.json")},
            "datasets": plan["protocol"]["datasets"]},
        analysis_source_sha256=file_sha(__file__), analysis_tests_sha256=file_sha(ROOT / "tests/test_review_controls_summary.py"))
    return report


def findings(report):
    lines = ["# Matched proposal-value controls", "",
        f"**{report['complete_primary_arms']}/{report['expected_primary_arms']} primary arms complete; {report['saved_requests']}/{report['expected_requests']} requests saved.** Every incomplete arm remains unscored. This report performs no API calls.", "",
        "[Audited machine report](COMPARISON.json) · [Frozen protocol](prepared_full/protocol.json)", "",
        "Each frozen case receives an actual Qwen3 four-shot proposal, an explicit no-proposal slot, and a shuffled proposal. Input, examples, choices and all other prompt wording are identical. The primary contrasts are actual minus no-proposal and actual minus shuffled, separately for each dataset. Existing direct Jev results provide context and are not substituted for this matched no-proposal arm.", "",
        "| Dataset | Arm | Saved/expected | Status | Accuracy | Balanced accuracy | Macro-F1 | Failures |",
        "|---|---|---:|---|---:|---:|---:|---:|"]
    def metric(value):
        return "pending" if value is None else f"{100 * value:.2f}%"
    for row in report["runs"]:
        m = row["metrics"] or {}
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['saved_requests']}/{row['expected_requests']} | {row['status']} | {metric(m.get('accuracy'))} | {metric(m.get('balanced_accuracy'))} | {metric(m.get('macro_f1'))} | {m.get('n_failures', 'pending')} |")
    lines += ["", "| Dataset | Contrast | Accuracy difference, pp [95% group bootstrap] | Fixed | Harmed |",
        "|---|---|---|---:|---:|"]
    for row in report["comparisons"]:
        interval, fixed, harmed = "pending", "pending", "pending"
        if row["status"] == "complete":
            estimate = row["paired_group_bootstrap"]["metrics"]["accuracy"]
            interval = f"{100 * estimate['estimate']:+.2f} [{100 * estimate['ci95'][0]:+.2f}, {100 * estimate['ci95'][1]:+.2f}]"
            fixed, harmed = row["transitions_B_to_A"]["wrong_to_correct"], row["transitions_B_to_A"]["correct_to_wrong"]
        lines.append(f"| {row['dataset']} | {row['contrast']} | {interval} | {fixed} | {harmed} |")
    repeated = report["serving_repeat_diagnostic"]
    lines += ["", f"**Serving-repeat diagnostic: {repeated['status']} ({repeated['available_complete_pairs']}/{repeated['expected_pairs']} pairs available).** Agreement metrics are withheld until every declared repeat and its reference is present."]
    if repeated["counts"] is not None:
        c = repeated["counts"]
        lines += ["", f"Among {c['both_valid']} pairs with two valid responses, {c['valid_label_agreement']} agree and {c['valid_label_disagreement']} disagree. Reference-only failures: {c['reference_failed_only']}; repeat-only failures: {c['repeat_failed_only']}; both failed: {c['both_failed']}. Both failures are not counted as label agreement. Resolved-model differences and per-dataset counts are in the JSON."]
    lines += ["", "All failures remain incorrect in primary metrics, including unknown transport outcomes; missing costs stay unknown. The repeat sample is predefined and serves a separate descriptive diagnostic. Repeated prompts and primary arms share test cases, so these observations are not independent architecture trials.", "",
        "These holdouts were already observed. Intervals are exploratory, unadjusted group-bootstrap intervals conditional on the frozen cases, demonstrations, prompts and observed responses; they do not establish generalization to new data or deterministic serving. Full matched arms are required for comparisons, and no partial result is promoted to a headline. No paid request, budget allocation, or historical producer/report modification occurs in this summarizer.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    args = parser.parse_args(argv)
    report = collect(args.prepared)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "COMPARISON.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (OUTPUT / "FINDINGS.md").write_text(findings(report))
    print(json.dumps({key: report[key] for key in ("status", "saved_requests", "expected_requests", "complete_primary_arms", "complete_primary_contrasts")}))


if __name__ == "__main__":
    main()
