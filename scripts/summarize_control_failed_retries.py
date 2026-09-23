"""Audit a secondary failure-recovery view of the completed matched controls.

Every primary and repeat first-attempt record remains unchanged. This report
uses only the earliest valid separately audited recovery of an original failed
request, without consulting its true label. It performs no inference or I/O to
any service. Partial retry executions are never scored.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import analyze_jev_recovery as analysis
import run_control_failed_retries as runner
import summarize_review_controls as primary
from jevbench.types import Prediction
from summarize_numeric_decisions import chosen, transitions
from summarize_tabular import group_bootstrap

OUTPUT = ROOT / "results/completion_20260923"
FILE = OUTPUT / "CONTROL_RECOVERY_COMPARISON.json"
require = analysis.require
METRICS = (*analysis.METRIC_FIELDS, "n_test")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(rows, predictions, classes, samples, seed):
    measured = primary.score_complete(rows, predictions, classes)
    metrics = {key: measured[key] for key in METRICS}
    metrics["n_rows"] = len(rows)
    values = chosen(predictions, classes)
    return {"metrics": metrics, "group_bootstrap": group_bootstrap(
        [r.label for r in rows], values, [hashlib.sha256(r.text.encode()).hexdigest() for r in rows],
        n_classes=classes, samples=samples, seed=seed)}


def overlay(plan, datasets, originals, audited):
    """Select on response validity alone; do not alter any original object."""
    require(audited["record"]["status"] == "complete", "Post-control recovery is not complete")
    requests = plan["requests"]
    require(len(originals) == len(requests), "All primary and repeat original requests are required")
    by_request, request_by_id = {}, {}
    for request, prediction in zip(requests, originals):
        primary.execution.validate_prediction(prediction, request)
        request_id = request["request_id"]
        require(request_id not in by_request, "Duplicate original control request")
        analysis.validate_prediction(asdict(prediction), request["row_id"], len(datasets[request["dataset"]].labels),
                                     require_probabilities=True)
        by_request[request_id], request_by_id[request_id] = prediction, request
    items = audited["plan"]["inventory"]["failures"]
    failure_by_id = {item["failure_id"]: item for item in items}
    failed_requests = {item["control_request_id"] for item in items}
    require(len(failure_by_id) == len(failed_requests) == len(items), "Duplicate control recovery inventory")
    require(failed_requests == {key for key, p in by_request.items() if p.error},
            "Recovery inventory must cover every original primary and repeat failure")
    require(audited["record"]["no_op"] == (not items), "No-op differs from the failure inventory")
    for item in items:
        request, original = request_by_id[item["control_request_id"]], by_request[item["control_request_id"]]
        require(item["dataset"] == request["dataset"] and item["control_arm"] == request["arm"]
                and item["original_prediction_sha256"] == runner.preparation.prediction_sha256(asdict(original)),
                "Original control failure lineage differs")
    attempts_by_id = defaultdict(list)
    selected = dict(by_request)
    outcomes = Counter()
    for attempt in audited["attempts"]:
        prediction = attempt["prediction"]
        failure_id = prediction["metadata"]["retry"]["failure_id"]
        require(failure_id in failure_by_id, "Attempt targets an original success or an unknown failure")
        item = failure_by_id[failure_id]
        request_id = item["control_request_id"]
        request, original = request_by_id[request_id], by_request[request_id]
        sequence = attempts_by_id[failure_id]
        require(attempt["stage"] == "retry" and attempt["attempt_index"] == len(sequence) + 1
                and len(sequence) < 2, "Control retries must be consecutive and bounded to two")
        require(not sequence or sequence[-1]["prediction"]["error"], "Attempt after a valid success is forbidden")
        identity = attempt["identity"]
        require(identity["row_id"] == request["row_id"] and identity["prompt_sha256"] == request["prompt_sha256"]
                and identity["original_prediction_sha256"] == analysis.digest(asdict(original)),
                "Retry does not match the original request and prediction")
        expected_lineage = {"control_request_id": request_id, "dataset": request["dataset"], "arm": request["arm"],
                            "original_control_prediction_sha256": item["original_prediction_sha256"]}
        require(prediction["metadata"].get("post_control") == expected_lineage, "Retry control lineage differs")
        valid = analysis.validate_prediction(prediction, request["row_id"], len(datasets[request["dataset"]].labels),
                                             require_probabilities=True)
        sequence.append(attempt)
        # A success, even an incorrect class, terminates selection. If neither
        # recovery succeeds, preserve the final failure as an incorrect result.
        selected[request_id] = Prediction(**deepcopy(prediction))
        outcomes[analysis.failure_kind(prediction)] += 1
        require(valid or selected[request_id].error, "Malformed failed control retry")
    require(set(attempts_by_id) == set(failure_by_id), "A completed recovery omitted an original failure")
    require(all(not values[-1]["prediction"]["error"] or len(values) == 2 for values in attempts_by_id.values()),
            "Failed request has an unfinished finite retry policy")
    if not items:
        require(not audited["attempts"] and audited["budget"]["reservations"] == 0,
                "No-op contains paid recovery attempts")
    return selected, dict(outcomes)


def assemble(plan, datasets, originals, audited, *, samples=2000):
    """Pure aggregate assembly; caller audits the original and retry ledgers."""
    require(samples >= 100, "Use at least 100 bootstrap samples")
    require(set(plan["protocol"]["datasets"]) == set(primary.preparation.DATASETS)
            and plan["protocol"]["primary_arms"] == list(primary.preparation.PRIMARY_ARMS),
            "Recovery must retain all four datasets and three original arms")
    first = primary.assemble(plan, datasets, originals, samples=samples)
    require(first["status"] == "complete", "Primary control study is incomplete")
    selected, outcomes = overlay(plan, datasets, originals, audited)
    original_by_id = {request["request_id"]: p for request, p in zip(plan["requests"], originals)}
    first_runs = {(r["dataset"], r["arm"]): r for r in first["runs"]}
    seed = plan["protocol"]["bootstrap_seed"]
    runs, payloads = [], {}
    for name, descriptor in plan["protocol"]["datasets"].items():
        dataset = datasets[name]
        row_by_id = {r.id: r for r in dataset.test}
        rows = [row_by_id[row_id] for row_id in descriptor["selected_row_ids"]]
        classes = len(dataset.labels)
        y, groups = [r.label for r in rows], [hashlib.sha256(r.text.encode()).hexdigest() for r in rows]
        for arm in plan["protocol"]["primary_arms"]:
            request_by_row = {r["row_id"]:r["request_id"] for r in plan["requests"] if r["dataset"] == name and r["arm"] == arm}
            old = [original_by_id[request_by_row[r.id]] for r in rows]
            new = [selected[request_by_row[r.id]] for r in rows]
            first_view, recovered = score(rows, old, classes, samples, seed), score(rows, new, classes, samples, seed)
            require(all(first_view["metrics"][key] == first_runs[name, arm]["metrics"][key] for key in METRICS),
                    "Original primary metrics changed during recovery analysis")
            old_values, values = chosen(old, classes), chosen(new, classes)
            changed = transitions(y, old_values, values)
            require(changed["correct_to_wrong"] == 0, "Failure-only recovery changed a correct original response")
            runs.append({"dataset": name, "arm": arm, "status": "complete", "n_rows": len(rows),
                "first_attempt": first_view, "recovered": recovered,
                "resolved_failures": first_view["metrics"]["n_failures"] - recovered["metrics"]["n_failures"],
                "transitions_first_to_recovered": changed,
                "paired_first_to_recovered": group_bootstrap(y, values, groups, old_values,
                    n_classes=classes, samples=samples, seed=seed)})
            payloads[name, arm] = {"y": y, "groups": groups, "classes": classes, "values": values,
                                   "metrics": recovered["metrics"]}
    comparisons = []
    for original in first["comparisons"]:
        name, a, b = original["dataset"], original["a"], original["b"]
        pa, pb = payloads[name, a], payloads[name, b]
        require(all(pa[k] == pb[k] for k in ("y", "groups", "classes")), "Recovery contrast is not paired")
        ma, mb = pa["metrics"], pb["metrics"]
        recovered = {"paired_group_bootstrap": group_bootstrap(pa["y"], pa["values"], pa["groups"], pb["values"],
                n_classes=pa["classes"], samples=samples, seed=seed),
            "balanced_accuracy_delta": None if ma["balanced_accuracy"] is None or mb["balanced_accuracy"] is None
                else ma["balanced_accuracy"] - mb["balanced_accuracy"],
            "transitions_B_to_A": transitions(pa["y"], pb["values"], pa["values"])}
        comparisons.append({"dataset": name, "a": a, "b": b, "contrast": original["contrast"], "status": "complete",
            "first_attempt": {k: deepcopy(original[k]) for k in recovered}, "recovered": recovered})
    failed_by_scope, remaining_by_scope, calls_by_scope = Counter(), Counter(), Counter()
    for request, original in zip(plan["requests"], originals):
        scope = "repeat" if request["arm"] == primary.preparation.REPEAT_ARM else "primary"
        failed_by_scope[scope] += bool(original.error)
        remaining_by_scope[scope] += bool(selected[request["request_id"]].error)
    for attempt in audited["attempts"]:
        scope = "repeat" if attempt["prediction"]["metadata"]["post_control"]["arm"] == primary.preparation.REPEAT_ARM else "primary"
        calls_by_scope[scope] += 1
    return {"schema_version": 1, "study": "recovered-control-sensitivity-v1", "status": "complete",
        "study_role": "secondary_failure_recovery_sensitivity", "no_op": audited["record"]["no_op"],
        "selection_rule": "Preserve every original success; otherwise select the earliest valid recovery regardless of correctness. Two failures remain incorrect.",
        "primary_report": "results/review_controls/COMPARISON.json", "runs": runs, "comparisons": comparisons,
        "first_attempt_serving_repeat_diagnostic": deepcopy(first["serving_repeat_diagnostic"]),
        "recovery_counts": {scope: {"original_failures": failed_by_scope[scope], "remaining_failures": remaining_by_scope[scope],
            "resolved_failures": failed_by_scope[scope] - remaining_by_scope[scope], "new_calls": calls_by_scope[scope]}
            for scope in ("primary", "repeat")},
        "new_call_outcomes": {"n_calls": len(audited["attempts"]), "counts": outcomes},
        "budget": deepcopy(audited["budget"]), "bootstrap_samples": samples, "bootstrap_seed": seed,
        "latency_measurement_status": "not_reported_for_mixed_attempts",
        "limitations": ["This is a secondary availability sensitivity view on the same frozen cases, not a new test set.",
            "Primary first-attempt controls and the original serving-repeat diagnostic remain unchanged.",
            "An incorrect but valid recovered label is retained; true labels never select a retry.",
            "Every error remaining after the finite policy counts as incorrect with the full original denominator.",
            "Intervals are exploratory unadjusted group bootstraps conditional on frozen cases and served responses; balanced-accuracy deltas have no interval.",
            "Retries use additional calls and retain their full conservative reservation; this does not estimate deployment savings."]}


def collect(samples=2000):
    audited = runner.audit_run()
    primary_path = primary.OUTPUT / "COMPARISON.json"
    require(primary_path.is_file() and not primary_path.is_symlink(), "Generate the audited primary control report first")
    primary_sha = file_sha(primary_path)
    require(json.loads(primary_path.read_text()) == primary.collect(),
            "Saved primary control report is stale; regenerate it before recovery analysis")
    plan = primary.preparation.load_frozen_plan(primary.PREPARED)
    originals, execution = primary.load_execution(plan)
    require(execution["status"] == "complete", "Original control execution is incomplete")
    jobs = primary.preparation.load_jobs()
    report = assemble(plan, {name: job["dataset"] for name, job in jobs.items()}, originals, audited, samples=samples)
    report["provenance"] = {"recovery_artifact_sha256": audited["artifact_sha256"],
        "original_control_artifact_sha256": execution["artifact_sha256"],
        "prepared_files_sha256": {name:file_sha(primary.PREPARED / name) for name in ("protocol.json", "manifest.json", "requests.jsonl")},
        "primary_report_sha256": primary_sha,
        "analysis_sha256": file_sha(__file__), "primary_analysis_sha256": file_sha(primary.__file__),
        "attempt_records_sha256": analysis.digest(audited["attempts"])}
    require(runner.audit_run()["artifact_sha256"] == audited["artifact_sha256"], "Recovery artifacts changed during analysis")
    require(all(file_sha(primary.execution.OUTPUT / name) == value for name, value in execution["artifact_sha256"].items()),
            "Primary artifacts changed during recovery analysis")
    require(file_sha(primary_path) == primary_sha, "Primary report changed during recovery analysis")
    return report


def optional_collect(samples=2000):
    path = runner.AREA / "run.json"
    if not path.exists():
        require(not any((runner.AREA / name).exists() for name in ("plan.json", "attempts.jsonl", "budget.jsonl", "budget.jsonl.lock")),
                "Orphan post-control recovery evidence")
        return None
    status = json.loads(path.read_text()).get("status")
    if status == "complete":
        return collect(samples=samples)
    require(status in {"running", "paused", "budget_exhausted", "halted_errors", "halted_billing", "interrupted"},
            "Unknown post-control recovery status")
    return None


def findings(report):
    counts = report["recovery_counts"]
    lines = ["# Control failure-recovery sensitivity", "",
        "[Audited secondary report](CONTROL_RECOVERY_COMPARISON.json) · [Original primary controls](../review_controls/COMPARISON.json)", "",
        "This separate overlay retains all 12 full primary arms and eight paired contrasts. Original first attempts and the original serving-repeat diagnostic remain unchanged.", "",
        "**No-op: every original control and repeat response was valid; secondary scores equal the primary scores.**" if report["no_op"] else
        f"**{report['new_call_outcomes']['n_calls']} additional calls.** Primary failures: {counts['primary']['original_failures']} → {counts['primary']['remaining_failures']}; repeat failures: {counts['repeat']['original_failures']} → {counts['repeat']['remaining_failures']}. Recovered repeat agreement is not substituted for the original diagnostic.", "",
        "| Dataset | Arm | Full N | First accuracy | Recovered accuracy | First / recovered failures |", "|---|---|---:|---:|---:|---:|"]
    for row in report["runs"]:
        a, b = row["first_attempt"]["metrics"], row["recovered"]["metrics"]
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['n_rows']} | {100*a['accuracy']:.2f}% | {100*b['accuracy']:.2f}% | {a['n_failures']} / {b['n_failures']} |")
    lines += ["", "| Dataset | Contrast | First difference, pp [95% CI] | Recovered difference, pp [95% CI] |", "|---|---|---|---|"]
    for row in report["comparisons"]:
        def interval(view):
            m = row[view]["paired_group_bootstrap"]["metrics"]["accuracy"]
            return f"{100*m['estimate']:+.2f} [{100*m['ci95'][0]:+.2f}, {100*m['ci95'][1]:+.2f}]"
        lines.append(f"| {row['dataset']} | {row['contrast']} | {interval('first_attempt')} | {interval('recovered')} |")
    lines += ["", f"Conservative added charge/reservation: ${report['budget']['charged_or_reserved_usd']}; allocation ${report['budget']['budget_usd']}. Every reservation is retained.", ""]
    lines += ["- " + note for note in report["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    report = collect(samples=args.bootstrap_samples)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runner.atomic_json(FILE, report)
    (OUTPUT / "CONTROL_RECOVERY_FINDINGS.md").write_text(findings(report))
    print(json.dumps({"status": report["status"], "no_op": report["no_op"], "new_calls": report["new_call_outcomes"]["n_calls"]}))


if __name__ == "__main__":
    main()
