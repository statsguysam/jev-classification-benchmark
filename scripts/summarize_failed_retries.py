"""Audited, aggregate recovery views; original evidence is never rewritten.

Scores are available only after the finite retry policy is complete and its
runner audits every source, request, prediction, reservation and settlement.
First actual service attempts and earliest valid recovered responses are kept
separate from the historical snapshot (which can include an upstream skip).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_failed_retries as runner
import analyze_jev_recovery as analysis
from summarize_tabular import group_bootstrap

OUTPUT = ROOT / "results/completion_20260923"
FILE = OUTPUT / "RECOVERY_COMPARISON.json"
require = analysis.require
VIEWS = ("original_snapshot", "first_attempt", "recovered")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_original(item):
    """Load the entire original condition, not just the selected failed rows."""
    path = runner.preparation.canonical_path(ROOT, item["run_path"])
    record = runner.preparation.read(path)
    dataset = runner.preparation.dataset_for(record, ROOT)
    values = runner.preparation.read_predictions(path.with_name("predictions.jsonl"))
    require(record.get("status") == "complete" and record.get("run_id") == item["run_id"],
            "An original condition is not complete or has a different identity")
    require([p["row_id"] for p in values] == [r.id for r in dataset.test],
            "Full original condition must match every ordered test row")
    measured = analysis.summarize_view(dataset.test, dataset.labels, {p["row_id"]:p for p in values})
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "n_failures"):
        require(measured["metrics"][key] == record["metrics"].get(key), "Original full-condition metric differs")
    return dataset, record, values


def original_reservations(plan):
    """Only read pinned earlier ledgers, never a concurrent later control ledger."""
    result = {}
    for relative, expected in plan["prior"]["files_sha256"].items():
        if relative.endswith(".lock"):
            continue
        path = runner.preparation.canonical_path(ROOT, relative)
        require(file_sha(path) == expected, "Pinned original ledger changed")
        events = runner.preparation.read_predictions(path)
        identity = events[0]["ledger_id"]
        require(identity not in result, "Duplicate original ledger identity")
        result[identity] = {e["reservation_id"]:e["details"] for e in events if e["type"] == "reserve"}
    return result


def original_identity(value, record, labels, reservations):
    """Recover an actually sent prompt digest from its pinned reservation."""
    meta = value.get("metadata", {})
    ownership = meta.get("budget", {})
    details = reservations.get(ownership.get("ledger_id"), {}).get(ownership.get("reservation_id"))
    require(details is not None and details.get("row_id") == value["row_id"]
            and details.get("model") == record["config"]["model"], "Original response lacks its exact paid reservation")
    prompt_sha = details.get("prompt_sha256")
    require(analysis.valid_sha(prompt_sha), "Original request has no valid prompt digest")
    if meta.get("review_prompt_sha256") is not None:
        require(meta["review_prompt_sha256"] == prompt_sha, "Original review and reservation prompt differ")
    return {"condition_id":record["run_id"],"row_id":value["row_id"],"prompt_sha256":prompt_sha,
        "choices_sha256":analysis.digest(labels),
        "source_sha256":analysis.digest(meta["proposal"]) if meta.get("proposal") is not None else None,
        "original_prediction_sha256":analysis.digest(value)}


def pending_service_views(rows, labels, originals, attempts):
    first, recovered = {}, {}
    for row in rows:
        prior = originals[row.id]
        values = ([] if analysis.failure_kind(prior) == "upstream_not_called" else [prior])
        values += [a["prediction"] for a in attempts[row.id]]
        first[row.id] = values[0] if values else None
        recovered[row.id] = next((p for p in values if not p["error"]), values[-1] if values else None)
    # Whole-fold summaries retain outcome counts, but cannot score any view with
    # an unattempted row. No prompt hash or partial-fold accuracy is fabricated.
    return (analysis.summarize_view(rows, labels, first), analysis.summarize_view(rows, labels, recovered))


def summarize_condition(dataset, record, originals, items, attempts, reservations, *, samples):
    rows, labels = dataset.test, dataset.labels
    by_row = {p["row_id"]:p for p in originals}
    require(len(by_row) == len(rows) and set(by_row) == {r.id for r in rows}, "Incomplete original condition")
    for value in originals:
        analysis.validate_prediction(value, value["row_id"], len(labels),
                                     require_probabilities=record["config"]["provider"] == "jev")
    failures = {item["row_id"]:item for item in items}
    require(len(failures) == len(items) and {p["row_id"] for p in originals if p["error"]} == set(failures),
            "Frozen inventory must cover every original failure in the condition")
    for item in items:
        require(runner.preparation.prediction_sha256(by_row[item["row_id"]]) == item["original_prediction_sha256"],
                "Full condition failure digest differs from the frozen inventory")
    attempts_by_row = defaultdict(list)
    for attempt in attempts:
        row_id = attempt["identity"]["row_id"]
        require(row_id in failures, "Recovery attempt targets an original success or unknown row")
        require(attempt["prediction"]["metadata"]["retry"]["failure_id"] == failures[row_id]["failure_id"],
                "Attempt does not belong to this original failure")
        attempts_by_row[row_id].append(attempt)
    dependent_rows = [r.id for r in rows if analysis.failure_kind(by_row[r.id]) == "upstream_not_called"]
    unresolved = [row_id for row_id in dependent_rows if not attempts_by_row[row_id]]
    paid_missing = [item["row_id"] for item in items if item["kind"] == "paid_failure" and not attempts_by_row[item["row_id"]]]
    require(not paid_missing, "Completed finite policy omitted a paid-failure attempt")
    source_models = {p["metadata"]["proposal"]["source_model"] for p in originals if p.get("metadata", {}).get("proposal")}
    require(len(source_models) <= 1, "Original condition mixes source models")
    descriptor = {"original_run_id":record["run_id"],"dataset":record["dataset"],"model":record["config"]["model"],
        "provider":record["config"]["provider"],"source_model":next(iter(source_models),None),
        "method":record["method"],"shots_per_class":record["config"]["shots_per_class"],
        "scope":items[0]["scope"],"n_rows":len(rows),"n_new_calls":len(attempts),
        "dependent_review_first_calls":sum(a["stage"] == "dependent_first_call" for a in attempts),
        "same_request_as_original_snapshot":not bool(dependent_rows)}
    outcome_counts = Counter(analysis.failure_kind(a["prediction"]) for a in attempts)
    descriptor["new_service_outcome_counts"] = dict(sorted(outcome_counts.items()))
    descriptor["new_call_stage_counts"] = dict(sorted(Counter(a["stage"] for a in attempts).items()))
    descriptor["identity_provenance"] = {
        "original_attempt_identity_sha256":[analysis.digest(a["identity"]) for a in attempts],
        "attempt_records_sha256":analysis.digest(attempts),
        "condition_grouping":"An analysis-only condition alias groups full-fold outcomes; original audited attempt identities and request hashes are retained."}
    descriptor['condition_variant'] = 'dependent_source_recovery_overlay' if dependent_rows else 'original_request_recovery'
    if unresolved:
        # An unsent request has no prompt hash. Do not invent one for the helper,
        # or publish selected-row scores in place of the complete condition.
        original_view = analysis.summarize_view(rows, labels, by_row)
        first_view, recovered_view = pending_service_views(rows, labels, by_row, attempts_by_row)
        return {**descriptor,"status":"incomplete","analysis_status":"unresolved_upstream_without_service_call",
            "unresolved_upstream_rows":len(unresolved),"original_snapshot":original_view,
            "first_attempt":first_view,"recovered":recovered_view,
            "paired_first_to_recovered":None,"paired_group_bootstrap":None,
            "group_bootstrap":{view:None for view in VIEWS},
            "selection_rule":"No full-fold service-attempt score while an upstream-dependent call is absent.",
            "latency_measurement_status":"not_reported_for_mixed_attempts"}
    alias = record["run_id"] + ("::dependent-source-recovery-overlay-v1" if dependent_rows else "")
    descriptor['identity_provenance']['condition_id_aliases'] = {
        value['identity']['condition_id']:alias for value in attempts}
    identities, normalized = {}, []
    for row in rows:
        values = attempts_by_row[row.id]
        identity = deepcopy(values[0]["identity"]) if values else original_identity(by_row[row.id], record, labels, reservations)
        require(identity["condition_id"] in {record["run_id"],record["run_id"]+"::dependent-source-recovery-v1"},
                "Unrecognized original or dependent condition identity")
        require(identity["row_id"] == row.id and identity["choices_sha256"] == analysis.digest(labels)
                and identity["original_prediction_sha256"] == analysis.digest(by_row[row.id]),
                "Recovery identity differs from original row/choices/outcome")
        identity["condition_id"] = alias
        identities[row.id] = identity
    for value in attempts:
        copied = deepcopy(value)
        copied["identity"]["condition_id"] = alias
        require(copied["identity"] == identities[copied["identity"]["row_id"]],
                "Repeated attempt changed its request or source identity")
        normalized.append(copied)
    result = analysis.analyze_condition(rows, labels, by_row, identities, normalized,
                                       require_probabilities=record["config"]["provider"] == "jev")
    require(not result["retry_policy_deviation"], "Completed recovery repeated an already successful response")
    groups = [hashlib.sha256(row.text.encode()).hexdigest() for row in rows]
    selections = result["rows"]
    values = {
        "original_snapshot":[-1 if by_row[r.id]["error"] else by_row[r.id]["label"] for r in rows],
        "first_attempt":[-1 if p["first_attempt_label"] is None else p["first_attempt_label"] for p in selections],
        "recovered":[-1 if p["recovered_label"] is None else p["recovered_label"] for p in selections]}
    uncertainty = {view:group_bootstrap([r.label for r in rows],values[view],groups,n_classes=len(labels),samples=samples,seed=42)
                   for view in VIEWS}
    paired = group_bootstrap([r.label for r in rows],values["recovered"],groups,values["first_attempt"],
                             n_classes=len(labels),samples=samples,seed=42)
    return {**descriptor,"status":"complete","analysis_status":"audited_full_condition",
        "unresolved_upstream_rows":0,**{view:result[view] for view in VIEWS},
        "selection_rule":result["selection_rule"],"paired_first_to_recovered":result["paired_first_to_recovered"],
        "paired_group_bootstrap":paired,"group_bootstrap":uncertainty,
        "latency_measurement_status":result["latency_measurement_status"]}


def assemble(audited, loaded, reservations, *, samples=2000):
    """Pure assembly; caller must supply audit_run output and audited originals."""
    require(type(samples) is int and samples >= 100, "Use at least 100 bootstrap draws")
    require(audited["record"]["status"] == "complete", "Recovery policy is not complete")
    items = audited["plan"]["inventory"]["failures"]
    by_run = defaultdict(list)
    by_id = {}
    for item in items:
        require(item["failure_id"] not in by_id, "Duplicate recovery failure identity")
        by_id[item["failure_id"]] = item
        by_run[item["run_id"]].append(item)
    attempts_by_run = defaultdict(list)
    for attempt in audited["attempts"]:
        fid = attempt["prediction"]["metadata"]["retry"]["failure_id"]
        require(fid in by_id, "Attempt is outside frozen failure inventory")
        attempts_by_run[by_id[fid]["run_id"]].append(attempt)
    require(set(loaded) == set(by_run), "Full-condition inventory differs")
    conditions = [summarize_condition(*loaded[run_id],entries,attempts_by_run[run_id],reservations,samples=samples)
                  for run_id,entries in sorted(by_run.items())]
    outcomes = Counter(analysis.failure_kind(a["prediction"]) for a in audited["attempts"])
    stages = Counter(a["stage"] for a in audited["attempts"])
    kinds = Counter(by_id[a["prediction"]["metadata"]["retry"]["failure_id"]]["kind"] for a in audited["attempts"])
    return {"schema_version":1,"study":"failed-call-recovery-report-v1","status":"complete",
        "execution_status":"complete","n_affected_conditions":len(conditions),
        "complete_service_condition_views":sum(c["status"]=="complete" for c in conditions),
        "conditions":conditions,"api_outcomes":{"new_calls":len(audited["attempts"]),
            "outcomes":dict(sorted(outcomes.items())),"stages":dict(sorted(stages.items())),
            "ordinary_paid_failure_retry_calls":kinds["paid_failure"],"dependent_review_calls":kinds["upstream_skip"],
            "dependent_first_calls":stages["dependent_first_call"],
            "upstream_unresolved_not_called":audited["record"]["counts"]["upstream_unresolved_not_called"]},
        "budget":deepcopy(audited["budget"]),"bootstrap_samples":samples,"bootstrap_seed":42,
        "provenance":{"retry_artifact_sha256":deepcopy(audited["artifact_sha256"]),
            "original_artifact_sha256":{name:sha for item in items for name,sha in item["artifact_sha256"].items()},
            "analysis_sha256":file_sha(__file__),"pure_helper_sha256":file_sha(analysis.__file__)},
        "limitations":["Original outcomes and reservations remain immutable; retries add calls and cost.",
            "An upstream skip is distinct from a paid failure. A dependent review begins only after its source has a valid recovered response.",
            "First attempt means the first actual service call; original_snapshot also includes historical upstream skips.",
            "Earliest valid success is selected without using truth labels. Original valid responses, including wrong ones, remain unchanged.",
            "Full-condition metrics include failures as incorrect. Unattempted dependent rows suppress full service-view metrics.",
            "Paired group bootstrap intervals are unadjusted and conditional on these cases and observed attempts; they measure recovery availability, not semantic model improvement.",
            "Affected conditions were selected for operational errors and are not a new independent benchmark. No overall accuracy winner or latency claim is reported."]}


def collect(*, samples=2000):
    audited = runner.audit_run()
    loaded = {}
    for item in audited["plan"]["inventory"]["failures"]:
        if item["run_id"] not in loaded:
            loaded[item["run_id"]] = load_original(item)
    result = assemble(audited, loaded, original_reservations(audited["plan"]), samples=samples)
    # Reaudit after loading/analysis so a concurrent mutation cannot be published.
    repeated = runner.audit_run()
    require(repeated["artifact_sha256"] == audited["artifact_sha256"], "Recovery changed during aggregation")
    return result


def optional_collect(*, samples=2000):
    """No scores while execution is missing/incomplete; malformed evidence fails."""
    record_path = runner.AREA / "run.json"
    if not record_path.exists():
        require(not any((runner.AREA/name).exists() for name in ("plan.json","attempts.jsonl","budget.jsonl","budget.jsonl.lock")),
                "Orphan recovery evidence cannot be treated as pending")
        return None
    record = json.loads(record_path.read_text())
    require(record.get("status") in {"running","paused","halted_billing","halted_errors","budget_exhausted","interrupted","complete"},
            "Unknown recovery status")
    return collect(samples=samples) if record["status"] == "complete" else None


def findings(report):
    def metric(view):
        value=view["metrics"]
        return "pending" if value is None else f'{value["accuracy"]:.2%} / {value["n_failures"]}'
    lines=["# Audited recovery: original outcomes remain visible","",
        f'Finite policy complete: {report["api_outcomes"]["new_calls"]} new service calls across {report["n_affected_conditions"]} affected original conditions.',
        "","Accuracy / failures below use the full original condition. First actual call and recovered views are separate from historical upstream skips.","",
        "| Dataset | Model / source | Examples/class | Original snapshot | First actual call | Recovered | New calls |",
        "|---|---|---:|---|---|---|---:|"]
    for c in report["conditions"]:
        model=c["model"]+(f' / proposal: {c["source_model"]}' if c["source_model"] else "")
        lines.append(f'| {c["dataset"]} | {model} | {c["shots_per_class"]} | {metric(c["original_snapshot"])} | {metric(c["first_attempt"])} | {metric(c["recovered"])} | {c["n_new_calls"]} |')
    lines+=["","## API outcomes","",json.dumps(report["api_outcomes"],sort_keys=True),"",
        "The paired intervals in RECOVERY_COMPARISON.json compare recovered minus first actual call. They do not attribute semantic improvements to retrying a valid response.",""]
    lines += [f'- {text}' for text in report["limitations"]]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples",type=int,default=2000)
    args=parser.parse_args(argv)
    report=collect(samples=args.bootstrap_samples)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    runner.atomic_json(FILE,report)
    (OUTPUT/"RECOVERY_FINDINGS.md").write_text(findings(report))
    print(json.dumps({"status":report["status"],"conditions":report["n_affected_conditions"],
                     "new_calls":report["api_outcomes"]["new_calls"]}))


if __name__=="__main__": main()
