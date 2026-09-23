#!/usr/bin/env python3
"""Sensitivity appendix: include constant-label sources in cached review ranks.

Reaudits the unchanged primary analysis and raw artifacts. No model/network calls,
new thresholds, primary-report changes, or dashboard changes. Writes only the new
results/review_value_sensitivity namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import analyze_review_value as primary


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligibility(base_row, predictions, values):
    original = primary.eligibility(base_row, predictions, values, review_complete=True)
    labels = set(v for v in values if v >= 0)
    constant = len(labels) == 1 and all(v >= 0 for v in values)
    reasons = [reason for reason in original["exclusion_reasons"]
               if not (reason == "constant_or_missing_source_labels" and constant)]
    return {"eligible": not reasons, "primary_eligible": original["eligible"],
            "constant_source_predictions": constant, "n_distinct_source_labels": len(labels),
            "exclusion_reasons": reasons,
            "primary_exclusion_reasons": original["exclusion_reasons"],
            "probability_semantics": original["probability_semantics"]}


def analyze_condition(condition, raw_base, raw_review, manifest, *, base_config):
    classes = len(manifest["labels"])
    ids = [row["id"] for row in manifest["rows"]]
    y = [row["label"] for row in manifest["rows"]]
    require([p.row_id for p in raw_base] == [p.row_id for p in raw_review] == ids,
            "Source/review predictions are not the same complete ordered fold")
    base, reviewed = primary.chosen(raw_base, classes), primary.chosen(raw_review, classes)
    gate = eligibility({"config": base_config}, raw_base, base)
    item = {key: condition[key] for key in ("dataset", "source_model", "shots_per_class", "n_rows",
        "source_run_id", "review_run_id", "direct_jev_run_id")}
    item.update(eligibility=gate, curve=None)
    if not gate["eligible"]:
        item["analysis_membership"] = "excluded_from_sensitivity"
        return item
    require(gate["primary_eligible"] == condition["selective_review_eligibility"]["eligible"],
            "Primary gate classification changed")
    curve = primary.selective_curve(y, base, reviewed, raw_base, ids, classes)
    require(curve["points"][0]["metrics"] == condition["metrics"]["never_review"] and
            curve["points"][-1]["metrics"] == condition["metrics"]["always_review"],
            "Sensitivity endpoints differ from the audited source/full review")
    require(all(p["required_inferences"]["review_api_requests"] == p["selected_rows"] for p in curve["points"]),
            "Eligible complete-probability source cannot silently skip requests")
    if gate["primary_eligible"]:
        require(curve == condition["selective_review"], "Primary rank, coverage, or random comparison changed")
    probabilities = list(curve["source_max_probabilities"].values())
    item.update(analysis_membership="primary_and_sensitivity" if gate["primary_eligible"] else "sensitivity_only_constant_labels",
        source_max_probability={"min": min(probabilities), "max": max(probabilities),
            "n_unique": len(set(probabilities)), "has_variation": len(set(probabilities)) > 1},
        metrics=condition["metrics"], curve=curve,
        failures=condition["failures"], observed_review_request_records=condition["observed_review_request_records"])
    return item


def protected_paths(root, report):
    paths = {root / "results/review_value/ANALYSIS.json", root / "results/review_value/FINDINGS.md",
             root / "results/numeric_expansion/COMPARISON.json", root / "results/numeric_decisions/COMPARISON.json",
             root / "results/TABULAR_COMPARISON.json", root / "dashboard/dist/review-data.json"}
    for artifact in report["artifacts"].values():
        paths.update((root / artifact["path"]).with_name(name) for name in artifact["sha256"])
    paths.update(root / path for path in report["audit"]["file_pins"])
    return paths


def random_comparison_pattern(condition):
    differences = [p["metrics"]["micro_accuracy"] - p["random_matched_rate"]["micro_accuracy"]
                   for p in condition["curve"]["points"] if 0 < p["requested_review_percent"] < 100]
    require(len(differences) == 4, "Intermediate coverage grid changed")
    signs = [0 if abs(value) <= 1e-12 else 1 if value > 0 else -1 for value in differences]
    return {1: "above_at_all_intermediate_rates", -1: "below_at_all_intermediate_rates", 0: "ties_at_all_intermediate_rates"}.get(
        signs[0], "mixed") if len(set(signs)) == 1 else "mixed"


def collect(root=ROOT):
    root = Path(root).resolve()
    saved = primary.source.read(root / "results/review_value/ANALYSIS.json")
    before = {path.relative_to(root).as_posix(): file_sha(path) for path in protected_paths(root, saved)}
    report = primary.collect(root)
    require(report == saved, "Primary report is stale; do not silently replace its evidence")
    conditions, excluded = [], []
    for condition in report["conditions"]:
        artifacts = [report["artifacts"][condition[key]] for key in ("source_run_id", "review_run_id")]
        rows = [{"source_path": artifact["path"], "artifact_sha256": artifact["sha256"]} for artifact in artifacts]
        (base, bm), (review, rm) = [primary.load_raw(root, row) for row in rows]
        require(bm == rm, "Source/review manifests differ")
        item = analyze_condition(condition, base, review, bm, base_config=artifacts[0]["config"])
        (conditions if item["eligibility"]["eligible"] else excluded).append(item)
    unchanged = {path: file_sha(root / path) for path in before}
    require(before == unchanged, "Existing evidence changed during the read-only analysis")
    primary_count = sum(c["eligibility"]["primary_eligible"] for c in conditions)
    added = [c for c in conditions if not c["eligibility"]["primary_eligible"]]
    require(all(c["eligibility"]["constant_source_predictions"] for c in added),
            "Sensitivity changed an exclusion other than constant source labels")
    patterns = {key: sum(random_comparison_pattern(c) == key for c in added) for key in
                ("above_at_all_intermediate_rates", "below_at_all_intermediate_rates", "ties_at_all_intermediate_rates", "mixed")}
    counters = {"completed_review_conditions_audited": report["complete_review_conditions"],
        "expected_review_conditions": report["expected_review_conditions"],
        "primary_eligible_conditions": primary_count, "sensitivity_eligible_conditions": len(conditions),
        "added_constant_label_conditions": len(added),
        "added_constant_conditions_with_varying_max_probability": sum(c["source_max_probability"]["has_variation"] for c in added),
        "excluded_complete_conditions": len(excluded), "unscored_incomplete_conditions": len(report["excluded_conditions"]),
        "scored_curve_points": len(conditions) * len(primary.RATES), "partial_condition_metrics": 0}
    return {"schema_version": 1, "scope": "post-hoc sensitivity appendix broadening primary eligibility to include constant-label local sources",
        "status": "complete_offline_appendix", "counters": counters, "coverage_rates_percent": list(primary.RATES),
        "ranking": "Identical to primary: ascending source max probability, SHA-256(row_id) tie-break, nearest-integer count (.5 upward)",
        "change_from_primary": "Remove only the constant-source-label exclusion; all complete local HF normalized-probability sources qualify regardless of output-label diversity",
        "primary_analysis_changed": False, "primary_dashboard_changed": False, "new_model_or_api_calls": 0,
        "added_condition_accuracy_vs_random_patterns": patterns,
        "conditions": conditions, "excluded_complete_conditions": excluded,
        "unscored_incomplete_conditions": report["excluded_conditions"],
        "provenance": {"primary_report_sha256": before["results/review_value/ANALYSIS.json"],
            "script_sha256": file_sha(__file__), "tests_sha256": file_sha(root / "tests/test_review_sensitivity.py"),
            "protected_files_sha256": before,
            "artifacts": {run_id: artifact for run_id, artifact in report["artifacts"].items()
                if any(run_id in (c["source_run_id"], c["review_run_id"], c["direct_jev_run_id"]) for c in conditions)}},
        "limitations": ["Both primary eligibility and this sensitivity analysis were chosen after earlier test outcomes were visible.",
            "A constant class prediction can have a varying score ranking; constant predictions alone do not imply rank information is absent.",
            "All fixed coverages and eligible conditions are reported, with no learned threshold or selection of a winning coverage.",
            "Curves simulate cached outcomes on one split, not independent serving reruns or deployment cost savings.",
            "Random accuracy/balanced accuracy are exact fixed-subset expectations; random macro-F1 uses the same fixed-seed 10000 draws as primary.",
            "Differences versus random are descriptive, without uncertainty testing or an inference of general source-model ranking quality."]}


def pct(value):
    return f"{100 * value:.1f}%"


def findings(report):
    c = report["counters"]
    patterns = report["added_condition_accuracy_vs_random_patterns"]
    lines = ["# Constant-label sensitivity appendix", "",
        f"**All {c['sensitivity_eligible_conditions']} complete local-model conditions with eligible probability scores are included:** {c['primary_eligible_conditions']} conditions shared with the primary analysis and {c['added_constant_label_conditions']} constant-label conditions added only here. The {c['unscored_incomplete_conditions']} incomplete review conditions receive no metrics, and hosted sources remain outside this probability-ranking analysis.", "",
        "[Machine report, complete curves and source pins](ANALYSIS.json) · [Unchanged primary findings](../review_value/FINDINGS.md)", "",
        "The primary analysis excluded sources that predicted one class throughout their test fold. That was a post-hoc scope restriction, not proof that their confidence ranks were unusable. This appendix removes only that restriction and reuses precisely the same source maximum-probability ranking, row-hash tie-break, six coverage rates, rounding, cached review outcomes and random reference. No threshold, model or coverage is selected using these test outcomes.", "",
        f"All {c['added_constant_conditions_with_varying_max_probability']}/{c['added_constant_label_conditions']} newly included constant-label conditions have varying maximum-probability scores. Output-label collapse and score variation are different properties. Every unchanged primary curve matches the earlier machine report exactly; source/full-review endpoints and request counts are checked against the same audited records.", "",
        "## All included conditions", "",
        "The 50% column is the same predetermined midpoint for every condition. It is descriptive, not a recommendation. Primary membership is explicit; all other coverages follow in the next table.", "",
        "| Dataset | Source | Examples/class | Membership | Constant label? | Unique max scores | No review accuracy | 50% review accuracy | Random 50% expectation | Full review accuracy |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for condition in report["conditions"]:
        points = condition["curve"]["points"]
        half = next(p for p in points if p["requested_review_percent"] == 50)
        gate = condition["eligibility"]
        label = "primary + sensitivity" if gate["primary_eligible"] else "sensitivity only"
        lines.append(f"| {primary.source.NAMES[condition['dataset']]} | {primary.source.MODEL_NAMES[condition['source_model']]} | {condition['shots_per_class']} | {label} | {'yes' if gate['constant_source_predictions'] else 'no'} | {condition['source_max_probability']['n_unique']} | {pct(points[0]['metrics']['micro_accuracy'])} | {pct(half['metrics']['micro_accuracy'])} | {pct(half['random_matched_rate']['micro_accuracy'])} | {pct(points[-1]['metrics']['micro_accuracy'])} |")
    lines += ["", "## Every fixed coverage", "",
        "Accuracy is within-condition micro accuracy. Random accuracy and balanced accuracy are exact expectations for a uniformly selected subset of the same size. Random macro-F1, its Monte Carlo standard error, exact selection IDs, full direct-Jev references, and all failure details remain in the JSON.", "",
        "| Dataset/source/shots | Membership | Reviewed rows | Accuracy | Balanced accuracy | Macro-F1 | Random accuracy | Fixed | Harmed | Source inferences + review requests |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in report["conditions"]:
        title = f"{primary.source.NAMES[condition['dataset']]} / {primary.source.MODEL_NAMES[condition['source_model']]} / {condition['shots_per_class']}"
        membership = "primary + sensitivity" if condition["eligibility"]["primary_eligible"] else "sensitivity only"
        for point in condition["curve"]["points"]:
            m, t = point["metrics"], point["transitions_from_never_review"]
            lines.append(f"| {title} | {membership} | {point['selected_rows']}/{condition['n_rows']} | {pct(m['micro_accuracy'])} | {pct(m['balanced_accuracy'])} | {m['macro_f1']:.3f} | {pct(point['random_matched_rate']['micro_accuracy'])} | {t['wrong_to_correct']} | {t['correct_to_wrong']} | {condition['n_rows']} + {point['required_inferences']['review_api_requests']} |")
    lines += ["", "## What survives the broader scope", "",
        f"Across the {c['added_constant_label_conditions']} added constant-label conditions, {patterns['above_at_all_intermediate_rates']} are above the random-selection accuracy expectation at all four intermediate coverages, {patterns['below_at_all_intermediate_rates']} are below it throughout, {patterns['mixed']} have mixed differences, and {patterns['ties_at_all_intermediate_rates']} tie throughout. These are descriptive signs of differences, not statistical tests; every condition and all six coverages are displayed above.", "",
        f"The two Qwen3 four-shot observations remain the same because their inputs, ranking and cached outcomes have not changed: 57/114 Breast Cancer reviews and 18/36 Wine reviews match their respective full-review accuracies. The broader appendix does not turn that arithmetic observation into general evidence for a confidence gate. Constant-label conditions can still have variable ranks, and their results must be considered alongside the {c['primary_eligible_conditions']} currently eligible primary conditions.", "",
        "The primary eligibility rule was chosen after earlier outcomes existed. This appendix is also post-hoc and does not repair that limitation by adding more curves. Model conditions share cases, and source sequence probabilities include class ID plus EOS; they are uncalibrated. Report all conditions rather than selecting successful routes or a best test-set coverage. This analysis command makes no new inference or paid request, changes no primary report/dashboard, and establishes no dollar or latency saving.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = collect(args.root)
    output = args.root / "results/review_value_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    (output / "ANALYSIS.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "FINDINGS.md").write_text(findings(report))
    print(json.dumps(report["counters"]))


if __name__ == "__main__":
    main()
