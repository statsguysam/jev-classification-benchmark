#!/usr/bin/env python3
"""Offline, complete-condition analysis of cached numeric review value.

Reuses the existing read-only provenance/metric auditors; never calls a provider.
Writes only results/review_value. Coverage rules are descriptive batch simulations,
not learned thresholds or validated deployment policies.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

import summarize_expanded_numeric as source
from summarize_numeric_decisions import chosen, transitions
from jevbench.types import Prediction

ROOT = Path(__file__).resolve().parents[1]
RATES = (0, 10, 25, 50, 75, 100)
RANDOM_SEED = 20260923
RANDOM_SAMPLES = 10000
FILES = ("run.json", "predictions.jsonl", "test_manifest.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def scores(y, pred, classes):
    """All cases remain in denominators; -1 is an incorrect/failed outcome."""
    y, pred = np.asarray(y), np.asarray(pred)
    require(y.shape == pred.shape and y.ndim == 1 and len(y), "Unaligned predictions")
    recalls, f1 = [], []
    for c in range(classes):
        tp = np.sum((y == c) & (pred == c))
        support, positives = np.sum(y == c), np.sum(pred == c)
        if support:
            recalls.append(tp / support)
        f1.append(2 * tp / (support + positives) if support + positives else 0.)
    return {"micro_accuracy": float(np.mean(y == pred)),
            "balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1)),
            "n_rows": len(y), "n_failures": int(np.sum(pred == -1))}


def joint_correctness(y, base, direct, review):
    require(len(y) == len(base) == len(direct) == len(review), "Unaligned joint outcomes")
    counts = {"".join(map(str, bits)): 0 for bits in itertools.product((0, 1), repeat=3)}
    for truth, a, b, c in zip(y, base, direct, review):
        counts[f"{int(a == truth)}{int(b == truth)}{int(c == truth)}"] += 1
    return {"bit_order": ["base_correct", "direct_jev_correct", "review_correct"],
            "counts": counts, "failures_count_as_incorrect": True}


def eligibility(base_row, predictions, values, review_complete=True):
    reasons = []
    if not review_complete:
        reasons.append("review_condition_incomplete")
    if base_row["config"].get("provider") != "hf":
        reasons.append("source_is_not_a_local_hf_model")
    if len(set(v for v in values if v >= 0)) < 2:
        reasons.append("constant_or_missing_source_labels")
    if any(v < 0 for v in values):
        reasons.append("source_failures_prevent_complete_probability_ranking")
    if any(p.probabilities is None or p.error is not None or
           p.metadata.get("probability_kind") != "label_sequence_likelihood_normalized"
           for p in predictions):
        reasons.append("source_probability_scores_unavailable_or_ineligible")
    return {"eligible": not reasons, "exclusion_reasons": reasons,
            "probability_semantics": "normalized numeric-class-ID-plus-EOS sequence likelihood; not calibrated confidence"}


def rank_rows(predictions, row_ids):
    """Low source maximum probability first; does not receive test truth."""
    require(len(predictions) == len(row_ids) and len(set(row_ids)) == len(row_ids), "Row IDs differ")
    confidences = []
    for p in predictions:
        require(p.probabilities is not None and p.error is None, "Ranking requires complete source probabilities")
        values = np.asarray(p.probabilities, dtype=float)
        require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all() and
                (values >= 0).all() and (values <= 1).all() and abs(values.sum() - 1) <= 1e-6,
                "Invalid source probabilities")
        confidences.append(float(values.max()))
    hashes = [hashlib.sha256(row_id.encode()).hexdigest() for row_id in row_ids]
    return sorted(range(len(row_ids)), key=lambda i: (confidences[i], hashes[i])), confidences


def review_count(n, rate):
    """Nearest integer, .5 upward; actual coverage is always reported."""
    return (n * rate + 50) // 100


def random_expectation(y, base, review, classes, m, *, samples=RANDOM_SAMPLES):
    """Uniform fixed-size subsets: exact linear metrics, Monte Carlo macro-F1."""
    n = len(y)
    require(0 <= m <= n and samples >= 2, "Invalid random comparison size")
    a, b = scores(y, base, classes), scores(y, review, classes)
    fraction = m / n
    result = {name: a[name] + fraction * (b[name] - a[name])
              for name in ("micro_accuracy", "balanced_accuracy")}
    changes = transitions(y, base, review)
    result.update(expected_fixed=fraction * changes["wrong_to_correct"],
                  expected_harmed=fraction * changes["correct_to_wrong"],
                  expected_net_fixed=fraction * changes["net_correct_change"],
                  expected_failures=a["n_failures"] + fraction * (b["n_failures"] - a["n_failures"]),
                  linear_metrics_method="exact uniform fixed-size subset expectation")
    if m in (0, n):
        result.update(macro_f1=a["macro_f1"] if m == 0 else b["macro_f1"],
                      macro_f1_method="exact endpoint", macro_f1_monte_carlo_se=0., samples=0)
        return result
    rng = np.random.default_rng(RANDOM_SEED)
    selected = np.zeros((samples, n), dtype=bool)
    indices = np.argsort(rng.random((samples, n)), axis=1)[:, :m]
    selected[np.arange(samples)[:, None], indices] = True
    predictions = np.where(selected, np.asarray(review), np.asarray(base))
    f1 = np.zeros(samples)
    truth = np.asarray(y)
    for c in range(classes):
        tp = ((predictions == c) & (truth == c)).sum(axis=1)
        denominator = (truth == c).sum() + (predictions == c).sum(axis=1)
        f1 += np.divide(2 * tp, denominator, out=np.zeros(samples), where=denominator != 0) / classes
    result.update(macro_f1=float(f1.mean()), macro_f1_method="Monte Carlo mean over uniform fixed-size subsets",
                  macro_f1_monte_carlo_se=float(f1.std(ddof=1) / np.sqrt(samples)),
                  samples=samples, random_seed=RANDOM_SEED)
    return result


def selective_curve(y, base, review, predictions, row_ids, classes):
    order, confidences = rank_rows(predictions, row_ids)
    n, result = len(y), []
    for rate in RATES:
        m = review_count(n, rate)
        selected = set(order[:m])
        mixed = [review[i] if i in selected else base[i] for i in range(n)]
        changes = transitions(y, base, mixed)
        result.append({"requested_review_percent": rate, "selected_rows": m, "actual_review_fraction": m / n,
            "selected_row_ids": [row_ids[i] for i in order[:m]], "metrics": scores(y, mixed, classes),
            "transitions_from_never_review": changes,
            "required_inferences": {"source": n, "review_api_requests": m, "direct_jev": 0},
            "random_matched_rate": random_expectation(y, base, review, classes, m)})
    return {"ranking": "ascending source max probability; ascending SHA-256(row_id) breaks ties",
            "rounding": "nearest integer, .5 upward", "ranked_row_ids": [row_ids[i] for i in order],
            "source_max_probabilities": dict(zip(row_ids, confidences)), "points": result}


def same_label_pairs(conditions):
    """Identical input/proposal produces identical prompt, regardless of source."""
    result = []
    for a, b in itertools.combinations(conditions, 2):
        if (a["dataset"], a["shots_per_class"]) != (b["dataset"], b["shots_per_class"]):
            continue
        require(a["row_ids"] == b["row_ids"] and a["y"] == b["y"], "Equivalent prompt rows are misaligned")
        eligible = [i for i, (x, z) in enumerate(zip(a["base"], b["base"])) if x == z and x >= 0]
        for i in eligible:
            require(a["prompt_hashes"][i] and a["prompt_hashes"][i] == b["prompt_hashes"][i],
                    "Same-row/same-example/same-label review prompts differ")
        valid = [i for i in eligible if a["review"][i] >= 0 and b["review"][i] >= 0]
        ar = a.get("resolved_models", [None] * len(a["row_ids"]))
        br = b.get("resolved_models", [None] * len(b["row_ids"]))
        result.append({"dataset": a["dataset"], "shots_per_class": a["shots_per_class"],
            "a_run_id": a["run_id"], "b_run_id": b["run_id"],
            "same_label_identical_prompt_rows": len(eligible), "both_valid_rows": len(valid),
            "valid_output_disagreements": sum(a["review"][i] != b["review"][i] for i in valid),
            "both_valid_resolved_model_mismatches": sum(ar[i] != br[i] for i in valid if ar[i] and br[i]),
            "both_valid_unknown_resolved_model_rows": sum(not ar[i] or not br[i] for i in valid),
            "rows_with_either_review_failure": len(eligible) - len(valid),
            "disagreement_row_ids": [a["row_ids"][i] for i in valid if a["review"][i] != b["review"][i]]})
    return result


def load_raw(root, row):
    path = source.verify_historical_artifacts(row, root)
    raw = [Prediction(**json.loads(line)) for line in path.with_name("predictions.jsonl").read_text().splitlines() if line]
    manifest = source.read(path.with_name("test_manifest.json"))
    require([p.row_id for p in raw] == [r["id"] for r in manifest["rows"]], "Raw prediction order differs")
    return raw, manifest


def collect(root=ROOT):
    root = Path(root).resolve()
    # Existing collectors audit family-specific provenance, complete coverage,
    # saved metrics, source/report pins, review prompt hashes, and reservations.
    # Their 100-replicate audit outputs are not used for new inference or claims.
    audited = source.collect(root, samples=100)
    by_key = {source.row_key(row): row for row in audited["runs"]}
    complete, excluded, gate_inventory, artifacts, internal = [], [], [], {}, []
    for dataset in source.DATASETS:
        for model in source.MODEL_NAMES:
            for shots in (0, 4):
                b = by_key[dataset, model, "base", shots]
                r = by_key[dataset, model, "review", shots]
                d = by_key[dataset, source.JEV, "direct", shots]
                require(b["status"] == d["status"] == "complete", "Missing audited base or direct Jev")
                raw_base, manifest = load_raw(root, b)
                classes = len(manifest["labels"])
                base_values = chosen(raw_base, classes)
                gate = eligibility(b, raw_base, base_values, r["status"] == "complete")
                ident = {"dataset": dataset, "source_model": model, "shots_per_class": shots,
                         "source_run_id": b["run_id"], "review_run_id": r["run_id"]}
                gate_inventory.append({**ident, **gate})
                if r["status"] != "complete":
                    excluded.append({**ident, "status": r["status"], "source_path": r["source_path"],
                                     "available_artifact_pins": ({name: source.file_sha((root / r["source_path"]).with_name(name))
                                         for name in FILES if (root / r["source_path"]).with_name(name).is_file()}
                                         if r["source_path"] else {}),
                                     "metrics": None, "reason": "Incomplete conditions receive no partial metrics"})
                    continue
                raw_review, review_manifest = load_raw(root, r)
                raw_direct, direct_manifest = load_raw(root, d)
                require(manifest == review_manifest == direct_manifest, "Triplet test manifests differ")
                require(b["train_labels"] == r["train_labels"] == d["train_labels"], "Triplet label budgets differ")
                require(r["source_run_id"] == b["run_id"], "Review points to another source")
                training_ids = [source.read(root / row["source_path"])["training_example_ids"] for row in (b, r, d)]
                require(training_ids[0] == training_ids[1] == training_ids[2], "Triplet examples differ")
                for row in (b, r, d):
                    record = source.read(root / row["source_path"])
                    artifacts[row["run_id"]] = {"path": row["source_path"], "sha256": row["artifact_sha256"],
                        "config": row["config"], "resolved_models": row.get("resolved_models", []),
                        "resolved_revisions": row.get("resolved_revisions", []),
                        "started_at": record.get("started_at"), "completed_at": record.get("completed_at"),
                        "execution_sessions": record.get("execution_sessions", [])}
                y = [row["label"] for row in manifest["rows"]]
                ids = [row["id"] for row in manifest["rows"]]
                rv, dv = chosen(raw_review, classes), chosen(raw_direct, classes)
                failures = {stage: [{"row_id": p.row_id, "error": p.error,
                                    "reported_cost_usd": p.metadata.get("openrouter", {}).get("reported_cost_usd")}
                                   for p, v in zip(predictions, values) if v == -1]
                            for stage, predictions, values in (("base", raw_base, base_values),
                                ("review", raw_review, rv), ("direct_jev", raw_direct, dv))}
                called = sum(value >= 0 for value in base_values)
                reservations = [p.metadata.get("budget", {}).get("reservation_id") for p in raw_review]
                observed_requests = sum(value is not None for value in reservations)
                require(observed_requests == called and len({v for v in reservations if v is not None}) == called,
                        "Review request records do not match valid proposals")
                item = {**ident, "direct_jev_run_id": d["run_id"], "n_rows": len(y), "labels": manifest["labels"],
                    "metrics": {"never_review": scores(y, base_values, classes), "always_review": scores(y, rv, classes),
                                "direct_jev": scores(y, dv, classes)},
                    "joint_correctness": joint_correctness(y, base_values, dv, rv),
                    "review_minus_base": transitions(y, base_values, rv),
                    "review_minus_direct": transitions(y, dv, rv), "failures": failures,
                    "observed_review_request_records": observed_requests,
                    "observed_skipped_source_failure_rows": len(y) - observed_requests,
                    "required_inferences": {"never_review": {"source": len(y), "review_api_requests": 0, "direct_jev": 0},
                        "always_review": {"source": len(y), "review_api_requests": called, "direct_jev": 0},
                        "direct_jev": {"source": 0, "review_api_requests": 0, "direct_jev": len(y)}},
                    "selective_review_eligibility": gate, "selective_review": None}
                if gate["eligible"]:
                    item["selective_review"] = selective_curve(y, base_values, rv, raw_base, ids, classes)
                complete.append(item)
                internal.append({"dataset": dataset, "shots_per_class": shots, "run_id": r["run_id"],
                    "row_ids": ids, "y": y, "base": base_values, "review": rv,
                    "prompt_hashes": [p.metadata.get("review_prompt_sha256") for p in raw_review],
                    "resolved_models": [p.metadata.get("resolved_model") for p in raw_review]})
    require(len(complete) == audited["complete_review_runs"], "Completed review inventory differs")
    pin_paths = ["scripts/analyze_review_value.py", "scripts/summarize_expanded_numeric.py",
        "scripts/summarize_numeric_decisions.py", "scripts/summarize_tabular.py",
        "scripts/run_numeric_jev_review.py", "scripts/run_expanded_numeric_review.py",
        "scripts/run_expanded_numeric_local.py", "tests/test_review_value.py", "docs/EXPANDED_NUMERIC_PROTOCOL.md",
        "results/TABULAR_COMPARISON.json", "results/numeric_decisions/COMPARISON.json",
        "results/numeric_expansion/COMPARISON.json"]
    return {"schema_version": 1, "scope": "offline post-hoc analysis of complete numeric review conditions",
        "expected_review_conditions": 24, "complete_review_conditions": len(complete),
        "scored_partial_conditions": 0, "unique_test_rows": {name: next(x["n_rows"] for x in complete if x["dataset"] == name) for name in source.DATASETS},
        "coverage_rates_percent": list(RATES), "selection_rule_uses_test_truth": False,
        "interpretation": "Fixed-coverage batch simulation on cached outcomes; not a learned threshold or validated deployable gate. No paid calls, confidence calibration, latency or dollar savings claim.",
        "audit": {"method": "unchanged summarize_expanded_numeric.collect(samples=100), then pinned raw artifact reread",
                  "bootstrap_outputs_used_for_findings": False, "prepared_manifests": audited["prepared_manifests"],
                  "file_pins": {p: source.file_sha(root / p) for p in pin_paths}},
        "artifacts": artifacts, "conditions": complete, "excluded_conditions": excluded,
        "gate_inventory": gate_inventory, "same_label_prompt_pairs": same_label_pairs(internal)}


def pct(x):
    return f"{100*x:.1f}%"


def findings(report):
    eligible_count = sum(c["selective_review"] is not None for c in report["conditions"])
    lines = ["# Added value of Jev review: offline numeric evidence", "",
        f"**{report['complete_review_conditions']}/24 complete numeric review conditions are scored.** The other conditions remain unscored; no partial-run accuracy is used. Breast Cancer has 114 test cases, Wine 36, each on one frozen split. No text review conclusions are available here.", "",
        "This analysis reconstructs raw predictions through the unchanged provenance/metric auditors. Source, direct Jev and review share the same held-out rows and matching zero/four examples per class. All failures count as incorrect. The machine report preserves each failure and its unknown reported charge; it does not convert unknown charges to zero.", "",
        "[Machine-readable report and exact artifact pins](ANALYSIS.json)", "",
        "## Does review add value over both available alternatives?", "",
        "Accuracy below is micro accuracy within each condition. Balanced accuracy and macro-F1 are in the JSON for all three arms. Fixed/harmed counts compare review with its own source, including failures; net versus direct compares with separately called Jev alone.", "",
        "| Dataset | Source | Examples/class | Source accuracy | Review accuracy | Direct Jev accuracy | Fixed | Harmed | Net vs source | Net vs direct |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in report["conditions"]:
        m, t = c["metrics"], c["review_minus_base"]
        lines.append(f"| {source.NAMES[c['dataset']]} | {source.MODEL_NAMES[c['source_model']]} | {c['shots_per_class']} | {pct(m['never_review']['micro_accuracy'])} | {pct(m['always_review']['micro_accuracy'])} | {pct(m['direct_jev']['micro_accuracy'])} | {t['wrong_to_correct']} | {t['correct_to_wrong']} | {t['net_correct_change']:+d} | {c['review_minus_direct']['net_correct_change']:+d} |")
    pairs = report["same_label_prompt_pairs"]
    same = sum(p["same_label_identical_prompt_rows"] for p in pairs)
    valid = sum(p["both_valid_rows"] for p in pairs)
    disagreement = sum(p["valid_output_disagreements"] for p in pairs)
    failed = sum(p["rows_with_either_review_failure"] for p in pairs)
    alias_mismatches = sum(p["both_valid_resolved_model_mismatches"] for p in pairs)
    unknown_aliases = sum(p["both_valid_unknown_resolved_model_rows"] for p in pairs)
    lines += ["", "The JSON supplies all eight joint correctness combinations for source/direct/review, so a correction can be distinguished from a gain already achieved by direct Jev. These conditions reuse the same cases and direct references; no pooled average or count is presented as independent evidence.", "",
        "## Same proposed label means the same reviewer prompt", "",
        f"Across {len(pairs)} within-dataset/shot source pairs, {same} row-pair comparisons had the same valid proposed label. All had exactly matching audited review-prompt hashes. Of {valid} comparisons with two valid review responses, {disagreement} produced different final labels; {failed} comparisons had at least one review failure. These are overlapping pairwise comparisons, not independent trials.", "",
        f"The both-valid comparisons contain {alias_mismatches} resolved-model-ID mismatches and {unknown_aliases} unknown resolved-model comparisons. Per-run resolved IDs, start/completion times and execution sessions are preserved in the artifact inventory. Matching returned IDs do not establish an unchanged serving backend.", "",
        "Source identity, confidence and rationale are absent from the reviewer prompt. Therefore, when two sources propose the same class for the same row and examples, the reviewer gets the same prompt. Cross-source differences on those matched prompts cannot be attributed to source identity being visible. The observed variability includes historical calls at different serving times and errors; these dependent comparisons were not designed as replicates and are not a repeatability estimate or causal anchoring test.", "",
        "## Fixed-coverage selective review", "",
        f"Eligible conditions require a complete review, a local HF source, complete normalized class-sequence probabilities and at least two distinct source labels. Constant-label sources and hosted sources without probability scores are excluded. This leaves {eligible_count} conditions in the current snapshot. Eligibility depends on source outputs and completion, not test correctness; it defines the scope of this descriptive analysis.", "",
        "Rank rows by ascending source maximum probability, with SHA-256(row ID) breaking ties. Select the nearest integer number of rows at predetermined 0/10/25/50/75/100% coverage (.5 rounds upward). No labels choose the rank, coverage, or threshold. This is a fixed-coverage batch simulation, not a validated deployable gate. Sequence likelihoods include numeric label plus EOS and are not calibrated probabilities of correctness.", "",
        "Random matched-rate accuracy and balanced accuracy are exact expectations over uniform fixed-size subsets. Random macro-F1 is a 10,000-draw estimate with fixed seed and reported Monte Carlo standard error, since macro-F1 is nonlinear. These are reference expectations, not confidence intervals.", "",
        "| Dataset/source/shots | Reviewed rows | Accuracy | Balanced accuracy | Macro-F1 | Random expected accuracy | Fixed | Harmed | Source inferences + review requests |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in report["conditions"]:
        if c["selective_review"] is None:
            continue
        for p in c["selective_review"]["points"]:
            m, t = p["metrics"], p["transitions_from_never_review"]
            title = f"{source.NAMES[c['dataset']]} / {source.MODEL_NAMES[c['source_model']]} / {c['shots_per_class']}"
            lines.append(f"| {title} | {p['selected_rows']}/{c['n_rows']} | {pct(m['micro_accuracy'])} | {pct(m['balanced_accuracy'])} | {m['macro_f1']:.3f} | {pct(p['random_matched_rate']['micro_accuracy'])} | {t['wrong_to_correct']} | {t['correct_to_wrong']} | {c['n_rows']} + {p['selected_rows']} |")
    lines += ["", "Never-review needs one source inference per row. Always-review additionally requests review for every valid source proposal; a failed proposal remains a pipeline failure with no fallback call. Direct Jev needs only its own request per row. Observed review reservation records are checked against valid proposals, so failed requests count and skipped source failures do not become requests. The JSON includes these counts and selection IDs. Curve counts are simulated logical inference/request requirements under the frozen one-attempt behavior, not new paid calls, measured wall-clock time, training compute, or dollar savings; source costs remain incomplete.", "",
        "## Limits and next evidence needed", "",
        f"These are exploratory analyses chosen after earlier test results were seen. The coverage grid is fixed before this analysis is calculated, but was not preregistered before the original outcomes existed. Do not select a winning coverage on these test cases and call it validated. Unknown serving outcomes remain recorded failures. {len(report['excluded_conditions'])} incomplete conditions and all text reviews remain outside scored comparisons.", "",
        "Review versus existing direct Jev changes prompt wording as well as adding a proposal, and uses separate serving occasions. A same-prompt no-proposal control and randomized-proposal control are still needed to identify proposal value or anchoring causally. The familiar public datasets, one split, tiny Wine sample, restricted-label source scoring, repeated cases and mixed runtimes limit generalization. No architecture-wide or universal improvement claim follows.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = collect(args.root)
    output = args.root / "results/review_value"
    output.mkdir(parents=True, exist_ok=True)
    (output / "ANALYSIS.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "FINDINGS.md").write_text(findings(report))
    print(json.dumps({"complete_review_conditions": report["complete_review_conditions"],
        "eligible_selective_review_conditions": sum(c["selective_review"] is not None for c in report["conditions"]),
        "output": str(output)}))


if __name__ == "__main__":
    main()
