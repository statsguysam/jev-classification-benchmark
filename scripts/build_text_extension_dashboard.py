"""Reaudit a text report and export aggregate-only dashboard data.

No inference, site editing, report regeneration, or hosted calls. Incomplete or
stale evidence fails before the existing dashboard asset can be replaced.
Partial publication requires explicit --allow-incomplete; unscored arms stay null.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import summarize_text_extension as scientific
from jevbench.runner import save_json

DATASETS = {
    "sst2": {"id": "sst2", "label": "SST-2 sentiment", "data_type": "text", "task": "binary",
        "n_classes": 2, "n_features": None, "feature_description": "Text · TF-IDF for classical models",
        "n_test": 200, "full_training_labels": 10000,
        "classes": [{"id":0,"label":"negative"},{"id":1,"label":"positive"}]},
    "trec": {"id": "trec", "label": "TREC question categories", "data_type": "text", "task": "multiclass",
        "n_classes": 6, "n_features": None, "feature_description": "Text · TF-IDF for classical models",
        "n_test": 200, "full_training_labels": 4886,
        "classes": [{"id":i,"label":label} for i,label in enumerate([
            "abbreviation","entity","description or abstract concept","human being","location","numeric value"])]},
}
MODEL_KEYS = {"qwen_small": "Qwen/Qwen2.5-0.5B-Instruct", "qwen_main": "Qwen/Qwen3-4B-Instruct-2507",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "granite": "ibm-granite/granite-3.3-2b-instruct",
    "luna": "gpt-5.6-luna", "astra": "gpt-6-astra"}
TRANSITION_COUNTS = ("wrong_to_correct", "correct_to_wrong", "correct_to_wrong_label", "correct_to_failure",
    "both_correct", "both_wrong", "changed_predictions", "source_failure_rows", "review_stage_failure_rows")
LIMITATIONS = [
    "Exploratory text extension chosen after prior pilot results were viewed. Two familiar public datasets, one split and example seed; pretraining exposure cannot be excluded.",
    "Each dataset has 200 held-out rows. Four examples per class supplies 8 SST-2 labels or 24 TREC labels. Equal new labels do not imply equal total pretraining exposure.",
    "Full prepared classical training uses 10,000 SST-2 or 4,886 TREC labels and excludes validation labels. It is a separately labeled unequal-data reference.",
    "All classical models use identical training-only sparse word/character TF-IDF matrices and fixed hyperparameters. This is not an optimized text leaderboard; native XGBoost interprets unstored sparse entries as missing.",
    "Jev reviews the original text, the same examples and a cached class proposal from a separate model. This does not isolate bounded decoding on the same LLM; bounded validity does not guarantee correctness.",
    "Compare the pipeline with both the source LLM and Jev alone. A gain over a weak source does not establish that the first stage adds value.",
    "All failures count as incorrect. Failed source proposals trigger no review call; failed reviews never silently fall back to the proposed label.",
    "Original Qwen/OpenAI rendering differs from the fixed-system SmolLM2/Granite wrapper; model differences are not a controlled same-prompt architecture ablation.",
    "Paired normalized-text group-bootstrap intervals condition on this split, examples and fitted models. Multiple comparisons are unadjusted; crossing zero does not establish equivalence.",
    "Restricted-label likelihoods, native Choice and classical probability scores have different meanings and are not guaranteed calibrated.",
    "Cached proposals save new source calls in this experiment; a deployed pipeline incurs both source and reviewer expense. Local compute is unpriced and mixed hardware prevents a controlled latency ranking.",
    "This extension adds no LoRA training. Public benchmark performance does not establish deployment readiness.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, low=None, high=None):
    require(type(value) in (int, float) and math.isfinite(value), "Nonfinite or nonnumeric aggregate")
    require((low is None or value >= low) and (high is None or value <= high), "Aggregate outside allowed range")
    return value


def integer(value, low=0, high=None):
    require(type(value) is int, "Aggregate count must be an integer")
    return number(value, low, high)


def usd(value):
    require(type(value) in (str, int, float), "Invalid USD aggregate")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("Invalid USD aggregate") from None
    require(parsed.is_finite() and parsed >= 0, "Invalid USD aggregate")
    return str(parsed)


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,180}", value), "Unsafe aggregate identifier")
    return value


def family(model):
    return ("classical" if model in scientific.CLASSICAL_NAMES else "jev" if model == scientific.JEV else
            "hosted_llm" if model.startswith("gpt-") else "open_weight_llm")


def intervals(bootstrap, *, paired, n_test, samples):
    method = "paired unstratified group percentile bootstrap" if paired else "unstratified group percentile bootstrap"
    require(bootstrap["method"] == method and bootstrap["samples"] == samples and bootstrap["seed"] == 42
            and bootstrap["n_rows"] == bootstrap["n_groups"] == n_test, "Unexpected grouped bootstrap provenance")
    result = {}
    for metric in ("accuracy", "macro_f1"):
        source = bootstrap["metrics"][metric]
        estimate = number(source["estimate"], -1 if paired else 0, 1)
        require(isinstance(source["ci95"], list) and len(source["ci95"]) == 2, "Invalid interval endpoints")
        lo, hi = [number(value, -1 if paired else 0, 1) for value in source["ci95"]]
        require(lo <= hi, "Reversed confidence interval")
        result[metric] = estimate
        result[metric + "_ci95"] = [lo, hi]
    return result


def transitions(source, n_test):
    result = {name: integer(source[name], 0, n_test) for name in TRANSITION_COUNTS}
    result["net_correct_change"] = integer(source["net_correct_change"], -n_test, n_test)
    result["accuracy_delta_pp"] = number(source["accuracy_delta_pp"], -100, 100)
    require(sum(result[key] for key in ("wrong_to_correct", "correct_to_wrong", "both_correct", "both_wrong")) == n_test,
            "Transition counts do not cover the full test set")
    require(result["correct_to_wrong"] == result["correct_to_wrong_label"] + result["correct_to_failure"]
            and result["net_correct_change"] == result["wrong_to_correct"] - result["correct_to_wrong"]
            and math.isclose(result["accuracy_delta_pp"], 100*result["net_correct_change"]/n_test, abs_tol=1e-10),
            "Transition fix/harm decomposition differs")
    return result


def completion(report, *, allow_incomplete=False):
    require(allow_incomplete or (report["status"] == "complete" and report["complete_runs"] == report["expected_runs"] == 68
            and report["complete_comparisons"] == report["expected_comparisons"] == 72),
            "Refusing dashboard export: all 68 conditions and 72 comparisons must be complete")
    require(report["expected_runs"] == len(report["runs"]) == 68 and report["expected_comparisons"] == len(report["comparisons"]) == 72
            and report["complete_runs"] == sum(row["status"] == "complete" for row in report["runs"])
            and report["complete_comparisons"] == sum(row["status"] == "complete" for row in report["comparisons"]),
            "Incomplete or duplicated aggregate inventory")
    require(report["status"] == ("complete" if report["complete_runs"] == 68 and report["complete_comparisons"] == 72
                                 else "in_progress_or_incomplete"), "Summary completion status differs")
    require(report["seed"] == 42 and type(report["bootstrap_samples"]) is int and report["bootstrap_samples"] >= 100,
            "Unexpected summary resampling protocol")


def cost_summary(source, *, allow_incomplete=False):
    require(source["status"] in ({"complete", "halted", "in_progress", "blocked_no_credit", "not_executed"} if allow_incomplete else {"complete"})
            and source["expected_new_review_conditions"] == 24
            and 0 <= source["complete_summary_conditions"] <= source["complete_checkpoint_conditions"] <= 24
            and (source["status"] != "complete" or source["complete_summary_conditions"] == source["complete_checkpoint_conditions"] == 24),
            "Cost reconciliation is incomplete")
    require((allow_incomplete and source["status"] != "complete") or all(source[name] == 0 for name in ("reservations_without_result", "results_without_saved_prediction", "partial_checkpoint_files")),
            "Unreconciled cost evidence")
    result = {"status": source["status"], "new_review_calls": integer(source["new_model_requests"]),
        "complete_review_conditions": integer(source["complete_summary_conditions"], 0, 24), "expected_review_conditions": 24,
        **{name: integer(source[name]) for name in ("reservations_without_result", "results_without_saved_prediction", "partial_checkpoint_files")},
        "known_reported_api_usd": usd(source["known_reported_api_usd"]),
        "unknown_cost_calls": integer(source["unknown_cost_requests"]),
        "new_review_reserved_usd": usd(source["review_conservative_usd"]),
        "prior_reserved_usd": usd(source["prior_conservative_usd"]),
        "cumulative_reserved_usd": usd(source["cumulative_conservative_usd"]),
        "authorized_usd": usd(source["cumulative_authorized_usd"]),
        "stage_authorized_usd": usd(source["stage_authorized_usd"]),
        "numeric_reserved_allowance_usd": usd(source["numeric_reserved_allowance_usd"]),
        "new_source_api_calls": integer(source["new_llm_generation_calls"]), "local_compute_usd": None,
        "source_failure_rows_without_jev_call": integer(source["source_failure_rows_without_jev_call"]),
        "reused_review_conditions": integer(source["reused_old_review_conditions"]), "per_condition": [],
        "notes": ["Known reported charges and conservative reservations are different quantities; unknown charges are not free calls.",
            "These per-condition costs cover only newly executed Jev reviews. Cached hosted proposals remain in prior accounting; all 24 text-review conditions are new.",
            "Local compute cost is unmeasured. A deployed chain incurs both source and reviewer expense."]}
    require(integer(source["reported_cost_requests"]) + result["unknown_cost_calls"] == result["new_review_calls"],
            "Cost request counts do not reconcile")
    require(result["new_source_api_calls"] == 0 and result["reused_review_conditions"] == 0, "Unexpected cost scope")
    require(Decimal(result["prior_reserved_usd"]) + Decimal(result["new_review_reserved_usd"]) == Decimal(result["cumulative_reserved_usd"])
            and Decimal(result["known_reported_api_usd"]) <= Decimal(result["new_review_reserved_usd"])
            and Decimal(result["cumulative_reserved_usd"]) <= Decimal(result["authorized_usd"]), "USD totals do not reconcile")
    require(Decimal(result["stage_authorized_usd"]) == Decimal("1.60")
            and Decimal(result["new_review_reserved_usd"]) <= Decimal(result["stage_authorized_usd"])
            and Decimal(result["numeric_reserved_allowance_usd"]) == Decimal("4.032000000")
            and Decimal(result["authorized_usd"]) == Decimal("25.00"), "Text stage or aggregate budget differs")
    expected = {(dataset, model, shots) for dataset in DATASETS for model in scientific.MODEL_NAMES for shots in (0, 4)
                if not (model in scientific.REUSED_REVIEW_MODELS and shots == 4)}
    seen = set()
    for item in source["condition_costs"]:
        require(item["model_key"] in MODEL_KEYS, "Unexpected cost model")
        model = MODEL_KEYS[item["model_key"]]
        key = item["dataset"], model, item["shots_per_class"]
        require(key in expected and key not in seen, "Unexpected or duplicate condition cost")
        seen.add(key)
        result["per_condition"].append({"dataset": key[0], "source_model": model, "shots_per_class": key[2],
            "model_requests": integer(item["model_requests"]), "known_reported_api_usd": usd(item["known_reported_api_usd"]),
            "unknown_cost_requests": integer(item["unknown_cost_requests"]), "conservative_usd": usd(item["conservative_usd"])})
    require(seen == expected and sum(item["model_requests"] for item in result["per_condition"]) == result["new_review_calls"]
            and sum(item["unknown_cost_requests"] for item in result["per_condition"]) == result["unknown_cost_calls"]
            and sum(Decimal(item["known_reported_api_usd"]) for item in result["per_condition"]) == Decimal(result["known_reported_api_usd"])
            and sum(Decimal(item["conservative_usd"]) for item in result["per_condition"]) == Decimal(result["new_review_reserved_usd"]),
            "Per-condition costs do not reconcile")
    return result


def public_run_id(row, condition):
    if not row["run_id"].startswith("pending::"):
        return identifier(row["run_id"])
    dataset, model, arm, budget = condition
    if arm == "review":
        model_key = next(key for key, value in MODEL_KEYS.items() if value == model)
        return f"{dataset}__{model_key}__k{budget}__jev-review"
    return "pending-" + hashlib.sha256(json.dumps(condition).encode()).hexdigest()[:24]


def sanitize(report, *, allow_incomplete=False):
    """Allowlist projection after build() independently verifies raw evidence."""
    completion(report, allow_incomplete=allow_incomplete)
    samples = report["bootstrap_samples"]
    result = {"schema_version": 1, "scope": "Exploratory text classification and cached-label Jev review",
        "completion": {name: report[name] for name in ("status", "complete_runs", "expected_runs", "complete_comparisons", "expected_comparisons")},
        "datasets": list(DATASETS.values()),
        "models": [{"id": model, "label": label, "family": family(model)} for model, label in
                   {**scientific.MODEL_NAMES, scientific.JEV: "Jev 1.13", **scientific.CLASSICAL_NAMES}.items()],
        "runs": [], "comparisons": [], "costs": cost_summary(report["costs"], allow_incomplete=allow_incomplete), "limitations": list(LIMITATIONS)}
    if report["status"] != "complete":
        result["limitations"].insert(0, "Incomplete operational snapshot: pending or halted conditions have no score, confidence interval or failure count. Partial checkpoints are never scored; completed arms alone do not establish an aggregate conclusion.")
        result["costs"]["notes"].insert(0, "Spending includes audited partial and halted attempts, including unknown charges, while only complete conditions receive performance scores.")
    expected = set(scientific.expected_conditions())
    by_condition, by_id, original_ids, original_rows = {}, {}, {}, {}
    for row in report["runs"]:
        key = scientific.row_key(row)
        require(key in expected and key not in by_condition and row["run_id"] not in original_ids, "Unexpected or duplicate run condition")
        dataset, model, arm, budget = key
        require(row["arm"] == arm, "Run arm differs from declared condition")
        descriptor = DATASETS[dataset]
        n_test = descriptor["n_test"]
        n_classes = descriptor["n_classes"]
        n_train = descriptor["full_training_labels"] if budget is None else budget*n_classes
        complete = row["status"] == "complete"
        require(row["status"] in {"complete", "running", "stopped_after_three_consecutive_errors", "stopped_after_billing_error", "pending: no complete audited artifact"}, "Unexpected run status")
        if complete:
            require(row["n_test"] == n_test and row["n_classes"] == n_classes, "Dataset test size/classes differ")
            require(row["train_labels"] == n_train, "Training-label budget differs")
            ci = intervals(row["group_bootstrap"], paired=False, n_test=n_test, samples=samples)
            for metric in ("accuracy", "macro_f1"):
                require(math.isclose(number(row[metric], 0, 1), ci[metric], abs_tol=1e-12), "CI estimate differs from score")
        else:
            require(all(row.get(name) is None for name in ("accuracy", "macro_f1", "balanced_accuracy", "n_failures", "failure_rate",
                "n_test", "train_labels", "group_bootstrap", "probability_coverage", "log_loss", "brier_sum", "review_transitions")),
                "Incomplete run contains an invented or partial score")
            ci = {name: None for name in ("accuracy", "accuracy_ci95", "macro_f1", "macro_f1_ci95")}
        labels = {**scientific.MODEL_NAMES, scientific.JEV: "Jev alone", **scientific.CLASSICAL_NAMES}
        item = {"run_id": public_run_id(row, key), "status": row["status"], "dataset": dataset, "model": row["model"],
            "display_model": labels[model] + (" → Jev" if arm == "review" else ""),
            "source_model": model if arm in ("base", "review") else None,
            "source_run_id": row["source_run_id"] if arm == "review" else None,
            "reviewer_model": scientific.JEV if arm == "review" else None,
            "model_family": family(model), "arm": arm,
            "jev_mode": "review" if arm == "review" else "alone" if arm == "direct" else "none",
            "label_budget": "full_training" if budget is None else "zero_shot" if budget == 0 else "four_per_class",
            "shots_per_class": None if arm == "classical" else budget,
            "train_per_class": budget, "train_labels": n_train if complete else None, "n_test": n_test if complete else None, "n_classes": n_classes,
            **ci, "balanced_accuracy": number(row["balanced_accuracy"], 0, 1) if complete else None,
            "n_failures": integer(row["n_failures"], 0, n_test) if complete else None, "failure_rate": number(row["failure_rate"], 0, 1) if complete else None,
            "probability_coverage": number(row["probability_coverage"], 0, 1) if complete else None,
            "log_loss": None if row.get("log_loss") is None else number(row["log_loss"], 0),
            "brier_sum": None if row.get("brier_sum") is None else number(row["brier_sum"], 0, 2),
            "ci_method": row["group_bootstrap"]["method"] if complete else None, "ci_samples": samples if complete else None, "ci_n_groups": n_test if complete else None,
            "review_transitions": transitions(row["review_transitions"], n_test) if arm == "review" and complete else None}
        if complete:
            require(math.isclose(item["failure_rate"], item["n_failures"]/n_test, abs_tol=1e-12), "Failure count differs from rate")
        if arm == "review" and complete:
            transition = item["review_transitions"]
            require(item["n_failures"] == transition["source_failure_rows"] + transition["review_stage_failure_rows"],
                    "Pipeline failure counts differ")
        result["runs"].append(item)
        require(item["run_id"] not in by_id, "Public run IDs collide")
        original_ids[row["run_id"]] = item["run_id"]
        original_rows[key] = row
        by_condition[key], by_id[item["run_id"]] = item, item
    require(set(by_condition) == expected, "Missing run conditions")
    for key, item in by_condition.items():
        if key[2] == "review":
            source_key = key[0], key[1], "base", key[3]
            require(item["source_run_id"] == original_rows[source_key]["run_id"], "Review source linkage differs")
            source = by_condition[source_key]
            require(item["status"] != "complete" or source["status"] == "complete", "Complete review has an incomplete source")
            item["source_run_id"] = source["run_id"]
    expected_comparisons = {(kind, by_condition[a]["run_id"], by_condition[b]["run_id"]): equal
                            for kind, a, b, equal in scientific.comparison_specs()}
    seen = set()
    for comparison in report["comparisons"]:
        require(comparison["a"] in original_ids and comparison["b"] in original_ids, "Paired contrast lacks a source run")
        key = comparison["kind"], original_ids[comparison["a"]], original_ids[comparison["b"]]
        require(key in expected_comparisons and key not in seen, "Unexpected or duplicate paired contrast")
        seen.add(key)
        a, b = by_id[key[1]], by_id[key[2]]
        require(comparison["dataset"] == a["dataset"] == b["dataset"] and
                comparison["equal_new_label_budget"] is expected_comparisons[key] and
                comparison["train_per_class_a"] == a["train_per_class"] and comparison["train_per_class_b"] == b["train_per_class"],
                "Paired comparison identities or label budgets differ")
        complete = comparison["status"] == "complete"
        require(comparison["status"] == ("complete" if a["status"] == b["status"] == "complete" else "pending"),
                "Paired comparison status differs from its runs")
        if complete:
            ci = intervals(comparison["paired_bootstrap"], paired=True, n_test=a["n_test"], samples=samples)
            require(all(math.isclose(ci[metric], a[metric]-b[metric], abs_tol=1e-12) for metric in ("accuracy", "macro_f1")),
                    "Paired delta differs from run scores")
        else:
            require(comparison.get("paired_bootstrap") is None and comparison.get("transitions") is None, "Pending contrast has partial scores")
            ci = {name: None for name in ("accuracy", "accuracy_ci95", "macro_f1", "macro_f1_ci95")}
        change = transitions(comparison["transitions"], a["n_test"]) if key[0] == "review_minus_source" and complete else None
        if change is not None:
            require(a["source_run_id"] == b["run_id"] and change == a["review_transitions"] and
                    math.isclose(change["accuracy_delta_pp"]/100, ci["accuracy"], abs_tol=1e-12), "Review source or fix/harm delta differs")
        result["comparisons"].append({"kind": key[0], "status": comparison["status"], "dataset": a["dataset"], "a": key[1], "b": key[2],
            "equal_new_label_budget": expected_comparisons[key], "train_per_class_a": a["train_per_class"],
            "train_per_class_b": b["train_per_class"], "accuracy_delta": ci["accuracy"], "accuracy_delta_ci95": ci["accuracy_ci95"],
            "macro_f1_delta": ci["macro_f1"], "macro_f1_delta_ci95": ci["macro_f1_ci95"],
            "ci_method": comparison["paired_bootstrap"]["method"] if complete else None, "ci_samples": samples if complete else None, "ci_n_groups": a["n_test"] if complete else None,
            "transitions": change})
    require(seen == set(expected_comparisons), "Missing paired contrasts")
    return result


def build(root=ROOT, output=None, *, allow_incomplete=False):
    root = Path(root).resolve()
    source = root / "results/text_extension/COMPARISON.json"
    raw = source.read_bytes()
    report = json.loads(raw)
    completion(report, allow_incomplete=allow_incomplete)  # Default fails before touching the output.
    audited = scientific.collect(root, samples=report["bootstrap_samples"])
    require(report == audited, "Saved report differs from independently revalidated evidence; regenerate the scientific report first")
    result = sanitize(report, allow_incomplete=allow_incomplete)
    result["provenance"] = {"source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "report_builder_sha256": report["source_sha256"], "protocol_sha256": report["protocol_sha256"],
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bootstrap_samples": report["bootstrap_samples"], "seed": report["seed"]}
    # JSON serialization also rejects any unexpected NaN before atomic replacement.
    json.dumps(result, allow_nan=False)
    target = Path(output) if output else root / "dashboard/dist/text-data.json"
    save_json(target, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicitly export audited pending/halted arms with null scores")
    args = parser.parse_args()
    result = build(args.root, args.output, allow_incomplete=args.allow_incomplete)
    print(json.dumps({"status": "exported", "runs": len(result["runs"]), "comparisons": len(result["comparisons"])}))
