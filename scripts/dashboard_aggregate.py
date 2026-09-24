"""Shared aggregate validation and projection for the two reviewer dashboards.

This module reads no benchmark artifacts and performs no inference. The study
exporters retain their own raw-evidence audits, dataset definitions and budgets.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re


MODEL_KEYS = {"qwen_small": "Qwen/Qwen2.5-0.5B-Instruct", "qwen_main": "Qwen/Qwen3-4B-Instruct-2507",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "granite": "ibm-granite/granite-3.3-2b-instruct",
    "luna": "gpt-5.6-luna", "astra": "gpt-6-astra"}
TRANSITION_COUNTS = ("wrong_to_correct", "correct_to_wrong", "correct_to_wrong_label", "correct_to_failure",
    "both_correct", "both_wrong", "changed_predictions", "source_failure_rows", "review_stage_failure_rows")


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


def family(model, scientific):
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


def public_run_id(row, condition):
    if not row["run_id"].startswith("pending::"):
        return identifier(row["run_id"])
    dataset, model, arm, budget = condition
    if arm == "review":
        model_key = next(key for key, value in MODEL_KEYS.items() if value == model)
        return f"{dataset}__{model_key}__k{budget}__jev-review"
    return "pending-" + hashlib.sha256(json.dumps(condition).encode()).hexdigest()[:24]


def sanitize_report(report, *, scientific, datasets, scope, limitations, cost_summary,
                    allowed_run_statuses, allow_incomplete=False):
    """Project the shared run/contrast schema after the caller audits raw evidence.

    Dataset descriptors, allowed statuses and cost validation remain with each
    study. Only explicitly selected aggregate fields enter the public payload.
    """
    completion(report, allow_incomplete=allow_incomplete)
    samples = report["bootstrap_samples"]
    result = {"schema_version": 1, "scope": scope,
        "completion": {name: report[name] for name in ("status", "complete_runs", "expected_runs", "complete_comparisons", "expected_comparisons")},
        "datasets": list(datasets.values()),
        "models": [{"id": model, "label": label, "family": family(model, scientific)} for model, label in
                   {**scientific.MODEL_NAMES, scientific.JEV: "Jev 1.13", **scientific.CLASSICAL_NAMES}.items()],
        "runs": [], "comparisons": [], "costs": cost_summary(report["costs"], allow_incomplete=allow_incomplete), "limitations": list(limitations)}
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
        descriptor = datasets[dataset]
        n_test = descriptor["n_test"]
        n_classes = descriptor["n_classes"]
        n_train = descriptor["full_training_labels"] if budget is None else budget*n_classes
        complete = row["status"] == "complete"
        require(row["status"] in allowed_run_statuses, "Unexpected run status")
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
            "model_family": family(model, scientific), "arm": arm,
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


def source_sha256():
    """Identify this shared implementation in newly generated exports."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
