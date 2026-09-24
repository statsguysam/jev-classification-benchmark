"""Reaudit a numeric report and export aggregate-only dashboard data.

No inference, site editing, report regeneration, or hosted calls. Incomplete or
stale evidence fails before the existing dashboard asset can be replaced.
Partial publication requires explicit --allow-incomplete; unscored arms stay null.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import summarize_expanded_numeric as scientific
from jevbench.runner import save_json
import dashboard_aggregate
from dashboard_aggregate import MODEL_KEYS, TRANSITION_COUNTS, completion, integer, require, usd

DATASETS = {
    "breast_cancer": {"id": "breast_cancer", "label": "Breast Cancer Wisconsin Diagnostic",
        "data_type": "numeric", "task": "binary", "n_classes": 2, "n_features": 30,
        "n_test": 114, "full_training_labels": 341,
        "classes": [{"id": 0, "label": "malignant"}, {"id": 1, "label": "benign"}]},
    "wine": {"id": "wine", "label": "Wine", "data_type": "numeric", "task": "multiclass",
        "n_classes": 3, "n_features": 13, "n_test": 36, "full_training_labels": 106,
        "classes": [{"id": i, "label": f"cultivar {i+1}"} for i in range(3)]},
}
LIMITATIONS = [
    "Seven of eight new SmolLM2/Granite source conditions predict one class under the fixed label-plus-EOS likelihood recipe. Some source models therefore yield identical review prompts; repeated calls are not independent architecture treatments.",
    "Exploratory model expansion selected after earlier results were observed; no preregistered or causal claim about bounded outputs.",
    "Only two familiar public numerical datasets, one split and seed, 114 binary and 36 multiclass test rows; pretraining exposure cannot be excluded.",
    "Wine cultivar IDs are arbitrary without labeled examples; zero-shot results may reflect pretrained familiarity rather than transferable classification ability.",
    "Four examples per class means 8 or 12 supplied labels. Full-training native references use 341 or 106 labels and form a separate comparison budget.",
    "All failures count as incorrect. Failed source proposals prevent a review call; failed reviews do not silently fall back to the source label.",
    "Jev reviews a cached proposed class together with the original numerical row and matching examples. Bounded validity does not guarantee correctness.",
    "Original Qwen/OpenAI rendering differs from the fixed-system SmolLM2/Granite wrapper; this is not a controlled same-prompt architecture ablation.",
    "Paired group-bootstrap intervals condition on this split, examples, serialization and fitted models. Multiple comparisons are unadjusted; crossing zero does not establish equivalence.",
    "Likelihood-normalized, Choice and classical probability scores have different meanings and are not guaranteed calibrated.",
    "A deployed two-stage chain pays for both stages. This expansion reuses cached source proposals; local compute is unpriced and mixed hardware prevents a controlled latency ranking.",
    "This expansion adds no LoRA training. Breast Cancer is a historical benchmark, not clinical validation.",
]


def cost_summary(source, *, allow_incomplete=False):
    require(source["status"] in ({"complete", "halted", "in_progress"} if allow_incomplete else {"complete"})
            and source["expected_new_review_conditions"] == 20
            and 0 <= source["complete_summary_conditions"] <= source["complete_checkpoint_conditions"] <= 20
            and (source["status"] != "complete" or source["complete_summary_conditions"] == source["complete_checkpoint_conditions"] == 20),
            "Cost reconciliation is incomplete")
    require((allow_incomplete and source["status"] != "complete") or all(source[name] == 0 for name in ("reservations_without_result", "results_without_saved_prediction", "partial_checkpoint_files")),
            "Unreconciled cost evidence")
    result = {"status": source["status"], "new_review_calls": integer(source["new_model_requests"]),
        "complete_review_conditions": integer(source["complete_summary_conditions"], 0, 20), "expected_review_conditions": 20,
        **{name: integer(source[name]) for name in ("reservations_without_result", "results_without_saved_prediction", "partial_checkpoint_files")},
        "known_reported_api_usd": usd(source["known_reported_api_usd"]),
        "unknown_cost_calls": integer(source["unknown_cost_requests"]),
        "new_review_reserved_usd": usd(source["review_conservative_usd"]),
        "prior_reserved_usd": usd(source["prior_conservative_usd"]),
        "cumulative_reserved_usd": usd(source["cumulative_conservative_usd"]),
        "authorized_usd": usd(source["cumulative_authorized_usd"]),
        "new_source_api_calls": integer(source["new_llm_generation_calls"]), "local_compute_usd": None,
        "source_failure_rows_without_jev_call": integer(source["source_failure_rows_without_jev_call"]),
        "reused_review_conditions": integer(source["reused_old_review_conditions"]), "per_condition": [],
        "notes": ["Known reported charges and conservative reservations are different quantities; unknown charges are not free calls.",
            "These per-condition costs cover only newly executed Jev reviews. Four reused reviews and cached hosted proposals remain in prior accounting.",
            "Local compute cost is unmeasured. A deployed chain incurs both source and reviewer expense."]}
    require(integer(source["reported_cost_requests"]) + result["unknown_cost_calls"] == result["new_review_calls"],
            "Cost request counts do not reconcile")
    require(result["new_source_api_calls"] == 0 and result["reused_review_conditions"] == 4, "Unexpected cost scope")
    require(Decimal(result["prior_reserved_usd"]) + Decimal(result["new_review_reserved_usd"]) == Decimal(result["cumulative_reserved_usd"])
            and Decimal(result["known_reported_api_usd"]) <= Decimal(result["new_review_reserved_usd"])
            and Decimal(result["cumulative_reserved_usd"]) <= Decimal(result["authorized_usd"]), "USD totals do not reconcile")
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


def sanitize(report, *, allow_incomplete=False):
    """Apply this study's rules to the shared aggregate projection."""
    return dashboard_aggregate.sanitize_report(
        report, scientific=scientific, datasets=DATASETS,
        scope="Exploratory numerical classification and cached-label Jev review",
        limitations=LIMITATIONS, cost_summary=cost_summary,
        allowed_run_statuses={"complete", "running", "stopped_after_three_consecutive_errors",
                              "pending: no complete audited artifact"},
        allow_incomplete=allow_incomplete,
    )


def build(root=ROOT, output=None, *, allow_incomplete=False):
    root = Path(root).resolve()
    source = root / "results/numeric_expansion/COMPARISON.json"
    raw = source.read_bytes()
    report = json.loads(raw)
    completion(report, allow_incomplete=allow_incomplete)  # Default fails before touching the output.
    audited = scientific.collect(root, samples=report["bootstrap_samples"])
    require(report == audited, "Saved report differs from independently revalidated evidence; regenerate the scientific report first")
    result = sanitize(report, allow_incomplete=allow_incomplete)
    result["provenance"] = {"source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "report_builder_sha256": report["source_sha256"], "protocol_sha256": report["protocol_sha256"],
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "exporter_helper_sha256": dashboard_aggregate.source_sha256(),
        "bootstrap_samples": report["bootstrap_samples"], "seed": report["seed"]}
    # JSON serialization also rejects any unexpected NaN before atomic replacement.
    json.dumps(result, allow_nan=False)
    target = Path(output) if output else root / "dashboard/dist/numeric-data.json"
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
