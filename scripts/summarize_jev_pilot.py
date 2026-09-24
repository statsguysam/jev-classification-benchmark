#!/usr/bin/env python3
"""Recompute Jev pilot and probability-quality tables from completed artifacts.

PYTHONPATH=src .venv/bin/python scripts/summarize_jev_pilot.py
No inference, fitting, calibration, or modification of original runs is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from audit_cross_environment import audit, load_json, read_run, require
from compare_combined_pilot import completed_candidate, validate_jev_route
from jevbench.metrics import evaluate
from jevbench.runner import environment
from jevbench.types import Prediction, Row

DISPLAY = {"jev": "Jev 1.13", "qwen05": "Qwen2.5 0.5B", "qwen4b": "Qwen3 4B", "nb_pilot": "TF-IDF MultinomialNB"}
PROBABILITY_FIELDS = ("n_probability_rows", "probability_coverage", "log_loss", "brier_sum", "ece_15_equal_width",
                      "roc_auc", "average_precision", "zero_true_class_probability_rows",
                      "probability_renormalized_rows", "chosen_label_argmax_disagreement_rows", "reliability_bins")


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def recompute(directory: Path, record: dict, test: dict):
    raw = (directory / "predictions.jsonl").read_bytes()
    values = [json.loads(line) for line in raw.splitlines() if line.strip()]
    predictions = [Prediction(**value) for value in values]
    rows = [Row(value["id"], "", value["label"]) for value in test["rows"]]
    metrics = evaluate(rows, predictions, len(record["labels"]))
    for key in ("n_test", "n_failures", "accuracy", "macro_f1", *PROBABILITY_FIELDS):
        require(equivalent(metrics.get(key), record["metrics"].get(key)), f"Recomputed probability/label metric {key} differs from saved run {record['run_id']}")
    valid = [value for value in values if value.get("probabilities") is not None and value.get("error") is None and type(value.get("label")) is int and 0 <= value["label"] < len(record["labels"])]
    kinds = sorted({value.get("metadata", {}).get("probability_kind", "unspecified") for value in valid})
    failed = [value for value in values if value.get("error") is not None or type(value.get("label")) is not int or not 0 <= value["label"] < len(record["labels"])]
    error_counts = Counter(value["error"] if isinstance(value.get("error"), str) else "missing/invalid label without a recorded error string" for value in failed)
    provider = record["config"].get("provider")
    if provider in ("jev", "typesafe"):
        validate_jev_route(record, values)
        require(not valid or kinds == ["jev_choice_distribution"], "Jev native probability provenance is missing or mixed")
        semantics = "API-returned Choice distribution; OpenRouter transport"
    elif provider in ("hf", "huggingface"):
        require(not valid or kinds == ["label_sequence_likelihood_normalized"], "Qwen restricted-label probability provenance is missing or mixed")
        semantics = "Normalized likelihood of numeric class ID plus EOS"
    elif record["method"] == "classical" and record["config"]["model"] == "multinomial_nb":
        require(record["training"].get("probability_kind") == "native_uncalibrated", "Unexpected Naive Bayes probability provenance")
        semantics = "MultinomialNB native predict_proba; no post-hoc calibration"
    else:
        raise ValueError("Unsupported probability model for this fixed report")
    return metrics, {"predictions_sha256": hashlib.sha256(raw).hexdigest(), "recorded_probability_kinds": kinds,
                     "failure_breakdown": dict(sorted(error_counts.items())), "failed_rows": len(failed),
                     "failed_rows_without_retained_probabilities": sum(value.get("probabilities") is None for value in failed),
                     "probability_semantics": semantics, "resolved_models": sorted({value.get("metadata", {}).get("resolved_model") for value in values if value.get("metadata", {}).get("resolved_model")})}


def collect(results_root: Path, data_root: Path, seed: int):
    rows, pending, evidence = [], [], []
    core_source_sha256 = environment()["source_sha256"]
    for dataset in ("sst2", "trec"):
        for model in DISPLAY:
            methods = ("classical",) if model == "nb_pilot" else ("zero_shot", "few_shot") if model == "jev" else ("zero_shot", "few_shot", "lora")
            for method in methods:
                directory, status = completed_candidate(results_root, dataset, model, method, seed)
                if directory is None:
                    pending.append(status)
                    continue
                anchor_key, anchor_method = ("qwen05", "zero_shot") if method == "zero_shot" else ("nb_pilot", "classical")
                anchor, anchor_status = completed_candidate(results_root, dataset, anchor_key, anchor_method, seed)
                require(anchor is not None, f"Required content/training reference unavailable: {anchor_status}")
                proof = audit(directory, anchor, data_root / dataset)
                require(proof["eligible_for_separately_labeled_matched_pairing"] and proof["raw_content_recomputed"], f"Data/training audit failed: {directory}")
                record, test = read_run(directory)
                require(record.get("implementation_sha256") == core_source_sha256 == proof["run_b"]["implementation_sha256"], "Probability report requires the same frozen local inference core as its reference")
                metrics, prediction_evidence = recompute(directory, record, test)
                require(metrics["n_test"] == 200 and record["dataset_manifest"]["text_policy"]["max_chars"] == 2000, "Update the fixed 200-row / 2,000-character pilot description for this run")
                row = {"dataset": dataset, "model": DISPLAY[model], "model_key": model, "method": method,
                       "training_labels": len(record["training_example_ids"]), "seed": seed,
                       "run_id": record["run_id"], "source_path": str(directory / "run.json"),
                       "manifest_sha256": record["manifest_sha256"], "prepared_content_sha256": record["dataset_manifest"]["prepared_content_sha256"],
                       "probability_semantics": prediction_evidence["probability_semantics"]}
                row["failed_rows_without_retained_probabilities"] = prediction_evidence["failed_rows_without_retained_probabilities"]
                row["method_label"] = (("QLoRA 4/class" if record["adapter_training"].get("load_in_4bit") else "LoRA 4/class") if method == "lora" else {"zero_shot": "zero-shot", "few_shot": "few-shot 4/class", "classical": "classical 4/class"}[method])
                row.update({key: metrics.get(key) for key in ("n_test", "accuracy", "macro_f1", "n_failures", *PROBABILITY_FIELDS) if key != "reliability_bins"})
                for metric in ("accuracy", "macro_f1"):
                    ci = record["metrics"].get("bootstrap", {}).get("metrics", {}).get(metric, {}).get("ci95")
                    row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = ci if ci else (None, None)
                row["bootstrap_samples"] = record["metrics"].get("bootstrap", {}).get("samples")
                rows.append(row)
                evidence.append({"run_id": record["run_id"], "content_audit": proof, "prediction_evidence": prediction_evidence,
                                 "recomputed_probability_metrics": {key: metrics.get(key) for key in PROBABILITY_FIELDS},
                                 "provider": record["config"].get("provider"), "requested_model": record["config"]["model"],
                                 "wrapper_provenance": record["config"].get("budget_guard")})
    return rows, pending, evidence


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def summarize(results_root: Path, data_root: Path, output: Path, seed=42):
    protected = [data_root, *(results_root / source for source in ("pilot", "colab", "hosted", "jev"))]
    require(not any(output.resolve().is_relative_to(path.resolve()) for path in protected), "Summary output cannot be inside original run/data directories")
    rows, pending, evidence = collect(results_root, data_root, seed)
    jev = [row for row in rows if row["model_key"] == "jev"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps({"schema_version": 1, "rows": rows, "pending": pending, "evidence": evidence}, indent=2, sort_keys=True) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["dataset", "model", "method"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Jev pilot and probability quality", "", f"**{len(jev)} completed Jev runs** are recorded below. Only completed artifacts supply measurements. Jev is accessed as `typesafe/jev-1.13` through OpenRouter; the frozen inference core remains unchanged and wrapper provenance is retained in the JSON evidence.", "", "This is the shared SST-2/TREC pilot: 200 held-out rows per dataset, selection seed 42 and a 2,000-character input prefix. Zero-shot uses no training examples. Few-shot uses exactly four per class (eight SST-2 labels or 24 TREC labels), matched to the classical and other few-shot/adapted arms; no development labels are used. The different regimes remain labeled separately. Jev adapter training was not run.", "", "| Dataset | Jev method | New labels | Accuracy | Macro F1 | Errors | Probability coverage | Artifact |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in jev:
        link = os.path.relpath(row["source_path"], output.parent).replace(os.sep, "/")
        lines.append(f"| {row['dataset']} | {row['method_label']} | {row['training_labels']} | {fmt(row['accuracy'])} | {fmt(row['macro_f1'])} | {row['n_failures']} | {row['n_probability_rows']}/{row['n_test']} | [run]({link}) |")
    lines += ["", "## Recorded Jev failures", "", "The fixed validator counts failed predictions as incorrect in accuracy and as missing predictions in macro F1. Failures include any recorded transport errors and output-validation errors; a choice-versus-maximum-probability rejection is an output-validation failure, not evidence of a timeout. The table below uses the original saved row error strings without reclassifying the failure or imputing a label.", "", "The [official Choice contract](https://docs.typesafe.ai/primitives/choice) defines the chosen option as the one with the highest probability. The frozen validator rejects a response when `p(choice) < max(p) - 1e-6`. A rejection through OpenRouter does not isolate whether its cause lies in the underlying model or a serving layer. The retained artifacts do not support a claim about that cause or the numerical size of the rejected disagreement.", "", "| Dataset | Method | Original recorded error | Rows |", "|---|---|---|---:|"]
    by_run = {item["run_id"]: item["prediction_evidence"] for item in evidence}
    for row in jev:
        errors = by_run[row["run_id"]]["failure_breakdown"]
        for error, count in errors.items():
            displayed_error = error.replace("|", "\\|").replace("`", "\\`")
            lines.append(f"| {row['dataset']} | {row['method_label']} | `{displayed_error}` | {count} |")
        if not errors:
            lines.append(f"| {row['dataset']} | {row['method_label']} | No recorded failures | 0 |")
    failures = sum(row["n_failures"] for row in jev)
    missing_distributions = sum(row["failed_rows_without_retained_probabilities"] for row in jev)
    lines += ["", f"Of {failures} recorded Jev failures, {missing_distributions} have no retained probability distribution. Probability scores exclude every failed prediction, including choice-versus-argmax rejections. The rejected raw distributions are unavailable when the saved field is null; their NLL, Brier and calibration errors cannot be reconstructed and are not imputed. Original prediction files retain available usage, request IDs, model and routing metadata. The valid-row `chosen_label_argmax_disagreement_rows` metric applies only to retained valid predictions: a zero value does not mean that no raw API response disagreed with its argmax.", ""]
    lines += ["", "## Probability metrics on the fixed datasets", "", "Every probability metric below was recomputed from saved predictions and checked against the recorded score. Exact test content and matched training IDs were verified against frozen prepared files. Jev uses its API-returned Choice distribution; Qwen scores complete numeric class IDs plus EOS and normalizes over the permitted labels; Naive Bayes uses its native `predict_proba`. These distributions have different semantics. Their observed quality can be compared on the same labels and rows, but a score difference does not isolate architecture or establish a general calibration advantage.", "", "Lower NLL, Brier sum and ECE are better within the same dataset. NLL clips probabilities at 1e-15 then renormalizes; a finite NLL does not remove the significance of an exact zero assigned to the true class. Brier is the sum of squared class-probability errors per row, without division by the class count. ECE uses 15 equal-width bins of the maximum predicted probability and the argmax class's correctness. Bin estimates are noisy at 200 rows; no post-hoc calibration, test-guided thresholds or probability-parameter tuning were performed.", "", "Probability metrics are conditional on rows with a valid class prediction and distribution. Coverage and failures are reported so missing distributions cannot masquerade as better calibration. These are point estimates, not confidence intervals or multiplicity-adjusted claims. Accuracy and macro F1 include inference failures; the table does not pool different datasets or training-label budgets.", ""]
    for dataset in ("sst2", "trec"):
        lines += [f"### {dataset.upper()}", "", "| Model | Method | New labels | Coverage | NLL | Brier sum | ECE | True-class zeros |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["dataset"] == dataset:
                link = os.path.relpath(row["source_path"], output.parent).replace(os.sep, "/")
                lines.append(f"| [{row['model']}]({link}) | {row['method_label']} | {row['training_labels']} | {row['n_probability_rows']}/{row['n_test']} | {fmt(row['log_loss'])} | {fmt(row['brier_sum'])} | {fmt(row['ece_15_equal_width'])} | {row['zero_true_class_probability_rows'] if row['zero_true_class_probability_rows'] is not None else 'N/A'} |")
        lines.append("")
    lines += ["## Matched-label paired comparisons", "", "The separate [comparison index](comparisons/combined/index.md) includes Jev few-shot minus fixed TF-IDF Naive Bayes, Astra few-shot and Qwen 4B few-shot. Each comparison audits the exact same training IDs/seed and test items. These are additional exploratory, unadjusted contrasts selected after earlier pilot results were visible; they are not prospectively registered. No equal-training paired claim is made for Jev zero-shot versus few-shot. Cross-model/protocol contrasts remain descriptive.", "", "OpenAI label-only runs do not expose a class-probability vector in this experiment, so they have no NLL, Brier or ECE entry. Jev probabilities alone do not establish an advantage over absent OpenAI probability measurements. [Combined accuracy report](COMPARISON.md) · [Colab content audit](COLAB_AUDIT.md).", ""]
    if pending:
        lines += ["## Pending", "", *[f"- {status}" for status in pending], ""]
    lines += [f"[CSV]({output.with_suffix('.csv').name}) · [Full audit, reliability bins and provenance]({output.with_suffix('.json').name})", "", "Rebuild with `PYTHONPATH=src .venv/bin/python scripts/summarize_jev_pilot.py`.", ""]
    output.write_text("\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--data-root", type=Path, default=Path("data/pilot"))
    parser.add_argument("--output", type=Path, default=Path("results/JEV_PILOT.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    require(args.seed == 42, "This fixed pilot narrative currently describes selection seed 42")
    rows = summarize(args.results_root, args.data_root, args.output, args.seed)
    print(f"Wrote {args.output}; {sum(row['model_key'] == 'jev' for row in rows)} completed Jev runs, {len(rows)} probability reference rows")


if __name__ == "__main__":
    main()
