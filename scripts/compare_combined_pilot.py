#!/usr/bin/env python3
"""Rebuild exploratory paired contrasts from completed immutable pilot runs.

Example: PYTHONPATH=src python scripts/compare_combined_pilot.py

This separate analysis never rewrites run manifests or relaxes compare_runs.
The frozen prepared corpus is required to independently verify the saved hashes.
The contrast list is fixed for this report, not a prospectively registered plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from audit_cross_environment import AuditError, audit, load_json, read_run, require
from jevbench.metrics import stratified_bootstrap
from sklearn.metrics import accuracy_score, f1_score

MODELS = {
    "qwen05": ("pilot", "Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen4b": ("colab", "Qwen/Qwen3-4B-Instruct-2507"),
    "luna": ("hosted", "gpt-5.6-luna"),
    "astra": ("hosted", "gpt-6-astra"),
    "jev": ("jev", "typesafe/jev-1.13"),
}
JEV_BASE_URL = "https://openrouter.ai/api/v1"
JEV_ENDPOINT = JEV_BASE_URL + "/systemone"
JEV_RESOLVED_MODELS = {"typesafe/jev-1.13", "typesafe/jev-1.13-20260917"}


def validate_jev_route(record: dict, predictions: list[dict] | None = None):
    """Bind this fixed Jev report to the saved, guarded OpenRouter route."""
    config = record["config"]
    require(config.get("provider") == "jev" and config.get("model") == MODELS["jev"][1], "Unexpected Jev provider/model for the fixed OpenRouter pilot")
    require(config.get("base_url") == JEV_BASE_URL, "Jev report requires the official OpenRouter base URL")
    guard = config.get("budget_guard", {})
    require(guard.get("route") == "OpenRouter" and guard.get("endpoint") == JEV_ENDPOINT, "Jev report requires recorded OpenRouter route/endpoint provenance")
    require(guard.get("native_protocol") == "frozen_JevClassifier_Choice", "Jev report requires frozen native Choice protocol provenance")
    require(set(guard.get("response_model_allowlist", [])) == JEV_RESOLVED_MODELS, "Unexpected Jev response model allowlist")
    wrapper_hash = hashlib.sha256(Path(__file__).with_name("run_openrouter_jev.py").read_bytes()).hexdigest()
    require(guard.get("wrapper_sha256") == wrapper_hash, "Jev wrapper SHA does not match the frozen local wrapper")
    require(guard.get("dataset") == record.get("dataset") and guard.get("seed") == record.get("seed") and guard.get("shots_per_class") == config.get("shots_per_class"), "Jev route provenance disagrees with the recorded dataset/seed/shot budget")
    for prediction in predictions or []:
        if prediction.get("error") is not None:
            continue
        metadata = prediction.get("metadata", {})
        route = metadata.get("openrouter", {})
        require(metadata.get("resolved_model") in JEV_RESOLVED_MODELS, "Successful Jev row resolved to an unverified model")
        require(metadata.get("requested_model") == MODELS["jev"][1], "Successful Jev row requested an unexpected model")
        require(route.get("provider") == "TypeSafe" and route.get("endpoint") == JEV_ENDPOINT, "Successful Jev row lacks the required TypeSafe/OpenRouter endpoint provenance")


def completed_candidate(results: Path, dataset: str, model: str, method: str, seed: int) -> tuple[Path | None, str]:
    source, model_name = MODELS[model] if model in MODELS else (model.removeprefix("nb_"), "multinomial_nb")
    candidates, incomplete = [], []
    for path in sorted((results / source).glob("*/run.json")):
        record = load_json(path)
        if (record.get("dataset"), record.get("config", {}).get("model"), record.get("method"), record.get("seed")) != (dataset, model_name, method, seed):
            continue
        config = record["config"]
        if method == "classical" and config.get("train_per_class") != 4:
            continue
        if method == "few_shot" and config.get("shots_per_class") != 4:
            continue
        if method == "lora" and record.get("adapter_training", {}).get("train_per_class") != 4:
            continue
        if model == "jev":
            validate_jev_route(record)
        (candidates if record.get("status") == "complete" else incomplete).append(path.parent)
    require(len(candidates) <= 1, f"Ambiguous completed runs for {dataset}/{model}/{method}/seed{seed}: {candidates}")
    if candidates:
        return candidates[0], "complete"
    return None, f"{dataset}/{model}/{method}/seed{seed}: " + ("present but incomplete" if incomplete else "not imported or not run")


def contrast_specs():
    specs = []
    for model in ("qwen05", "qwen4b"):
        specs.append((f"{model}_lora_minus_few4", model, "lora", model, "few_shot", False))
    for model in MODELS:
        nb = "nb_colab" if model == "qwen4b" else "nb_pilot"
        specs.append((f"{model}_few4_minus_nb_k4", model, "few_shot", nb, "classical", True))
    specs += [("astra_few4_minus_luna_few4", "astra", "few_shot", "luna", "few_shot", True),
              ("qwen4b_few4_minus_astra_few4", "qwen4b", "few_shot", "astra", "few_shot", True),
              ("jev_few4_minus_astra_few4", "jev", "few_shot", "astra", "few_shot", True),
              ("jev_few4_minus_qwen4b_few4", "jev", "few_shot", "qwen4b", "few_shot", True)]
    return specs


def read_predictions(directory: Path, record: dict, test: dict) -> tuple[list[int], dict]:
    raw = (directory / "predictions.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if record["config"].get("provider") in ("jev", "typesafe"):
        validate_jev_route(record, rows)
    by_id = {row["row_id"]: row for row in rows}
    expected = [row["id"] for row in test["rows"]]
    require(len(by_id) == len(rows) and set(by_id) == set(expected), "Prediction IDs must exactly match held-out IDs with no duplicates")
    classes = len(record["labels"])
    labels = [by_id[ident].get("label") if by_id[ident].get("error") is None and type(by_id[ident].get("label")) is int and 0 <= by_id[ident]["label"] < classes else -1 for ident in expected]
    targets = [row["label"] for row in test["rows"]]
    measured = {"accuracy": float(accuracy_score(targets, labels)),
                "macro_f1": float(f1_score(targets, labels, labels=list(range(classes)), average="macro", zero_division=0)),
                "n_failures": labels.count(-1)}
    for metric, value in measured.items():
        require(abs(value - record["metrics"][metric]) < 1e-12, f"Recomputed {metric} differs from saved run metrics")
    return labels, {"predictions_sha256": hashlib.sha256(raw).hexdigest(),
                    "prediction_order_matches_test_order": [row["row_id"] for row in rows] == expected,
                    "recomputed_metrics": measured}


def compare_pair(a_dir: Path, b_dir: Path, prepared: Path, samples: int, bootstrap_seed: int) -> dict:
    proof = audit(a_dir, b_dir, prepared)
    require(proof["eligible_for_separately_labeled_matched_pairing"] and proof["raw_content_recomputed"], "Content and matched-training audit did not pass")
    a, test_a = read_run(a_dir)
    b, test_b = read_run(b_dir)
    if a.get("method") == "lora" and b.get("method") == "few_shot":
        require(a["config"].get("model") == b["config"].get("model") and a["config"].get("revision") == b["config"].get("revision"), "LoRA versus few-shot contrast requires the same pinned base model revision")
        require(a.get("adapter_training", {}).get("resolved_revision") == a["config"].get("revision"), "Adapter training revision differs from the evaluated base model")
    prediction_a, evidence_a = read_predictions(a_dir, a, test_a)
    prediction_b, evidence_b = read_predictions(b_dir, b, test_b)
    result = stratified_bootstrap([row["label"] for row in test_a["rows"]], prediction_a, prediction_b,
                                 n_classes=len(a["labels"]), samples=samples, seed=bootstrap_seed)
    strict_manifest_match = test_a == test_b
    result.update({"run_a": a["run_id"], "run_b": b["run_id"], "delta_direction": "run_a minus run_b",
                   "dataset": a["dataset"], "n_test": len(test_a["rows"]),
                   "equal_training_ids_and_seed": True, "training_rows_per_run": len(a["training_example_ids"]),
                   "content_audit": proof, "prediction_evidence_a": evidence_a, "prediction_evidence_b": evidence_b,
                   "strict_compare_runs_manifest_requirement_satisfied": strict_manifest_match,
                   "strict_compare_runs_note": "Original exact-manifest requirement is satisfied." if strict_manifest_match else "Original compare_runs remains unavailable because original manifests differ. This separately labeled contrast uses exact content and training evidence verified against frozen prepared files; no original manifest was changed.",
                   "multiple_comparisons_adjustment": "none",
                   "interpretation": "Exploratory paired test-item contrast, conditional on one selection seed and fixed model/prompt/training runs. Unadjusted 95% percentile intervals do not include training-seed, prompt-choice, or model-selection uncertainty; this report's fixed contrast list was not prospectively registered."})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--data-root", type=Path, default=Path("data/pilot"))
    parser.add_argument("--output", type=Path, default=Path("results/comparisons/combined"))
    parser.add_argument("--datasets", nargs="+", choices=("sst2", "trec"), default=["sst2", "trec"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args(argv)
    require(args.bootstrap >= 100, "At least 100 bootstrap samples required")
    protected = [args.data_root, *(args.results_root / source for source in ("pilot", "colab", "hosted", "jev"))]
    require(not any(args.output.resolve().is_relative_to(path.resolve()) for path in protected), "Output must not be inside original run or prepared-data directories")
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for dataset in args.datasets:
        for key, model_a, method_a, model_b, method_b, cross_protocol in contrast_specs():
            slug = f"{dataset}_{key}_seed{args.seed}"
            item = {"contrast": slug, "dataset": dataset, "cross_protocol_or_model_comparison": cross_protocol}
            try:
                a, status_a = completed_candidate(args.results_root, dataset, model_a, method_a, args.seed)
                b, status_b = completed_candidate(args.results_root, dataset, model_b, method_b, args.seed)
                if a is None or b is None:
                    item.update({"status": "pending", "reason": "; ".join(status for path, status in [(a, status_a), (b, status_b)] if path is None)})
                else:
                    result = compare_pair(a, b, args.data_root / dataset, args.bootstrap, args.bootstrap_seed)
                    result["contrast"] = slug
                    result["cross_protocol_or_model_comparison"] = cross_protocol
                    result["protocol_note"] = "Descriptive comparison across model/scoring families; it does not isolate a method effect." if cross_protocol else "Same base model: adapted weights with no demonstrations versus frozen weights with demonstrations; this comparison changes both adaptation and inference context."
                    filename = slug + ".json"
                    (args.output / filename).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                    item.update({"status": "complete", "file": filename, "metrics": result["metrics"],
                                 "manifest_hashes_are_equal": result["content_audit"]["manifest_hashes_are_equal"]})
            except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
                item.update({"status": "blocked", "reason": str(exc)})
            index.append(item)
            print(f"{item['status']}: {slug}", flush=True)
    (args.output / "index.json").write_text(json.dumps({"schema_version": 1, "contrasts": index}, indent=2, sort_keys=True) + "\n")
    lines = ["# Exploratory paired comparisons", "", "Differences are A minus B. All completed contrasts pass exact ordered test-content and matched-training checks, recomputed against frozen prepared files. Original manifest hashes remain visible and unchanged. The classical reference is the fixed TF-IDF MultinomialNB model, not the best observed classifier.", "", "Intervals use 2,000 paired stratified test-item bootstrap resamples by default; each JSON records the actual count and bootstrap seed. These are multiple unadjusted exploratory contrasts, conditional on one selection seed and fixed runs. The contrast list was fixed for this report after some pilot outcomes were already visible; it is not a preregistration. Cross-family comparisons are descriptive and do not isolate model size, adaptation, or scoring protocol.", "", "| Dataset / contrast | Accuracy delta [95% CI] | Macro F1 delta [95% CI] | Original manifest hashes |", "|---|---|---|---|"]
    for item in index:
        if item["status"] != "complete":
            continue
        values = []
        for metric in ("accuracy", "macro_f1"):
            m = item["metrics"][metric]
            values.append(f"{m['estimate']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]")
        identity = "equal" if item["manifest_hashes_are_equal"] else "different; content audited"
        lines.append(f"| [{item['contrast']}]({item['file']}) | {values[0]} | {values[1]} | {identity} |")
    waiting = [item for item in index if item["status"] != "complete"]
    if waiting:
        lines += ["", "## Pending or blocked", ""]
        lines += [f"- **{item['status']}** `{item['contrast']}`: {item['reason']}" for item in waiting]
    lines += ["", "Rebuild with `PYTHONPATH=src .venv/bin/python scripts/compare_combined_pilot.py`. Missing or incomplete runs are listed, never replaced with estimates. Existing comparison artifacts outside this index are not evidence of completion in the current rebuild.", ""]
    (args.output / "index.md").write_text("\n".join(lines))
    return int(any(item["status"] == "blocked" for item in index))


if __name__ == "__main__":
    sys.exit(main())
