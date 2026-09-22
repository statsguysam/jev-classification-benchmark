#!/usr/bin/env python3
"""Build a public, aggregate-only dashboard asset from measured execution artifacts.

Schema v1: ``runs`` retains every complete execution. Its unique ``id`` and
``source_path`` are repository-relative run.json paths (raw run_id is not unique).
Scores and CI bounds use [0, 1], never percentages. Missing metrics/intervals are
null, never zero. ``train_per_class`` is 0 for zero-shot, an integer for matched
arms, and null for full-training classical runs; ``label_budget`` disambiguates
that null. ``canonical`` selects the first equivalent execution by the declared
source priority, independently of scores. All repeated executions remain visible.
``costs`` contains decimal USD strings, at study/stage/provider scope only.

Run from the repository root: python scripts/build_dashboard_data.py
No model execution, network access, or scientific artifact changes are performed.
Only explicit aggregate fields are exported, not arbitrary config/metadata maps.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import quote

REPO_URL = "https://github.com/statsguysam/jev-classification-benchmark"
TABULAR = {"titanic", "breast_cancer", "wine"}
DATASET_NAMES = {
    "sst2": "SST-2", "imdb": "IMDb", "ag_news": "AG News", "trec": "TREC",
    "banking77": "Banking77", "titanic": "Titanic",
    "breast_cancer": "Breast Cancer Wisconsin (Diagnostic)", "wine": "Wine",
}
MODEL_NAMES = {
    "typesafe/jev-1.13": "Jev 1.13", "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-6-astra": "GPT-6 Astra", "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3 4B", "majority": "Majority",
    "logistic_regression": "Logistic regression", "linear_svc": "Linear SVM",
    "multinomial_nb": "Multinomial Naive Bayes", "rbf_svc": "RBF SVM",
    "random_forest": "Random forest", "hist_gradient_boosting": "Histogram gradient boosting",
}
PROBABILITY_METRICS = (
    "log_loss", "brier_sum", "ece_15_equal_width", "roc_auc", "average_precision",
    "zero_true_class_probability_rows", "chosen_label_argmax_disagreement_rows",
    "probability_renormalized_rows",
)
METADATA_KEYS = ("probability_kind", "resolved_model", "resolved_revision", "device", "dtype", "scoring")
SOURCE_PRIORITY = {"pilot": 0, "matched": 1, "colab": 2}


def read_json(path: Path):
    return json.loads(path.read_text())


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_url(path: str) -> str:
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("Source links must remain inside the repository")
    return REPO_URL + "/blob/main/" + quote(path, safe="/")


def data_type(dataset: str) -> str:
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {dataset}")
    return "mixed" if dataset == "titanic" else "numeric" if dataset in TABULAR else "text"


def finite(value):
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("Metric must be a finite number or null")
    return value


def confidence(metrics: dict, audited: dict | None = None) -> dict:
    bootstrap = audited["group_bootstrap"] if audited else metrics.get("bootstrap")
    out = {"ci": {key: None for key in ("accuracy", "macro_f1", "balanced_accuracy")},
           "ci_method": None, "ci_samples": None, "ci_n_groups": None,
           "ci_source_path": "results/TABULAR_COMPARISON.json" if audited else None,
           "ci_conditional_on": "fixed split, fitted model and prompt; not training-seed or pretraining uncertainty"}
    if not bootstrap:
        return out
    for key in out["ci"]:
        item = bootstrap.get("metrics", {}).get(key)
        if item is None:
            continue
        bounds = item.get("ci95")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("Invalid confidence interval")
        low, high = map(finite, bounds)
        if low is None or high is None or not 0 <= low <= high <= 1:
            raise ValueError("Invalid confidence interval bounds")
        if not math.isclose(item["estimate"], metrics[key], abs_tol=1e-12):
            raise ValueError("Confidence interval point estimate does not match execution")
        out["ci"][key] = [low, high]
    out.update(ci_method=bootstrap["method"], ci_samples=bootstrap.get("samples"),
               ci_n_groups=bootstrap.get("n_groups"),
               ci_conditional_on=bootstrap.get("conditional_on", out["ci_conditional_on"]))
    return out


def audit_prediction_aggregates(path: Path, record: dict, test: dict) -> dict:
    """Check row coverage/counts while retaining only safe aggregate metadata."""
    metrics = record["metrics"]
    classes = len(record["labels"])
    expected = {r["id"] for r in test["rows"]}
    if len(expected) != len(test["rows"]):
        raise ValueError("Duplicate test manifest IDs")
    seen = set()
    failures = probability_rows = 0
    metadata = {key: set() for key in METADATA_KEYS}
    confusion_matrix = [[0] * (classes + 1) for _ in range(classes)]
    targets = {r["id"]: r["label"] for r in test["rows"]}
    for line in path.read_text().splitlines():
        p = json.loads(line)
        rid = p["row_id"]
        if rid in seen or rid not in expected:
            raise ValueError("Prediction IDs do not match test manifest")
        seen.add(rid)
        valid = p.get("error") is None and isinstance(p.get("label"), int) and 0 <= p["label"] < classes
        failures += not valid
        probability_rows += valid and p.get("probabilities") is not None
        confusion_matrix[targets[rid]][p["label"] if valid else classes] += 1
        for key in METADATA_KEYS:
            value = p.get("metadata", {}).get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("Expected string aggregate prediction metadata")
                metadata[key].add(value)
    if seen != expected or len(seen) != metrics["n_test"]:
        raise ValueError("Prediction coverage or test count mismatch")
    if failures != metrics["n_failures"] or probability_rows != metrics["n_probability_rows"]:
        raise ValueError("Failure or probability count mismatch")
    if not math.isclose(probability_rows / len(seen), metrics["probability_coverage"], abs_tol=1e-12):
        raise ValueError("Probability coverage mismatch")
    if confusion_matrix != metrics["confusion_matrix"]:
        raise ValueError("Confusion matrix mismatch")
    recalls, f1s = [], []
    for i, counts in enumerate(confusion_matrix):
        tp, support = counts[i], sum(counts)
        predicted = sum(r[i] for r in confusion_matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        if support:
            recalls.append(recall)
        original = metrics["per_class"][i]
        for key, expected_value in {"precision": precision, "recall": recall, "f1": f1, "support": support}.items():
            if not math.isclose(original[key], expected_value, abs_tol=1e-12):
                raise ValueError("Per-class metric mismatch")
    expected_metrics = {"accuracy": sum(r[i] for i, r in enumerate(confusion_matrix)) / len(seen),
                        "macro_f1": sum(f1s) / classes, "balanced_accuracy": sum(recalls) / len(recalls),
                        "failure_rate": failures / len(seen)}
    for key, expected_value in expected_metrics.items():
        if not math.isclose(metrics[key], expected_value, abs_tol=1e-12):
            raise ValueError("Classification metric mismatch")
    return {key: sorted(values) for key, values in metadata.items()}


def equivalence_fingerprint(record: dict, row: dict, test: dict) -> str:
    """Scientific input/configuration equivalence; hardware repetitions are retained.

    Durations, run IDs, scores, absolute adapter paths and environment versions
    are deliberately excluded. Declared pipeline/parameters, selected ordered
    labels, data content, prompt, precision and adapter recipe are included.
    """
    config = record["config"]
    training = record.get("training", {})
    adapter = record.get("adapter_training", {})
    selected_ids = adapter.get("training_row_ids", record.get("training_example_ids", []))
    relevant_config = {key: config.get(key) for key in (
        "revision", "shots_per_class", "max_context_tokens", "reasoning_effort",
        "max_output_tokens", "temperature", "feature_schema_sha256", "native_records_sha256",
        "selection", "tabular_classical_sha256", "tabular_data_sha256",
    )}
    return digest({
        "dataset": row["dataset"], "model": row["model"], "method": row["method"],
        "seed": row["seed"], "train_per_class": row["train_per_class"],
        "input_representation": row["input_representation"], "train_labels": row["train_labels"],
        "dev_labels": row["dev_labels"], "ordered_training_ids": selected_ids,
        "ordered_test_rows": test["rows"], "labels": record["labels"],
        "prepared_content_sha256": record["dataset_manifest"].get("prepared_content_sha256"),
        "text_policy": record["dataset_manifest"].get("text_policy"),
        "prompt_template_sha256": record.get("prompt_template_sha256"),
        "implementation_sha256": record.get("implementation_sha256"),
        "config": relevant_config,
        "selected_parameters": training.get("selected_parameters"),
        "features": training.get("features"),
        "selection_metric": training.get("selection_metric"),
        "refit_with_validation": training.get("refit_with_validation"),
        "inference_dtypes": row["hardware"]["inference_dtypes"],
        "adapter_recipe": {key: adapter.get(key) for key in (
            "dtype", "epochs", "gradient_accumulation_steps", "learning_rate", "load_in_4bit",
            "lora_alpha", "max_length", "max_steps", "r", "seed", "selection", "target_modules",
            "batch_size", "prompt_sha256", "supervision", "optimizer_steps", "skipped_optimizer_steps",
        )},
    })


def normalize_run(record: dict, source_path: str, metadata: dict, audited: dict | None = None) -> dict:
    metrics, config = record["metrics"], record["config"]
    dataset, model = record["dataset"], config["model"]
    classical = record["method"].startswith("classical")
    adapter, training = record.get("adapter_training", {}), record.get("training", {})
    family = "classical" if classical else "jev" if model.startswith("typesafe/jev") else "open" if config.get("provider") == "hf" else "hosted"
    method = "classical" if classical else "qlora" if adapter.get("load_in_4bit") else record["method"]
    train_per_class = config.get("train_per_class") if classical else adapter.get("train_per_class") if method in {"lora", "qlora"} else config.get("shots_per_class", 0)
    selected_ids = adapter.get("training_row_ids", record.get("training_example_ids", []))
    train_labels = training.get("training_rows", adapter.get("training_rows", len(selected_ids)))
    dev_labels = training.get("validation_rows", adapter.get("validation_rows", 0))
    if len(selected_ids) != train_labels:
        raise ValueError("Training label count does not match selected IDs")
    representation = "native_columns" if record["method"] == "classical_tabular" else "tfidf_text" if classical else "serialized_columns" if dataset in TABULAR else "text"
    if model == "majority":
        representation = "native_columns" if dataset in TABULAR else "text_labels_only"
    if classical:
        latency_basis = "amortized batch inference including feature preprocessing; not individual request latency"
        latency_scope = "same hardware, preprocessing, and batch-size protocol only"
    elif family == "open":
        latency_basis = "sequential per-row normalized class-sequence likelihood scoring"
        latency_scope = "same hardware, precision, context and candidate classes only"
    else:
        latency_basis = "end-to-end API request observed by client; includes network/provider delay"
        latency_scope = "same client/request protocol; backend hardware is undisclosed"
    hardware = {
        "client_platform": record.get("environment", {}).get("platform"),
        "inference_devices": metadata.get("device", []) or (["cpu"] if classical else []),
        "inference_dtypes": metadata.get("dtype", []),
        "training_device": adapter.get("device", "cpu" if classical else None),
        "training_dtype": adapter.get("dtype"),
        "provider_hardware": None,
        "basis": "recorded prediction metadata and execution environment; hosted client is not provider hardware",
    }
    row = {
        "id": source_path, "source_path": source_path, "source_url": source_url(source_path),
        "run_id": record["run_id"], "status": record["status"],
        "dataset": dataset, "data_type": data_type(dataset), "study": "tabular" if dataset in TABULAR else "text",
        "task": "binary" if len(record["labels"]) == 2 else "multiclass",
        "n_classes": len(record["labels"]), "labels": record["labels"],
        "model": model, "display_model": MODEL_NAMES.get(model, model), "family": family, "method": method,
        "input_representation": representation, "train_per_class": train_per_class,
        "label_budget": "full" if classical and train_per_class is None else "zero" if train_per_class == 0 else f"k{train_per_class}",
        "train_labels": train_labels, "dev_labels": dev_labels, "seed": record["seed"],
        "accuracy": finite(metrics.get("accuracy")), "macro_f1": finite(metrics.get("macro_f1")),
        "balanced_accuracy": finite(metrics.get("balanced_accuracy")),
        "n_test": metrics["n_test"], "n_failures": metrics["n_failures"],
        "failure_rate": finite(metrics.get("failure_rate")),
        "valid_response_accuracy": finite(metrics.get("valid_response_accuracy")),
        "probability_coverage": finite(metrics.get("probability_coverage")),
        "n_probability_rows": metrics.get("n_probability_rows"),
        "probability_kinds": metadata.get("probability_kind", []) or ([training["probability_kind"]] if training.get("probability_kind") else []),
        "probability_metrics": {key: finite(metrics.get(key)) for key in PROBABILITY_METRICS},
        "probability_metrics_basis": "only rows with a valid response and class probability distribution; ECE uses argmax confidence",
        "reliability_bins": [{key: finite(b.get(key)) for key in ("bin", "count", "mean_confidence", "accuracy")} for b in metrics.get("reliability_bins", [])],
        "per_class": [{**{key: finite(p[key]) for key in ("label", "precision", "recall", "f1", "support")}, "name": record["labels"][p["label"]]} for p in metrics.get("per_class", [])],
        "confusion_matrix": metrics.get("confusion_matrix"),
        "confusion_matrix_columns": metrics.get("confusion_matrix_columns"),
        "latency": {"p50_s": finite(metrics.get("latency_p50_s")), "p95_s": finite(metrics.get("latency_p95_s")),
                    "prediction_time_s": finite(metrics.get("prediction_time_s")),
                    "basis": latency_basis, "comparable_scope": latency_scope},
        "hardware": hardware, "training_quantized_4bit": adapter.get("load_in_4bit"),
        "training_time_s": finite(adapter.get("training_s", training.get("fit_and_selection_s"))),
        "training_time_basis": "adapter training loop" if adapter else "classical fitting and selection" if classical else None,
        "resolved_models": metadata.get("resolved_model", []), "requested_revision": config.get("revision"),
        "resolved_revisions": metadata.get("resolved_revision", []), "scoring_rules": metadata.get("scoring", []),
        "input_tokens": metrics.get("input_tokens"), "output_tokens": metrics.get("output_tokens"),
        "input_tokens_coverage": metrics.get("input_tokens_coverage"), "output_tokens_coverage": metrics.get("output_tokens_coverage"),
        "input_tokens_known_total": metrics.get("input_tokens_known_total"), "output_tokens_known_total": metrics.get("output_tokens_known_total"),
        "training_selection_sha256": digest(selected_ids),
        "prepared_content_sha256": record["dataset_manifest"].get("prepared_content_sha256"),
        "test_ids_sha256": record.get("test_ids_sha256"),
        "metric_source": "measured run.json; tabular confidence intervals from audited comparison" if audited else "measured run.json",
    }
    row.update(confidence(metrics, audited))
    if row["ci_source_path"] is None and row["ci_method"] is not None:
        row["ci_source_path"] = source_path
    return row


def mark_repetitions(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[row["duplicate_group_id"]].append(row)
    for members in groups.values():
        chosen = min(members, key=lambda r: (SOURCE_PRIORITY.get(Path(r["id"]).parts[1], 3), r["id"]))
        for row in members:
            row.update(canonical=row is chosen, canonical_id=chosen["id"], repetition_count=len(members),
                       duplicate_of=None if row is chosen else chosen["id"])


def cost_summary(root: Path) -> dict:
    text_path, tab_path = "results/JEV_COSTS.json", "results/tabular/API_COST_SUMMARY.json"
    text, tab = read_json(root / text_path), read_json(root / tab_path)
    if Decimal(text["combined_conservative_charged_or_reserved_usd"]) != Decimal(tab["prior_text_study_accounted_usd"]):
        raise ValueError("Text/tabular cumulative cost anchor differs")
    if Decimal(tab["prior_text_study_accounted_usd"]) + Decimal(tab["tabular"]["accounted_conservative_usd"]) != Decimal(tab["cumulative_accounted_conservative_usd"]):
        raise ValueError("Cumulative cost does not reconcile")
    provider_rows = [{
        "study": "tabular", "model": model,
        **{key: value.get(key) for key in (
            "accounted_conservative_usd", "reported_api_cost_usd", "reported_api_cost_known_subtotal_usd",
            "reported_api_cost_coverage", "token_rate_estimate_usd", "token_rate_estimate_known_subtotal_usd",
            "token_rate_estimate_coverage", "reservations", "retained_full_reservations",
        )},
    } for model, value in sorted(tab["by_model"].items())]
    provider_rows.extend([
        {"study": "text", "model": "OpenAI models combined", "accounted_conservative_usd": text["openai"]["conservative_settlement_usd"],
         "token_rate_estimate_usd": text["openai"]["reported_usage_standard_rate_estimate_usd"],
         "reported_api_cost_usd": None, "requests": text["openai"]["requests"]},
        {"study": "text", "model": "typesafe/jev-1.13", "accounted_conservative_usd": text["jev"]["retained_reservation_usd"],
         "reported_api_cost_usd": text["jev"]["reported_api_cost_usd"],
         "reported_api_cost_known_subtotal_usd": text["jev"]["reported_api_cost_known_subtotal_usd"],
         "reported_api_cost_coverage": float(text["jev"]["reported_cost_coverage"]), "requests": text["jev"]["requests"]},
    ])
    return {
        "currency": "USD", "amount_encoding": "decimal strings", "basis": tab["basis"],
        "authorized_usd": tab["cumulative_authorized_cap_usd"],
        "conservative_accounted_usd": tab["cumulative_accounted_conservative_usd"],
        "remaining_capacity_usd": tab["remaining_cumulative_capacity_usd"],
        "known_mixed_cost_subtotal_usd": str(Decimal(text["combined_openai_estimate_plus_jev_reported_known_subtotal_usd"]) + Decimal(tab["tabular"]["token_rate_estimate_known_subtotal_usd"])),
        "known_mixed_cost_subtotal_basis": "OpenAI standard-rate token estimates plus Jev reported API dollars for known requests; incomplete coverage, not invoice or total billed spend",
        "compute_cost_usd": None,
        "stages": [{"study": "text", "accounted_conservative_usd": text["combined_conservative_charged_or_reserved_usd"]},
                   {"study": "tabular", "accounted_conservative_usd": tab["tabular"]["accounted_conservative_usd"]}],
        "providers": provider_rows,
        "sources": [{"path": p, "url": source_url(p), "sha256": file_digest(root / p)} for p in (text_path, tab_path)],
    }


def assert_public(value) -> None:
    """Fail closed if selected source strings contain credentials or machine paths."""
    serialized = json.dumps(value, allow_nan=False)
    for pattern in (r"/Users/", r"/home/", r"/content/", r"/private/", r"(?<![A-Za-z0-9])sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{16,}"):
        if re.search(pattern, serialized):
            raise ValueError("Dashboard asset contains a forbidden secret/path pattern")


def build(root: Path) -> dict:
    root = root.resolve()
    audit_path = root / "results/TABULAR_COMPARISON.json"
    audited = read_json(audit_path)
    audit_by_path = {row["source_path"]: row for row in audited["runs"] if row["status"] == "complete"}
    rows, datasets = [], {}
    for path in sorted((root / "results").rglob("run.json")):
        record = read_json(path)
        if record.get("status") != "complete":
            raise ValueError("Unmeasured execution encountered; do not present as a measured result")
        relative = path.relative_to(root).as_posix()
        audit = audit_by_path.get(relative)
        if record["dataset"] in TABULAR:
            if audit is None:
                raise ValueError("Tabular run missing from audited comparison")
            for filename in ("run.json", "test_manifest.json", "predictions.jsonl"):
                if file_digest(path.parent / filename) != audit["artifact_sha256"][filename]:
                    raise ValueError("Audited tabular artifact hash mismatch")
        test = read_json(path.parent / "test_manifest.json")
        metadata = audit_prediction_aggregates(path.parent / "predictions.jsonl", record, test)
        row = normalize_run(record, relative, metadata, audit)
        if "cuda" in row["hardware"]["inference_devices"]:
            environment_path = "results/tabular/colab/environment.json" if row["study"] == "tabular" else "results/colab_environment_after_fix.json"
            recorded_environment = read_json(root / environment_path)
            if recorded_environment["platform"] != row["hardware"]["client_platform"]:
                raise ValueError("Colab hardware environment does not match execution")
            row["hardware"].update(accelerator=recorded_environment.get("gpu"),
                                   accelerator_memory_bytes=recorded_environment.get("gpu_memory_bytes"),
                                   source_path=environment_path)
        else:
            row["hardware"].update(accelerator=None, accelerator_memory_bytes=None, source_path=relative)
        row["source_sha256"] = file_digest(path)
        row["duplicate_group_id"] = equivalence_fingerprint(record, row, test)
        rows.append(row)
        if row["dataset"] not in datasets:
            manifest = record["dataset_manifest"]
            datasets[row["dataset"]] = {
                "id": row["dataset"], "name": DATASET_NAMES[row["dataset"]],
                "data_type": row["data_type"], "task": row["task"], "n_classes": row["n_classes"],
                "labels": row["labels"], "n_test": row["n_test"],
                "splits": {key: {"rows": value["rows"], "class_counts": value["class_counts"]} for key, value in manifest["splits"].items()},
                "source_url": manifest.get("source", {}).get("homepage"),
                "test_feature_groups": audit["group_bootstrap"]["n_groups"] if audit else None,
                "ci_method": row["ci_method"],
            }
        elif datasets[row["dataset"]]["labels"] != row["labels"] or datasets[row["dataset"]]["n_test"] != row["n_test"]:
            raise ValueError("Dataset label map or heldout size differs across runs")
    if {row["source_path"] for row in rows if row["study"] == "tabular"} != set(audit_by_path):
        raise ValueError("Audited tabular inventory does not match execution inventory")
    mark_repetitions(rows)
    canonical = [row for row in rows if row["canonical"]]
    result = {
        "schema_version": 1, "repository_url": REPO_URL,
        "summary": {
            "execution_records": len(rows), "canonical_configurations": len(canonical),
            "repeated_execution_records": len(rows) - len(canonical),
            "prediction_records": sum(row["n_test"] for row in rows),
            "canonical_prediction_records": sum(row["n_test"] for row in canonical),
            "failures": sum(row["n_failures"] for row in rows),
            "canonical_failures": sum(row["n_failures"] for row in canonical),
            "datasets": len(datasets), "models": len({row["model"] for row in rows}),
            "by_study": dict(sorted(Counter(row["study"] for row in rows).items())),
            "by_family": dict(sorted(Counter(row["family"] for row in rows).items())),
        },
        "datasets": list(datasets.values()), "runs": rows, "costs": cost_summary(root),
        "provenance": {"builder_path": "scripts/build_dashboard_data.py", "builder_sha256": file_digest(Path(__file__)),
                       "tabular_audit_path": "results/TABULAR_COMPARISON.json", "tabular_audit_sha256": file_digest(audit_path)},
        "notes": [
            "Measured executions only. Missing model/dataset/method combinations were not run and must not be scored as zero.",
            "Canonical rows group equivalent dataset, model, method, seed, label budget, ordered training/test inputs and declared recipes. Choose pilot before matched before Colab; selection never uses scores. Repeated hardware/environment executions remain available with separate latency and provenance.",
            "Do not average ranks across datasets or unequal label budgets. Full-training classical runs may also use validation labels; inspect train_labels and dev_labels.",
            "All test failures remain in classification denominators and in the final confusion-matrix column. Probability metrics use their explicitly recorded subset; missing probabilities are not reconstructed.",
            "Text CIs are 1,000-replicate stratified row percentile bootstrap; tabular CIs use 2,000-replicate unstratified whole-feature-group bootstrap from the audited summary. No balanced-accuracy CI was estimated.",
            "Intervals condition on one split, prompt/serialization, training selection and fitted model; they do not cover training randomness, prompt search or pretraining contamination.",
            "Tabular LLMs receive fixed name=value serialization; native ML receives original feature columns. Public benchmark memorization remains possible. Wine cultivar IDs are arbitrary; zero-shot performance does not establish semantic understanding.",
            "This is a small public benchmark, not a clinical validation. Latency protocols/hardware differ; API costs are aggregate accounting, not provider invoices or per-run price comparisons.",
        ],
    }
    assert_public(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("artifacts/dashboard-data.json"))
    args = parser.parse_args()
    result = build(args.root)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
