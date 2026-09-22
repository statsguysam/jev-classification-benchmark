#!/usr/bin/env python3
"""Audit the 68-condition numeric expansion without making model/API calls.

Keep original measurements immutable. Missing/incomplete conditions remain visible
with null scores. Every scored condition is reconstructed from raw predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import digest
from jevbench.types import Prediction
from summarize_numeric_decisions import chosen, transitions
from summarize_tabular import audit_run as audit_historical_run, group_bootstrap
from tabular_data import load_native_prepared

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SHA = "7e3df4dc6a04aa0a88068e9639e3d04d95f86bcae38d3086ad12d23b4f11a7e2"
NUMERIC_SHA = "881dfb668ea8fca161de42fcba8ca8f0635a0d314507afc542567be11c471e76"
DATASETS = ("breast_cancer", "wine")
NAMES = {"breast_cancer": "Breast Cancer", "wine": "Wine"}
MODEL_NAMES = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3 4B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2 1.7B",
    "ibm-granite/granite-3.3-2b-instruct": "Granite 3.3 2B",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-6-astra": "GPT-6 Astra",
}
NEW_MODELS = {"HuggingFaceTB/SmolLM2-1.7B-Instruct", "ibm-granite/granite-3.3-2b-instruct"}
NEW_MODEL_KEYS = {"HuggingFaceTB/SmolLM2-1.7B-Instruct": "smollm2", "ibm-granite/granite-3.3-2b-instruct": "granite"}
JEV = "typesafe/jev-1.13"
CLASSICAL_NAMES = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
                   "xgboost": "XGBoost", "lightgbm": "LightGBM"}
REUSED_REVIEW_MODELS = {"Qwen/Qwen3-4B-Instruct-2507", "gpt-6-astra"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_conditions():
    result = []
    for dataset in DATASETS:
        result.extend((dataset, model, arm, shots) for model in MODEL_NAMES for shots in (0, 4)
                      for arm in ("base", "review"))
        result.extend((dataset, JEV, "direct", shots) for shots in (0, 4))
        result.extend((dataset, model, "classical", budget) for model in CLASSICAL_NAMES for budget in (4, None))
    return result


def row_key(row):
    if "source_run_id" in row or row["model"].endswith("+jev_review"):
        return row["dataset"], row["model"].removesuffix("+jev_review"), "review", row["train_per_class"]
    arm = "direct" if row["model"] == JEV else "classical" if row["model"] in CLASSICAL_NAMES else "base"
    return row["dataset"], row["model"], arm, row["train_per_class"]


def placeholder(key):
    dataset, model, arm, budget = key
    display = "Jev alone" if model == JEV else CLASSICAL_NAMES.get(model, MODEL_NAMES.get(model))
    if arm == "review":
        display += " → Jev"
    return {"dataset": dataset, "model": model+"+jev_review" if arm == "review" else model,
        "display_model": display, "arm": arm, "method": "cached_label_jev_review" if arm == "review" else
        "classical_reference" if arm == "classical" else "zero_shot" if budget == 0 else "few_shot",
        "train_per_class": budget, "train_labels": None, "seed": 42,
        "run_id": "pending::"+"::".join(map(str, key)), "source_path": None,
        "status": "pending: no complete audited artifact", "accuracy": None, "macro_f1": None,
        "n_failures": None, "n_test": None, "group_bootstrap": None,
        "probability_coverage": None, "probability_kind": None}


def audit_predictions(path, dataset, *, model, budget, display, root=ROOT, samples=2000):
    """Common metric/row-budget checks after a family-specific provenance audit."""
    path, root = Path(path), Path(root)
    record = read(path)
    require(record.get("status") == "complete", "Incomplete artifacts must remain unscored")
    selected = dataset.train if budget is None else select_examples(dataset.train, dataset.labels, budget, 42)
    ids = [row.id for row in selected]
    require(record["dataset"] == dataset.name and record["labels"] == dataset.labels and record["seed"] == 42,
            "Dataset/labels/seed differ")
    require(record["training_example_ids"] == ids, "Training/example IDs differ from declared label budget")
    require(record["manifest_sha256"] == digest(dataset.manifest) and record["dataset_manifest"] == dataset.manifest,
            "Prepared dataset manifest differs")
    expected_test = {"dataset": dataset.name, "labels": dataset.labels, "manifest_sha256": digest(dataset.manifest),
        "rows": [{"id": row.id, "label": row.label, "text_sha256": hashlib.sha256(row.text.encode()).hexdigest()}
                 for row in dataset.test]}
    require(read(path.with_name("test_manifest.json")) == expected_test, "Ordered held-out records differ")
    predictions = [Prediction(**json.loads(line)) for line in path.with_name("predictions.jsonl").read_text().splitlines() if line]
    require([p.row_id for p in predictions] == [r.id for r in dataset.test], "Predictions do not cover the ordered test fold")
    metrics = evaluate(dataset.test, predictions, len(dataset.labels))
    for name, value in metrics.items():
        saved = record["metrics"].get(name)
        equal = math.isclose(value, saved, rel_tol=1e-10, abs_tol=1e-12) if isinstance(value, float) and isinstance(saved, (float, int)) else value == saved
        require(equal, f"Saved metric disagrees with raw predictions: {name}")
    values = chosen(predictions, len(dataset.labels))
    payload = {"y": [row.label for row in dataset.test], "predictions": values,
        "groups": [row["text_sha256"] for row in expected_test["rows"]], "n_classes": len(dataset.labels),
        "training_example_ids": ids, "test_row_ids": [row.id for row in dataset.test],
        "manifest_sha256": digest(dataset.manifest)}
    row = {"dataset": dataset.name, "model": model, "display_model": display,
        "method": record["method"], "train_per_class": budget, "train_labels": len(ids), "seed": 42,
        "status": "complete", "run_id": record["run_id"], "source_path": path.relative_to(root).as_posix(),
        "group_bootstrap": group_bootstrap(payload["y"], values, payload["groups"], n_classes=len(dataset.labels), samples=samples),
        "artifact_sha256": {name: file_sha(path.with_name(name)) for name in ("run.json", "predictions.jsonl", "test_manifest.json")},
        "probability_kinds": sorted({p.metadata.get("probability_kind", "unreported") for p in predictions}),
        "resolved_models": sorted({p.metadata["resolved_model"] for p in predictions if p.metadata.get("resolved_model")}),
        "resolved_revisions": sorted({p.metadata["resolved_revision"] for p in predictions if p.metadata.get("resolved_revision")}),
        "inference_dtypes": sorted({p.metadata["dtype"] for p in predictions if p.metadata.get("dtype")}),
        "devices": sorted({p.metadata["device"] for p in predictions if p.metadata.get("device")}),
        "config": record["config"], **metrics}
    return row, payload


def verify_historical_artifacts(row, root):
    path = Path(root) / row["source_path"]
    require(set(row["artifact_sha256"]) == {"run.json", "predictions.jsonl", "test_manifest.json"}, "Historical hash inventory differs")
    require(all(file_sha(path.with_name(name)) == expected for name, expected in row["artifact_sha256"].items()),
            "Historical artifact changed after its pinned report")
    return path


def comparison_specs():
    result = []
    for dataset in DATASETS:
        for model in MODEL_NAMES:
            for shots in (0, 4):
                review = dataset, model, "review", shots
                result.append(("review_minus_source", review, (dataset, model, "base", shots), True))
                result.append(("review_minus_jev_alone", review, (dataset, JEV, "direct", shots), True))
            for arm in ("base", "review"):
                result.append(("few_minus_zero_descriptive", (dataset, model, arm, 4), (dataset, model, arm, 0), False))
    return result


def compare(rows, payloads, *, samples=2000):
    result = []
    for kind, ka, kb, equal_budget in comparison_specs():
        a, b = rows[ka], rows[kb]
        item = {"kind": kind, "dataset": ka[0], "a": a["run_id"], "b": b["run_id"],
            "a_display": a["display_model"], "b_display": b["display_model"],
            "train_per_class_a": ka[3], "train_per_class_b": kb[3],
            "status": "pending", "paired_bootstrap": None,
            "interpretation": "Exploratory, unadjusted, A minus B; fixed split/models/serialization",
            "equal_new_label_budget": equal_budget}
        if a["status"] != "complete" or b["status"] != "complete":
            result.append(item)
            continue
        pa, pb = payloads[ka], payloads[kb]
        require(all(pa[field] == pb[field] for field in ("manifest_sha256", "y", "groups", "n_classes", "test_row_ids")), "Paired conditions use unequal held-out records")
        if equal_budget:
            require(pa["training_example_ids"] == pb["training_example_ids"], "Matched comparison uses unequal training IDs")
        item.update(status="complete", paired_bootstrap=group_bootstrap(pa["y"], pa["predictions"], pa["groups"], pb["predictions"], n_classes=pa["n_classes"], samples=samples))
        if kind == "review_minus_source":
            require(a["source_run_id"] == b["run_id"], "Review is linked to a different source run")
            counts = transitions(pa["y"], pb["predictions"], pa["predictions"])
            require(math.isclose(counts["accuracy_delta_pp"]/100, item["paired_bootstrap"]["metrics"]["accuracy"]["estimate"], abs_tol=1e-12), "Transition counts disagree with accuracy delta")
            item["transitions"] = counts
            a["review_transitions"] = counts
        result.append(item)
    return result


def assemble(audited, payloads, *, samples=2000):
    rows = {key: placeholder(key) for key in expected_conditions()}
    for key, row in audited.items():
        require(key in rows and row_key(row) == key, "Unexpected expanded condition")
        rows[key] = row
        rows[key]["arm"] = key[2]
    for key, row in rows.items():
        if key[2] == "review":
            source = rows[key[0], key[1], "base", key[3]]
            if row["status"] == "complete":
                require(source["status"] == "complete" and row.get("source_run_id") == source["run_id"], "Complete review has an unavailable/different source")
            else:
                row["source_run_id"] = source["run_id"]
    comparisons = compare(rows, payloads, samples=samples)
    counts = {arm: sum(row["status"] == "complete" for key, row in rows.items() if key[2] == arm)
              for arm in ("base", "review", "direct", "classical")}
    return {"schema_version": 1, "scope": "exploratory post-hoc numeric-only expansion",
        "datasets": list(DATASETS), "models": MODEL_NAMES, "expected_runs": 68,
        "complete_runs": sum(counts.values()), "status": "complete" if sum(counts.values()) == 68 else "in_progress_or_incomplete",
        "expected_base_runs": 24, "complete_base_runs": counts["base"],
        "expected_review_runs": 24, "complete_review_runs": counts["review"],
        "expected_direct_jev_runs": 4, "complete_direct_jev_runs": counts["direct"],
        "expected_classical_runs": 16, "complete_classical_runs": counts["classical"],
        "expected_new_local_runs": 8, "complete_new_local_runs": sum(row["status"] == "complete" and key[1] in NEW_MODELS and key[2] == "base" for key, row in rows.items()),
        "expected_new_review_runs": 20, "complete_new_review_runs": sum(row["status"] == "complete" and key[2] == "review" and not (key[1] in REUSED_REVIEW_MODELS and key[3] == 4) for key, row in rows.items()),
        "expected_reused_review_runs": 4, "complete_reused_review_runs": sum(row["status"] == "complete" and key[2] == "review" and key[1] in REUSED_REVIEW_MODELS and key[3] == 4 for key, row in rows.items()),
        "bootstrap_samples": samples, "seed": 42, "runs": list(rows.values()), "comparisons": comparisons,
        "expected_comparisons": 72, "complete_comparisons": sum(item["status"] == "complete" for item in comparisons)}


def collect(root=ROOT, *, samples=2000):
    root = Path(root).resolve()
    require(samples >= 100, "Use at least 100 bootstrap replicates")
    historical_path, numeric_path = root/"results/TABULAR_COMPARISON.json", root/"results/numeric_decisions/COMPARISON.json"
    require(file_sha(historical_path) == HISTORICAL_SHA, "Original tabular report differs from the pinned source")
    require(file_sha(numeric_path) == NUMERIC_SHA, "Original numeric supplement differs from the pinned source")
    historical, numeric = read(historical_path), read(numeric_path)
    datasets = {name: load_native_prepared(root/"data/tabular-full"/name) for name in DATASETS}
    audited, payloads = {}, {}
    def add(row, payload):
        key = row_key(row)
        require(key in expected_conditions() and key not in audited, "Duplicate or unexpected expanded condition")
        audited[key], payloads[key] = row, payload
    for previous in historical["runs"]:
        model, method = previous["model"], previous["method"]
        if previous["dataset"] not in DATASETS or method == "lora":
            continue
        if model not in {*MODEL_NAMES, JEV, "logistic_regression", "random_forest"}:
            continue
        path = verify_historical_artifacts(previous, root)
        dataset, native = datasets[previous["dataset"]]
        audit_historical_run(path, dataset, native, root, samples=samples)
        display = "Jev alone" if model == JEV else CLASSICAL_NAMES.get(model, MODEL_NAMES.get(model))
        row, payload = audit_predictions(path, dataset, model=model, budget=previous["train_per_class"], display=display, root=root, samples=samples)
        row["measurement_origin"] = "reused_original_tabular"
        row["source_prompt_family"] = "frozen original provider/model rendering"
        add(row, payload)
    for previous in numeric["runs"]:
        model = previous["model"]
        if model not in {"xgboost", "lightgbm"} and not model.endswith("+jev_review"):
            continue
        path = verify_historical_artifacts(previous, root)
        dataset, native = datasets[previous["dataset"]]
        if model.endswith("+jev_review"):
            from summarize_numeric_decisions import audit_review_provenance
            source_model = model.removesuffix("+jev_review")
            require(source_model in REUSED_REVIEW_MODELS and previous["train_per_class"] == 4, "Unknown reused review condition")
            audit_review_provenance(path, dataset)
        else:
            from audit_numeric_boosting import audit_boosting_run
            audit_boosting_run(path, dataset, native, root=root)
        row, payload = audit_predictions(path, dataset, model=model, budget=previous["train_per_class"], display=previous["display_model"], root=root, samples=samples)
        row["measurement_origin"] = "reused_original_numeric_supplement"
        if model.endswith("+jev_review"):
            row["source_run_id"] = read(path)["config"]["source_run_id"]
        add(row, payload)
    for path in sorted((root/"results/numeric_expansion/local").glob("*/run.json")):
        record = read(path)
        model, shots = record["config"]["model"], record["config"].get("shots_per_class")
        require(model in NEW_MODELS and record["dataset"] in DATASETS and shots in (0, 4), "Unexpected new local condition")
        key = record["dataset"], model, "base", shots
        require(key not in audited, "Duplicate new local condition")
        if record.get("status") != "complete":
            row = placeholder(key)
            row.update(status=record.get("status", "incomplete"), run_id=record["run_id"], source_path=path.relative_to(root).as_posix())
            audited[key] = row
            continue
        from run_expanded_numeric_local import audit_source_run
        dataset, native = datasets[record["dataset"]]
        audit_source_run(path, dataset, NEW_MODEL_KEYS[model], shots)
        row, payload = audit_predictions(path, dataset, model=model, budget=shots, display=MODEL_NAMES[model], root=root, samples=samples)
        row.update(measurement_origin="new_local_expansion", source_prompt_family="fixed-system-no-thinking-v1")
        add(row, payload)
    for path in sorted((root/"results/numeric_expansion/review").glob("*/run.json")):
        record = read(path)
        source_model, shots = record["config"]["source_model"], record["config"]["shots_per_class"]
        require(source_model in MODEL_NAMES and record["dataset"] in DATASETS and shots in (0, 4), "Unexpected expanded review condition")
        key = record["dataset"], source_model, "review", shots
        require(key not in audited, "Duplicate review, including a reused historical condition")
        if record.get("status") != "complete":
            row = placeholder(key)
            row.update(status=record.get("status", "incomplete"), run_id=record["run_id"], source_path=path.relative_to(root).as_posix())
            audited[key] = row
            continue
        from run_expanded_numeric_review import audit_completed_review
        dataset, _ = datasets[record["dataset"]]
        audit_completed_review(path, dataset)
        row, payload = audit_predictions(path, dataset, model=source_model+"+jev_review", budget=shots,
            display=MODEL_NAMES[source_model]+" → Jev", root=root, samples=samples)
        row.update(source_run_id=record["config"]["source_run_id"], measurement_origin="new_review_expansion")
        add(row, payload)
    summary = assemble(audited, payloads, samples=samples)
    summary.update(source_sha256=file_sha(__file__), protocol_sha256=file_sha(root/"docs/EXPANDED_NUMERIC_PROTOCOL.md"),
        pinned_source_reports={"results/TABULAR_COMPARISON.json": HISTORICAL_SHA,
                               "results/numeric_decisions/COMPARISON.json": NUMERIC_SHA},
        prepared_manifests={name: digest(value[0].manifest) for name, value in datasets.items()})
    if (root/"scripts/audit_expanded_numeric_costs.py").is_file():
        from audit_expanded_numeric_costs import collect_costs
        summary["costs"] = collect_costs([row for row in summary["runs"] if row["status"] == "complete"])
    else:
        summary["costs"] = {"status": "pending: expanded cost auditor not available"}
    return summary


def pct(value):
    return "pending" if value is None else f"{100*value:.1f}%"


def interval(comparison):
    if comparison["status"] != "complete":
        return "pending"
    result = comparison["paired_bootstrap"]["metrics"]["accuracy"]
    low, high = result["ci95"]
    return f"{100*result['estimate']:+.2f} [{100*low:+.2f}, {100*high:+.2f}]"


def write(summary, root=ROOT):
    root = Path(root)
    output = root/"results/numeric_expansion"
    output.mkdir(parents=True, exist_ok=True)
    (output/"COMPARISON.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    if "costs" in summary:
        (output/"COSTS.json").write_text(json.dumps(summary["costs"], indent=2, allow_nan=False)+"\n")
    fields = ("dataset", "model", "display_model", "arm", "method", "train_per_class", "train_labels", "status", "n_test", "accuracy", "macro_f1", "balanced_accuracy", "n_failures", "probability_coverage", "log_loss", "brier_sum", "run_id", "source_run_id", "measurement_origin", "source_path")
    with (output/"COMPARISON.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summary["runs"])
    by_key = {row_key(row): row for row in summary["runs"]}
    paired = {(item["a"], item["kind"]): item for item in summary["comparisons"]}
    lines = ["# Expanded numerical classification and Jev review", "",
        f"**{summary['complete_runs']}/68 conditions are complete and audited.** Base LLMs: {summary['complete_base_runs']}/24; reviews: {summary['complete_review_runs']}/24; Jev direct: {summary['complete_direct_jev_runs']}/4; native references: {summary['complete_classical_runs']}/16. Pending conditions have no score.", "",
        "This is an exploratory expansion chosen after earlier results were observed. It compares six LLMs on two frozen numerical datasets at zero-shot and four examples per class, then asks Jev to review each saved proposed label with the same labeled examples. Bounded outputs do not guarantee accurate decisions.", "",
        "[Protocol](../../docs/EXPANDED_NUMERIC_PROTOCOL.md) · [All metrics and paired intervals](COMPARISON.json) · [CSV](COMPARISON.csv)", "",
        f"New local runs complete: {summary['complete_new_local_runs']}/8. New review runs complete: {summary['complete_new_review_runs']}/20. Historical reviews reused without additional calls: {summary['complete_reused_review_runs']}/4.", ""]
    for dataset in DATASETS:
        for shots in (0, 4):
            title = "zero-shot" if shots == 0 else "four examples per class"
            jev = by_key[dataset, JEV, "direct", shots]
            lines += [f"## {NAMES[dataset]}: {title}", "",
                f"Direct Jev reference accuracy: **{pct(jev['accuracy'])}**. Source and review use the same {0 if shots == 0 else 8 if dataset == 'breast_cancer' else 12} newly supplied labels.", "",
                "| LLM | Base accuracy | After Jev | Fixed | Correct → wrong label | Correct → failure | Review − base, pp [95% CI] | Review − Jev alone, pp [95% CI] |",
                "|---|---:|---:|---:|---:|---:|---|---|"]
            for model in MODEL_NAMES:
                base, review = by_key[dataset, model, "base", shots], by_key[dataset, model, "review", shots]
                comparison = paired[review["run_id"], "review_minus_source"]
                changes = comparison.get("transitions", {})
                lines.append(f"| {MODEL_NAMES[model]} | {pct(base['accuracy'])} | {pct(review['accuracy'])} | {changes.get('wrong_to_correct', '—')} | {changes.get('correct_to_wrong_label', '—')} | {changes.get('correct_to_failure', '—')} | {interval(comparison)} | {interval(paired[review['run_id'], 'review_minus_jev_alone'])} |")
            lines.append("")
    for budget, title in ((4, "Native references: matched four examples per class"), (None, "Native references: full training, additional labels")):
        lines += ["## "+title, "", "| Dataset | Model | Training labels | Test rows | Accuracy | Macro-F1 | Failures |", "|---|---|---:|---:|---:|---:|---:|"]
        for dataset in DATASETS:
            for model in CLASSICAL_NAMES:
                row = by_key[dataset, model, "classical", budget]
                f1 = "pending" if row["macro_f1"] is None else f"{row['macro_f1']:.4f}"
                lines.append(f"| {NAMES[dataset]} | {CLASSICAL_NAMES[model]} | {row['train_labels'] if row['train_labels'] is not None else '—'} | {row['n_test'] if row['n_test'] is not None else '—'} | {pct(row['accuracy'])} | {f1} | {row['n_failures'] if row['n_failures'] is not None else '—'} |")
        lines.append("")
    lines += ["## Incremental review cost", "",
        "New review spending is reconciled separately in [COSTS.json](COSTS.json), including partial or interrupted requests. Four historical reviews are reused and remain accounted in earlier spending. Known API charges and conservative reservations are different quantities; missing charge records are not treated as free calls.", "",
        "## Interpretation and limits", "",
        "All source and review failures count as incorrect, including an invalid source proposal that prevents a review call. Correction counts separate newly wrong labels from review failures. A positive net correction count is an observed change on these test rows; intervals are paired group-bootstrap intervals, not a guarantee of improvement elsewhere.", "",
        f"All intervals use {summary['bootstrap_samples']} resamples and seed 42, condition on the fixed split and fitted models, and are exploratory without adjustment for multiple comparisons. The JSON also contains few-shot-minus-zero-shot contrasts labeled as comparisons with unequal new-label budgets. A confidence interval that includes zero does not establish equivalence.", "",
        "Original Qwen/OpenAI proposals are reused from their original provider/model prompt rendering. New SmolLM2 and Granite runs use an explicit fixed system message with thinking disabled; this is not a controlled same-prompt architecture ablation. The numeric serialization and selected examples stay fixed. Class-probability scores from restricted-label likelihoods, native Choice and classical estimators are not interchangeable or guaranteed calibrated.", "",
        "Only 114 Breast Cancer and 36 Wine test rows, one split/seed, and familiar public datasets are covered. Pretraining exposure cannot be excluded. Wine cultivar IDs have arbitrary class meaning without examples: zero-shot primarily probes pretrained familiarity and inference from this serialization, not supervised learnability. Even perfect Wine accuracy does not establish general numerical-classification ability or a causal benefit from bounded outputs. Full-training native baselines use extra labels. This expansion adds no LoRA training.", "",
        "Review prompts include the original numerical row, matching-shot examples and a cached proposed class, without a newly generated rationale. The review-versus-Jev-alone contrast includes both the extra proposal/prompt wording and separate serving variability. A deployed chain pays and waits for both source and review; this experiment reuses existing proposals where available. Mixed MPS/CUDA and hosted latency are not directly comparable.", "",
        ("All 68 requested conditions are complete and audited. Interpretation remains conditional on the datasets and protocol above."
         if summary["complete_runs"] == summary["expected_runs"] else
         "Only fully audited rows receive scores. The matrix remains incomplete until every requested condition finishes; no aggregate headline is inferred from a partial subset."), ""]
    (output/"FINDINGS.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    summary = collect(args.root, samples=args.bootstrap_samples)
    write(summary, args.root)
    print(json.dumps({name: summary[name] for name in ("status", "complete_runs", "expected_runs", "complete_review_runs", "complete_comparisons")}))


if __name__ == "__main__":
    main()
