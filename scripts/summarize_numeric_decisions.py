#!/usr/bin/env python3
"""Recompute numeric-only comparisons and review corrections from saved predictions.

No provider calls. Historical rows must match their previously audited hashes;
new artifacts are checked against the same native/serialized prepared records.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import digest
from jevbench.types import Prediction
from summarize_tabular import audit_run, group_bootstrap
from tabular_data import load_native_prepared

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/numeric_decisions"
DATASETS = ("breast_cancer", "wine")
NAMES = {"breast_cancer": "Breast Cancer", "wine": "Wine"}
MODELS = {
    "typesafe/jev-1.13": "Jev alone",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3 4B alone",
    "gpt-6-astra": "GPT-6 Astra alone",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def chosen(predictions, n_classes):
    return [p.label if p.error is None and type(p.label) is int and 0 <= p.label < n_classes else -1
            for p in predictions]


def transitions(y, original, reviewed):
    require(len(y) == len(original) == len(reviewed) and len(y) > 0, "Unaligned review arrays")
    counts = {"wrong_to_correct": 0, "correct_to_wrong": 0, "both_correct": 0, "both_wrong": 0}
    for truth, before, after in zip(y, original, reviewed):
        key = ("both_correct" if after == truth else "correct_to_wrong") if before == truth else (
            "wrong_to_correct" if after == truth else "both_wrong")
        counts[key] += 1
    counts["changed_predictions"] = sum(a != b for a, b in zip(original, reviewed))
    counts["correct_to_failure"] = sum(before == truth and after == -1 for truth, before, after in zip(y, original, reviewed))
    counts["correct_to_wrong_label"] = counts["correct_to_wrong"] - counts["correct_to_failure"]
    counts["source_failure_rows"] = sum(before == -1 for before in original)
    counts["review_stage_failure_rows"] = sum(before != -1 and after == -1 for before, after in zip(original, reviewed))
    counts["net_correct_change"] = counts["wrong_to_correct"] - counts["correct_to_wrong"]
    counts["accuracy_delta_pp"] = 100 * counts["net_correct_change"] / len(y)
    return counts


def audit_supplement(path, dataset, *, model, train_per_class, display):
    record = read(path)
    require(record["status"] == "complete", f"Incomplete supplemental run: {path}")
    require(record["dataset"] == dataset.name and record["labels"] == dataset.labels, "Dataset/labels differ")
    require(record["seed"] == 42 and record["manifest_sha256"] == digest(dataset.manifest), "Dataset identity differs")
    expected_ids = [r.id for r in (dataset.train if train_per_class is None else
                                  select_examples(dataset.train, dataset.labels, train_per_class, 42))]
    require(record["training_example_ids"] == expected_ids, "Supplement has unequal training IDs")
    expected_manifest = {"dataset": dataset.name, "labels": dataset.labels, "manifest_sha256": digest(dataset.manifest),
        "rows": [{"id": r.id, "label": r.label, "text_sha256": hashlib.sha256(r.text.encode()).hexdigest()}
                 for r in dataset.test]}
    require(read(path.with_name("test_manifest.json")) == expected_manifest, "Supplement test manifest differs")
    predictions = [Prediction(**json.loads(line)) for line in path.with_name("predictions.jsonl").read_text().splitlines() if line]
    require([p.row_id for p in predictions] == [r.id for r in dataset.test], "Supplement prediction order differs")
    metrics = evaluate(dataset.test, predictions, len(dataset.labels))
    for key, value in metrics.items():
        saved = record["metrics"].get(key)
        require(math.isclose(value, saved, rel_tol=1e-10, abs_tol=1e-12) if isinstance(value, float) and
                isinstance(saved, (int, float)) else value == saved, f"Saved metric disagrees: {key}")
    values = chosen(predictions, len(dataset.labels))
    payload = {"y": [r.label for r in dataset.test], "predictions": values,
               "groups": [r["text_sha256"] for r in expected_manifest["rows"]], "n_classes": len(dataset.labels),
               "training_example_ids": expected_ids, "manifest_sha256": digest(dataset.manifest)}
    row = {"dataset": dataset.name, "model": model, "display_model": display, "method": record["method"],
        "train_per_class": train_per_class, "train_labels": len(expected_ids), "seed": 42,
        "run_id": record["run_id"], "source_path": path.relative_to(ROOT).as_posix(),
        "group_bootstrap": group_bootstrap(payload["y"], values, payload["groups"], n_classes=len(dataset.labels)),
        "artifact_sha256": {name: file_sha(path.with_name(name)) for name in ("run.json", "predictions.jsonl", "test_manifest.json")},
        **metrics}
    return row, payload


def audit_review_provenance(path, dataset):
    # These helpers only validate local records; they do not construct a provider
    # or call main(). Reuse execution's exact source/checkpoint/billing guards.
    import run_numeric_jev_review as review
    record = read(path)
    guard = record["config"]["budget_guard"]
    key = guard["proposal_key"]
    source, original, proposals, examples = review.load_source(dataset, key)
    config = read(ROOT / "configs/tabular_hosted.json")["jev_openrouter"]
    identity, _ = review.make_identity(dataset, key, source, original, examples, config,
                                       guard["ledger_id"], record["bootstrap_samples"])
    _, saved = review.load_checkpoint(path.parent, identity, dataset, source, original, proposals, examples)
    ledger = review.AnchoredLedger(review.budget.Ledger(review.LEDGER, review.STAGE_CAP))
    review.validate_reservations(ledger, dataset.name, key, saved, identity)
    for prediction in saved:
        if not prediction.error:
            require(prediction.metadata.get("resolved_model") in review.jev_route.ALLOWED_RESPONSE_MODELS,
                    "Review returned an unapproved model")
            require(prediction.probabilities is not None and prediction.probabilities[prediction.label] >=
                    max(prediction.probabilities) - 1e-6, "Review choice/distribution disagrees")


def costs(rows):
    from audit_numeric_review_costs import collect_costs
    return collect_costs(rows)


def contrast(a, b, rows, payloads):
    pa, pb = payloads[a], payloads[b]
    require(pa["manifest_sha256"] == pb["manifest_sha256"] and pa["y"] == pb["y"] and pa["groups"] == pb["groups"], "Unequal test records")
    require(pa["training_example_ids"] == pb["training_example_ids"], "Unequal training labels in matched contrast")
    return {"dataset": rows[a]["dataset"], "a": a, "b": b,
            "a_display": rows[a]["display_model"], "b_display": rows[b]["display_model"],
            "paired_bootstrap": group_bootstrap(pa["y"], pa["predictions"], pa["groups"], pb["predictions"], n_classes=pa["n_classes"])}


def collect():
    historical = read(ROOT / "results/TABULAR_COMPARISON.json")
    datasets = {name: load_native_prepared(ROOT / "data/tabular-full" / name) for name in DATASETS}
    rows, payloads = {}, {}
    for row in historical["runs"]:
        if row["dataset"] not in DATASETS or row["model"] not in MODELS or row["method"] == "lora":
            continue
        if row["method"] == "zero_shot" and row["model"] != "typesafe/jev-1.13":
            continue
        path = ROOT / row["source_path"]
        for name, expected in row["artifact_sha256"].items():
            require(file_sha(path.with_name(name)) == expected, "Historical artifact changed")
        audited, payload = audit_run(path, *datasets[row["dataset"]], ROOT)
        audited["display_model"] = MODELS[row["model"]]
        rows[row["run_id"]], payloads[row["run_id"]] = audited, payload
    for path in sorted((OUT / "boosting").glob("*/run.json")):
        from audit_numeric_boosting import audit_boosting_run
        record = read(path)
        model = record["config"]["model"]
        require(model in ("xgboost", "lightgbm"), "Unexpected booster")
        audit_boosting_run(path, *datasets[record["dataset"]])
        row, payload = audit_supplement(path, datasets[record["dataset"]][0], model=model,
            train_per_class=record["config"]["train_per_class"], display=MODELS[model])
        rows[row["run_id"]], payloads[row["run_id"]] = row, payload
    for path in sorted((OUT / "review").glob("*/run.json")):
        record = read(path)
        if record["status"] != "complete":
            continue
        audit_review_provenance(path, datasets[record["dataset"]][0])
        model = record["config"]["source_model"]
        source_id = record["config"]["source_run_id"]
        require(source_id in rows and rows[source_id]["model"] == model, "Unknown review source")
        row, payload = audit_supplement(path, datasets[record["dataset"]][0], model=model + "+jev_review",
            train_per_class=4, display=MODELS[model].removesuffix(" alone") + " → Jev")
        row["source_run_id"] = source_id
        row["review_transitions"] = transitions(payload["y"], payloads[source_id]["predictions"], payload["predictions"])
        rows[row["run_id"]], payloads[row["run_id"]] = row, payload
    comparisons = []
    for ident, row in rows.items():
        if "source_run_id" in row:
            comparisons.append({"kind": "review_minus_source", **contrast(ident, row["source_run_id"], rows, payloads),
                                "transitions": row["review_transitions"]})
            jev_id = next(key for key, value in rows.items() if value["dataset"] == row["dataset"]
                          and value["model"] == "typesafe/jev-1.13" and value["train_per_class"] == 4)
            comparisons.append({"kind": "review_minus_jev_alone", **contrast(ident, jev_id, rows, payloads)})
        if row["train_per_class"] == 4 and (row["model"] == "typesafe/jev-1.13" or "source_run_id" in row):
            for peer_id, peer in rows.items():
                if peer["dataset"] == row["dataset"] and peer["train_per_class"] == 4 and peer["model"] in ("xgboost", "lightgbm"):
                    comparisons.append({"kind": "jev_system_minus_matched_boosting", **contrast(ident, peer_id, rows, payloads)})
    identities = [(r["dataset"], r["model"], r["train_per_class"]) for r in rows.values()]
    require(len(identities) == len(set(identities)), "Duplicate conditions")
    return {"schema_version": 1, "scope": "numeric-only exploratory supplement", "datasets": list(DATASETS),
        "source_sha256": file_sha(__file__), "protocol_sha256": file_sha(ROOT / "docs/NUMERIC_DECISIONS_PROTOCOL.md"),
        "runs": list(rows.values()), "comparisons": comparisons, "costs": costs(list(rows.values())),
        "expected_new_review_runs": 4, "complete_new_review_runs": sum("source_run_id" in r for r in rows.values()),
        "expected_boosting_runs": 8, "complete_boosting_runs": sum(r["model"] in ("xgboost", "lightgbm") for r in rows.values())}


def pct(value):
    return f"{100 * value:.1f}%"


def interval(result, metric="accuracy", scale=100):
    value = result["metrics"][metric]
    lo, hi = value["ci95"]
    return f"{scale * value['estimate']:+.2f} [{scale * lo:+.2f}, {scale * hi:+.2f}]"


def write(summary):
    OUT.mkdir(exist_ok=True)
    (OUT / "COMPARISON.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (OUT / "COSTS.json").write_text(json.dumps(summary["costs"], indent=2, allow_nan=False) + "\n")
    fields = ("dataset", "display_model", "method", "train_per_class", "train_labels", "n_test", "accuracy", "macro_f1",
              "balanced_accuracy", "n_failures", "log_loss", "brier_sum", "probability_coverage", "run_id", "source_path")
    with (OUT / "COMPARISON.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["runs"])
    lines = ["# Numerical bounded decisions: measured pilot", "",
        "This supplement tests Jev on numerical tabular classification, including an LLM proposed-class → Jev review arm and true XGBoost/LightGBM baselines. "
        "It is an exploratory two-dataset study, not evidence that bounded outputs guarantee correctness or that Jev improves every LLM.", "",
        "[Protocol](../../docs/NUMERIC_DECISIONS_PROTOCOL.md) · [All metrics and paired intervals](COMPARISON.json) · [CSV](COMPARISON.csv)", "",
        f"New review conditions complete: **{summary['complete_new_review_runs']}/4**. Boosting conditions complete: **{summary['complete_boosting_runs']}/8**.", "",
        "## Does Jev fix more LLM errors than it introduces?", "",
        "The exact saved four-per-class LLM predictions are reused as advisory proposals. Jev also sees the original numerical row and the same demonstrations. "
        "No new LLM rationale is generated. Source or reviewer failures count as incorrect. Intervals are paired 95% group-bootstrap intervals, conditional on one split and example selection.", "",
        "| Dataset | Source → Jev | Before | After | Fixed | Correct → wrong label | Correct → failure | Accuracy change, pp [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    by_id = {r["run_id"]: r for r in summary["runs"]}
    for comparison in summary["comparisons"]:
        if comparison["kind"] != "review_minus_source":
            continue
        a, b = by_id[comparison["a"]], by_id[comparison["b"]]
        changes = comparison["transitions"]
        lines.append(f"| {NAMES[a['dataset']]} | {a['display_model']} | {pct(b['accuracy'])} | {pct(a['accuracy'])} | "
                     f"{changes['wrong_to_correct']} | {changes['correct_to_wrong_label']} | {changes['correct_to_failure']} | {interval(comparison['paired_bootstrap'])} |")
    if summary["complete_new_review_runs"] < 4:
        lines += ["", "Incomplete review arms are not assigned scores or conclusions."]
    lines += ["", "## Does the LLM proposal add value over Jev alone?", "",
        "Both arms receive the original numerical row and the same labeled examples. The review arm additionally receives the saved proposed class. "
        "These are separate Jev calls, so any observed difference can also include serving/sampling variability.", "",
        "| Dataset | Review pipeline | Accuracy change vs Jev alone, pp [95% CI] |", "|---|---|---|"]
    for comparison in summary["comparisons"]:
        if comparison["kind"] == "review_minus_jev_alone":
            lines.append(f"| {NAMES[comparison['dataset']]} | {comparison['a_display']} | {interval(comparison['paired_bootstrap'])} |")
    for title, budget in (("Same small label budget: four records per class", 4), ("Full-training supervised reference: additional labels", None), ("Jev zero-shot reference: no supplied labels", 0)):
        lines += ["", "## " + title, "", "| Dataset | System | Training labels | Test rows | Accuracy | Macro-F1 | Failures |", "|---|---|---:|---:|---:|---:|---:|"]
        for row in sorted(summary["runs"], key=lambda r: (r["dataset"], r["display_model"])):
            if row["train_per_class"] != budget:
                continue
            lines.append(f"| {NAMES[row['dataset']]} | {row['display_model']} | {row['train_labels']} | {row['n_test']} | "
                         f"{pct(row['accuracy'])} | {row['macro_f1']:.3f} | {row['n_failures']} |")
    lines += ["", "## Jev systems versus boosted trees with matched labels", "",
        "All differences are the named Jev system minus the tree baseline, in accuracy percentage points. A confidence interval crossing zero does not prove equivalence.", "",
        "| Dataset | Jev system | Tree baseline | Difference [95% paired CI] |", "|---|---|---|---|"]
    for comparison in summary["comparisons"]:
        if comparison["kind"] == "jev_system_minus_matched_boosting":
            lines.append(f"| {NAMES[comparison['dataset']]} | {comparison['a_display']} | {comparison['b_display']} | {interval(comparison['paired_bootstrap'])} |")
    cost = summary["costs"]
    lines += ["", "## Measured incremental cost", ""]
    if cost["status"] == "complete":
        lines += [f"The new review made **{cost['new_model_requests']} Jev calls** and no new LLM-generation calls. "
            f"Known API-reported charges total **US${cost['known_reported_api_usd']}** across {cost['reported_cost_requests']} calls; "
            f"Calls with unknown reported charges: {cost['unknown_cost_requests']}. Conservative review reservations are "
            f"**US${cost['review_conservative_usd']}**, bringing cumulative study accounting to **US${cost['cumulative_conservative_usd']} / US$20**. "
            "Reservations are not an invoice. [Machine-readable cost reconciliation](COSTS.json)."]
    else:
        lines += ["Review cost reconciliation is not final while review conditions remain incomplete. [Current accounting](COSTS.json)."]
    lines += ["", "## Limits", "",
        "Only 114 Breast Cancer and 36 Wine test rows, one split, seed 42, one serialization, fixed untuned tree recipes, and already familiar public datasets. "
        "Four labels per class is an extreme low-data setting; it does not characterize normally trained boosting. Full-training trees use additional labels. "
        "Pretraining exposure cannot be ruled out. Model architecture and output protocol are not isolated. Jev probabilities may still be miscalibrated on this task.", "",
        "The review's extra cost is incremental to the source LLM call. Reusing cached proposals avoids new LLM charges in this experiment; a live chain still pays and waits for both stages. "
        "Local amortized batch timings and hosted request timings are not controlled speed comparisons.", ""]
    (OUT / "FINDINGS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    result = collect()
    write(result)
    print(json.dumps({"runs": len(result["runs"]), "reviews_complete": result["complete_new_review_runs"],
                      "boosting_complete": result["complete_boosting_runs"], "report": "results/numeric_decisions/FINDINGS.md"}))
