import csv
import json
from pathlib import Path


def make_report(results_dir: Path, output: Path):
    runs = [json.loads(path.read_text()) for path in sorted(results_dir.rglob("run.json"))]
    complete = [r for r in runs if r.get("status") == "complete"]
    lines = ["# Recorded benchmark results", "", "Only completed, measured runs are listed. Missing models have not been benchmarked; no result is inferred.", "",
             "Compare rows only when test-manifest hashes, label budgets, preprocessing, and protocol match. These are dataset-specific results, not an overall model ranking.", "",
             "| Dataset | Model | Method / output protocol | Labels per class | Total train / dev labels | Seed | N | Accuracy | Macro F1 | 95% F1 CI | Failures | Prob. coverage |", "|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|"]
    flattened = []
    for run in complete:
        m, c = run["metrics"], run["config"]
        budget = c.get("train_per_class") if run["method"] == "classical" else c.get("shots_per_class", 0)
        training = run.get("training", run.get("adapter_training", {}))
        if run["method"] == "lora":
            budget = training["train_per_class"]
        if budget is None:
            budget = "full prepared train"
        train_labels = len(run["training_example_ids"])
        dev_labels = training.get("validation_rows", 0)
        protocol = "classical native" if run["method"] == "classical" else "closed-label likelihood" if c.get("provider") in {"hf", "huggingface"} else "native Choice" if c.get("provider") in {"jev", "typesafe"} else "label generation"
        ci = m["bootstrap"]["metrics"]["macro_f1"]["ci95"]
        lines.append(f"| {run['dataset']} | {c['model']} | {run['method']} / {protocol} | {budget} | {train_labels} / {dev_labels} | {run['seed']} | {m['n_test']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {m['n_failures']} | {m['probability_coverage']:.0%} |")
        flattened.append({"run_id": run["run_id"], "dataset": run["dataset"], "model": c["model"], "method": run["method"], "output_protocol": protocol, "train_labels": train_labels, "dev_labels": dev_labels, "budget_per_class": budget, "seed": run["seed"], "manifest_sha256": run["manifest_sha256"], **{k:v for k,v in m.items() if isinstance(v, (str, int, float)) or v is None}})
    lines.extend(["", "Probability metrics are computed only where actual class probabilities are available. Do not compare their quality without checking coverage and probability semantics.", "", "Classical prediction timings are amortized batch timings; hosted latency measures sequential end-to-end requests. They are not direct serving-speed comparisons.", "", f"Incomplete runs: {len(runs)-len(complete)}.", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    if flattened:
        fields = list(dict.fromkeys(key for row in flattened for key in row))
        with output.with_suffix(".csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flattened)
    return output
