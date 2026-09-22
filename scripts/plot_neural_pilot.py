#!/usr/bin/env python3
"""Reproduce the SST-2/TREC neural pilot plot from saved run artifacts.

Run from the repository root:
    MPLCONFIGDIR=artifacts/matplotlib /opt/anaconda3/bin/python scripts/plot_neural_pilot.py

The classical comparator is always MultinomialNB at four training rows/class,
selected by method/configuration rather than by test performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
METHODS = ("zero_shot", "few_shot", "lora", "classical")
DISPLAY = ("Qwen · zero-shot", "Qwen · few-shot, 4/class", "Qwen · LoRA, 4/class", "TF–IDF + Naive Bayes, 4/class")
COLORS = ("#64748B", "#147D92", "#8656A5", "#C57A19")


def find_runs(input_dir: Path) -> dict[str, dict[str, dict]]:
    result = {dataset: {} for dataset in ("sst2", "trec")}
    for path in sorted(input_dir.glob("*/run.json")):
        record = json.loads(path.read_text())
        dataset, method = record.get("dataset"), record.get("method")
        if dataset not in result or method not in METHODS or record.get("seed") != 42 or record.get("status") != "complete":
            continue
        config = record["config"]
        if method == "classical":
            selected = config.get("model") == "multinomial_nb" and config.get("train_per_class") == 4
        else:
            selected = config.get("model") == MODEL
            if method == "few_shot":
                selected &= config.get("shots_per_class") == 4
            elif method == "lora":
                selected &= record.get("adapter_training", {}).get("train_per_class") == 4
            else:
                selected &= config.get("shots_per_class") == 0
        if not selected:
            continue
        if method in result[dataset]:
            raise ValueError(f"Ambiguous completed runs for {dataset}/{method}; use an input directory with one matching run")
        result[dataset][method] = record
    for dataset, runs in result.items():
        if set(runs) != set(METHODS):
            raise ValueError(f"Missing runs for {dataset}: {set(METHODS)-set(runs)}")
        if any(run["metrics"]["n_test"] != 200 for run in runs.values()):
            raise ValueError("This pilot figure requires exactly 200 test rows per run")
        if len({run["manifest_sha256"] for run in runs.values()}) != 1:
            raise ValueError(f"Dataset manifests differ across {dataset} runs")
        matched = [runs[method]["training_example_ids"] for method in ("few_shot", "lora", "classical")]
        if not all(ids == matched[0] for ids in matched):
            raise ValueError(f"Training IDs differ across matched arms for {dataset}")
        if runs["zero_shot"]["training_example_ids"]:
            raise ValueError("Zero-shot run must use no labeled examples")
        expected_count = 8 if dataset == "sst2" else 24
        if len(matched[0]) != expected_count:
            raise ValueError(f"Wrong labeled-example count for {dataset}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/pilot"))
    parser.add_argument("--output-stem", type=Path, default=Path("results/figures/neural-pilot"))
    args = parser.parse_args()
    runs = find_runs(args.input)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "svg.fonttype": "none", "svg.hashsalt": "jevbench-neural-pilot-v1", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 6.1), sharey=True)
    fig.subplots_adjust(left=0.275, right=0.945, bottom=0.27, top=0.74, wspace=0.25)
    fig.text(0.06, 0.94, "Small-model pilot: results depend on the task", fontsize=20, fontweight="bold", color="#182333")
    fig.text(0.06, 0.88, "Qwen2.5-0.5B-Instruct and a fixed TF–IDF + Multinomial Naive Bayes comparator · seed 42", fontsize=11.5, color="#475569")
    positions = np.arange(4)[::-1]
    for ax, dataset, title, label_budget in zip(axes, ("sst2", "trec"), ("SST-2 · sentiment", "TREC · question type"), (8, 24)):
        ax.set_title(title, loc="left", fontweight="bold", pad=45)
        ax.text(0, 1.085, f"200 test examples · {label_budget} labels for each 4/class arm", transform=ax.transAxes, fontsize=10, color="#475569")
        for position, method, color in zip(positions, METHODS, COLORS):
            record = runs[dataset][method]
            value = record["metrics"]["macro_f1"]
            interval = record["metrics"].get("bootstrap", {}).get("metrics", {}).get("macro_f1", {}).get("ci95")
            if interval is not None:
                low, high = interval
                ax.hlines(position, low, high, color=color, linewidth=3, alpha=0.65, zorder=2)
                ax.vlines([low, high], position - 0.07, position + 0.07, color=color, linewidth=1.5, zorder=2)
            ax.scatter([value], [position], s=86, color=color, edgecolors="white", linewidths=1.2, zorder=3)
            ax.text(1.035, position, f"{value:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=11, fontweight="bold", color=color)
        ax.set_yticks(positions, DISPLAY)
        ax.set_ylim(-0.45, 3.5)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("Macro-F1")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        ax.tick_params(axis="both", length=0, pad=9, labelcolor="#334155")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.06, 0.165, "Dots: macro-F1. Lines: 95% test-item bootstrap intervals (1,000 resamples).", fontsize=10.5, color="#475569")
    fig.text(0.06, 0.12, "Few-shot, LoRA, and Naive Bayes share the same training examples. Zero-shot uses no labeled examples.", fontsize=10.5, color="#475569")
    fig.text(0.06, 0.075, "One-seed pilot with 200 held-out examples per task; intervals exclude training-seed and prompt-selection uncertainty.", fontsize=10.5, color="#475569")
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        destination = args.output_stem.with_suffix(f".{suffix}")
        metadata = {"Date": None} if suffix == "svg" else {"Software": "jevbench plot_neural_pilot.py"}
        fig.savefig(destination, dpi=200, bbox_inches="tight", pad_inches=0.22, metadata=metadata)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
