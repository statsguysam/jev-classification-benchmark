#!/usr/bin/env python3
"""Export figures for the numeric-only pilot from its audited JSON summary.

MPLCONFIGDIR=artifacts/matplotlib /opt/anaconda3/bin/python scripts/plot_numeric_decisions.py
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/numeric_decisions"
ORDER = ["Logistic regression", "XGBoost", "LightGBM", "Jev alone", "Qwen3 4B alone", "Qwen3 4B → Jev", "GPT-6 Astra alone", "GPT-6 Astra → Jev"]
COLORS = {"Logistic regression": "#596579", "XGBoost": "#b97825", "LightGBM": "#b97825", "Jev alone": "#087e8b",
          "Qwen3 4B alone": "#7053ab", "Qwen3 4B → Jev": "#087e8b", "GPT-6 Astra alone": "#365fbc", "GPT-6 Astra → Jev": "#087e8b"}


def main():
    path = OUT / "COMPARISON.json"
    summary = json.loads(path.read_text())
    if summary["complete_new_review_runs"] != 4 or summary["complete_boosting_runs"] != 8:
        raise ValueError("All planned numeric conditions must be complete before drawing the publication figures")
    rows = {(r["dataset"], r["display_model"], r["train_per_class"]): r for r in summary["runs"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5), sharey=True)
    fig.patch.set_facecolor("#f7f8fb")
    fig.suptitle("Can a bounded-decision model compete on numerical data?", x=.055, ha="left", y=.96, fontsize=20, fontweight="bold", color="#14243b")
    fig.text(.055, .907, "Same four labeled examples per class · Jev alone and LLM proposals reviewed by Jev", fontsize=12, color="#526075")
    for ax, dataset, title in zip(axes, ("breast_cancer", "wine"), ("Breast Cancer · binary\n114 test rows · 8 training labels", "Wine · 3 classes\n36 test rows · 12 training labels")):
        ax.set_facecolor("#f7f8fb")
        for y, name in enumerate(ORDER):
            row = rows[dataset, name, 4]
            low, high = [x * 100 for x in row["group_bootstrap"]["metrics"]["accuracy"]["ci95"]]
            point = row["accuracy"] * 100
            color = COLORS[name]
            ax.hlines(y, low, high, color=color, linewidth=2, alpha=.55)
            ax.plot(point, y, "o", markersize=8, color=color)
            ax.text(105, y, f"{point:.1f}%", va="center", ha="left", fontsize=11, fontweight="bold", color=color)
        ax.set_xlim(45, 118)
        ax.set_xticks([50, 60, 70, 80, 90, 100])
        ax.set_xlabel("Accuracy (%)")
        ax.set_yticks(range(len(ORDER)), ORDER)
        ax.set_ylim(len(ORDER)-.5, -.6)
        ax.grid(axis="x", color="#dde2eb", linewidth=.8)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=13, loc="left", pad=22, color="#14243b")
        ax.tick_params(axis="y", length=0, pad=12)
    fig.subplots_adjust(left=.20, right=.97, top=.79, bottom=.22, wspace=.14)
    fig.text(.055, .125, "Whiskers: 95% bootstrap intervals conditional on this split and sample selection. Failures count as incorrect.", fontsize=10, color="#526075")
    fig.text(.055, .09, "Exploratory pilot: 2 familiar public datasets, 1 seed, fixed untuned recipes. Full-training ML uses more labels and is reported separately.", fontsize=10, color="#526075")
    fig.text(.055, .055, "Source: github.com/statsguysam/jev-classification-benchmark · results/numeric_decisions", fontsize=10, color="#526075")
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"numeric-comparison.{suffix}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    reviews = [c for c in summary["comparisons"] if c["kind"] == "review_minus_source"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("#f7f8fb")
    ax.set_facecolor("#f7f8fb")
    labels = []
    extent = max(max(c["transitions"]["wrong_to_correct"], c["transitions"]["correct_to_wrong"]) for c in reviews) + 3
    for y, c in enumerate(reviews):
        t = c["transitions"]
        ax.barh(y, t["wrong_to_correct"], color="#138678", height=.48)
        ax.barh(y, -t["correct_to_wrong"], color="#c56155", height=.48)
        ax.text(t["wrong_to_correct"]+.25, y, str(t["wrong_to_correct"]), va="center", color="#11695f", fontweight="bold")
        if t["correct_to_wrong"]:
            ax.text(-t["correct_to_wrong"]-.25, y, str(t["correct_to_wrong"]), va="center", ha="right", color="#a0443c", fontweight="bold")
        labels.append(("Breast Cancer" if c["dataset"] == "breast_cancer" else "Wine") + "\n" + c["a_display"])
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(-extent, extent)
    ax.axvline(0, color="#6e7787", linewidth=1)
    ax.set_xlabel("Predictions made wrong  ←                 →  Predictions corrected")
    ax.tick_params(axis="y", length=0, pad=12)
    fig.suptitle("Does Jev fix more LLM errors than it introduces?", x=.055, ha="left", y=.94, fontsize=21, fontweight="bold", color="#14243b")
    fig.text(.055, .865, "The same saved LLM proposal, reviewed against the original numeric features and training examples", fontsize=12, color="#526075")
    fig.subplots_adjust(left=.24, right=.95, top=.77, bottom=.24)
    fig.text(.055, .11, "4 training examples/class · 114 + 36 test rows · red counts include failures (detailed separately in report) · one split/seed", fontsize=10, color="#526075")
    fig.text(.055, .06, "Exploratory results for this label-review pipeline; not proof that bounded output alone improves accuracy. See the paired intervals in FINDINGS.md.", fontsize=10, color="#526075")
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"review-corrections.{suffix}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    (OUT / "FIGURE_PROVENANCE.json").write_text(json.dumps({"summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figures": ["numeric-comparison.png", "numeric-comparison.svg", "review-corrections.png", "review-corrections.svg"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
