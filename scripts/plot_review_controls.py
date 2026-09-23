"""Render the complete, audited proposal controls as a publication figure.

No API calls. The audit is rerun before plotting; partial arms never appear as
completed results. Both prespecified contrasts and all four datasets are shown.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import summarize_review_controls as summary

NAMES = {"breast_cancer": "Breast Cancer", "wine": "Wine", "sst2": "SST-2", "trec": "TREC"}
ARMS = (("actual", "Real proposal", "#087f8c"),
        ("no_proposal", "No proposal", "#6b778d"),
        ("shuffled", "Shuffled proposal", "#cd732f"))


def main():
    source = ROOT / "results/review_controls/COMPARISON.json"
    recorded = json.loads(source.read_text())
    audited = summary.collect()
    if audited != recorded or audited["status"] != "complete":
        raise ValueError("Regenerate the complete audited controls report before plotting")
    report = audited
    names = list(NAMES)
    runs = {(r["dataset"], r["arm"]): r for r in report["runs"]}
    comparisons = {(r["dataset"], r["contrast"]): r for r in report["comparisons"]}
    if len(runs) != 12 or len(comparisons) != 8:
        raise ValueError("Publication figure requires every frozen dataset, arm and contrast")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c5ceda", "text.color": "#14223b",
        "axes.labelcolor": "#14223b", "xtick.color": "#52627a",
        "ytick.color": "#52627a", "svg.hashsalt": "jev-proposal-controls-v1"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 9.5), facecolor="white",
                                     gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.14, right=.96, top=.72, bottom=.27, wspace=.24)
    fig.text(.055, .94, "Does the LLM's proposal help Jev?", fontsize=26, weight="bold")
    fig.text(.055, .89, "Same row. Same examples. Only the proposed class changes.",
             fontsize=15, color="#52627a")
    fig.text(.055, .845, "First attempts · Qwen3 4B · 4 examples per class · 550 held-out rows",
             fontsize=12, color="#52627a")
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=8,
                    color=color, label=label) for _, label, color in ARMS],
               loc="upper left", bbox_to_anchor=(.048, .81), ncol=3, frameon=False)

    offsets = (-.19, 0, .19)
    for i, name in enumerate(names):
        for (arm, _, color), offset in zip(ARMS, offsets):
            value = 100 * runs[name, arm]["metrics"]["accuracy"]
            left.plot(value, i + offset, "o", color=color, ms=8, zorder=3)
            left.annotate(f"{value:.1f}%", (value, i + offset), xytext=(8, 0),
                          textcoords="offset points", va="center", fontsize=10, color=color)
        for offset, contrast, color in ((-.12, "actual_minus_no_proposal", "#087f8c"),
                                        (.12, "actual_minus_shuffled", "#cd732f")):
            m = comparisons[name, contrast]["paired_group_bootstrap"]["metrics"]["accuracy"]
            value = 100 * m["estimate"]
            lo, hi = (100 * v for v in m["ci95"])
            right.hlines(i + offset, lo, hi, color=color, lw=2)
            right.plot(value, i + offset, "o", color=color, ms=7, zorder=3)
            right.annotate(f"{value:+.1f}", (value, i + offset), xytext=(0, -14),
                           textcoords="offset points", ha="center", fontsize=9, color=color)

    labels = [f"{NAMES[name]}\nn = {runs[name, 'actual']['expected_requests']}" for name in names]
    left.set_yticks(range(len(names)), labels)
    left.set(xlim=(0, 113), ylim=(3.5, -.5), xlabel="Jev accuracy", title="All three prompt conditions")
    left.set_xticks([0, 25, 50, 75, 100])
    left.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    left.grid(axis="x", color="#e7ecf2", zorder=0)
    right.axvline(0, color="#9ba7b8", lw=1.2, zorder=0)
    maximum = max(abs(100 * end) for c in report["comparisons"]
                  for end in c["paired_group_bootstrap"]["metrics"]["accuracy"]["ci95"])
    limit = max(5, maximum * 1.18)
    right.set(xlim=(-limit, limit), ylim=(3.5, -.5), xlabel="Accuracy difference (percentage points)",
              title="Added value of the real proposal")
    right.set_yticks(range(len(names)), [""] * len(names))
    right.grid(axis="x", color="#e7ecf2", zorder=0)
    for ax in (left, right):
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.title.set_fontsize(12)
        ax.title.set_weight("bold")
        for boundary in (.5, 1.5, 2.5):
            ax.axhline(boundary, color="#eff2f6", lw=1, zorder=0)
    right.legend(handles=[Line2D([], [], color="#087f8c", marker="o", label="Real minus none"),
                          Line2D([], [], color="#cd732f", marker="o", label="Real minus shuffled")],
                 loc="upper center", bbox_to_anchor=(.5, -.19), frameon=False, ncol=2, fontsize=10)
    fig.text(.055, .155, "Error bars: paired 95% group-bootstrap intervals. Positive differences favor real proposals.",
             fontsize=10, color="#52627a")
    repeat = report["serving_repeat_diagnostic"]["counts"]
    failed_pairs = sum(repeat[key] for key in ("reference_failed_only", "repeat_failed_only", "both_failed"))
    fig.text(.055, .12,
             f"Service check: {repeat['valid_label_disagreement']}/{repeat['both_valid']} valid identical-prompt pairs disagreed; "
             f"{failed_pairs}/64 pairs had a failure. Primary calls with errors: {sum(len(r['failures']) for r in report['runs'])}/1,650.",
             fontsize=10, color="#52627a")
    fig.text(.055, .050, "Exploratory: familiar public holdouts already inspected; one split; unadjusted intervals. Failures count as incorrect.\n"
             "Intervals are conditional on these cases and responses, not new training or serving runs. No pooled winner.\n"
             "Method + results: github.com/statsguysam/jev-classification-benchmark · results/review_controls/FINDINGS.md",
             fontsize=8.5, color="#52627a", linespacing=1.5)
    out = ROOT / "results/review_controls/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "proposal_value.png", dpi=180, facecolor="white", metadata={"Software": "matplotlib"})
    fig.savefig(out / "proposal_value.svg", facecolor="white", metadata={"Date": None})
    metadata = {"report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "matplotlib_version": matplotlib.__version__,
                "data_scope": "All four datasets, three primary arms and both prespecified paired contrasts; first attempts",
                "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
                            if p.suffix in {".png", ".svg"}}}
    (out / "MANIFEST.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
