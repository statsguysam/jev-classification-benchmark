#!/usr/bin/env python3
"""Draw the expanded numerical pilot from the audited, complete summary only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/numeric_expansion"
MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2 1.7B"),
    ("ibm-granite/granite-3.3-2b-instruct", "Granite 3.3 2B"),
    ("Qwen/Qwen3-4B-Instruct-2507", "Qwen3 4B"),
    ("gpt-5.6-luna", "GPT-5.6 Luna"),
    ("gpt-6-astra", "GPT-6 Astra"),
]
DATASETS = [("breast_cancer", "Breast Cancer · binary · 114 test rows"),
            ("wine", "Wine · 3 classes · 36 test rows")]
BG, INK, MUTED = "#f7f8fb", "#16263e", "#566578"
BASE, REVIEW, POS, NEG = "#66748f", "#078c88", "#078c88", "#c15953"


def prepare(summary):
    if summary.get("status") != "complete" or summary.get("complete_runs") != 68:
        raise ValueError("All 68 planned conditions must be audited before drawing publication figures")
    rows = summary["runs"]
    by_id = {r["run_id"]: r for r in rows}
    panels = {}
    for dataset, _ in DATASETS:
        for shots in (0, 4):
            pairs = []
            for model, name in MODELS:
                sources = [r for r in rows if r["dataset"] == dataset and r["model"] == model
                           and r["train_per_class"] == shots and "source_run_id" not in r]
                if len(sources) != 1:
                    raise ValueError(f"Expected one completed base condition: {dataset}/{model}/{shots}")
                base = sources[0]
                reviews = [r for r in rows if r.get("source_run_id") == base["run_id"]]
                comparisons = [c for c in summary["comparisons"] if c["kind"] == "review_minus_source"
                               and c["b"] == base["run_id"]]
                if len(reviews) != 1 or len(comparisons) != 1:
                    raise ValueError("All planned review conditions must be complete before drawing figures")
                if by_id[comparisons[0]["a"]]["run_id"] != reviews[0]["run_id"]:
                    raise ValueError("Unaligned comparison")
                pairs.append((name, base, reviews[0], comparisons[0]))
            direct = [r for r in rows if r["dataset"] == dataset and r["model"] == "typesafe/jev-1.13"
                      and r["train_per_class"] == shots]
            if len(direct) != 1:
                raise ValueError("Jev direct reference missing")
            panels[dataset, shots] = (pairs, direct[0])
    return panels


def canvas(title, subtitle):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11), sharey=True)
    fig.patch.set_facecolor(BG)
    fig.suptitle(title, x=.035, y=.972, ha="left", fontsize=23, weight="bold", color=INK)
    fig.text(.035, .926, subtitle, fontsize=12, color=MUTED)
    fig.subplots_adjust(left=.17, right=.97, top=.83, bottom=.16, hspace=.52, wspace=.14)
    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.set_yticks(range(6), [name for _, name in MODELS])
        ax.set_ylim(5.6, -.7)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.grid(axis="x", color="#dfe4eb", linewidth=.8)
        ax.set_axisbelow(True)
    return fig, axes


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    summary_path = OUT / "COMPARISON.json"
    summary = json.loads(summary_path.read_text())
    panels = prepare(summary)
    intervals = [c["paired_bootstrap"]["metrics"]["accuracy"]["ci95"] for c in summary["comparisons"]
                 if c["kind"] == "review_minus_source"]
    delta_min = min(-10, min(100*ci[0] for ci in intervals)-5)
    delta_max = max(10, max(100*ci[1] for ci in intervals)+5)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    fig, axes = canvas("What changes when Jev reviews an LLM's decision?",
        "Numerical tabular classification · Same test rows and labeled examples · Cached proposed class → Jev Choice")
    for i, (dataset, title) in enumerate(DATASETS):
        for j, shots in enumerate((0, 4)):
            ax = axes[i, j]
            pairs, direct = panels[dataset, shots]
            for y, (_, base, reviewed, _) in enumerate(pairs):
                a, b = 100*base["accuracy"], 100*reviewed["accuracy"]
                ax.plot([a, b], [y, y], color=POS if b >= a else NEG, linewidth=3, alpha=.5)
                ax.plot(a, y, "o", color=BASE, markersize=7)
                ax.plot(b, y, "D", color=REVIEW, markersize=6)
                ax.text(144, y, f"{a:.1f} → {b:.1f}", va="center", ha="right", fontsize=10, color=INK)
            ax.axvline(100*direct["accuracy"], color=REVIEW, ls="--", lw=1, alpha=.8)
            ax.set_xlim(0, 148)
            ax.set_xticks((0, 25, 50, 75, 100))
            ax.set_xlabel("Accuracy (%)")
            setting = "Zero-shot" if shots == 0 else f"Few-shot · {8 if dataset == 'breast_cancer' else 12} labels total"
            ax.set_title(f"{title}\n{setting}", loc="left", color=INK, fontsize=12, pad=13)
    fig.legend(handles=[Line2D([0],[0],marker="o",color=BASE,ls="",label="LLM alone"),
        Line2D([0],[0],marker="D",color=REVIEW,ls="",label="LLM → Jev"),
        Line2D([0],[0],ls="--",color=REVIEW,label="Jev alone (same examples)")],
        loc="upper left",bbox_to_anchor=(.035,.9),ncol=3,frameon=False,fontsize=11)
    fig.text(.035,.09,"Few-shot means 4 examples per class, shared by both stages. Failures remain in the denominator. No new fine-tuning.",fontsize=10,color=MUTED)
    fig.text(.035,.055,"Exploratory: two familiar datasets, one split/seed, fixed recipes. See FINDINGS.md for paired uncertainty and matched classical ML.",fontsize=10,color=MUTED)
    fig.text(.035,.025,"github.com/statsguysam/jev-classification-benchmark · results/numeric_expansion",fontsize=10,color=MUTED)
    save(fig,"expanded-accuracy")

    fig, axes = canvas("Does the second decision improve accuracy?",
        "Jev review minus the exact original LLM prediction · Paired 95% bootstrap intervals · Percentage points")
    for i,(dataset,title) in enumerate(DATASETS):
        for j,shots in enumerate((0,4)):
            ax=axes[i,j]
            for y,(_,base,reviewed,comparison) in enumerate(panels[dataset,shots][0]):
                value=comparison["paired_bootstrap"]["metrics"]["accuracy"]
                delta=100*value["estimate"]
                lo,hi=[100*v for v in value["ci95"]]
                color=POS if delta>=0 else NEG
                ax.hlines(y,lo,hi,color=color,lw=2)
                ax.plot(delta,y,"o",color=color,ms=7)
                t=comparison["transitions"]
                ax.text(delta_max+39,y,f"{delta:+.1f}  ({t['wrong_to_correct']} / {t['correct_to_wrong']})",va="center",ha="right",fontsize=10,color=INK)
            ax.axvline(0,color=MUTED,lw=1)
            ax.set_xlim(delta_min,delta_max+43)
            ax.set_xticks([v for v in range(-100,101,20) if delta_min<=v<=delta_max])
            ax.set_xlabel("Change in accuracy (percentage points)")
            ax.set_title(f"{title}\n{'Zero-shot' if shots==0 else 'Few-shot · 4 examples per class'}",loc="left",color=INK,fontsize=12,pad=13)
    fig.text(.035,.875,"Row annotation: net accuracy change (predictions corrected / predictions made wrong)",fontsize=11,color=MUTED)
    fig.text(.035,.09,"Intervals condition on this split and example selection; 2,000 paired group resamples. Exploratory comparisons, no multiplicity correction.",fontsize=10,color=MUTED)
    fig.text(.035,.055,"Bounded output is not a guarantee of correctness. A review can introduce errors or failures as well as correct them.",fontsize=10,color=MUTED)
    fig.text(.035,.025,"github.com/statsguysam/jev-classification-benchmark · results/numeric_expansion",fontsize=10,color=MUTED)
    save(fig,"expanded-review-deltas")

    fig, axes = plt.subplots(1,2,figsize=(14,7.5),sharey=True)
    fig.patch.set_facecolor(BG)
    fig.suptitle("How does Jev compare with native tabular models?",x=.04,y=.965,ha="left",fontsize=22,weight="bold",color=INK)
    fig.text(.04,.905,"Matched examples and held-out rows · Full-training ML shown separately because it uses more labels",fontsize=12,color=MUTED)
    native=[("typesafe/jev-1.13","Jev alone"),("xgboost","XGBoost"),("lightgbm","LightGBM"),
            ("logistic_regression","Logistic regression"),("random_forest","Random forest")]
    for ax,(dataset,title) in zip(axes,DATASETS):
        ax.set_facecolor(BG)
        for y,(model,name) in enumerate(native):
            for budget,offset,color,marker in ((4,-.12,REVIEW,"o"),(None,.12,BASE,"^")):
                if model=="typesafe/jev-1.13" and budget is None:
                    continue
                row=next(r for r in summary["runs"] if r["dataset"]==dataset and r["model"]==model and r["train_per_class"]==budget)
                lo,hi=[100*v for v in row["group_bootstrap"]["metrics"]["accuracy"]["ci95"]]
                point=100*row["accuracy"]
                ax.hlines(y+offset,lo,hi,color=color,lw=2,alpha=.6)
                ax.plot(point,y+offset,marker,color=color,ms=7)
                ax.text(112,y+offset,f"{point:.1f}%",ha="right",va="center",fontsize=10,color=color)
        ax.set_yticks(range(len(native)),[name for _,name in native])
        ax.set_ylim(4.6,-.6)
        ax.set_xlim(40,115)
        ax.set_xticks((40,60,80,100))
        ax.set_xlabel("Accuracy (%)")
        ax.tick_params(axis="y",length=0,pad=10)
        ax.grid(axis="x",color="#dfe4eb",lw=.8)
        ax.set_axisbelow(True)
        ax.set_title(title+f"\nMatched: {8 if dataset=='breast_cancer' else 12} labels · Full ML: {341 if dataset=='breast_cancer' else 106}",loc="left",fontsize=12,color=INK,pad=15)
    fig.legend(handles=[Line2D([0],[0],marker="o",color=REVIEW,ls="",label="4 examples per class"),
        Line2D([0],[0],marker="^",color=BASE,ls="",label="Full training split (ML only)")],
        loc="upper left",bbox_to_anchor=(.04,.86),ncol=2,frameon=False)
    fig.subplots_adjust(left=.19,right=.97,top=.72,bottom=.23,wspace=.13)
    fig.text(.04,.125,"Whiskers: 95% group-bootstrap intervals conditional on this split and training sample. Fixed untuned recipes; one seed.",fontsize=10,color=MUTED)
    fig.text(.04,.08,"Numerical features: 30 Breast Cancer / 13 Wine. Jev receives lossless named-feature serialization; ML receives native columns.",fontsize=10,color=MUTED)
    fig.text(.04,.035,"github.com/statsguysam/jev-classification-benchmark · results/numeric_expansion",fontsize=10,color=MUTED)
    save(fig,"expanded-classical-reference")
    (OUT/"FIGURE_PROVENANCE.json").write_text(json.dumps({
        "summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "plot_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figures":[f"{stem}.{ext}" for stem in ("expanded-accuracy","expanded-review-deltas","expanded-classical-reference") for ext in ("png","svg")],
    },indent=2)+"\n")


if __name__=="__main__":
    main()
