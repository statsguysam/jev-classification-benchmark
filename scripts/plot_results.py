"""Plot recorded results only. Requires matplotlib (pip install matplotlib)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=Path("results/pilot"))
parser.add_argument("--output", type=Path, default=Path("results/figures"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
runs = [json.loads(p.read_text()) for p in sorted(args.results.rglob("run.json"))]
runs = [r for r in runs if r["status"] == "complete"]
datasets = ["sst2", "imdb", "ag_news", "trec", "banking77"]
names = {"logistic_regression": "TF-IDF + logistic regression", "linear_svc": "TF-IDF + linear SVM", "multinomial_nb": "TF-IDF + Naive Bayes"}
colors = ["#2874a6", "#15927f", "#bf7434"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
for axis, budget, title in zip(axes, [4, None], ["4 labeled examples per class; no development labels", "Full prepared training; development-set selection"]):
    x = np.arange(len(datasets))
    for j, (model, label) in enumerate(names.items()):
        selected = []
        for dataset in datasets:
            matches = [r for r in runs if r["method"] == "classical" and r["dataset"] == dataset and r["config"]["model"] == model and r["config"]["train_per_class"] == budget and r["seed"] == 42]
            if len(matches) > 1:
                raise ValueError("Ambiguous duplicate runs; select one frozen result directory")
            selected.append(matches[0] if matches else None)
        values = [r["metrics"]["macro_f1"] if r else np.nan for r in selected]
        lows = [max(0, r["metrics"]["macro_f1"]-r["metrics"]["bootstrap"]["metrics"]["macro_f1"]["ci95"][0]) if r else 0 for r in selected]
        highs = [max(0, r["metrics"]["bootstrap"]["metrics"]["macro_f1"]["ci95"][1]-r["metrics"]["macro_f1"]) if r else 0 for r in selected]
        axis.bar(x+(j-1)*.24, values, width=.22, color=colors[j], label=label, yerr=[lows, highs], capsize=2, error_kw={"elinewidth":.8})
    axis.set_xticks(x, ["SST-2", "IMDb", "AG News", "TREC", "Banking77"])
    axis.set_title(title, fontsize=10, pad=12)
    axis.set_ylim(0,1)
    axis.grid(axis="y", alpha=.18)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Macro F1 (200 held-out rows per dataset)")
fig.suptitle("Classical baselines: measured pilot, seed 42", fontsize=15, x=.06, ha="left")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.02),fontsize=9)
fig.text(.06,.005,"95% stratified bootstrap intervals. Full prepared train ≤10,000; dev ≤1,000. All inputs capped at 2,000 characters. Banking77 is exploratory.",fontsize=8,color="#444444")
fig.tight_layout(rect=[0,.09,1,.94])
fig.savefig(args.output/"classical-pilot.png",dpi=180,bbox_inches="tight")
fig.savefig(args.output/"classical-pilot.svg",bbox_inches="tight")
print(args.output/"classical-pilot.png")
