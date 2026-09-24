"""Offline figures for the Medium narrative, derived from frozen study artifacts.

The selected examples are explicit, not chosen by a best-score query. Primary
control results stay primary. No inference, price lookup, or report mutation.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import plot_review_outcomes as audit

OUT = ROOT / "results/medium_story/figures"
INK = "#183448"
MUTED = "#536978"
GRID = "#e4eaf0"
TEAL = "#087f8c"
LIGHT_TEAL = "#69b5b3"
BLUE = "#527ba1"
GOLD = "#c28b36"
RED = "#bd5662"
NAMES = {"breast_cancer": "Breast Cancer", "wine": "Wine", "sst2": "SST-2", "trec": "TREC"}
SOURCES = ["results/numeric_expansion/COMPARISON.json", "results/text_extension/COMPARISON.json",
           "results/review_controls/COMPARISON.json", "results/review_value/LINKEDIN_COST_NOTE.md"]
CAPTIONS = {}
PROVENANCE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, name, caption, data):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=160, facecolor="white", metadata={"Software": "matplotlib"})
    fig.savefig(OUT / f"{name}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    CAPTIONS[name] = caption
    PROVENANCE[name] = data


def header(fig, title, subtitle):
    fig.text(.055, .93, title, fontsize=24, weight="bold", color=INK)
    fig.text(.055, .872, subtitle, fontsize=12.5, color=MUTED)


def clean_axis(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelcolor=MUTED)


def find_run(reports, dataset, model, arm, shots=4):
    rows = [r for report in reports for r in report["runs"] if r["dataset"] == dataset
            and r["model"] == model and r["arm"] == arm and r["train_per_class"] == shots]
    assert len(rows) == 1, (dataset, model, arm)
    return rows[0]


def wine(reports):
    identities = [("Qwen/Qwen2.5-0.5B-Instruct", "base", "Qwen2.5 0.5B alone", BLUE),
                  ("Qwen/Qwen2.5-0.5B-Instruct+jev_review", "review", "Qwen2.5 + Jev review", TEAL),
                  ("typesafe/jev-1.13", "direct", "Jev alone", LIGHT_TEAL),
                  ("logistic_regression", "classical", "Logistic regression", GOLD)]
    rows = [find_run(reports, "wine", m, a) for m, a, _, _ in identities]
    assert [round(r["accuracy"] * r["n_test"]) for r in rows] == [12, 33, 33, 35]
    predictions = [[json.loads(line) for line in (ROOT / Path(row["source_path"]).parent / "predictions.jsonl").read_text().splitlines()]
                   for row in rows[1:3]]
    assert [(r["row_id"], r["label"]) for r in predictions[0]] == [(r["row_id"], r["label"]) for r in predictions[1]]
    fig = plt.figure(figsize=(10.5, 6.7), facecolor="white")
    header(fig, "The rescue was real. The extra model was optional.",
           "Wine classification | 36 test rows | 12 training examples for each method")
    ax = fig.add_axes([.29, .26, .65, .53])
    for i, (row, (_, _, _, color)) in enumerate(zip(rows, identities)):
        value = row["accuracy"] * 100
        ax.barh(i, value, height=.6, color=color)
        ax.text(value - 2, i, f"{value:.1f}%", color="white", ha="right", va="center", weight="bold", fontsize=14)
        ax.text(103, i, f"{round(row['accuracy'] * 36)}/36", va="center", fontsize=13, color=INK)
    ax.set(ylim=(3.6, -.6), xlim=(0, 118), xlabel="Accuracy")
    ax.set_yticks(range(4), [t[2] for t in identities], fontsize=12)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    clean_axis(ax)
    fig.text(.055, .145, "All 36 reviewed predictions matched Jev alone.", fontsize=15, weight="bold", color=TEAL)
    fig.text(.055, .072, "Selected few-shot condition: four examples per class. Numerical features used directly by logistic regression.\n"
             "One fixed split; descriptive comparison. Qwen2.5 chose a single class throughout this condition.",
             fontsize=10.2, color=MUTED, linespacing=1.5)
    save(fig, "01_wine_rescue", "On the 36-row Wine test set, adding Jev improved Qwen2.5 from 12 to 33 correct answers. "
         "Jev alone made exactly the same final predictions. Logistic regression scored 35/36 with the same 12 training labels.",
         {"scope": "Selected Wine, four examples per class, original saved runs", "runs": rows})


def trec(reports, panels):
    row = next(r for r in panels["trec"] if r["model"] == "gpt-6-astra" and r["shots_per_class"] == 4)
    assert (row["corrected"], row["harmed_wrong_label"], row["harmed_failure"]) == (3, 24, 0)
    fig = plt.figure(figsize=(10.5, 6.7), facecolor="white")
    header(fig, "A second opinion can undo a good decision.",
           "TREC question classification | 200 test questions | 24 examples in each prompt")
    ax = fig.add_axes([.075, .33, .37, .43])
    values = [row["source_accuracy"] * 100, row["review_accuracy"] * 100]
    ax.bar([0, 1], values, color=[BLUE, TEAL], width=.58)
    for x, value, count in zip([0, 1], values, [194, 173]):
        ax.text(x, value + 3, f"{value:.1f}%", ha="center", fontsize=17, weight="bold", color=INK)
        ax.text(x, value - 9, f"{count}/200", ha="center", color="white", fontsize=13)
    ax.set(ylim=(0, 110), ylabel="Accuracy")
    ax.set_xticks([0, 1], ["GPT-6 Astra", "Astra + Jev"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    clean_axis(ax, "y")
    fig.text(.535, .734, "What changed?", fontsize=17, weight="bold", color=INK)
    fig.text(.535, .60, "3", fontsize=34, weight="bold", color=TEAL)
    fig.text(.625, .62, "wrong answers corrected", fontsize=13, color=INK)
    fig.text(.535, .455, "24", fontsize=34, weight="bold", color=RED)
    fig.text(.625, .475, "correct answers made wrong", fontsize=13, color=INK)
    fig.text(.535, .34, "21 fewer correct answers overall", fontsize=14, weight="bold", color=RED)
    fig.text(.055, .17, "97.0% to 86.5% accuracy after review", fontsize=17, weight="bold", color=INK)
    fig.text(.055, .077, "Selected few-shot condition: four examples per class. Jev saw the original question and Astra's proposed class.\n"
             "Counts follow each question from the source model to the reviewer. One fixed test set.",
             fontsize=10.2, color=MUTED, linespacing=1.5)
    save(fig, "02_trec_overrides", "On TREC, Jev corrected three Astra mistakes but changed 24 correct answers to wrong ones. "
         "The net effect was 21 fewer correct answers out of 200, a drop of 10.5 percentage points.", row)


def controls(report):
    assert report["status"] == "complete" and report["complete_primary_arms"] == 12
    rows = {r["dataset"]: r for r in report["comparisons"] if r["contrast"] == "actual_minus_no_proposal"}
    assert set(rows) == set(NAMES)
    fig = plt.figure(figsize=(10.5, 7.0), facecolor="white")
    header(fig, "Does showing Jev the suggested answer help?",
           "Same rows, examples and prompt wording | Qwen3 4B supplies the suggestion")
    ax = fig.add_axes([.21, .28, .63, .51])
    ax.axvline(0, color="#8b9ba7", lw=1.4)
    n_by_dataset = {"breast_cancer": 114, "wine": 36, "sst2": 200, "trec": 200}
    selected = []
    for i, dataset in enumerate(NAMES):
        metric = rows[dataset]["paired_group_bootstrap"]["metrics"]["accuracy"]
        estimate = metric["estimate"] * 100
        lo, hi = [x * 100 for x in metric["ci95"]]
        ax.hlines(i, lo, hi, color=TEAL, lw=3)
        ax.plot(estimate, i, "o", color=TEAL, markersize=9)
        ax.plot([lo, hi], [i, i], "|", color=TEAL, markersize=12)
        ax.text(17, i, f"{estimate:+.1f} pp", va="center", fontsize=12, weight="bold", color=INK)
        selected.append({"dataset": dataset, "estimate_pp": estimate, "ci95_pp": [lo, hi]})
    ax.set(xlim=(-16, 16), ylim=(3.55, -.55), xlabel="Accuracy change when the suggestion is included (percentage points)")
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    ax.set_yticks(range(4), [f"{NAMES[d]}\nn = {n_by_dataset[d]}" for d in NAMES], fontsize=12)
    clean_axis(ax)
    fig.text(.24, .81, "Suggestion hurts", fontsize=10, color=MUTED)
    fig.text(.685, .81, "Suggestion helps", fontsize=10, color=MUTED)
    fig.text(.055, .145, "Every interval includes zero. A consistent benefit is not established here.",
             fontsize=14, weight="bold", color=INK)
    fig.text(.055, .062, "Dots show the observed change; lines show paired 95% bootstrap intervals. Zero means no accuracy change.\n"
             "Original primary comparison, all four datasets, four examples per class. Wider intervals mean greater uncertainty.\n"
             "One fixed split. These results do not establish that the two approaches are equivalent.",
             fontsize=10, color=MUTED, linespacing=1.5)
    save(fig, "03_proposal_value", "Including Qwen3's proposed answer changed Jev's observed accuracy by no more than half a "
         "percentage point in either direction in the original primary comparison. The 95% intervals all include zero. "
         "The Wine interval is wide because its test set contains only 36 rows. This does not prove equivalence.",
         {"scope": "All four actual-minus-no-proposal primary contrasts; no secondary overlay", "values": selected,
          "bootstrap_samples": report["bootstrap_samples"], "bootstrap_seed": report["bootstrap_seed"]})


def cost(reports):
    source = find_run(reports, "trec", "gpt-6-astra", "base")
    jev = find_run(reports, "trec", "typesafe/jev-1.13", "direct")
    review = find_run(reports, "trec", "gpt-6-astra+jev_review", "review")
    costs = []
    ids = []
    example_ids = []
    for row in (source, jev, review):
        metadata = json.loads((ROOT / row["source_path"]).read_text())
        example_ids.append(metadata["training_example_ids"])
        path = Path(row["source_path"]).parent / "predictions.jsonl"
        SOURCES.append(str(path))
        predictions = [json.loads(line) for line in (ROOT / path).read_text().splitlines()]
        assert len(predictions) == 200
        ids.append([r["row_id"] for r in predictions])
        values = [Decimal(r["metadata"]["budget"]["reported_usage_estimate"]["usd"])
                  if row["arm"] == "base" else Decimal(r["metadata"]["openrouter"]["reported_cost_usd"])
                  for r in predictions]
        costs.append(sum(values))
    assert ids[0] == ids[1] == ids[2]
    assert example_ids[0] == example_ids[1] == example_ids[2]
    assert costs == [Decimal("1.356350000"), Decimal("0.008723022"), Decimal("0.009218622")]
    displayed = [costs[0], costs[1], costs[0] + costs[2]]
    fig = plt.figure(figsize=(10.5, 7.0), facecolor="white")
    header(fig, "The review was cheap. Accuracy still fell.",
           "TREC | Same 200 questions | Four examples per class | API costs for one pass")
    left = fig.add_axes([.24, .33, .29, .43])
    right = fig.add_axes([.64, .33, .25, .43])
    colors = [BLUE, LIGHT_TEAL, TEAL]
    labels = ["GPT-6 Astra alone", "Jev alone", "Astra + Jev review"]
    for i, (row, money, color) in enumerate(zip((source, jev, review), displayed, colors)):
        acc = row["accuracy"] * 100
        left.barh(i, acc, color=color, height=.58)
        left.text(acc - 3, i, f"{acc:.1f}%", va="center", ha="right", color="white", fontsize=13, weight="bold")
        right.barh(i, float(money), color=color, height=.58)
        right.text(float(money) + .045, i, f"${money:.4f}", va="center", fontsize=12.5, color=INK, weight="bold")
    left.set(ylim=(2.6, -.6), xlim=(0, 100), xlabel="Accuracy")
    left.set_xticks([0, 50, 100])
    left.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    left.set_yticks(range(3), labels, fontsize=12)
    right.set(ylim=(2.6, -.6), xlim=(0, 1.7), xlabel="API cost (USD)")
    right.set_yticks([])
    right.set_xticks([0, .5, 1, 1.5], ["$0", "$0.50", "$1.00", "$1.50"])
    clean_axis(left)
    clean_axis(right)
    fig.text(.055, .205, "Jev review added about $0.0092, or 0.68%, to the Astra estimate.",
             fontsize=14, weight="bold", color=INK)
    fig.text(.055, .066, "Astra: estimated from recorded tokens and dated standard rates. Jev: provider-reported charges.\n"
             "The pipeline total counts both stages once. Local compute, training and engineering were not priced.\n"
             "These are saved experimental costs, not current price quotes or a production savings estimate.",
             fontsize=10.2, color=MUTED, linespacing=1.5)
    save(fig, "04_trec_cost", "For the same 200 TREC questions, Jev alone cost about $0.0087 and scored 85.5%; Astra cost an "
         "estimated $1.3564 and scored 97.0%. Combining them cost about $1.3656 and scored 86.5%. The comparison counts "
         "both pipeline stages once and excludes local compute, training and engineering.",
         {"scope": "TREC four examples per class, same ordered 200 rows", "astra_estimated_usd": str(costs[0]),
          "jev_alone_reported_usd": str(costs[1]), "jev_review_reported_usd": str(costs[2]),
          "chain_reconstructed_usd": str(displayed[2]), "runs": [source["run_id"], jev["run_id"], review["run_id"]]})


def all_matched(reports):
    identities = [("gpt-6-astra", "base", "GPT-6 Astra", BLUE),
                  ("typesafe/jev-1.13", "direct", "Jev", TEAL),
                  ("logistic_regression", "classical", "Logistic regression", GOLD),
                  ("random_forest", "classical", "Random forest", GOLD),
                  ("xgboost", "classical", "XGBoost", GOLD),
                  ("lightgbm", "classical", "LightGBM", GOLD)]
    fig, axes = plt.subplots(2, 2, figsize=(11.25, 9.5), facecolor="white")
    fig.subplots_adjust(left=.19, right=.965, top=.76, bottom=.185, wspace=.62, hspace=.53)
    header(fig, "The data type changed the comparison.",
           "Same new training labels: four per class | All four classical methods shown")
    values = []
    for (dataset, title), ax in zip(NAMES.items(), axes.flat):
        rows = [find_run(reports, dataset, model, arm) for model, arm, _, _ in identities]
        for i, (row, (_, _, _, color)) in enumerate(zip(rows, identities)):
            acc = row["accuracy"] * 100
            ax.barh(i, acc, height=.65, color=color)
            ax.text(102, i, f"{acc:.1f}%", va="center", fontsize=10.5, color=INK, weight="bold")
        n, nclass = rows[0]["n_test"], rows[0]["n_classes"]
        dtype = "Numerical" if dataset in ("wine", "breast_cancer") else "Text"
        ax.set_title(f"{title}  |  {dtype}\n{n} test rows, {4*nclass} training labels", fontsize=12, weight="bold", loc="left", pad=14)
        ax.set(xlim=(0, 122), ylim=(5.6, -.6))
        ax.set_yticks(range(6), [t[2] for t in identities], fontsize=10.5)
        ax.set_xticks([0, 50, 100])
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        clean_axis(ax)
        values.extend({"dataset": dataset, "model": r["model"], "run_id": r["run_id"],
                       "accuracy": r["accuracy"], "n_test": r["n_test"], "train_labels": r["train_labels"]} for r in rows)
    fig.text(.055, .098, "Classical models used numerical features directly and TF-IDF features for text.\n"
             "All methods received the same task-specific labels. LLMs and Jev also bring prior pretraining.\n"
             "One split and fixed recipes; these are few-shot results, not full-training rankings.",
             fontsize=10.8, color=MUTED, linespacing=1.5)
    save(fig, "05_matched_label_context", "With only four training examples per class, classical methods were competitive on "
         "these numerical tasks but much weaker on the text tasks. Text baselines used TF-IDF. Equal numbers of task-specific "
         "labels do not equalize the prior pretraining available to Jev and Astra. These are fixed-recipe, one-split results.",
         {"scope": "All four classical methods, Jev, Astra; all four datasets; four examples per class", "values": values})


def jev_few_shot(reports):
    fig = plt.figure(figsize=(10.5, 6.7), facecolor="white")
    header(fig, "A few examples made a large difference for Jev.",
           "Jev alone | Same held-out rows | No examples versus four examples per class")
    ax = fig.add_axes([.21, .30, .72, .48])
    values = []
    for i, dataset in enumerate(NAMES):
        pair = [find_run(reports, dataset, "typesafe/jev-1.13", "direct", shots) for shots in (0, 4)]
        zero, few = [r["accuracy"] * 100 for r in pair]
        ax.plot([zero, few], [i, i], color="#b8c8d4", lw=4, zorder=1)
        ax.plot(zero, i, "o", ms=11, color=BLUE, zorder=2)
        ax.plot(few, i, "o", ms=11, color=TEAL, zorder=2)
        ax.text(zero, i + .28, f"{zero:.1f}%", ha="center", va="center", color=BLUE, fontsize=12, weight="bold")
        ax.text(few, i - .28, f"{few:.1f}%", ha="center", va="center", color=TEAL, fontsize=12, weight="bold")
        values.append({"dataset": dataset, "zero_accuracy": zero/100, "few_accuracy": few/100,
                       "runs": [r["run_id"] for r in pair]})
    ax.set(xlim=(0, 103), ylim=(3.58, -.58), xlabel="Accuracy")
    ax.set_yticks(range(4), list(NAMES.values()), fontsize=12)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    clean_axis(ax)
    fig.text(.30, .195, "● No examples", fontsize=12, color=BLUE, weight="bold")
    fig.text(.55, .195, "● Four examples per class", fontsize=12, color=TEAL, weight="bold")
    fig.text(.055, .078, "Four per class means 8 labels for Breast Cancer and SST-2, 12 for Wine, and 24 for TREC.\n"
             "Original standalone Jev runs. The examples are part of the prompt; no model weights are updated.\n"
             "One fixed split and example selection. Gains need not repeat with different examples or datasets.",
             fontsize=10.2, color=MUTED, linespacing=1.5)
    save(fig, "06_jev_few_shot", "Jev's standalone accuracy improved after adding four labeled examples per class to the "
         "prompt on each dataset, with the largest observed gains on Wine and TREC. This was in-context learning, not fine-tuning.",
         {"scope": "All four datasets, original standalone Jev zero-shot and four-example-per-class runs", "values": values})


def all_review_conditions(panels):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    models = audit.MODELS
    data = np.zeros((len(models) * 2, 4))
    selected = []
    for i, (model, display) in enumerate(models):
        for j, shots in enumerate((0, 4)):
            for k, dataset in enumerate(NAMES):
                row = next(r for r in panels[dataset] if r["model"] == model and r["shots_per_class"] == shots)
                data[2 * i + j, k] = row["net_accuracy_pp"]
                selected.append(row)
    improved, tied, declined = int((data > 1e-8).sum()), int((abs(data) < 1e-8).sum()), int((data < -1e-8).sum())
    assert (improved, tied, declined) == (32, 3, 13)
    fig = plt.figure(figsize=(10.5, 9.3), facecolor="white")
    header(fig, "Review helped often. It did not help everywhere.",
           "All 48 source-model conditions | Change in accuracy after adding Jev review")
    ax = fig.add_axes([.27, .26, .65, .55])
    cmap = LinearSegmentedColormap.from_list("review", [RED, "#fafafa", TEAL])
    scale = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, cmap=cmap, norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale), aspect="auto")
    for (y, x), value in np.ndenumerate(data):
        color = "white" if abs(value) > scale*.53 else INK
        ax.text(x, y, f"{value:+.1f}" if abs(value)>1e-8 else "0.0", ha="center", va="center", fontsize=12, color=color, weight="bold")
    ax.set_xticks(range(4), ["Breast Cancer\nNumeric", "Wine\nNumeric", "SST-2\nText", "TREC\nText"], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(range(12), [f"{display}\n{'Zero-shot' if shots == 0 else '4 examples/class'}" for _, display in models for shots in (0,4)], fontsize=10)
    ax.tick_params(length=0, pad=10)
    for y in [1.5, 3.5, 5.5, 7.5, 9.5]:
        ax.axhline(y, color="white", lw=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([.40, .19, .38, .018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Accuracy change (percentage points)", fontsize=10, labelpad=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    fig.text(.055, .118, f"{improved} improved   ·   {tied} unchanged   ·   {declined} declined", fontsize=15, weight="bold", color=INK)
    fig.text(.055, .054, "Original full-denominator results. Positive values favor review. Repeated conditions share the same four test sets.\n"
             "These are descriptive changes, not 48 independent replications or a general model ranking.",
             fontsize=10.2, color=MUTED, linespacing=1.5)
    save(fig, "07_all_review_conditions", "Across all 48 original source-model and prompting conditions, Jev review improved "
         "accuracy in 32, tied in three, and reduced it in 13. Cells show percentage-point changes. The conditions reuse four "
         "test sets and are not independent replications.", {"scope": "Complete original 48-condition source-to-review matrix", "values": selected})


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                         "axes.labelcolor": MUTED, "svg.fonttype": "none", "svg.hashsalt": "jev-medium-v1"})
    reports = [json.loads((ROOT / p).read_text()) for p in SOURCES[:2]]
    panels = audit.prepare(reports)
    # Existing reports pin their raw data. Refuse silently changed inputs.
    pins = {p: sha(ROOT / p) for p in SOURCES}
    for report in reports:
        for row in report["runs"]:
            for name, expected in row["artifact_sha256"].items():
                path = Path(row["source_path"]).parent / name
                assert sha(ROOT / path) == expected, path
                pins[str(path)] = expected
    wine(reports)
    trec(reports, panels)
    controls(json.loads((ROOT / SOURCES[2]).read_text()))
    cost(reports)
    all_matched(reports)
    jev_few_shot(reports)
    all_review_conditions(panels)
    for path in SOURCES:
        pins[path] = sha(ROOT / path)
    manifest = {"script_sha256": sha(__file__), "matplotlib_version": matplotlib.__version__,
                "source_sha256": pins, "figures": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.suffix in (".png", ".svg")},
                "captions": CAPTIONS, "chart_data": PROVENANCE}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "CAPTIONS.md").write_text("# Medium figure captions\n\n" + "\n\n".join(f"## {name}\n\n{text}" for name, text in CAPTIONS.items()) + "\n")
    print(OUT)
    for name in CAPTIONS:
        print(name)


if __name__ == "__main__":
    main()
