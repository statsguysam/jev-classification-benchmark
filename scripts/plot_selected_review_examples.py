"""Create a LinkedIn-sized figure from two explicitly selected first-attempt examples.

Offline, complete-data only. Every displayed value is derived from reports that
must match fresh raw-data audits. No retries, model calls or producer changes.
Run only after the original numeric/text phases have stopped and both primary
reports are current. These audits do not read the separate control execution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_review_outcomes as audited

OUT = ROOT / "results/completion_20260923/figures/selected_examples"
SELECTION = (("wine", "Qwen/Qwen2.5-0.5B-Instruct", 4), ("trec", "gpt-6-astra", 4))
FILENAMES = ("when_second_model_helps.png", "when_second_model_helps.svg")


def bar(row, label, kind, n, training_labels):
    audited.require(row["status"] == "complete" and row["n_test"] == n
                    and row["train_per_class"] == 4 and row["train_labels"] == training_labels,
                    "Selected comparison must use the same full test fold and four training examples per class")
    correct = row["accuracy"] * n
    audited.require(math.isfinite(correct) and math.isclose(correct, round(correct), abs_tol=1e-8)
                    and 0 <= round(correct) <= n and type(row["n_failures"]) is int
                    and 0 <= row["n_failures"] <= n - round(correct), "Invalid selected count or failure total")
    return {"run_id": row["run_id"], "label": label, "kind": kind, "n_correct": round(correct),
            "n_test": n, "accuracy": row["accuracy"], "n_failures": row["n_failures"], "training_labels": training_labels}


def prepare(reports):
    """Select by fixed condition identity, never by sorting observed performance."""
    panels = audited.prepare(reports)
    selected = []
    for report, (dataset, model, shots) in zip(reports, SELECTION):
        matches = [row for row in panels[dataset] if row["model"] == model and row["shots_per_class"] == shots]
        audited.require(len(matches) == 1, "Selected source→review condition is unavailable")
        pair = matches[0]
        n, classes = audited.DATASETS[dataset][2:]
        training_labels = shots * classes
        by_id = {row["run_id"]: row for row in report["runs"]}
        source, review = by_id[pair["source_run_id"]], by_id[pair["review_run_id"]]
        bars = [bar(source, pair["display_model"], "source", n, training_labels),
                bar(review, "Source → Jev", "review", n, training_labels)]
        if dataset == "wine":
            for model_id, arm, label, kind in (("typesafe/jev-1.13", "direct", "Jev alone", "direct"),
                                              ("logistic_regression", "classical", "Logistic regression", "classical")):
                candidates = [row for row in report["runs"] if row["dataset"] == dataset and row["model"] == model_id
                              and row["arm"] == arm and row["train_per_class"] == shots]
                audited.require(len(candidates) == 1, "Selected matched-label native reference is unavailable")
                bars.append(bar(candidates[0], label, kind, n, training_labels))
            direct_pairs = [p for p in report["comparisons"] if p["kind"] == "review_minus_jev_alone"
                            and p["a"] == review["run_id"] and p["b"] == bars[2]["run_id"]]
            audited.require(len(direct_pairs) == 1 and direct_pairs[0]["equal_new_label_budget"],
                            "Direct Jev reference must have audited matching examples")
        selected.append({**pair, "title": audited.DATASETS[dataset][0], "n_classes": classes,
                         "training_labels": training_labels, "bars": bars})
    return selected


def net_text(panel):
    lo, hi = panel["paired_ci95_pp"]
    return f"Review − source: {panel['net_accuracy_pp']:+.1f} pp  (paired 95% CI {lo:+.1f} to {hi:+.1f})"


def transition_text(panel):
    return (f"{panel['corrected']} corrected  ·  {panel['harmed_wrong_label']} harmed by a wrong label  ·  "
            f"{panel['harmed_failure']} harmed by failure")


def direct_text(panel):
    reviewed, direct = panel["bars"][1:3]
    if reviewed["n_correct"] == direct["n_correct"]:
        return "The reviewed pipeline matched Jev alone's accuracy here."
    delta = 100 * (reviewed["accuracy"] - direct["accuracy"])
    return f"Reviewed pipeline − Jev alone: {delta:+.1f} percentage points here."


def render(selected, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    colors = {"source": "#687891", "review": "#087f8c", "direct": "#64aaa5", "classical": "#b98535"}
    ink, muted, background = "#132b40", "#516477", "#f3f6fa"
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "jev-selected-first-attempt-examples-v1", "text.color": ink})
    fig = plt.figure(figsize=(8, 8), facecolor=background)
    fig.text(.055, .951, "When does a second model help?", fontsize=22.5, weight="bold")
    fig.text(.055, .913, "Two selected first-attempt examples · 4 training examples per class", fontsize=10.5, color=muted)
    for y, height in ((.49, .38), (.174, .296)):
        fig.add_artist(FancyBboxPatch((.042, y), .916, height, boxstyle="round,pad=0.008,rounding_size=0.018",
                                     transform=fig.transFigure, facecolor="white", edgecolor="#dde5ee", linewidth=.8, zorder=0))
    for index, panel in enumerate(selected):
        numeric = panel["dataset"] == "wine"
        title_y, subtitle_y, rect = ((.836, .805, [.29, .603, .63, .166]) if numeric
                                     else (.434, .404, [.29, .292, .63, .093]))
        fig.text(.068, title_y, f"{index + 1:02d}  {panel['title']} · {'numerical data' if numeric else 'text classification'}",
                 fontsize=15, weight="bold")
        fig.text(.068, subtitle_y, f"{panel['n_classes']} classes · N = {panel['n_test']} · {panel['training_labels']} training labels for each method shown",
                 fontsize=9.6, color=muted)
        ax = fig.add_axes(rect, zorder=1)
        for y, row in enumerate(panel["bars"]):
            ax.barh(y, 100 * row["accuracy"], height=.59, color=colors[row["kind"]], zorder=2)
            ax.text(104, y, f"{row['n_correct']}/{row['n_test']}", va="center", fontsize=12.5, weight="bold", color=ink)
        ax.set_xlim(0, 131)
        ax.set_ylim(len(panel["bars"]) - .47, -.53)
        ax.set_yticks(range(len(panel["bars"])), [r["label"] for r in panel["bars"]], fontsize=10)
        ax.set_xticks([0, 50, 100], ["0%", "50%", "100%"], fontsize=8, color=muted)
        ax.grid(axis="x", color="#e9eef4", linewidth=.7, zorder=0)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        if numeric:
            fig.text(.068, .567, net_text(panel), fontsize=10, weight="bold")
            fig.text(.068, .540, transition_text(panel), fontsize=9.4, color=muted)
            fig.text(.068, .512, direct_text(panel), fontsize=10, color=ink)
        else:
            fig.text(.068, .255, net_text(panel), fontsize=10, weight="bold")
            fig.text(.068, .227, transition_text(panel), fontsize=9.4, color=muted)
            fig.text(.068, .200, f"Failed source rows: {panel['source_failures']} · failed review rows (including upstream skips): "
                     f"{panel['review_failures_including_upstream_skips']}", fontsize=9.2, color=muted)
    fig.text(.055, .136, "Different datasets and denominators. The classical-ML comparison is Wine only.", fontsize=9.3, color=muted)
    fig.text(.055, .113, "Exploratory: familiar public holdouts; intervals condition on fixed rows, prompts and responses.", fontsize=8.8, color=muted)
    fig.text(.055, .091, "Unadjusted intervals. Failures count as incorrect in full N. One fixed split.", fontsize=9, color=muted)
    fig.text(.055, .069, "Selected examples are not an overall ranking. See all 48 conditions in the repository.", fontsize=9, color=muted)
    fig.text(.055, .039, "github.com/statsguysam/jev-classification-benchmark", fontsize=10, weight="bold")
    fig.text(.055, .019, "Original results preserved · Retry recoveries excluded · Values and source hashes in MANIFEST.json", fontsize=8, color=muted)
    out = Path(out)
    fig.savefig(out / FILENAMES[0], dpi=200, facecolor=background, metadata={"Software": "matplotlib"})
    fig.savefig(out / FILENAMES[1], facecolor=background, metadata={"Date": None})
    plt.close(fig)
    return matplotlib.__version__


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    reports, pins = audited.load_audited()
    pins[Path(audited.__file__).relative_to(ROOT).as_posix()] = audited.file_sha(audited.__file__)
    selected = prepare(reports)
    with tempfile.TemporaryDirectory(prefix="jev-selected-review-examples-") as temp:
        temp = Path(temp)
        version = render(selected, temp)
        audited.verify_pins(pins)
        manifest = {"schema_version": 1, "status": "complete_audited", "scope": "Two selected first-attempt examples, not an aggregate ranking",
            "selection_basis": "Fixed named conditions chosen for explanation; no outcome sorting in this script",
            "selection": [list(key) for key in SELECTION], "outcome_numbers_hardcoded": False,
            "retry_results_used": False, "pixels": [1600, 1600], "script_sha256": audited.file_sha(__file__),
            "matplotlib_version": version, "evidence_sha256": pins, "examples": selected,
            "figures": {name: audited.file_sha(temp / name) for name in FILENAMES}}
        (temp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in (*FILENAMES, "MANIFEST.json"):
            (args.output_dir / name).write_bytes((temp / name).read_bytes())
    print(args.output_dir)


if __name__ == "__main__":
    main()
