"""Plot every completed source→Jev condition, preserving first-attempt failures.

Offline only. Both saved 68-condition reports must equal fresh raw-data audits.
This script never runs inference, updates reports, or substitutes retry results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
SOURCES = (("results/numeric_expansion/COMPARISON.json", "summarize_expanded_numeric"),
           ("results/text_extension/COMPARISON.json", "summarize_text_extension"))
OUT = ROOT / "results/completion_20260923/figures"
MODELS = (("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B"),
          ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2 1.7B"),
          ("ibm-granite/granite-3.3-2b-instruct", "Granite 3.3 2B"),
          ("Qwen/Qwen3-4B-Instruct-2507", "Qwen3 4B"),
          ("gpt-5.6-luna", "GPT-5.6 Luna"), ("gpt-6-astra", "GPT-6 Astra"))
DATASETS = {"breast_cancer": ("Breast Cancer", "Numeric · binary", 114, 2),
            "wine": ("Wine", "Numeric · 3 classes", 36, 3),
            "sst2": ("SST-2", "Text · binary", 200, 2),
            "trec": ("TREC", "Text · 6 classes", 200, 6)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_complete(report, datasets):
    """Reject a partial inventory before invoking any raw prediction collector."""
    require(report.get("status") == "complete" and report.get("expected_runs") == report.get("complete_runs") == 68
            and report.get("expected_comparisons") == report.get("complete_comparisons") == 72,
            "Both studies must contain 68/68 audited runs and 72/72 comparisons")
    runs, comparisons = report["runs"], report["comparisons"]
    require(len(runs) == 68 and len(comparisons) == 72 and all(r["status"] == "complete" for r in runs + comparisons),
            "Incomplete or duplicated report inventory")
    require(set(report["datasets"]) == set(datasets) and {r["dataset"] for r in runs} == set(datasets),
            "Unexpected dataset inventory")
    require(len({r["run_id"] for r in runs}) == 68, "Duplicate run identity")
    require(len({(r["kind"], r["a"], r["b"]) for r in comparisons}) == 72, "Duplicate comparison identity")
    for arm, expected in (("base", 24), ("review", 24), ("direct", 4), ("classical", 16)):
        require(sum(r["arm"] == arm for r in runs) == expected, "Incomplete condition family")
    require(type(report["bootstrap_samples"]) is int and report["bootstrap_samples"] >= 100,
            "Invalid bootstrap specification")


def prepare(reports):
    """Pure structural/metric checks; input reports are audited by load_audited."""
    require(len(reports) == 2, "Require numerical and text reports together")
    groups = (("breast_cancer", "wine"), ("sst2", "trec"))
    for report, dataset_names in zip(reports, groups):
        validate_complete(report, dataset_names)
    panels = {}
    for report, dataset_names in zip(reports, groups):
        by_id = {r["run_id"]: r for r in report["runs"]}
        pairs = [p for p in report["comparisons"] if p["kind"] == "review_minus_source"]
        require(len(pairs) == 24, "Require all 24 source→review contrasts in each study")
        for dataset in dataset_names:
            panel = []
            n = DATASETS[dataset][2]
            for model, name in MODELS:
                for shots in (0, 4):
                    bases = [r for r in report["runs"] if r["dataset"] == dataset and r["arm"] == "base"
                             and r["model"] == model and r["train_per_class"] == shots]
                    require(len(bases) == 1, "Missing or duplicate source model/shot condition")
                    base = bases[0]
                    matches = [p for p in pairs if p["b"] == base["run_id"]]
                    require(len(matches) == 1, "Missing or duplicated paired review")
                    pair = matches[0]
                    review = by_id[pair["a"]]
                    require(review["arm"] == "review" and review["source_run_id"] == base["run_id"]
                            and review["model"] == model + "+jev_review" and review["dataset"] == dataset
                            and pair["dataset"] == dataset and review["train_per_class"] == shots
                            and base["n_test"] == review["n_test"] == n, "Review source or denominator mismatch")
                    counts = pair["transitions"]
                    fields = ("wrong_to_correct", "correct_to_wrong", "both_correct", "both_wrong",
                              "correct_to_wrong_label", "correct_to_failure", "source_failure_rows", "review_stage_failure_rows")
                    require(all(type(counts[k]) is int and 0 <= counts[k] <= n for k in fields), "Invalid transition count")
                    corrected, wrong, failed = (counts[k] for k in ("wrong_to_correct", "correct_to_wrong_label", "correct_to_failure"))
                    require(counts["correct_to_wrong"] == wrong + failed
                            and sum(counts[k] for k in fields[:4]) == n, "Transition partition does not cover the full test set")
                    delta = 100 * (corrected - wrong - failed) / n
                    require(counts["net_correct_change"] == corrected - wrong - failed
                            and math.isclose(counts["accuracy_delta_pp"], delta, abs_tol=1e-10), "Net transition count differs")
                    require(math.isclose(base["accuracy"], (counts["both_correct"] + wrong + failed) / n, abs_tol=1e-12)
                            and math.isclose(review["accuracy"], (counts["both_correct"] + corrected) / n, abs_tol=1e-12),
                            "Accuracy must retain failures in the full denominator")
                    metric = pair["paired_bootstrap"]["metrics"]["accuracy"]
                    lo, hi = metric["ci95"]
                    require(all(type(v) in (int, float) and math.isfinite(v) for v in (lo, hi, metric["estimate"]))
                            and -1 <= lo <= hi <= 1 and math.isclose(100 * metric["estimate"], delta, abs_tol=1e-10),
                            "Paired interval or accuracy estimate differs")
                    require(base["n_failures"] == counts["source_failure_rows"]
                            and review["n_failures"] == counts["source_failure_rows"] + counts["review_stage_failure_rows"]
                            and failed <= counts["review_stage_failure_rows"], "Failure accounting differs from the frozen no-fallback pipeline")
                    panel.append({"dataset": dataset, "model": model, "display_model": name, "shots_per_class": shots,
                        "n_test": n, "source_run_id": base["run_id"], "review_run_id": review["run_id"],
                        "source_accuracy": base["accuracy"], "review_accuracy": review["accuracy"],
                        "corrected": corrected, "harmed_wrong_label": wrong, "harmed_failure": failed,
                        "source_failures": base["n_failures"], "review_failures_including_upstream_skips": review["n_failures"],
                        "review_stage_failures": counts["review_stage_failure_rows"], "net_accuracy_pp": delta,
                        "paired_ci95_pp": [100 * lo, 100 * hi], "bootstrap_samples": report["bootstrap_samples"]})
            panels[dataset] = panel
    require(sum(map(len, panels.values())) == 48, "The chart must contain all 48 review conditions")
    return panels


def verify_pins(pins, root=ROOT):
    for name, expected in pins.items():
        path = root / name
        require(not path.is_symlink() and path.is_file() and file_sha(path) == expected, f"Audited evidence changed: {name}")


def load_audited(root=ROOT):
    root = Path(root)
    reports = [json.loads((root / name).read_text()) for name, _ in SOURCES]
    # Validate BOTH saved statuses before either collector can inspect outcomes.
    prepare(reports)
    pins = {name: file_sha(root / name) for name, _ in SOURCES}
    for report, (_, module_name) in zip(reports, SOURCES):
        module = importlib.import_module(module_name)
        source_path = Path(module.__file__).resolve().relative_to(ROOT).as_posix()
        pins[source_path] = file_sha(root / source_path)
        require(report == module.collect(root=root, samples=report["bootstrap_samples"]),
                "Saved report differs from fresh raw-data audit; regenerate both summaries before plotting")
        for row in report["runs"]:
            folder = Path(row["source_path"]).parent
            for filename, expected in row["artifact_sha256"].items():
                key = (folder / filename).as_posix()
                require(key not in pins or pins[key] == expected, "Conflicting original artifact identity")
                pins[key] = expected
    verify_pins(pins, root)
    return reports, pins


def render(panels, out):
    """Render only a validated aggregate; matplotlib performs no data collection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    colors = {"correct": "#007e87", "wrong": "#b74e58", "failure": "#e39c49", "ink": "#17293b", "muted": "#506174"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none",
        "svg.hashsalt": "jev-all-review-outcomes-v1", "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "text.color": colors["ink"], "axes.labelcolor": colors["ink"]})
    fig, axes = plt.subplots(2, 2, figsize=(19, 14.5), facecolor="white")
    fig.subplots_adjust(left=.125, right=.845, top=.805, bottom=.18, wspace=.81, hspace=.41)
    fig.text(.038, .966, "When Jev reviews an LLM: corrections and new errors", fontsize=25, weight="bold")
    fig.text(.038, .931, "All 48 source→review conditions · Six models · Zero-shot and four examples per class · First attempts", fontsize=13)
    fig.legend(handles=[Patch(facecolor=colors["correct"], label="Incorrect → correct"),
        Patch(facecolor=colors["wrong"], label="Correct → wrong label"),
        Patch(facecolor=colors["failure"], hatch="////", edgecolor=colors["wrong"], label="Correct → failed review"),
        Line2D([], [], marker="D", color=colors["ink"], markersize=5, label="Net change + paired 95% interval")],
        loc="upper left", bbox_to_anchor=(.033, .904), ncol=4, frameon=False, fontsize=10)
    fig.text(.038, .858, "Bars: percentage of the full test set corrected (right) or harmed (left). Row annotation: corrected / wrong / failed; net pp.",
             fontsize=10, color=colors["muted"])
    maximum = max(max(100 * r["corrected"] / r["n_test"], 100 * (r["harmed_wrong_label"] + r["harmed_failure"]) / r["n_test"],
                      *(abs(v) for v in r["paired_ci95_pp"])) for rows in panels.values() for r in rows)
    limit = max(10, math.ceil((maximum + 2) / 10) * 10)
    for ax, (dataset, rows) in zip(axes.flat, panels.items()):
        title, kind, n, classes = DATASETS[dataset]
        for y, row in enumerate(rows):
            corrected, wrong, failed = (100 * row[key] / n for key in ("corrected", "harmed_wrong_label", "harmed_failure"))
            ax.barh(y, corrected, height=.55, color=colors["correct"], alpha=.76, zorder=2)
            ax.barh(y, -wrong, height=.55, color=colors["wrong"], alpha=.76, zorder=2)
            ax.barh(y, -failed, left=-wrong, height=.55, color=colors["failure"], hatch="////",
                    edgecolor=colors["wrong"], linewidth=.6, zorder=2)
            ax.hlines(y, *row["paired_ci95_pp"], color=colors["ink"], linewidth=1, zorder=3)
            ax.plot(row["net_accuracy_pp"], y, "D", color=colors["ink"], markersize=3.5, zorder=4)
            annotation = f"{row['corrected']:>3} / {row['harmed_wrong_label']:>3} / {row['harmed_failure']:>2}   {row['net_accuracy_pp']:+.1f}"
            ax.text(1.025, y, annotation, transform=ax.get_yaxis_transform(), va="center", fontsize=9, family="DejaVu Sans Mono")
        labels = [f"{r['display_model']} · {'zero' if r['shots_per_class'] == 0 else '4/class'}" for r in rows]
        ax.set_yticks(range(12), labels)
        ax.set_ylim(11.7, -.8)
        ax.set_xlim(-limit, limit)
        ax.axvline(0, color=colors["ink"], linewidth=1, zorder=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, symmetric=True))
        ax.grid(axis="x", color="#e5eaf0", linewidth=.7, zorder=0)
        ax.tick_params(axis="y", length=0, pad=8, labelsize=9)
        ax.set_xlabel("Percentage points of test-set accuracy", fontsize=9, labelpad=7)
        ax.set_title(f"{title}  |  N = {n}\n{kind} · few-shot = {4 * classes} training labels", loc="left", pad=21, fontsize=12, weight="bold")
        ax.text(1.025, 1.018, "C / W / F    net pp", transform=ax.transAxes, fontsize=8.5, color=colors["muted"])
        for boundary in (1.5, 3.5, 5.5, 7.5, 9.5):
            ax.axhline(boundary, color="#edf0f4", linewidth=.8, zorder=0)
    samples = sorted({r["bootstrap_samples"] for rows in panels.values() for r in rows})
    fig.text(.038, .126, "Failures stay in the full N, including upstream failures that prevented review. Hatching isolates damage to an originally correct decision;\n"
        "other failed rows remain incorrect. Retry recoveries are excluded. Exact counts, accuracies and failure totals are recorded in MANIFEST.json.",
        fontsize=10, color=colors["muted"], linespacing=1.6)
    fig.text(.038, .070, f"Exploratory, unadjusted paired group-bootstrap intervals ({'/'.join(map(str, samples))} resamples; seed 42). One fixed split and prompt recipe per condition.\n"
        "Models retain their original prompt rendering. This compares two-stage pipelines; it does not isolate bounded decoding on the same LLM. No pooled winner.",
        fontsize=9.5, color=colors["muted"], linespacing=1.6)
    fig.text(.038, .023, "github.com/statsguysam/jev-classification-benchmark · Numerical and text COMPARISON.json · All first-attempt source→Jev conditions", fontsize=9)
    out = Path(out)
    for ext in ("png", "svg"):
        fig.savefig(out / f"review_outcomes.{ext}", dpi=200, facecolor="white", metadata={"Date": None} if ext == "svg" else {"Software": "matplotlib"})
    plt.close(fig)
    return matplotlib.__version__


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    reports, pins = load_audited()
    panels = prepare(reports)
    # Stage outside the results tree; no partial publication figure is emitted.
    with tempfile.TemporaryDirectory(prefix="jev-review-outcomes-") as temp:
        temp = Path(temp)
        version = render(panels, temp)
        verify_pins(pins)
        manifest = {"schema_version": 1, "status": "complete_audited", "data_scope": "first-attempt source→Jev; all 48 conditions",
            "retry_results_used": False, "frozen_producers_modified": False,
            "script_sha256": file_sha(__file__), "matplotlib_version": version, "evidence_sha256": pins,
            "conditions": [r for rows in panels.values() for r in rows],
            "figures": {name: file_sha(temp / name) for name in ("review_outcomes.png", "review_outcomes.svg")}}
        (temp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("review_outcomes.png", "review_outcomes.svg", "MANIFEST.json"):
            (args.output_dir / name).write_bytes((temp / name).read_bytes())
    print(args.output_dir)


if __name__ == "__main__":
    main()
