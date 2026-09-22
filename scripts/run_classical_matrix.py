#!/usr/bin/env python3
"""Run the same prepared splits through the complete classical baseline matrix.

Example:
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/run_classical_matrix.py \
        --data-root data/pilot --output results/pilot --budgets 4 --include-full

Use --budgets 1 4 8 --seeds 13 42 87 for a larger study. Full means every row in
that prepared training split; it does not imply the complete original corpus.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from jevbench.classical import CLASSICAL_MODELS
from jevbench.data import load_prepared
from jevbench.runner import run_classical, save_json

DEFAULT_DATASETS = ["sst2", "imdb", "ag_news", "trec", "banking77"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/pilot"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot"))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", choices=CLASSICAL_MODELS, default=list(CLASSICAL_MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--budgets", type=int, nargs="+", default=[4], help="Labeled training examples per class")
    parser.add_argument("--include-full", action="store_true", help="Also use all prepared training rows, with validation model selection")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args(argv)
    if any(budget < 1 for budget in args.budgets):
        parser.error("--budgets must be positive")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    budgets = list(dict.fromkeys(args.budgets)) + ([None] if args.include_full else [])
    seeds = list(dict.fromkeys(args.seeds))
    # Fail early on unavailable/corrupt datasets, before fitting any model.
    datasets = [load_prepared(args.data_root / name) for name in dict.fromkeys(args.datasets)]
    for dataset in datasets:
        class_sizes = [sum(row.label == label for row in dataset.train) for label in range(len(dataset.labels))]
        if max(args.budgets) > min(class_sizes):
            parser.error(f"{dataset.name} has only {min(class_sizes)} training rows in its smallest class")
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "data_root": str(args.data_root), "output": str(args.output),
        "datasets": args.datasets, "models": args.models, "seeds": seeds,
        "budgets_per_class": budgets, "bootstrap_samples": args.bootstrap_samples,
        "full_definition": "all prepared training rows; may be capped relative to original corpus",
        "runs": [],
    }
    summary_path = args.output / "classical_matrix_summary.json"
    save_json(summary_path, summary)
    total = len(datasets) * len(args.models) * len(seeds) * len(budgets)
    index = 0
    for dataset in datasets:
        for seed in seeds:
            for budget in budgets:
                for model in args.models:
                    index += 1
                    track = f"{budget}/class" if budget is not None else "full-prepared"
                    print(f"[{index}/{total}] {dataset.name} | {model} | {track} | seed={seed}", flush=True)
                    record = run_classical(dataset, model, args.output, seed=seed,
                                           train_per_class=budget, bootstrap_samples=args.bootstrap_samples)
                    metrics, training = record["metrics"], record["training"]
                    summary["runs"].append({
                        "run_id": record["run_id"], "dataset": dataset.name, "model": model,
                        "seed": seed, "train_per_class": budget, "training_rows": training["training_rows"],
                        "n_test": metrics["n_test"], "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"], "failure_rate": metrics["failure_rate"],
                        "wall_time_s": record["wall_time_s"], "status": record["status"],
                    })
                    save_json(summary_path, summary)
                    print(f"  accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
                          f"n_train={training['training_rows']} n_test={metrics['n_test']} "
                          f"wall_s={record['wall_time_s']:.2f}", flush=True)
    summary.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
    save_json(summary_path, summary)
    print(f"Completed {len(summary['runs'])} runs. Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
