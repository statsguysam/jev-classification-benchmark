"""Explicit plan/execute driver; hosted execution requires opt-in and a matrix cap."""
import argparse
import json
from pathlib import Path
from jevbench.data import load_prepared
from jevbench.runner import run_model, save_json

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
parser.add_argument("--model-keys", nargs="+", required=True)
parser.add_argument("--datasets", nargs="+", default=["sst2", "imdb", "ag_news", "trec"])
parser.add_argument("--data-root", type=Path, default=Path("data/pilot"))
parser.add_argument("--output", type=Path, default=Path("results/models"))
parser.add_argument("--shots", nargs="+", type=int, default=[0, 1, 4, 8])
parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 87])
parser.add_argument("--bootstrap-samples", type=int, default=1000)
parser.add_argument("--execute", action="store_true", help="Without this, print plan only")
parser.add_argument("--allow-paid", action="store_true")
parser.add_argument("--max-total-requests", type=int, default=0)
args = parser.parse_args()
configs = json.loads(args.config.read_text())
datasets = {name: load_prepared(args.data_root / name) for name in args.datasets}
jobs = []
for name, dataset in datasets.items():
    for key in args.model_keys:
        config = configs[key]
        if config.get("adapter_path"):
            raise ValueError("Use the single-run CLI for dataset-specific adapters")
        for shots in args.shots:
            # Zero-shot has no example selection; evaluate only once, fixed seed42.
            for seed in ([42] if shots == 0 else args.seeds):
                jobs.append({"dataset": name, "model_key": key, "shots_per_class": shots,
                             "seed": seed, "test_requests": len(dataset.test)})
total = sum(job["test_requests"] for job in jobs)
plan = {"runs": len(jobs), "test_requests_upper_bound": total,
        "request_cap_is_not_a_dollar_limit": True, "jobs": jobs}
print(json.dumps(plan, indent=2))
if args.execute:
    if total > args.max_total_requests:
        raise ValueError(f"Matrix upper bound is {total} requests, above explicit cap {args.max_total_requests}")
    paid = any(configs[key]["provider"] not in {"hf", "huggingface"} and not configs[key].get("local", False) for key in args.model_keys)
    if paid and not args.allow_paid:
        raise ValueError("Hosted matrix requires --allow-paid; no requests made")
    save_json(args.output / "matrix_plan.json", plan)
    for job in jobs:
        record = run_model(datasets[job["dataset"]], configs[job["model_key"]], args.output,
                           shots=job["shots_per_class"], seed=job["seed"], allow_paid=args.allow_paid,
                           max_requests=job["test_requests"], bootstrap_samples=args.bootstrap_samples)
        print(json.dumps({"completed": record["run_id"], "macro_f1": record["metrics"]["macro_f1"]}), flush=True)
