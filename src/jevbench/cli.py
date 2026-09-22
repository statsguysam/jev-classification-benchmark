import argparse
import json
from pathlib import Path

from .runner import cap_text, compare_runs, run_classical, run_model, save_json


def main():
    parser = argparse.ArgumentParser(description="Reproducible classification benchmark. See docs/PROTOCOL.md.")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Download immutable public dataset snapshots and freeze splits")
    prep.add_argument("datasets", nargs="+", choices=["sst2", "imdb", "ag_news", "trec", "banking77"])
    prep.add_argument("--cache-dir", type=Path, default=Path("data"))
    prep.add_argument("--output", type=Path, default=Path("data/prepared"))
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--train-limit", type=int)
    prep.add_argument("--validation-limit", type=int)
    prep.add_argument("--test-limit", type=int)
    prep.add_argument("--max-text-chars", type=int, default=2000)
    classical = commands.add_parser("classical", help="Run TF-IDF supervised baselines")
    classical.add_argument("--models", nargs="+", default=["majority", "logistic_regression", "linear_svc", "multinomial_nb"])
    classical.add_argument("--train-per-class", type=int)
    model = commands.add_parser("model", help="Run Jev, frontier or local HF models")
    model.add_argument("--config", type=Path, required=True)
    model.add_argument("--model-key", required=True)
    model.add_argument("--shots", type=int, default=0, help="Demonstrations PER CLASS")
    model.add_argument("--allow-paid", action="store_true")
    model.add_argument("--max-requests", type=int, default=1000)
    for command in (classical, model):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, default=Path("results/runs"))
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--bootstrap-samples", type=int, default=1000)
    lora = commands.add_parser("train-lora", help="Train a response-only PEFT adapter on open weights")
    lora.add_argument("--data", type=Path, required=True)
    lora.add_argument("--model", required=True)
    lora.add_argument("--output", type=Path, required=True)
    lora.add_argument("--revision")
    lora.add_argument("--seed", type=int, default=42)
    lora.add_argument("--train-per-class", type=int)
    lora.add_argument("--epochs", type=int, default=3)
    lora.add_argument("--max-length", type=int, default=2048)
    lora.add_argument("--max-steps", type=int)
    lora.add_argument("--device", default="auto")
    lora.add_argument("--load-in-4bit", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--results", type=Path, default=Path("results/runs"))
    report.add_argument("--output", type=Path, default=Path("results/REPORT.md"))
    compare = commands.add_parser("compare")
    compare.add_argument("run_a", type=Path)
    compare.add_argument("run_b", type=Path)
    compare.add_argument("--samples", type=int, default=2000)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--allow-unequal-training", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare_dataset, save_prepared
        for name in args.datasets:
            dataset = prepare_dataset(name, args.cache_dir, seed=args.seed, train_limit=args.train_limit, test_limit=args.test_limit, validation_limit=args.validation_limit)
            cap_text(dataset, args.max_text_chars)
            dataset.manifest.pop("prepared_path", None)
            save_prepared(dataset, args.output / name)
            print(json.dumps({"dataset": name, "path": str(args.output / name), "sizes": {split: len(getattr(dataset, split)) for split in ("train", "validation", "test")}}), flush=True)
    elif args.command in {"classical", "model", "train-lora"}:
        from .data import load_prepared
        dataset = load_prepared(args.data)
        if args.command == "classical":
            for name in args.models:
                result = run_classical(dataset, name, args.output, seed=args.seed, train_per_class=args.train_per_class, bootstrap_samples=args.bootstrap_samples)
                print(json.dumps({"run": result["run_id"], "accuracy": result["metrics"]["accuracy"], "macro_f1": result["metrics"]["macro_f1"]}), flush=True)
        elif args.command == "model":
            configs = json.loads(args.config.read_text())
            result = run_model(dataset, configs[args.model_key], args.output, shots=args.shots, seed=args.seed, allow_paid=args.allow_paid, max_requests=args.max_requests, bootstrap_samples=args.bootstrap_samples)
            print(json.dumps({"run": result["run_id"], "accuracy": result["metrics"]["accuracy"], "macro_f1": result["metrics"]["macro_f1"]}), flush=True)
        else:
            from .lora import train_lora
            metadata = train_lora(dataset, args.model, args.output, revision=args.revision, seed=args.seed, train_per_class=args.train_per_class, epochs=args.epochs, max_length=args.max_length, max_steps=args.max_steps, device=args.device, load_in_4bit=args.load_in_4bit)
            print(json.dumps(metadata, indent=2))
    elif args.command == "report":
        from .report import make_report
        print(make_report(args.results, args.output))
    else:
        result = compare_runs(args.run_a, args.run_b, samples=args.samples, allow_unequal_training=args.allow_unequal_training)
        if args.output:
            save_json(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
