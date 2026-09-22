"""Run the tabular extension within the expanded, cumulative US$20 ceiling.

Prior text-study ledgers remain immutable at $5.293007 conservatively accounted.
This separately identified extension can reserve at most $14.70, including all
its Jev and OpenAI calls. Reuses the frozen transports, ledger and inference core.
Dry-run by default; credentials enter only through the inherited hidden prompt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import run_openrouter_jev as jev_route

budget = jev_route.budget
ROOT = Path(__file__).resolve().parents[1]
STAGE_CAP = "14.70"
TOTAL_CAP = "20.00"
PRIOR_TOTAL = "5.293007000"
FROZEN_ROUTE_SHA = "12a4e66649a2f9533ba8bb16fd7bd386a0c9dfc808f400b4da2a6df26e5ce89f"
PRIOR = {
    "openai-budget.jsonl": {
        "cap": "7.50", "sha256": "81ed86d640896c5775191fc055979fdde4f0a6b906f79e864596ac7007f8352a",
        "lock_sha256": "a76fae681be49c00a0138b9c3e52274a3cc853fc8f50e27c7917a01defc38278"},
    "jev-budget.jsonl": {
        "cap": "2.50", "sha256": "2cee1170917a008bf750ac077595ead54f8d55081e9a6f3d8f90e7daa4c822e5",
        "lock_sha256": "9dbd526daa535fbaa32584ab840c6d15d87fb655a1c7eea5d200581b3efc8f4d"},
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_prior(root=ROOT):
    """Reject any change to earlier spending; never reset or settle old records."""
    snapshots = {}
    for name, expected in PRIOR.items():
        path = root / "results" / name
        if (file_sha(path) != expected["sha256"] or
                file_sha(path.with_name(path.name + ".lock")) != expected["lock_sha256"]):
            raise budget.GuardError("Prior study ledger/anchor changed; reconcile the cumulative budget before further calls")
        snapshot = budget.Ledger(path, expected["cap"]).snapshot()
        if snapshot["halted"]:
            raise budget.GuardError("Prior study ledger is halted")
        snapshots[name] = snapshot
    total = sum((Decimal(s["charged_or_reserved_usd"]) for s in snapshots.values()), Decimal(0))
    if total != Decimal(PRIOR_TOTAL) or total + Decimal(STAGE_CAP) > Decimal(TOTAL_CAP):
        raise budget.GuardError("Cumulative spending allocation does not fit the authorized US$20")
    return snapshots


class AnchoredLedger:
    """One new ledger plus checked immutable evidence for the earlier study."""
    def __init__(self, ledger, root=ROOT, stop_after=None):
        self.inner, self.root, self.stop_after = ledger, root, stop_after
        self.identity = ledger.identity
        self.new_requests = 0

    def reserve(self, amount, details):
        verify_prior(self.root)
        if self.stop_after is not None and self.new_requests >= self.stop_after:
            raise jev_route.StopRequested("Requested checkpoint reached before another HTTP call")
        ident = self.inner.reserve(amount, details)
        self.new_requests += 1
        return ident

    def result(self, *args, **kwargs):
        return self.inner.result(*args, **kwargs)

    def settle(self, *args, **kwargs):
        return self.inner.settle(*args, **kwargs)

    def snapshot(self):
        return self.inner.snapshot()


def validate_models(path, keys):
    configured = json.loads(Path(path).read_text())
    expected_openai = json.loads((ROOT / "configs/hosted_budget.json").read_text())
    models = {}
    for key in keys:
        if key == "jev_openrouter":
            models[key] = jev_route.validate_config(configured[key])
        elif key in {"openai_economical", "openai_frontier"}:
            if configured[key] != expected_openai[key]:
                raise budget.GuardError("Tabular OpenAI settings must match the recorded text-study presets")
            models[key] = budget.validate_config(configured[key])
        else:
            raise budget.GuardError("Unrecognized tabular hosted model key")
    if not models or len(models) != len(keys):
        raise budget.GuardError("Model keys must be nonempty and unique")
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tabular_hosted.json")
    parser.add_argument("--model-keys", nargs="+", default=["jev_openrouter", "openai_economical", "openai_frontier"])
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--output", type=Path, default=ROOT / "results/tabular/hosted")
    parser.add_argument("--ledger", type=Path, default=ROOT / "results/tabular/api-budget.jsonl")
    parser.add_argument("--max-requests", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--stop-after-new-requests", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--init-ledger", action="store_true")
    parser.add_argument("--prompt-api-key", action="store_true")
    args = parser.parse_args(argv)
    if args.ledger.resolve() != (ROOT / "results/tabular/api-budget.jsonl").resolve():
        raise budget.GuardError("Use the one canonical tabular ledger; alternate ledgers cannot create another allowance")
    if file_sha(jev_route.__file__) != FROZEN_ROUTE_SHA or file_sha(jev_route._BUDGET_PATH) != jev_route.FROZEN_BUDGET_SHA256:
        raise budget.GuardError("Frozen route/ledger source changed")
    if datetime.now(timezone.utc).date().isoformat() != jev_route.VERIFIED_ON:
        raise budget.GuardError("Reverify dated prices before creating a new version of this wrapper")
    if (args.max_requests <= 0 or args.bootstrap_samples < 100 or not args.shots or not set(args.shots) <= {0, 4} or
            len(set(args.shots)) != len(args.shots) or
            (args.stop_after_new_requests is not None and args.stop_after_new_requests <= 0)):
        raise budget.GuardError("Invalid request/checkpoint limit or shot conditions")
    prior = verify_prior()
    models = validate_models(args.config, args.model_keys)
    from tabular_data import load_native_prepared
    from jevbench.prompts import build_prompt, select_examples
    from jevbench.runner import run_model
    from jevbench import providers
    loaded = [load_native_prepared(path) for path in args.data]
    datasets = [item[0] for item in loaded]
    if len({dataset.name for dataset in datasets}) != len(datasets):
        raise budget.GuardError("Repeated dataset job")
    prices = {key: budget.select_price(config) for key, config in models.items() if config["provider"] == "openai"}
    plan = []
    for dataset in datasets:
        if len(dataset.test) > args.max_requests:
            raise budget.GuardError("Full prepared test set exceeds request cap; no test rows are dropped")
        for shots in args.shots:
            examples = select_examples(dataset.train, dataset.labels, shots, 42)
            for key, config in models.items():
                amounts = ([jev_route.RESERVE_NANO] * len(dataset.test) if key == "jev_openrouter" else
                    [budget.reservation(config, prices[key], build_prompt(row, dataset.labels, examples))[0] for row in dataset.test])
                plan.append({"dataset": dataset.name, "model_key": key, "shots_per_class": shots,
                             "requests": len(dataset.test), "fresh_reservations_before_settlement_usd": budget.usd_string(sum(amounts)),
                             "maximum_one_request_reservation_usd": budget.usd_string(max(amounts))})
    summary = {"mode": "execute" if args.execute else "dry_run", "prior_study_conservative_usd": PRIOR_TOTAL,
               "new_stage_cap_usd": STAGE_CAP, "approved_cumulative_cap_usd": TOTAL_CAP,
               "matrix": plan, "fresh_matrix_requests": sum(row["requests"] for row in plan),
               "note": "OpenAI successful usage settles; Jev and failures retain reservations. Plan totals before settlements are not a spend estimate."}
    print(json.dumps(summary, indent=2), flush=True)
    if not args.execute:
        return summary
    ledger = AnchoredLedger(budget.Ledger(args.ledger, STAGE_CAP, initialize=args.init_ledger), stop_after=args.stop_after_new_requests)
    if args.prompt_api_key:
        budget.prompt_credentials(list(models.values()))
    if not all(os.environ.get(config["api_key_env"]) for config in models.values()):
        raise budget.GuardError("Missing selected provider credentials; no HTTP calls made")
    if args.workers > 1:
        if args.stop_after_new_requests is not None:
            raise budget.GuardError("Use a single worker for an exact request checkpoint")
        # Each dataset has one process, keeping global transport patches isolated.
        # Children inherit credentials in memory; they never appear in argv/files.
        from concurrent.futures import ThreadPoolExecutor
        def child(data_path):
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                "--data", str(data_path.resolve()), "--model-keys", *args.model_keys,
                "--shots", *map(str, args.shots), "--output", str(args.output.resolve()),
                "--ledger", str(args.ledger.resolve()), "--max-requests", str(args.max_requests),
                "--bootstrap-samples", str(args.bootstrap_samples), "--execute"]
            return subprocess.run(command, check=False).returncode
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            statuses = list(pool.map(child, args.data))
        if any(statuses):
            raise budget.GuardError("A dataset worker stopped; preserve ledger and partial results before resuming")
        return {"worker_exit_codes": statuses, "budget": ledger.snapshot()}
    original_factory = providers.build_provider
    completed = []
    for dataset in datasets:
        for shots in args.shots:
            for key, config in models.items():
                provenance = {"tabular_wrapper_sha256": file_sha(__file__), "wrapper_sha256": FROZEN_ROUTE_SHA,
                    "ledger_driver_sha256": jev_route.FROZEN_BUDGET_SHA256,
                    "ledger_id": ledger.identity["ledger_id"], "prior_ledgers": PRIOR,
                    "prior_charged_or_reserved_usd": PRIOR_TOTAL, "stage_cap_usd": STAGE_CAP,
                    "aggregate_authorized_usd": TOTAL_CAP, "dataset": dataset.name, "seed": 42,
                    "shots_per_class": shots, "model_key": key}
                if key == "jev_openrouter":
                    provenance.update({"route": "OpenRouter", "endpoint": jev_route.ENDPOINT,
                        "price_verified_on": jev_route.VERIFIED_ON, "price_source": jev_route.PRICE["source"],
                        "response_model_allowlist": sorted(jev_route.ALLOWED_RESPONSE_MODELS),
                        "native_protocol": "frozen_JevClassifier_Choice", "reserve_checks_included_in_latency": True})
                    def factory(_):
                        return jev_route.GuardedOpenRouterJev(config, ledger, provenance)
                else:
                    provenance.update({"pricing_sha256": budget.sha(prices[key]), "verified_on": prices[key]["verified_on"]})
                    def factory(_):
                        return budget.BudgetedProvider(original_factory(config), config, prices[key], ledger, provenance)
                try:
                    with patch.object(providers, "build_provider", factory):
                        run = run_model(dataset, {**config, "budget_guard": provenance}, args.output,
                            shots=shots, seed=42, allow_paid=True, max_requests=args.max_requests,
                            bootstrap_samples=args.bootstrap_samples)
                except jev_route.StopRequested:
                    stopped = {"status": "checkpoint", "new_requests": ledger.new_requests, "budget": ledger.snapshot()}
                    print(json.dumps(stopped), flush=True)
                    return stopped
                completed.append(run["run_id"])
                print(json.dumps({"run": run["run_id"], "accuracy": run["metrics"]["accuracy"],
                                  "failures": run["metrics"]["n_failures"], "budget": ledger.snapshot()}), flush=True)
    return {"completed": completed, "budget": ledger.snapshot()}


if __name__ == "__main__":
    try:
        main()
    except (budget.BudgetStop, budget.GuardError) as exc:
        print(f"Stopped: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
