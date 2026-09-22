"""Expand cached-label Jev review with matched zero/four-shot source stages.

The earlier four review arms and their ledgers stay immutable. This driver has
one new US$5 allocation, at most 1,500 new Jev calls, and a cumulative US$25 cap.
Source manifests are frozen individually before use; no source model is called.
Missing local source runs remain pending. Dry-run is the default.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path

import run_numeric_jev_review as base
from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import _identity, digest, environment, finish_run, save_json
from jevbench.types import Prediction
from tabular_data import load_native_prepared

budget, route = base.budget, base.jev_route
ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / "results/numeric_expansion"
OUTPUT, LOCAL, SOURCES_DIR = AREA / "review", AREA / "local", AREA / "sources"
LEDGER = AREA / "review-budget.jsonl"
STAGE_CAP, PRIOR_TOTAL, TOTAL_CAP = "5.00", "19.325827700", "25.00"
MAX_REQUESTS, SEED = 1500, 42
BASE_REVIEW_SHA = "6282eb8b74c89ebbaf24d34e7ba29e06b4dc8cab9c700b0b3620fc613f6d3653"
HISTORICAL_SUMMARY_SHA = "7e3df4dc6a04aa0a88068e9639e3d04d95f86bcae38d3086ad12d23b4f11a7e2"
PRIOR = {**base.PRIOR,
    "results/numeric_decisions/review-budget.jsonl": {"cap": "0.90",
        "sha256": "19d362e5e9894894f60014c580da7f6ed0771020864842e516831cb35a09bc6a",
        "lock_sha256": "cf1dab0ccde5136c6a28df0297ab057f264ac6c9c529c6bf65bfa3cf6efc0a49"}}
MODELS = {
    "qwen_small": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "provider": "hf", "revision": "7ae557604adf67be50417f59c2c2f167def9a775"},
    "qwen_main": {"model": "Qwen/Qwen3-4B-Instruct-2507", "provider": "hf", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554"},
    "luna": {"model": "gpt-5.6-luna", "provider": "openai"},
    "astra": {"model": "gpt-6-astra", "provider": "openai"},
    "smollm2": {"model": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "provider": "hf", "revision": "31b70e2e869a7173562077fd711b654946d38674", "new_local": True},
    "granite": {"model": "ibm-granite/granite-3.3-2b-instruct", "provider": "hf", "revision": "707f574c62054322f6b5b04b6d075f0a8f05e0f0", "new_local": True},
}
DATASETS = ("breast_cancer", "wine")
SHOTS = (0, 4)
REUSED_KEYS = {"qwen_main": "qwen3_4b", "astra": "astra"}
FILE_NAMES = ("run.json", "predictions.jsonl", "test_manifest.json")


def require(test, message):
    if not test:
        raise budget.GuardError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_transport():
    base.verify_transport()
    require(file_sha(base.__file__) == BASE_REVIEW_SHA, "Original review implementation changed")
    require(environment()["source_sha256"] == base.FROZEN_CORE_SHA, "Frozen benchmark inference core changed")


def verify_prior(root=ROOT):
    snapshots = {}
    for relative, expected in PRIOR.items():
        path = root / relative
        require(file_sha(path) == expected["sha256"] and
                file_sha(path.with_name(path.name + ".lock")) == expected["lock_sha256"],
                "Prior study ledger/lock changed; reconcile the cumulative budget")
        snapshots[relative] = budget.Ledger(path, expected["cap"]).snapshot()
        require(not snapshots[relative]["halted"], "A prior study ledger is halted")
    total = sum((Decimal(value["charged_or_reserved_usd"]) for value in snapshots.values()), Decimal(0))
    require(total == Decimal(PRIOR_TOTAL) and total + Decimal(STAGE_CAP) <= Decimal(TOTAL_CAP),
            "Expanded allocation exceeds the cumulative authorization")
    return snapshots


def condition_key(dataset, model_key, shots):
    require(dataset in DATASETS and model_key in MODELS and type(shots) is int and shots in SHOTS,
            "Unsupported dataset/model/shot condition")
    return f"{dataset}__{model_key}__k{shots}"


def is_reused(model_key, shots):
    return model_key in REUSED_KEYS and shots == 4


def historical_source(dataset, model_key, shots):
    path = ROOT / "results/TABULAR_COMPARISON.json"
    require(file_sha(path) == HISTORICAL_SUMMARY_SHA, "Pinned historical summary changed")
    summary = json.loads(path.read_text())
    matches = [row for row in summary["runs"] if row["dataset"] == dataset and
        row["model"] == MODELS[model_key]["model"] and row["method"] == ("zero_shot" if shots == 0 else "few_shot") and
        row["train_per_class"] == shots]
    require(len(matches) == 1, "Historical source condition is missing or ambiguous")
    row = matches[0]
    return {"model": row["model"], "provider": MODELS[model_key]["provider"],
        "path": Path(row["source_path"]).parent.as_posix(), "hashes": row["artifact_sha256"]}


def discover_local_source(dataset, model_key, shots):
    candidates, incomplete = [], []
    for path in sorted(LOCAL.glob("*/run.json")):
        record = json.loads(path.read_text())
        config = record.get("config", {})
        if record.get("dataset") == dataset and config.get("model") == MODELS[model_key]["model"] and config.get("shots_per_class") == shots:
            (candidates if record.get("status") == "complete" else incomplete).append(path)
    require(len(candidates) <= 1, "Multiple completed local sources match; never select using test scores")
    if not candidates:
        return None
    path = candidates[0]
    require(all(path.with_name(name).is_file() for name in FILE_NAMES), "Completed local source lacks final artifacts")
    return {"model": MODELS[model_key]["model"], "provider": "hf",
        "path": path.parent.relative_to(ROOT).as_posix(),
        "hashes": {name: file_sha(path.with_name(name)) for name in FILE_NAMES}}


def validate_source(dataset, model_key, shots, source):
    condition_key(dataset.name, model_key, shots)
    spec = MODELS[model_key]
    require(source["model"] == spec["model"] and source["provider"] == spec["provider"], "Frozen source model differs")
    directory = (ROOT / source["path"]).resolve()
    require(directory.is_relative_to(ROOT.resolve()) and set(source["hashes"]) == set(FILE_NAMES), "Source path/hash schema differs")
    if spec.get("new_local"):
        require(directory.parent == LOCAL.resolve(), "New source must reside in the canonical local artifact directory")
    else:
        require(source == historical_source(dataset.name, model_key, shots), "Source differs from pinned historical artifacts")
    require(all(file_sha(directory / name) == expected for name, expected in source["hashes"].items()), "Frozen source artifact changed")
    record = json.loads((directory / "run.json").read_text())
    predictions = [Prediction(**json.loads(line)) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    examples = select_examples(dataset.train, dataset.labels, shots, SEED)
    expected = {"status": "complete", "dataset": dataset.name, "labels": dataset.labels, "seed": SEED,
        "method": "zero_shot" if shots == 0 else "few_shot", "manifest_sha256": digest(dataset.manifest),
        "dataset_manifest": dataset.manifest, "test_ids_sha256": digest([row.id for row in dataset.test]),
        "training_example_ids": [row.id for row in examples], "implementation_sha256": base.FROZEN_CORE_SHA,
        "run_id": directory.name}
    require(all(record.get(key) == value for key, value in expected.items()), "Source dataset/training/test identity differs")
    config = record["config"]
    require(config.get("model") == spec["model"] and config.get("provider") == spec["provider"] and
            config.get("shots_per_class") == shots and not config.get("adapter_path"), "Source model or label budget differs")
    if "revision" in spec:
        require(config.get("revision") == spec["revision"], "Source model revision differs")
    require(json.loads((directory / "test_manifest.json").read_text()) == base.expected_test_manifest(dataset), "Source test manifest differs")
    require([prediction.row_id for prediction in predictions] == [row.id for row in dataset.test], "Source predictions are not a complete ordered test set")
    for prediction in predictions:
        require(isinstance(prediction.metadata, dict), "Source prediction metadata is malformed")
        if prediction.error:
            require(isinstance(prediction.error, str) and prediction.label is None, "Malformed source failure")
        else:
            require(base.valid_proposal(prediction, len(dataset.labels)), "Source class ID is outside the bounded labels")
    measured = evaluate(dataset.test, predictions, len(dataset.labels))
    require(all(record["metrics"].get(key) == value for key, value in measured.items()), "Source metrics differ from raw predictions")
    if spec.get("new_local"):
        # This API must validate config, renderer/source identities and per-row
        # rendered/template hashes without constructing or loading an HF model.
        local = importlib.import_module("run_expanded_numeric_local")
        audited_record, audited_predictions = local.audit_source_run(directory / "run.json", dataset, model_key, shots)
        require(audited_record == record and [asdict(value) for value in audited_predictions] == [asdict(value) for value in predictions],
                "Local renderer audit returned different source records")
    return record, predictions, examples


def source_manifest(dataset, model_key, shots, source, record):
    return {"schema_version": 1, "dataset": dataset.name, "model_key": model_key, "shots_per_class": shots,
        "source": source, "source_model_config": record["config"], "source_config_sha256": digest(record["config"]),
        "dataset_manifest_sha256": digest(dataset.manifest), "test_ids_sha256": record["test_ids_sha256"],
        "training_example_ids": record["training_example_ids"], "implementation_sha256": record["implementation_sha256"]}


def load_source(dataset, model_key, shots, *, freeze=False):
    slug = condition_key(dataset.name, model_key, shots)
    path = SOURCES_DIR / f"{slug}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        source = saved["source"]
    else:
        saved = None
        source = (discover_local_source(dataset.name, model_key, shots) if MODELS[model_key].get("new_local") else
                  historical_source(dataset.name, model_key, shots))
    if source is None:
        return None
    record, predictions, examples = validate_source(dataset, model_key, shots, source)
    manifest = source_manifest(dataset, model_key, shots, source, record)
    if saved is not None:
        require(saved == manifest, "Individual frozen source manifest changed")
    elif freeze:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            file.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
    return {"dataset": dataset, "model_key": model_key, "shots": shots, "source": source,
        "record": record, "predictions": predictions, "examples": examples,
        "manifest_path": path.relative_to(ROOT).as_posix(), "manifest_sha256": file_sha(path) if path.exists() else None}


def validate_reused(job, config):
    dataset, key = job["dataset"], REUSED_KEYS[job["model_key"]]
    source, original, proposals, examples = base.load_source(dataset, key)
    old_ledger = budget.Ledger(base.LEDGER, base.STAGE_CAP)
    directory = base.OUTPUT / f"{dataset.name}__{key}__jev-review"
    record = json.loads((directory / "run.json").read_text())
    identity, _ = base.make_identity(dataset, key, source, original, examples, config,
        old_ledger.identity["ledger_id"], record["bootstrap_samples"])
    record, saved = base.load_checkpoint(directory, identity, dataset, source, original, proposals, examples)
    require(record is not None and record["status"] == "complete", "Prior review is not complete and cannot be reused")
    base.validate_reservations(base.AnchoredLedger(old_ledger), dataset.name, key, saved, identity)
    measured = evaluate(dataset.test, saved, len(dataset.labels))
    require(all(record["metrics"].get(key) == value for key, value in measured.items()), "Reused review metrics differ")
    return {"run_id": record["run_id"], "path": directory.relative_to(ROOT).as_posix(),
        "hashes": {name: file_sha(directory / name) for name in FILE_NAMES}}


class AnchoredLedger:
    def __init__(self, ledger, root=ROOT, stop_after=None):
        self.inner, self.root, self.stop_after = ledger, root, stop_after
        self.identity, self.new_requests = ledger.identity, 0

    def reserve(self, amount, details):
        verify_transport()
        verify_prior(self.root)
        require(amount == route.RESERVE_NANO, "Unexpected Jev reservation")
        events, _ = self.inner._read()
        reservations = [event for event in events if event["type"] == "reserve"]
        outcomes = [event["details"].get("outcome") for event in events if event["type"] == "result"]
        require(len(outcomes) < 3 or not all(value in {"prediction_error", "raised"} for value in outcomes[-3:]),
                "Three consecutive errors halted this expansion ledger")
        if len(reservations) >= MAX_REQUESTS:
            raise budget.BudgetStop("Expansion is limited to 1500 new Jev calls")
        key = details.get("dataset"), details.get("proposal_key"), details.get("row_id")
        require(not any((event["details"].get("dataset"), event["details"].get("proposal_key"), event["details"].get("row_id")) == key
                        for event in reservations), "This expanded review row was already reserved; no automatic retry")
        if self.stop_after is not None and self.new_requests >= self.stop_after:
            raise route.StopRequested("Requested checkpoint reached before another API call")
        ident = self.inner.reserve(amount, details)
        self.new_requests += 1
        return ident

    def result(self, *args, **kwargs):
        return self.inner.result(*args, **kwargs)

    def snapshot(self):
        return self.inner.snapshot()


def make_identity(job, config, ledger_id, bootstrap_samples):
    dataset, model_key, shots = job["dataset"], job["model_key"], job["shots"]
    require(job["manifest_sha256"] is not None and file_sha(ROOT / job["manifest_path"]) == job["manifest_sha256"],
            "Freeze this individual source manifest before paid review")
    source = job["source"]
    provenance = {"expansion_wrapper_sha256": file_sha(__file__), "base_review_wrapper_sha256": BASE_REVIEW_SHA,
        "wrapper_sha256": base.FROZEN_ROUTE_SHA, "ledger_driver_sha256": route.FROZEN_BUDGET_SHA256,
        "ledger_id": ledger_id, "prior_ledgers": PRIOR, "prior_charged_or_reserved_usd": PRIOR_TOTAL,
        "stage_cap_usd": STAGE_CAP, "aggregate_authorized_usd": TOTAL_CAP,
        "dataset": dataset.name, "proposal_key": f"{model_key}_k{shots}", "model_key": model_key,
        "seed": SEED, "shots_per_class": shots, "source_shots_per_class": shots,
        "route": "OpenRouter", "endpoint": route.ENDPOINT, "price": route.PRICE,
        "source_run_id": job["record"]["run_id"], "source_files_sha256": source["hashes"],
        "source_manifest_path": job["manifest_path"], "source_manifest_sha256": job["manifest_sha256"],
        "review_prompt_version": "cached-label-proposal-review-v1"}
    identity = _identity(dataset, "cached_label_jev_review", {**config, "shots_per_class": shots,
        "source_model": source["model"], "source_run_id": job["record"]["run_id"],
        "source_shots_per_class": shots, "budget_guard": provenance}, SEED, job["examples"])
    identity.update({"bootstrap_samples": bootstrap_samples, "proposal": source,
        "method_label": "cached-label proposal + Jev review", "new_proposal_generation": False,
        "source_manifest_sha256": job["manifest_sha256"]})
    return identity, provenance


def run_review(job, config, ledger, *, bootstrap_samples=1000):
    dataset, model_key, shots = job["dataset"], job["model_key"], job["shots"]
    require(not is_reused(model_key, shots), "Existing review must be reused, never charged again")
    identity, provenance = make_identity(job, config, ledger.identity["ledger_id"], bootstrap_samples)
    directory = OUTPUT / (condition_key(dataset.name, model_key, shots) + "__jev-review")
    source, original, proposals, examples = job["source"], job["record"], job["predictions"], job["examples"]
    record, previous = base.load_checkpoint(directory, identity, dataset, source, original, proposals, examples)
    base.validate_reservations(ledger, dataset.name, provenance["proposal_key"], previous, identity)
    if record is not None and record["status"] == "complete":
        measured = evaluate(dataset.test, previous, len(dataset.labels))
        require(all(record["metrics"].get(key) == value for key, value in measured.items()), "Completed expanded review metrics differ")
        return record
    record = record or {**identity, "run_id": directory.name, "started_at": budget.utc_now(),
        "environment": environment(), "dataset_manifest": dataset.manifest,
        "execution": {"concurrency": 1, "automatic_retries": 0, "max_total_review_requests": MAX_REQUESTS,
            "proposal_source": "cached_labels_only", "source_and_reviewer_shots_matched": True,
            "latency_scope": "Jev review only; source runtime is stored in the source run"}}
    record["status"] = "running"
    record.setdefault("execution_sessions", []).append({"started_at": budget.utc_now(), "cached_rows": len(previous)})
    save_json(directory / "run.json", record)
    provider = route.GuardedOpenRouterJev(config, ledger, provenance)
    with (directory / "predictions.jsonl").open("a") as file:
        for row, proposal in zip(dataset.test[len(previous):], proposals[len(previous):]):
            if base.valid_proposal(proposal, len(dataset.labels)):
                prompt = base.build_review_prompt(row, dataset.labels, examples, proposal.label)
                prediction = provider.predict(row, dataset.labels, prompt)
                prediction.metadata["review_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            else:
                prediction = Prediction(row.id, None, error="source_proposal_failed: Jev review not called",
                    metadata={"probability_kind": "unavailable", "jev_review_called": False})
            prediction.metadata["proposal"] = base.proposal_metadata(source, original, proposal)
            file.write(json.dumps(asdict(prediction), allow_nan=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
            previous.append(prediction)
            if len(previous) >= 3 and all(prediction.error for prediction in previous[-3:]):
                record.update(status="stopped_after_three_consecutive_errors", n_predictions=len(previous))
                save_json(directory / "run.json", record)
                raise budget.GuardError("Stopped after three consecutive pipeline errors")
            if len(previous) % 25 == 0:
                print(json.dumps({"condition": directory.name, "completed_rows": len(previous), "test_rows": len(dataset.test)}), flush=True)
    return finish_run(dataset, previous, directory, record, bootstrap_samples)


def audit_completed_review(path, dataset):
    """Validate a completed expanded arm without constructing a provider."""
    path = Path(path)
    record = json.loads(path.read_text())
    guard = record["config"]["budget_guard"]
    model_key, shots = guard["model_key"], guard["shots_per_class"]
    require(not is_reused(model_key, shots), "Previously measured review cannot appear as a new expanded run")
    expected_dir = OUTPUT / (condition_key(dataset.name, model_key, shots) + "__jev-review")
    require(path.resolve() == (expected_dir / "run.json").resolve(), "Expanded review path differs from its condition")
    job = load_source(dataset, model_key, shots)
    require(job is not None and job["manifest_sha256"] is not None, "Review lacks a frozen source")
    config = route.validate_config(json.loads((ROOT / "configs/tabular_hosted.json").read_text())["jev_openrouter"])
    ledger = AnchoredLedger(budget.Ledger(LEDGER, STAGE_CAP))
    identity, _ = make_identity(job, config, ledger.identity["ledger_id"], record["bootstrap_samples"])
    record, predictions = base.load_checkpoint(path.parent, identity, dataset, job["source"], job["record"], job["predictions"], job["examples"])
    require(record is not None and record["status"] == "complete", "Expanded review is incomplete")
    base.validate_reservations(ledger, dataset.name, guard["proposal_key"], predictions, identity)
    measured = evaluate(dataset.test, predictions, len(dataset.labels))
    require(all(record["metrics"].get(key) == value for key, value in measured.items()), "Saved expanded review scores differ")
    for prediction in predictions:
        if not prediction.error:
            require(prediction.metadata.get("resolved_model") in route.ALLOWED_RESPONSE_MODELS,
                    "Expanded review returned an unapproved model")
    return record, predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-keys", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--shots", nargs="+", type=int, choices=list(SHOTS), default=list(SHOTS))
    parser.add_argument("--freeze-sources", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--init-ledger", action="store_true")
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--stop-after-new-requests", type=int)
    args = parser.parse_args(argv)
    require(all(len(values) == len(set(values)) for values in (args.model_keys, args.datasets, args.shots)) and
            args.bootstrap_samples >= 100 and (args.stop_after_new_requests is None or args.stop_after_new_requests > 0),
            "Duplicate condition or invalid checkpoint/bootstrap limit")
    verify_transport()
    verify_prior()
    config = route.validate_config(json.loads((ROOT / "configs/tabular_hosted.json").read_text())["jev_openrouter"])
    jobs, plan, reused, pending = [], [], [], []
    for dataset_name in args.datasets:
        dataset, _ = load_native_prepared(ROOT / "data/tabular-full" / dataset_name)
        require(dataset.name == dataset_name and all(feature["kind"] == "numeric" for feature in dataset.manifest["tabular"]["features"]),
                "Only the exact numerical datasets may be reviewed")
        for model_key in args.model_keys:
            for shots in args.shots:
                condition = condition_key(dataset.name, model_key, shots)
                job = load_source(dataset, model_key, shots, freeze=args.freeze_sources)
                if job is None:
                    pending.append(condition)
                    plan.append({"condition": condition, "status": "pending_source", "test_rows": len(dataset.test), "fresh_jev_calls": None})
                    continue
                if is_reused(model_key, shots):
                    prior_review = validate_reused(job, config)
                    reused.append(prior_review)
                    plan.append({"condition": condition, "status": "reused", "test_rows": len(dataset.test), "fresh_jev_calls": 0, "review": prior_review})
                    continue
                count = sum(base.valid_proposal(prediction, len(dataset.labels)) for prediction in job["predictions"])
                jobs.append(job)
                plan.append({"condition": condition, "status": "ready" if job["manifest_sha256"] else "needs_source_freeze",
                    "source_run_id": job["record"]["run_id"], "source_manifest": job["manifest_path"],
                    "test_rows": len(dataset.test), "fresh_jev_calls": count,
                    "source_failures_without_jev_call": len(dataset.test) - count})
    count = sum(row["fresh_jev_calls"] or 0 for row in plan)
    require(count <= MAX_REQUESTS and Decimal(budget.usd_string(count * route.RESERVE_NANO)) <= Decimal(STAGE_CAP),
            "Selected expansion matrix exceeds its call or spending cap")
    summary = {"mode": "execute" if args.execute else "dry_run", "method": "cached-label proposal + Jev review",
        "matrix": plan, "fresh_jev_calls_ready": count, "fresh_ready_reservations_usd": budget.usd_string(count * route.RESERVE_NANO),
        "reused_prior_review_runs": len(reused), "pending_source_conditions": pending,
        "prior_conservative_usd": PRIOR_TOTAL, "new_stage_cap_usd": STAGE_CAP, "cumulative_authorized_usd": TOTAL_CAP,
        "maximum_expansion_calls": MAX_REQUESTS, "maximum_expansion_reservations_usd": budget.usd_string(MAX_REQUESTS * route.RESERVE_NANO),
        "note": "Both stages use the same zero or four demonstrations/class. No source LLM calls are made. Missing source models remain pending."}
    print(json.dumps(summary, indent=2), flush=True)
    if not args.execute:
        return summary
    require(not pending, "Selected sources are pending; select completed --model-keys or finish their local inference first")
    require(all(job["manifest_sha256"] for job in jobs), "Use --freeze-sources for every requested condition before execution")
    if not jobs:
        return {"completed": [], "reused": reused, "new_calls": 0}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.with_name("review-execution.lock").open("a+") as execution_lock:
        try:
            fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise budget.GuardError("Another expanded review worker is already running") from None
        ledger = AnchoredLedger(budget.Ledger(LEDGER, STAGE_CAP, initialize=args.init_ledger), stop_after=args.stop_after_new_requests)
        if args.prompt_api_key:
            budget.prompt_credentials([config])
        require(bool(os.environ.get(config["api_key_env"])), "Missing Jev credential; no API calls made")
        completed = []
        for job in jobs:
            try:
                record = run_review(job, config, ledger, bootstrap_samples=args.bootstrap_samples)
            except route.StopRequested:
                checkpoint = {"status": "checkpoint", "new_calls": ledger.new_requests, "budget": ledger.snapshot()}
                print(json.dumps(checkpoint), flush=True)
                return checkpoint
            completed.append(record["run_id"])
            print(json.dumps({"completed": record["run_id"], "accuracy": record["metrics"]["accuracy"],
                "failures": record["metrics"]["n_failures"], "budget": ledger.snapshot()}), flush=True)
        return {"completed": completed, "reused": reused, "budget": ledger.snapshot()}


if __name__ == "__main__":
    try:
        main()
    except (budget.BudgetStop, budget.GuardError) as exc:
        print(f"Stopped: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
