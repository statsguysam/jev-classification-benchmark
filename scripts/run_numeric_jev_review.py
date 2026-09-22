"""Cached-label proposal + Jev review on the two frozen numerical datasets.

The cached Qwen3/Astra label is the only new state given to Jev. This is not a
reasoning cascade or a same-base decoding ablation. The original numerical row
and the same four training demonstrations per class remain in the prompt.
Dry-run by default; this module never reads credentials at import or during plan.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import run_openrouter_jev as jev_route
from jevbench.metrics import evaluate
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import _identity, digest, environment, finish_run, save_json
from jevbench.types import Prediction
from tabular_data import load_native_prepared

budget = jev_route.budget
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/numeric_decisions/review"
LEDGER = ROOT / "results/numeric_decisions/review-budget.jsonl"
STAGE_CAP, PRIOR_TOTAL, TOTAL_CAP = "0.90", "18.522115700", "20.00"
MAX_REQUESTS = 300
SEED, SHOTS = 42, 4
FROZEN_ROUTE_SHA = "12a4e66649a2f9533ba8bb16fd7bd386a0c9dfc808f400b4da2a6df26e5ce89f"
FROZEN_CORE_SHA = "d5547ccde4224e182653315d81c0a631c789bbe294ad7bcf95ef0cddd25ce608"
PRIOR = {
    "results/openai-budget.jsonl": {"cap": "7.50",
        "sha256": "81ed86d640896c5775191fc055979fdde4f0a6b906f79e864596ac7007f8352a",
        "lock_sha256": "a76fae681be49c00a0138b9c3e52274a3cc853fc8f50e27c7917a01defc38278"},
    "results/jev-budget.jsonl": {"cap": "2.50",
        "sha256": "2cee1170917a008bf750ac077595ead54f8d55081e9a6f3d8f90e7daa4c822e5",
        "lock_sha256": "9dbd526daa535fbaa32584ab840c6d15d87fb655a1c7eea5d200581b3efc8f4d"},
    "results/tabular/api-budget.jsonl": {"cap": "14.70",
        "sha256": "e829c35af5e4bb0630c327f2e4d9efe4e1ded399f161e25d87c04a9001d84b1c",
        "lock_sha256": "2f2b85695440d144a8dc10484cc5515a71aacc3fb529c607b5162e6854cc8a89"},
}
SOURCES = {
    ("breast_cancer", "qwen3_4b"): {
        "model": "Qwen/Qwen3-4B-Instruct-2507", "provider": "hf",
        "path": "results/tabular/colab/breast_cancer__Qwen3-4B-Instruct-2507__cd3baa813fe3",
        "hashes": {"run.json": "a5d870f25fb6c13fd66bd686e77b5ca60cf5c7f1c43991b21297a325eea2146e",
            "predictions.jsonl": "e5c1b86c06a40de1724e263df2849e46ff24c8957df163d1424b4008a1c58917",
            "test_manifest.json": "d3445c1ee998cbebc299755c9c4284295416ad8a1f4165cfad80ddf85b987685"}},
    ("breast_cancer", "astra"): {
        "model": "gpt-6-astra", "provider": "openai",
        "path": "results/tabular/hosted/breast_cancer__gpt-6-astra__6be368014170",
        "hashes": {"run.json": "bb1c82612d9c4787d057d78be96ea225ebe592738ee8090a27b08d08963a1176",
            "predictions.jsonl": "872826ba524654e0e4bdabccca79dc69e9403e884bf5ac9a29407d600705b177",
            "test_manifest.json": "d3445c1ee998cbebc299755c9c4284295416ad8a1f4165cfad80ddf85b987685"}},
    ("wine", "qwen3_4b"): {
        "model": "Qwen/Qwen3-4B-Instruct-2507", "provider": "hf",
        "path": "results/tabular/colab/wine__Qwen3-4B-Instruct-2507__cea3e01ea8d0",
        "hashes": {"run.json": "91378022c1e86e476bd4f3b59a301d411d804d50bed3ca8ccc7ba744c25bfb43",
            "predictions.jsonl": "b9b11638881c31a9853035c1803a4daf61cbf82f83c136ba57494463f16607d4",
            "test_manifest.json": "edbfa3c757d7142e13a35ba40b53c590587e3e591508d70161dd433f881a5c76"}},
    ("wine", "astra"): {
        "model": "gpt-6-astra", "provider": "openai",
        "path": "results/tabular/hosted/wine__gpt-6-astra__d5c2863bd80f",
        "hashes": {"run.json": "c3162e3b5defb51b2d0fe24d9d05a5e131c67f2ff7ce1f0dfa02077256999622",
            "predictions.jsonl": "dbe5ff17012aec89bf5588be1493217451e25607eb725b6a422fedda986156ab",
            "test_manifest.json": "edbfa3c757d7142e13a35ba40b53c590587e3e591508d70161dd433f881a5c76"}},
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_transport():
    if (file_sha(jev_route.__file__) != FROZEN_ROUTE_SHA or
            file_sha(jev_route._BUDGET_PATH) != jev_route.FROZEN_BUDGET_SHA256):
        raise budget.GuardError("Frozen Jev route or budget source changed")
    if datetime.now(timezone.utc).date().isoformat() != jev_route.VERIFIED_ON:
        raise budget.GuardError("Reverify the dated Jev price and route before new calls")


def verify_prior(root=ROOT):
    snapshots = {}
    for relative, expected in PRIOR.items():
        path = root / relative
        if (file_sha(path) != expected["sha256"] or
                file_sha(path.with_name(path.name + ".lock")) != expected["lock_sha256"]):
            raise budget.GuardError("Prior ledger/lock changed; reconcile before any new calls")
        snapshots[relative] = budget.Ledger(path, expected["cap"]).snapshot()
        if snapshots[relative]["halted"]:
            raise budget.GuardError("A prior ledger is halted")
    total = sum((Decimal(item["charged_or_reserved_usd"]) for item in snapshots.values()), Decimal(0))
    if total != Decimal(PRIOR_TOTAL) or total + Decimal(STAGE_CAP) > Decimal(TOTAL_CAP):
        raise budget.GuardError("Cumulative review allowance exceeds the authorized total")
    return snapshots


def expected_test_manifest(dataset):
    return {"dataset": dataset.name, "labels": dataset.labels,
        "manifest_sha256": digest(dataset.manifest),
        "rows": [{"id": row.id, "label": row.label,
            "text_sha256": hashlib.sha256(row.text.encode()).hexdigest()} for row in dataset.test]}


def valid_proposal(prediction, n_classes):
    return not prediction.error and type(prediction.label) is int and 0 <= prediction.label < n_classes


def validate_source(dataset, source, record, predictions, test_manifest):
    """Audit alignment and identities before constructing any paid request."""
    examples = select_examples(dataset.train, dataset.labels, SHOTS, SEED)
    expected = {"status": "complete", "dataset": dataset.name, "labels": dataset.labels,
        "seed": SEED, "method": "few_shot", "manifest_sha256": digest(dataset.manifest),
        "dataset_manifest": dataset.manifest, "test_ids_sha256": digest([r.id for r in dataset.test]),
        "training_example_ids": [r.id for r in examples], "implementation_sha256": FROZEN_CORE_SHA,
        "run_id": Path(source["path"]).name}
    if any(record.get(key) != value for key, value in expected.items()):
        raise budget.GuardError("Proposal run dataset, training/test rows, seed or provenance differs")
    config = record.get("config", {})
    if (config.get("model") != source["model"] or config.get("provider") != source["provider"] or
            config.get("shots_per_class") != SHOTS or config.get("adapter_path")):
        raise budget.GuardError("Only the exact cached four-per-class proposal models are permitted")
    if test_manifest != expected_test_manifest(dataset):
        raise budget.GuardError("Proposal test manifest differs from the complete frozen numerical data")
    if [p.row_id for p in predictions] != [r.id for r in dataset.test]:
        raise budget.GuardError("Proposal predictions must align exactly with every ordered test row")
    for prediction in predictions:
        if prediction.error:
            if not isinstance(prediction.error, str) or prediction.label is not None:
                raise budget.GuardError("Malformed failed proposal")
        elif not valid_proposal(prediction, len(dataset.labels)):
            raise budget.GuardError("Proposal has an invalid class ID")
    measured = evaluate(dataset.test, predictions, len(dataset.labels))
    for key in ("accuracy", "macro_f1", "n_failures", "n"):
        if key in measured and measured[key] != record["metrics"].get(key):
            raise budget.GuardError("Cached proposal score differs from its predictions")
    return examples


def load_source(dataset, proposal_key, root=ROOT):
    if (dataset.name, proposal_key) not in SOURCES:
        raise budget.GuardError("Unsupported numerical dataset/proposal model")
    source = SOURCES[dataset.name, proposal_key]
    directory = root / source["path"]
    if any(file_sha(directory / name) != sha for name, sha in source["hashes"].items()):
        raise budget.GuardError("Pinned cached proposal files changed")
    record = json.loads((directory / "run.json").read_text())
    predictions = [Prediction(**json.loads(line)) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    manifest = json.loads((directory / "test_manifest.json").read_text())
    examples = validate_source(dataset, source, record, predictions, manifest)
    return source, record, predictions, examples


def build_review_prompt(row, labels, examples, proposal_label):
    """Uses only serialized features, train demonstrations, and cached class ID.

    Evaluation row.id and row.label are never interpolated. No prediction
    confidence, ground truth, free-form rationale or source-model identity enters.
    """
    if type(proposal_label) is not int or not 0 <= proposal_label < len(labels):
        raise budget.GuardError("A valid cached class ID is required for review")
    base = build_prompt(row, labels, examples).removesuffix("\nClass ID:")
    return (base + "\nCached proposal from a separate model: class ID " + str(proposal_label) + "."
        "\nThe proposal may be incorrect. Classify the original numerical row in Final text. "
        "Use that row's numerical features and the labeled examples above to make the final decision. "
        "You may keep or override the proposal. Return only the numeric class ID."
        "\nClass ID:")


def proposal_metadata(source, source_record, prediction):
    return {"source_run_id": source_record["run_id"], "source_path": source["path"],
        "source_model": source["model"], "source_files_sha256": source["hashes"],
        "source_row_id": prediction.row_id, "source_label": prediction.label,
        "source_prediction_sha256": digest(asdict(prediction)),
        "source_failed": bool(prediction.error), "new_proposal_generation": False}


class AnchoredLedger:
    """Frozen prior spend plus one never-settled, maximum-300-request ledger."""
    def __init__(self, ledger, root=ROOT, stop_after=None):
        self.inner, self.root, self.stop_after = ledger, root, stop_after
        self.identity = ledger.identity
        self.new_requests = 0

    def reserve(self, amount, details):
        verify_transport()
        verify_prior(self.root)
        if amount != jev_route.RESERVE_NANO:
            raise budget.GuardError("Unexpected per-request reservation")
        events, _ = self.inner._read()
        outcomes = [event["details"].get("outcome") for event in events if event["type"] == "result"]
        if len(outcomes) >= 3 and all(outcome in {"prediction_error", "raised"} for outcome in outcomes[-3:]):
            raise budget.GuardError("Three consecutive provider errors halted the review ledger")
        reservations = [event for event in events if event["type"] == "reserve"]
        if len(reservations) >= MAX_REQUESTS:
            raise budget.BudgetStop("The entire review is limited to 300 new Jev calls")
        key = (details.get("dataset"), details.get("proposal_key"), details.get("row_id"))
        if any((event["details"].get("dataset"), event["details"].get("proposal_key"),
                event["details"].get("row_id")) == key for event in reservations):
            raise budget.GuardError("This review row was already reserved; no automatic retry or duplicate charge")
        if self.stop_after is not None and self.new_requests >= self.stop_after:
            raise jev_route.StopRequested("Requested checkpoint reached before another HTTP call")
        ident = self.inner.reserve(amount, details)
        self.new_requests += 1
        return ident

    def result(self, *args, **kwargs):
        return self.inner.result(*args, **kwargs)

    def snapshot(self):
        return self.inner.snapshot()


def make_identity(dataset, proposal_key, source, record, examples, config, ledger_id, bootstrap_samples):
    provenance = {"review_wrapper_sha256": file_sha(__file__), "wrapper_sha256": FROZEN_ROUTE_SHA,
        "ledger_driver_sha256": jev_route.FROZEN_BUDGET_SHA256, "ledger_id": ledger_id,
        "prior_ledgers": PRIOR, "prior_charged_or_reserved_usd": PRIOR_TOTAL,
        "stage_cap_usd": STAGE_CAP, "aggregate_authorized_usd": TOTAL_CAP,
        "dataset": dataset.name, "proposal_key": proposal_key, "seed": SEED, "shots_per_class": SHOTS,
        "route": "OpenRouter", "endpoint": jev_route.ENDPOINT, "price": jev_route.PRICE,
        "source_run_id": record["run_id"], "source_files_sha256": source["hashes"],
        "review_prompt_version": "cached-label-proposal-review-v1"}
    ident = _identity(dataset, "cached_label_jev_review",
        {**config, "shots_per_class": SHOTS, "source_run_id": record["run_id"],
            "source_model": source["model"], "budget_guard": provenance}, SEED, examples)
    ident.update({"bootstrap_samples": bootstrap_samples, "proposal": source,
        "method_label": "cached-label proposal + Jev review", "new_proposal_generation": False})
    return ident, provenance


def load_checkpoint(directory, identity, dataset, source, source_record, proposals, examples):
    record_path, predictions_path = directory / "run.json", directory / "predictions.jsonl"
    if not record_path.exists():
        if predictions_path.exists():
            raise budget.GuardError("Predictions without their provenance record cannot be resumed")
        return None, []
    record = json.loads(record_path.read_text())
    if any(record.get(key) != value for key, value in identity.items()):
        raise budget.GuardError("Checkpoint identity changed; preserve the original run and investigate")
    if record.get("status") == "stopped_after_three_consecutive_errors":
        raise budget.GuardError("Three consecutive errors halted this condition; investigate before resuming")
    raw = predictions_path.read_text() if predictions_path.exists() else ""
    if raw and not raw.endswith("\n"):
        raise budget.GuardError("Incomplete prediction write; no automatic repeat of a possibly paid call")
    previous = [Prediction(**json.loads(line)) for line in raw.splitlines()]
    if [p.row_id for p in previous] != [r.id for r in dataset.test[:len(previous)]]:
        raise budget.GuardError("Checkpoint is not an exact ordered prefix of the complete test set")
    for row, cached, proposal in zip(dataset.test, previous, proposals):
        if not isinstance(cached.metadata, dict) or not isinstance(cached.latency_s, (int, float)) or not math.isfinite(cached.latency_s) or cached.latency_s < 0:
            raise budget.GuardError("Malformed checkpoint prediction schema")
        if cached.error:
            if not isinstance(cached.error, str) or cached.label is not None or cached.probabilities is not None:
                raise budget.GuardError("Malformed checkpoint failure")
        else:
            if (type(cached.label) is not int or not 0 <= cached.label < len(dataset.labels) or
                    not isinstance(cached.probabilities, list) or len(cached.probabilities) != len(dataset.labels) or
                    any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in cached.probabilities) or
                    not math.isclose(sum(cached.probabilities), 1, rel_tol=0, abs_tol=1e-6) or
                    cached.probabilities[cached.label] < max(cached.probabilities) - 1e-6):
                raise budget.GuardError("Malformed checkpoint Choice label/probabilities")
        expected = proposal_metadata(source, source_record, proposal)
        if cached.metadata.get("proposal") != expected:
            raise budget.GuardError("Checkpoint proposal provenance differs")
        if valid_proposal(proposal, len(dataset.labels)):
            expected_hash = hashlib.sha256(build_review_prompt(row, dataset.labels, examples, proposal.label).encode()).hexdigest()
            if cached.metadata.get("review_prompt_sha256") != expected_hash or not cached.metadata.get("budget", {}).get("reservation_id"):
                raise budget.GuardError("Checkpoint prompt or paid-call identity differs")
        elif cached.error != "source_proposal_failed: Jev review not called" or cached.label is not None or cached.probabilities is not None:
            raise budget.GuardError("Failed proposals must remain pipeline failures without a fallback")
    if record.get("status") == "complete" and len(previous) != len(dataset.test):
        raise budget.GuardError("Complete checkpoint is missing test predictions")
    return record, previous


def validate_reservations(ledger, dataset_name, proposal_key, previous, identity):
    """A reserved call without a durable prediction is uncertain, never replayed."""
    events, _ = ledger.inner._read()
    reservations = {event["reservation_id"]: event for event in events if event["type"] == "reserve"
        and event["details"].get("dataset") == dataset_name and event["details"].get("proposal_key") == proposal_key}
    saved = {p.metadata["budget"]["reservation_id"]: p for p in previous if "budget" in p.metadata}
    if set(saved) != set(reservations):
        raise budget.GuardError("Unreconciled reserved call or checkpoint; do not repeat a possibly paid request")
    for reservation_id, prediction in saved.items():
        details = reservations[reservation_id]["details"]
        guard = identity["config"]["budget_guard"]
        if (prediction.metadata["budget"].get("ledger_id") != ledger.identity["ledger_id"] or
                details.get("row_id") != prediction.row_id or details.get("prompt_sha256") != prediction.metadata["review_prompt_sha256"] or
                any(details.get(key) != value for key, value in guard.items())):
            raise budget.GuardError("Saved prediction and durable reservation provenance disagree")


def run_review(dataset, proposal_key, source, source_record, proposals, examples, config, ledger, *, bootstrap_samples=1000):
    identity, provenance = make_identity(dataset, proposal_key, source, source_record, examples, config,
        ledger.identity["ledger_id"], bootstrap_samples)
    directory = OUTPUT / f"{dataset.name}__{proposal_key}__jev-review"
    record, previous = load_checkpoint(directory, identity, dataset, source, source_record, proposals, examples)
    validate_reservations(ledger, dataset.name, proposal_key, previous, identity)
    if record is not None and record.get("status") == "complete":
        measured = evaluate(dataset.test, previous, len(dataset.labels))
        if any(record["metrics"].get(key) != value for key, value in measured.items()):
            raise budget.GuardError("Completed review metrics differ from saved predictions")
        return record
    record = record or {**identity, "run_id": directory.name, "started_at": budget.utc_now(),
        "environment": environment(), "dataset_manifest": dataset.manifest,
        "execution": {"concurrency": 1, "automatic_retries": 0, "max_total_review_requests": MAX_REQUESTS,
            "proposal_source": "cached_labels_only", "original_proposal_generation_cost_in_prior_study": True,
            "latency_scope": "Jev review stage only; original proposal timing stored in source run"}}
    record["status"] = "running"
    record.setdefault("execution_sessions", []).append({"started_at": budget.utc_now(), "cached_rows": len(previous)})
    save_json(directory / "run.json", record)
    provider = jev_route.GuardedOpenRouterJev(config, ledger, provenance)
    with (directory / "predictions.jsonl").open("a") as file:
        for row, proposal in zip(dataset.test[len(previous):], proposals[len(previous):]):
            if valid_proposal(proposal, len(dataset.labels)):
                prompt = build_review_prompt(row, dataset.labels, examples, proposal.label)
                prediction = provider.predict(row, dataset.labels, prompt)
                prediction.metadata["review_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            else:
                prediction = Prediction(row.id, None, error="source_proposal_failed: Jev review not called",
                    metadata={"probability_kind": "unavailable", "jev_review_called": False})
            prediction.metadata["proposal"] = proposal_metadata(source, source_record, proposal)
            file.write(json.dumps(asdict(prediction), allow_nan=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
            previous.append(prediction)
            if len(previous) >= 3 and all(p.error for p in previous[-3:]):
                record.update({"status": "stopped_after_three_consecutive_errors", "n_predictions": len(previous)})
                save_json(directory / "run.json", record)
                raise budget.GuardError("Stopped after three consecutive pipeline errors")
            if len(previous) % 25 == 0:
                print(json.dumps({"condition": directory.name, "completed_rows": len(previous), "test_rows": len(dataset.test)}), flush=True)
    return finish_run(dataset, previous, directory, record, bootstrap_samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["breast_cancer", "wine"], default=["breast_cancer", "wine"])
    parser.add_argument("--proposals", nargs="+", choices=["qwen3_4b", "astra"], default=["qwen3_4b", "astra"])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--stop-after-new-requests", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--init-ledger", action="store_true")
    parser.add_argument("--prompt-api-key", action="store_true")
    args = parser.parse_args(argv)
    if (len(set(args.datasets)) != len(args.datasets) or len(set(args.proposals)) != len(args.proposals)
            or args.bootstrap_samples < 100 or (args.stop_after_new_requests is not None and args.stop_after_new_requests <= 0)):
        raise budget.GuardError("Duplicate conditions or invalid checkpoint/bootstrap limit")
    verify_transport()
    verify_prior()
    if environment()["source_sha256"] != FROZEN_CORE_SHA:
        raise budget.GuardError("Frozen benchmark inference core changed")
    config = jev_route.validate_config(json.loads((ROOT / "configs/tabular_hosted.json").read_text())["jev_openrouter"])
    jobs, plan = [], []
    for name in args.datasets:
        dataset, _ = load_native_prepared(ROOT / "data/tabular-full" / name)
        if dataset.name != name or any(feature["kind"] != "numeric" for feature in dataset.manifest["tabular"]["features"]):
            raise budget.GuardError("Only the frozen numerical feature datasets are permitted")
        for proposal_key in args.proposals:
            loaded = load_source(dataset, proposal_key)
            jobs.append((dataset, proposal_key, *loaded))
            requests = sum(valid_proposal(prediction, len(dataset.labels)) for prediction in loaded[2])
            plan.append({"dataset": name, "proposal": proposal_key, "source_run_id": loaded[1]["run_id"],
                "test_rows": len(dataset.test), "fresh_jev_calls": requests,
                "source_failures_without_jev_call": len(dataset.test) - requests})
    count = sum(row["fresh_jev_calls"] for row in plan)
    if count > MAX_REQUESTS or Decimal(budget.usd_string(count * jev_route.RESERVE_NANO)) > Decimal(STAGE_CAP):
        raise budget.GuardError("Review matrix exceeds the new allocation")
    summary = {"mode": "execute" if args.execute else "dry_run", "method": "cached-label proposal + Jev review",
        "matrix": plan, "fresh_jev_calls": count, "fresh_reservations_usd": budget.usd_string(count * jev_route.RESERVE_NANO),
        "prior_charged_or_reserved_usd": PRIOR_TOTAL, "review_cap_usd": STAGE_CAP,
        "cumulative_authorized_usd": TOTAL_CAP,
        "note": "No new LLM proposals; original cached labels only. Review is not a same-base decoding ablation."}
    print(json.dumps(summary, indent=2), flush=True)
    if not args.execute:
        return summary
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    # Hold one process lock for the complete execution, including credential entry.
    with LEDGER.with_name("review-execution.lock").open("a+") as execution_lock:
        try:
            fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise budget.GuardError("Another review worker is already running") from None
        ledger = AnchoredLedger(budget.Ledger(LEDGER, STAGE_CAP, initialize=args.init_ledger), stop_after=args.stop_after_new_requests)
        if args.prompt_api_key:
            budget.prompt_credentials([config])
        if not os.environ.get(config["api_key_env"]):
            raise budget.GuardError("Missing Jev credential; no model calls made")
        completed = []
        for job in jobs:
            try:
                record = run_review(*job, config, ledger, bootstrap_samples=args.bootstrap_samples)
            except jev_route.StopRequested:
                checkpoint = {"status": "checkpoint", "new_calls": ledger.new_requests, "budget": ledger.snapshot()}
                print(json.dumps(checkpoint), flush=True)
                return checkpoint
            completed.append(record["run_id"])
            print(json.dumps({"completed": record["run_id"], "accuracy": record["metrics"]["accuracy"],
                "failures": record["metrics"]["n_failures"], "budget": ledger.snapshot()}), flush=True)
        return {"completed": completed, "budget": ledger.snapshot()}


if __name__ == "__main__":
    try:
        main()
    except (budget.BudgetStop, budget.GuardError) as exc:
        print(f"Stopped: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
