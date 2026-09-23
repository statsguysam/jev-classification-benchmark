"""Offline immutable inventory of failed first-pass inference attempts.

No credentials, inference, file replacement, or accuracy-based retry selection.
Reconstructs the exact original request in memory and checks its reserved prompt
hash. Rescan after the pending first passes; write a new named inventory snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import digest, environment
import run_numeric_jev_review as numeric_prompt
import run_text_jev_review as text_prompt
import text_extension_sources as text_sources
from tabular_data import load_native_prepared

CORE_SHA = "d5547ccde4224e182653315d81c0a631c789bbe294ad7bcf95ef0cddd25ce608"
DEFAULT_OUTPUT = ROOT / "results/completion_20260923/FAILED_INVENTORY.json"
CONFIG_KEYS = {"provider", "model", "base_url", "api_key_env", "timeout", "max_output_tokens",
               "reasoning_effort", "token_limit_parameter", "temperature", "seed"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def prediction_sha256(value):
    """SHA256 of sorted compact UTF-8 JSON, not the physical JSONL line."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical_path(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Expected repository-relative path")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), "Noncanonical evidence path")
    return path


def read_predictions(path):
    raw = Path(path).read_bytes()
    require(not raw or raw.endswith(b"\n"), "Incomplete prediction append; snapshot after writer stops")
    return [json.loads(line) for line in raw.splitlines()]


def ledger_inventory(root):
    ledgers = {}
    for path in sorted((root / "results").rglob("*budget*.jsonl")):
        events = read_predictions(path)
        require(events and events[0].get("type") == "header", "Budget evidence lacks its header")
        identity = events[0]["ledger_id"]
        require(identity not in ledgers, "Duplicate ledger identity")
        cap = str(Decimal(events[0]["budget_nano"]) / Decimal(1_000_000_000))
        ledger = numeric_prompt.budget.Ledger(path, cap)
        audited, _ = ledger._read()
        require(audited == events, "Ledger changed while taking retry inventory")
        reserves = {e["reservation_id"]: e for e in events if e["type"] == "reserve"}
        results = Counter(e["reservation_id"] for e in events if e["type"] == "result")
        ledgers[identity] = {"path": path.relative_to(root).as_posix(), "reserves": reserves, "results": results,
                             "pins": {path.relative_to(root).as_posix(): file_sha(path),
                                      ledger.lock_path.relative_to(root).as_posix(): file_sha(ledger.lock_path)}}
    return ledgers


def request_recipe(record):
    if record.get("method") == "cached_label_jev_review":
        return "text_cached_proposal" if record["dataset"] in text_sources.DATASETS else "numeric_cached_proposal"
    require(record.get("method") in ("zero_shot", "few_shot"), "Failed run has no audited request reconstruction recipe")
    return "standard_prompt"


def dataset_for(record, root):
    name = record["dataset"]
    dataset = (text_sources.load_dataset(name) if name in text_sources.DATASETS else
               load_native_prepared(root / "data/tabular-full" / name)[0])
    require(dataset.labels == record["labels"] and digest(dataset.manifest) == record["manifest_sha256"],
            "Original dataset/label/manifest identity differs")
    require(record.get("implementation_sha256") == CORE_SHA, "Original inference core differs")
    return dataset


def original_prediction(item, root):
    for relative, expected in item["artifact_sha256"].items():
        require(file_sha(canonical_path(root, relative)) == expected, "Inventory evidence changed; prepare a new immutable snapshot")
    path = canonical_path(root, item["prediction_path"])
    values = read_predictions(path)
    prediction = values[item["prediction_line_1based"] - 1]
    require(prediction["row_id"] == item["row_id"] and prediction.get("error") == item["error"] and
            prediction_sha256(prediction) == item["original_prediction_sha256"], "Original failure identity changed")
    return prediction


def reconstruct_request(item, root=ROOT, recovered_source_prediction=None):
    """Return request data in memory only; never include a test target or key.

    For upstream skips, an explicitly supplied recovered source prediction builds a new
dependent review request. It is not an exact retry of a previously sent prompt.
The caller must pin the separate successful source-retry artifact/provenance.
"""
    root = Path(root).resolve()
    require(root == ROOT, "Reconstruction uses the canonical frozen dataset registry")
    require(environment()["source_sha256"] == CORE_SHA, "Frozen request/data core changed")
    prediction = original_prediction(item, root)
    record = read(canonical_path(root, item["run_path"]))
    recovered = None
    if item["request_recipe"] == "control_frozen_request":
        prepared = root / "results/review_controls/prepared_full"
        require(file_sha(prepared / "requests.jsonl") == record.get("requests_sha256"),
                "Prepared control requests differ from the original execution identity")
        require(file_sha(prepared / "protocol.json") == record.get("protocol_sha256"),
                "Prepared control protocol differs from the original execution identity")
        requests = read_predictions(prepared / "requests.jsonl")
        matches = [q for q in requests if q["request_id"] == item["control_request_id"]]
        require(len(matches) == 1, "Control request identity is missing or ambiguous")
        request = matches[0]
        prompt, labels = request["prompt"], request["choices"]
        config = {"provider": "jev", "model": "typesafe/jev-1.13", "base_url": "https://openrouter.ai/api/v1",
                  "api_key_env": "OPENROUTER_API_KEY", "timeout": 120}
    else:
        dataset = dataset_for(record, root)
        rows = [row for row in dataset.test if row.id == item["row_id"]]
        require(len(rows) == 1, "Failed row is not uniquely present in the frozen test fold")
        row = rows[0]
        labels = dataset.labels
        shots = record["config"]["shots_per_class"]
        examples = select_examples(dataset.train, labels, shots, record["seed"])
        require([r.id for r in examples] == record["training_example_ids"], "Original demonstrations differ")
        config = {k: v for k, v in record["config"].items() if k in CONFIG_KEYS}
        if item["request_recipe"] == "standard_prompt":
            prompt = build_prompt(row, labels, examples)
        else:
            proposal = prediction["metadata"]["proposal"]
            source_folder = canonical_path(root, proposal["source_path"])
            require(all(file_sha(source_folder / name) == h for name, h in proposal["source_files_sha256"].items()),
                    "Original cached source changed")
            source_values = [p for p in read_predictions(source_folder / "predictions.jsonl") if p["row_id"] == row.id]
            require(len(source_values) == 1 and digest(source_values[0]) == proposal["source_prediction_sha256"],
                    "Cached source prediction identity differs")
            if item["kind"] == "upstream_skip":
                recovered = asdict(recovered_source_prediction) if is_dataclass(recovered_source_prediction) else recovered_source_prediction
                require(isinstance(recovered, dict) and source_values[0].get("error") and
                        recovered.get("row_id") == row.id and not recovered.get("error") and
                        type(recovered.get("label")) is int and 0 <= recovered["label"] < len(labels),
                        "Successful separately recorded source retry required")
                label = recovered["label"]
            else:
                label = source_values[0]["label"]
                require(not source_values[0].get("error") and label == proposal["source_label"], "Source proposal was not valid")
            render = text_prompt.build_review_prompt if item["request_recipe"] == "text_cached_proposal" else numeric_prompt.build_review_prompt
            prompt = render(row, labels, examples, label)
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    if item["kind"] == "paid_failure":
        require(prompt_sha == item["original_prompt_sha256"], "Reconstructed prompt differs from the paid original")
    provider = config["provider"]
    if provider == "jev":
        endpoint = "https://openrouter.ai/api/v1/systemone"
        body = {"model": config["model"], "state": prompt, "questions": {"classification": {
            "type": "choice", "instructions": "Choose the numeric class id for the final text in the state, following its classification task and examples.",
            "criteria": {str(i): label for i, label in enumerate(labels)}}}}
    else:
        require(provider == "openai", "Unsupported paid retry provider")
        endpoint = config.get("base_url", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        body = {"model": config["model"], "messages": [{"role": "user", "content": prompt}],
                config.get("token_limit_parameter", "max_completion_tokens"): config["max_output_tokens"]}
        body.update({key: config[key] for key in ("temperature", "seed", "reasoning_effort") if key in config})
    result = {"row_id": item["row_id"], "prompt": prompt, "labels": labels, "config": config,
            "endpoint": endpoint, "request_body": body, "original_prompt_sha256": item["original_prompt_sha256"],
            "prompt_sha256": prompt_sha, "is_new_dependent_request": item["kind"] == "upstream_skip"}
    if recovered is not None:
        result["dependency"] = {"source_original_run_id": item["source_condition"]["source_run_id"],
            "source_original_row_id": item["row_id"], "recovered_prediction_sha256": prediction_sha256(recovered)}
    return result


def make_item(path, record, prediction, line, ledgers, root, *, recipe=None):
    meta = prediction.get("metadata", {})
    skipped = prediction["error"] == "source_proposal_failed: Jev review not called"
    kind = "upstream_skip" if skipped else "paid_failure"
    model, provider = record["config"]["model"], record["config"]["provider"]
    require(provider in ("jev", "openai"), "Local/non-hosted inference failure needs a separate recovery design")
    pp = path.with_name("predictions.jsonl")
    pins = {p.relative_to(root).as_posix(): file_sha(p) for p in (path, pp)}
    mp = path.with_name("test_manifest.json")
    if mp.exists(): pins[mp.relative_to(root).as_posix()] = file_sha(mp)
    item = {"kind": kind, "run_path": path.relative_to(root).as_posix(), "prediction_path": pp.relative_to(root).as_posix(),
            "run_id": record["run_id"], "row_id": prediction["row_id"], "prediction_line_1based": line,
            "original_prediction_sha256": prediction_sha256(prediction), "error": prediction["error"],
            "provider": provider, "model": model, "dataset": record["dataset"],
            "shots_per_class": record["config"]["shots_per_class"], "method": record["method"],
            "request_recipe": recipe or request_recipe(record), "artifact_sha256": pins,
            "scope": "current_four_dataset_study" if record["dataset"] in ("sst2", "trec", "breast_cancer", "wine") else "historical_additional_dataset",
            "original_prompt_sha256": None, "source_condition": None}
    item["failure_id"] = hashlib.sha256(canonical_bytes([item["run_id"], item["row_id"], item["original_prediction_sha256"]])).hexdigest()
    if "proposal" in meta:
        proposal = meta["proposal"]
        item["source_condition"] = {key: proposal[key] for key in ("source_run_id", "source_path", "source_model", "source_row_id", "source_prediction_sha256", "source_failed")}
        for name, value in proposal["source_files_sha256"].items():
            relative = (Path(proposal["source_path"]) / name).as_posix()
            require(file_sha(canonical_path(root, relative)) == value, "Cached source pin changed")
            pins[relative] = value
    if skipped:
        require(not meta.get("budget", {}).get("reservation_id"), "Source skip unexpectedly contains a paid reservation")
        item.update(original_ledger_path=None, original_ledger_id=None, original_reservation_id=None,
                    original_retained_reservation_usd="0", dependency="Recover the failed source separately, then record a new dependent Jev attempt")
    else:
        identity, reservation = meta["budget"]["ledger_id"], meta["budget"]["reservation_id"]
        ledger = ledgers[identity]
        reserve = ledger["reserves"][reservation]
        require(ledger["results"][reservation] == 1 and reserve["details"]["row_id"] == prediction["row_id"] and
                reserve["details"]["model"] == model, "Failure does not match exactly one paid result")
        item.update(original_ledger_path=ledger["path"], original_ledger_id=identity, original_reservation_id=reservation,
                    original_prompt_sha256=reserve["details"]["prompt_sha256"],
                    original_retained_reservation_usd=numeric_prompt.budget.usd_string(reserve["reserved_nano"]))
        pins.update(ledger["pins"])
    return item


def pending_counts(root, runs):
    details, sources = [], []
    for family, expected in (("numeric_expansion", 24), ("text_extension", 24)):
        summary = read(root / "results" / family / "COMPARISON.json")
        for row in summary["runs"]:
            if row["arm"] == "base":
                path = canonical_path(root, row["source_path"])
                record = read(path); ps = read_predictions(path.with_name("predictions.jsonl"))
                sources.append({"run_id": record["run_id"], "dataset": record["dataset"], "n_rows": len(ps),
                    "failed_rows": sum(bool(p.get("error")) for p in ps),
                    "invalid_success_labels": sum(not p.get("error") and (type(p.get("label")) is not int or
                        not 0 <= p["label"] < len(record["labels"])) for p in ps)})
            if row["arm"] != "review": continue
            source_id = row["source_run_id"]
            matches = [record for _, record in runs if record.get("config", {}).get("source_run_id") == source_id
                       and record.get("dataset") == row["dataset"] and record.get("method") == "cached_label_jev_review"]
            require(len(matches) <= 1, "Ambiguous first-pass review condition")
            source = next(x for x in summary["runs"] if x["run_id"] == source_id)
            count = 0
            if matches:
                path = next(path for path, record in runs if record is matches[0])
                count = len(read_predictions(path.with_name("predictions.jsonl")))
            require(count <= source["n_test"], "First-pass row count exceeds source fold")
            details.append({"family": family, "dataset": row["dataset"], "source_run_id": source_id,
                "shots_per_class": row["train_per_class"], "expected_rows": source["n_test"],
                "saved_rows": count, "pending_rows": source["n_test"] - count})
    return {"by_condition": details, "numeric_pending_rows": sum(x["pending_rows"] for x in details if x["family"] == "numeric_expansion"),
            "text_pending_rows": sum(x["pending_rows"] for x in details if x["family"] == "text_extension"),
            "source_runs": sources, "invalid_success_source_labels": sum(x["invalid_success_labels"] for x in sources),
            "failed_source_rows": sum(x["failed_rows"] for x in sources)}


def build_inventory(root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT and environment()["source_sha256"] == CORE_SHA, "Canonical frozen checkout required")
    ledgers, runs, failures = ledger_inventory(root), [], []
    counts = Counter()
    for path in sorted((root / "results").rglob("run.json")):
        if "completion_20260923" in path.parts: continue  # Never recursively retry retry attempts.
        record = read(path)
        if record.get("study") == "proposal-value-controls-v1" and path.with_name("predictions.jsonl").exists():
            request_path = root / "results/review_controls/prepared_full/requests.jsonl"
            requests = {q["request_id"]: q for q in read_predictions(request_path)}
            predictions = read_predictions(path.with_name("predictions.jsonl"))
            counts["control_runs_scanned"] += 1; counts["control_prediction_rows_scanned"] += len(predictions)
            for line, prediction in enumerate(predictions, 1):
                request_id = prediction["metadata"]["control"]["request_id"]
                request = requests[request_id]
                require(prediction["row_id"] == request["row_id"], "Control row identity differs")
                if not prediction.get("error"):
                    require(type(prediction.get("label")) is int and 0 <= prediction["label"] < len(request["choices"]),
                            "Invalid successful control label")
                    continue
                synthetic_record = {"run_id": "proposal-value-controls-v1::" + request_id, "dataset": request["dataset"],
                    "method": "proposal_value_control", "config": {"provider": "jev", "model": "typesafe/jev-1.13", "shots_per_class": 4}}
                item = make_item(path, synthetic_record, prediction, line, ledgers, root, recipe="control_frozen_request")
                item["control_request_id"] = request_id
                item["control_arm"] = request["arm"]
                item["artifact_sha256"][request_path.relative_to(root).as_posix()] = file_sha(request_path)
                reconstruct_request(item, root)
                failures.append(item)
            continue
        if "labels" not in record or not path.with_name("predictions.jsonl").exists(): continue
        runs.append((path, record))
        predictions = read_predictions(path.with_name("predictions.jsonl"))
        require(len({p["row_id"] for p in predictions}) == len(predictions), "Duplicate original inference row")
        counts["standard_runs_scanned"] += 1; counts["prediction_rows_scanned"] += len(predictions)
        for line, prediction in enumerate(predictions, 1):
            if not prediction.get("error"):
                require(type(prediction.get("label")) is int and 0 <= prediction["label"] < len(record["labels"]),
                        "A successful original label is invalid")
                continue
            item = make_item(path, record, prediction, line, ledgers, root)
            if item["kind"] == "paid_failure": reconstruct_request(item, root)
            failures.append(item)
    by_key = {(item["run_id"], item["row_id"]): item for item in failures}
    for item in failures:
        if item["kind"] == "upstream_skip":
            dependency = by_key[(item["source_condition"]["source_run_id"], item["row_id"])]
            item["depends_on_failure_id"] = dependency["failure_id"]
    counts.update(paid_failed_calls=sum(f["kind"] == "paid_failure" for f in failures),
                  paid_jev_failures=sum(f["kind"] == "paid_failure" and f["provider"] == "jev" for f in failures),
                  paid_openai_failures=sum(f["kind"] == "paid_failure" and f["provider"] == "openai" for f in failures),
                  upstream_skips=sum(f["kind"] == "upstream_skip" for f in failures))
    pending = pending_counts(root, runs)
    pins = {name: value for item in failures for name, value in item["artifact_sha256"].items()}
    require(all(file_sha(root / name) == value for name, value in pins.items()), "Evidence changed during inventory; rescan after writer stops")
    return {"schema_version": 1, "scope": "All saved original standard inference failures; no retries of wrong-but-valid labels or poor LoRA outcomes",
        "counts": dict(counts), "failures": failures, "pending_first_pass": pending,
        "frozen_control_requests": read(root / "results/review_controls/prepared_full/protocol.json")["request_count"],
        "retry_policy": {"maximum_additional_attempts_per_original_failure": 2, "selection": "Original error only; never accuracy or confidence",
            "stop_on_first_valid_success": True, "rounds": "At most two separate failure-only rounds; never repeat a successful response",
            "preserve_originals": True, "separate_recovered_report": True, "new_failures": "Rescan completed first-pass outputs into a new immutable inventory; no recursive retries",
            "unresolved_orphan_requests": "Stop for accounting reconciliation; do not infer free failures or retry uncertain calls"},
        "cost_note": "Original unknown-cost reservations remain retained. Reverify prices; allocate each additional call inside the unchanged aggregate $25, counting the health probe and outstanding scopes.",
        "prediction_hash_algorithm": "SHA256 of sorted compact UTF-8 JSON with ensure_ascii=False and allow_nan=False",
        "producer_sha256": file_sha(__file__), "frozen_core_sha256": CORE_SHA}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_inventory()
    if args.write:
        output = args.output.resolve()
        require(output.is_relative_to((ROOT / "results/completion_20260923").resolve()), "Inventory output must stay in its separate namespace")
        raw = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        if output.exists():
            require(output.read_text() == raw, "Immutable inventory already exists; choose a new snapshot filename")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"mode": "write_immutable_snapshot" if args.write else "dry_run_no_writes", **report["counts"],
        "numeric_pending_rows": report["pending_first_pass"]["numeric_pending_rows"],
        "text_pending_rows": report["pending_first_pass"]["text_pending_rows"], "control_requests": report["frozen_control_requests"]}))
    return report


if __name__ == "__main__":
    main()
