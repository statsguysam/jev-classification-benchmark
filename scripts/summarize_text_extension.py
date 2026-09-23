#!/usr/bin/env python3
"""Reaudit frozen SST-2/TREC evidence; never infer or score incomplete conditions."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import text_extension_sources as sources
from jevbench.data import normalized_text
from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import digest, save_json
from jevbench.types import Prediction
from summarize_numeric_decisions import chosen, transitions
from summarize_tabular import group_bootstrap as _group_bootstrap

DATASETS = ("sst2", "trec")
NAMES = {"sst2": "SST-2 sentiment", "trec": "TREC question categories"}
MODEL_NAMES = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5 0.5B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3 4B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2 1.7B",
    "ibm-granite/granite-3.3-2b-instruct": "Granite 3.3 2B",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-6-astra": "GPT-6 Astra"}
MODEL_KEYS = {spec["model"]: key for key, spec in sources.MODELS.items()}
NEW_MODELS = {sources.MODELS[key]["model"] for key in ("smollm2", "granite")}
JEV = "typesafe/jev-1.13"
CLASSICAL_NAMES = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
                   "xgboost": "XGBoost", "lightgbm": "LightGBM"}
REUSED_REVIEW_MODELS = set()
SOURCE_VALIDATOR_SHA = "6f33dea3df79cb8523bcdb06b52105cdb7c0bb4be29f4d2ad1ceacf07c56c290"
CLASSICAL_HELPER_SHA = "d0a6d452185c5487e32b01160bfcdb4d044ee5d804b969f4fe95517dc81f9a2c"
CLASSICAL_SUMMARY_SHA = "6db64ddb3a3117cb6c47c4c2e1e537ab45e808ac66392cbbe3b287bebe0be38e"
REVIEW_RUNNER_SHA = "bdc3d1cac8e52791359023b57abac8833d898af6ff7b0240512e337baa3ba43e"
FIXED_PRIOR = Decimal("19.325827700")
STAGE_CAP = Decimal("1.60")
TOTAL_CAP = Decimal("25.00")
NUMERIC_ENVELOPE = Decimal("4.032000000")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_bootstrap(*args, **kwargs):
    value = _group_bootstrap(*args, **kwargs)
    value["group_definition"] = "Unicode NFKC, casefold, whitespace-collapsed text SHA256"
    return value


def expected_conditions():
    result = []
    for dataset in DATASETS:
        result.extend((dataset, model, arm, shots) for model in MODEL_NAMES for shots in (0, 4)
                      for arm in ("base", "review"))
        result.extend((dataset, JEV, "direct", shots) for shots in (0, 4))
        result.extend((dataset, model, "classical", budget) for model in CLASSICAL_NAMES for budget in (4, None))
    return result


def row_key(row):
    if "source_run_id" in row or row["model"].endswith("+jev_review"):
        return row["dataset"], row["model"].removesuffix("+jev_review"), "review", row["train_per_class"]
    arm = "direct" if row["model"] == JEV else "classical" if row["model"] in CLASSICAL_NAMES else "base"
    return row["dataset"], row["model"], arm, row["train_per_class"]


def placeholder(key):
    dataset, model, arm, budget = key
    display = "Jev alone" if model == JEV else CLASSICAL_NAMES.get(model, MODEL_NAMES.get(model))
    if arm == "review":
        display += " → Jev"
    return {"dataset": dataset, "model": model+"+jev_review" if arm == "review" else model,
        "display_model": display, "arm": arm, "method": "cached_label_jev_review" if arm == "review" else
        "classical_reference" if arm == "classical" else "zero_shot" if budget == 0 else "few_shot",
        "train_per_class": budget, "train_labels": None, "seed": 42,
        "run_id": "pending::"+"::".join(map(str, key)), "source_path": None,
        "status": "pending: no complete audited artifact", "accuracy": None, "macro_f1": None,
        "n_failures": None, "n_test": None, "group_bootstrap": None,
        "probability_coverage": None, "probability_kind": None, "balanced_accuracy": None, "failure_rate": None, "n_classes": None, "log_loss": None, "brier_sum": None, "review_transitions": None}


def comparison_specs():
    result = []
    for dataset in DATASETS:
        for model in MODEL_NAMES:
            for shots in (0, 4):
                review = dataset, model, "review", shots
                result.append(("review_minus_source", review, (dataset, model, "base", shots), True))
                result.append(("review_minus_jev_alone", review, (dataset, JEV, "direct", shots), True))
            for arm in ("base", "review"):
                result.append(("few_minus_zero_descriptive", (dataset, model, arm, 4), (dataset, model, arm, 0), False))
    return result


def compare(rows, payloads, *, samples=2000):
    result = []
    for kind, ka, kb, equal_budget in comparison_specs():
        a, b = rows[ka], rows[kb]
        item = {"kind": kind, "dataset": ka[0], "a": a["run_id"], "b": b["run_id"],
            "a_display": a["display_model"], "b_display": b["display_model"],
            "train_per_class_a": ka[3], "train_per_class_b": kb[3],
            "status": "pending", "paired_bootstrap": None, "transitions": None,
            "interpretation": "Exploratory, unadjusted, A minus B; fixed split/models/serialization",
            "equal_new_label_budget": equal_budget}
        if a["status"] != "complete" or b["status"] != "complete":
            result.append(item)
            continue
        pa, pb = payloads[ka], payloads[kb]
        require(all(pa[field] == pb[field] for field in ("manifest_sha256", "y", "groups", "n_classes", "test_row_ids")), "Paired conditions use unequal held-out records")
        if equal_budget:
            require(pa["training_example_ids"] == pb["training_example_ids"], "Matched comparison uses unequal training IDs")
        item.update(status="complete", paired_bootstrap=group_bootstrap(pa["y"], pa["predictions"], pa["groups"], pb["predictions"], n_classes=pa["n_classes"], samples=samples))
        if kind == "review_minus_source":
            require(a["source_run_id"] == b["run_id"], "Review is linked to a different source run")
            counts = transitions(pa["y"], pb["predictions"], pa["predictions"])
            require(math.isclose(counts["accuracy_delta_pp"]/100, item["paired_bootstrap"]["metrics"]["accuracy"]["estimate"], abs_tol=1e-12), "Transition counts disagree with accuracy delta")
            item["transitions"] = counts
            a["review_transitions"] = counts
        result.append(item)
    return result


def assemble(audited, payloads, *, samples=2000):
    rows = {key: placeholder(key) for key in expected_conditions()}
    for key, row in audited.items():
        require(key in rows and row_key(row) == key, "Unexpected expanded condition")
        rows[key] = row
        rows[key]["arm"] = key[2]
    for key, row in rows.items():
        if key[2] == "review":
            source = rows[key[0], key[1], "base", key[3]]
            if row["status"] == "complete":
                require(source["status"] == "complete" and row.get("source_run_id") == source["run_id"], "Complete review has an unavailable/different source")
            else:
                row["source_run_id"] = source["run_id"]
    comparisons = compare(rows, payloads, samples=samples)
    counts = {arm: sum(row["status"] == "complete" for key, row in rows.items() if key[2] == arm)
              for arm in ("base", "review", "direct", "classical")}
    return {"schema_version": 1, "scope": "exploratory text-classification extension",
        "datasets": list(DATASETS), "models": MODEL_NAMES, "expected_runs": 68,
        "complete_runs": sum(counts.values()), "status": "complete" if sum(counts.values()) == 68 else "in_progress_or_incomplete",
        "expected_base_runs": 24, "complete_base_runs": counts["base"],
        "expected_review_runs": 24, "complete_review_runs": counts["review"],
        "expected_direct_jev_runs": 4, "complete_direct_jev_runs": counts["direct"],
        "expected_classical_runs": 16, "complete_classical_runs": counts["classical"],
        "expected_new_local_runs": 8, "complete_new_local_runs": sum(row["status"] == "complete" and key[1] in NEW_MODELS and key[2] == "base" for key, row in rows.items()),
        "expected_new_review_runs": 24, "complete_new_review_runs": sum(row["status"] == "complete" and key[2] == "review" and not (key[1] in REUSED_REVIEW_MODELS and key[3] == 4) for key, row in rows.items()),
        "expected_reused_review_runs": 0, "complete_reused_review_runs": sum(row["status"] == "complete" and key[2] == "review" and key[1] in REUSED_REVIEW_MODELS and key[3] == 4 for key, row in rows.items()),
        "bootstrap_samples": samples, "seed": 42, "runs": list(rows.values()), "comparisons": comparisons,
        "expected_comparisons": 72, "complete_comparisons": sum(item["status"] == "complete" for item in comparisons)}


def pct(value):
    return "pending" if value is None else f"{100*value:.1f}%"


def interval(comparison):
    if comparison["status"] != "complete":
        return "pending"
    result = comparison["paired_bootstrap"]["metrics"]["accuracy"]
    low, high = result["ci95"]
    return f"{100*result['estimate']:+.2f} [{100*low:+.2f}, {100*high:+.2f}]"


def protocol_plan(root=ROOT, *, frozen=False):
    root = Path(root)
    paths = ["scripts/text_extension_sources.py", "configs/text_extension_sources.json",
             "scripts/run_text_classical_extension.py", "configs/text_classical_extension.json"]
    result = {"schema_version": 1, "study_status": "exploratory_after_prior_text_results_viewed",
        "datasets": list(DATASETS), "seed": 42, "shots_per_class": [0, 4],
        "classical_training_budgets_per_class": [4, None], "expected_conditions": 68,
        "expected_comparisons": 72, "model_registry": sources.MODELS,
        "source_sha256": {name: file_sha(root/name) for name in paths},
        "text_source_validator_sha256": SOURCE_VALIDATOR_SHA,
        "classical_helper_sha256": CLASSICAL_HELPER_SHA,
        "review_prompt_version": "cached-label-text-review-v1",
        "review_design": "Original text, exactly matched examples, cached proposed class; no source model identity or generated rationale in reviewer prompt",
        "group_definition": "Unicode NFKC, casefold, whitespace-collapsed text SHA256",
        "intervals": "Unadjusted paired group percentile bootstrap, conditional on fixed split/examples/models",
        "classical_features": "Training-only sparse word (1,2) plus char_wb (3,5) TF-IDF; one fixed estimator configuration; no validation fitting",
        "new_review_stage_cap_usd": str(STAGE_CAP), "cumulative_authorized_usd": str(TOTAL_CAP),
        "numeric_study_reserved_allowance_usd": str(NUMERIC_ENVELOPE),
        "fixed_prior_usd": str(FIXED_PRIOR),
        "review_execution_frozen": frozen,
        "review_runner_sha256": file_sha(root/"scripts/run_text_jev_review.py") if frozen else None}
    require(result["source_sha256"]["scripts/text_extension_sources.py"] == SOURCE_VALIDATOR_SHA,
            "Text source validator changed")
    require(result["source_sha256"]["scripts/run_text_classical_extension.py"] == CLASSICAL_HELPER_SHA,
            "Text classical producer changed")
    require(not frozen or result["review_runner_sha256"] == REVIEW_RUNNER_SHA,
            "Final text review producer changed")
    return result


def prepare_protocol(root=ROOT, *, freeze=False):
    root = Path(root)
    path = root/"results/text_extension/protocol.json"
    previous = read(path) if path.exists() else None
    frozen = bool(freeze or previous and previous["review_execution_frozen"])
    plan = protocol_plan(root, frozen=frozen)
    if previous and previous["review_execution_frozen"]:
        require(previous == plan, "Frozen text execution protocol changed")
    elif previous != plan:
        require(not list((root/"results/text_extension/review").glob("*/run.json")),
                "Cannot change a draft protocol after review execution")
        save_json(path, plan)
    return plan


def audited_row(dataset, record, predictions, *, model, arm, budget, path, samples, root=ROOT):
    require(record.get("status") == "complete", "Incomplete condition cannot be scored")
    selected = dataset.train if budget is None else select_examples(dataset.train, dataset.labels, budget, 42)
    ids = [row.id for row in selected]
    require(record["training_example_ids"] == ids and record["seed"] == 42,
            "Training examples/seed differ from the exact frozen budget")
    require(record["dataset"] == dataset.name and record["labels"] == dataset.labels,
            "Prediction dataset/class order differs")
    require([p.row_id for p in predictions] == [r.id for r in dataset.test],
            "Predictions must cover every ordered frozen test row")
    for prediction in predictions:
        if prediction.error:
            require(prediction.label is None and prediction.probabilities is None, "Failure cannot retain a class")
        else:
            require(type(prediction.label) is int and 0 <= prediction.label < len(dataset.labels), "Invalid class ID")
            if prediction.probabilities is not None:
                p = prediction.probabilities
                require(len(p) == len(dataset.labels) and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in p)
                    and math.isclose(sum(p), 1, abs_tol=1e-6), "Invalid probabilities or decision rule")
                # Jev supplies its own Choice label. Match the frozen adapter's
                # tolerance without imposing a first-index argmax tie break.
                # Other recipes retain their declared first-argmax decision.
                decision_valid = (p[prediction.label] >= max(p) - 1e-6 if record["config"].get("provider") == "jev"
                    else prediction.label == max(range(len(p)), key=p.__getitem__))
                require(decision_valid, "Invalid probabilities or decision rule")
    metrics = evaluate(dataset.test, predictions, len(dataset.labels))
    require(all(record["metrics"].get(k) == v for k, v in metrics.items()), "Saved metrics disagree with predictions")
    values = chosen(predictions, len(dataset.labels))
    payload = {"y": [r.label for r in dataset.test], "predictions": values,
        "groups": [hashlib.sha256(normalized_text(r.text).encode()).hexdigest() for r in dataset.test],
        "n_classes": len(dataset.labels), "training_example_ids": ids, "test_row_ids": [r.id for r in dataset.test],
        "manifest_sha256": digest(dataset.manifest)}
    display = "Jev alone" if model == JEV else CLASSICAL_NAMES.get(model, MODEL_NAMES.get(model.removesuffix("+jev_review")))
    if arm == "review":
        display += " → Jev"
    row = {"dataset": dataset.name, "model": model, "display_model": display, "arm": arm,
        "method": record["method"], "train_per_class": budget, "train_labels": len(ids), "seed": 42,
        "status": "complete", "run_id": record["run_id"], "source_path": Path(path).relative_to(root).as_posix(),
        "n_classes": len(dataset.labels), "original_source_manifest_sha256": record["manifest_sha256"],
        "group_bootstrap": group_bootstrap(payload["y"], values, payload["groups"], n_classes=len(dataset.labels), samples=samples),
        "artifact_sha256": {name: file_sha(Path(path).with_name(name)) for name in sources.FILE_NAMES},
        "config": record["config"], "review_transitions": None, **metrics}
    return row, payload


def classical_artifacts(root=ROOT):
    """Recheck fixed producer/data/parameters before accepting stored predictions."""
    import run_text_classical_extension as producer
    root = Path(root)
    folder = root/"results/text_extension/classical"
    require(file_sha(folder/"summary.json") == CLASSICAL_SUMMARY_SHA, "Frozen classical summary changed")
    summary, protocol = read(folder/"summary.json"), read(folder/"protocol.json")
    require(summary.get("status") == "complete" and len(summary["runs"]) == 16, "Classical matrix incomplete")
    config = read(root/"configs/text_classical_extension.json")
    require(protocol["config"] == config and protocol["config_file_sha256"] == file_sha(root/"configs/text_classical_extension.json")
        and protocol["source_sha256"] == producer.source_hashes() and file_sha(producer.__file__) == CLASSICAL_HELPER_SHA,
        "Classical producer/configuration changed")
    require(summary["protocol_sha256"] == digest(protocol), "Classical protocol linkage differs")
    require({p.parent.name for p in folder.glob("*/run.json")} == {r["run_id"] for r in summary["runs"]},
            "Classical artifact inventory differs")
    result, seen, features = [], set(), {}
    for item in summary["runs"]:
        path = folder/item["run_id"]/"run.json"
        record = read(path)
        name, model, budget = item["dataset"], item["model"], item["train_per_class"]
        key = name, model, "classical", budget
        require(key in expected_conditions() and key not in seen and file_sha(path) == item["run_json_sha256"],
                "Duplicate, unexpected or changed classical run")
        seen.add(key)
        dataset = sources.load_dataset(name)
        require(record["config"]["protocol_sha256"] == digest(protocol)
            and record["config"]["experiment_config_sha256"] == digest(config)
            and record["config"]["source_sha256"] == producer.source_hashes(), "Classical source provenance differs")
        for filename, expected in protocol["prepared_files_sha256"][name].items():
            require(file_sha(root/"data/pilot"/name/filename) == expected, "Classical prepared artifact changed")
        require(record["status"] == "complete" and record["method"] == "classical_text_extension"
            and record["dataset_manifest"] == dataset.manifest and record["manifest_sha256"] == digest(dataset.manifest),
            "Classical dataset/method differs")
        for filename, expected in record["artifacts_sha256"].items():
            require(file_sha(path.with_name(filename)) == expected, "Classical prediction/test artifact changed")
        selected = dataset.train if budget is None else select_examples(dataset.train, dataset.labels, budget, 42)
        training = record["training"]
        params = producer.model_parameters(model, config, len(dataset.labels), 42)
        require(training["training_rows"] == len(selected) and training["training_row_ids"] == [r.id for r in selected]
            and training["training_row_ids_sha256"] == digest([r.id for r in selected])
            and training["selected_parameters"] == params == record["config"]["parameters"]
            and training["validation_rows"] == 0 and training["candidate_budget"] == 1
            and training["early_stopping"] is False and training["test_used_for_selection"] is False
            and training["library_versions"] == config["versions"], "Classical fitting budget/parameters differ")
        f = training["features"]
        require(f["fit_split"] == "selected training rows only" and f["matrix_format"] == "csr"
            and f["dense_conversion"] is False and f["validation_rows_seen"] == f["test_rows_seen_during_fit"] == 0
            and f["training_records_sha256"] == digest([asdict(r) for r in selected])
            and f["training_text_sha256"] == digest([r.text for r in selected]), "Classical feature provenance differs")
        fingerprint = f["matrix_sha256"], training["test_matrix_sha256"]
        require((name, budget) not in features or features[name, budget] == fingerprint, "Models use unequal TF-IDF matrices")
        features[name, budget] = fingerprint
        expected_test = {"dataset": name, "labels": dataset.labels, "manifest_sha256": digest(dataset.manifest),
            "rows": [{"id": r.id, "label": r.label, "text_sha256": hashlib.sha256(r.text.encode()).hexdigest(),
                      "normalized_text_sha256": hashlib.sha256(normalized_text(r.text).encode()).hexdigest()} for r in dataset.test]}
        require(read(path.with_name("test_manifest.json")) == expected_test, "Classical test identity differs")
        predictions = [Prediction(**json.loads(line)) for line in path.with_name("predictions.jsonl").read_text().splitlines()]
        result.append((dataset, record, predictions, model, budget, path))
    return result


def audit_text_ledger(root=ROOT):
    import run_text_jev_review as review
    ledger = review.TextLedger(review.budget.Ledger(review.LEDGER, review.STAGE_CAP))
    before = file_sha(review.LEDGER)
    audited = review.audit_global(ledger, allow_halted=True)
    events, charged_nano = ledger.inner._read()
    require(file_sha(review.LEDGER) == before, "Ledger changed during reporting; retry a stable snapshot")
    return {"checkpoints": audited, "events": events, "charged_nano": charged_nano,
            "ledger_sha256": before, "ledger_id": ledger.identity["ledger_id"]}


def cost_snapshot(rows, root=ROOT, audited_ledger=None):
    root = Path(root)
    path = root/"results/text_extension/review-budget.jsonl"
    if not path.exists():
        review_folder = root/"results/text_extension/review"
        require(not review_folder.is_symlink() and not any(
            candidate.is_file() or candidate.is_symlink() for candidate in review_folder.rglob("*")),
            "Review artifacts or partial files exist without a ledger")
    import run_text_jev_review as review
    numeric_path = root/"results/numeric_expansion/review-budget.jsonl"
    before = file_sha(numeric_path)
    # The reviewer pins the original numeric ledger prefix and identity as well
    # as the spending envelope. A replacement valid hash chain is not enough.
    snapshot = review.verify_envelope(root)
    require(file_sha(numeric_path) == before, "Numeric ledger changed during reporting; retry a stable snapshot")
    numeric_reserved = Decimal(snapshot["charged_or_reserved_usd"])
    require(numeric_reserved <= NUMERIC_ENVELOPE and FIXED_PRIOR+NUMERIC_ENVELOPE+STAGE_CAP <= TOTAL_CAP,
            "Combined study budget exceeds authorized envelope")
    cumulative = FIXED_PRIOR + numeric_reserved
    empty = {"status": "blocked_no_credit", "new_model_requests": 0, "reported_cost_requests": 0,
        "unknown_cost_requests": 0, "known_reported_api_usd": "0", "review_conservative_usd": "0",
        "prior_conservative_usd": str(cumulative), "cumulative_conservative_usd": str(cumulative),
        "cumulative_authorized_usd": str(TOTAL_CAP), "stage_authorized_usd": str(STAGE_CAP),
        "numeric_current_reserved_usd": str(numeric_reserved), "numeric_reserved_allowance_usd": str(NUMERIC_ENVELOPE),
        "new_llm_generation_calls": 0, "ledger_sha256": None, "numeric_ledger_sha256": before,
        "expected_new_review_conditions": 24, "maximum_new_model_requests": 4800,
        "complete_checkpoint_conditions": 0, "complete_summary_conditions": 0,
        "reservations_without_result": 0, "results_without_saved_prediction": 0,
        "partial_checkpoint_files": 0, "source_failure_rows_without_jev_call": 0,
        "reused_old_review_conditions": 0,
        "condition_costs": [{"dataset": dataset, "model_key": key, "shots_per_class": shots,
            "model_requests": 0, "reported_cost_requests": 0, "unknown_cost_requests": 0,
            "known_reported_api_usd": "0", "conservative_usd": "0"}
            for dataset in DATASETS for key in sources.MODELS for shots in (0,4)],
        "note": "No text-review ledger or model requests exist. Prior funding check found no purchased provider credits; funding must be checked again before execution. Historical and numeric reservations remain unchanged."}
    if not path.exists():
        return empty
    audit = audited_ledger or audit_text_ledger(root)
    events = audit["events"]
    reserves = {event["reservation_id"]: event for event in events if event["type"] == "reserve"}
    results = {event["reservation_id"]: event for event in events if event["type"] == "result"}
    settles = {event["reservation_id"]: event for event in events if event["type"] == "settle"}
    known = {ident: Decimal(str(event["details"]["reported_cost_usd"])) for ident,event in results.items()
             if event["details"].get("reported_cost_usd") is not None}
    amounts = {ident: settles[ident]["charged_nano"] if ident in settles else event["reserved_nano"] for ident,event in reserves.items()}
    require(sum(amounts.values()) == audit["charged_nano"], "Settled conservative accounting differs")
    charged = Decimal(audit["charged_nano"])/Decimal(1_000_000_000)
    require(charged <= STAGE_CAP and cumulative+charged <= TOTAL_CAP, "Text review budget exceeded")
    completed = {record["run_id"] for _,record,_ in audit["checkpoints"] if record["status"] == "complete"}
    measured = {row["run_id"] for row in rows if row["arm"] == "review" and row["status"] == "complete"}
    require(completed == measured, "Cost checkpoint and measured review inventory differ")
    halted = any(record["status"].startswith("stopped_") for _,record,_ in audit["checkpoints"])
    conditions = []
    for item in empty["condition_costs"]:
        ids = {ident for ident,event in reserves.items() if event["details"]["dataset"] == item["dataset"]
               and event["details"]["proposal_key"] == f"{item['model_key']}_k{item['shots_per_class']}"}
        known_ids = ids & set(known)
        conditions.append({**item,"model_requests":len(ids),"reported_cost_requests":len(known_ids),
            "unknown_cost_requests":len(ids-known_ids),"known_reported_api_usd":str(sum((known[i] for i in known_ids),Decimal(0))),
            "conservative_usd":str(Decimal(sum(amounts[i] for i in ids))/Decimal(1_000_000_000))})
    return {**empty,"status":"complete" if len(completed)==24 else "halted" if halted else "in_progress",
        "new_model_requests":len(reserves),"reported_cost_requests":len(known),"unknown_cost_requests":len(reserves)-len(known),
        "known_reported_api_usd":str(sum(known.values(),Decimal(0))),"review_conservative_usd":str(charged),
        "cumulative_conservative_usd":str(cumulative+charged),"ledger_sha256":audit["ledger_sha256"],
        "settled_requests":len(settles),"complete_checkpoint_conditions":len(completed),"complete_summary_conditions":len(measured),
        "partial_checkpoint_files":sum(record["status"]!='complete' for _,record,_ in audit["checkpoints"]),
        "source_failure_rows_without_jev_call":sum(p.metadata.get("jev_review_called") is False for _,_,ps in audit["checkpoints"] for p in ps),
        "condition_costs":conditions,"note":"Every text event and checkpoint is globally audited. Successful fully verified new responses may settle at the guarded conservative cost; failed or unknown responses retain their full reservation. Historical ledgers remain unchanged."}


def collect(root=ROOT, *, samples=2000):
    root = Path(root).resolve()
    require(root == ROOT and samples >= 100, "Use the canonical checkout and at least 100 bootstrap replicates")
    require(file_sha(sources.__file__) == SOURCE_VALIDATOR_SHA, "Frozen text validator changed")
    protocol = read(root/"results/text_extension/protocol.json")
    require(protocol == protocol_plan(root, frozen=protocol["review_execution_frozen"]), "Text protocol differs from current frozen sources")
    datasets = {name: sources.load_dataset(name) for name in DATASETS}
    audited, payloads = {}, {}
    def add(row, payload):
        key = row_key(row)
        require(key in expected_conditions() and key not in audited, "Duplicate/unexpected text condition")
        audited[key], payloads[key] = row, payload
    for dataset in datasets.values():
        for model, model_key in MODEL_KEYS.items():
            for shots in (0,4):
                job = sources.load_source(dataset, model_key, shots, freeze=False)
                if job is None:
                    continue
                path = root/job["source"]["path"]/"run.json"
                row, payload = audited_row(dataset, job["record"], job["predictions"], model=model,
                    arm="base", budget=shots, path=path, samples=samples, root=root)
                row.update(measurement_origin="new_local_text_extension" if model in NEW_MODELS else "reused_original_text_pilot",
                    content_audit=job["content_audit"], source_validator_sha256=SOURCE_VALIDATOR_SHA)
                add(row,payload)
        for shots in (0,4):
            job = sources.load_direct(dataset, shots, freeze=False)
            require(job is not None, "Frozen direct Jev reference missing")
            row, payload = audited_row(dataset,job["record"],job["predictions"],model=JEV,arm="direct",budget=shots,
                path=root/job["source"]["path"]/"run.json",samples=samples,root=root)
            row.update(measurement_origin="reused_original_text_pilot", content_audit=job["content_audit"],
                       source_validator_sha256=SOURCE_VALIDATOR_SHA)
            add(row,payload)
    for dataset,record,predictions,model,budget,path in classical_artifacts(root):
        row,payload = audited_row(dataset,record,predictions,model=model,arm="classical",budget=budget,path=path,samples=samples,root=root)
        row.update(measurement_origin="new_classical_text_extension", training=record["training"])
        add(row,payload)
    ledger_audit = audit_text_ledger(root) if (root/"results/text_extension/review-budget.jsonl").exists() else None
    checkpoints = [] if ledger_audit is None else ledger_audit["checkpoints"]
    require({path.parent for path in (root/"results/text_extension/review").glob("*/run.json")} == {Path(directory) for directory,_,_ in checkpoints}, "Unaudited review checkpoints exist")
    for directory,record,predictions in checkpoints:
        path=Path(directory)/"run.json"; cfg=record["config"]; model,shots=cfg["source_model"],cfg["shots_per_class"]
        key=record["dataset"],model,"review",shots
        require(key in expected_conditions() and key not in audited, "Duplicate/unplanned text review")
        require(protocol["review_execution_frozen"], "Review execution lacks a frozen study protocol")
        if record["status"] != "complete":
            row=placeholder(key)
            row.update(status=record["status"],run_id=record["run_id"],source_path=path.relative_to(root).as_posix())
            audited[key]=row
            continue
        row,payload=audited_row(datasets[record["dataset"]],record,predictions,model=model+"+jev_review",arm="review",
            budget=shots,path=path,samples=samples,root=root)
        row.update(source_run_id=cfg["source_run_id"],measurement_origin="new_text_review")
        add(row,payload)
    report=assemble(audited,payloads,samples=samples)
    report.update(source_sha256=file_sha(__file__),protocol_sha256=file_sha(root/"results/text_extension/protocol.json"),
        source_validator_sha256=SOURCE_VALIDATOR_SHA,prepared_manifests={name:digest(ds.manifest) for name,ds in datasets.items()},
        review_execution_frozen=protocol["review_execution_frozen"])
    report["costs"]=cost_snapshot(report["runs"],root,ledger_audit)
    return report


def write(report, root=ROOT):
    folder=Path(root)/"results/text_extension"
    save_json(folder/"COMPARISON.json",report)
    rows={row_key(row):row for row in report["runs"]}
    pairs={(row["a"],row["kind"]):row for row in report["comparisons"]}
    lines=["# Text classification and Jev review", "",
        f"**{report['complete_runs']}/68 conditions are complete and audited.** LLM alone: {report['complete_base_runs']}/24; LLM → Jev: {report['complete_review_runs']}/24; Jev alone: {report['complete_direct_jev_runs']}/4; classical references: {report['complete_classical_runs']}/16.","",
        "SST-2 has two sentiment classes; TREC has six question categories. Each uses the same frozen 200 held-out rows across every arm. Zero-shot supplies no labels; few-shot supplies exactly four examples per class (8 SST-2 and 24 TREC labels), with identical ordered examples at the source and reviewer stages.","",
        "Missing or partial conditions have no score, confidence interval or inferred improvement. Cached LLM labels and direct Jev measurements are reused after full-content and matched-training audits; original cross-environment manifest hashes are preserved.","",
        "[Full metrics and paired contrasts](COMPARISON.json) · [Protocol](protocol.json) · [Classical evidence](classical/README.md)",""]
    if report['complete_review_runs']==0:
        lines += ["**No LLM → Jev text-review effect has been measured yet.** Completed direct-Jev and classical scores cannot answer whether Jev improves an LLM proposal. Review calls await funded provider credit and the guarded execution stage.",""]
    for dataset in DATASETS:
        for shots in (0,4):
            direct=rows[dataset,JEV,"direct",shots]
            lines += [f"## {NAMES[dataset]} · {'zero-shot' if shots==0 else 'four examples per class'}","",
                f"Jev alone: **{pct(direct['accuracy'])} accuracy** on 200 rows.","",
                "| LLM | LLM alone | LLM → Jev | Fixed / harmed | Review − LLM, pp [95% CI] | Review − Jev alone, pp [95% CI] |",
                "|---|---:|---:|---:|---|---|"]
            for model in MODEL_NAMES:
                base,review=rows[dataset,model,"base",shots],rows[dataset,model,"review",shots]
                diff=pairs[review['run_id'],'review_minus_source']; t=diff.get('transitions')
                fixed=f"{t['wrong_to_correct']} / {t['correct_to_wrong']}" if t else '—'
                lines.append(f"| {MODEL_NAMES[model]} | {pct(base['accuracy'])} | {pct(review['accuracy'])} | {fixed} | {interval(diff)} | {interval(pairs[review['run_id'],'review_minus_jev_alone'])} |")
            lines.append("")
    for budget,title in ((4,"Classical ML · same four labels per class"),(None,"Classical ML · full prepared training, additional labels")):
        lines += [f"## {title}","","| Dataset | Method | Training labels | Accuracy | Macro-F1 | Failures |","|---|---|---:|---:|---:|---:|"]
        for dataset in DATASETS:
            for model in CLASSICAL_NAMES:
                r=rows[dataset,model,"classical",budget]
                lines.append(f"| {NAMES[dataset]} | {CLASSICAL_NAMES[model]} | {r['train_labels']} | {pct(r['accuracy'])} | {r['macro_f1']:.4f} | {r['n_failures']} |")
        lines.append("")
    costs=report['costs']
    lines += ["## Scope and costs","",
        f"New Jev text-review requests: **{costs['new_model_requests']}**. Known reported cost: **${costs['known_reported_api_usd']}**; unknown-cost calls: **{costs['unknown_cost_requests']}**. Cumulative conservative accounting, including earlier work: **${costs['cumulative_conservative_usd']} / ${costs['cumulative_authorized_usd']}**. The new text stage is capped at $1.60 and preserves the remaining numerical-review allocation.","",
        "Known provider charges and conservative reservations are different quantities. Local compute is unpriced; historical hosted source generation is already included in earlier accounting. A deployed chain pays for both stages.","",
        "## Interpretation limits","",
        "This exploratory extension was chosen after viewing earlier text-pilot results. It uses one split and example-selection seed on two familiar public datasets; pretraining exposure cannot be excluded. Equal newly supplied labels do not mean equal total training exposure.","",
        "Jev receives the original text, the same examples and a cached proposed class from a separate model. This does not isolate the causal effect of bounded decoding on the same LLM. Compare each pipeline with both the source and Jev alone; count corrected and harmed predictions. Failed source proposals trigger no review call, and failed reviews receive no fallback label.","",
        "The four classical models share training-only sparse word/character TF-IDF features and fixed hyperparameters. Full prepared training uses 10,000 SST-2 or 4,886 TREC labels and excludes validation labels. This is not an optimized text leaderboard. XGBoost treats unstored sparse entries as missing; other estimators treat them as zero. Four retained LightGBM wrapper feature-name warnings do not change the fixed vectorizer column ordering.","",
        f"Intervals use {report['bootstrap_samples']} unstratified group-bootstrap resamples with seed 42 and normalized-text groups; both tests contain 200 distinct groups. Paired intervals condition on the split, selected examples and fitted models, and have no multiple-comparison adjustment. Few-shot-minus-zero-shot contrasts use unequal newly supplied label budgets. Restricted-label likelihoods, native Choice probabilities and classical probabilities are not guaranteed calibrated.","",
        "Original Qwen/OpenAI sources retain their historical prompt rendering. New SmolLM2/Granite sources use the explicit fixed-system likelihood recipe. Model differences therefore do not isolate architecture alone. No new LoRA training is included.",""]
    (folder/"FINDINGS.md").write_text("\n".join(lines))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--bootstrap-samples",type=int,default=2000)
    parser.add_argument("--freeze-protocol",action="store_true",help="Pin final review wrapper before the first paid review")
    args=parser.parse_args(argv)
    prepare_protocol(args.root,freeze=args.freeze_protocol)
    report=collect(args.root,samples=args.bootstrap_samples)
    write(report,args.root)
    print(json.dumps({k:report[k] for k in ('status','complete_runs','complete_review_runs','complete_comparisons','review_execution_frozen')}))


if __name__=='__main__':
    main()
