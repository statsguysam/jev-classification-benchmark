#!/usr/bin/env python3
"""Reconcile the mixed tabular ledger, including unfinished and resumed attempts.

No requests, fitting, settlement, or ledger mutation. A short ledger lock captures
one consistent ledger prefix and prediction-file snapshot while workers run.
Unpersisted outcomes and requests without final outcomes remain fully accounted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

import run_tabular_hosted as hosted
from tabular_data import load_native_prepared
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import digest, save_json

ROOT = Path(__file__).resolve().parents[1]
budget, route = hosted.budget, hosted.jev_route
DATASETS = ("titanic", "breast_cancer", "wine")
MODEL_KEYS = ("jev_openrouter", "openai_economical", "openai_frontier")
NANO, MILLION = Decimal(10**9), Decimal(10**6)


def require(value, message):
    if not value:
        raise ValueError(message)


def money(value, field):
    require(isinstance(value, str), f"{field}: explicit decimal string required")
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"{field}: invalid decimal") from None
    require(result.is_finite() and result >= 0, f"{field}: invalid nonnegative amount")
    return result


def file_sha(content):
    return hashlib.sha256(content).hexdigest()


def price_for(config):
    # Historical analysis validates the frozen price declaration; it must remain
    # reproducible after the execution date, without permitting new paid calls.
    return (dict(route.PRICE) if config["provider"] == "jev" else
            {**budget.PRICES[config["model"]], "verification_kind": "bundled_official_reference"})


def expected_guard(dataset, shots, key, ledger_id, sources, configs):
    result = {"tabular_wrapper_sha256": sources["tabular_wrapper_sha256"],
        "wrapper_sha256": hosted.FROZEN_ROUTE_SHA, "ledger_driver_sha256": route.FROZEN_BUDGET_SHA256,
        "ledger_id": ledger_id, "prior_ledgers": hosted.PRIOR,
        "prior_charged_or_reserved_usd": hosted.PRIOR_TOTAL, "stage_cap_usd": hosted.STAGE_CAP,
        "aggregate_authorized_usd": hosted.TOTAL_CAP, "dataset": dataset, "seed": 42,
        "shots_per_class": shots, "model_key": key}
    if key == "jev_openrouter":
        result.update(route="OpenRouter", endpoint=route.ENDPOINT, price_verified_on=route.VERIFIED_ON,
            price_source=route.PRICE["source"], response_model_allowlist=sorted(route.ALLOWED_RESPONSE_MODELS),
            native_protocol="frozen_JevClassifier_Choice", reserve_checks_included_in_latency=True)
    else:
        price = price_for(configs[key])
        result.update(pricing_sha256=budget.sha(price), verified_on=price["verified_on"])
    return result


def parse_predictions(content, complete):
    lines = content.splitlines()
    rows, partial = [], False
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeError):
            if index == len(lines)-1 and not complete and not content.endswith(b"\n"):
                partial = True
            else:
                raise ValueError("Corrupt prediction JSONL before the unfinished final line") from None
    return rows, partial


def capture(root=ROOT):
    """Capture files under the existing ledger lock, never reset/create a ledger."""
    root = Path(root).resolve()
    prior = hosted.verify_prior(root)
    sources = {
        "tabular_wrapper_sha256": file_sha((root / "scripts/run_tabular_hosted.py").read_bytes()),
        "jev_wrapper_sha256": file_sha((root / "scripts/run_openrouter_jev.py").read_bytes()),
        "ledger_driver_sha256": file_sha((root / "scripts/run_budgeted_hosted.py").read_bytes()),
    }
    require(sources["jev_wrapper_sha256"] == hosted.FROZEN_ROUTE_SHA and
            sources["ledger_driver_sha256"] == route.FROZEN_BUDGET_SHA256, "Frozen transport/ledger code differs")
    configs = hosted.validate_models(root / "configs/tabular_hosted.json", list(MODEL_KEYS))
    prepared = {name: load_native_prepared(root / "data/tabular-full" / name)[0] for name in DATASETS}
    ledger_path = root / "results/tabular/api-budget.jsonl"
    ledger = budget.Ledger(ledger_path, hosted.STAGE_CAP)
    artifacts, hashes = [], {}
    with ledger._locked():
        events, accounted_nano = ledger._read()
        for path in (ledger.path, ledger.lock_path):
            hashes[path.relative_to(root).as_posix()] = file_sha(path.read_bytes())
        for path in sorted((root / "results/tabular/hosted").glob("*/run.json")):
            raw = path.read_bytes()
            record = json.loads(raw)
            hashes[path.relative_to(root).as_posix()] = file_sha(raw)
            predicted = path.with_name("predictions.jsonl")
            content = predicted.read_bytes() if predicted.exists() else b""
            rows, partial = parse_predictions(content, record.get("status") == "complete")
            if predicted.exists():
                hashes[predicted.relative_to(root).as_posix()] = file_sha(content)
            test_path = path.with_name("test_manifest.json")
            test = None
            if test_path.exists():
                content = test_path.read_bytes()
                hashes[test_path.relative_to(root).as_posix()] = file_sha(content)
                test = json.loads(content)
            artifacts.append({"record": record, "rows": rows, "test": test,
                "partial_final_prediction_line": partial, "source_path": path.relative_to(root).as_posix()})
        captured_at = datetime.now(timezone.utc).isoformat()
    return {"events": events, "accounted_nano": accounted_nano, "ledger_id": ledger.identity["ledger_id"],
        "captured_at": captured_at, "artifacts": artifacts, "files_sha256": hashes, "sources": sources,
        "prior": prior}, prepared, configs


def unique_events(events, kind):
    chosen = [event for event in events if event["type"] == kind]
    result = {event["reservation_id"]: event for event in chosen}
    require(len(chosen) == len(result), f"Duplicate {kind} event for one reservation")
    return result


def usage_value(value, name):
    require(value is None or (type(value) is int and value >= 0), f"Invalid {name} token usage")
    return value


def analyze(snapshot, prepared, configs):
    """Pure reconciliation of an already captured, hash-validated ledger prefix."""
    events, ledger_id = snapshot["events"], snapshot["ledger_id"]
    reserves = unique_events(events, "reserve")
    results, settlements = unique_events(events, "result"), unique_events(events, "settle")
    require(set(results) <= set(reserves) and set(settlements) <= set(results), "Outcome/settlement lacks its reservation or result")
    contexts, accounting = {}, {}
    for name, dataset in prepared.items():
        for shots in (0, 4):
            examples = select_examples(dataset.train, dataset.labels, shots, 42)
            for key, config in configs.items():
                guard = expected_guard(name, shots, key, ledger_id, snapshot["sources"], configs)
                contexts[name, shots, key] = {"dataset": dataset, "examples": examples,
                    "rows": {r.id: r for r in dataset.test}, "config": config,
                    "guard": guard, "price": price_for(config)}
    request_ids = set()
    for rid, reservation in reserves.items():
        details = reservation["details"]
        condition = (details.get("dataset"), details.get("shots_per_class"), details.get("model_key"))
        require(condition in contexts, "Reservation condition is outside the declared hosted matrix")
        context = contexts[condition]
        config, price, guard = context["config"], context["price"], context["guard"]
        require(all(details.get(key) == value for key, value in guard.items()), "Reservation provenance differs from frozen wrapper/budget")
        require(details.get("provider") == config["provider"] and details.get("model") == config["model"], "Reserved provider/model differs")
        require(details.get("row_id") in context["rows"], "Reserved row is outside the frozen test fold")
        prompt = build_prompt(context["rows"][details["row_id"]], context["dataset"].labels, context["examples"])
        require(details.get("prompt_sha256") == file_sha(prompt.encode()), "Reserved prompt differs from serialized test row/demonstrations")
        require(details.get("pricing") == price, "Reserved pricing differs from frozen declaration")
        expected_nano, bounds = ((route.RESERVE_NANO, None) if config["provider"] == "jev" else budget.reservation(config, price, prompt))
        require(reservation["reserved_nano"] == expected_nano, "Reservation amount does not match its prompt/token bound")
        require(bounds is None or details.get("bound") == bounds, "Reservation token bounds differ")
        result = results.get(rid, {}).get("details")
        settlement = settlements.get(rid)
        usage, api_cost, token_estimate = {"input_tokens": None, "output_tokens": None}, None, None
        if result is not None:
            require(result.get("outcome") in {"returned", "prediction_error", "raised"}, "Unknown request outcome")
            require(not result.get("usage_exceeded_reservation_assumptions"), "Ledger halted after a usage/route guard violation; billing must be reconciled")
            if result["outcome"] == "raised":
                require(result.get("usage_unknown") is True and settlement is None, "Raised request has invalid usage/settlement")
            else:
                usage = result.get("usage", {})
                require(set(usage) == {"input_tokens", "output_tokens"}, "Incomplete usage schema")
                inputs, outputs = usage_value(usage["input_tokens"], "input"), usage_value(usage["output_tokens"], "output")
                if config["provider"] == "jev":
                    require(settlement is None and result.get("reservation_released") is False, "Jev reservation must remain retained")
                    require(result.get("route") == "OpenRouter" and not any(result.get("guard_reasons", {}).values()), "Jev route guard differs")
                    require(result.get("resolved_model") in route.ALLOWED_RESPONSE_MODELS or (result["outcome"] == "prediction_error" and result.get("resolved_model") is None), "Unrecognized returned Jev model")
                    require(result.get("provider") == "TypeSafe" or (result["outcome"] == "prediction_error" and result.get("provider") is None), "Unrecognized returned Jev provider")
                    require(inputs is None or inputs <= price["reserve_input_tokens"], "Jev input usage exceeds bound")
                    request_id = result.get("request_id")
                    if request_id is not None:
                        require(isinstance(request_id, str) and bool(request_id) and request_id not in request_ids, "Duplicate/invalid Jev API request ID")
                        request_ids.add(request_id)
                    require(result["outcome"] != "returned" or request_id is not None, "Successful Jev result lacks request ID")
                    if result.get("reported_cost_usd") is not None:
                        api_cost = money(result["reported_cost_usd"], "Jev API cost")
                        require(api_cost <= Decimal(expected_nano)/NANO, "Reported Jev cost exceeds reserved bound")
                    if inputs is not None:
                        token_estimate = Decimal(inputs)*Decimal(price["input_usd_per_million"])/MILLION
                else:
                    require((inputs is None or inputs <= bounds["input_token_bound"]) and (outputs is None or outputs <= bounds["output_token_cap"]), "OpenAI usage exceeds reserved token bound")
                    if inputs is not None and outputs is not None:
                        token_estimate = Decimal(budget.usd_nano((Decimal(inputs)*Decimal(price["input_usd_per_million"])+Decimal(outputs)*Decimal(price["output_usd_per_million"]))/MILLION))/NANO
                    estimate = result.get("reported_usage_estimate")
                    require((estimate is None) == (token_estimate is None), "Reported-token estimate coverage differs")
                    if estimate is not None:
                        require(money(estimate["usd"], "reported-token estimate") == token_estimate, "Reported-token estimate differs from usage")
        if settlement is not None:
            require(config["provider"] == "openai" and result["outcome"] == "returned", "Only successful OpenAI results may settle")
            inputs, outputs = usage["input_tokens"], usage["output_tokens"]
            require(type(inputs) is int and inputs > 0 and type(outputs) is int and outputs > 0, "Settlement requires complete positive usage")
            actual = result.get("resolved_model")
            require(isinstance(actual, str) and (actual == config["model"] or re.fullmatch(re.escape(config["model"])+r"-\d{4}-\d{2}-\d{2}", actual)), "Settlement model differs from priced model")
            charge = budget.usd_nano((Decimal(inputs)*Decimal(price["input_usd_per_million"])*Decimal(price["input_multiplier"])+Decimal(outputs)*Decimal(price["output_usd_per_million"]))/MILLION)
            require(settlement["charged_nano"] == charge and 0 <= charge <= expected_nano, "Settlement differs from conservative token-rate calculation")
            evidence = settlement.get("evidence", {})
            require(evidence.get("usage") == usage and evidence.get("model") == actual and evidence.get("pricing_sha256") == budget.sha(price), "Settlement evidence differs from result/pricing")
        accounting[rid] = {"condition": condition, "provider": config["provider"], "model": config["model"],
            "result": result, "usage": usage, "reported_api_cost": api_cost, "token_estimate": token_estimate,
            "accounted_nano": settlement["charged_nano"] if settlement else expected_nano,
            "reserved_nano": expected_nano, "settled": settlement is not None}
    seen, run_conditions, runs = set(), set(), []
    for artifact in snapshot["artifacts"]:
        record, rows = artifact["record"], artifact["rows"]
        config = record["config"]
        guard = config["budget_guard"]
        condition = (record["dataset"], config["shots_per_class"], guard["model_key"])
        require(condition in contexts and condition not in run_conditions, "Duplicate/unknown hosted run condition")
        run_conditions.add(condition)
        context = contexts[condition]
        dataset = context["dataset"]
        require(guard == context["guard"] and config == {**context["config"], "budget_guard": guard, "shots_per_class": condition[1]}, "Run configuration/provenance differs")
        require(record["seed"] == 42 and record["method"] == ("zero_shot" if condition[1] == 0 else "few_shot"), "Run seed/method differs")
        require(record["labels"] == dataset.labels and record["manifest_sha256"] == digest(dataset.manifest) and record["dataset_manifest"] == dataset.manifest, "Run frozen prepared manifest differs")
        require(record["training_example_ids"] == [r.id for r in context["examples"]], "Run demonstration IDs differ")
        require([row["row_id"] for row in rows] == [row.id for row in dataset.test[:len(rows)]] and len(rows) <= len(dataset.test), "Persisted predictions are not the ordered test prefix")
        complete = record.get("status") == "complete"
        if complete:
            expected_test = {"dataset": dataset.name, "labels": dataset.labels, "manifest_sha256": digest(dataset.manifest),
                "rows": [{"id": row.id, "label": row.label, "text_sha256": file_sha(row.text.encode())} for row in dataset.test]}
            require(len(rows) == len(dataset.test) and artifact["test"] == expected_test and not artifact["partial_final_prediction_line"], "Complete run lacks complete frozen test artifacts")
        failed, no_dispatch = 0, 0
        for row in rows:
            is_failed = row.get("error") is not None
            require(is_failed or (type(row.get("label")) is int and 0 <= row["label"] < len(dataset.labels)), "Successful prediction has an invalid label")
            failed += int(is_failed)
            meta = row.get("metadata", {})
            if "budget" not in meta:
                require(is_failed and row.get("input_tokens") is None and row.get("output_tokens") is None, "Prediction without reservation cannot claim successful/billable usage")
                no_dispatch += 1
                continue
            pay = meta["budget"]; rid = pay.get("reservation_id")
            require(rid in accounting and rid not in seen, "Duplicate or unmatched prediction reservation")
            seen.add(rid)
            item, reserve = accounting[rid], reserves[rid]
            outcome = item["result"]
            require(item["condition"] == condition and reserve["details"]["row_id"] == row["row_id"], "Prediction condition/row differs from reservation")
            require(pay.get("ledger_id") == ledger_id and money(pay["reserved_upper_bound_usd"], "prediction reservation") == Decimal(item["reserved_nano"])/NANO, "Prediction ledger/reservation amount differs")
            require(outcome is not None and outcome["outcome"] == ("prediction_error" if is_failed else "returned"), "Prediction/result outcome differs")
            require(outcome["usage"] == {"input_tokens": row.get("input_tokens"), "output_tokens": row.get("output_tokens")}, "Prediction/result token usage differs")
            require(meta.get("resolved_model") == outcome.get("resolved_model"), "Prediction/result model differs")
            if item["provider"] == "jev":
                require(pay.get("reservation_released") is False and "settled_conservative_usd" not in pay, "Prediction incorrectly releases Jev reservation")
                routed = meta.get("openrouter", {})
                require(routed.get("endpoint") == route.ENDPOINT and routed.get("id") == outcome.get("request_id") and routed.get("provider") == outcome.get("provider"), "Prediction/ledger Jev request ID or route differs")
                cost = None if routed.get("reported_cost_usd") is None else money(routed["reported_cost_usd"], "prediction Jev API cost")
                require(cost == item["reported_api_cost"], "Prediction/ledger API cost differs")
            else:
                require(pay.get("reported_usage_estimate") == outcome.get("reported_usage_estimate"), "Prediction/ledger token estimate differs")
                require(all(pay.get(key) == value for key, value in reserve["details"]["bound"].items()), "Prediction token reservation bounds differ")
                charged = pay.get("settled_conservative_usd")
                require((charged is not None) == item["settled"], "Prediction/ledger settlement coverage differs")
                if charged is not None:
                    require(money(charged, "prediction settlement") == Decimal(item["accounted_nano"])/NANO, "Prediction/ledger settlement amount differs")
        if complete:
            require(record["metrics"]["n_test"] == len(rows) and record["metrics"]["n_failures"] == failed, "Complete run counts differ from predictions")
        runs.append({"dataset": condition[0], "model": context["config"]["model"], "shots_per_class": condition[1],
            "model_key": condition[2],
            "run_id": record["run_id"], "status": record.get("status"), "expected_predictions": len(dataset.test),
            "persisted_predictions": len(rows), "prediction_failures": failed, "predictions_without_dispatch": no_dispatch,
            "partial_final_prediction_line": artifact["partial_final_prediction_line"], "source_path": artifact["source_path"]})
    def totals(items):
        items = list(items)
        n = len(items)
        api_known = [item["reported_api_cost"] for item in items if item["reported_api_cost"] is not None]
        token_known = [item["token_estimate"] for item in items if item["token_estimate"] is not None]
        api_sum, token_sum = sum(api_known, Decimal(0)), sum(token_known, Decimal(0))
        result = {"reservations": n, "final_outcomes": sum(item["result"] is not None for item in items),
            "requests_without_final_outcome": sum(item["result"] is None for item in items),
            "settlements": sum(item["settled"] for item in items),
            "retained_full_reservations": sum(not item["settled"] for item in items),
            "accounted_conservative_usd": str(Decimal(sum(item["accounted_nano"] for item in items))/NANO),
            "initial_reservations_before_settlements_usd": str(Decimal(sum(item["reserved_nano"] for item in items))/NANO),
            "reported_api_cost_known_requests": len(api_known), "reported_api_cost_unknown_requests": n-len(api_known),
            "reported_api_cost_known_subtotal_usd": str(api_sum), "reported_api_cost_usd": str(api_sum) if n and len(api_known) == n else None,
            "reported_api_cost_coverage": len(api_known)/n if n else None,
            "token_rate_estimate_known_requests": len(token_known), "token_rate_estimate_unknown_requests": n-len(token_known),
            "token_rate_estimate_known_subtotal_usd": str(token_sum), "token_rate_estimate_usd": str(token_sum) if n and len(token_known) == n else None,
            "token_rate_estimate_coverage": len(token_known)/n if n else None,
            "outcomes": dict(Counter(item["result"]["outcome"] if item["result"] else "pending_or_interrupted" for item in items))}
        for token in ("input_tokens", "output_tokens"):
            known = [item["usage"][token] for item in items if item["usage"][token] is not None]
            result[token+"_known_total"] = sum(known)
            result[token+"_known_requests"] = len(known)
            result[token] = sum(known) if n and len(known) == n else None
        return result
    summary = totals(accounting.values())
    by_condition = [{"dataset": key[0], "shots_per_class": key[1], "model_key": key[2],
        "model": context["config"]["model"], "expected_prediction_rows": len(context["dataset"].test),
        **totals(item for item in accounting.values() if item["condition"] == key)}
        for key, context in contexts.items()]
    require(money(summary["accounted_conservative_usd"], "recomputed total") == Decimal(snapshot["accounted_nano"])/NANO, "Recomputed request accounting differs from ledger")
    combined = Decimal(hosted.PRIOR_TOTAL)+Decimal(snapshot["accounted_nano"])/NANO
    require(Decimal(snapshot["accounted_nano"])/NANO <= Decimal(hosted.STAGE_CAP) and combined <= Decimal(hosted.TOTAL_CAP), "Authorized cumulative allocation exceeded")
    expected_requests = sum(len(dataset.test) for dataset in prepared.values())*2*len(configs)
    completed_runs = sum(row["status"] == "complete" for row in runs)
    unrepresented = set(accounting)-seen
    return {"schema_version": 1, "captured_at": snapshot["captured_at"],
        "basis": "Conservative ledger accounting plus separately labeled API-reported Jev dollars and standard-rate token estimates. None is an invoice; unrelated account activity, taxes and compute are excluded.",
        "status": "complete" if completed_runs == len(contexts) and not unrepresented else "in_progress_or_incomplete",
        "ledger_id": ledger_id, "ledger_events": len(events), "ledger_head_sha256": events[-1]["event_sha256"],
        "expected_runs": len(contexts), "complete_runs": completed_runs, "expected_prediction_rows": expected_requests,
        "persisted_prediction_rows": sum(run["persisted_predictions"] for run in runs),
        "predictions_without_dispatch": sum(run["predictions_without_dispatch"] for run in runs),
        "prediction_reservations_reconciled": len(seen), "reservations_without_persisted_prediction": len(unrepresented),
        "unpersisted_reservation_ids": sorted(unrepresented),
        "prior_text_study_accounted_usd": hosted.PRIOR_TOTAL, "tabular_stage_cap_usd": hosted.STAGE_CAP,
        "cumulative_authorized_cap_usd": hosted.TOTAL_CAP, "cumulative_accounted_conservative_usd": str(combined),
        "remaining_tabular_reservation_capacity_usd": str(Decimal(hosted.STAGE_CAP)-Decimal(snapshot["accounted_nano"])/NANO),
        "remaining_cumulative_capacity_usd": str(Decimal(hosted.TOTAL_CAP)-combined),
        "tabular": summary, "by_model": {config["model"]: totals(item for item in accounting.values() if item["model"] == config["model"]) for config in configs.values()},
        "runs": runs, "by_condition": by_condition, "prior_ledger_snapshots": snapshot.get("prior", {}), "sources": snapshot["sources"],
        "files_sha256_at_capture": snapshot.get("files_sha256", {}),
        "request_id_note": "Jev response IDs are checked against ledger outcomes and must be unique. The frozen OpenAI adapter did not retain provider request IDs; its durable reservation IDs, usage, model and settlements are reconciled instead.",
        "snapshot_note": "Ledger and prediction files captured while holding the existing ledger lock. Outcomes without persisted predictions, including interrupted or currently active requests, remain included at their full reservation or validated settlement."}


def build_report(root=ROOT):
    snapshot, prepared, configs = capture(root)
    return analyze(snapshot, prepared, configs)


def write_report(report, root=ROOT):
    root = Path(root)
    save_json(root / "results/tabular/API_COST_SUMMARY.json", report)
    total = report["tabular"]
    lines = ["# Tabular API cost reconciliation", "",
        f"Snapshot: {report['captured_at']}. **{report['complete_runs']}/{report['expected_runs']} hosted runs complete**, with {report['persisted_prediction_rows']:,}/{report['expected_prediction_rows']:,} persisted prediction rows.", "",
        f"The tabular ledger conservatively accounts for **US${Decimal(total['accounted_conservative_usd']):.9f}**. Including the unchanged text-study US${Decimal(report['prior_text_study_accounted_usd']):.9f}, cumulative accounting is **US${Decimal(report['cumulative_accounted_conservative_usd']):.9f}**, within the approved **US${report['cumulative_authorized_cap_usd']}** ceiling. The tabular allocation is US${report['tabular_stage_cap_usd']}.", "",
        f"All {total['reservations']:,} captured reservations are included: {total['settlements']:,} validated settlements and {total['retained_full_reservations']:,} full retained reservations. {total['requests_without_final_outcome']} requests lack a final outcome; {report['reservations_without_persisted_prediction']} reservations lack a persisted prediction. These can be active or interrupted attempts and are never discarded on resume.", "",
        "API-reported dollar costs are available only for some Jev responses. OpenAI amounts are independently recomputed standard-rate token estimates. Missing costs/usage remain unknown; their known subtotals do not represent a complete bill. Conservative accounting retains unresolved reserves. None of these figures is a provider invoice.", "",
        "| Model | Reservations / settled | Known API-cost requests | Known API-cost subtotal USD | Known token-estimate requests | Token-estimate subtotal USD | Conservative accounted USD |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model, item in report["by_model"].items():
        lines.append(f"| {model} | {item['reservations']} / {item['settlements']} | {item['reported_api_cost_known_requests']}/{item['reservations']} | {Decimal(item['reported_api_cost_known_subtotal_usd']):.9f} | {item['token_rate_estimate_known_requests']}/{item['reservations']} | {Decimal(item['token_rate_estimate_known_subtotal_usd']):.9f} | {Decimal(item['accounted_conservative_usd']):.9f} |")
    lines += ["", report["request_id_note"], "", report["snapshot_note"], "",
        "[Machine-readable snapshot](tabular/API_COST_SUMMARY.json). Preserve the canonical ledger and its `.lock` anchor together. This report performs no API calls or settlements. Refresh with `python scripts/summarize_tabular_costs.py`; a snapshot taken during execution is deliberately incomplete.", ""]
    (root / "results/TABULAR_API_COST_SUMMARY.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = build_report(args.root)
    write_report(report, args.root)
    print(json.dumps({key: report[key] for key in ("status", "complete_runs", "expected_runs", "persisted_prediction_rows", "cumulative_accounted_conservative_usd")}))


if __name__ == "__main__":
    main()
