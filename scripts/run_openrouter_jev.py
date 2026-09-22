"""Guarded native Jev Choice runs through OpenRouter, preserving frozen core code.

Dry-run by default. Each network request reserves $0.002688 in the existing
durable ledger implementation; no releases, redirects, retries or model fallbacks.
Reported OpenRouter cost is recorded separately from the reservation. Published
prices bound the intended calls conditionally, not other account activity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import threading
import urllib.error
import urllib.request
from unittest.mock import patch

_BUDGET_PATH = Path(__file__).with_name("run_budgeted_hosted.py")
_SPEC = importlib.util.spec_from_file_location("jev_frozen_budget_driver", _BUDGET_PATH)
budget = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(budget)
FROZEN_BUDGET_SHA256 = "3a6447218cba5025fef3aa58eb56d9045bd89173a35c4150544a87e7a55b6f92"
MODEL = "typesafe/jev-1.13"
BASE_URL = "https://openrouter.ai/api/v1"
ENDPOINT = BASE_URL + "/systemone"
VERIFIED_ON = "2026-09-22"
ALLOWED_RESPONSE_MODELS = {MODEL, "typesafe/jev-1.13-20260917"}
PRICE = {"input_usd_per_million": "0.042", "output_usd_per_million": "0",
         "reserve_input_tokens": 64000, "catalog_context_tokens": 32000,
         "per_request_reserved_usd": "0.002688000", "verified_on": VERIFIED_ON,
         "source": "https://openrouter.ai/typesafe/jev-1.13",
         "endpoint_source": "https://openrouter.ai/docs/guides/community/typesafe-sdk"}
RESERVE_NANO = budget.usd_nano(PRICE["per_request_reserved_usd"])
_LOCK = threading.RLock()


class StopRequested(RuntimeError):
    """Stop before a request without writing a failed prediction."""


def validate_config(config):
    required = {"provider": "jev", "model": MODEL, "base_url": BASE_URL, "api_key_env": "OPENROUTER_API_KEY"}
    if set(config)-set(required)-{"timeout"} or any(config.get(key) != value for key, value in required.items()):
        raise budget.GuardError("Only the pinned Jev model, official System One endpoint and OPENROUTER_API_KEY environment are accepted")
    timeout = config.get("timeout", 120)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise budget.GuardError("A finite positive timeout is required")
    return dict(config)


def parse_cost(value):
    """Keep zero and decimal precision without treating missing cost as zero."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise budget.GuardError("Invalid reported OpenRouter cost")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise budget.GuardError("Invalid reported OpenRouter cost") from None
    if not number.is_finite() or number < 0:
        raise budget.GuardError("Invalid reported OpenRouter cost")
    return str(number)


def post_json_exact_cost(url, body, headers, timeout):
    """Keep the wire cost's decimal precision; preserve core float probabilities."""
    from jevbench.providers import _NoRedirect, ProviderError
    request = urllib.request.Request(url, data=json.dumps(body, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read()
            result = json.loads(raw)
            exact = json.loads(raw, parse_float=Decimal)
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"http_error: status={exc.code}; no automatic retry") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ProviderError("network_error: connection failed or timed out; no automatic retry") from None
    except (ValueError, UnicodeError):
        raise ProviderError("invalid_output: API returned invalid JSON") from None
    if not isinstance(result, dict):
        raise ProviderError("invalid_output: API response is not an object")
    if isinstance(exact.get("usage"), dict) and "cost" in exact["usage"]:
        cost = exact["usage"]["cost"]
        result["usage"]["cost"] = str(cost) if isinstance(cost, Decimal) else cost
    return result


class GuardedOpenRouterJev:
    def __init__(self, config, ledger, provenance, quota=None):
        from jevbench.providers import JevClassifier
        self.config = validate_config(config)
        self.inner = JevClassifier(self.config)
        self.ledger, self.provenance = ledger, provenance
        self.quota = quota if quota is not None else {"count": 0, "limit": None}

    def predict(self, row, labels, prompt):
        from jevbench import providers
        captured, reservations = {}, []

        def guarded_post(url, body, headers, timeout):
            if reservations or url != ENDPOINT or set(body) != {"model", "state", "questions"} or body["model"] != MODEL or body["state"] != prompt:
                raise budget.GuardError("Unexpected endpoint, model, state or repeated request; no HTTP call made")
            expected_question = {"type": "choice",
                "instructions": "Choose the numeric class id for the final text in the state, following its classification task and examples.",
                "criteria": {str(i): label for i, label in enumerate(labels)}}
            if body["questions"] != {"classification": expected_question}:
                raise budget.GuardError("Request differs from the frozen native Choice protocol")
            if self.quota["limit"] is not None and self.quota["count"] >= self.quota["limit"]:
                raise StopRequested("Reached the requested new-request limit; partial predictions remain resumable")
            details = {"row_id": row.id, "provider": "jev", "model": MODEL, "route": "OpenRouter",
                       "endpoint": ENDPOINT, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "pricing": PRICE, **self.provenance}
            ident = self.ledger.reserve(RESERVE_NANO, details)
            reservations.append(ident)
            self.quota["count"] += 1
            result = post_json_exact_cost(url, body, headers, timeout)
            if isinstance(result, dict):
                model, provider, request_id = result.get("model"), result.get("provider"), result.get("id")
                valid_model = isinstance(model, str) and model in ALLOWED_RESPONSE_MODELS
                captured["resolved_model"] = model if valid_model else None
                captured["provider"] = provider if provider == "TypeSafe" else None
                captured["request_id"] = request_id if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,200}", request_id) else None
                captured["routing_violation"] = not valid_model or provider != "TypeSafe"
                usage = result.get("usage")
                try:
                    captured["reported_cost_usd"] = parse_cost(usage.get("cost") if isinstance(usage, dict) else None)
                except budget.GuardError:
                    captured["invalid_cost"] = True
                    captured["reported_cost_usd"] = None
            return result

        with _LOCK:
            try:
                with patch.object(providers, "_post_json", guarded_post):
                    prediction = self.inner.predict(row, labels, prompt)
            except BaseException as exc:
                for ident in reservations:
                    self.ledger.result(ident, {"outcome": "raised", "exception_type": type(exc).__name__, "usage_unknown": True})
                raise
        if not reservations:
            return prediction
        cost = captured.get("reported_cost_usd")
        overrun = ((prediction.input_tokens is not None and prediction.input_tokens > PRICE["reserve_input_tokens"])
                   or (cost is not None and Decimal(cost) > Decimal(PRICE["per_request_reserved_usd"])))
        halt = overrun or captured.get("routing_violation", False) or captured.get("invalid_cost", False)
        details = {"outcome": "prediction_error" if prediction.error else "returned",
                   "usage": {"input_tokens": prediction.input_tokens, "output_tokens": prediction.output_tokens},
                   "route": "OpenRouter", "request_id": captured.get("request_id"),
                   "provider": captured.get("provider"), "resolved_model": captured.get("resolved_model"),
                   "reported_cost_usd": cost, "reservation_released": False,
                   "usage_exceeded_reservation_assumptions": bool(halt),
                   "guard_reasons": {"reported_usage_or_cost_overrun": overrun,
                                     "unexpected_route_or_snapshot": captured.get("routing_violation", False),
                                     "invalid_reported_cost": captured.get("invalid_cost", False)}}
        self.ledger.result(reservations[0], details)
        prediction.metadata["openrouter"] = {"id": captured.get("request_id"), "provider": captured.get("provider"),
                                               "reported_cost_usd": cost, "endpoint": ENDPOINT}
        prediction.metadata["budget"] = {"ledger_id": self.ledger.identity["ledger_id"],
            "reservation_id": reservations[0], "reserved_upper_bound_usd": PRICE["per_request_reserved_usd"],
            "reservation_released": False}
        if halt:
            raise budget.GuardError("OpenRouter route, snapshot, reported cost or usage violated the verified assumptions; ledger halted")
        return prediction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/jev_openrouter.json"))
    parser.add_argument("--model-key", default="jev_openrouter")
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/jev"))
    parser.add_argument("--ledger", type=Path, default=Path("results/jev-budget.jsonl"))
    parser.add_argument("--budget-usd", default="2.50")
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--stop-after-new-requests", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--init-ledger", action="store_true")
    parser.add_argument("--prompt-api-key", action="store_true")
    args = parser.parse_args(argv)
    if hashlib.sha256(_BUDGET_PATH.read_bytes()).hexdigest() != FROZEN_BUDGET_SHA256:
        raise budget.GuardError("Frozen ledger driver changed; review before running this wrapper")
    if datetime.now(timezone.utc).date().isoformat() != VERIFIED_ON:
        raise budget.GuardError("Verify current OpenRouter Jev pricing/route before updating the dated wrapper")
    if not 0 < budget.usd_nano(args.budget_usd) <= budget.usd_nano("2.50"):
        raise budget.GuardError("This separately allocated Jev budget must not exceed $2.50")
    if args.max_requests <= 0 or args.bootstrap_samples < 100 or any(value < 0 for value in args.shots) or (args.stop_after_new_requests is not None and args.stop_after_new_requests <= 0):
        raise budget.GuardError("Invalid request cap, bootstrap count (minimum 100), new-request limit or shot count")
    config = validate_config(json.loads(args.config.read_text())[args.model_key])
    from jevbench.data import load_prepared
    from jevbench.prompts import select_examples
    from jevbench.runner import run_model
    from jevbench import providers
    datasets = [load_prepared(path) for path in args.data]
    for dataset in datasets:
        if len(dataset.test) > args.max_requests:
            raise budget.GuardError("Dataset exceeds --max-requests; test rows are never silently removed")
        for shots in args.shots:
            select_examples(dataset.train, dataset.labels, shots, args.seed)
    count = sum(len(dataset.test) for dataset in datasets)*len(args.shots)
    plan = {"mode": "execute" if args.execute else "dry_run", "model": MODEL, "endpoint": ENDPOINT,
            "fresh_matrix_requests": count, "full_reservation_usd": budget.usd_string(count*RESERVE_NANO),
            "per_call_reserved_usd": PRICE["per_request_reserved_usd"], "budget_usd": args.budget_usd,
            "scope": "Native Choice through OpenRouter; full reservations retained; reported cost is separate; no other account spend included"}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return plan
    ledger = budget.Ledger(args.ledger, args.budget_usd, initialize=args.init_ledger)
    if args.prompt_api_key:
        budget.prompt_credentials([config])
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise budget.GuardError("Set OPENROUTER_API_KEY or use --prompt-api-key; no calls made")
    quota = {"count": 0, "limit": args.stop_after_new_requests}
    wrapper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    completed = []
    for dataset in datasets:
        for shots in args.shots:
            provenance = {"wrapper_sha256": wrapper_sha, "ledger_driver_sha256": FROZEN_BUDGET_SHA256,
                          "ledger_id": ledger.identity["ledger_id"], "route": "OpenRouter",
                          "endpoint": ENDPOINT, "price_verified_on": VERIFIED_ON, "price_source": PRICE["source"],
                          "response_model_allowlist": sorted(ALLOWED_RESPONSE_MODELS),
                          "native_protocol": "frozen_JevClassifier_Choice", "reserve_checks_included_in_latency": True,
                          "dataset": dataset.name, "shots_per_class": shots, "seed": args.seed}
            run_config = {**config, "budget_guard": provenance}

            def factory(_):
                return GuardedOpenRouterJev(config, ledger, provenance, quota)

            try:
                with patch.object(providers, "build_provider", factory):
                    result = run_model(dataset, run_config, args.output, shots=shots, seed=args.seed,
                        allow_paid=True, max_requests=args.max_requests, bootstrap_samples=args.bootstrap_samples)
            except StopRequested:
                summary = {"status": "paused_after_requested_new_requests", "new_requests": quota["count"], "budget": ledger.snapshot()}
                print(json.dumps(summary), flush=True)
                return summary
            completed.append(result["run_id"])
            print(json.dumps({"run": result["run_id"], "accuracy": result["metrics"]["accuracy"], "budget": ledger.snapshot()}), flush=True)
    return {"completed": completed, "budget": ledger.snapshot()}


if __name__ == "__main__":
    try:
        main()
    except (budget.BudgetStop, budget.GuardError) as exc:
        print(f"Stopped: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
