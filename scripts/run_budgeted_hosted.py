"""Run hosted benchmarks under one durable, conservatively reserved USD budget.

Dry-run is the default. Every actual HTTP call is reserved before dispatch.
Successful OpenAI responses with complete usage settle once at conservative
reported-token rates; errors, unknown usage and crashes keep full reservations.
The ledger cap is
exact. Its correspondence to an invoice is conditional on published prices,
token-accounting assumptions, standard service, and no unrelated account spend.
Keep BOTH ledger and .lock files together on a local POSIX filesystem. Never
delete them or create a second ledger to resume the same authorized budget.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import getpass
import hashlib
import json
import math
import os
import re
import sys
import threading
import uuid
import warnings
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from unittest.mock import patch

NANO = Decimal(1_000_000_000)
MILLION = Decimal(1_000_000)
VERIFIED_ON = "2026-09-22"
BASE_URLS = {"jev": "https://api.typesafe.ai/v1", "openai": "https://api.openai.com/v1",
             "anthropic": "https://api.anthropic.com/v1", "gemini": "https://generativelanguage.googleapis.com/v1beta"}
KEY_ENVS = {"jev": "TYPESAFE_API_KEY", "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
OFFICIAL_PRICE_HOSTS = {"jev": {"docs.typesafe.ai"}, "openai": {"developers.openai.com", "openai.com"},
                        "anthropic": {"platform.claude.com", "claude.com"}, "gemini": {"ai.google.dev"}}
PRICES = {
    "jev-1.13.0": {"provider": "jev", "input_usd_per_million": "0.042", "output_usd_per_million": "0",
        "input_multiplier": "1", "mode": "full_context", "max_input_tokens": 64000,
        "max_output_tokens": 0, "source": "https://docs.typesafe.ai/models", "verified_on": VERIFIED_ON},
    "gpt-5.6-luna": {"provider": "openai", "input_usd_per_million": "0.2", "output_usd_per_million": "1.2",
        "input_multiplier": "1.25", "mode": "utf8_bound", "max_input_tokens": 272000,
        "max_output_tokens": 128000, "source": "https://developers.openai.com/api/docs/models/gpt-5.6-luna", "verified_on": VERIFIED_ON},
    "gpt-6-astra": {"provider": "openai", "input_usd_per_million": "10", "output_usd_per_million": "50",
        "input_multiplier": "1.25", "mode": "utf8_bound", "max_input_tokens": 272000,
        "max_output_tokens": 128000, "source": "https://developers.openai.com/api/docs/models/gpt-6-astra", "verified_on": VERIFIED_ON},
}
_TRANSPORT_LOCK = threading.RLock()


class BudgetStop(RuntimeError):
    pass


class GuardError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def usd_nano(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise GuardError("Invalid USD amount") from None
    if not number.is_finite() or number < 0:
        raise GuardError("USD amounts must be finite and nonnegative")
    return int((number * NANO).to_integral_value(rounding=ROUND_CEILING))


def usd_string(value):
    return format(Decimal(value) / NANO, ".9f")


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class Ledger:
    """Append-only fsynced events protected by a persistent POSIX flock file."""

    def __init__(self, path, budget_usd, *, initialize=False):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.budget = usd_nano(budget_usd)
        if self.budget <= 0:
            raise GuardError("Budget must be positive")
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if not initialize and (not self.path.exists() or not self.lock_path.exists()):
            raise GuardError("Existing ledger AND .lock required; use --init-ledger once for a new authorized budget")
        with self._locked(create=initialize) as lock:
            marker = lock.read()
            if not marker:
                if not initialize or self.path.exists():
                    raise GuardError("Missing ledger identity marker; refusing to reset history")
                marker = json.dumps({"schema": 1, "ledger_id": str(uuid.uuid4()), "budget_nano": self.budget})
                lock.write(marker)
                lock.flush()
                os.fsync(lock.fileno())
                identity = json.loads(marker)
                with self.path.open("x", encoding="utf-8") as file:
                    header = {"type": "header", **identity, "created_at": utc_now(), "previous_sha256": None}
                    header["event_sha256"] = sha(header)
                    file.write(json.dumps(header, sort_keys=True) + "\n")
                    file.flush()
                    os.fsync(file.fileno())
                marker = json.dumps({**identity, "head_sha256": header["event_sha256"], "event_count": 1})
                lock.seek(0)
                lock.write(marker)
                lock.truncate()
                lock.flush()
                os.fsync(lock.fileno())
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            try:
                loaded_marker = json.loads(marker)
                self.identity = {key: loaded_marker[key] for key in ("schema", "ledger_id", "budget_nano")}
            except (ValueError, TypeError, KeyError):
                raise GuardError("Corrupt ledger identity marker") from None
            if self.identity.get("budget_nano") != self.budget:
                raise GuardError("Existing ledger budget is immutable; use its original --budget-usd")
            self._read()

    @contextlib.contextmanager
    def _locked(self, *, create=False):
        if self.path.is_symlink() or self.lock_path.is_symlink():
            raise GuardError("Ledger paths must not be symlinks")
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        # O_APPEND would make marker rewrites append a second JSON object.
        descriptor = os.open(self.lock_path, flags, 0o600)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                lock.seek(0)
                yield lock
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self):
        try:
            data = self.path.read_text()
            if not data.endswith("\n"):
                raise GuardError("Incomplete ledger write; refusing to discard possible reservations")
            events = [json.loads(line) for line in data.splitlines()]
        except (OSError, ValueError):
            raise GuardError("Missing or corrupt ledger; refusing to reset history") from None
        previous, reserved, amounts, settlements = None, 0, {}, set()
        for index, event in enumerate(events):
            check = dict(event)
            checksum = check.pop("event_sha256", None)
            if check.get("previous_sha256") != previous or sha(check) != checksum:
                raise GuardError("Ledger hash chain mismatch")
            previous = checksum
            if index == 0:
                if event.get("type") != "header" or any(event.get(k) != v for k, v in self.identity.items()):
                    raise GuardError("Ledger header does not match persistent identity")
            elif event.get("type") == "reserve":
                cost, ident = event.get("reserved_nano"), event.get("reservation_id")
                if type(cost) is not int or cost <= 0 or not isinstance(ident, str) or ident in amounts:
                    raise GuardError("Invalid ledger reservation")
                reserved += cost
                amounts[ident] = cost
            elif event.get("type") == "settle":
                ident, charge = event.get("reservation_id"), event.get("charged_nano")
                if ident not in amounts or ident in settlements or type(charge) is not int or not 0 <= charge <= amounts[ident]:
                    raise GuardError("Invalid or duplicate ledger settlement")
                reserved -= amounts[ident]-charge
                settlements.add(ident)
            elif event.get("type") == "result":
                if event.get("reservation_id") not in amounts:
                    raise GuardError("Ledger result has no reservation")
            else:
                raise GuardError("Unknown ledger event")
        try:
            anchor = json.loads(self.lock_path.read_text())
        except (OSError, ValueError):
            raise GuardError("Corrupt durable ledger head marker") from None
        if any(anchor.get(k) != v for k, v in self.identity.items()) or anchor.get("head_sha256") != previous or anchor.get("event_count") != len(events):
            raise GuardError("Ledger/anchor mismatch; possible rollback or interrupted write; fail closed")
        if not events or reserved > self.budget:
            raise GuardError("Ledger is empty or already exceeds its authorized budget")
        return events, reserved

    def _append(self, event, events):
        event = {**event, "recorded_at": utc_now(), "previous_sha256": events[-1]["event_sha256"]}
        event["event_sha256"] = sha(event)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        # Separate durable anchor detects loss of a complete valid event suffix.
        # A crash between ledger and anchor fsync fails closed on resume.
        with self.lock_path.open("r+", encoding="utf-8") as marker:
            marker.write(json.dumps({**self.identity, "head_sha256": event["event_sha256"], "event_count": len(events)+1}))
            marker.truncate()
            marker.flush()
            os.fsync(marker.fileno())

    def reserve(self, cost_nano, details):
        if type(cost_nano) is not int or cost_nano <= 0:
            raise GuardError("Reservation must be a positive integer number of nanodollars")
        with self._locked():
            events, reserved = self._read()
            if any(e.get("details", {}).get("usage_exceeded_reservation_assumptions") for e in events):
                raise GuardError("Ledger is halted after a usage bound violation; reconcile billing first")
            if reserved + cost_nano > self.budget:
                raise BudgetStop(f"Budget exhausted: reserved ${usd_string(reserved)}, next ${usd_string(cost_nano)}, cap ${usd_string(self.budget)}")
            ident = str(uuid.uuid4())
            self._append({"type": "reserve", "reservation_id": ident, "reserved_nano": cost_nano, "details": details}, events)
            return ident

    def settle(self, reservation_id, charged_nano, evidence):
        """Release unused reserve only once, for verified complete usage."""
        with self._locked():
            events, _ = self._read()
            reserved = next((e["reserved_nano"] for e in events
                             if e["type"] == "reserve" and e["reservation_id"] == reservation_id), None)
            if reserved is None or any(e["type"] == "settle" and e["reservation_id"] == reservation_id for e in events):
                raise GuardError("Unknown or already settled reservation")
            if type(charged_nano) is not int or not 0 <= charged_nano <= reserved:
                self._append({"type": "result", "reservation_id": reservation_id,
                    "details": {"usage_exceeded_reservation_assumptions": True}}, events)
                raise GuardError("Reported charge exceeds reservation; ledger halted")
            self._append({"type": "settle", "reservation_id": reservation_id, "charged_nano": charged_nano,
                          "evidence": evidence}, events)

    def result(self, reservation_id, details):
        with self._locked():
            events, _ = self._read()
            if not any(e.get("reservation_id") == reservation_id and e["type"] == "reserve" for e in events):
                raise GuardError("Cannot finalize an unknown reservation")
            self._append({"type": "result", "reservation_id": reservation_id, "details": details}, events)

    def snapshot(self):
        with self._locked():
            events, reserved = self._read()
            return {"ledger_id": self.identity["ledger_id"], "budget_usd": usd_string(self.budget),
                    "charged_or_reserved_usd": usd_string(reserved), "remaining_reservation_usd": usd_string(self.budget-reserved),
                    "reservations": sum(e["type"] == "reserve" for e in events),
                    "settlements": sum(e["type"] == "settle" for e in events), "events": len(events),
                    "halted": any(e.get("details", {}).get("usage_exceeded_reservation_assumptions") for e in events)}


def validate_config(config):
    config = dict(config)
    allowed = {"provider", "model", "base_url", "api_key_env", "timeout", "max_output_tokens",
               "temperature", "seed", "reasoning_effort", "token_limit_parameter", "budget_guard"}
    if set(config)-allowed:
        raise GuardError(f"Unsupported hosted config fields: {sorted(set(config)-allowed)}")
    provider = config.get("provider")
    if provider not in BASE_URLS:
        raise GuardError("Budget driver accepts only canonical official hosted providers")
    if config.get("base_url", BASE_URLS[provider]) != BASE_URLS[provider]:
        raise GuardError("Only the exact official base URL is permitted")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(config.get("model", ""))):
        raise GuardError("An explicit model ID is required")
    env = config.get("api_key_env", KEY_ENVS[provider])
    if not isinstance(env, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", env):
        raise GuardError("api_key_env must name an environment variable; key values are forbidden")
    config["api_key_env"] = env
    if config.get("token_limit_parameter", "max_completion_tokens") != "max_completion_tokens":
        raise GuardError("Use max_completion_tokens to cap both reasoning and visible OpenAI output")
    if provider != "jev":
        maximum = config.get("max_output_tokens")
        if type(maximum) is not int or not 1 <= maximum <= 128000:
            raise GuardError("An explicit max_output_tokens in [1,128000] is required")
    timeout = config.get("timeout", 120)
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise GuardError("A finite positive timeout is required")
    return config


def select_price(config, declared=None):
    from urllib.parse import urlparse
    model = config["model"]
    custom = model in (declared or {})
    price = dict((declared or {}).get(model, PRICES.get(model, {})))
    required = {"provider", "input_usd_per_million", "output_usd_per_million", "input_multiplier",
                "mode", "max_input_tokens", "max_output_tokens", "source", "verified_on"}
    if not required <= set(price) or price["provider"] != config["provider"]:
        raise GuardError("No verified pricing for this exact model; supply explicit --prices-json")
    if price["verified_on"] != datetime.now(timezone.utc).date().isoformat():
        raise GuardError("Pricing must be verified today; refresh a cited --prices-json declaration")
    source = urlparse(price["source"])
    if source.scheme != "https" or source.hostname not in OFFICIAL_PRICE_HOSTS[config["provider"]] or source.username or source.query:
        raise GuardError("Pricing needs a direct official HTTPS source")
    for field in ("input_usd_per_million", "output_usd_per_million", "input_multiplier"):
        usd_nano(price[field])
    if Decimal(str(price["input_multiplier"])) < 1:
        raise GuardError("Input multiplier must include the most expensive cache-write/input tier")
    for field in ("max_input_tokens", "max_output_tokens"):
        if type(price[field]) is not int or price[field] < 0:
            raise GuardError("Pricing token maxima must be nonnegative integers")
    if price["max_input_tokens"] < 1:
        raise GuardError("A documented maximum input context is required")
    if price["mode"] not in {"utf8_bound", "full_context"}:
        raise GuardError("Unknown reservation calculation")
    if price["mode"] == "utf8_bound" and (config["provider"] != "openai" or model not in {"gpt-5.6-luna", "gpt-6-astra"}):
        raise GuardError("Unknown models require full-context reservations, not assumed tokenization")
    if config["provider"] != "jev" and config["max_output_tokens"] > price["max_output_tokens"]:
        raise GuardError("Requested output cap exceeds declared model maximum")
    if custom and price.get("declaration") != "I verified the maximum standard request rates and context limits at the cited official source today":
        raise GuardError("Custom pricing requires an explicit verification declaration")
    price["verification_kind"] = "operator_declared" if custom else "bundled_official_reference"
    return price


def reservation(config, price, prompt):
    if price["mode"] == "utf8_bound":
        # Byte BPE content tokens cannot exceed UTF-8 bytes. Chat serialization
        # adds tokens: 2048 is an intentionally generous protocol assumption,
        # NOT a provider-certified exact overhead or invoice guarantee.
        inputs = len(prompt.encode("utf-8")) + 2048
        if inputs > min(272000, price["max_input_tokens"]):
            raise GuardError("Input reservation exceeds 272k tier; refusing long-context price ambiguity")
        bound_kind = "utf8_bytes_plus_2048_chat_overhead_assumption"
    else:
        inputs, bound_kind = price["max_input_tokens"], "declared_full_context_maximum"
    outputs = 0 if config["provider"] == "jev" else config["max_output_tokens"]
    amount = (Decimal(inputs)*Decimal(str(price["input_usd_per_million"]))*Decimal(str(price["input_multiplier"]))
              + Decimal(outputs)*Decimal(str(price["output_usd_per_million"]))) / MILLION
    return usd_nano(amount), {"input_token_bound": inputs, "output_token_cap": outputs, "bound_kind": bound_kind}


def _usage_estimate(prediction, config, price):
    inputs, outputs = prediction.input_tokens, prediction.output_tokens
    if inputs is None or (outputs is None and config["provider"] != "jev"):
        return None
    amount = (Decimal(inputs)*Decimal(str(price["input_usd_per_million"]))
              + Decimal(outputs or 0)*Decimal(str(price["output_usd_per_million"]))) / MILLION
    return {"usd": usd_string(usd_nano(amount)), "basis": "reported_tokens_at_uncached_standard_rates; excludes unknown cache-write/other billing details; not invoice"}


class BudgetedProvider:
    def __init__(self, provider, config, price, ledger, provenance):
        self.provider, self.config, self.price, self.ledger = provider, config, price, ledger
        self.provenance = provenance

    def predict(self, row, labels, prompt):
        from jevbench import providers
        amount, bound = reservation(self.config, self.price, prompt)
        reservations = []
        provider_name = self.config["provider"]

        def guarded_post(url, body, headers, timeout):
            expected = BASE_URLS[provider_name] + {"jev": "/systemone", "openai": "/chat/completions", "anthropic": "/messages",
                        "gemini": f"/models/{self.config['model']}:generateContent"}[provider_name]
            if reservations or url != expected or body.get("model", self.config["model"]) != self.config["model"]:
                raise GuardError("Unexpected endpoint/model or repeated request; no network call made")
            if provider_name == "openai":
                allowed = {"model", "messages", "max_completion_tokens", "temperature", "seed", "reasoning_effort"}
                if set(body)-allowed or body.get("messages") != [{"role": "user", "content": prompt}] or body.get("max_completion_tokens") != self.config["max_output_tokens"]:
                    raise GuardError("OpenAI request differs from the priced text-only protocol")
                body = {**body, "service_tier": "default"}
            elif provider_name == "jev":
                if set(body) != {"state", "model", "questions"} or body["state"] != prompt or len(body["questions"]) != 1:
                    raise GuardError("Unexpected Jev request shape")
                question = body["questions"].get("classification", {})
                if set(question) != {"type", "instructions", "criteria"} or question.get("type") != "choice" or question.get("criteria") != {str(i): label for i, label in enumerate(labels)}:
                    raise GuardError("Unexpected Jev classification question")
            elif provider_name == "anthropic":
                if set(body)-{"model", "messages", "max_tokens", "temperature"} or body.get("max_tokens") != self.config["max_output_tokens"] or body.get("messages") != [{"role": "user", "content": prompt}]:
                    raise GuardError("Unexpected Anthropic request shape")
            elif provider_name == "gemini":
                if set(body) != {"contents", "generationConfig"} or body["generationConfig"].get("maxOutputTokens") != self.config["max_output_tokens"] or set(body["generationConfig"])-{"maxOutputTokens", "temperature"} or body["contents"] != [{"role": "user", "parts": [{"text": prompt}]}]:
                    raise GuardError("Unexpected Gemini request shape")
            details = {"row_id": row.id, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "provider": provider_name, "model": self.config["model"], "bound": bound,
                       "pricing": self.price, **self.provenance}
            ident = self.ledger.reserve(amount, details)
            reservations.append(ident)
            return original_post(url, body, headers, timeout)

        with _TRANSPORT_LOCK:
            original_post = providers._post_json
            try:
                with patch.object(providers, "_post_json", guarded_post):
                    prediction = self.provider.predict(row, labels, prompt)
            except BaseException as exc:
                for ident in reservations:
                    self.ledger.result(ident, {"outcome": "raised", "exception_type": type(exc).__name__, "usage_unknown": True})
                raise
        if reservations:
            usage = {"input_tokens": prediction.input_tokens, "output_tokens": prediction.output_tokens}
            estimate = _usage_estimate(prediction, self.config, self.price)
            violation = ((prediction.input_tokens is not None and prediction.input_tokens > bound["input_token_bound"])
                         or (provider_name != "jev" and prediction.output_tokens is not None and prediction.output_tokens > bound["output_token_cap"]))
            details = {"outcome": "prediction_error" if prediction.error else "returned", "usage": usage,
                       "reported_usage_estimate": estimate, "resolved_model": prediction.metadata.get("resolved_model"),
                       "usage_exceeded_reservation_assumptions": violation}
            self.ledger.result(reservations[0], details)
            prediction.metadata["budget"] = {"ledger_id": self.ledger.identity["ledger_id"], "reservation_id": reservations[0],
                "reserved_upper_bound_usd": usd_string(amount), "reported_usage_estimate": estimate, **bound}
            if violation:
                raise GuardError("Provider usage exceeded reservation assumptions; stop and reconcile billing before resuming")
            resolved = prediction.metadata.get("resolved_model")
            trustworthy_model = isinstance(resolved, str) and (resolved == self.config["model"] or re.fullmatch(re.escape(self.config["model"])+r"-\d{4}-\d{2}-\d{2}", resolved))
            if provider_name == "openai" and not prediction.error and trustworthy_model and all(type(x) is int and x > 0 for x in (prediction.input_tokens, prediction.output_tokens)):
                charge = (Decimal(prediction.input_tokens)*Decimal(str(self.price["input_usd_per_million"]))*Decimal(str(self.price["input_multiplier"]))
                          + Decimal(prediction.output_tokens)*Decimal(str(self.price["output_usd_per_million"]))) / MILLION
                charged_nano = usd_nano(charge)
                self.ledger.settle(reservations[0], charged_nano, {"usage": usage, "pricing_sha256": sha(self.price),
                    "model": resolved, "basis": "complete successful OpenAI usage; maximum input/cache-write rate; no cache discount"})
                prediction.metadata["budget"]["settled_conservative_usd"] = usd_string(charged_nano)
        return prediction


def prompt_credentials(configs):
    """Use a controlling terminal only, never a pipe or notebook cell output."""
    environments = sorted({config["api_key_env"] for config in configs})
    # TextIOWrapper's update mode can require seeking on macOS character
    # devices. getpass opens its own terminal for input; this stream only writes
    # the prompt, so output-only mode is portable and never seeks.
    with open("/dev/tty", "w") as tty:
        if not tty.isatty():
            raise GuardError("Credential entry requires a controlling TTY")
        for env in environments:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                value = getpass.getpass(f"Enter {env} (hidden; memory only): ", stream=tty)
            if not value:
                raise GuardError(f"Empty credential for {env}")
            os.environ[env] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--model-keys", nargs="+", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/hosted"))
    parser.add_argument("--budget-usd", default="10")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--prices-json", type=Path)
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--init-ledger", action="store_true", help="Initialize ONCE; never reset or replace a ledger to resume")
    parser.add_argument("--prompt-api-key", action="store_true")
    args = parser.parse_args(argv)
    if args.max_requests <= 0 or any(k < 0 for k in args.shots):
        raise GuardError("Invalid request cap or negative shot count")
    raw = json.loads(args.config.read_text())
    configs = [validate_config(raw[key]) for key in args.model_keys]
    declared = json.loads(args.prices_json.read_text()) if args.prices_json else None
    prices = [select_price(config, declared) for config in configs]
    from jevbench.data import load_prepared
    from jevbench.prompts import build_prompt, select_examples
    datasets = [load_prepared(path) for path in args.data]
    if any(len(dataset.test) > args.max_requests for dataset in datasets):
        raise GuardError("Prepared test set exceeds --max-requests; no dataset is silently shortened")
    jobs, total = [], 0
    for dataset in datasets:
        for config, price in zip(configs, prices):
            for shots in args.shots:
                examples = select_examples(dataset.train, dataset.labels, shots, args.seed)
                costs = [reservation(config, price, build_prompt(row, dataset.labels, examples))[0] for row in dataset.test]
                total += sum(costs)
                jobs.append((dataset, config, price, shots, sum(costs)))
    plan = {"mode": "execute" if args.execute else "dry_run", "fresh_full_matrix_reserved_upper_bound_usd": usd_string(total),
            "budget_usd": usd_string(usd_nano(args.budget_usd)), "calls_if_no_cached_predictions": sum(len(job[0].test) for job in jobs),
            "scope": "exact ledger reservation cap; conditional monetary bound, not provider invoice or account-wide spend cap",
            "assumptions": "current cited standard rates; text-only requests; UTF8 byte-BPE content bound plus 2048 assumed chat-overhead tokens for known OpenAI models; full context for other models; complete successful OpenAI usage can settle once at maximum configured rates; errors/crashes keep reserves; no retries"}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return plan
    ledger = Ledger(args.ledger, args.budget_usd, initialize=args.init_ledger)
    if args.prompt_api_key:
        prompt_credentials(configs)
    if any(not os.environ.get(config["api_key_env"]) for config in configs):
        raise GuardError("Missing selected provider credential; no calls made")
    from jevbench import providers
    from jevbench.runner import run_model
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_factory = providers.build_provider
    completed = []
    for dataset, config, price, shots, cost in jobs:
        provenance = {"driver_sha256": script_sha, "ledger_id": ledger.identity["ledger_id"],
                      "pricing_sha256": sha(price), "verified_on": price["verified_on"],
                      "service_tier": "default" if config["provider"] == "openai" else "provider_standard",
                      "price_source": price["source"], "dataset": dataset.name, "shots_per_class": shots, "seed": args.seed}
        run_config = {**config, "budget_guard": provenance}

        def factory(inner_config):
            return BudgetedProvider(original_factory(inner_config), config, price, ledger, provenance)

        try:
            with patch.object(providers, "build_provider", factory):
                result = run_model(dataset, run_config, args.output, shots=shots, seed=args.seed,
                    allow_paid=True, max_requests=args.max_requests, bootstrap_samples=args.bootstrap_samples)
            completed.append({"run": result["run_id"], "accuracy": result["metrics"]["accuracy"]})
            print(json.dumps({"completed": completed[-1], "budget": ledger.snapshot()}), flush=True)
        except (BudgetStop, GuardError):
            # runner preserves partial predictions; do not relabel them complete.
            print(json.dumps({"status": "stopped_before_next_request_or_guard_violation", "budget": ledger.snapshot()}), flush=True)
            raise
    return {"runs": completed, "budget": ledger.snapshot()}


if __name__ == "__main__":
    try:
        main()
    except (BudgetStop, GuardError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        raise SystemExit(2)
