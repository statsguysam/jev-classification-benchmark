"""One isolated Jev request, dry-run by default; never resumes an old study.

The original route implementation remains frozen, including its historical price
date. This new probe independently verifies today's identical route and rates.
It retains the whole $0.002688 reservation, including after a failed request.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_text_jev_review as historical
from jevbench.providers import _NoRedirect
from jevbench.types import Row

route, budget = historical.route, historical.budget
DATE = "2026-09-23"
RELATIVE_AREA = "results/health_checks/jev-20260923"
CAP = "0.002688000"
ENVELOPE = Decimal("24.957827700")
CATALOG_URL = "https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints"
ROUTE_DOCS = "https://openrouter.ai/docs/guides/community/typesafe-sdk.md"
MODEL_DOCS = "https://docs.typesafe.ai/models"
ROW = Row("jev-health-20260923", "An apple is a fruit.", 0)
LABELS = ["fruit", "vehicle"]
PROMPT = ("Classify the final text.\nClasses:\n0: fruit\n1: vehicle\n"
          "Final text: An apple is a fruit.\nClass ID:")
CONFIG = {"provider": "jev", "model": "typesafe/jev-1.13",
          "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "timeout": 30}


def require(condition, message):
    if not condition:
        raise budget.GuardError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh(timestamp):
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()
    except (TypeError, ValueError):
        raise budget.GuardError("Invalid verification timestamp") from None
    require(0 <= age <= 60, "Fresh verification required before the one request")


def ensure_unused(root):
    require(not (root / "results/review_controls/execution").exists(),
            "Controls execution exists; reconcile its allocation before a probe")
    parent, area = root / "results/health_checks", root / RELATIVE_AREA
    if parent.exists():
        require(not parent.is_symlink(), "Health-check directory must be canonical")
        require(all(path == area for path in parent.iterdir()),
                "Another health-check allocation exists; reconcile it first")
    if area.exists():
        require(not area.is_symlink() and area.is_dir(), "Probe directory must be canonical")
        require(all(path.name == "execution.lock" for path in area.iterdir()),
                "This canonical probe already has evidence; no retry or reset")


def protected_files(root):
    paths = [root / path for path in historical.numeric.PRIOR]
    paths += [root / "results/numeric_expansion/review-budget.jsonl"]
    if (root / "results/text_extension/review-budget.jsonl").exists():
        paths.append(root / "results/text_extension/review-budget.jsonl")
    paths = [item for path in paths for item in (path, path.with_name(path.name + ".lock"))]
    paths += [Path(route.__file__), Path(route._BUDGET_PATH), Path(historical.__file__),
              Path(historical.numeric.__file__), Path(historical.base.__file__)]
    require(all(path.is_file() and not path.is_symlink() for path in paths), "Protected evidence is missing or linked")
    return {str(path): sha(path) for path in paths}


def verify_envelope(root):
    require(datetime.now(timezone.utc).date().isoformat() == DATE,
            "This separately verified probe is restricted to 2026-09-23 UTC")
    historical.verify_frozen_helpers()  # Hash checks only; never changes frozen dates.
    historical.verify_envelope(root)
    require(historical.FIXED_PRIOR + historical.NUMERIC_RESERVED_ALLOWANCE + Decimal(historical.STAGE_CAP) == ENVELOPE,
            "Historical outstanding allocations changed")
    require(ENVELOPE + Decimal(CAP) <= Decimal("25.00") and CAP == route.PRICE["per_request_reserved_usd"],
            "Probe plus full unfinished allocations exceed $25")
    text_ledger = root / "results/text_extension/review-budget.jsonl"
    require(text_ledger.exists() == text_ledger.with_name(text_ledger.name + ".lock").exists(),
            "Text ledger or durable identity is missing")
    if text_ledger.exists():
        snapshot = budget.Ledger(text_ledger, historical.STAGE_CAP).snapshot()
        require(not snapshot["halted"], "Text ledger is halted on a usage or routing violation")
    return protected_files(root)


def public_get(url):
    request = urllib.request.Request(url, headers={"Accept": "application/json, text/markdown, text/html",
        "User-Agent": "jev-classification-benchmark/1.0"}, method="GET")
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=20) as response:
            require(response.status == 200 and response.geturl() == url, "Public verification route/status differs")
            raw = response.read(4_000_001)
            require(len(raw) <= 4_000_000, "Public verification response is too large")
            return raw
    except urllib.error.HTTPError as error:
        raise budget.GuardError(f"Public verification GET returned HTTP {error.code} for {url}; no prediction attempted") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise budget.GuardError("Public verification GET failed; no prediction attempted") from None


def validate_catalog(payload):
    data = payload.get("data", {})
    require(data.get("id") == route.MODEL and data.get("architecture", {}).get("modality") == "text->decisions",
            "Jev catalog model/modality changed")
    endpoints = data.get("endpoints", [])
    require(len(endpoints) == 1, "Provider inventory changed; review before probing")
    endpoint = endpoints[0]
    require(endpoint.get("name") == "TypeSafe | typesafe/jev-1.13-20260917" and
            endpoint.get("model_id") == route.MODEL and endpoint.get("provider_name") == "TypeSafe" and
            endpoint.get("context_length") == 32000 and endpoint.get("status") == 0,
            "Jev provider, snapshot, context or availability changed")
    pricing = endpoint.get("pricing", {})
    require(pricing.get("prompt") == "0.000000042" and pricing.get("completion") == "0" and
            pricing.get("discount", 0) == 0 and set(pricing) <= {"prompt", "completion", "discount"},
            "Jev published pricing changed")
    return {key: endpoint[key] for key in ("name", "model_id", "provider_name", "context_length", "status", "pricing")}


def verify_public():
    raw = public_get(CATALOG_URL)
    try:
        endpoint = validate_catalog(json.loads(raw))
    except (ValueError, TypeError, AttributeError, KeyError):
        raise budget.GuardError("Public model catalog is invalid or changed") from None
    route_raw, model_raw = public_get(ROUTE_DOCS), public_get(MODEL_DOCS)
    require(b"/api/v1/systemone" in route_raw and b"typesafe/jev-1.13" in route_raw,
            "Official docs no longer verify the frozen native route/model")
    require(b"64k" in model_raw and b"32k" in model_raw and b"jev-1.13.0" in model_raw,
            "Official TypeSafe context/model documentation changed")
    return {"verified_at": budget.utc_now(), "verified_on": DATE, "endpoint": route.ENDPOINT,
            "model_catalog": endpoint, "input_usd_per_million": "0.042", "output_usd_per_million": "0",
            "reserved_input_tokens": 64000, "retained_reservation_usd": CAP,
            "sources": {url: hashlib.sha256(blob).hexdigest() for url, blob in
                        ((CATALOG_URL, raw), (ROUTE_DOCS, route_raw), (MODEL_DOCS, model_raw))},
            "historical_route_price_declaration": route.PRICE,
            "note": "Fresh independent verification; the imported frozen transport retains its historical Sep 22 declaration."}


def write_exclusive(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def hidden_key():
    require(sys.stdin.isatty(), "A terminal is required for hidden credential entry")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            key = getpass.getpass("OpenRouter API key (hidden; never saved): ")
        except getpass.GetPassWarning:
            raise budget.GuardError("Hidden credential input is unavailable; no plaintext fallback") from None
    require(bool(key.strip()), "No credential entered")
    return key.strip()


def perform_one(area, verification, receipt, pins):
    """Credential is in process environment only; this function never reads files for it."""
    require(not any((area / name).exists() for name in ("budget.jsonl", "budget.jsonl.lock", "run.json", "prediction.json")),
            "A probe attempt already exists; never retry or reset")
    fresh(verification["verified_at"])
    fresh(receipt["checked_at"])
    inner = budget.Ledger(area / "budget.jsonl", CAP, initialize=True)
    provenance = {"purpose": "isolated_health_probe_not_benchmark", "probe_sha256": sha(__file__),
                  "fresh_verification_sha256": sha(area / "verification.json"), "verified_on": DATE,
                  "historical_envelope_usd": str(ENVELOPE), "authorized_total_usd": "25.00"}
    provider = route.GuardedOpenRouterJev(CONFIG, inner, provenance, quota={"count": 0, "limit": 1})
    status, prediction, raised = "failed", None, None
    try:
        require(all(sha(path) == value for path, value in pins.items()), "Historical evidence changed before the request")
        prediction = provider.predict(ROW, LABELS, PROMPT)
        require(inner.snapshot()["reservations"] == 1, "Probe did not make exactly one accounted request")
        if prediction.error is None:
            require(prediction.label in (0, 1) and len(prediction.probabilities or []) == 2,
                    "Invalid health response")
            status = "responded_expected_choice" if prediction.label == ROW.label else "responded_unexpected_choice"
        write_exclusive(area / "prediction.json", asdict(prediction))
    except BaseException as error:
        raised = type(error).__name__
        status = "stopped_or_raised_no_retry"
        raise
    finally:
        snapshot = inner.snapshot()
        unchanged = all(sha(path) == value for path, value in pins.items())
        write_exclusive(area / "run.json", {"status": status, "completed_at": budget.utc_now(),
            "exception_type": raised, "budget": snapshot, "historical_files_unchanged": unchanged,
            "request": {"row_id": ROW.id, "prompt": PROMPT, "labels": LABELS, "expected_choice": ROW.label},
            "credit_receipt": receipt, "provenance": provenance,
            "prediction_error": prediction.error if prediction else None,
            "no_automatic_retry": True, "no_benchmark_metrics": True})
        require(unchanged, "Historical evidence changed during the isolated probe")
    return {"status": status, "request_count": snapshot["reservations"], "budget": snapshot,
            "result_path": str(area / "run.json")}


def execute(root):
    key = hidden_key()
    # Positive credit before any durable probe allocation. Public GETs use no key.
    receipt = historical.public_receipt(historical.check_credit(key))
    verification = verify_public()
    area = root / RELATIVE_AREA
    with ExitStack() as stack:
        for path in (root / "results/numeric_expansion/review-execution.lock",
                     root / "results/text_extension/review-execution.lock"):
            require(path.parent.is_dir() and not path.is_symlink(), "Noncanonical historical execution lock")
            lock = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise budget.GuardError("An earlier worker is active; no probe request") from None
        ensure_unused(root)
        pins = verify_envelope(root)
        area.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context((area / "execution.lock").open("a+"))
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise budget.GuardError("Another probe is active") from None
        ensure_unused(root)
        fresh(receipt["checked_at"]); fresh(verification["verified_at"])
        verification["protected_files_sha256"] = pins
        write_exclusive(area / "verification.json", verification)
        previous = os.environ.get("OPENROUTER_API_KEY")
        try:
            os.environ["OPENROUTER_API_KEY"] = key
            return perform_one(area, verification, receipt, pins)
        finally:
            key = None
            if previous is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    ensure_unused(ROOT)
    verify_envelope(ROOT)
    if args.execute:
        result = execute(ROOT)
    else:
        result = {"status": "dry_run_no_network_credentials_or_writes", "maximum_new_requests": 1,
                  "retained_reservation_usd": CAP, "historical_full_envelope_usd": str(ENVELOPE),
                  "combined_envelope_usd": str(ENVELOPE + Decimal(CAP)), "authorized_total_usd": "25.00",
                  "model": route.MODEL, "endpoint": route.ENDPOINT, "area": str(ROOT / RELATIVE_AREA)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    try:
        main()
    except (budget.GuardError, budget.BudgetStop) as error:
        raise SystemExit(f"STOP: {error}") from None
