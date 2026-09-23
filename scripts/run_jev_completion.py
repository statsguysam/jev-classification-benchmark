"""Resume frozen Jev studies with independently refreshed public transport evidence.

Usage: run_jev_completion.py MODE [the original runner's arguments]. MODE is
numeric-recovery, numeric, text or controls. Without --execute this is read-only,
offline and credential-free; --verify-public explicitly permits public GETs.
No failed prediction is retried or replaced by this compatibility wrapper.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import probe_jev_health as probe

historical, budget, route = probe.historical, probe.budget, probe.route
base = historical.base
AREA = ROOT / "results/completion_sessions"
MAX_VERIFICATION_AGE_SECONDS = 3600
PROBE_SHA = "f59e921cb36d9b1a59a5c9a00826347fc6e409d7abaed4228945546d81e1bb09"
MODES = {"numeric-recovery": "recover_expanded_numeric_billing",
         "numeric": "run_expanded_numeric_review", "text": "run_text_jev_review",
         "controls": "run_review_controls"}
FROZEN_TARGET_SHA = {
    "numeric-recovery": "045abae3f02797e921d5eef4758e95e246922e39a03a69bddb65a0f1882fc416",
    "numeric": "68a16db205cceb81b46916630c776827ccdb09e7df6aed60dd1db7413d4d7a88",
    "text": "bdc3d1cac8e52791359023b57abac8833d898af6ff7b0240512e337baa3ba43e",
}
require = historical.require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc)


def verify_sources(target, selected_sha, wrapper_sha):
    # These are all original hash/core checks, without changing their source.
    historical.verify_frozen_helpers()
    require(sha(probe.__file__) == PROBE_SHA, "Public verification helper changed")
    require(sha(target.__file__) == selected_sha, "Selected producer changed during execution")
    require(sha(__file__) == wrapper_sha, "Completion wrapper changed during execution")


def validate_verification(receipt, now):
    require(now.date().isoformat() == probe.DATE, "Current UTC date needs a new reviewed compatibility wrapper")
    require(receipt.get("verified_on") == probe.DATE and receipt.get("endpoint") == route.ENDPOINT,
            "Independent route verification differs")
    try:
        checked = datetime.fromisoformat(receipt["verified_at"])
        require(checked.tzinfo is not None, "Verification timestamp needs a timezone")
        age = (now - checked).total_seconds()
    except (KeyError, ValueError, TypeError):
        raise budget.GuardError("Invalid independent verification timestamp") from None
    require(0 <= age <= MAX_VERIFICATION_AGE_SECONDS and checked.date() == now.date(),
            "Independent public verification is stale")
    require(receipt.get("input_usd_per_million") == route.PRICE["input_usd_per_million"] and
            receipt.get("output_usd_per_million") == route.PRICE["output_usd_per_million"] and
            receipt.get("reserved_input_tokens") == route.PRICE["reserve_input_tokens"] and
            receipt.get("retained_reservation_usd") == route.PRICE["per_request_reserved_usd"] and
            receipt.get("historical_route_price_declaration") == route.PRICE,
            "Fresh pricing differs from the unchanged reservation policy")
    endpoint = receipt.get("model_catalog", {})
    probe.validate_catalog({"data": {"id": route.MODEL, "architecture": {"modality": "text->decisions"},
                                    "endpoints": [endpoint]}})
    require(route.ALLOWED_RESPONSE_MODELS == {"typesafe/jev-1.13", "typesafe/jev-1.13-20260917"},
            "Response snapshot allowlist changed")
    sources = receipt.get("sources", {})
    require(set(sources) == {probe.CATALOG_URL, probe.ROUTE_DOCS, probe.MODEL_DOCS} and
            all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in sources.values()),
            "Public verification source evidence differs")
    return receipt


class VerifiedTransport:
    def __init__(self, target, selected_sha, *, execute=False, receipt=None):
        self.target, self.selected_sha, self.execute = target, selected_sha, execute
        self.wrapper_sha = sha(__file__)
        self.receipts = [] if receipt is None else [receipt]
        self.area = None

    def refresh(self):
        # GETs carry no model credential. The independent helper rejects redirects,
        # price/provider/context changes and unknown model snapshots.
        value = validate_verification(probe.verify_public(), utc_now())
        self.receipts.append(value)
        if self.area is not None:
            probe.write_exclusive(self.area / f"public-verification-{len(self.receipts):03d}.json", value)
        return value

    def check(self):
        verify_sources(self.target, self.selected_sha, self.wrapper_sha)
        if not self.execute:
            return
        now = utc_now()
        require(now.date().isoformat() == probe.DATE, "UTC date changed; stop and reverify before new calls")
        refresh = not self.receipts
        if self.receipts:
            try:
                checked = datetime.fromisoformat(self.receipts[-1]["verified_at"])
                age = (now - checked).total_seconds()
            except (ValueError, TypeError, KeyError):
                raise budget.GuardError("Invalid cached public verification") from None
            require(age >= 0, "Public verification timestamp is in the future")
            refresh = age > MAX_VERIFICATION_AGE_SECONDS or checked.date() != now.date()
        if refresh:
            self.refresh()
        validate_verification(self.receipts[-1], utc_now())


def evidence_paths(root=ROOT):
    """Source/checkpoint evidence only; exclude regenerable aggregate reports."""
    paths = set()
    for directory in (root / "results", root / "data/pilot", root / "data/tabular-full"):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or "completion_sessions" in path.parts:
                continue
            if (path.name in {"run.json", "predictions.jsonl", "test_manifest.json", "manifest.json",
                    "protocol.json", "requests.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl"}
                    or ("budget" in path.name and path.name.endswith((".jsonl", ".jsonl.lock")))
                    or ("sources" in path.parts and path.suffix == ".json")):
                paths.add(path)
    paths.update((root / "src/jevbench").glob("*.py"))
    paths.update((root / "configs").glob("*.json"))
    paths.update(root / p for p in ("scripts/run_openrouter_jev.py", "scripts/run_budgeted_hosted.py",
        "scripts/run_numeric_jev_review.py", "scripts/run_expanded_numeric_review.py",
        "scripts/run_text_jev_review.py", "scripts/recover_expanded_numeric_billing.py"))
    return sorted(paths)


def snapshot(root=ROOT):
    values = {}
    for path in evidence_paths(root):
        require(path.is_file() and not path.is_symlink(), "Evidence path is missing or linked")
        values[path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return values


def mutable_kind(relative, mode):
    if mode in {"numeric", "numeric-recovery"}:
        area = "results/numeric_expansion"
    elif mode == "text":
        area = "results/text_extension"
    else:
        area = "results/review_controls/execution"
    if mode == "controls":
        if relative == area + "/run.json": return "run"
        if relative == area + "/predictions.jsonl": return "append"
        ledger = area + "/budget.jsonl"
    else:
        if re.fullmatch(re.escape(area) + r"/review/[^/]+/run\.json", relative): return "run"
        if re.fullmatch(re.escape(area) + r"/review/[^/]+/predictions\.jsonl", relative): return "append"
        ledger = area + "/review-budget.jsonl"
    if relative == ledger: return "append"
    if relative == ledger + ".lock": return "ledger_anchor"
    return None


def audit_preservation(before, after, mode, root=ROOT):
    changed = []
    for relative, original in before.items():
        require(relative in after, f"Original evidence disappeared: {relative}")
        if after[relative] == original:
            continue
        kind = mutable_kind(relative, mode)
        require(kind is not None, f"Protected evidence changed: {relative}")
        if kind == "append":
            with (root / relative).open("rb") as stream:
                prefix = stream.read(original["bytes"])
            require(len(prefix) == original["bytes"] and hashlib.sha256(prefix).hexdigest() == original["sha256"],
                    f"Historical evidence prefix changed: {relative}")
        changed.append(relative)
    return changed


def validate_arguments(mode, arguments):
    require(mode in MODES, "Unknown completion mode")
    require(not any("sk-" in value or "API_KEY=" in value for value in arguments),
            "Credentials must not be supplied as command arguments")
    execute = "--execute" in arguments
    require(execute or not any(value in arguments for value in
            ("--freeze-sources", "--init-ledger", "--prompt-api-key")),
            "Write/credential flags require explicit --execute")
    # The frozen parsers validate the remaining exact arguments. This prevents
    # argparse abbreviation from turning an apparently dry run into execution.
    known_flags = {"--execute", "--dry-run", "--datasets", "--dataset", "--model-keys", "--model-key",
                   "--shots", "--freeze-sources", "--init-ledger", "--prompt-api-key",
                   "--bootstrap-samples", "--stop-after-new-requests", "--help"}
    require(all(not arg.startswith("--") or arg in known_flags for arg in arguments),
            "Only exact original option names are allowed; no abbreviations or inline values")
    return execute


def execute_session(mode, arguments, target, transport):
    AREA.mkdir(parents=True, exist_ok=True)
    require(not AREA.is_symlink(), "Completion receipt directory must be canonical")
    lock_path = AREA / "execution.lock"
    require(not lock_path.is_symlink(), "Completion lock must be canonical")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise budget.GuardError("Another completion wrapper is active") from None
        transport.check()
        session = AREA / (utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + mode + "-" + uuid.uuid4().hex[:12])
        session.mkdir(exist_ok=False)
        transport.area = session
        for index, value in enumerate(transport.receipts, 1):
            probe.write_exclusive(session / f"public-verification-{index:03d}.json", value)
        before = snapshot()
        receipt = {"schema_version": 1, "mode": mode, "delegated_argv": arguments,
            "started_at": budget.utc_now(), "wrapper_sha256": transport.wrapper_sha,
            "selected_producer": Path(target.__file__).name, "selected_producer_sha256": transport.selected_sha,
            "replacement": "Scoped base.verify_transport: original hash checks plus independent current public evidence",
            "maximum_public_verification_age_seconds": MAX_VERIFICATION_AGE_SECONDS,
            "original_price_declaration_unchanged": route.PRICE,
            "original_failed_predictions_retained": True, "automatic_retries_added": 0,
            "before_evidence": before}
        probe.write_exclusive(session / "start.json", receipt)
        status, exception_type = "returned", None
        try:
            return target.main(arguments)
        except BaseException as error:
            status, exception_type = "raised", type(error).__name__
            raise
        finally:
            after = snapshot()
            try:
                changed = audit_preservation(before, after, mode)
                # No new requests remain. Verify unchanged code without making
                # an unnecessary public GET or failing only because UTC rolled.
                verify_sources(target, transport.selected_sha, transport.wrapper_sha)
                preserved = True
            except BaseException:
                changed, preserved = [], False
                raise
            finally:
                probe.write_exclusive(session / "completion.json", {"schema_version": 1, "status": status,
                    "exception_type": exception_type, "completed_at": budget.utc_now(),
                    "start_sha256": sha(session / "start.json"), "after_evidence": after,
                    "historical_evidence_preserved": preserved, "allowed_mutable_paths_changed": changed,
                    "public_verification_sha256": {p.name: sha(p) for p in sorted(session.glob("public-verification-*.json"))}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=list(MODES))
    parser.add_argument("--verify-public", action="store_true", help="Allow public GET verification during an otherwise offline plan")
    args, arguments = parser.parse_known_args(argv)
    execute = validate_arguments(args.mode, arguments)
    target = importlib.import_module(MODES[args.mode])
    selected_sha = sha(target.__file__)
    if args.mode in FROZEN_TARGET_SHA:
        require(selected_sha == FROZEN_TARGET_SHA[args.mode], "Frozen selected producer changed")
    transport = VerifiedTransport(target, selected_sha, execute=execute)
    verify_sources(target, selected_sha, transport.wrapper_sha)
    if execute or args.verify_public:
        transport.refresh()
    if args.verify_public and not execute:
        print(json.dumps({"public_verification": transport.receipts[-1]}, sort_keys=True))
    # The transport object verifies all original immutable code and independently
    # refreshed evidence. No dates, prices, request bodies or identities change.
    with patch.object(base, "verify_transport", transport.check):
        if not execute:
            return target.main(arguments)
        return execute_session(args.mode, arguments, target, transport)


if __name__ == "__main__":
    try:
        main()
    except (budget.GuardError, budget.BudgetStop) as exc:
        raise SystemExit(f"STOP: {exc}") from None
