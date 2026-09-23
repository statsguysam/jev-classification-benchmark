"""Reconcile all completed benchmark ledgers under the cumulative USD 25 cap.

Read-only with respect to execution evidence. This is conservative accounting,
not provider invoicing, and does not value local or Colab compute.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_control_failed_retries as recovery


def collect():
    audited = recovery.audit_run()
    inventory = recovery.preparation.ledger_inventory(ROOT)
    expected = recovery.prior_ledger_paths()
    if not audited["record"]["no_op"]:
        expected = expected | {(recovery.AREA / "budget.jsonl").relative_to(ROOT).as_posix()}
    recovery.require({v["path"] for v in inventory.values()} == expected,
                     "Final ledger inventory is missing or includes an unaccounted allocation")
    rows, total = [], 0
    for item in sorted(inventory.values(), key=lambda value: value["path"]):
        path = ROOT / item["path"]
        header = recovery.preparation.read_predictions(path)[0]
        state = recovery.budget.Ledger(path, recovery.budget.usd_string(header["budget_nano"])).snapshot()
        recovery.require(not state["halted"], "A final budget ledger is halted")
        counted = (recovery.route.RESERVE_NANO if path == recovery.base.HEALTH else
                   recovery.budget.usd_nano(state["charged_or_reserved_usd"]))
        total += counted
        rows.append({"path": item["path"], "ledger_id": state["ledger_id"],
                     "allocated_usd": state["budget_usd"],
                     "charged_or_reserved_usd": state["charged_or_reserved_usd"],
                     "counted_toward_cumulative_cap_usd": recovery.budget.usd_string(counted),
                     "reservations": state["reservations"], "settlements": state["settlements"],
                     "artifact_sha256": item["pins"]})
    expected_total = (recovery.budget.usd_nano(audited["plan"]["prior"]["prior_conservative_usd"]) +
                      recovery.budget.usd_nano(audited["budget"]["charged_or_reserved_usd"]))
    recovery.require(total == expected_total and total <= recovery.base.TOTAL_NANO,
                     "Final accounting differs from the audited cumulative authorization")
    after = recovery.audit_run()
    recovery.require(after["artifact_sha256"] == audited["artifact_sha256"],
                     "Completion evidence changed during budget reconciliation")
    final_inventory = recovery.preparation.ledger_inventory(ROOT)
    inventory_pins = {item["path"]: item["pins"] for item in inventory.values()}
    final_pins = {item["path"]: item["pins"] for item in final_inventory.values()}
    recovery.require(set(final_pins) == expected and final_pins == inventory_pins,
                     "Final ledger inventory or its evidence changed during reconciliation")
    for row in rows:
        recovery.require(all(recovery.file_sha(ROOT / path) == sha
                             for path, sha in row["artifact_sha256"].items()),
                         "A budget ledger changed during reconciliation")
    return {"schema_version": 1, "status": "complete", "authorized_usd": "25.000000000",
            "conservative_total_usd": recovery.budget.usd_string(total),
            "remaining_under_authorization_usd": recovery.budget.usd_string(recovery.base.TOTAL_NANO - total),
            "ledgers": rows, "control_recovery_no_op": audited["record"]["no_op"],
            "analysis_sha256": recovery.file_sha(__file__),
            "interpretation": "Retained conservative charges and unresolved reservations across all study phases. "
                "Not a provider invoice or account-wide cap. Local/Colab compute is unpriced. "
                "Historical reservations remain retained; phase allocation ceilings must not be summed as actual spending."}


def main():
    report = collect()
    out = ROOT / "results/completion_20260923"
    (out / "COSTS.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Final conservative budget reconciliation", "",
             f"**US${report['conservative_total_usd']} counted against the US$25 authorization; "
             f"US${report['remaining_under_authorization_usd']} remains.**", "",
             report["interpretation"], "",
             "| Ledger | Reservations | Settlements | Counted USD |",
             "|---|---:|---:|---:|"]
    for row in report["ledgers"]:
        lines.append(f"| {row['path']} | {row['reservations']} | {row['settlements']} | "
                     f"{row['counted_toward_cumulative_cap_usd']} |")
    lines += ["", "[Machine-readable accounting and ledger hashes](COSTS.json)", ""]
    (out / "COSTS.md").write_text("\n".join(lines))
    print(json.dumps({key: report[key] for key in ("status", "conservative_total_usd", "remaining_under_authorization_usd")}))


if __name__ == "__main__":
    main()
