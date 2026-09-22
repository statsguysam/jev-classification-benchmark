"""A billing recovery cannot erase failures, replay paid rows, or weaken caps."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recover_expanded_numeric_billing as recovery
from jevbench.metrics import evaluate
from jevbench.runner import digest
from jevbench.types import Prediction, PreparedDataset, Row

review = recovery.review


def receipt():
    return recovery.parse_credit_response({"data": {"total_credits": "5", "total_usage": ".2",
        "account_id": "must-not-persist"}}, review.budget.utc_now())


@pytest.fixture
def halted(tmp_path, monkeypatch):
    area = tmp_path / "results/numeric_expansion"
    for key, value in {"ROOT": tmp_path, "AREA": area, "OUTPUT": area / "review",
            "LOCAL": area / "local", "SOURCES_DIR": area / "sources", "LEDGER": area / "review-budget.jsonl"}.items():
        monkeypatch.setattr(review, key, value)
    monkeypatch.setattr(review, "PRIOR", {})
    monkeypatch.setattr(review, "verify_transport", lambda: None)
    monkeypatch.setattr(review, "verify_prior", lambda *args: {})
    dataset = PreparedDataset("breast_cancer", ["negative", "positive"],
        [Row(f"train-{i}", f"feature={i/10}", i%2) for i in range(10)], [],
        [Row(f"test-{i}", f"feature={i+10}", i%2) for i in range(7)],
        {"tabular": {"features": [{"name": "feature", "kind": "numeric"}]}})
    monkeypatch.setattr(review, "load_native_prepared", lambda path: (dataset, None))
    spec = review.MODELS["qwen_small"]
    source_dir = tmp_path / "results/tabular/local/source"
    source_dir.mkdir(parents=True)
    proposals = [Prediction(row.id, row.label) for row in dataset.test]
    source_record = {"status": "complete", "dataset": dataset.name, "labels": dataset.labels, "seed": 42,
        "method": "zero_shot", "manifest_sha256": digest(dataset.manifest), "dataset_manifest": dataset.manifest,
        "test_ids_sha256": digest([r.id for r in dataset.test]), "training_example_ids": [],
        "implementation_sha256": review.base.FROZEN_CORE_SHA, "run_id": source_dir.name,
        "config": {"provider": "hf", "model": spec["model"], "revision": spec["revision"], "shots_per_class": 0},
        "metrics": evaluate(dataset.test, proposals, 2)}
    review.save_json(source_dir / "run.json", source_record)
    (source_dir / "predictions.jsonl").write_text("".join(json.dumps(asdict(p))+"\n" for p in proposals))
    review.save_json(source_dir / "test_manifest.json", review.base.expected_test_manifest(dataset))
    source = {"model": spec["model"], "provider": "hf", "path": source_dir.relative_to(tmp_path).as_posix(),
        "hashes": {name: review.file_sha(source_dir / name) for name in review.FILE_NAMES}}
    monkeypatch.setattr(review, "historical_source", lambda *args: source)
    job = review.load_source(dataset, "qwen_small", 0, freeze=True)
    config = {"provider": "jev", "model": review.route.MODEL, "base_url": review.route.BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY", "timeout": 120}
    review.save_json(tmp_path / "configs/tabular_hosted.json", {"jev_openrouter": config})
    inner = review.budget.Ledger(review.LEDGER, review.STAGE_CAP, initialize=True)
    review.LEDGER.with_name("review-execution.lock").write_text("")
    identity, guard = review.make_identity(job, config, inner.identity["ledger_id"], 100)
    directory = review.OUTPUT / "breast_cancer__qwen_small__k0__jev-review"
    directory.mkdir(parents=True)
    predictions = []
    for i, (row, proposal) in enumerate(zip(dataset.test[:4], proposals)):
        prompt_sha = recovery.sha(review.base.build_review_prompt(row, dataset.labels, [], proposal.label).encode())
        ident = inner.reserve(review.route.RESERVE_NANO, {**guard, "row_id": row.id, "prompt_sha256": prompt_sha,
            "model": review.route.MODEL, "provider": "jev", "pricing": review.route.PRICE})
        error = recovery.BILLING_ERROR if i else None
        inner.result(ident, {"outcome": "prediction_error" if error else "returned", "route": "OpenRouter",
            "reservation_released": False, "usage_exceeded_reservation_assumptions": False})
        prediction = Prediction(row.id, None if error else row.label, None if error else [1.,0.], latency_s=.1,
            error=error, metadata={"requested_model": review.route.MODEL,
                "budget": {"ledger_id": inner.identity["ledger_id"], "reservation_id": ident,
                    "reserved_upper_bound_usd": review.route.PRICE["per_request_reserved_usd"], "reservation_released": False},
                "review_prompt_sha256": prompt_sha,
                "proposal": review.base.proposal_metadata(source, source_record, proposal)})
        predictions.append(prediction)
    record = {**identity, "run_id": directory.name, "status": recovery.HALTED, "n_predictions": len(predictions),
        "execution": {"automatic_retries": 0}, "dataset_manifest": dataset.manifest}
    review.save_json(directory / "run.json", record)
    (directory / "predictions.jsonl").write_text("".join(json.dumps(asdict(p))+"\n" for p in predictions))

    def reconciled(_rows):
        events, anchor = recovery.snapshot_ledger(inner)
        n = sum(e["type"] == "reserve" for e in events)
        return {"status": "halted", "reservations_without_result": 0, "results_without_saved_prediction": 0,
            "partial_checkpoint_files": 0, "ledger_sha256": anchor["ledger_sha256"],
            "new_model_requests": n, "review_conservative_usd": review.budget.usd_string(n*review.route.RESERVE_NANO)}
    monkeypatch.setattr(recovery.costs, "collect_costs", reconciled)
    return {"job": job, "directory": directory, "inner": inner, "predictions": predictions,
        "config": config, "dataset": dataset, "record": record}


def audit():
    return recovery.audit_halt("breast_cancer", "qwen_small", 0)


def details_for(a, offset=0):
    row = a["job"]["dataset"].test[len(a["predictions"])+offset]
    proposal = a["job"]["predictions"][len(a["predictions"])+offset]
    prompt = review.base.build_review_prompt(row,a["job"]["dataset"].labels,a["job"]["examples"],proposal.label)
    return {**a["guard"], "row_id": row.id, "provider":"jev", "model":review.route.MODEL,
        "pricing":review.route.PRICE, "prompt_sha256":recovery.sha(prompt.encode())}


def test_dry_run_never_reads_key_or_calls_credit_and_preserves_every_byte(halted, monkeypatch):
    before = recovery.protected_hashes(halted["job"], halted["directory"])
    monkeypatch.setattr(recovery, "check_credit", lambda *_: pytest.fail("Dry run attempted account authentication"))
    monkeypatch.setattr(review.budget, "prompt_credentials", lambda *_: pytest.fail("Dry run requested a secret"))
    plan = recovery.main(["--dataset", "breast_cancer", "--model-key", "qwen_small", "--shots", "0", "--prompt-api-key"])
    assert plan["remaining_calls"] == 3 and plan["preserved_predictions"] == 4
    assert recovery.protected_hashes(halted["job"], halted["directory"]) == before
    assert not (review.AREA / "billing_recovery").exists()


def test_frozen_validator_rejects_changed_prompt_order_and_partial_write(halted):
    path = halted["directory"] / "predictions.jsonl"
    raw = path.read_text()
    rows = raw.splitlines()
    path.write_text("\n".join([rows[1], rows[0], *rows[2:]])+"\n")
    with pytest.raises(review.budget.GuardError, match="exact ordered prefix"):
        audit()
    path.write_text(raw.rstrip())
    with pytest.raises(review.budget.GuardError, match="Incomplete prediction write"):
        audit()
    changed = json.loads(rows[-1]); changed["metadata"]["review_prompt_sha256"] = "0"*64
    path.write_text("\n".join([*rows[:-1], json.dumps(changed)])+"\n")
    with pytest.raises(review.budget.GuardError, match="prompt or paid-call"):
        audit()


def test_nonbilling_failure_and_missing_reservation_are_ineligible(halted):
    path = halted["directory"] / "predictions.jsonl"
    original = path.read_text(); rows = original.splitlines()
    changed = json.loads(rows[-1]); changed["error"] = "http_error: status=429; no automatic retry"
    path.write_text("\n".join([*rows[:-1],json.dumps(changed)])+"\n")
    with pytest.raises(review.budget.GuardError, match="exact final triplet"):
        audit()
    path.write_text(original)
    halted["inner"].reserve(review.route.RESERVE_NANO, {"dataset":"breast_cancer","proposal_key":"qwen_small_k0","row_id":"uncertain"})
    with pytest.raises(review.budget.GuardError, match="Unreconciled reserved call"):
        audit()


@pytest.mark.parametrize("credits,usage", [(0,0),(0,".2"),("5","5"),("NaN",0),(True,0),("-1",0),("5","Infinity")])
def test_nonpositive_or_invalid_account_credit_is_rejected(credits, usage):
    with pytest.raises(review.budget.GuardError):
        recovery.parse_credit_response({"data":{"total_credits":credits,"total_usage":usage}},review.budget.utc_now())


def test_receipt_allowlist_and_freshness():
    value = receipt()
    assert value["available_credit_usd"] == "4.8"
    assert "account" not in json.dumps(value) and "must-not-persist" not in json.dumps(value)
    recovery.verify_receipt(value)
    value["checked_at"] = (datetime.now(timezone.utc)-timedelta(seconds=61)).isoformat()
    with pytest.raises(review.budget.GuardError, match="stale"):
        recovery.verify_receipt(value)


def test_authenticated_get_uses_fixed_route_no_redirect_and_sanitizes_response(monkeypatch):
    seen = []
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def geturl(self): return recovery.CREDITS_URL
        def read(self, limit):
            assert limit == 65537
            return b'{"data":{"total_credits":5,"total_usage":0.2,"account_id":"hidden"}}'
    class Opener:
        def open(self, request, timeout):
            seen.append((request.full_url,request.method,request.get_header("Authorization"),timeout))
            return Response()
    def opener(handler):
        assert isinstance(handler, recovery._NoRedirect)
        return Opener()
    monkeypatch.setattr(recovery.urllib.request, "build_opener", opener)
    value = recovery.check_credit("fixture-only-key")
    assert seen == [(recovery.CREDITS_URL,"GET","Bearer fixture-only-key",15)]
    assert "fixture-only-key" not in json.dumps(value) and "hidden" not in json.dumps(value)


def test_funding_failure_cannot_mutate_or_create_a_recovery(halted, monkeypatch):
    before = recovery.protected_hashes(halted["job"], halted["directory"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only-key")
    monkeypatch.setattr(recovery, "check_credit", lambda *_: recovery.parse_credit_response(
        {"data":{"total_credits":0,"total_usage":".2"}}, review.budget.utc_now()))
    with pytest.raises(review.budget.GuardError, match="not positive"):
        recovery.main(["--dataset","breast_cancer","--model-key","qwen_small","--shots","0","--execute"])
    assert recovery.protected_hashes(halted["job"],halted["directory"]) == before
    assert not (review.AREA / "billing_recovery").exists()


def test_clear_preserves_snapshot_predictions_ledgers_and_identity(halted):
    a = audit(); before = dict(a["protected"])
    destination, manifest = recovery.prepare_recovery(a, receipt())
    assert (destination / "halted-run.json").read_bytes() == a["record_raw"]
    record = json.loads((halted["directory"] / "run.json").read_text())
    assert record["status"] == "running"
    for key, value in halted["record"].items():
        if key not in {"status","execution","n_predictions"}: assert record[key] == value
    assert "n_predictions" not in record  # Frozen producer otherwise leaves a stale halt count after completion.
    assert record["execution"]["automatic_retries"] == 0
    assert record["execution"]["billing_recoveries"][0]["helper_sha256"] == review.file_sha(recovery.__file__)
    saved_receipt = record["execution"]["billing_recoveries"][0]["credit_receipt"]
    assert saved_receipt["positive_available_credit"] is True
    assert set(saved_receipt) == {"endpoint","method","authenticated","http_status","checked_at","positive_available_credit"}
    assert manifest["credit_receipt"] == saved_receipt
    assert "total_credits_usd" not in (destination / "manifest.json").read_text()
    assert "total_usage_usd" not in (halted["directory"] / "run.json").read_text()
    before[(halted["directory"] / "run.json").relative_to(review.ROOT).as_posix()] = manifest["cleared_run_sha256"]
    assert recovery.protected_hashes(halted["job"],halted["directory"]) == before
    with pytest.raises(review.budget.GuardError): recovery.prepare_recovery(a, receipt())


def test_source_or_ledger_change_after_audit_prevents_clear(halted):
    a = audit()
    source_path = review.ROOT / a["job"]["source"]["path"] / "predictions.jsonl"
    source_path.write_text(source_path.read_text()+"\n")
    with pytest.raises(review.budget.GuardError, match="Evidence changed"):
        recovery.prepare_recovery(a, receipt())
    assert json.loads((halted["directory"] / "run.json").read_text())["status"] == recovery.HALTED
    assert not (review.AREA / "billing_recovery").exists()


def test_one_time_exemption_then_original_guard_and_duplicate_protection(halted):
    a = audit(); ledger = recovery.BillingRecoveryLedger(a["inner"],a,receipt())
    with pytest.raises(review.budget.GuardError, match="next unseen row"):
        ledger.reserve(review.route.RESERVE_NANO, details_for(a,1))
    ident = ledger.reserve(review.route.RESERVE_NANO,details_for(a))
    assert ledger.exemption_used and ledger.new_requests == 1
    ledger.result(ident,{"outcome":"returned"})
    with pytest.raises(review.budget.GuardError, match="already reserved"):
        ledger.reserve(review.route.RESERVE_NANO,details_for(a))
    second = ledger.reserve(review.route.RESERVE_NANO,details_for(a,1))
    assert second != ident and ledger.new_requests == 2


def test_fresh_failure_does_not_get_another_exemption(halted):
    a = audit(); ledger = recovery.BillingRecoveryLedger(a["inner"],a,receipt())
    ident = ledger.reserve(review.route.RESERVE_NANO,details_for(a))
    ledger.result(ident,{"outcome":"prediction_error"})
    with pytest.raises(review.budget.GuardError, match="Three consecutive errors"):
        ledger.reserve(review.route.RESERVE_NANO,details_for(a,1))
    assert ledger.new_requests == 1


def test_exempted_request_must_keep_frozen_prompt_and_underlying_usd_cap(halted):
    a = audit(); ledger = recovery.BillingRecoveryLedger(a["inner"],a,receipt())
    changed = {**details_for(a), "prompt_sha256":"0"*64}
    with pytest.raises(review.budget.GuardError,match="provenance"):
        ledger.reserve(review.route.RESERVE_NANO,changed)
    a["inner"].budget = 4*review.route.RESERVE_NANO
    with pytest.raises(review.budget.BudgetStop,match="Budget exhausted"):
        ledger.reserve(review.route.RESERVE_NANO,details_for(a))
    assert not ledger.exemption_used and ledger.new_requests == 0


def test_head_change_cap_and_duplicate_guards_apply_to_exemption(halted, monkeypatch):
    a = audit(); ledger = recovery.BillingRecoveryLedger(a["inner"],a,receipt())
    monkeypatch.setattr(review,"MAX_REQUESTS",4)
    with pytest.raises(review.budget.BudgetStop): ledger.reserve(review.route.RESERVE_NANO,details_for(a))
    monkeypatch.setattr(review,"MAX_REQUESTS",1500)
    duplicate = {"dataset":"breast_cancer","proposal_key":"qwen_small_k0","row_id":"test-3"}
    with pytest.raises(review.budget.GuardError,match="already reserved"):
        ledger.reserve(review.route.RESERVE_NANO,duplicate)
    a["inner"].reserve(review.route.RESERVE_NANO,{"different":"condition"})
    with pytest.raises(review.budget.GuardError,match="head changed"):
        ledger.reserve(review.route.RESERVE_NANO,details_for(a))
    assert not ledger.exemption_used


def test_execute_uses_frozen_producer_and_keeps_prior_prefix(halted, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY","fixture-only-key")
    monkeypatch.setattr(recovery,"check_credit",lambda *_: receipt())
    class Provider:
        def __init__(self, config, ledger, guard): self.ledger,self.guard=ledger,guard
        def predict(self,row,labels,prompt):
            ident=self.ledger.reserve(review.route.RESERVE_NANO,{**self.guard,"row_id":row.id,"prompt_sha256":recovery.sha(prompt.encode()),
                "provider":"jev","model":review.route.MODEL,"pricing":review.route.PRICE})
            self.ledger.result(ident,{"outcome":"returned"})
            return Prediction(row.id,0,[1.,0.],metadata={"budget":{"reservation_id":ident,"ledger_id":self.ledger.identity["ledger_id"]}})
    monkeypatch.setattr(review.route,"GuardedOpenRouterJev",Provider)
    before=(halted["directory"] / "predictions.jsonl").read_bytes()
    result=recovery.main(["--dataset","breast_cancer","--model-key","qwen_small","--shots","0","--execute","--stop-after-new-requests","1"])
    assert result["status"] == "checkpoint" and result["new_calls"] == 1
    raw=(halted["directory"] / "predictions.jsonl").read_bytes()
    assert raw.startswith(before) and len(raw.splitlines()) == 5
    assert json.loads((halted["directory"] / "run.json").read_text())["status"] == "running"
    assert len(list((review.AREA / "billing_recovery").glob("*/completion.json"))) == 1
    # All original three billing failures remain in the source prefix.
    assert sum(json.loads(line)["error"] == recovery.BILLING_ERROR for line in raw.splitlines()) == 3


def test_execution_lock_blocks_a_second_worker(halted):
    import fcntl
    with review.LEDGER.with_name("review-execution.lock").open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(review.budget.GuardError,match="Another expanded review worker"):
            recovery.main(["--dataset","breast_cancer","--model-key","qwen_small","--shots","0"])
