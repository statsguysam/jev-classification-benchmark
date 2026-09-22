import copy
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import summarize_tabular_costs as costs
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import _identity
from jevbench.types import Prediction, PreparedDataset, Row


class Fixture:
    def __init__(self, root):
        self.root = root
        self.configs = costs.hosted.validate_models(costs.ROOT / "configs/tabular_hosted.json", list(costs.MODEL_KEYS))
        self.dataset = PreparedDataset("titanic", ["no", "yes"],
            [Row(f"train-{label}-{index}", f"feature={label * 10 + index}", label) for label in range(2) for index in range(4)],
            [Row("val-0", "feature=100", 0), Row("val-1", "feature=110", 1)],
            [Row("test-0", "feature=200", 0), Row("test-1", "feature=210", 1)], {"fixture": True})
        self.ledger = costs.budget.Ledger(root / "results/tabular/api-budget.jsonl", costs.hosted.STAGE_CAP, initialize=True)
        self.sources = {"tabular_wrapper_sha256": costs.file_sha((costs.ROOT / "scripts/run_tabular_hosted.py").read_bytes())}
        self.artifacts = {}

    def add(self, key="jev_openrouter", *, index=0, outcome="returned", reported="0.000004200",
            settle=False, persist=True, shots=0):
        config, dataset = self.configs[key], self.dataset
        guard = costs.expected_guard(dataset.name, shots, key, self.ledger.identity["ledger_id"], self.sources, self.configs)
        examples = select_examples(dataset.train, dataset.labels, shots, 42)
        row = dataset.test[index]
        prompt = build_prompt(row, dataset.labels, examples)
        price = costs.price_for(config)
        amount, bound = ((costs.route.RESERVE_NANO, None) if key == "jev_openrouter" else costs.budget.reservation(config, price, prompt))
        details = {"row_id": row.id, "provider": config["provider"], "model": config["model"],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "pricing": price, **guard}
        if bound is not None:
            details["bound"] = bound
        rid = self.ledger.reserve(amount, details)
        if outcome == "pending":
            return rid
        if outcome == "raised":
            self.ledger.result(rid, {"outcome": "raised", "exception_type": "RuntimeError", "usage_unknown": True})
            return rid
        failed = outcome == "prediction_error"
        inputs, outputs = (None, None) if failed else (100, 3)
        actual = None if failed else ("typesafe/jev-1.13-20260917" if key == "jev_openrouter" else config["model"])
        pay = {"ledger_id": self.ledger.identity["ledger_id"], "reservation_id": rid,
               "reserved_upper_bound_usd": costs.budget.usd_string(amount)}
        metadata = {"requested_model": config["model"], "resolved_model": actual, "budget": pay}
        result = {"outcome": outcome, "usage": {"input_tokens": inputs, "output_tokens": outputs},
                  "resolved_model": actual, "usage_exceeded_reservation_assumptions": False}
        if key == "jev_openrouter":
            request_id = None if failed else "fixture-" + rid
            cost = None if failed else reported
            result.update(route="OpenRouter", request_id=request_id, provider=None if failed else "TypeSafe",
                reported_cost_usd=cost, reservation_released=False,
                guard_reasons={"reported_usage_or_cost_overrun": False, "unexpected_route_or_snapshot": False, "invalid_reported_cost": False})
            metadata["openrouter"] = {"id": request_id, "provider": result["provider"], "reported_cost_usd": cost, "endpoint": costs.route.ENDPOINT}
            pay["reservation_released"] = False
        else:
            prediction = Prediction(row.id, None if failed else row.label, input_tokens=inputs, output_tokens=outputs)
            result["reported_usage_estimate"] = costs.budget._usage_estimate(prediction, config, price)
            pay.update(reported_usage_estimate=result["reported_usage_estimate"], **bound)
        self.ledger.result(rid, result)
        if settle:
            charge = costs.budget.usd_nano((Decimal(inputs)*Decimal(price["input_usd_per_million"])*Decimal(price["input_multiplier"])+Decimal(outputs)*Decimal(price["output_usd_per_million"]))/costs.MILLION)
            self.ledger.settle(rid, charge, {"usage": result["usage"], "model": actual,
                                          "pricing_sha256": costs.budget.sha(price)})
            pay["settled_conservative_usd"] = costs.budget.usd_string(charge)
        if persist:
            condition = (key, shots)
            if condition not in self.artifacts:
                run_config = {**config, "budget_guard": guard, "shots_per_class": shots}
                record = _identity(dataset, "zero_shot" if shots == 0 else "few_shot", run_config, 42, examples)
                run_id = f"fixture-{key}-{shots}"
                record.update(run_id=run_id, status="running", dataset_manifest=dataset.manifest)
                self.artifacts[condition] = {"record": record, "rows": [], "test": None,
                    "partial_final_prediction_line": False, "source_path": f"results/tabular/hosted/{run_id}/run.json"}
            self.artifacts[condition]["rows"].append(asdict(Prediction(row.id, None if failed else row.label,
                input_tokens=inputs, output_tokens=outputs, error="network_error" if failed else None, metadata=metadata)))
        return rid

    def snapshot(self):
        with self.ledger._locked():
            events, accounted = self.ledger._read()
        return {"events": events, "accounted_nano": accounted, "ledger_id": self.ledger.identity["ledger_id"],
            "captured_at": datetime.now(timezone.utc).isoformat(), "artifacts": list(self.artifacts.values()),
            "sources": self.sources}

    def analyze(self):
        return costs.analyze(self.snapshot(), {"titanic": self.dataset}, self.configs)


@pytest.fixture
def fixture(tmp_path):
    return Fixture(tmp_path)


def test_mixed_provider_exact_accounting_preserves_ledger_and_distinguishes_cost_basis(fixture):
    fixture.add()
    fixture.add("openai_economical", settle=True)
    before = fixture.ledger.path.read_bytes(), fixture.ledger.lock_path.read_bytes()
    result = fixture.analyze()
    assert before == (fixture.ledger.path.read_bytes(), fixture.ledger.lock_path.read_bytes())
    assert result["tabular"]["reservations"] == 2 and result["tabular"]["settlements"] == 1
    assert result["prediction_reservations_reconciled"] == 2
    assert Decimal(result["tabular"]["accounted_conservative_usd"]) == Decimal("0.002716600")
    assert Decimal(result["cumulative_accounted_conservative_usd"]) == Decimal("5.295723600")
    assert result["by_model"]["gpt-5.6-luna"]["reported_api_cost_usd"] is None
    assert result["by_model"]["gpt-5.6-luna"]["reported_api_cost_known_requests"] == 0
    assert Decimal(result["by_model"]["gpt-5.6-luna"]["token_rate_estimate_usd"]) == Decimal("0.0000236")
    assert result["tabular"]["reported_api_cost_coverage"] == .5
    assert result["status"] == "in_progress_or_incomplete"


def test_interrupted_and_resumed_attempts_both_remain_accounted(fixture):
    pending = fixture.add(outcome="pending", persist=False)
    resumed = fixture.add()
    result = fixture.analyze()
    assert pending != resumed
    assert result["tabular"]["reservations"] == 2
    assert result["tabular"]["requests_without_final_outcome"] == 1
    assert result["unpersisted_reservation_ids"] == [pending]
    assert result["persisted_prediction_rows"] == 1
    assert Decimal(result["tabular"]["accounted_conservative_usd"]) == Decimal("0.005376")


def test_final_outcome_without_persisted_prediction_is_still_costed(fixture):
    rid = fixture.add("openai_frontier", settle=True, persist=False)
    result = fixture.analyze()
    assert result["unpersisted_reservation_ids"] == [rid]
    assert result["tabular"]["final_outcomes"] == 1 and result["tabular"]["settlements"] == 1
    assert result["persisted_prediction_rows"] == 0
    assert Decimal(result["tabular"]["accounted_conservative_usd"]) == Decimal("0.0014")


def test_failed_and_raised_requests_preserve_unknown_usage_and_full_reservations(fixture):
    fixture.add(outcome="prediction_error")
    fixture.add("openai_economical", outcome="raised", persist=False)
    result = fixture.analyze()
    assert result["tabular"]["reported_api_cost_usd"] is None
    assert result["tabular"]["token_rate_estimate_usd"] is None
    assert result["tabular"]["reported_api_cost_known_requests"] == 0
    assert result["tabular"]["retained_full_reservations"] == 2
    assert result["tabular"]["outcomes"] == {"prediction_error": 1, "raised": 1}


@pytest.mark.parametrize("reported,known", [("0", 1), (None, 0)])
def test_zero_api_cost_is_distinct_from_missing(fixture, reported, known):
    fixture.add(reported=reported)
    result = fixture.analyze()["by_model"]["typesafe/jev-1.13"]
    assert result["reported_api_cost_known_requests"] == known
    assert result["reported_api_cost_usd"] == ("0" if known else None)
    assert Decimal(result["token_rate_estimate_usd"]) == Decimal("0.0000042")
    assert Decimal(result["accounted_conservative_usd"]) == Decimal("0.002688")


@pytest.mark.parametrize("field,change,match", [
    ("input_tokens", 99, "token usage differs"),
    ("error", "network_error", "outcome differs"),
])
def test_prediction_usage_and_outcome_tampering_rejected(fixture, field, change, match):
    fixture.add()
    fixture.artifacts["jev_openrouter", 0]["rows"][0][field] = change
    with pytest.raises(ValueError, match=match):
        fixture.analyze()


@pytest.mark.parametrize("field,value,match", [
    ("id", "wrong", "request ID or route differs"),
    ("reported_cost_usd", "0.00000001", "API cost differs"),
])
def test_provider_request_id_and_reported_cost_must_match_ledger(fixture, field, value, match):
    fixture.add()
    fixture.artifacts["jev_openrouter", 0]["rows"][0]["metadata"]["openrouter"][field] = value
    with pytest.raises(ValueError, match=match):
        fixture.analyze()


def test_prediction_cannot_reuse_another_reservation(fixture):
    first = fixture.add()
    fixture.add(index=1)
    fixture.artifacts["jev_openrouter", 0]["rows"][1]["metadata"]["budget"]["reservation_id"] = first
    with pytest.raises(ValueError, match="Duplicate or unmatched"):
        fixture.analyze()


def test_duplicate_final_outcomes_rejected(fixture):
    rid = fixture.add()
    fixture.ledger.result(rid, {"outcome": "raised", "usage_unknown": True})
    with pytest.raises(ValueError, match="Duplicate result"):
        fixture.analyze()


def test_openai_settlement_recomputed_instead_of_trusted(fixture):
    fixture.add("openai_economical", settle=True)
    snapshot = fixture.snapshot()
    next(event for event in snapshot["events"] if event["type"] == "settle")["charged_nano"] = 1
    with pytest.raises(ValueError, match="Settlement differs"):
        costs.analyze(snapshot, {"titanic": fixture.dataset}, fixture.configs)


def test_frozen_prompt_and_ancestral_budget_provenance_checked(fixture):
    fixture.add()
    for field, bad, match in (("prompt_sha256", "0"*64, "prompt differs"),
                              ("prior_charged_or_reserved_usd", "0", "provenance differs"),
                              ("tabular_wrapper_sha256", "0"*64, "provenance differs")):
        snapshot = copy.deepcopy(fixture.snapshot())
        next(event for event in snapshot["events"] if event["type"] == "reserve")["details"][field] = bad
        with pytest.raises(ValueError, match=match):
            costs.analyze(snapshot, {"titanic": fixture.dataset}, fixture.configs)


def test_falsely_completed_run_rejected(fixture):
    fixture.add()
    fixture.artifacts["jev_openrouter", 0]["record"]["status"] = "complete"
    with pytest.raises(ValueError, match="Complete run lacks"):
        fixture.analyze()


def test_complete_matrix_requires_every_prediction_and_includes_each_condition(fixture):
    dataset = fixture.dataset
    for key in costs.MODEL_KEYS:
        for shots in (0, 4):
            for index in range(len(dataset.test)):
                fixture.add(key, index=index, shots=shots, settle=key != "jev_openrouter")
            artifact = fixture.artifacts[key, shots]
            artifact["record"].update(status="complete", metrics={"n_test": 2, "n_failures": 0})
            artifact["test"] = {"dataset": dataset.name, "labels": dataset.labels,
                "manifest_sha256": costs.digest(dataset.manifest), "rows": [{"id": row.id,
                "label": row.label, "text_sha256": costs.file_sha(row.text.encode())} for row in dataset.test]}
    result = fixture.analyze()
    assert result["status"] == "complete"
    assert result["complete_runs"] == result["expected_runs"] == 6
    assert result["persisted_prediction_rows"] == result["expected_prediction_rows"] == 12
    assert result["reservations_without_persisted_prediction"] == 0
    assert len(result["by_condition"]) == 6
    assert all(item["reservations"] == 2 for item in result["by_condition"])


def test_mismatched_prediction_settlement_and_snapshot_total_rejected(fixture):
    fixture.add("openai_economical", settle=True)
    snapshot = fixture.snapshot()
    snapshot["accounted_nano"] += 1
    with pytest.raises(ValueError, match="accounting differs from ledger"):
        costs.analyze(snapshot, {"titanic": fixture.dataset}, fixture.configs)
    fixture.artifacts["openai_economical", 0]["rows"][0]["metadata"]["budget"]["settled_conservative_usd"] = "0"
    with pytest.raises(ValueError, match="settlement amount differs"):
        fixture.analyze()


def test_only_incomplete_last_jsonl_line_can_be_deferred():
    assert costs.parse_predictions(b'{"row_id":"a"}\n{"row_', False) == ([{"row_id": "a"}], True)
    with pytest.raises(ValueError):
        costs.parse_predictions(b'{"row_id":"a"}\n{"row_', True)
    with pytest.raises(ValueError):
        costs.parse_predictions(b'{"broken":\n{"row_id":"a"}\n', False)


def test_capture_existing_ledger_is_read_only_and_report_writes_only_new_outputs(fixture, monkeypatch):
    fixture.add()
    fixture.add("openai_economical", settle=True)
    root = fixture.root
    (root / "scripts").mkdir()
    (root / "configs").mkdir()
    for name in ("run_tabular_hosted.py", "run_openrouter_jev.py", "run_budgeted_hosted.py"):
        (root / "scripts" / name).write_bytes((costs.ROOT / "scripts" / name).read_bytes())
    (root / "configs/tabular_hosted.json").write_bytes((costs.ROOT / "configs/tabular_hosted.json").read_bytes())
    for artifact in fixture.artifacts.values():
        path = root / artifact["source_path"]
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(artifact["record"]))
        path.with_name("predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in artifact["rows"]))
    monkeypatch.setattr(costs.hosted, "verify_prior", lambda _: {"fixture_prior": {"charged_or_reserved_usd": costs.hosted.PRIOR_TOTAL}})
    monkeypatch.setattr(costs, "DATASETS", ("titanic",))
    monkeypatch.setattr(costs, "load_native_prepared", lambda _: (fixture.dataset, {}))
    before = fixture.ledger.path.read_bytes(), fixture.ledger.lock_path.read_bytes()
    report = costs.build_report(root)
    costs.write_report(report, root)
    assert before == (fixture.ledger.path.read_bytes(), fixture.ledger.lock_path.read_bytes())
    assert (root / "results/tabular/API_COST_SUMMARY.json").exists()
    text = (root / "results/TABULAR_API_COST_SUMMARY.md").read_text()
    assert "Missing costs/usage remain unknown" in text
    assert "did not retain provider request IDs" in text
