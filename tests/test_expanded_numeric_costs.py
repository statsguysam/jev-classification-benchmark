"""Accounting must count calls, not rows or duplicate result records."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_expanded_numeric_costs as costs
from jevbench.types import Prediction


@pytest.fixture
def accounting(monkeypatch):
    key = ("wine", "astra_k0")
    monkeypatch.setattr(costs, "CONDITIONS", {key: ("astra", 0)})
    details = {"dataset": "wine", "proposal_key": "astra_k0", "row_id": "row-one", "prompt_sha256": "prompt-one"}
    result = {"outcome": "returned", "route": "OpenRouter", "provider": "TypeSafe",
        "resolved_model": "typesafe/jev-1.13-20260917", "request_id": "request-one",
        "reported_cost_usd": "0.000100", "reservation_released": False,
        "usage_exceeded_reservation_assumptions": False, "usage": {"input_tokens": 10, "output_tokens": 0}}
    events = [{"type": "header", "ledger_id": "ledger-one"},
        {"type": "reserve", "reservation_id": "reserve-one", "reserved_nano": costs.review.route.RESERVE_NANO, "details": details},
        {"type": "result", "reservation_id": "reserve-one", "details": result}]
    prediction = Prediction("row-one", 1, [0.1, 0.9], input_tokens=10, output_tokens=0,
        metadata={"resolved_model": result["resolved_model"], "review_prompt_sha256": "prompt-one",
            "budget": {"ledger_id": "ledger-one", "reservation_id": "reserve-one",
                "reservation_released": False, "reserved_upper_bound_usd": costs.review.route.PRICE["per_request_reserved_usd"]},
            "openrouter": {"endpoint": costs.review.route.ENDPOINT, "provider": "TypeSafe",
                "id": "request-one", "reported_cost_usd": "0.0001"}})
    kwargs = {"ledger_id": "ledger-one", "expected_reservations": {(*key, "row-one"): details},
              "complete_conditions": {key}, "complete_summary_conditions": {key}, "ledger_sha256": "ledgerhash"}
    return events, {"reserve-one": prediction}, kwargs


def test_unique_calls_costs_and_source_failures_reconcile(accounting):
    events, predictions, kwargs = accounting
    result = costs.reconcile_events(events, predictions, 1, **kwargs)
    assert result["status"] == "complete"
    assert result["new_model_requests"] == result["reported_cost_requests"] == 1
    assert result["unknown_cost_requests"] == 0
    assert result["known_reported_api_usd"] == "0.000100"
    assert result["review_conservative_usd"] == "0.002688000"
    assert result["source_failure_rows_without_jev_call"] == 1
    assert result["cumulative_conservative_usd"] == "19.328515700"


def test_duplicate_results_cannot_double_count_costs(accounting):
    events, predictions, kwargs = accounting
    events.append(copy.deepcopy(events[-1]))
    with pytest.raises(ValueError, match="Multiple result"):
        costs.reconcile_events(events, predictions, 0, **kwargs)


def test_inflight_reservation_remains_unknown_and_reserved(accounting):
    events, _, kwargs = accounting
    kwargs.update(complete_conditions=set(), complete_summary_conditions=set())
    result = costs.reconcile_events(events[:-1], {}, 0, **kwargs)
    assert result["status"] == "in_progress"
    assert result["reservations_without_result"] == 1
    assert result["reported_cost_requests"] == 0 and result["unknown_cost_requests"] == 1
    assert result["known_reported_api_usd"] == "0"
    assert result["review_conservative_usd"] == "0.002688000"


def test_reported_result_without_checkpoint_cost_still_counted(accounting):
    events, _, kwargs = accounting
    kwargs.update(complete_conditions=set(), complete_summary_conditions=set())
    result = costs.reconcile_events(events, {}, 0, **kwargs)
    assert result["status"] == "in_progress"
    assert result["results_without_saved_prediction"] == 1
    assert result["reported_cost_requests"] == 1
    assert result["known_reported_api_usd"] == "0.000100"


def test_completed_condition_cannot_hide_uncertain_call(accounting):
    events, _, kwargs = accounting
    with pytest.raises(ValueError, match="Completed condition.*unreconciled"):
        costs.reconcile_events(events, {}, 0, **kwargs)


def test_saved_prediction_requires_result_evidence(accounting):
    events, predictions, kwargs = accounting
    with pytest.raises(ValueError, match="no result evidence"):
        costs.reconcile_events(events[:-1], predictions, 0, **kwargs)


@pytest.mark.parametrize("field,value", [("reported_cost_usd", "0.0002"), ("provider", "other-provider"),
                                        ("id", "other-request"), ("endpoint", "https://elsewhere.test")])
def test_prediction_evidence_must_match_ledger(accounting, field, value):
    events, predictions, kwargs = accounting
    predictions["reserve-one"].metadata["openrouter"][field] = value
    with pytest.raises(ValueError, match="cost/provider/model/request"):
        costs.reconcile_events(events, predictions, 0, **kwargs)


def test_model_budget_and_prompt_provenance_are_checked(accounting):
    events, predictions, kwargs = accounting
    prediction = predictions["reserve-one"]
    prediction.metadata["resolved_model"] = "other-model"
    with pytest.raises(ValueError, match="cost/provider/model/request"):
        costs.reconcile_events(events, predictions, 0, **kwargs)
    prediction.metadata["resolved_model"] = events[-1]["details"]["resolved_model"]
    prediction.metadata["budget"]["ledger_id"] = "other-ledger"
    with pytest.raises(ValueError, match="budget provenance"):
        costs.reconcile_events(events, predictions, 0, **kwargs)
    prediction.metadata["budget"]["ledger_id"] = "ledger-one"
    prediction.metadata["review_prompt_sha256"] = "changed-prompt"
    with pytest.raises(ValueError, match="row or prompt"):
        costs.reconcile_events(events, predictions, 0, **kwargs)


def test_failure_unknown_cost_is_not_zero(accounting):
    events, predictions, kwargs = accounting
    result = events[-1]["details"]
    result.update(outcome="prediction_error", provider=None, resolved_model=None,
                  reported_cost_usd=None, request_id=None, usage={"input_tokens": None, "output_tokens": None})
    prediction = predictions["reserve-one"]
    prediction.label, prediction.probabilities, prediction.error = None, None, "network_error"
    prediction.input_tokens = prediction.output_tokens = None
    prediction.metadata["resolved_model"] = None
    prediction.metadata["openrouter"].update(provider=None, reported_cost_usd=None, id=None)
    result = costs.reconcile_events(events, predictions, 0, **kwargs)
    assert result["reported_cost_requests"] == 0
    assert result["unknown_cost_requests"] == 1
    assert result["review_conservative_usd"] == "0.002688000"


def test_zero_reported_cost_is_known(accounting):
    events, predictions, kwargs = accounting
    events[-1]["details"]["reported_cost_usd"] = "0"
    predictions["reserve-one"].metadata["openrouter"]["reported_cost_usd"] = "0.0000"
    result = costs.reconcile_events(events, predictions, 0, **kwargs)
    assert result["reported_cost_requests"] == 1 and result["unknown_cost_requests"] == 0


def test_released_reservations_and_unrecognized_paid_rows_rejected(accounting):
    events, predictions, kwargs = accounting
    events.append({"type": "settle", "reservation_id": "reserve-one"})
    with pytest.raises(ValueError, match="never be released"):
        costs.reconcile_events(events, predictions, 0, **kwargs)
    events.pop()
    events[1]["details"]["row_id"] = "unexpected-row"
    with pytest.raises(ValueError, match="Unrecognized or repeated paid row"):
        costs.reconcile_events(events, predictions, 0, **kwargs)


def test_partial_checkpoint_is_not_final(accounting):
    events, predictions, kwargs = accounting
    result = costs.reconcile_events(events, predictions, 0, partial_checkpoint_files=1, **kwargs)
    assert result["status"] == "in_progress"


def test_catalog_excludes_four_previously_paid_reviews():
    assert len(costs.CONDITIONS) == len(costs.RUN_IDS) == 20
    assert costs.review.MAX_REQUESTS == 1500
    assert costs.review.STAGE_CAP == "5.00" and costs.review.TOTAL_CAP == "25.00"
    for dataset in ("breast_cancer", "wine"):
        assert (dataset, "qwen_main_k4") not in costs.CONDITIONS
        assert (dataset, "astra_k4") not in costs.CONDITIONS
        assert (dataset, "qwen_main_k0") in costs.CONDITIONS
        assert (dataset, "astra_k0") in costs.CONDITIONS
        assert (dataset, "granite_k4") in costs.CONDITIONS


def test_reused_four_shot_review_cannot_be_charged_to_new_allowance(accounting):
    events, predictions, kwargs = accounting
    events[1]["details"]["proposal_key"] = "astra_k4"
    kwargs["expected_reservations"]["wine", "astra_k4", "row-one"] = events[1]["details"]
    with pytest.raises(ValueError, match="Old or unplanned review"):
        costs.reconcile_events(events, predictions, 0, **kwargs)


def test_reported_cost_above_bound_cannot_be_hidden_by_reservation(accounting):
    events, predictions, kwargs = accounting
    events[-1]["details"]["reported_cost_usd"] = "0.002688001"
    predictions["reserve-one"].metadata["openrouter"]["reported_cost_usd"] = "0.002688001"
    with pytest.raises(ValueError, match="exceeds the bound"):
        costs.reconcile_events(events, predictions, 0, **kwargs)


def test_cumulative_cap_is_separate_from_stage_cap(accounting, monkeypatch):
    events, predictions, kwargs = accounting
    monkeypatch.setattr(costs.review, "PRIOR_TOTAL", "24.999000")
    with pytest.raises(ValueError, match="Cumulative authorization"):
        costs.reconcile_events(events, predictions, 0, **kwargs)
    monkeypatch.setattr(costs.review, "PRIOR_TOTAL", "19.325827700")
    monkeypatch.setattr(costs.review, "STAGE_CAP", "0.002")
    with pytest.raises(ValueError, match="stage cap"):
        costs.reconcile_events(events, predictions, 0, **kwargs)


def test_repeated_row_with_new_reservation_id_is_not_a_new_experiment(accounting):
    events, predictions, kwargs = accounting
    duplicate = copy.deepcopy(events[1])
    duplicate["reservation_id"] = "reserve-two"
    events.append(duplicate)
    with pytest.raises(ValueError, match="repeated paid row"):
        costs.reconcile_events(events, predictions, 0, **kwargs)


def test_zero_and_four_shot_calls_are_distinct_allowed_conditions(accounting, monkeypatch):
    events, predictions, kwargs = accounting
    # Use a source whose zero/four conditions are both newly reviewed.
    conditions = {("wine", "luna_k0"): ("luna", 0), ("wine", "luna_k4"): ("luna", 4)}
    monkeypatch.setattr(costs, "CONDITIONS", conditions)
    events[1]["details"]["proposal_key"] = "luna_k0"
    second_reserve, second_result = copy.deepcopy(events[1]), copy.deepcopy(events[2])
    second_reserve["reservation_id"] = second_result["reservation_id"] = "reserve-two"
    second_reserve["details"]["proposal_key"] = "luna_k4"
    second_reserve["details"]["prompt_sha256"] = "prompt-four"
    second_result["details"]["request_id"] = "request-two"
    second_prediction = copy.deepcopy(predictions["reserve-one"])
    second_prediction.metadata["budget"]["reservation_id"] = "reserve-two"
    second_prediction.metadata["openrouter"]["id"] = "request-two"
    second_prediction.metadata["review_prompt_sha256"] = "prompt-four"
    events.extend([second_reserve, second_result])
    predictions["reserve-two"] = second_prediction
    kwargs.update(expected_reservations={
        ("wine", "luna_k0", "row-one"): events[1]["details"],
        ("wine", "luna_k4", "row-one"): second_reserve["details"]},
        complete_conditions=set(conditions), complete_summary_conditions=set(conditions))
    result = costs.reconcile_events(events, predictions, 0, **kwargs)
    assert result["status"] == "complete" and result["new_model_requests"] == 2
    assert [item["shots_per_class"] for item in result["condition_costs"]] == [0, 4]
    assert result["known_reported_api_usd"] == "0.000200"
    monkeypatch.setattr(costs.review, "MAX_REQUESTS", 1)
    with pytest.raises(ValueError, match="request limit"):
        costs.reconcile_events(events, predictions, 0, **kwargs)
