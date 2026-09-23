"""Offline request reconstruction and immutable-original retry selection."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_failed_retries as retries
from jevbench.types import Prediction


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Inventory must not access network")
    monkeypatch.setattr("socket.socket.connect", reject)


@pytest.fixture(scope="module")
def inventory():
    return retries.build_inventory()


def test_all_known_original_failures_are_included_without_truth_or_prompts(inventory):
    counts = inventory["counts"]
    initial = retries.read(retries.DEFAULT_OUTPUT)
    identity = lambda item: (item["run_path"], item["row_id"], item["original_prediction_sha256"])
    # Whole JSONL/run/ledger hashes in the old snapshot may change by legitimate
    # append/completion; individual original failed records must remain present.
    assert {identity(item) for item in initial["failures"]} <= {identity(item) for item in inventory["failures"]}
    assert counts["paid_failed_calls"] == sum(f["kind"] == "paid_failure" for f in inventory["failures"])
    assert counts["upstream_skips"] == sum(f["kind"] == "upstream_skip" for f in inventory["failures"])
    assert any(f["dataset"] == "titanic" and f["scope"] == "historical_additional_dataset" for f in inventory["failures"])
    assert inventory["pending_first_pass"]["invalid_success_source_labels"] == 0
    assert inventory["pending_first_pass"]["failed_source_rows"] == 1
    assert len(inventory["pending_first_pass"]["source_runs"]) == 48
    def check(value):
        if isinstance(value, dict):
            assert not ({"prompt", "request_body", "labels", "label", "text", "truth", "api_key", "Authorization"} & set(value))
            for item in value.values(): check(item)
        elif isinstance(value, list):
            for item in value: check(item)
    check(inventory)


def test_every_paid_prompt_matches_its_original_reservation(inventory):
    for item in inventory["failures"]:
        if item["kind"] != "paid_failure": continue
        request = retries.reconstruct_request(item)
        assert request["prompt_sha256"] == item["original_prompt_sha256"]
        assert hashlib.sha256(request["prompt"].encode()).hexdigest() == item["original_prompt_sha256"]
        assert request["is_new_dependent_request"] is False
        assert "label" not in request and "row" not in request
        assert "api_key" not in request["config"]
        if item["provider"] == "openai":
            assert request["request_body"]["max_completion_tokens"] == 1024
            assert request["request_body"]["reasoning_effort"] == "low"
            assert item["original_retained_reservation_usd"] == "0.156650000"


def test_upstream_skip_requires_a_valid_new_source_and_stays_a_new_call(inventory):
    item = next(f for f in inventory["failures"] if f["kind"] == "upstream_skip")
    with pytest.raises(ValueError, match="Successful separately recorded"):
        retries.reconstruct_request(item)
    for prediction in [Prediction(item["row_id"], None, error="failed"), Prediction("wrong-row", 0),
                       Prediction(item["row_id"], 99)]:
        with pytest.raises(ValueError, match="Successful separately recorded"):
            retries.reconstruct_request(item, recovered_source_prediction=prediction)
    prediction = Prediction(item["row_id"], 0, metadata={"attempt": "separate-source-retry"})
    request = retries.reconstruct_request(item, recovered_source_prediction=prediction)
    assert request["original_prompt_sha256"] is None and request["is_new_dependent_request"] is True
    assert request["dependency"]["recovered_prediction_sha256"] == retries.prediction_sha256(asdict(prediction))
    assert request["dependency"]["source_original_run_id"] == item["source_condition"]["source_run_id"]
    assert item["depends_on_failure_id"] in {f["failure_id"] for f in inventory["failures"]}


def test_changed_identity_and_hash_are_rejected(inventory):
    from copy import deepcopy
    item = deepcopy(inventory["failures"][0])
    item["original_prediction_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Original failure identity"):
        retries.reconstruct_request(item)
    item = deepcopy(inventory["failures"][0])
    item["original_prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Reconstructed prompt"):
        retries.reconstruct_request(item)


def test_no_correctness_selection_or_unlimited_retry(inventory):
    policy = inventory["retry_policy"]
    assert policy["maximum_additional_attempts_per_original_failure"] == 2
    assert policy["stop_on_first_valid_success"] is True
    assert all(f["error"] for f in inventory["failures"])
    assert all("lora" not in f["method"] for f in inventory["failures"])
    assert len({f["failure_id"] for f in inventory["failures"]}) == len(inventory["failures"])


def test_scanner_leaves_all_pinned_originals_unchanged(inventory):
    for item in inventory["failures"]:
        assert all(retries.file_sha(retries.ROOT / path) == digest for path, digest in item["artifact_sha256"].items())


@pytest.mark.parametrize("changed", ["requests.jsonl", "protocol.json"])
def test_control_reconstruction_requires_original_execution_hashes(monkeypatch, changed):
    prompt = "fixture prompt"
    item = {"request_recipe": "control_frozen_request", "run_path": "results/review_controls/execution/run.json",
            "control_request_id": "request-1", "row_id": "row-1", "kind": "paid_failure",
            "original_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    record = {"requests_sha256": "original-requests", "protocol_sha256": "original-protocol"}
    monkeypatch.setattr(retries, "original_prediction", lambda *args: {})
    monkeypatch.setattr(retries, "read", lambda path: record)
    monkeypatch.setattr(retries, "file_sha", lambda path: "changed" if path.name == changed else
                        {"requests.jsonl": "original-requests", "protocol.json": "original-protocol"}[path.name])
    monkeypatch.setattr(retries, "read_predictions", lambda path: pytest.fail("Changed controls must not be reconstructed"))
    with pytest.raises(ValueError, match="original execution identity"):
        retries.reconstruct_request(item)
