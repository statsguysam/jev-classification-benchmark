"""Safety, provenance and alignment checks for the cached-label review extension."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_numeric_jev_review as review
from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import digest
from jevbench.types import Prediction, PreparedDataset, Row


@pytest.fixture
def study():
    dataset = PreparedDataset("breast_cancer", ["negative", "positive"],
        [Row(f"train-{i}", f"feature={i / 10}", i % 2) for i in range(10)],
        [Row("validation-0", "feature=99", 0)],
        [Row("SECRET_TEST_IDENTIFIER_A", "feature=2.1", 1), Row("SECRET_TEST_IDENTIFIER_B", "feature=2.2", 0)],
        {"tabular": {"features": [{"name": "feature", "kind": "numeric"}]}})
    source = {"model": "Qwen/Qwen3-4B-Instruct-2507", "provider": "hf", "path": "results/example", "hashes": {"run.json": "abc"}}
    proposals = [Prediction(row.id, row.label) for row in dataset.test]
    examples = select_examples(dataset.train, dataset.labels, 4, 42)
    record = {"run_id": "example", "status": "complete", "dataset": dataset.name, "labels": dataset.labels,
        "seed": 42, "method": "few_shot", "manifest_sha256": digest(dataset.manifest),
        "dataset_manifest": dataset.manifest, "test_ids_sha256": digest([r.id for r in dataset.test]),
        "training_example_ids": [r.id for r in examples], "implementation_sha256": review.FROZEN_CORE_SHA,
        "config": {"model": source["model"], "provider": "hf", "shots_per_class": 4},
        "metrics": evaluate(dataset.test, proposals, len(dataset.labels))}
    return dataset, source, record, proposals, examples


def test_review_prompt_excludes_test_id_target_and_validation(study):
    dataset, _, _, _, examples = study
    row = dataset.test[0]
    prompt = review.build_review_prompt(row, dataset.labels, examples, 0)
    altered_target = replace(row, label=0)
    assert prompt == review.build_review_prompt(altered_target, dataset.labels, examples, 0)
    assert row.id not in prompt
    assert dataset.validation[0].text not in prompt
    assert row.text in prompt
    assert "Cached proposal from a separate model: class ID 0." in prompt
    assert "Classify the original numerical row in Final text" in prompt
    assert all(example.text in prompt for example in examples)
    assert "Qwen" not in prompt and "source_run_id" not in prompt


@pytest.mark.parametrize("proposal", [None, True, -1, 2, "1"])
def test_review_rejects_invalid_proposal(study, proposal):
    dataset, _, _, _, examples = study
    with pytest.raises(review.budget.GuardError, match="class ID"):
        review.build_review_prompt(dataset.test[0], dataset.labels, examples, proposal)


def test_source_requires_complete_ordered_rows_and_exact_manifest(study):
    dataset, source, record, proposals, examples = study
    manifest = review.expected_test_manifest(dataset)
    assert review.validate_source(dataset, source, record, proposals, manifest) == examples
    with pytest.raises(review.budget.GuardError, match="align"):
        review.validate_source(dataset, source, record, proposals[::-1], manifest)
    with pytest.raises(review.budget.GuardError, match="align"):
        review.validate_source(dataset, source, record, proposals[:1], manifest)
    changed = copy.deepcopy(manifest)
    changed["rows"][0]["label"] = 0
    with pytest.raises(review.budget.GuardError, match="manifest"):
        review.validate_source(dataset, source, record, proposals, changed)


@pytest.mark.parametrize("field,value", [("shots_per_class", 0), ("model", "another-model"), ("provider", "openai")])
def test_source_rejects_wrong_model_and_shots(study, field, value):
    dataset, source, record, proposals, _ = study
    record["config"][field] = value
    with pytest.raises(review.budget.GuardError, match="exact cached"):
        review.validate_source(dataset, source, record, proposals, review.expected_test_manifest(dataset))


def test_source_rejects_changed_training_ids(study):
    dataset, source, record, proposals, _ = study
    record["training_example_ids"][0] = dataset.validation[0].id
    with pytest.raises(review.budget.GuardError, match="training/test"):
        review.validate_source(dataset, source, record, proposals, review.expected_test_manifest(dataset))


def test_prior_anchors_detect_any_change_and_do_not_mutate(tmp_path, monkeypatch):
    path = tmp_path / "prior.jsonl"
    ledger = review.budget.Ledger(path, "1.00", initialize=True)
    ledger.reserve(review.budget.usd_nano("0.20"), {"purpose": "old"})
    lock = path.with_name(path.name + ".lock")
    before = path.read_bytes(), lock.read_bytes()
    monkeypatch.setattr(review, "PRIOR", {"prior.jsonl": {"cap": "1.00", "sha256": review.file_sha(path), "lock_sha256": review.file_sha(lock)}})
    monkeypatch.setattr(review, "PRIOR_TOTAL", "0.20")
    assert review.verify_prior(tmp_path)["prior.jsonl"]["charged_or_reserved_usd"] == "0.200000000"
    assert before == (path.read_bytes(), lock.read_bytes())
    with lock.open("a") as file:
        file.write(" ")
    with pytest.raises(review.budget.GuardError, match="Prior ledger/lock"):
        review.verify_prior(tmp_path)


def test_reserve_checks_prior_each_time_and_blocks_repeat(tmp_path, monkeypatch):
    checks = []
    monkeypatch.setattr(review, "verify_prior", lambda root: checks.append(root))
    monkeypatch.setattr(review, "verify_transport", lambda: None)
    inner = review.budget.Ledger(tmp_path / "new.jsonl", review.STAGE_CAP, initialize=True)
    ledger = review.AnchoredLedger(inner, root=tmp_path)
    details = {"dataset": "wine", "proposal_key": "astra", "row_id": "one"}
    ledger.reserve(review.jev_route.RESERVE_NANO, details)
    with pytest.raises(review.budget.GuardError, match="already reserved"):
        ledger.reserve(review.jev_route.RESERVE_NANO, details)
    assert len(checks) == 2
    assert inner.snapshot()["reservations"] == 1
    monkeypatch.setattr(review, "MAX_REQUESTS", 1)
    with pytest.raises(review.budget.BudgetStop, match="300"):
        ledger.reserve(review.jev_route.RESERVE_NANO, {**details, "row_id": "two"})


def test_checkpoint_refuses_changed_identity_proposal_or_partial_write(tmp_path, study):
    dataset, source, record, proposals, examples = study
    identity = {"method": "cached_label_jev_review", "proposal_sha": "original"}
    cache = {**identity, "status": "running"}
    (tmp_path / "run.json").write_text(json.dumps(cache))
    prediction = Prediction(dataset.test[0].id, 1, [0.1, 0.9], metadata={
        "proposal": review.proposal_metadata(source, record, proposals[0]),
        "review_prompt_sha256": review.hashlib.sha256(review.build_review_prompt(dataset.test[0], dataset.labels, examples, 1).encode()).hexdigest(),
        "budget": {"reservation_id": "reservation-one"}})
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(asdict(prediction)) + "\n")
    assert len(review.load_checkpoint(tmp_path, identity, dataset, source, record, proposals, examples)[1]) == 1
    with pytest.raises(review.budget.GuardError, match="identity changed"):
        review.load_checkpoint(tmp_path, {**identity, "proposal_sha": "changed"}, dataset, source, record, proposals, examples)
    proposals[0].label = 0
    with pytest.raises(review.budget.GuardError, match="proposal provenance"):
        review.load_checkpoint(tmp_path, identity, dataset, source, record, proposals, examples)
    path.write_text('{"row_id":')
    with pytest.raises(review.budget.GuardError, match="Incomplete prediction"):
        review.load_checkpoint(tmp_path, identity, dataset, source, record, proposals, examples)


def test_checkpoint_rejects_unbounded_or_missing_choice_probabilities(tmp_path, study):
    dataset, source, record, proposals, examples = study
    identity = {"method": "cached_label_jev_review"}
    (tmp_path / "run.json").write_text(json.dumps({**identity, "status": "running"}))
    prediction = Prediction(dataset.test[0].id, 99, [0.5, 0.5])
    (tmp_path / "predictions.jsonl").write_text(json.dumps(asdict(prediction)) + "\n")
    with pytest.raises(review.budget.GuardError, match="Choice label/probabilities"):
        review.load_checkpoint(tmp_path, identity, dataset, source, record, proposals, examples)


def test_three_provider_errors_halt_across_condition_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "verify_prior", lambda root: None)
    monkeypatch.setattr(review, "verify_transport", lambda: None)
    inner = review.budget.Ledger(tmp_path / "new.jsonl", review.STAGE_CAP, initialize=True)
    for i in range(3):
        ident = inner.reserve(review.jev_route.RESERVE_NANO, {"dataset": "wine", "proposal_key": "qwen3_4b", "row_id": str(i)})
        inner.result(ident, {"outcome": "prediction_error"})
    ledger = review.AnchoredLedger(inner)
    with pytest.raises(review.budget.GuardError, match="Three consecutive"):
        ledger.reserve(review.jev_route.RESERVE_NANO, {"dataset": "breast_cancer", "proposal_key": "astra", "row_id": "next"})
    assert inner.snapshot()["reservations"] == 3


def test_failed_source_proposal_stays_failure_without_call(tmp_path, monkeypatch, study):
    dataset, source, record, proposals, examples = study
    dataset.test = [dataset.test[0]]
    proposals = [Prediction(dataset.test[0].id, None, error="upstream_timeout")]
    monkeypatch.setattr(review, "OUTPUT", tmp_path)
    monkeypatch.setattr(review, "validate_reservations", lambda *args: None)

    class NoCalls:
        def __init__(self, *args):
            pass

        def predict(self, *args):
            raise AssertionError("A failed cached proposal must not call Jev")

    class FakeLedger:
        identity = {"ledger_id": "test-ledger"}

    monkeypatch.setattr(review.jev_route, "GuardedOpenRouterJev", NoCalls)
    result = review.run_review(dataset, "qwen3_4b", source, record, proposals, examples,
        {"provider": "jev", "model": "typesafe/jev-1.13"}, FakeLedger(), bootstrap_samples=100)
    assert result["metrics"]["n_failures"] == 1
    prediction = json.loads(next(tmp_path.glob("*/predictions.jsonl")).read_text())
    assert prediction["label"] is None and prediction["probabilities"] is None
    assert prediction["metadata"]["jev_review_called"] is False
    assert prediction["metadata"]["proposal"]["source_failed"] is True


def test_orphan_reservation_is_never_replayed(tmp_path):
    inner = review.budget.Ledger(tmp_path / "new.jsonl", review.STAGE_CAP, initialize=True)
    inner.reserve(review.jev_route.RESERVE_NANO, {"dataset": "wine", "proposal_key": "astra", "row_id": "one"})
    ledger = review.AnchoredLedger(inner)
    with pytest.raises(review.budget.GuardError, match="Unreconciled"):
        review.validate_reservations(ledger, "wine", "astra", [], {"config": {"budget_guard": {}}})


def test_stop_resume_and_completed_replay_never_pay_twice(tmp_path, monkeypatch, study):
    dataset, source, source_record, proposals, examples = study
    monkeypatch.setattr(review, "OUTPUT", tmp_path / "runs")
    monkeypatch.setattr(review, "verify_prior", lambda root: None)
    monkeypatch.setattr(review, "verify_transport", lambda: None)
    inner = review.budget.Ledger(tmp_path / "new.jsonl", review.STAGE_CAP, initialize=True)
    ledger = review.AnchoredLedger(inner, stop_after=1)
    paid_rows = []

    class FakeProvider:
        def __init__(self, config, ledger, provenance):
            self.ledger, self.provenance = ledger, provenance

        def predict(self, row, labels, prompt):
            ident = self.ledger.reserve(review.jev_route.RESERVE_NANO, {**self.provenance,
                "row_id": row.id, "prompt_sha256": review.hashlib.sha256(prompt.encode()).hexdigest()})
            paid_rows.append(row.id)
            self.ledger.result(ident, {"outcome": "returned"})
            return Prediction(row.id, 0, [0.8, 0.2], metadata={"budget": {
                "ledger_id": self.ledger.identity["ledger_id"], "reservation_id": ident}})

    monkeypatch.setattr(review.jev_route, "GuardedOpenRouterJev", FakeProvider)
    config = {"provider": "jev", "model": "typesafe/jev-1.13"}
    args = dataset, "qwen3_4b", source, source_record, proposals, examples, config, ledger
    with pytest.raises(review.jev_route.StopRequested):
        review.run_review(*args, bootstrap_samples=100)
    assert paid_rows == [dataset.test[0].id]
    assert inner.snapshot()["reservations"] == 1
    ledger.stop_after = None
    result = review.run_review(*args, bootstrap_samples=100)
    assert result["status"] == "complete"
    assert paid_rows == [row.id for row in dataset.test]
    assert inner.snapshot()["reservations"] == 2
    replay = review.run_review(*args, bootstrap_samples=100)
    assert replay == result
    assert paid_rows == [row.id for row in dataset.test]
    assert inner.snapshot()["reservations"] == 2
