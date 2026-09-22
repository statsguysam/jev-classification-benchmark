"""Expanded review preserves matched shots, immutable sources and paid checkpoints."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_expanded_numeric_review as expanded
from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import digest
from jevbench.types import Prediction, PreparedDataset, Row


@pytest.fixture
def fixture_source(tmp_path, monkeypatch):
    monkeypatch.setattr(expanded, "ROOT", tmp_path)
    monkeypatch.setattr(expanded, "AREA", tmp_path / "results/numeric_expansion")
    monkeypatch.setattr(expanded, "OUTPUT", tmp_path / "results/numeric_expansion/review")
    monkeypatch.setattr(expanded, "LOCAL", tmp_path / "results/numeric_expansion/local")
    monkeypatch.setattr(expanded, "SOURCES_DIR", tmp_path / "results/numeric_expansion/sources")
    dataset = PreparedDataset("breast_cancer", ["negative", "positive"],
        [Row(f"train-{i}", f"feature={i / 10}", i % 2) for i in range(10)],
        [Row("val-one", "feature=99", 0)],
        [Row("SECRET_TEST_A", "feature=2.1", 1), Row("SECRET_TEST_B", "feature=2.2", 0)],
        {"tabular": {"features": [{"name": "feature", "kind": "numeric"}]}})
    source_by_shots = {}

    def create(shots):
        directory = tmp_path / f"results/tabular/local/source-{shots}"
        directory.mkdir(parents=True, exist_ok=True)
        spec = expanded.MODELS["qwen_small"]
        config = {"provider": "hf", "model": spec["model"], "revision": spec["revision"], "shots_per_class": shots}
        predictions = [Prediction(row.id, row.label) for row in dataset.test]
        examples = select_examples(dataset.train, dataset.labels, shots, 42)
        record = {"status": "complete", "dataset": dataset.name, "labels": dataset.labels, "seed": 42,
            "method": "zero_shot" if shots == 0 else "few_shot", "manifest_sha256": digest(dataset.manifest),
            "dataset_manifest": dataset.manifest, "test_ids_sha256": digest([r.id for r in dataset.test]),
            "training_example_ids": [r.id for r in examples], "implementation_sha256": expanded.base.FROZEN_CORE_SHA,
            "run_id": directory.name, "config": config, "metrics": evaluate(dataset.test, predictions, 2)}
        (directory / "run.json").write_text(json.dumps(record))
        (directory / "predictions.jsonl").write_text("".join(json.dumps(asdict(p)) + "\n" for p in predictions))
        (directory / "test_manifest.json").write_text(json.dumps(expanded.base.expected_test_manifest(dataset)))
        source = {"model": spec["model"], "provider": "hf", "path": directory.relative_to(tmp_path).as_posix(),
            "hashes": {name: expanded.file_sha(directory / name) for name in expanded.FILE_NAMES}}
        source_by_shots[shots] = source
        return source

    monkeypatch.setattr(expanded, "historical_source", lambda name, model_key, shots: source_by_shots[shots])
    create(0)
    create(4)
    return dataset, create, source_by_shots


def test_zero_and_four_shot_sources_keep_exact_training_budgets(fixture_source):
    dataset, _, _ = fixture_source
    zero = expanded.load_source(dataset, "qwen_small", 0)
    few = expanded.load_source(dataset, "qwen_small", 4)
    assert zero["examples"] == []
    assert len(few["examples"]) == 8
    prompt = expanded.base.build_review_prompt(dataset.test[0], dataset.labels, zero["examples"], 0)
    assert "Example:" not in prompt
    assert dataset.test[0].id not in prompt
    assert prompt == expanded.base.build_review_prompt(replace(dataset.test[0], label=0), dataset.labels, [], 0)


def test_freezing_is_individual_idempotent_and_never_overwrites(fixture_source):
    dataset, _, sources = fixture_source
    job = expanded.load_source(dataset, "qwen_small", 0, freeze=True)
    path = expanded.ROOT / job["manifest_path"]
    original = path.read_bytes()
    assert expanded.load_source(dataset, "qwen_small", 0, freeze=True)["manifest_sha256"] == job["manifest_sha256"]
    expanded.load_source(dataset, "qwen_small", 4, freeze=True)
    assert path.read_bytes() == original
    changed = json.loads(original)
    changed["shots_per_class"] = 4
    path.write_text(json.dumps(changed))
    with pytest.raises(expanded.budget.GuardError, match="manifest changed"):
        expanded.load_source(dataset, "qwen_small", 0, freeze=True)
    assert json.loads(path.read_text())["shots_per_class"] == 4


def test_paid_identity_does_not_depend_on_later_source_inventory(fixture_source):
    dataset, _, _ = fixture_source
    job = expanded.load_source(dataset, "qwen_small", 0, freeze=True)
    config = {"provider": "jev", "model": expanded.route.MODEL}
    before, guard = expanded.make_identity(job, config, "new-ledger", 100)
    expanded.load_source(dataset, "qwen_small", 4, freeze=True)
    after, _ = expanded.make_identity(job, config, "new-ledger", 100)
    assert before == after
    assert guard["shots_per_class"] == guard["source_shots_per_class"] == 0
    assert before["training_example_ids"] == []


def test_changed_source_files_cannot_be_refrozen(fixture_source):
    dataset, _, sources = fixture_source
    job = expanded.load_source(dataset, "qwen_small", 0, freeze=True)
    path = expanded.ROOT / sources[0]["path"] / "predictions.jsonl"
    path.write_text(path.read_text() + " ")
    with pytest.raises(expanded.budget.GuardError, match="artifact changed"):
        expanded.load_source(dataset, "qwen_small", 0, freeze=True)
    assert expanded.file_sha(expanded.ROOT / job["manifest_path"]) == job["manifest_sha256"]


def test_source_shot_mismatch_rejected_even_if_hashes_updated(fixture_source):
    dataset, _, sources = fixture_source
    source = sources[0]
    path = expanded.ROOT / source["path"] / "run.json"
    record = json.loads(path.read_text())
    record["config"]["shots_per_class"] = 4
    path.write_text(json.dumps(record))
    source["hashes"]["run.json"] = expanded.file_sha(path)
    with pytest.raises(expanded.budget.GuardError, match="label budget differs"):
        expanded.validate_source(dataset, "qwen_small", 0, source)


def test_missing_new_local_models_stay_pending(fixture_source):
    dataset, _, _ = fixture_source
    assert expanded.load_source(dataset, "smollm2", 0, freeze=True) is None
    assert not (expanded.SOURCES_DIR / "breast_cancer__smollm2__k0.json").exists()


def test_existing_review_conditions_are_never_charged_again(fixture_source):
    dataset, _, _ = fixture_source
    assert expanded.is_reused("qwen_main", 4)
    assert expanded.is_reused("astra", 4)
    assert not expanded.is_reused("astra", 0)
    with pytest.raises(expanded.budget.GuardError, match="reused, never charged"):
        expanded.run_review({"dataset": dataset, "model_key": "astra", "shots": 4}, {}, None)


def test_all_prior_ledgers_are_immutable_and_cumulative_cap_checked(tmp_path, monkeypatch):
    prior = {}
    for i in range(4):
        path = tmp_path / f"prior-{i}.jsonl"
        ledger = expanded.budget.Ledger(path, "1.00", initialize=True)
        ledger.reserve(expanded.budget.usd_nano("0.10"), {"prior": i})
        prior[path.name] = {"cap": "1.00", "sha256": expanded.file_sha(path),
            "lock_sha256": expanded.file_sha(path.with_name(path.name + ".lock"))}
    monkeypatch.setattr(expanded, "PRIOR", prior)
    monkeypatch.setattr(expanded, "PRIOR_TOTAL", "0.40")
    assert len(expanded.verify_prior(tmp_path)) == 4
    (tmp_path / "prior-3.jsonl.lock").write_text("changed")
    with pytest.raises(expanded.budget.GuardError, match="ledger/lock changed"):
        expanded.verify_prior(tmp_path)


def test_request_guard_distinguishes_shots_and_limits_global_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(expanded, "verify_prior", lambda root: None)
    monkeypatch.setattr(expanded, "verify_transport", lambda: None)
    inner = expanded.budget.Ledger(tmp_path / "new.jsonl", expanded.STAGE_CAP, initialize=True)
    ledger = expanded.AnchoredLedger(inner)
    details = {"dataset": "wine", "proposal_key": "luna_k0", "row_id": "one"}
    ledger.reserve(expanded.route.RESERVE_NANO, details)
    ledger.reserve(expanded.route.RESERVE_NANO, {**details, "proposal_key": "luna_k4"})
    with pytest.raises(expanded.budget.GuardError, match="already reserved"):
        ledger.reserve(expanded.route.RESERVE_NANO, details)
    monkeypatch.setattr(expanded, "MAX_REQUESTS", 2)
    with pytest.raises(expanded.budget.BudgetStop, match="1500"):
        ledger.reserve(expanded.route.RESERVE_NANO, {**details, "row_id": "next"})


def test_three_errors_halt_across_models_and_shots(tmp_path, monkeypatch):
    monkeypatch.setattr(expanded, "verify_prior", lambda root: None)
    monkeypatch.setattr(expanded, "verify_transport", lambda: None)
    inner = expanded.budget.Ledger(tmp_path / "new.jsonl", expanded.STAGE_CAP, initialize=True)
    for i in range(3):
        ident = inner.reserve(expanded.route.RESERVE_NANO, {"dataset": "wine", "proposal_key": "luna_k0", "row_id": str(i)})
        inner.result(ident, {"outcome": "prediction_error"})
    with pytest.raises(expanded.budget.GuardError, match="Three consecutive"):
        expanded.AnchoredLedger(inner).reserve(expanded.route.RESERVE_NANO,
            {"dataset": "breast_cancer", "proposal_key": "smollm2_k4", "row_id": "new"})


def test_stop_resume_and_completed_replay_do_not_repeat_paid_calls(fixture_source, tmp_path, monkeypatch):
    dataset, _, _ = fixture_source
    job = expanded.load_source(dataset, "qwen_small", 0, freeze=True)
    monkeypatch.setattr(expanded, "verify_prior", lambda root: None)
    monkeypatch.setattr(expanded, "verify_transport", lambda: None)
    inner = expanded.budget.Ledger(tmp_path / "new.jsonl", expanded.STAGE_CAP, initialize=True)
    ledger = expanded.AnchoredLedger(inner, stop_after=1)
    called = []

    class FakeProvider:
        def __init__(self, config, ledger, provenance):
            self.ledger, self.provenance = ledger, provenance

        def predict(self, row, labels, prompt):
            ident = self.ledger.reserve(expanded.route.RESERVE_NANO, {**self.provenance,
                "row_id": row.id, "prompt_sha256": expanded.hashlib.sha256(prompt.encode()).hexdigest()})
            called.append(row.id)
            self.ledger.result(ident, {"outcome": "returned"})
            return Prediction(row.id, 0, [0.9, 0.1], metadata={"budget": {
                "ledger_id": self.ledger.identity["ledger_id"], "reservation_id": ident}})

    monkeypatch.setattr(expanded.route, "GuardedOpenRouterJev", FakeProvider)
    config = {"provider": "jev", "model": expanded.route.MODEL}
    with pytest.raises(expanded.route.StopRequested):
        expanded.run_review(job, config, ledger, bootstrap_samples=100)
    assert called == [dataset.test[0].id]
    # A new unrelated frozen condition must not invalidate this paid checkpoint.
    expanded.load_source(dataset, "qwen_small", 4, freeze=True)
    ledger.stop_after = None
    completed = expanded.run_review(job, config, ledger, bootstrap_samples=100)
    assert completed["status"] == "complete" and inner.snapshot()["reservations"] == 2
    assert called == [row.id for row in dataset.test]
    assert expanded.run_review(job, config, ledger, bootstrap_samples=100) == completed
    assert inner.snapshot()["reservations"] == 2


def test_failed_proposal_is_retained_without_jumping_to_jev(fixture_source, tmp_path, monkeypatch):
    dataset, _, _ = fixture_source
    job = expanded.load_source(dataset, "qwen_small", 0, freeze=True)
    job["dataset"].test = job["dataset"].test[:1]
    job["predictions"] = [Prediction(dataset.test[0].id, None, error="source_failure")]
    monkeypatch.setattr(expanded.base, "validate_reservations", lambda *args: None)

    class NoCalls:
        def __init__(self, *args):
            pass

        def predict(self, *args):
            raise AssertionError("No Jev fallback for failed source")

    class FakeLedger:
        identity = {"ledger_id": "new-ledger"}

    monkeypatch.setattr(expanded.route, "GuardedOpenRouterJev", NoCalls)
    result = expanded.run_review(job, {"provider": "jev", "model": expanded.route.MODEL}, FakeLedger(), bootstrap_samples=100)
    assert result["metrics"]["n_failures"] == 1
    prediction = json.loads(next(expanded.OUTPUT.glob("*/predictions.jsonl")).read_text())
    assert prediction["probabilities"] is None and prediction["label"] is None
    assert prediction["metadata"]["jev_review_called"] is False
