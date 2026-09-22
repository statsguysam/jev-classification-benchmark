import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import summarize_review_controls as summary
from jevbench.types import Prediction, PreparedDataset, Row


@pytest.fixture
def study():
    datasets, requests, descriptors = {}, [], {}
    for name in summary.preparation.DATASETS:
        rows = [Row(f"{name}-{i}", f"input {i // 2}", i % 2) for i in range(16)]
        datasets[name] = PreparedDataset(name, ["a", "b"], [], [], rows)
        descriptors[name] = {"selected_row_ids": [row.id for row in rows], "n_classes": 2,
                             "training_example_ids": ["training-a", "training-b"]}
        for arm in (*summary.preparation.PRIMARY_ARMS, summary.preparation.REPEAT_ARM):
            for i, row in enumerate(rows):
                proposal = "not provided" if arm.startswith("no_proposal") else i % 2
                prompt = f"input {i}; proposal={proposal}"
                requests.append({"request_id": f"{name}:{arm}:{i}", "case_id": f"{name}:case:{i}",
                    "dataset": name, "row_id": row.id, "arm": arm, "proposal_value": proposal,
                    "donor_row_id": None, "execution_index": len(requests), "choices": ["a", "b"],
                    "prompt": prompt, "prompt_sha256": summary.hashlib.sha256(prompt.encode()).hexdigest(),
                    "repeat_of_request_id": f"{name}:no_proposal:{i}" if arm == summary.preparation.REPEAT_ARM else None})
    plan = {"protocol": {"study": "fixture", "study_role": "primary_full_fixed_holdouts",
        "datasets": descriptors, "primary_arms": list(summary.preparation.PRIMARY_ARMS),
        "bootstrap_samples": 100, "bootstrap_seed": 42, "repeat_request_count": 64}, "requests": requests}
    predictions = []
    for request in requests:
        index = int(request["row_id"].rsplit("-", 1)[-1])
        truth = index % 2
        label = truth if request["arm"] == "actual" else 0 if request["arm"].startswith("no_proposal") else 1 - truth
        predictions.append(Prediction(request["row_id"], label,
            probabilities=[.9, .1] if label == 0 else [.1, .9],
            metadata={"resolved_model": "fixture-model", "control": {key: request[key] for key in (
                "request_id", "case_id", "dataset", "arm", "proposal_value", "donor_row_id", "execution_index", "prompt_sha256")}}))
    return plan, datasets, predictions


def test_pending_inventory_has_no_performance_metrics(study):
    plan, datasets, _ = study
    report = summary.assemble(plan, datasets, [])
    assert report["complete_primary_arms"] == report["complete_primary_contrasts"] == 0
    assert report["expected_primary_arms"] == 12 and report["expected_primary_contrasts"] == 8
    assert all(row["metrics"] is row["group_bootstrap"] is row["failures"] is None for row in report["runs"])
    assert report["serving_repeat_diagnostic"]["expected_pairs"] == 64
    assert report["serving_repeat_diagnostic"]["counts"] is None
    assert "0/12" in summary.findings(report)


def test_partial_arm_is_not_scored_but_completed_arm_is(study):
    plan, datasets, predictions = study
    partial = summary.assemble(plan, datasets, predictions[:15])
    assert partial["complete_primary_arms"] == 0
    assert partial["runs"][0]["saved_requests"] == 15 and partial["runs"][0]["metrics"] is None
    complete = summary.assemble(plan, datasets, predictions[:17])
    assert complete["complete_primary_arms"] == 1
    assert complete["runs"][0]["metrics"]["accuracy"] == 1.
    assert complete["runs"][1]["metrics"] is None
    assert all(c["paired_group_bootstrap"] is None for c in complete["comparisons"])


def test_complete_contrasts_recompute_paired_accuracy_f1_and_group_count(study):
    plan, datasets, predictions = study
    report = summary.assemble(plan, datasets, predictions)
    assert report["complete_primary_arms"] == 12 and report["complete_primary_contrasts"] == 8
    for contrast in report["comparisons"]:
        expected = .5 if contrast["b"] == "no_proposal" else 1.
        bootstrap = contrast["paired_group_bootstrap"]
        assert bootstrap["metrics"]["accuracy"]["estimate"] == expected
        assert bootstrap["n_groups"] == 8  # Duplicate texts stay in the same group.
        assert bootstrap["n_rows"] == 16
        assert contrast["transitions_B_to_A"]["net_correct_change"] == int(16 * expected)
        assert contrast["balanced_accuracy_delta"] == expected
    repeat = report["serving_repeat_diagnostic"]
    assert repeat["counts"]["valid_label_agreement"] == repeat["counts"]["both_valid"] == 64


def test_repeat_metrics_require_all_64_pairs_even_if_63_are_available(study):
    plan, datasets, predictions = study
    report = summary.assemble(plan, datasets, predictions[:-1])
    repeated = report["serving_repeat_diagnostic"]
    assert repeated["available_complete_pairs"] == 63
    assert repeated["status"] == "pending" and repeated["counts"] is None and repeated["by_dataset"] is None
    assert report["complete_primary_arms"] == 12


def test_repeat_failure_and_label_disagreement_are_distinct(study):
    plan, datasets, predictions = study
    by_id = {request["request_id"]: p for request, p in zip(plan["requests"], predictions)}
    for request_id in ("breast_cancer:no_proposal:0", "breast_cancer:no_proposal_repeat:1",
                       "breast_cancer:no_proposal:2", "breast_cancer:no_proposal_repeat:2"):
        p = by_id[request_id]
        p.error, p.label, p.probabilities = "unknown network outcome", None, None
    p = by_id["breast_cancer:no_proposal_repeat:3"]
    p.label, p.probabilities = 1, [.1, .9]
    report = summary.assemble(plan, datasets, predictions)
    counts = report["serving_repeat_diagnostic"]["counts"]
    assert counts["both_valid"] == 61
    assert counts["valid_label_disagreement"] == 1 and counts["valid_label_agreement"] == 60
    assert counts["both_failed"] == counts["reference_failed_only"] == counts["repeat_failed_only"] == 1
    assert len(report["saved_failure_records"]) == 4
    assert all(p["reported_cost_usd"] is None for p in report["saved_failure_records"])
    no_proposal = next(row for row in report["runs"] if row["dataset"] == "breast_cancer" and row["arm"] == "no_proposal")
    assert no_proposal["metrics"]["n_failures"] == 2
    assert no_proposal["metrics"]["accuracy"] == 6 / 16


def test_unknown_or_wrong_request_identity_is_rejected(study):
    plan, datasets, predictions = study
    predictions[0].metadata["control"]["arm"] = "shuffled"
    with pytest.raises(summary.execution.budget.GuardError, match="identity differs"):
        summary.assemble(plan, datasets, predictions)


def test_repeat_prompt_mismatch_is_rejected(study):
    plan, datasets, predictions = study
    repeated = next(r for r in plan["requests"] if r["arm"] == summary.preparation.REPEAT_ARM)
    repeated["prompt"] = "different wording"
    with pytest.raises(ValueError, match="identical no-proposal"):
        summary.assemble(plan, datasets, predictions)


def test_balanced_accuracy_is_unavailable_when_a_declared_class_is_absent():
    rows = [Row("a", "a", 0), Row("b", "b", 0)]
    predictions = [Prediction("a", 0), Prediction("b", 0)]
    m = summary.score_complete(rows, predictions, 2)
    assert m["accuracy"] == 1 and m["macro_f1"] == .5
    assert m["balanced_accuracy"] is None and m["balanced_accuracy_missing_classes"] == [1]


def test_orphan_execution_evidence_is_not_reported_as_not_started(tmp_path, monkeypatch):
    monkeypatch.setattr(summary.execution, "OUTPUT", tmp_path)
    (tmp_path / "predictions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Orphan execution"):
        summary.load_execution({})


def test_real_pending_report_is_offline_and_has_no_execution_mutation(monkeypatch):
    import socket
    if not (summary.PREPARED / "protocol.json").exists():
        pytest.skip("Frozen plan not yet available")
    if summary.execution.OUTPUT.exists():
        pytest.skip("Execution exists; pending-only integration test no longer applies")
    protocol = summary.read(summary.PREPARED / "protocol.json")
    if protocol.get("repeat_request_count") != 64:
        pytest.skip("Updated repeat protocol not yet frozen")
    def no_connect(*args, **kwargs):
        raise AssertionError("Control report attempted network access")
    monkeypatch.setattr(socket.socket, "connect", no_connect)
    before = {path: path.read_bytes() for path in summary.PREPARED.iterdir()}
    report = summary.collect()
    assert report["saved_requests"] == report["complete_primary_arms"] == 0
    assert report["expected_requests"] == 1714
    assert not summary.execution.OUTPUT.exists()
    assert before == {path: path.read_bytes() for path in before}
