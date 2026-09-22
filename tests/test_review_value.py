import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import analyze_review_value as analysis
from jevbench.types import Prediction


def prediction(row, label, probabilities, error=None):
    return Prediction(row_id=row, label=label, probabilities=probabilities, error=error,
                      metadata={"probability_kind": "label_sequence_likelihood_normalized"})


def test_failures_remain_in_all_metric_denominators_and_joint_cells():
    measured = analysis.scores([0, 0, 1, 1], [0, -1, 0, 1], 2)
    assert measured == {"micro_accuracy": .5, "balanced_accuracy": .5, "macro_f1": pytest.approx(7 / 12),
                        "n_rows": 4, "n_failures": 1}
    joint = analysis.joint_correctness([0, 1], [0, -1], [1, 1], [-1, 1])
    assert joint["counts"]["100"] == joint["counts"]["011"] == 1
    assert sum(joint["counts"].values()) == 2


def test_gate_rejects_constant_labels_hosted_probabilities_and_partial_reviews():
    values = [prediction("a", 0, [.7, .3]), prediction("b", 0, [.8, .2])]
    assert "constant_or_missing_source_labels" in analysis.eligibility({"config": {"provider": "hf"}}, values, [0, 0])["exclusion_reasons"]
    out = analysis.eligibility({"config": {"provider": "openai"}}, values, [0, 1], False)
    assert not out["eligible"]
    assert "source_is_not_a_local_hf_model" in out["exclusion_reasons"]
    assert "review_condition_incomplete" in out["exclusion_reasons"]
    values[1].probabilities = None
    assert "source_probability_scores_unavailable_or_ineligible" in analysis.eligibility({"config": {"provider": "hf"}}, values, [0, 1])["exclusion_reasons"]


def test_fixed_ranking_is_truth_blind_and_ties_follow_row_hash():
    rows = ["r2", "r1", "r3", "r4"]
    predictions = [prediction(rows[0], 0, [.8, .2]), prediction(rows[1], 1, [.2, .8]),
                   prediction(rows[2], 0, [.55, .45]), prediction(rows[3], 1, [.1, .9])]
    rank, _ = analysis.rank_rows(predictions, rows)
    assert rank[0] == 2 and rank[-1] == 3
    assert rank[1:3] == sorted([0, 1], key=lambda i: analysis.hashlib.sha256(rows[i].encode()).hexdigest())
    a = analysis.selective_curve([0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, -1], predictions, rows, 2)
    b = analysis.selective_curve([1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, -1], predictions, rows, 2)
    assert [p["selected_row_ids"] for p in a["points"]] == [p["selected_row_ids"] for p in b["points"]]
    assert [p["requested_review_percent"] for p in a["points"]] == [0, 10, 25, 50, 75, 100]
    assert a["points"][0]["metrics"] == analysis.scores([0, 0, 1, 1], [0, 1, 0, 1], 2)
    assert a["points"][-1]["metrics"] == analysis.scores([0, 0, 1, 1], [1, 0, 1, -1], 2)
    for p in a["points"]:
        assert p["required_inferences"]["review_api_requests"] == len(p["selected_row_ids"])
        assert p["required_inferences"]["source"] == 4
        t = p["transitions_from_never_review"]
        assert t["net_correct_change"] == t["wrong_to_correct"] - t["correct_to_wrong"]


def test_random_expectations_match_exhaustive_small_case_and_do_not_use_f1_of_expected_counts():
    y, base, review, m = [0, 0, 0, 1], [0, 0, 1, 1], [1, -1, 0, 0], 2
    exact = []
    for subset in itertools.combinations(range(4), m):
        pred = [review[i] if i in subset else base[i] for i in range(4)]
        exact.append(analysis.scores(y, pred, 2))
    actual = analysis.random_expectation(y, base, review, 2, m, samples=30000)
    for name in ("micro_accuracy", "balanced_accuracy"):
        assert actual[name] == pytest.approx(np.mean([r[name] for r in exact]))
    assert actual["macro_f1"] == pytest.approx(np.mean([r["macro_f1"] for r in exact]), abs=.005)
    assert actual["macro_f1_monte_carlo_se"] > 0
    for endpoint, predictions in ((0, base), (4, review)):
        out = analysis.random_expectation(y, base, review, 2, endpoint)
        assert out["macro_f1"] == analysis.scores(y, predictions, 2)["macro_f1"]
        assert out["samples"] == 0


def test_same_label_equivalence_requires_matching_prompt_and_separates_failures():
    a = {"dataset": "wine", "shots_per_class": 4, "run_id": "a", "row_ids": ["x", "y", "z"],
         "y": [0, 1, 1], "base": [0, 1, 1], "review": [0, 1, -1], "prompt_hashes": ["p", "q", "r"]}
    b = {**a, "run_id": "b", "review": [1, 1, 1]}
    out = analysis.same_label_pairs([a, b])[0]
    assert out["same_label_identical_prompt_rows"] == 3
    assert out["both_valid_rows"] == 2 and out["valid_output_disagreements"] == 1
    assert out["rows_with_either_review_failure"] == 1
    b["prompt_hashes"] = ["changed", "q", "r"]
    with pytest.raises(ValueError, match="prompts differ"):
        analysis.same_label_pairs([a, b])


def test_bad_probability_or_duplicate_ids_fail_closed():
    with pytest.raises(ValueError, match="Invalid source probabilities"):
        analysis.rank_rows([prediction("a", 0, [float("nan"), .2])], ["a"])
    with pytest.raises(ValueError, match="Row IDs differ"):
        analysis.rank_rows([prediction("a", 0, [.5, .5])] * 2, ["a", "a"])


def test_fixed_coverage_rounding():
    assert [analysis.review_count(36, rate) for rate in analysis.RATES] == [0, 4, 9, 18, 27, 36]
    assert [analysis.review_count(114, rate) for rate in analysis.RATES] == [0, 11, 29, 57, 86, 114]


def test_real_collection_is_offline_complete_only_and_preserves_existing_artifacts(monkeypatch):
    import socket
    root = analysis.ROOT
    if not (root / "results/numeric_expansion/COMPARISON.json").exists():
        pytest.skip("Local result archive unavailable")
    def no_connect(*args, **kwargs):
        raise AssertionError("Offline analysis attempted network access")
    monkeypatch.setattr(socket.socket, "connect", no_connect)
    prior = analysis.source.read(root / "results/numeric_expansion/COMPARISON.json")
    paths = set()
    for row in prior["runs"]:
        if row.get("source_path"):
            paths.update((root / row["source_path"]).with_name(name) for name in analysis.FILES
                         if (root / row["source_path"]).with_name(name).exists())
    paths.update(root.glob("results/**/*budget*.jsonl*"))
    paths.update(root / name for name in ("results/TABULAR_COMPARISON.json",
        "results/numeric_decisions/COMPARISON.json", "results/numeric_expansion/COMPARISON.json"))
    before = {str(p): analysis.source.file_sha(p) for p in paths if p.is_file()}
    report = analysis.collect(root)
    assert report["complete_review_conditions"] == prior["complete_review_runs"]
    assert report["scored_partial_conditions"] == 0
    assert all(r["metrics"] is None for r in report["excluded_conditions"])
    assert len(report["conditions"]) + len(report["excluded_conditions"]) == 24
    assert before == {p: analysis.source.file_sha(p) for p in before}
    for c in report["conditions"]:
        assert sum(c["joint_correctness"]["counts"].values()) == c["n_rows"]
        for arm, row_key in (("never_review", "source_run_id"), ("always_review", "review_run_id"), ("direct_jev", "direct_jev_run_id")):
            artifact = analysis.source.read(root / report["artifacts"][c[row_key]]["path"])
            assert c["metrics"][arm]["micro_accuracy"] == artifact["metrics"]["accuracy"]
            assert c["metrics"][arm]["balanced_accuracy"] == pytest.approx(artifact["metrics"]["balanced_accuracy"])
            assert c["metrics"][arm]["macro_f1"] == pytest.approx(artifact["metrics"]["macro_f1"])
