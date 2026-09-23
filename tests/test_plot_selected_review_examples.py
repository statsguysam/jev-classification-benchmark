"""Synthetic selection/denominator checks; no real collectors or rendering."""
import copy
from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"), str(Path(__file__).resolve().parent)]
import plot_selected_review_examples as figure
from test_plot_review_outcomes import fixture_report


@pytest.fixture
def reports():
    values = [fixture_report(("breast_cancer", "wine")), fixture_report(("sst2", "trec"))]
    for report in values:
        for row in report["runs"]:
            dataset = row["dataset"]
            n, classes = figure.audited.DATASETS[dataset][2:]
            if row["arm"] in ("base", "review"):
                row["train_labels"] = row["train_per_class"] * classes
            elif row["arm"] == "direct":
                shots = int(row["run_id"].rsplit("-", 1)[1])
                row.update(model="typesafe/jev-1.13", train_per_class=shots, train_labels=shots * classes,
                           n_test=n, n_failures=0, accuracy=18 / n)
            else:
                index = int(row["run_id"].rsplit("-", 1)[1])
                model = ("logistic_regression", "random_forest", "xgboost", "lightgbm")[index // 2]
                shots = 4 if index % 2 == 0 else None
                row.update(model=model, train_per_class=shots, train_labels=classes * 4 if shots else 1000,
                           n_test=n, n_failures=0, accuracy=17 / n)
    wine = values[0]
    reviewed = next(row for row in wine["runs"] if row["dataset"] == "wine" and row["arm"] == "review"
                    and row["model"] == "Qwen/Qwen2.5-0.5B-Instruct+jev_review" and row["train_per_class"] == 4)
    direct = next(row for row in wine["runs"] if row["dataset"] == "wine" and row["arm"] == "direct" and row["train_per_class"] == 4)
    wine["comparisons"][-1] = {"kind": "review_minus_jev_alone", "dataset": "wine", "a": reviewed["run_id"],
        "b": direct["run_id"], "status": "complete", "equal_new_label_budget": True}
    return values


def test_values_are_derived_from_audited_selected_identities_not_expected_outcomes(reports):
    wine, trec = figure.prepare(reports)
    assert (wine["dataset"], wine["model"], trec["dataset"], trec["model"]) == (
        "wine", "Qwen/Qwen2.5-0.5B-Instruct", "trec", "gpt-6-astra")
    assert [r["n_correct"] for r in wine["bars"]] == [13, 14, 18, 17]
    assert [r["n_correct"] for r in trec["bars"]] == [13, 14]
    assert wine["training_labels"] == 12 and trec["training_labels"] == 24
    assert wine["n_test"] == 36 and trec["n_test"] == 200
    assert wine["harmed_failure"] == 1 and trec["review_failures_including_upstream_skips"] == 3
    assert all(r["kind"] != "classical" for r in trec["bars"])


@pytest.mark.parametrize("mutation", ["unrelated_pending", "full_training_reference", "denominator", "different_examples", "missing_reference"])
def test_partial_or_unmatched_reference_rejected(reports, mutation):
    classical = next(row for row in reports[0]["runs"] if row["dataset"] == "wine" and row.get("model") == "logistic_regression"
                     and row["train_per_class"] == 4)
    if mutation == "unrelated_pending": reports[1]["runs"][-1]["status"] = "pending"
    elif mutation == "full_training_reference": classical["train_labels"] = 1000
    elif mutation == "denominator": classical["n_test"] -= 1
    elif mutation == "different_examples": reports[0]["comparisons"][-1]["equal_new_label_budget"] = False
    else: classical["model"] = "unplanned"
    with pytest.raises(ValueError): figure.prepare(reports)


def test_match_statement_is_only_used_for_observed_equal_accuracy(reports):
    wine, _ = figure.prepare(reports)
    assert "matched" not in figure.direct_text(wine)
    equal = copy.deepcopy(wine)
    equal["bars"][2].update(n_correct=wine["bars"][1]["n_correct"], accuracy=wine["bars"][1]["accuracy"])
    assert "matched Jev alone's accuracy" in figure.direct_text(equal)


def test_transition_and_interval_annotations_use_measured_inputs(reports):
    wine, _ = figure.prepare(reports)
    assert figure.transition_text(wine) == "4 corrected  ·  2 harmed by a wrong label  ·  1 harmed by failure"
    assert "+2.8 pp" in figure.net_text(wine)
    assert "-5.6 to +11.1" in figure.net_text(wine)


def test_no_render_if_fresh_audit_rejects(tmp_path, monkeypatch):
    def reject():
        raise ValueError("Incomplete report")
    monkeypatch.setattr(figure.audited, "load_audited", reject)
    monkeypatch.setattr(figure, "render", lambda *_: pytest.fail("Rendered incomplete evidence"))
    with pytest.raises(ValueError, match="Incomplete"):
        figure.main(["--output-dir", str(tmp_path / "figure")])
    assert not (tmp_path / "figure").exists()
