"""Semantic checks for the public metrics asset; no models/APIs are executed."""
import copy
from decimal import Decimal
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dashboard_data", ROOT / "scripts/build_dashboard_data.py")
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


@pytest.fixture(scope="module")
def asset():
    return dashboard.build(ROOT)


def sample_raw(dataset="sst2", model="logistic_regression", method="classical"):
    for p in dashboard.study_run_paths(ROOT):
        record = json.loads(p.read_text())
        if (record["dataset"], record["config"]["model"], record["method"]) == (dataset, model, method):
            return p, record
    raise AssertionError("Fixture not found")


def test_every_measured_execution_retained_and_totals(asset):
    paths = {p.relative_to(ROOT).as_posix() for p in dashboard.study_run_paths(ROOT)}
    assert {r["id"] for r in asset["runs"]} == paths
    assert len(asset["runs"]) == len(paths) == 318
    assert asset["summary"]["prediction_records"] == 59464
    assert asset["summary"]["failures"] == 9
    assert asset["summary"]["by_study"] == {"tabular": 66, "text": 252}
    assert asset["summary"]["canonical_configurations"] == 290
    assert len({r["run_id"] for r in asset["runs"]}) < len(paths)  # run_id is not a UI key


def test_corrected_numeric_study_is_not_silently_added_to_historical_dashboard(tmp_path):
    historical = tmp_path / "results/tabular/condition/run.json"
    historical.parent.mkdir(parents=True)
    historical.write_text("{}")
    corrected = tmp_path / "results/numeric_decisions/review/condition/run.json"
    corrected.parent.mkdir(parents=True)
    corrected.write_text("{}")
    assert dashboard.study_run_paths(tmp_path) == [historical]


def test_repetitions_preserve_evidence_and_have_one_score_independent_canonical(asset):
    groups = {}
    for row in asset["runs"]:
        groups.setdefault(row["duplicate_group_id"], []).append(row)
    assert sum(len(g) - 1 for g in groups.values()) == 28
    for rows in groups.values():
        assert sum(r["canonical"] for r in rows) == 1
        canonical = next(r for r in rows if r["canonical"])
        assert all(r["canonical_id"] == canonical["id"] for r in rows)
        assert all(r["repetition_count"] == len(rows) for r in rows)
        if len(rows) > 1:
            assert canonical["source_path"].startswith("results/pilot/")
    synthetic = [{"id": "results/colab/z/run.json", "duplicate_group_id": "same", "accuracy": 1.0},
                 {"id": "results/pilot/a/run.json", "duplicate_group_id": "same", "accuracy": 0.0}]
    dashboard.mark_repetitions(synthetic)
    assert synthetic[1]["canonical"] and not synthetic[0]["canonical"]


def test_equivalence_changes_when_inputs_or_recipe_change():
    p, raw = sample_raw()
    test = json.loads((p.parent / "test_manifest.json").read_text())
    row = dashboard.normalize_run(raw, p.relative_to(ROOT).as_posix(), {})
    baseline = dashboard.equivalence_fingerprint(raw, row, test)
    raw_changed = copy.deepcopy(raw)
    raw_changed["training_example_ids"][0] = "other-training-row"
    assert dashboard.equivalence_fingerprint(raw_changed, row, test) != baseline
    raw_changed = copy.deepcopy(raw)
    raw_changed["training"]["selected_parameters"]["C"] = 20
    assert dashboard.equivalence_fingerprint(raw_changed, row, test) != baseline
    changed_test = copy.deepcopy(test)
    changed_test["rows"][0]["text_sha256"] = "different-input"
    assert dashboard.equivalence_fingerprint(raw, row, changed_test) != baseline
    changed_row = dict(row, train_per_class=8)
    assert dashboard.equivalence_fingerprint(raw, changed_row, test) != baseline


def test_titanic_is_mixed_not_numeric_and_task_sizes_are_correct(asset):
    datasets = {d["id"]: d for d in asset["datasets"]}
    assert datasets["titanic"]["data_type"] == "mixed"
    assert datasets["titanic"]["task"] == "binary"
    assert datasets["titanic"]["n_test"] == 262
    assert datasets["titanic"]["test_feature_groups"] == 218
    assert datasets["breast_cancer"]["data_type"] == "numeric"
    assert datasets["wine"]["data_type"] == "numeric"
    assert datasets["wine"]["task"] == "multiclass"
    assert datasets["wine"]["n_classes"] == 3
    assert datasets["banking77"]["n_classes"] == 77
    assert all(r["data_type"] == "mixed" for r in asset["runs"] if r["dataset"] == "titanic")


def test_tabular_audited_group_intervals_replace_raw_row_intervals(asset):
    audits = {r["source_path"]: r for r in json.loads((ROOT / "results/TABULAR_COMPARISON.json").read_text())["runs"]}
    changed = 0
    for row in asset["runs"]:
        if row["study"] != "tabular":
            assert row["ci_samples"] == 1000
            assert row["ci_method"] == "stratified percentile bootstrap"
            continue
        audited = audits[row["source_path"]]["group_bootstrap"]
        assert row["ci_samples"] == 2000
        assert row["ci_n_groups"] == audited["n_groups"]
        assert row["ci_method"] == "unstratified group percentile bootstrap"
        assert row["ci_source_path"] == "results/TABULAR_COMPARISON.json"
        raw = json.loads((ROOT / row["source_path"]).read_text())
        for metric in ("accuracy", "macro_f1"):
            assert row["ci"][metric] == audited["metrics"][metric]["ci95"]
            changed += row["ci"][metric] != raw["metrics"]["bootstrap"]["metrics"][metric]["ci95"]
    assert changed > 0
    assert all(r["ci"]["balanced_accuracy"] is None for r in asset["runs"])


def test_missing_values_remain_null_and_zero_probability_coverage_is_real(asset):
    hosted = [r for r in asset["runs"] if r["family"] == "hosted"]
    assert hosted and all(r["probability_coverage"] == 0 for r in hosted)
    assert all(r["probability_metrics"]["log_loss"] is None for r in hosted)
    assert all(r["probability_metrics"]["ece_15_equal_width"] is None for r in hosted)
    assert dashboard.confidence({})["ci"] == {"accuracy": None, "macro_f1": None, "balanced_accuracy": None}
    assert dashboard.finite(None) is None
    assert dashboard.finite(0) == 0
    with pytest.raises(ValueError, match="finite"):
        dashboard.finite(float("nan"))


def test_ci_need_not_contain_the_point_estimate_but_must_match_it():
    # Percentile intervals do not mathematically have to bracket the estimate.
    m = {"accuracy": 0.3, "bootstrap": {"method": "fixture", "samples": 100,
         "metrics": {"accuracy": {"estimate": 0.3, "ci95": [0.4, 0.7]}}}}
    assert dashboard.confidence(m)["ci"]["accuracy"] == [0.4, 0.7]
    m["bootstrap"]["metrics"]["accuracy"]["estimate"] = 0.8
    with pytest.raises(ValueError, match="point estimate"):
        dashboard.confidence(m)


def test_failures_and_detail_confusion_are_not_silently_filtered(asset):
    failures = 0
    for row in asset["runs"]:
        matrix = row["confusion_matrix"]
        assert len(matrix) == row["n_classes"]
        assert all(len(line) == row["n_classes"] + 1 for line in matrix)
        assert row["confusion_matrix_columns"][-1] == "failure"
        assert sum(map(sum, matrix)) == row["n_test"]
        assert sum(line[-1] for line in matrix) == row["n_failures"]
        assert sum(p["support"] for p in row["per_class"]) == row["n_test"]
        assert row["accuracy"] == pytest.approx(sum(matrix[i][i] for i in range(row["n_classes"])) / row["n_test"])
        assert row["n_probability_rows"] / row["n_test"] == pytest.approx(row["probability_coverage"])
        failures += row["n_failures"]
    assert failures == 9
    assert any(0 < r["probability_coverage"] < 1 for r in asset["runs"] if r["family"] == "jev")


def test_label_budgets_and_quantized_training_are_distinct(asset):
    for row in asset["runs"]:
        if row["method"] == "zero_shot":
            assert row["train_per_class"] == row["train_labels"] == 0
            assert row["label_budget"] == "zero"
        elif row["label_budget"] != "full":
            assert row["train_labels"] == row["n_classes"] * row["train_per_class"]
        if row["method"] == "qlora":
            assert row["training_quantized_4bit"] is True
            assert row["hardware"]["inference_dtypes"] == ["float16"]
            assert row["hardware"]["accelerator"] == "Tesla T4"
        elif row["method"] == "lora":
            assert row["training_quantized_4bit"] is False
    full = [r for r in asset["runs"] if r["label_budget"] == "full"]
    assert all(r["train_per_class"] is None for r in full)
    assert any(r["dev_labels"] > 0 for r in full)
    assert all(r["dev_labels"] == 0 for r in full if r["study"] == "tabular")


def test_sanitization_and_source_links_do_not_expose_raw_config_or_rows(asset):
    serialized = json.dumps(asset)
    for forbidden in ("adapter_path", "training_row_ids", "row_id", "api_key", "budget_guard", '"prompt":', "/Users/", "/content/"):
        assert forbidden not in serialized
    dashboard.assert_public(asset)
    for row in asset["runs"]:
        assert "text" not in row and "prompt" not in row and "config" not in row
        assert (ROOT / row["source_path"]).is_file()
        assert row["source_url"].startswith(dashboard.REPO_URL + "/blob/main/results/")
    with pytest.raises(ValueError, match="inside"):
        dashboard.source_url("../../secret")
    with pytest.raises(ValueError, match="forbidden"):
        dashboard.assert_public({"arbitrary": "/Users/private/file"})
    with pytest.raises(ValueError, match="forbidden"):
        dashboard.assert_public({"arbitrary": "sk-proj-" + "x" * 32})


def test_unapproved_config_fields_are_not_exported():
    p, raw = sample_raw()
    raw["config"]["private_unknown_key"] = "sk-proj-" + "x" * 32
    row = dashboard.normalize_run(raw, p.relative_to(ROOT).as_posix(), {})
    dashboard.assert_public(row)
    assert "private_unknown_key" not in json.dumps(row)


def test_cost_scope_is_aggregate_not_invented_per_run(asset):
    costs = asset["costs"]
    assert costs["authorized_usd"] == "20.00"
    assert costs["conservative_accounted_usd"] == "18.522115700"
    assert costs["compute_cost_usd"] is None
    assert Decimal(costs["authorized_usd"]) - Decimal(costs["conservative_accounted_usd"]) == Decimal(costs["remaining_capacity_usd"])
    assert sum(Decimal(x["accounted_conservative_usd"]) for x in costs["stages"]) == Decimal(costs["conservative_accounted_usd"])
    assert "incomplete coverage" in costs["known_mixed_cost_subtotal_basis"]
    assert all(not any("cost" in key for key in r) for r in asset["runs"])


def test_classification_scores_cannot_diverge_from_saved_predictions():
    p, raw = sample_raw()
    test = json.loads((p.parent / "test_manifest.json").read_text())
    dashboard.audit_prediction_aggregates(p.parent / "predictions.jsonl", raw, test)
    raw["metrics"]["accuracy"] = 0.012345
    with pytest.raises(ValueError, match="Classification metric mismatch"):
        dashboard.audit_prediction_aggregates(p.parent / "predictions.jsonl", raw, test)


def test_stale_audited_tabular_artifacts_are_rejected(tmp_path):
    real_summary = json.loads((ROOT / "results/TABULAR_COMPARISON.json").read_text())
    audited = real_summary["runs"][0]
    raw_path = ROOT / audited["source_path"]
    dest = tmp_path / audited["source_path"]
    dest.parent.mkdir(parents=True)
    mutated = json.loads(raw_path.read_text())
    mutated["metrics"]["macro_f1"] = 0.0001
    dest.write_text(json.dumps(mutated))
    (tmp_path / "results/TABULAR_COMPARISON.json").write_text(json.dumps({"runs": [audited]}))
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        dashboard.build(tmp_path)
