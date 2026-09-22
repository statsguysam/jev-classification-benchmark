"""Synthetic outcomes only; no model, tokenizer, cache, or active-run reads."""
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_validation_scoring_ablation as auditor
from jevbench.types import PreparedDataset, Row


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@pytest.fixture
def complete(tmp_path, monkeypatch):
    # Reuse the immutable, score-free context schema; every outcome below is fake.
    protocol = json.loads((auditor.OUTPUT / "protocol.json").read_text())
    config = protocol["config"]
    presets = json.loads(auditor.producer.frozen.PRESETS.read_text())
    datasets, samples, examples = {}, {}, {}
    for name, spec in config["datasets"].items():
        nclasses = 2 if name == "breast_cancer" else 3
        rows = [Row(row_id, f"synthetic validation {name} {i}", i % nclasses)
                for i, row_id in enumerate(spec["validation_ids"])]
        train = [Row(row_id, f"synthetic train {name} {i}", i % nclasses)
                 for i, row_id in enumerate(spec["training_example_ids_k4"])]
        datasets[name] = PreparedDataset(name, [str(i) for i in range(nclasses)], train, rows, [])
        samples[name], examples[name] = rows, train
    protocol["validation_content_sha256"] = {
        name: auditor.producer.digest([auditor.asdict(r) for r in rows]) for name, rows in samples.items()}
    protocol["all_validation_ids_sha256"] = {
        name: auditor.producer.digest([r.id for r in ds.validation]) for name, ds in datasets.items()}
    monkeypatch.setattr(auditor.producer, "load_inputs", lambda: (config, presets, datasets, samples, examples))
    save(tmp_path / "protocol.json", protocol)
    protocol_sha = auditor.sha((tmp_path / "protocol.json").read_bytes())
    monkeypatch.setattr(auditor, "PROTOCOL_SHA", protocol_sha)
    values = []
    for i, context in enumerate(protocol["contexts"]):
        nclasses = len(context["candidate_token_ids"])
        label = [-2.0 if c == i % nclasses else -3.0 for c in range(nclasses)]
        eos = [-4.0 if c == i % nclasses and i % 3 == 0 else -0.5 for c in range(nclasses)]
        joint = [a + b for a, b in zip(label, eos)]
        value = {"context": context, "protocol_sha256": protocol_sha, "label_logp": label,
            "conditional_eos_logp": eos, "joint_logp": joint,
            "per_candidate_token_logp": [list(pair) for pair in zip(label, eos)],
            "context_forward_passes": nclasses}
        for arm, scores in (("label_only", label), ("label_plus_eos", joint)):
            probabilities = auditor.producer.normalize(scores)
            value[arm] = {"label": max(range(nclasses), key=probabilities.__getitem__), "probabilities": probabilities}
        values.append(value)
    def write_predictions(new_values):
        (tmp_path / "predictions.jsonl").write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in new_values))
        pred_sha = auditor.sha((tmp_path / "predictions.jsonl").read_bytes())
        run = {"status": "complete", "n_completed": 136, "n_expected": 136, "device": "mps",
               "dtype": "float16", "torch_version": protocol["packages"]["torch"],
               "protocol_sha256": protocol_sha, "prediction_sha256": pred_sha}
        save(tmp_path / "run.json", run)
        return pred_sha
    pred_sha = write_predictions(values)
    raw = {"status": "complete", "protocol_sha256": protocol_sha, "n_contexts": 136,
        "prediction_sha256": pred_sha, "conditions": auditor.producer.summarize(values, datasets, samples),
        "test_results_changed": False, "method_selected": False, "confidence_gate_fitted": False}
    save(tmp_path / "SUMMARY.json", raw)
    (tmp_path / "FINDINGS.md").write_text("Synthetic unchanged producer report.\n")
    return tmp_path, values, raw, write_predictions


def test_complete_projection_only_exports_measured_metrics(complete):
    output, _, raw, _ = complete
    result = auditor.audit(output)
    assert result["n_contexts"] == 136 and result["n_distinct_validation_rows"] == 34
    assert result["latency_measurement_status"] == "not_measured"
    assert len(result["conditions"]) == 8
    for condition, original in zip(result["conditions"], raw["conditions"]):
        for arm in auditor.ARMS:
            assert set(condition["metrics"][arm]) == set(auditor.METRIC_FIELDS)
            assert condition["metrics"][arm]["n_rows"] == condition["n_validation"]
            assert original["metrics"][arm]["prediction_time_s"] == 0
        assert condition["predicted_class_counts"] == original["predicted_class_counts"]
    encoded = json.dumps(result)
    assert "latency_p50" not in encoded and "latency_p95" not in encoded and "prediction_time_s" not in encoded
    assert result["provenance"]["raw_summary_sha256"] == auditor.sha((output / "SUMMARY.json").read_bytes())


def test_offline_audit_does_not_require_inference_runtime_or_cache(complete, monkeypatch):
    output, _, _, _ = complete
    monkeypatch.setattr(importlib.metadata, "version", lambda *_: pytest.fail("Queried local inference versions"))
    monkeypatch.setattr(auditor.producer, "build_protocol", lambda *_: pytest.fail("Loaded tokenizer preflight"))
    monkeypatch.setattr(auditor.producer.frozen, "FixedChatClassifier", lambda *_: pytest.fail("Loaded model"))
    assert auditor.audit(output)["status"] == "complete"


def test_incomplete_run_rejected_before_reading_scores(complete, monkeypatch):
    output, _, _, _ = complete
    run = json.loads((output / "run.json").read_text())
    run.update(status="running", n_completed=135)
    save(output / "run.json", run)
    monkeypatch.setattr(auditor.producer, "read_checkpoint", lambda *_: pytest.fail("Read incomplete scores"))
    with pytest.raises(ValueError, match="incomplete"):
        auditor.audit(output)
    assert not (output / "AUDITED_SUMMARY.json").exists()


def test_complete_flag_cannot_hide_missing_prediction(complete):
    output, values, _, write = complete
    write(values[:-1])
    with pytest.raises(ValueError, match="136 ordered"):
        auditor.audit(output)


def test_reordered_or_duplicate_context_is_rejected(complete):
    output, values, _, write = complete
    values[1] = deepcopy(values[0])
    write(values)
    with pytest.raises(ValueError, match="context/protocol"):
        auditor.audit(output)


def test_decomposition_tamper_is_rejected(complete):
    output, values, _, write = complete
    values[0]["joint_logp"][0] -= 0.25
    write(values)
    with pytest.raises(ValueError, match="reconstruct joint"):
        auditor.audit(output)


@pytest.mark.parametrize("field", ["accuracy", "class_counts", "fixed", "harmed"])
def test_raw_summary_must_match_independent_recomputation(complete, field):
    output, _, raw, _ = complete
    condition = raw["conditions"][0]
    if field == "accuracy":
        condition["metrics"]["label_only"]["accuracy"] += 0.01
    elif field == "class_counts":
        condition["predicted_class_counts"]["label_only"][0] += 1
    else:
        key = "label_only_corrected_joint_errors" if field == "fixed" else "label_only_harmed_joint_correct"
        condition[key] += 1
    save(output / "SUMMARY.json", raw)
    with pytest.raises(ValueError, match="Raw SUMMARY differs"):
        auditor.audit(output)


def test_protocol_hash_is_independent_of_run_claim(complete):
    output, _, _, _ = complete
    protocol = json.loads((output / "protocol.json").read_text())
    protocol["contexts"][0]["rendered_input_tokens"] += 1
    save(output / "protocol.json", protocol)
    with pytest.raises(ValueError, match="Frozen protocol hash differs"):
        auditor.audit(output)


def test_source_pin_change_is_rejected(complete, monkeypatch):
    output, _, _, _ = complete
    protocol = json.loads((output / "protocol.json").read_text())
    protocol["source_sha256"]["src/jevbench/metrics.py"] = "0" * 64
    save(output / "protocol.json", protocol)
    monkeypatch.setattr(auditor, "PROTOCOL_SHA", auditor.sha((output / "protocol.json").read_bytes()))
    with pytest.raises(ValueError, match="Source pin differs"):
        auditor.audit(output)


def test_main_preserves_all_raw_artifacts_and_writes_separate_report(complete):
    output, _, _, _ = complete
    originals = {p.name: p.read_bytes() for p in output.iterdir()}
    auditor.main(["--output", str(output)])
    assert all((output / name).read_bytes() == content for name, content in originals.items())
    audited = json.loads((output / "AUDITED_SUMMARY.json").read_text())
    assert audited["status"] == "complete"
    report = (output / "AUDITED_FINDINGS.md").read_text()
    assert "placeholders, not measurements" in report
    assert "34 distinct validation cases" in report
    assert "must not be compared directly with historical test accuracy" in report
