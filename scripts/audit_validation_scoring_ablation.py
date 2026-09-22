#!/usr/bin/env python3
"""Audit the completed frozen scoring diagnostic without models or cache access.

Writes separate AUDITED_SUMMARY.json and AUDITED_FINDINGS.md. The producer's
protocol, predictions, run record, SUMMARY.json and FINDINGS.md are read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation_scoring_ablation as producer

ROOT = producer.ROOT
OUTPUT = producer.OUTPUT
PROTOCOL_SHA = "12f2712810562c3481f90894eb0fe4569a999e619ccf0a59ce4f5eba26182f30"
PRODUCER_SHA = "51a5d69a6e5a7eda7ca9f334fd638d9b72443936ef780ba7494d93db7dafdb97"
PRODUCER_PATH = "scripts/run_validation_scoring_ablation.py"
RENDERER_PATH = "scripts/run_expanded_numeric_local.py"
METRIC_FIELDS = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
                 "macro_f1": "macro_f1", "n_rows": "n_test", "n_failures": "n_failures"}
ARMS = ("label_only", "label_plus_eos")
require = producer.require


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verify_protocol(protocol: dict, protocol_sha: str):
    """Recheck the immutable preflight and its data; never load a tokenizer.

    The independently pinned protocol binds the token IDs and tokenizer hashes
    from the original tokenizer preflight. Data identities and context order are
    reconstructed below; no claim of rerunning tokenization is made.
    """
    require(protocol_sha == PROTOCOL_SHA, "Frozen protocol hash differs")
    sources = protocol["source_sha256"]
    require(sources.get(PRODUCER_PATH) == PRODUCER_SHA, "Frozen producer hash differs")
    require(sources.get(RENDERER_PATH) == producer.HELPER_SHA, "Frozen renderer hash differs")
    for relative, expected in sources.items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe source path")
        require(sha((ROOT / path).read_bytes()) == expected, f"Source pin differs: {relative}")
    # The pinned protocol records the inference environment. A read-only audit
    # does not require that runtime (e.g. MPS torch versus a CPU CI wheel).
    config, presets, datasets, samples, examples = producer.load_inputs()
    require(protocol["config"] == config, "Frozen config differs")
    require(protocol["models"] == {key: producer.frozen.model_config(key, "mps", presets)
            for key in config["model_keys"]}, "Frozen model configuration differs")
    require(protocol["frozen_core_sha256"] == producer.frozen.FROZEN_CORE_SHA,
            "Frozen core identity differs")
    require(protocol["validation_content_sha256"] == {
        name: producer.digest([asdict(row) for row in samples[name]]) for name in datasets},
        "Validation content differs")
    require(protocol["all_validation_ids_sha256"] == {
        name: producer.digest([row.id for row in ds.validation]) for name, ds in datasets.items()},
        "Validation split identity differs")
    expected = [(key, name, shots, row.id) for key in config["model_keys"]
                for name in datasets for shots in config["shots_per_class"] for row in samples[name]]
    contexts = protocol["contexts"]
    require(len(expected) == len(contexts) == protocol["n_model_contexts"] == 136,
            "Diagnostic requires all 136 contexts")
    require(protocol["n_distinct_validation_rows"] == sum(map(len, samples.values())) == 34
            and protocol["n_paired_decisions"] == 272, "Diagnostic row/decision count differs")
    require(protocol["test_predictions_read_or_changed"] is False, "Unexpected test reevaluation")
    for context, identity in zip(contexts, expected):
        require(tuple(context[k] for k in ("model_key", "dataset", "shots_per_class", "row_id")) == identity
                and context["key"] == producer.condition_key(*identity), "Context order or identity differs")
        candidates = context["candidate_token_ids"]
        require(len(candidates) == len(datasets[identity[1]].labels)
                and all(len(c) == 2 and all(type(t) is int and t >= 0 for t in c) for c in candidates)
                and len({c[0] for c in candidates}) == len(candidates)
                and len({c[1] for c in candidates}) == 1, "Candidate token identities differ")
        require(type(context["rendered_input_tokens"]) is int
                and 0 < context["rendered_input_tokens"] <= 8190, "Invalid frozen context length")
        for field in ("rendered_chat_prompt_sha256", "rendered_input_ids_sha256",
                      "chat_template_sha256", "tokenizer_fingerprint_sha256"):
            require(isinstance(context[field], str) and re.fullmatch("[0-9a-f]{64}", context[field]),
                    "Invalid frozen context provenance")
    return datasets, samples


def audit(output: Path = OUTPUT) -> dict:
    """Return an audited metric projection, failing before reading partial scores."""
    output = Path(output)
    paths = {name: output / name for name in ("run.json", "protocol.json", "predictions.jsonl", "SUMMARY.json")}
    require(all(path.is_file() and not path.is_symlink() for path in paths.values()),
            "Complete diagnostic artifacts are required")
    snapshots = {"run.json": paths["run.json"].read_bytes()}
    run = json.loads(snapshots["run.json"])
    require(run.get("status") == "complete" and run.get("n_completed") == run.get("n_expected") == 136,
            "Diagnostic is incomplete; partial scores must not be exported")
    snapshots.update({name: path.read_bytes() for name, path in paths.items() if name != "run.json"})
    protocol_sha = sha(snapshots["protocol.json"])
    protocol = json.loads(snapshots["protocol.json"])
    datasets, samples = verify_protocol(protocol, protocol_sha)
    prediction_sha = sha(snapshots["predictions.jsonl"])
    require(run.get("protocol_sha256") == protocol_sha and run.get("prediction_sha256") == prediction_sha
            and run.get("device") == "mps" and run.get("dtype") == "float16"
            and run.get("torch_version") == protocol["packages"]["torch"], "Completed run provenance differs")
    values = producer.read_checkpoint(paths["predictions.jsonl"], protocol, protocol_sha)
    require(len(values) == 136, "Diagnostic requires all 136 ordered predictions")
    conditions = producer.summarize(values, datasets, samples)
    require(len(conditions) == 8, "Diagnostic requires all eight conditions")
    expected_raw = {"status": "complete", "protocol_sha256": protocol_sha, "n_contexts": len(values),
        "prediction_sha256": prediction_sha, "conditions": conditions, "test_results_changed": False,
        "method_selected": False, "confidence_gate_fitted": False}
    require(json.loads(snapshots["SUMMARY.json"]) == expected_raw,
            "Raw SUMMARY differs from recomputed predictions, counts or provenance")
    projected = []
    for condition in conditions:
        projected.append({**{key: condition[key] for key in (
            "model_key", "dataset", "shots_per_class", "n_validation", "predicted_class_counts",
            "class_decisions_changed", "label_only_corrected_joint_errors", "label_only_harmed_joint_correct")},
            "metrics": {arm: {new: condition["metrics"][arm][old] for new, old in METRIC_FIELDS.items()}
                        for arm in ARMS}})
    require(all(path.read_bytes() == snapshots[name] for name, path in paths.items()),
            "Diagnostic artifacts changed during audit")
    require(all(sha((ROOT / relative).read_bytes()) == expected
                for relative, expected in protocol["source_sha256"].items()), "Source changed during audit")
    return {"schema_version": 1, "status": "complete", "n_contexts": 136,
        "n_distinct_validation_rows": 34, "n_paired_decisions": 272,
        "conditions": projected, "latency_measurement_status": "not_measured",
        "test_results_changed": False, "method_selected": False, "confidence_gate_fitted": False,
        "interpretation": "Exploratory balanced validation diagnostic; compare scoring rules within the same selected rows.",
        "provenance": {"protocol_sha256": protocol_sha, "predictions_sha256": prediction_sha,
            "raw_summary_sha256": sha(snapshots["SUMMARY.json"]), "producer_sha256": PRODUCER_SHA,
            "frozen_renderer_sha256": producer.HELPER_SHA,
            "inference_packages": protocol["packages"],
            "auditor_sha256": sha(Path(__file__).read_bytes())}}


def findings(summary: dict) -> str:
    lines = ["# Audited validation scoring diagnostic", "",
        "The completed diagnostic covers 136 model contexts on 34 distinct validation cases: "
        "16 Breast Cancer cases (8 per class) and 18 Wine cases (6 per class), evaluated with "
        "SmolLM2 and Granite at zero and four training examples per class. Each context compares "
        "numeric ID scoring with numeric ID plus EOS scoring from the same forward pass per candidate.", "",
        "[AUDITED_SUMMARY.json](AUDITED_SUMMARY.json) contains the verified classification metrics. "
        "All 136 ordered records, frozen source and protocol identities, token-score decomposition, "
        "class counts, and fixed/harmed counts were checked against the unchanged "
        "[raw summary](SUMMARY.json) and predictions. Tokenization was bound by the independently "
        "pinned original preflight; this offline audit did not load tokenizers or models.", "",
        "The raw evaluator's zero latency values are placeholders, not measurements. The audited "
        "summary omits numeric timing fields and marks latency as not measured. No inference-time "
        "or speed comparison is supported by this diagnostic.", "",
        "| Dataset | Model | Examples/class | Validation cases | ID+EOS accuracy | ID-only accuracy | Changed | Fixed / harmed by ID-only |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in summary["conditions"]:
        lines.append(f"| {c['dataset']} | {c['model_key']} | {c['shots_per_class']} | {c['n_validation']} | "
            f"{c['metrics']['label_plus_eos']['accuracy']:.1%} | {c['metrics']['label_only']['accuracy']:.1%} | "
            f"{c['class_decisions_changed']} | {c['label_only_corrected_joint_errors']} / {c['label_only_harmed_joint_correct']} |")
    lines += ["", "Compare the scoring rules only within identical selected validation rows. These small "
        "balanced samples have a different class prevalence from the earlier test sets; their absolute "
        "accuracies must not be compared directly with historical test accuracy. Reusing the same 34 "
        "cases across models and shot counts does not create 136 independent cases.", "",
        "This is exploratory development evidence after earlier test behavior was observed. The "
        "34 validation labels add a diagnostic label budget. No test results were rescored, no winning "
        "method was selected, and no confidence gate was fitted. This isolates two scoring rules "
        "within each frozen model context, not architecture differences. Free generation, adapters, "
        "and claims about performance on new held-out data remain outside this diagnostic."]
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Completed producer artifact directory")
    args = parser.parse_args(argv)
    summary = audit(args.output)
    atomic_write(args.output / "AUDITED_SUMMARY.json", json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    atomic_write(args.output / "AUDITED_FINDINGS.md", findings(summary))
    print(json.dumps({"status": "audited", "n_contexts": summary["n_contexts"],
        "summary": str(args.output / "AUDITED_SUMMARY.json"), "latency_measurement_status": "not_measured"}))


if __name__ == "__main__":
    main()
