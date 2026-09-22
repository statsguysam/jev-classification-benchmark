#!/usr/bin/env python3
"""Rebuild the read-only 14-run Colab versus local pilot import audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_cross_environment import audit, load_json, require
from compare_combined_pilot import read_predictions
from jevbench.runner import environment

LABEL_METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "failure_rate", "n_failures",
                 "confusion_matrix", "per_class", "valid_response_accuracy")


def matches(imported: dict, local: dict) -> bool:
    if any(imported[key] != local[key] for key in ("dataset", "method", "seed")):
        return False
    if imported["method"] == "classical":
        return imported["config"] == local["config"]
    return (local["config"]["model"] == "Qwen/Qwen2.5-0.5B-Instruct" and
            imported["config"].get("shots_per_class") == local["config"].get("shots_per_class") and
            imported.get("adapter_training", {}).get("train_per_class") == local.get("adapter_training", {}).get("train_per_class"))


def fingerprint(paths: set[Path]) -> dict:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--data-root", type=Path, default=Path("data/pilot"))
    parser.add_argument("--output", type=Path, default=Path("results/COLAB_AUDIT.json"))
    args = parser.parse_args()
    imported = [(p.parent, load_json(p)) for p in sorted((args.results_root / "colab").glob("*/run.json"))]
    local = [(p.parent, load_json(p)) for p in sorted((args.results_root / "pilot").glob("*/run.json"))]
    require(len(imported) == 14 and all(record["status"] == "complete" for _, record in imported), "This audit requires the completed fixed 14-run Colab pilot")
    source_hash = environment()["source_sha256"]
    originals, pairs = set(), []
    for directory, record in imported:
        candidates = [(path, row) for path, row in local if row["status"] == "complete" and matches(record, row)]
        require(len(candidates) == 1, f"Expected one exact local protocol counterpart for {directory.name}")
        other, counterpart = candidates[0]
        pairs.append((directory, record, other, counterpart))
        originals.update(path / filename for path in (directory, other) for filename in ("run.json", "test_manifest.json", "predictions.jsonl"))
        originals.update(args.data_root / record["dataset"] / filename for filename in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"))
    require(args.output.resolve() not in {path.resolve() for path in originals} and args.output.with_suffix(".md").resolve() not in {path.resolve() for path in originals}, "Audit output cannot overwrite original artifacts")
    before = fingerprint(originals)
    results = []
    for directory, record, other, counterpart in pairs:
        proof = audit(directory, other, args.data_root / record["dataset"])
        require(proof["eligible_for_separately_labeled_matched_pairing"] and proof["raw_content_recomputed"], f"Content audit failed for {directory.name}")
        recorded_hashes = {record.get("implementation_sha256"), record.get("environment", {}).get("source_sha256"), counterpart.get("implementation_sha256"), counterpart.get("environment", {}).get("source_sha256")}
        recorded_hashes.update(session.get("environment", {}).get("source_sha256") for session in record.get("execution_sessions", []))
        require(recorded_hashes == {source_hash}, f"Core source differs for {directory.name}")
        test = load_json(directory / "test_manifest.json")
        local_test = load_json(other / "test_manifest.json")
        predictions, predicted_evidence = read_predictions(directory, record, test)
        other_predictions, other_evidence = read_predictions(other, counterpart, local_test)
        item = {"colab_run": directory.name, "local_run": other.name, "dataset": record["dataset"], "method": record["method"],
                "identity_only_for_different_neural_models": record["method"] != "classical", "core_source_sha256": source_hash,
                "content_audit": proof, "colab_prediction_evidence": predicted_evidence, "local_prediction_evidence": other_evidence}
        if record["method"] == "classical":
            label_metrics_equal = all(record["metrics"][key] == counterpart["metrics"][key] for key in LABEL_METRICS)
            require(predictions == other_predictions and label_metrics_equal, f"Classical labels or scores differ for {directory.name}")
            item["classical_replication"] = {"all_ordered_predicted_labels_identical": True, "label_metrics_exactly_equal": True,
                                              "metrics_checked": list(LABEL_METRICS), "config_exactly_equal": record["config"] == counterpart["config"]}
        results.append(item)
    require(fingerprint(originals) == before, "An original artifact changed while the audit ran")
    require(sum(item["method"] == "classical" for item in results) == 8, "Expected eight classical replications")
    report = {"audit_type": "full_colab_import_content_and_classical_replication_audit", "schema_version": 1, "passed": True,
              "runs_verified": len(results), "classical_replications_verified": 8, "neural_identity_only_checks": 6,
              "current_local_core_source_sha256": source_hash, "original_artifacts_modified": False,
              "original_artifact_file_hashes": before, "runs": results,
              "interpretation": "All 14 imported runs preserve identical prepared content, ordered held-out IDs/labels/text, selected training IDs and seed relative to their local counterparts. Unequal original manifest hashes remain unequal; the only differences are supplemental audit fields. Neural comparisons here establish data/training identity only, not equal models, protocols, probabilities, or predictions. Eight classical runs have exactly identical predicted class labels and label-based scores across environments."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Colab import audit", "", "**All 14 completed imported runs passed.** The eight classical replications produced exactly the same ordered class predictions, accuracy, macro F1, balanced accuracy, confusion matrices and per-class scores as their local counterparts. Both environments used the same frozen data and selected training rows.", "", f"The current local inference core and every imported run record share source SHA-256 `{source_hash}`. Colab used Python 3.13; local runs used Python 3.12. The audit checks realized rows and file hashes rather than inferring reproducibility from equal seeds.", "", "Original manifest hashes differ and remain unchanged. Their only differing keys are `splits_before_text_transform` and `post_transform_overlap_removed_ids`, which were added as supplemental preparation audits. For every pair, all three frozen JSONL file hashes and the full prepared-content hash were recomputed from local files and matched both recorded manifests. Ordered test IDs, labels and text hashes; selected training IDs; selection seed; and validation-label use also matched.", "", "The six neural checks compare the 4B and 0.5B runs only for data/training identity. They do not establish identical models, numerical execution, probabilities or predictions. Probability and timing equality is not claimed for the classical replications. Original `compare_runs` retains its exact-manifest requirement; separately labeled paired analyses retain both hashes and embed this content evidence.", "", "| Dataset | Method / imported model | Local counterpart | Data/training audit | Classical class-label replication |", "|---|---|---|---|---|"]
    for item, (_, record, _, _) in zip(results, pairs):
        classical = "exact" if item["method"] == "classical" else "not applicable"
        lines.append(f"| {item['dataset']} | [{item['method']} / {record['config']['model']}](colab/{item['colab_run']}/run.json) | [run](pilot/{item['local_run']}/run.json) | passed | {classical} |")
    lines += ["", f"The [complete JSON evidence]({args.output.name}) preserves both full manifest hashes, content/file hashes, exact metadata differences and before/after hashes confirming original artifacts were unchanged. [Exploratory paired comparisons](comparisons/combined/index.md).", "", "Rebuild with `PYTHONPATH=src .venv/bin/python scripts/audit_colab_import.py`.", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(f"PASS: {len(results)} content audits, 8 exact classical label/score replications; {args.output}")


if __name__ == "__main__":
    main()
