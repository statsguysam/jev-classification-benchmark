#!/usr/bin/env python3
"""Read-only saved-source scoring audit; no model loading or API calls.

Only new audit reports are written. Historical inputs are independently audited
through the existing source validators; no source manifest is frozen or changed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
from sklearn.metrics import roc_auc_score
from jevbench.data import normalized_text
from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import digest

MODEL_KEYS = ("qwen_small", "qwen_main", "smollm2", "granite")
SAMPLE_PER_CLASS = {"breast_cancer": 8, "wine": 6, "sst2": 8, "trec": 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    require(np.isfinite(a).all(), "Nonfinite audit statistic")
    return {"n": len(values), "min": float(a.min()), "p25": float(np.quantile(a, .25)),
            "median": float(np.median(a)), "p75": float(np.quantile(a, .75)), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std()), "unique_values": len(set(values))}


def profile(dataset, predictions):
    """Characterize fixed predictions without fitting a gate or choosing a cutoff."""
    require([p.row_id for p in predictions] == [r.id for r in dataset.test], "Prediction order/coverage differs")
    k = len(dataset.labels)
    require(k >= 2, "Classification requires at least two classes")
    pmax, margins, entropies, wrong = [], [], [], []
    counts = [0] * k
    vectors, candidate_sizes, devices, dtypes = [], set(), set(), set()
    zeros = saturated = ties = 0
    for row, prediction in zip(dataset.test, predictions):
        require(prediction.metadata.get("scoring") == "sum_logp_numeric_id_plus_eos" and
                prediction.metadata.get("probability_kind") == "label_sequence_likelihood_normalized",
                "Unexpected source scoring protocol")
        devices.add(prediction.metadata.get("device")); dtypes.add(prediction.metadata.get("dtype"))
        if prediction.error is not None:
            require(prediction.label is None and prediction.probabilities is None, "Malformed source failure")
            continue
        values = prediction.probabilities
        require(isinstance(values, list) and len(values) == k and
                all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values) and
                abs(sum(values) - 1) <= 1e-6, "Invalid source probability vector")
        require(type(prediction.label) is int and prediction.label == max(range(k), key=values.__getitem__),
                "Source argmax differs from class probabilities")
        ordered = sorted(values, reverse=True)
        pmax.append(ordered[0]); margins.append(ordered[0] - ordered[1])
        entropies.append(-sum(v * math.log(v) for v in values if v > 0) / math.log(k))
        wrong.append(int(prediction.label != row.label)); counts[prediction.label] += 1
        vectors.append(values); candidate_sizes.add(prediction.metadata.get("candidate_tokens_scored"))
        zeros += sum(v == 0 for v in values)
        saturated += ordered[0] >= .99
        ties += ordered[0] == ordered[1]
    valid = len(vectors)
    measured = evaluate(dataset.test, predictions, k)
    constant = valid > 0 and sum(c > 0 for c in counts) == 1
    auc = lambda scores: float(roc_auc_score(wrong, [-v for v in scores])) if len(set(wrong)) == 2 else None
    return {"n_test": len(dataset.test), "n_probability_rows": valid, "n_failures": len(dataset.test) - valid,
        "accuracy": measured["accuracy"], "macro_f1": measured["macro_f1"],
        "class_labels": dataset.labels, "true_class_counts": [sum(r.label == i for r in dataset.test) for i in range(k)],
        "predicted_class_counts": counts, "constant_class": constant,
        "constant_class_id": counts.index(valid) if constant else None,
        "dominant_class_fraction": max(counts) / valid if valid else None,
        "top_probability": distribution(pmax), "top_two_margin": distribution(margins),
        "normalized_entropy": distribution(entropies), "n_top_probability_at_least_0_99": saturated,
        "n_zero_probability_entries": zeros, "n_top_probability_ties": ties,
        "unique_probability_vectors": len({tuple(v) for v in vectors}),
        "probability_vectors_sha256": digest(vectors),
        "predicted_labels_sha256": digest([p.label for p in predictions]),
        "candidate_tokens_scored_values": sorted(candidate_sizes, key=str),
        "recorded_devices": sorted(devices, key=str), "recorded_dtypes": sorted(dtypes, key=str),
        "correct_top_probability": distribution([v for v, error in zip(pmax, wrong) if not error]),
        "incorrect_top_probability": distribution([v for v, error in zip(pmax, wrong) if error]),
        "descriptive_error_auroc_negative_pmax": auc(pmax),
        "descriptive_error_auroc_negative_margin": auc(margins),
        "confidence_gate": {"finite_scores_available": valid == len(dataset.test),
            "margin_ranking_has_variation": len(set(margins)) > 1,
            "prospective_threshold_validated": False, "threshold_selected_from_test": False,
            "constant_label_does_not_imply_constant_confidence": constant and len(set(margins)) > 1,
            "eligibility": "candidate_for_new_validation_only_study" if valid else "no_usable_scores",
            "warning": "Restricted ID+EOS probability is uncalibrated. Test error AUROC is descriptive, not gate selection or evidence of reviewer benefit."}}


def validation_inventory(dataset):
    """Select diagnostic rows only from validation; train examples remain fixed."""
    train_ids, val_ids, test_ids = ({r.id for r in getattr(dataset, name)} for name in ("train", "validation", "test"))
    require(not train_ids & val_ids and not train_ids & test_ids and not val_ids & test_ids, "Split IDs overlap")
    groups = {name: {normalized_text(r.text) for r in getattr(dataset, name)} for name in ("train", "validation", "test")}
    require(not groups["train"] & groups["validation"] and not groups["train"] & groups["test"]
            and not groups["validation"] & groups["test"], "Normalized text crosses splits")
    selected = select_examples(dataset.validation, dataset.labels, SAMPLE_PER_CLASS[dataset.name], 42)
    demos = select_examples(dataset.train, dataset.labels, 4, 42)
    require(not {r.id for r in selected} & {r.id for r in demos}, "Ablation rows entered demonstrations")
    return {"dataset": dataset.name, "n_train": len(dataset.train), "n_validation": len(dataset.validation),
        "n_test": len(dataset.test), "validation_class_counts": dict(sorted(Counter(r.label for r in dataset.validation).items())),
        "validation_ids_sha256": digest([r.id for r in dataset.validation]),
        "validation_content_sha256": digest([{ "id": r.id, "text": r.text, "label": r.label} for r in dataset.validation]),
        "split_overlap_ids": 0, "split_overlap_normalized_text": 0,
        "proposed_diagnostic_sample_per_class": SAMPLE_PER_CLASS[dataset.name],
        "proposed_diagnostic_sample_ids": [r.id for r in selected],
        "proposed_diagnostic_sample_ids_sha256": digest([r.id for r in selected]),
        "fixed_four_per_class_demonstration_ids": [r.id for r in demos],
        "sample_selection": "unchanged select_examples(validation, labels, per_class, seed=42); demonstrations selected from train only",
        "n_remaining_validation_rows_after_diagnostic_sample": len(dataset.validation) - len(selected),
        "validation_previously_used_by_some_historical_experiments": dataset.name in ("sst2", "trec"),
        "interpretation": "Diagnostic development data, not guaranteed globally untouched validation; retain independent future test data for confirmation."}


def inspect_cache(specs):
    roots = {"workspace_direct": ROOT / "data/hf", "workspace_hub": ROOT / "data/hf/hub",
             "user_hub": Path.home() / ".cache/huggingface/hub"}
    result = {}
    for key in MODEL_KEYS:
        spec = specs[key]; found = []
        for location, root in roots.items():
            path = root / ("models--" + spec["model"].replace("/", "--")) / "snapshots" / spec["revision"]
            if not path.is_dir():
                continue
            weights = list(path.glob("*.safetensors")); expected = set()
            for index in path.glob("*.safetensors.index.json"):
                expected.update(json.loads(index.read_text())["weight_map"].values())
            found.append({"location": location, "config_present": (path / "config.json").is_file(),
                "tokenizer_config_present": (path / "tokenizer_config.json").is_file(),
                "weights_complete_by_inventory": bool(weights) and all((path / name).is_file() for name in expected),
                "weight_files": len(weights), "weight_bytes": sum(p.stat().st_size for p in weights)})
        result[key] = {"model": spec["model"], "revision": spec["revision"], "cached_snapshots": found}
    return {"models": result, "scope": "Only exact pinned snapshot directories; no credential files or environment secrets accessed. Availability is not a cryptographic weight-content audit."}


def inspect_hardware():
    result = {"os": platform.system(), "architecture": platform.machine(), "cpu_count": os.cpu_count(),
              "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
              "scope": "Current local process visibility only; sandbox permissions can hide MPS. No remote Colab probe or tensor/model allocation."}
    try:
        import torch
        result.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                      mps_built=torch.backends.mps.is_built(), mps_available=torch.backends.mps.is_available())
        if result["mps_available"]:
            result["mps_recommended_memory_bytes"] = torch.mps.recommended_max_memory()
    except ImportError:
        result["torch_installed"] = False
    return result


def collect():
    import run_expanded_numeric_review as numeric
    import text_extension_sources as text
    from run_tabular_classical import validate_native
    from tabular_data import load_native_prepared
    runs, datasets = [], []
    for domain, module, names in (("numeric", numeric, numeric.DATASETS), ("text", text, text.DATASETS)):
        for name in names:
            if domain == "numeric":
                dataset, native = load_native_prepared(ROOT / "data/tabular-full" / name)
                validate_native(dataset, native)
            else:
                dataset = text.load_dataset(name)
            datasets.append({"domain": domain, **validation_inventory(dataset)})
            for key in MODEL_KEYS:
                for shots in (0, 4):
                    job = module.load_source(dataset, key, shots, freeze=False)
                    require(job is not None, "Required complete source is missing")
                    record = job["record"]
                    runs.append({"domain": domain, "dataset": name, "model_key": key, "shots_per_class": shots,
                        "run_id": record["run_id"], "model": record["config"]["model"], "revision": record["config"]["revision"],
                        "source_artifacts": job["source"], "source_manifest_sha256": job["manifest_sha256"],
                        "training_example_ids": record["training_example_ids"], "test_ids_sha256": record["test_ids_sha256"],
                        "renderer": record["config"].get("renderer", {"version": "frozen_original_model_template"}),
                        **profile(dataset, job["predictions"])})
    equal = []
    for domain, names in (("numeric", numeric.DATASETS), ("text", text.DATASETS)):
        for name in names:
            for shots in (0, 4):
                condition = [r for r in runs if (r["domain"], r["dataset"], r["shots_per_class"]) == (domain, name, shots)]
                for index, left in enumerate(condition):
                    for right in condition[index + 1:]:
                        if left["predicted_labels_sha256"] == right["predicted_labels_sha256"]:
                            equal.append({"domain": domain, "dataset": name, "shots_per_class": shots,
                                          "models": [left["model_key"], right["model_key"]]})
    return {"schema_version": 1, "scope": "post-hoc descriptive audit of 32 frozen local source conditions; no new inference",
        "audit_script_sha256": file_sha(__file__), "input_validator_hashes": {
            str(Path(module.__file__).relative_to(ROOT)): file_sha(module.__file__) for module in (numeric, text)},
        "frozen_scoring_source_sha256": file_sha(ROOT / "src/jevbench/providers.py"),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn", "transformers")},
        "n_conditions": len(runs), "n_predictions": sum(r["n_test"] for r in runs),
        "n_failures": sum(r["n_failures"] for r in runs),
        "constant_conditions_by_domain": {domain: sum(r["constant_class"] for r in runs if r["domain"] == domain) for domain in ("numeric", "text")},
        "runs": runs, "identical_proposal_vectors": equal, "validation": datasets,
        "cache_inventory": inspect_cache(numeric.MODELS), "hardware": inspect_hardware(),
        "no_new_inference": True, "no_downloads": True, "no_hosted_calls": True,
        "confidence_gate_selected": False, "historical_results_changed": False}


def markdown(report):
    lines = ["# Saved source scoring audit", "",
        f"Audited **{report['n_conditions']} conditions / {report['n_predictions']:,} saved predictions**, with {report['n_failures']} failures. Existing validators rechecked source hashes, frozen data, ordered test IDs, selected training examples, model identity and metrics. No model inference, downloads, hosted calls or historical edits were performed.", "",
        f"Constant-label behavior occurs in **{report['constant_conditions_by_domain']['numeric']}/16 numerical** and **{report['constant_conditions_by_domain']['text']}/16 text** conditions. Constant predicted labels can coexist with varying confidence. These are descriptive findings on already viewed test data, not evidence for selecting a deployment threshold.", "",
        "| Dataset | Model | Examples/class | Accuracy | Predicted class counts | Median top probability | Median margin | Error AUROC (−margin) |",
        "|---|---|---:|---:|---|---:|---:|---:|"]
    for run in report["runs"]:
        auc = run["descriptive_error_auroc_negative_margin"]
        lines.append(f"| {run['dataset']} | {run['model_key']} | {run['shots_per_class']} | {run['accuracy']:.1%} | {run['predicted_class_counts']} | {run['top_probability']['median']:.4f} | {run['top_two_margin']['median']:.4f} | {auc:.3f} |" if auc is not None else
                     f"| {run['dataset']} | {run['model_key']} | {run['shots_per_class']} | {run['accuracy']:.1%} | {run['predicted_class_counts']} | {run['top_probability']['median']:.4f} | {run['top_two_margin']['median']:.4f} | unavailable |")
    lines.extend(["", "Class counts follow each dataset's saved class order; the JSON report includes those labels, full probability distributions of summary statistics, artifacts and training IDs. Accuracy retains failed rows. Confidence summaries and error AUROC use successful probability-bearing rows. The AUROC uses low margin as the error signal and is unavailable when correctness has only one class. No cutoffs are fitted to these scores.", "",
        "## What the saved scores establish", "",
        "The frozen scorer computes `log P(class-ID tokens | prompt) + log P(EOS | prompt, class-ID tokens)` and softmax-normalizes those joint sequence scores over the allowed labels. The label and EOS terms are summed, not averaged. EOS makes complete candidate strings nonoverlapping events. All inference contexts remain complete; no truncation or adapter change is inferred by this audit.", "",
        "These restricted probabilities are not calibrated probabilities that a class is correct. They condition on the candidate set and the requested immediate stopping behavior. A model may strongly prefer one ID+EOS response while being wrong on most examples. Stored normalized vectors do not preserve separate label/EOS log probabilities or total probability mass outside the candidate set. Consequently this audit cannot establish that EOS caused collapse, recover label-only scores, or predict free-generation performance.", "",
        "Historical Qwen and new Smol/Granite runs also differ in their fixed chat wrappers and recorded hardware. The new models share a fixed system message and disabled thinking, while historical sources retain their original templates. Public-data exposure, arbitrary class IDs/order, numerical serialization and numeric dtype are additional confounds. Identical proposal vectors yield identical downstream Jev prompts under the same dataset/shot condition because confidence and model identity are not sent to Jev.", "",
        "## Confidence-gated review eligibility", "",
        "Finite saved probabilities permit an exploratory retrospective ranking analysis for these local models. A constant label alone does not disqualify a margin-based ranking; an exactly constant margin gives only ties. Neither variation nor high confidence validates a threshold. Do not pool raw margins across models, datasets, shot counts or scoring protocols. Do not interpret source-error ranking as reviewer benefit: Jev may harm confident correct decisions or fail to repair uncertain errors.", "",
        "Learn any confidence threshold using separately declared validation labels and assess it on independent rows. Preserve a random-review baseline at the same request budget and compare full pipelines, including source/reviewer failures and request counts. The historical protocol does not review invalid source proposals; routing those directly to Jev would be a separately declared policy change. Existing hosted generated-label sources lack equivalent normalized vectors, so applying this local confidence rule to them is unsupported.", "",
        "## Proposed validation-only scoring ablation", "",
        "Keep historical tests unchanged and do not rescore or optimize on them. Use pinned SmolLM2 and Granite as the initial two families, selected as an exploratory diagnostic after the observed collapses; Qwen0.5B is an available follow-up and Qwen4B requires a separately authorized cache/download. Use both zero-shot and the original train-only four-per-class examples. No fitting, adapters or label remapping. Preserve each model's exact renderer, tokenizer, float16, 8,192-token limit and explicit device.", "",
        "Select the deterministic seed-42 diagnostic sample **from validation only** using the original selector: 8/class for Breast Cancer and SST-2 (16 rows each), 6/class for Wine (18), and 4/class for TREC (24). The JSON pins all 74 selected row IDs and the unchanged training-demonstration IDs. TREC has only eight validation examples of its rarest class, so four remain outside the small diagnostic sample. Validation labels used for this follow-up are an additional development budget and must be disclosed.", "",
        "For the same rendered context, compare (A) the frozen numeric-ID-plus-EOS joint score and (B) numeric-ID-token likelihood without EOS. Record the label term and conditional EOS term separately from the same teacher-forced forward passes, candidate token IDs, context hash, selected class, normalized vector, margins and any invalid outputs. Check both terms sum to the frozen joint score before interpreting differences. There are 296 model/shot/validation contexts and 592 paired scoring decisions; both scores can be obtained without repeating the forward pass.", "",
        "A separately declared third arm uses greedy free generation on the identical prompt, `do_sample=False`, `max_new_tokens=8`, and the pinned tokenizer EOS; accept only an exact allowed numeric class ID after surrounding-whitespace removal. Explanations, missing IDs, truncation and out-of-range IDs count as failures, not selectively parsed successes. Save generated tokens and termination reason. This arm tests answer generation, so output validity and class accuracy must be reported separately.", "",
        "Freeze the diagnostic recipe before execution. Use the 74-row sample only to diagnose scoring behavior; it is too small to establish a ranking or fit a reliable gate. Before threshold selection, partition the remaining validation groups deterministically into calibration and assessment sets, preserve disjoint normalized-text groups, and record exact IDs and class counts. Fit any recipe/threshold on calibration alone, lock it, then assess it once; do not return to the already viewed test sets as fresh confirmation. Historical text validation has been used in some earlier supervised experiments, so it is not guaranteed globally untouched. A final confirmation needs newly held-out examples or a new dataset, plus reported added validation-label and API budgets.", "",
        "## Available data and compute", "", "| Dataset | Train | Validation | Existing test | Proposed diagnostic |", "|---|---:|---:|---:|---:|"])
    for dataset in report["validation"]:
        lines.append(f"| {dataset['dataset']} | {dataset['n_train']} | {dataset['n_validation']} | {dataset['n_test']} | {len(dataset['proposed_diagnostic_sample_ids'])} |")
    lines.extend(["", "Exact pinned cache inventories and local hardware visibility are recorded in the JSON. Cache inspection reads only snapshot configuration/index metadata and file sizes, not credentials. It is an availability check, not a weight-content authenticity proof. The audit allocates no model or tensor and does not inspect a remote Colab runtime. Sandbox visibility can differ from an authorized local MPS process.", "",
        "Reproduce this saved-score report without inference:", "", "```bash", "python scripts/audit_source_scoring.py", "```", "",
        "`SOURCE_SCORING_AUDIT.json` records input hashes and this audit producer's hash. Its historical-test diagnostics are exploratory. No validation ablation or confidence-gated review result is claimed by this report."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/review_value")
    args = parser.parse_args(argv)
    report = collect()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "SOURCE_SCORING_AUDIT.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output / "SOURCE_SCORING_AUDIT.md").write_text(markdown(report))
    print(json.dumps({"conditions": report["n_conditions"], "predictions": report["n_predictions"],
                      "constant_conditions": report["constant_conditions_by_domain"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
