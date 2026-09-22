# Saved source scoring audit

Audited **32 conditions / 4,400 saved predictions**, with 0 failures. Existing validators rechecked source hashes, frozen data, ordered test IDs, selected training examples, model identity and metrics. No model inference, downloads, hosted calls or historical edits were performed.

Constant-label behavior occurs in **11/16 numerical** and **3/16 text** conditions. Constant predicted labels can coexist with varying confidence. These are descriptive findings on already viewed test data, not evidence for selecting a deployment threshold.

| Dataset | Model | Examples/class | Accuracy | Predicted class counts | Median top probability | Median margin | Error AUROC (−margin) |
|---|---|---:|---:|---|---:|---:|---:|
| breast_cancer | qwen_small | 0 | 37.7% | [114, 0] | 0.8180 | 0.6359 | 0.314 |
| breast_cancer | qwen_small | 4 | 61.4% | [1, 113] | 0.6370 | 0.2740 | 0.206 |
| breast_cancer | qwen_main | 0 | 61.4% | [13, 101] | 0.9466 | 0.8932 | 0.529 |
| breast_cancer | qwen_main | 4 | 86.0% | [59, 55] | 1.0000 | 1.0000 | 0.849 |
| breast_cancer | smollm2 | 0 | 37.7% | [114, 0] | 0.6784 | 0.3569 | 0.479 |
| breast_cancer | smollm2 | 4 | 37.7% | [114, 0] | 0.5487 | 0.0974 | 0.426 |
| breast_cancer | granite | 0 | 37.7% | [114, 0] | 0.9996 | 0.9992 | 0.607 |
| breast_cancer | granite | 4 | 61.4% | [87, 27] | 0.9172 | 0.8345 | 0.741 |
| wine | qwen_small | 0 | 33.3% | [36, 0, 0] | 0.5956 | 0.2253 | 0.253 |
| wine | qwen_small | 4 | 33.3% | [36, 0, 0] | 0.6221 | 0.2914 | 0.160 |
| wine | qwen_main | 0 | 38.9% | [0, 36, 0] | 0.9740 | 0.9481 | 0.653 |
| wine | qwen_main | 4 | 80.6% | [9, 11, 16] | 1.0000 | 1.0000 | 0.591 |
| wine | smollm2 | 0 | 33.3% | [36, 0, 0] | 0.4608 | 0.1605 | 0.729 |
| wine | smollm2 | 4 | 33.3% | [36, 0, 0] | 0.5912 | 0.3792 | 0.087 |
| wine | granite | 0 | 27.8% | [0, 0, 36] | 0.9788 | 0.9660 | 0.296 |
| wine | granite | 4 | 33.3% | [36, 0, 0] | 0.9828 | 0.9663 | 0.868 |
| sst2 | qwen_small | 0 | 50.0% | [198, 2] | 0.6745 | 0.3490 | 0.735 |
| sst2 | qwen_small | 4 | 85.0% | [98, 102] | 0.7041 | 0.4083 | 0.858 |
| sst2 | qwen_main | 0 | 91.5% | [91, 109] | 1.0000 | 1.0000 | 0.831 |
| sst2 | qwen_main | 4 | 95.5% | [101, 99] | 1.0000 | 1.0000 | 0.817 |
| sst2 | smollm2 | 0 | 49.0% | [200, 0] | 0.8543 | 0.7086 | 0.798 |
| sst2 | smollm2 | 4 | 70.0% | [42, 158] | 0.7716 | 0.5431 | 0.847 |
| sst2 | granite | 0 | 80.0% | [136, 64] | 0.9996 | 0.9992 | 0.677 |
| sst2 | granite | 4 | 94.0% | [108, 92] | 0.9999 | 0.9998 | 0.860 |
| trec | qwen_small | 0 | 15.5% | [109, 89, 0, 2, 0, 0] | 0.4751 | 0.1879 | 0.544 |
| trec | qwen_small | 4 | 17.5% | [0, 142, 17, 41, 0, 0] | 0.3952 | 0.1325 | 0.444 |
| trec | qwen_main | 0 | 52.5% | [2, 1, 126, 39, 26, 6] | 1.0000 | 1.0000 | 0.655 |
| trec | qwen_main | 4 | 83.0% | [4, 25, 63, 30, 42, 36] | 1.0000 | 1.0000 | 0.758 |
| trec | smollm2 | 0 | 2.0% | [200, 0, 0, 0, 0, 0] | 0.3589 | 0.1430 | 0.903 |
| trec | smollm2 | 4 | 27.5% | [0, 0, 200, 0, 0, 0] | 0.4654 | 0.3119 | 0.738 |
| trec | granite | 0 | 46.0% | [5, 0, 124, 37, 34, 0] | 0.9602 | 0.9321 | 0.647 |
| trec | granite | 4 | 63.5% | [7, 2, 73, 47, 33, 38] | 0.9389 | 0.8956 | 0.727 |

Class counts follow each dataset's saved class order; the JSON report includes those labels, full probability distributions of summary statistics, artifacts and training IDs. Accuracy retains failed rows. Confidence summaries and error AUROC use successful probability-bearing rows. The AUROC uses low margin as the error signal and is unavailable when correctness has only one class. No cutoffs are fitted to these scores.

## What the saved scores establish

The frozen scorer computes `log P(class-ID tokens | prompt) + log P(EOS | prompt, class-ID tokens)` and softmax-normalizes those joint sequence scores over the allowed labels. The label and EOS terms are summed, not averaged. EOS makes complete candidate strings nonoverlapping events. All inference contexts remain complete; no truncation or adapter change is inferred by this audit.

These restricted probabilities are not calibrated probabilities that a class is correct. They condition on the candidate set and the requested immediate stopping behavior. A model may strongly prefer one ID+EOS response while being wrong on most examples. Stored normalized vectors do not preserve separate label/EOS log probabilities or total probability mass outside the candidate set. Consequently this audit cannot establish that EOS caused collapse, recover label-only scores, or predict free-generation performance.

Historical Qwen and new Smol/Granite runs also differ in their fixed chat wrappers and recorded hardware. The new models share a fixed system message and disabled thinking, while historical sources retain their original templates. Public-data exposure, arbitrary class IDs/order, numerical serialization and numeric dtype are additional confounds. Identical proposal vectors yield identical downstream Jev prompts under the same dataset/shot condition because confidence and model identity are not sent to Jev.

## Confidence-gated review eligibility

Finite saved probabilities permit an exploratory retrospective ranking analysis for these local models. A constant label alone does not disqualify a margin-based ranking; an exactly constant margin gives only ties. Neither variation nor high confidence validates a threshold. Do not pool raw margins across models, datasets, shot counts or scoring protocols. Do not interpret source-error ranking as reviewer benefit: Jev may harm confident correct decisions or fail to repair uncertain errors.

Learn any confidence threshold using separately declared validation labels and assess it on independent rows. Preserve a random-review baseline at the same request budget and compare full pipelines, including source/reviewer failures and request counts. The historical protocol does not review invalid source proposals; routing those directly to Jev would be a separately declared policy change. Existing hosted generated-label sources lack equivalent normalized vectors, so applying this local confidence rule to them is unsupported.

## Proposed validation-only scoring ablation

Keep historical tests unchanged and do not rescore or optimize on them. Use pinned SmolLM2 and Granite as the initial two families, selected as an exploratory diagnostic after the observed collapses; Qwen0.5B is an available follow-up and Qwen4B requires a separately authorized cache/download. Use both zero-shot and the original train-only four-per-class examples. No fitting, adapters or label remapping. Preserve each model's exact renderer, tokenizer, float16, 8,192-token limit and explicit device.

Select the deterministic seed-42 diagnostic sample **from validation only** using the original selector: 8/class for Breast Cancer and SST-2 (16 rows each), 6/class for Wine (18), and 4/class for TREC (24). The JSON pins all 74 selected row IDs and the unchanged training-demonstration IDs. TREC has only eight validation examples of its rarest class, so four remain outside the small diagnostic sample. Validation labels used for this follow-up are an additional development budget and must be disclosed.

For the same rendered context, compare (A) the frozen numeric-ID-plus-EOS joint score and (B) numeric-ID-token likelihood without EOS. Record the label term and conditional EOS term separately from the same teacher-forced forward passes, candidate token IDs, context hash, selected class, normalized vector, margins and any invalid outputs. Check both terms sum to the frozen joint score before interpreting differences. There are 296 model/shot/validation contexts and 592 paired scoring decisions; both scores can be obtained without repeating the forward pass.

A separately declared third arm uses greedy free generation on the identical prompt, `do_sample=False`, `max_new_tokens=8`, and the pinned tokenizer EOS; accept only an exact allowed numeric class ID after surrounding-whitespace removal. Explanations, missing IDs, truncation and out-of-range IDs count as failures, not selectively parsed successes. Save generated tokens and termination reason. This arm tests answer generation, so output validity and class accuracy must be reported separately.

Freeze the diagnostic recipe before execution. Use the 74-row sample only to diagnose scoring behavior; it is too small to establish a ranking or fit a reliable gate. Before threshold selection, partition the remaining validation groups deterministically into calibration and assessment sets, preserve disjoint normalized-text groups, and record exact IDs and class counts. Fit any recipe/threshold on calibration alone, lock it, then assess it once; do not return to the already viewed test sets as fresh confirmation. Historical text validation has been used in some earlier supervised experiments, so it is not guaranteed globally untouched. A final confirmation needs newly held-out examples or a new dataset, plus reported added validation-label and API budgets.

## Available data and compute

| Dataset | Train | Validation | Existing test | Proposed diagnostic |
|---|---:|---:|---:|---:|
| breast_cancer | 341 | 114 | 114 | 16 |
| wine | 106 | 36 | 36 | 18 |
| sst2 | 10000 | 1000 | 200 | 16 |
| trec | 4886 | 545 | 200 | 24 |

Exact pinned cache inventories and local hardware visibility are recorded in the JSON. Cache inspection reads only snapshot configuration/index metadata and file sizes, not credentials. It is an availability check, not a weight-content authenticity proof. The audit allocates no model or tensor and does not inspect a remote Colab runtime. Sandbox visibility can differ from an authorized local MPS process.

Reproduce this saved-score report without inference:

```bash
python scripts/audit_source_scoring.py
```

`SOURCE_SCORING_AUDIT.json` records input hashes and this audit producer's hash. Its historical-test diagnostics are exploratory. No validation ablation or confidence-gated review result is claimed by this report.
