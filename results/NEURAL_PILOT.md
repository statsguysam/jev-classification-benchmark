# Open-model pilot: measured results

**6 completed runs and 1,200 recorded predictions** for **Qwen2.5-0.5B-Instruct**, pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`. These are exploratory local-MPS results. This document covers only the small-model arm; see [COMPARISON.md](COMPARISON.md) for Colab, OpenAI and Jev measurements; this report does not summarize those arms.

Every row uses the same 200 held-out examples per dataset and the shared 2,000-character input policy. Selection seed is 42. Local inference scores the complete numeric class ID plus EOS and normalizes likelihood over the permitted labels. This is a specific verbalizer/scoring protocol, not a claim about the best achievable performance of this model.

Few-shot and LoRA both use exactly four training examples per class: eight for SST-2 and 24 for TREC. Matched classical references use those same examples, with no development labels. LoRA uses a fixed three epochs, rank 8, alpha 16, learning rate 0.0002 and response-only supervision. Both adapters use their final checkpoint; no test-guided retuning or checkpoint selection was performed.

| Dataset | Method | New training labels | Accuracy | Macro F1 | 95% macro-F1 CI | NLL | Brier sum | ECE |
|---|---|---:|---:|---:|---|---:|---:|---:|
| sst2 | zero_shot | 0 | 0.5000 | 0.3503 | [0.3289, 0.3812] | 0.7051 | 0.5120 | 0.1849 |
| sst2 | few_shot | 8 | 0.8500 | 0.8499 | [0.8000, 0.8950] | 0.4122 | 0.2496 | 0.1802 |
| sst2 | lora | 8 | 0.8950 | 0.8944 | [0.8488, 0.9348] | 0.3265 | 0.1925 | 0.1055 |
| trec | zero_shot | 0 | 0.1550 | 0.0817 | [0.0677, 0.0954] | 2.7296 | 1.0899 | 0.3375 |
| trec | few_shot | 24 | 0.1750 | 0.0881 | [0.0638, 0.1153] | 2.4058 | 0.9237 | 0.2338 |
| trec | lora | 24 | 0.0400 | 0.0445 | [0.0158, 0.0799] | 2.0089 | 0.9209 | 0.2912 |

Intervals above use 1,000 stratified test-item bootstrap resamples. They are conditional on these fixed test sets, training subsets and one selection seed. NLL, Brier and 15-bin ECE describe normalized restricted-label likelihoods; these are not Jev-native probabilities.

## Fixed classical references

These references use the same test items. The larger supervised track spends extra training and development labels and is not an equal-label comparison.

| Dataset | Model | Training labels | Development labels | Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| sst2 | logistic_regression | 8 | 0 | 0.4950 | 0.4826 |
| sst2 | logistic_regression | 10000 | 1000 | 0.7950 | 0.7949 |
| sst2 | linear_svc | 8 | 0 | 0.5000 | 0.4869 |
| sst2 | linear_svc | 10000 | 1000 | 0.7950 | 0.7949 |
| sst2 | multinomial_nb | 8 | 0 | 0.5400 | 0.5388 |
| sst2 | multinomial_nb | 10000 | 1000 | 0.8450 | 0.8443 |
| trec | logistic_regression | 24 | 0 | 0.5000 | 0.4857 |
| trec | logistic_regression | 4886 | 545 | 0.8700 | 0.8679 |
| trec | linear_svc | 24 | 0 | 0.4950 | 0.4757 |
| trec | linear_svc | 4886 | 545 | 0.9000 | 0.8980 |
| trec | multinomial_nb | 24 | 0 | 0.5000 | 0.4857 |
| trec | multinomial_nb | 4886 | 545 | 0.7750 | 0.7914 |

## Interpretation and limits

* SST-2 LoRA minus few-shot has an observed macro-F1 difference of +0.0445. The paired 2,000-resample 95% interval is [-0.0007, +0.0900] and includes zero. This small, single-seed pilot does not establish a clear LoRA advantage.
* TREC LoRA collapsed toward **abbreviation**, choosing it on **188/200 examples (94%)**. Its accuracy is 0.0400 and macro-F1 is 0.0445. This unfavorable result is retained. No prompt, verbalizer, training recipe or checkpoint was retuned against the observed test scores; its cause is not established.
* TREC LoRA minus few-shot macro-F1 is -0.0435, with paired 2,000-resample 95% interval [-0.0804, -0.0095]. This describes a decline for these fixed runs; it does not identify the cause or incorporate training-seed/prompt-selection uncertainty. [Paired comparison](comparisons/trec_lora_k4_minus_few4_seed42.json).
* TREC LoRA used 9 optimizer steps on 24 rows. Three epochs and falling training loss do not establish convergence. The [pairing audit](comparisons/trec_pairing_audit_seed42.json) verifies shared test/training IDs, four examples per class, no development-label use, no train/test text overlap and the training prompt hash.
* The weak zero-shot and TREC results are specific to the fixed numeric-label/EOS likelihood protocol. Native generation, other verbalizers, prompt sensitivity and contextual calibration are separate, unrun experiments.
* There are 0 inference errors across 1,200 predictions and 1,200/1,200 valid class distributions. Valid output structure does not imply correct classification.
* Local training uses float32 MPS; inference uses float16 MPS. Inference recomputes the context for each candidate class. Some neural execution overlapped local CPU benchmark/analysis work. Timing is exploratory and not an isolated serving or architecture-level speed comparison. Model download/loading is outside per-row timing.
* LLM pretraining data and compute are not matched to classical models. Widely known public datasets may have appeared in pretraining. Equal new task labels do not imply equal total learning resources.
* Classical seed sensitivity is reported separately in [MATCHED_ANALYSIS.md](MATCHED_ANALYSIS.md). Neural pilots use one seed; full-test-set and multi-seed neural results remain pending.

## Recorded neural artifacts

* sst2 / zero_shot: [run metadata and metrics](pilot/sst2__Qwen2.5-0.5B-Instruct__594a827ceab2/run.json), [per-example predictions](pilot/sst2__Qwen2.5-0.5B-Instruct__594a827ceab2/predictions.jsonl).
* sst2 / few_shot: [run metadata and metrics](pilot/sst2__Qwen2.5-0.5B-Instruct__9800b4c889cc/run.json), [per-example predictions](pilot/sst2__Qwen2.5-0.5B-Instruct__9800b4c889cc/predictions.jsonl).
* sst2 / lora: [run metadata and metrics](pilot/sst2__Qwen2.5-0.5B-Instruct__c29d9f2be13f/run.json), [per-example predictions](pilot/sst2__Qwen2.5-0.5B-Instruct__c29d9f2be13f/predictions.jsonl).
* trec / zero_shot: [run metadata and metrics](pilot/trec__Qwen2.5-0.5B-Instruct__67b435872393/run.json), [per-example predictions](pilot/trec__Qwen2.5-0.5B-Instruct__67b435872393/predictions.jsonl).
* trec / few_shot: [run metadata and metrics](pilot/trec__Qwen2.5-0.5B-Instruct__5d00e79e1d60/run.json), [per-example predictions](pilot/trec__Qwen2.5-0.5B-Instruct__5d00e79e1d60/predictions.jsonl).
* trec / lora: [run metadata and metrics](pilot/trec__Qwen2.5-0.5B-Instruct__23d9b42b5b4e/run.json), [per-example predictions](pilot/trec__Qwen2.5-0.5B-Instruct__23d9b42b5b4e/predictions.jsonl).

Rebuild this document with `python scripts/summarize_neural_pilot.py`.

[Execution status](STATUS.md) · [SST-2 paired comparison](comparisons/sst2-qwen-lora-k4-vs-few4.json) · [Full pilot table](REPORT.md) · [Protocol](../docs/PROTOCOL.md)
