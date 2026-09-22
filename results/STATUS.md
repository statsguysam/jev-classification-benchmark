# Measured execution status

Snapshot: 22 September 2026. **248 completed executions, 49,600 recorded predictions, zero inference errors.** These predictions reuse the same 200 held-out rows per dataset; the count is not 49,600 distinct test examples. Jev is unmeasured because the user has no API access yet.

| Component | Completed execution | Evidence |
|---|---|---|
| Classical pilot | 40 runs: five datasets × four classifiers × matched-k4/full-prepared tracks, seed 42 | [Execution report](EXECUTION.md) |
| Matched classical study | 180 runs: five datasets × four classifiers × budgets 1/4/8 per class × seeds 13/42/87 | [Seed analysis](MATCHED_ANALYSIS.md) |
| Qwen2.5-0.5B | Six runs: SST-2/TREC × zero-shot/few-shot/LoRA, local MPS | [Small-model report](NEURAL_PILOT.md) |
| Qwen3-4B-Instruct-2507 | Six runs: SST-2/TREC × zero-shot/few-shot/QLoRA, Colab T4 | [Colab results](COLAB_REPORT.md) |
| Colab classical replication | Eight matched-k4, seed-42 runs; class predictions reproduce the local counterparts exactly | [Cross-environment audit](COLAB_AUDIT.md) |
| OpenAI Luna and Astra | Eight runs: two datasets × two models × zero/few-shot; 1,600 requests | [Hosted results](HOSTED_REPORT.md) |
| Jev | No API calls or measured scores; US$2.50 allocation remains unspent | [Access](../docs/MODEL_ACCESS.md) |

The 228 classical executions cover 200 distinct dataset/model/budget/seed configurations. Twenty local k4 configurations repeat between the two local batches; eight repeat on Colab. These are reproducibility checks, not extra independent seeds. The remaining 20 executions are neural/hosted conditions. Classical predictions total 45,600 and neural/hosted predictions total 4,000.

## Main comparison

| Dataset | Model | Zero-shot accuracy | Four-per-class accuracy | LoRA / QLoRA accuracy |
|---|---|---:|---:|---:|
| SST-2 | Qwen2.5-0.5B | 0.500 | 0.850 | 0.895 (LoRA) |
| SST-2 | Qwen3-4B | 0.915 | 0.955 | 0.935 (QLoRA) |
| SST-2 | OpenAI Luna | 0.925 | 0.955 | Unavailable |
| SST-2 | OpenAI Astra | 0.970 | 0.975 | Unavailable |
| TREC | Qwen2.5-0.5B | 0.155 | 0.175 | 0.040 (LoRA) |
| TREC | Qwen3-4B | 0.525 | 0.830 | 0.800 (QLoRA) |
| TREC | OpenAI Luna | 0.525 | 0.850 | Unavailable |
| TREC | OpenAI Astra | 0.970 | 0.970 | Unavailable |

Few-shot, adapted models and matched classical references spend the same four labels per class: eight on SST-2 and 24 on TREC, without development labels. Zero-shot uses none. The full-prepared classical arm uses additional training/development labels and is reported separately. The fixed matched TF-IDF MultinomialNB reference reaches 0.540 SST-2 and 0.500 TREC accuracy. See [COMPARISON.md](COMPARISON.md) for macro-F1, bootstrap intervals, model provenance, timing and all classical reference choices.

The 4B QLoRA recipe did not beat few-shot in either observed task. The smaller model improved on SST-2, but the paired macro-F1 difference interval includes zero; its TREC adapter predicts abbreviation on 188/200 rows and performs poorly. All results are retained. No test-guided prompt, checkpoint or recipe retuning was performed.

## Reproducibility and budget

All runs use preparation seed 42, the first 2,000 characters of each text and declared train/dev/test caps of 10,000/1,000/200. Pinned source snapshots, raw-file/content hashes, labels and ordered held-out IDs are recorded. Colab added two supplemental audit fields to the manifest; [the separate audit](COLAB_AUDIT.md) recomputes identical raw prepared content while preserving the unequal original manifest hashes. The inference core remained frozen across runs. Package versions, precision and hardware differ and are retained in each run.

[All 1,600 OpenAI requests reconcile](API_COSTS.md): US$2.568174 at reported-token standard rates, or US$3.142607 under conservative ledger accounting, below its US$7.50 allocation. These are estimates, not a provider invoice or account-wide bill. No Jev spending occurred. Credentials are excluded from repository/release artifacts.

The [pilot release](https://github.com/statsguysam/jev-classification-benchmark/releases/tag/v0.1.0-pilot) contains the source bundle and four adapter packages in one archive, with upstream licenses, modification notices and SHA-256 manifests. Base weights and raw corpus text are excluded.

## Remaining scope and limits

* Obtain Jev access and execute its zero/few-shot arm before drawing any Jev comparison.
* Extend open/hosted models beyond SST-2/TREC, then increase test coverage, training budgets and neural selection seeds. IMDb, AG News and Banking77 currently have classical measurements only.
* Full-prepared LoRA, additional frontier providers, prompt/verbalizer sensitivity and calibration variants are unrun extensions. Use development data for future tuning.
* One neural selection seed and 200 test examples per task support a pilot, not a publication-scale general ranking. Bootstrap intervals are conditional test-item uncertainty; multiple exploratory contrasts are unadjusted.
* Open models use numeric-label-plus-EOS likelihood; hosted models generate a numeric label. Their probabilities and timings are not interchangeable. Public-dataset contamination and unequal pretraining resources remain unresolved.

[Combined results](COMPARISON.md) · [Paired comparisons](comparisons/combined/index.md) · [Protocol](../docs/PROTOCOL.md) · [Sources](../docs/SOURCES.md)
