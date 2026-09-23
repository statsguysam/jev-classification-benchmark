# Text classification and Jev review

**68/68 conditions are complete and audited.** LLM alone: 24/24; LLM → Jev: 24/24; Jev alone: 4/4; classical references: 16/16.

SST-2 has two sentiment classes; TREC has six question categories. Each uses the same frozen 200 held-out rows across every arm. Zero-shot supplies no labels; few-shot supplies exactly four examples per class (8 SST-2 and 24 TREC labels), with identical ordered examples at the source and reviewer stages.

Missing or partial conditions have no score, confidence interval or inferred improvement. Cached LLM labels and direct Jev measurements are reused after full-content and matched-training audits; original cross-environment manifest hashes are preserved.

[Full metrics and paired contrasts](COMPARISON.json) · [Protocol](protocol.json) · [Classical evidence](classical/README.md)

## SST-2 sentiment · zero-shot

Jev alone: **93.5% accuracy** on 200 rows.

| LLM | LLM alone | LLM → Jev | Fixed / harmed | Review − LLM, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 50.0% | 93.5% | 88 / 1 | +43.50 [+36.50, +50.50] | +0.00 [-2.00, +2.00] |
| Qwen3 4B | 91.5% | 94.0% | 12 / 7 | +2.50 [-2.00, +7.00] | +0.50 [-1.00, +2.50] |
| SmolLM2 1.7B | 49.0% | 93.5% | 90 / 1 | +44.50 [+37.50, +51.50] | +0.00 [-2.00, +2.00] |
| Granite 3.3 2B | 80.0% | 93.5% | 29 / 2 | +13.50 [+8.50, +19.00] | +0.00 [-2.00, +2.00] |
| GPT-5.6 Luna | 92.5% | 94.5% | 5 / 1 | +2.00 [+0.00, +4.50] | +1.00 [-1.00, +3.00] |
| GPT-6 Astra | 97.0% | 93.5% | 2 / 9 | -3.50 [-7.00, -0.50] | +0.00 [-2.00, +2.00] |

## SST-2 sentiment · four examples per class

Jev alone: **96.5% accuracy** on 200 rows.

| LLM | LLM alone | LLM → Jev | Fixed / harmed | Review − LLM, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 85.0% | 96.5% | 26 / 3 | +11.50 [+6.50, +17.00] | +0.00 [+0.00, +0.00] |
| Qwen3 4B | 95.5% | 96.0% | 6 / 5 | +0.50 [-2.50, +4.00] | -0.50 [-1.50, +0.00] |
| SmolLM2 1.7B | 70.0% | 96.5% | 57 / 4 | +26.50 [+19.50, +33.00] | +0.00 [+0.00, +0.00] |
| Granite 3.3 2B | 94.0% | 96.0% | 6 / 2 | +2.00 [-0.50, +5.00] | -0.50 [-1.50, +0.00] |
| GPT-5.6 Luna | 95.5% | 96.5% | 4 / 2 | +1.00 [-1.01, +3.50] | +0.00 [+0.00, +0.00] |
| GPT-6 Astra | 97.5% | 96.5% | 1 / 3 | -1.00 [-3.00, +1.00] | +0.00 [+0.00, +0.00] |

## TREC question categories · zero-shot

Jev alone: **33.5% accuracy** on 200 rows.

| LLM | LLM alone | LLM → Jev | Fixed / harmed | Review − LLM, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 15.5% | 32.5% | 60 / 26 | +17.00 [+8.00, +25.50] | -1.00 [-4.00, +2.00] |
| Qwen3 4B | 52.5% | 41.5% | 6 / 28 | -11.00 [-16.50, -5.50] | +8.00 [+4.00, +12.00] |
| SmolLM2 1.7B | 2.0% | 32.5% | 61 / 0 | +30.50 [+24.00, +37.00] | -1.00 [-4.50, +2.00] |
| Granite 3.3 2B | 46.0% | 39.5% | 7 / 20 | -6.50 [-12.00, -1.50] | +6.00 [+2.50, +10.00] |
| GPT-5.6 Luna | 52.5% | 39.5% | 4 / 30 | -13.00 [-18.50, -7.50] | +6.00 [+2.00, +10.01] |
| GPT-6 Astra | 97.0% | 42.0% | 0 / 110 | -55.00 [-62.00, -48.00] | +8.50 [+4.00, +13.01] |

## TREC question categories · four examples per class

Jev alone: **85.5% accuracy** on 200 rows.

| LLM | LLM alone | LLM → Jev | Fixed / harmed | Review − LLM, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 17.5% | 87.5% | 143 / 3 | +70.00 [+63.00, +76.50] | +2.00 [+0.50, +4.00] |
| Qwen3 4B | 83.0% | 86.0% | 23 / 17 | +3.00 [-3.50, +8.50] | +0.50 [+0.00, +1.50] |
| SmolLM2 1.7B | 27.5% | 84.5% | 133 / 19 | +57.00 [+47.50, +66.00] | -1.00 [-3.00, +1.00] |
| Granite 3.3 2B | 63.5% | 85.5% | 57 / 13 | +22.00 [+14.50, +29.50] | +0.00 [-3.00, +3.00] |
| GPT-5.6 Luna | 85.0% | 85.0% | 20 / 20 | +0.00 [-6.00, +6.00] | -0.50 [-2.50, +1.50] |
| GPT-6 Astra | 97.0% | 86.5% | 3 / 24 | -10.50 [-15.50, -6.00] | +1.00 [-1.50, +3.50] |

## Classical ML · same four labels per class

| Dataset | Method | Training labels | Accuracy | Macro-F1 | Failures |
|---|---|---:|---:|---:|---:|
| SST-2 sentiment | Logistic regression | 8 | 49.5% | 0.4826 | 0 |
| SST-2 sentiment | Random forest | 8 | 53.0% | 0.5206 | 0 |
| SST-2 sentiment | XGBoost | 8 | 44.5% | 0.4439 | 0 |
| SST-2 sentiment | LightGBM | 8 | 47.5% | 0.4749 | 0 |
| TREC question categories | Logistic regression | 24 | 50.0% | 0.4857 | 0 |
| TREC question categories | Random forest | 24 | 49.0% | 0.4289 | 0 |
| TREC question categories | XGBoost | 24 | 35.0% | 0.2536 | 0 |
| TREC question categories | LightGBM | 24 | 37.5% | 0.3270 | 0 |

## Classical ML · full prepared training, additional labels

| Dataset | Method | Training labels | Accuracy | Macro-F1 | Failures |
|---|---|---:|---:|---:|---:|
| SST-2 sentiment | Logistic regression | 10000 | 79.0% | 0.7892 | 0 |
| SST-2 sentiment | Random forest | 10000 | 76.0% | 0.7596 | 0 |
| SST-2 sentiment | XGBoost | 10000 | 69.0% | 0.6796 | 0 |
| SST-2 sentiment | LightGBM | 10000 | 67.5% | 0.6670 | 0 |
| TREC question categories | Logistic regression | 4886 | 86.5% | 0.8635 | 0 |
| TREC question categories | Random forest | 4886 | 81.0% | 0.7862 | 0 |
| TREC question categories | XGBoost | 4886 | 70.5% | 0.7324 | 0 |
| TREC question categories | LightGBM | 4886 | 72.5% | 0.7376 | 0 |

## Scope and costs

New Jev text-review requests: **4800**. Known reported cost: **$0.138994044**; unknown-cost calls: **6**. Cumulative conservative accounting, including earlier work: **$23.659305471 / $25.00**. The new text stage is capped at $1.60 and preserves the remaining numerical-review allocation.

Known provider charges and conservative reservations are different quantities. Local compute is unpriced; historical hosted source generation is already included in earlier accounting. A deployed chain pays for both stages.

## Interpretation limits

This exploratory extension was chosen after viewing earlier text-pilot results. It uses one split and example-selection seed on two familiar public datasets; pretraining exposure cannot be excluded. Equal newly supplied labels do not mean equal total training exposure.

Jev receives the original text, the same examples and a cached proposed class from a separate model. This does not isolate the causal effect of bounded decoding on the same LLM. Compare each pipeline with both the source and Jev alone; count corrected and harmed predictions. Failed source proposals trigger no review call, and failed reviews receive no fallback label.

The four classical models share training-only sparse word/character TF-IDF features and fixed hyperparameters. Full prepared training uses 10,000 SST-2 or 4,886 TREC labels and excludes validation labels. This is not an optimized text leaderboard. XGBoost treats unstored sparse entries as missing; other estimators treat them as zero. Four retained LightGBM wrapper feature-name warnings do not change the fixed vectorizer column ordering.

Intervals use 2000 unstratified group-bootstrap resamples with seed 42 and normalized-text groups; both tests contain 200 distinct groups. Paired intervals condition on the split, selected examples and fitted models, and have no multiple-comparison adjustment. Few-shot-minus-zero-shot contrasts use unequal newly supplied label budgets. Restricted-label likelihoods, native Choice probabilities and classical probabilities are not guaranteed calibrated.

Original Qwen/OpenAI sources retain their historical prompt rendering. New SmolLM2/Granite sources use the explicit fixed-system likelihood recipe. Model differences therefore do not isolate architecture alone. No new LoRA training is included.
