# Text classification and Jev review

**44/68 conditions are complete and audited.** LLM alone: 24/24; LLM → Jev: 0/24; Jev alone: 4/4; classical references: 16/16.

SST-2 has two sentiment classes; TREC has six question categories. Each uses the same frozen 200 held-out rows across every arm. Zero-shot supplies no labels; few-shot supplies exactly four examples per class (8 SST-2 and 24 TREC labels), with identical ordered examples at the source and reviewer stages.

Missing or partial conditions have no score, confidence interval or inferred improvement. Cached LLM labels and direct Jev measurements are reused after full-content and matched-training audits; original cross-environment manifest hashes are preserved.

[Full metrics and paired contrasts](COMPARISON.json) · [Protocol](protocol.json) · [Classical evidence](classical/README.md)

**No LLM → Jev text-review effect has been measured yet.** Completed direct-Jev and classical scores cannot answer whether Jev improves an LLM proposal. Review calls await funded provider credit and the guarded execution stage.

## SST-2 sentiment · zero-shot

Jev alone: **93.5% accuracy** on 200 rows.

| LLM | LLM alone | LLM → Jev | Fixed / harmed | Review − LLM, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 50.0% | pending | — | pending | pending |
| Qwen3 4B | 91.5% | pending | — | pending | pending |
| SmolLM2 1.7B | 49.0% | pending | — | pending | pending |
| Granite 3.3 2B | 80.0% | pending | — | pending | pending |
| GPT-5.6 Luna | 92.5% | pending | — | pending | pending |
| GPT-6 Astra | 97.0% | pending | — | pending | pending |

## SST-2 sentiment · four examples per class

Jev alone: **96.5% accuracy** on 200 rows.

| LLM | LLM alone | LLM → Jev | Fixed / harmed | Review − LLM, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 85.0% | pending | — | pending | pending |
| Qwen3 4B | 95.5% | pending | — | pending | pending |
| SmolLM2 1.7B | 70.0% | pending | — | pending | pending |
| Granite 3.3 2B | 94.0% | pending | — | pending | pending |
| GPT-5.6 Luna | 95.5% | pending | — | pending | pending |
| GPT-6 Astra | 97.5% | pending | — | pending | pending |

## TREC question categories · zero-shot

Jev alone: **33.5% accuracy** on 200 rows.

| LLM | LLM alone | LLM → Jev | Fixed / harmed | Review − LLM, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 15.5% | pending | — | pending | pending |
| Qwen3 4B | 52.5% | pending | — | pending | pending |
| SmolLM2 1.7B | 2.0% | pending | — | pending | pending |
| Granite 3.3 2B | 46.0% | pending | — | pending | pending |
| GPT-5.6 Luna | 52.5% | pending | — | pending | pending |
| GPT-6 Astra | 97.0% | pending | — | pending | pending |

## TREC question categories · four examples per class

Jev alone: **85.5% accuracy** on 200 rows.

| LLM | LLM alone | LLM → Jev | Fixed / harmed | Review − LLM, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 17.5% | pending | — | pending | pending |
| Qwen3 4B | 83.0% | pending | — | pending | pending |
| SmolLM2 1.7B | 27.5% | pending | — | pending | pending |
| Granite 3.3 2B | 63.5% | pending | — | pending | pending |
| GPT-5.6 Luna | 85.0% | pending | — | pending | pending |
| GPT-6 Astra | 97.0% | pending | — | pending | pending |

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

New Jev text-review requests: **0**. Known reported cost: **$0**; unknown-cost calls: **0**. Cumulative conservative accounting, including earlier work: **$22.279939700 / $25.00**. The new text stage is capped at $1.60 and preserves the remaining numerical-review allocation.

Known provider charges and conservative reservations are different quantities. Local compute is unpriced; historical hosted source generation is already included in earlier accounting. A deployed chain pays for both stages.

## Interpretation limits

This exploratory extension was chosen after viewing earlier text-pilot results. It uses one split and example-selection seed on two familiar public datasets; pretraining exposure cannot be excluded. Equal newly supplied labels do not mean equal total training exposure.

Jev receives the original text, the same examples and a cached proposed class from a separate model. This does not isolate the causal effect of bounded decoding on the same LLM. Compare each pipeline with both the source and Jev alone; count corrected and harmed predictions. Failed source proposals trigger no review call, and failed reviews receive no fallback label.

The four classical models share training-only sparse word/character TF-IDF features and fixed hyperparameters. Full prepared training uses 10,000 SST-2 or 4,886 TREC labels and excludes validation labels. This is not an optimized text leaderboard. XGBoost treats unstored sparse entries as missing; other estimators treat them as zero. Four retained LightGBM wrapper feature-name warnings do not change the fixed vectorizer column ordering.

Intervals use 2000 unstratified group-bootstrap resamples with seed 42 and normalized-text groups; both tests contain 200 distinct groups. Paired intervals condition on the split, selected examples and fitted models, and have no multiple-comparison adjustment. Few-shot-minus-zero-shot contrasts use unequal newly supplied label budgets. Restricted-label likelihoods, native Choice probabilities and classical probabilities are not guaranteed calibrated.

Original Qwen/OpenAI sources retain their historical prompt rendering. New SmolLM2/Granite sources use the explicit fixed-system likelihood recipe. Model differences therefore do not isolate architecture alone. No new LoRA training is included.
