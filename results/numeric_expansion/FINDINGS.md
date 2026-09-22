# Expanded numerical classification and Jev review

**61/68 conditions are complete and audited.** Base LLMs: 24/24; reviews: 17/24; Jev direct: 4/4; native references: 16/16. Pending conditions have no score.

This is an exploratory expansion chosen after earlier results were observed. It compares six LLMs on two frozen numerical datasets at zero-shot and four examples per class, then asks Jev to review each saved proposed label with the same labeled examples. Bounded outputs do not guarantee accurate decisions.

[Protocol](../../docs/EXPANDED_NUMERIC_PROTOCOL.md) · [All metrics and paired intervals](COMPARISON.json) · [CSV](COMPARISON.csv)

New local runs complete: 8/8. New review runs complete: 13/20. Historical reviews reused without additional calls: 4/4.

## Breast Cancer: zero-shot

Direct Jev reference accuracy: **84.2%**. Source and review use the same 0 newly supplied labels.

| LLM | Base accuracy | After Jev | Fixed | Correct → wrong label | Correct → failure | Review − base, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 37.7% | 85.1% | 54 | 0 | 0 | +47.37 [+38.60, +56.14] | +0.88 [-4.39, +6.14] |
| Qwen3 4B | 61.4% | 93.0% | 38 | 2 | 0 | +31.58 [+21.93, +40.35] | +8.77 [+0.00, +17.54] |
| SmolLM2 1.7B | 37.7% | 86.8% | 57 | 1 | 0 | +49.12 [+39.47, +57.89] | +2.63 [-3.51, +8.77] |
| Granite 3.3 2B | 37.7% | pending | — | — | — | pending | pending |
| GPT-5.6 Luna | 75.4% | 87.7% | 15 | 1 | 0 | +12.28 [+6.14, +18.42] | +3.51 [-1.75, +8.77] |
| GPT-6 Astra | 99.1% | 95.6% | 0 | 4 | 0 | -3.51 [-7.02, -0.86] | +11.40 [+3.51, +19.30] |

## Breast Cancer: four examples per class

Direct Jev reference accuracy: **93.0%**. Source and review use the same 8 newly supplied labels.

| LLM | Base accuracy | After Jev | Fixed | Correct → wrong label | Correct → failure | Review − base, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 61.4% | 93.9% | 41 | 4 | 0 | +32.46 [+23.68, +42.11] | +0.88 [+0.00, +2.63] |
| Qwen3 4B | 86.0% | 93.0% | 10 | 2 | 0 | +7.02 [+1.75, +13.16] | +0.00 [-3.51, +3.51] |
| SmolLM2 1.7B | 37.7% | pending | — | — | — | pending | pending |
| Granite 3.3 2B | 61.4% | pending | — | — | — | pending | pending |
| GPT-5.6 Luna | 91.2% | 92.1% | 3 | 1 | 1 | +0.88 [-2.63, +5.26] | -0.88 [-4.39, +2.63] |
| GPT-6 Astra | 98.2% | 93.9% | 0 | 4 | 1 | -4.39 [-8.77, -0.88] | +0.88 [+0.00, +2.63] |

## Wine: zero-shot

Direct Jev reference accuracy: **33.3%**. Source and review use the same 0 newly supplied labels.

| LLM | Base accuracy | After Jev | Fixed | Correct → wrong label | Correct → failure | Review − base, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 33.3% | 33.3% | 0 | 0 | 0 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| Qwen3 4B | 38.9% | 36.1% | 12 | 13 | 0 | -2.78 [-30.56, +25.00] | +2.78 [+0.00, +8.33] |
| SmolLM2 1.7B | 33.3% | pending | — | — | — | pending | pending |
| Granite 3.3 2B | 27.8% | pending | — | — | — | pending | pending |
| GPT-5.6 Luna | 47.2% | 38.9% | 7 | 10 | 0 | -8.33 [-30.56, +13.89] | +5.56 [-5.56, +16.67] |
| GPT-6 Astra | 100.0% | 44.4% | 0 | 20 | 0 | -55.56 [-72.22, -38.89] | +11.11 [+2.78, +22.22] |

## Wine: four examples per class

Direct Jev reference accuracy: **91.7%**. Source and review use the same 12 newly supplied labels.

| LLM | Base accuracy | After Jev | Fixed | Correct → wrong label | Correct → failure | Review − base, pp [95% CI] | Review − Jev alone, pp [95% CI] |
|---|---:|---:|---:|---:|---:|---|---|
| Qwen2.5 0.5B | 33.3% | 91.7% | 21 | 0 | 0 | +58.33 [+41.67, +75.00] | +0.00 [+0.00, +0.00] |
| Qwen3 4B | 80.6% | 88.9% | 3 | 0 | 0 | +8.33 [+0.00, +19.44] | -2.78 [-11.11, +5.56] |
| SmolLM2 1.7B | 33.3% | pending | — | — | — | pending | pending |
| Granite 3.3 2B | 33.3% | pending | — | — | — | pending | pending |
| GPT-5.6 Luna | 88.9% | 91.7% | 2 | 1 | 0 | +2.78 [-5.56, +11.11] | +0.00 [-8.33, +8.33] |
| GPT-6 Astra | 97.2% | 94.4% | 0 | 1 | 0 | -2.78 [-8.33, +0.00] | +2.78 [-5.56, +11.18] |

## Native references: matched four examples per class

| Dataset | Model | Training labels | Test rows | Accuracy | Macro-F1 | Failures |
|---|---|---:|---:|---:|---:|---:|
| Breast Cancer | Logistic regression | 8 | 114 | 94.7% | 0.9435 | 0 |
| Breast Cancer | Random forest | 8 | 114 | 95.6% | 0.9521 | 0 |
| Breast Cancer | XGBoost | 8 | 114 | 88.6% | 0.8725 | 0 |
| Breast Cancer | LightGBM | 8 | 114 | 92.1% | 0.9147 | 0 |
| Wine | Logistic regression | 12 | 36 | 97.2% | 0.9740 | 0 |
| Wine | Random forest | 12 | 36 | 94.4% | 0.9475 | 0 |
| Wine | XGBoost | 12 | 36 | 72.2% | 0.6923 | 0 |
| Wine | LightGBM | 12 | 36 | 83.3% | 0.8205 | 0 |

## Native references: full training, additional labels

| Dataset | Model | Training labels | Test rows | Accuracy | Macro-F1 | Failures |
|---|---|---:|---:|---:|---:|---:|
| Breast Cancer | Logistic regression | 341 | 114 | 99.1% | 0.9907 | 0 |
| Breast Cancer | Random forest | 341 | 114 | 98.2% | 0.9813 | 0 |
| Breast Cancer | XGBoost | 341 | 114 | 97.4% | 0.9719 | 0 |
| Breast Cancer | LightGBM | 341 | 114 | 96.5% | 0.9619 | 0 |
| Wine | Logistic regression | 106 | 36 | 97.2% | 0.9718 | 0 |
| Wine | Random forest | 106 | 36 | 100.0% | 1.0000 | 0 |
| Wine | XGBoost | 106 | 36 | 94.4% | 0.9410 | 0 |
| Wine | LightGBM | 106 | 36 | 94.4% | 0.9381 | 0 |

## Incremental review cost

New review spending is reconciled separately in [COSTS.json](COSTS.json), including partial or interrupted requests. Four historical reviews are reused and remain accounted in earlier spending. Known API charges and conservative reservations are different quantities; missing charge records are not treated as free calls.

## Interpretation and limits

All source and review failures count as incorrect, including an invalid source proposal that prevents a review call. Correction counts separate newly wrong labels from review failures. A positive net correction count is an observed change on these test rows; intervals are paired group-bootstrap intervals, not a guarantee of improvement elsewhere.

All intervals use 2000 resamples and seed 42, condition on the fixed split and fitted models, and are exploratory without adjustment for multiple comparisons. The JSON also contains few-shot-minus-zero-shot contrasts labeled as comparisons with unequal new-label budgets. A confidence interval that includes zero does not establish equivalence.

Original Qwen/OpenAI proposals are reused from their original provider/model prompt rendering. New SmolLM2 and Granite runs use an explicit fixed system message with thinking disabled; this is not a controlled same-prompt architecture ablation. The numeric serialization and selected examples stay fixed. Class-probability scores from restricted-label likelihoods, native Choice and classical estimators are not interchangeable or guaranteed calibrated.

Only 114 Breast Cancer and 36 Wine test rows, one split/seed, and familiar public datasets are covered. Pretraining exposure cannot be excluded. Wine cultivar IDs have arbitrary class meaning without examples: zero-shot primarily probes pretrained familiarity and inference from this serialization, not supervised learnability. Even perfect Wine accuracy does not establish general numerical-classification ability or a causal benefit from bounded outputs. Full-training native baselines use extra labels. This expansion adds no LoRA training.

Review prompts include the original numerical row, matching-shot examples and a cached proposed class, without a newly generated rationale. The review-versus-Jev-alone contrast includes both the extra proposal/prompt wording and separate serving variability. A deployed chain pays and waits for both source and review; this experiment reuses existing proposals where available. Mixed MPS/CUDA and hosted latency are not directly comparable.

Only fully audited rows receive scores. The matrix remains incomplete until every requested condition finishes; no aggregate headline is inferred from a partial subset.
