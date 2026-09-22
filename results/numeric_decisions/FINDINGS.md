# Numerical bounded decisions: measured pilot

This supplement tests Jev on numerical tabular classification, including an LLM proposed-class → Jev review arm and true XGBoost/LightGBM baselines. It is an exploratory two-dataset study, not evidence that bounded outputs guarantee correctness or that Jev improves every LLM.

[Protocol](../../docs/NUMERIC_DECISIONS_PROTOCOL.md) · [All metrics and paired intervals](COMPARISON.json) · [CSV](COMPARISON.csv)

New review conditions complete: **4/4**. Boosting conditions complete: **8/8**.

## Does Jev fix more LLM errors than it introduces?

The exact saved four-per-class LLM predictions are reused as advisory proposals. Jev also sees the original numerical row and the same demonstrations. No new LLM rationale is generated. Source or reviewer failures count as incorrect. Intervals are paired 95% group-bootstrap intervals, conditional on one split and example selection.

| Dataset | Source → Jev | Before | After | Fixed | Correct → wrong label | Correct → failure | Accuracy change, pp [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|
| Breast Cancer | GPT-6 Astra → Jev | 98.2% | 93.9% | 0 | 4 | 1 | -4.39 [-8.77, -0.88] |
| Breast Cancer | Qwen3 4B → Jev | 86.0% | 93.0% | 10 | 2 | 0 | +7.02 [+1.75, +13.16] |
| Wine | GPT-6 Astra → Jev | 97.2% | 94.4% | 0 | 1 | 0 | -2.78 [-8.33, +0.00] |
| Wine | Qwen3 4B → Jev | 80.6% | 88.9% | 3 | 0 | 0 | +8.33 [+0.00, +19.44] |

## Does the LLM proposal add value over Jev alone?

Both arms receive the original numerical row and the same labeled examples. The review arm additionally receives the saved proposed class. These are separate Jev calls, so any observed difference can also include serving/sampling variability.

| Dataset | Review pipeline | Accuracy change vs Jev alone, pp [95% CI] |
|---|---|---|
| Breast Cancer | GPT-6 Astra → Jev | +0.88 [+0.00, +2.63] |
| Breast Cancer | Qwen3 4B → Jev | +0.00 [-3.51, +3.51] |
| Wine | GPT-6 Astra → Jev | +2.78 [-5.56, +11.18] |
| Wine | Qwen3 4B → Jev | -2.78 [-11.11, +5.56] |

## Same small label budget: four records per class

| Dataset | System | Training labels | Test rows | Accuracy | Macro-F1 | Failures |
|---|---|---:|---:|---:|---:|---:|
| Breast Cancer | GPT-6 Astra alone | 8 | 114 | 98.2% | 0.981 | 0 |
| Breast Cancer | GPT-6 Astra → Jev | 8 | 114 | 93.9% | 0.938 | 1 |
| Breast Cancer | Jev alone | 8 | 114 | 93.0% | 0.929 | 1 |
| Breast Cancer | LightGBM | 8 | 114 | 92.1% | 0.915 | 0 |
| Breast Cancer | Logistic regression | 8 | 114 | 94.7% | 0.943 | 0 |
| Breast Cancer | Qwen3 4B alone | 8 | 114 | 86.0% | 0.858 | 0 |
| Breast Cancer | Qwen3 4B → Jev | 8 | 114 | 93.0% | 0.927 | 0 |
| Breast Cancer | Random forest | 8 | 114 | 95.6% | 0.952 | 0 |
| Breast Cancer | XGBoost | 8 | 114 | 88.6% | 0.873 | 0 |
| Wine | GPT-6 Astra alone | 12 | 36 | 97.2% | 0.988 | 1 |
| Wine | GPT-6 Astra → Jev | 12 | 36 | 94.4% | 0.958 | 1 |
| Wine | Jev alone | 12 | 36 | 91.7% | 0.916 | 0 |
| Wine | LightGBM | 12 | 36 | 83.3% | 0.820 | 0 |
| Wine | Logistic regression | 12 | 36 | 97.2% | 0.974 | 0 |
| Wine | Qwen3 4B alone | 12 | 36 | 80.6% | 0.809 | 0 |
| Wine | Qwen3 4B → Jev | 12 | 36 | 88.9% | 0.891 | 0 |
| Wine | Random forest | 12 | 36 | 94.4% | 0.947 | 0 |
| Wine | XGBoost | 12 | 36 | 72.2% | 0.692 | 0 |

## Full-training supervised reference: additional labels

| Dataset | System | Training labels | Test rows | Accuracy | Macro-F1 | Failures |
|---|---|---:|---:|---:|---:|---:|
| Breast Cancer | LightGBM | 341 | 114 | 96.5% | 0.962 | 0 |
| Breast Cancer | Logistic regression | 341 | 114 | 99.1% | 0.991 | 0 |
| Breast Cancer | Random forest | 341 | 114 | 98.2% | 0.981 | 0 |
| Breast Cancer | XGBoost | 341 | 114 | 97.4% | 0.972 | 0 |
| Wine | LightGBM | 106 | 36 | 94.4% | 0.938 | 0 |
| Wine | Logistic regression | 106 | 36 | 97.2% | 0.972 | 0 |
| Wine | Random forest | 106 | 36 | 100.0% | 1.000 | 0 |
| Wine | XGBoost | 106 | 36 | 94.4% | 0.941 | 0 |

## Jev zero-shot reference: no supplied labels

| Dataset | System | Training labels | Test rows | Accuracy | Macro-F1 | Failures |
|---|---|---:|---:|---:|---:|---:|
| Breast Cancer | Jev alone | 0 | 114 | 84.2% | 0.845 | 1 |
| Wine | Jev alone | 0 | 36 | 33.3% | 0.167 | 0 |

## Jev systems versus boosted trees with matched labels

All differences are the named Jev system minus the tree baseline, in accuracy percentage points. A confidence interval crossing zero does not prove equivalence.

| Dataset | Jev system | Tree baseline | Difference [95% paired CI] |
|---|---|---|---|
| Breast Cancer | Jev alone | LightGBM | +0.88 [-4.39, +6.14] |
| Breast Cancer | Jev alone | XGBoost | +4.39 [-0.88, +10.53] |
| Wine | Jev alone | LightGBM | +8.33 [-5.56, +22.22] |
| Wine | Jev alone | XGBoost | +19.44 [+2.78, +36.11] |
| Breast Cancer | GPT-6 Astra → Jev | LightGBM | +1.75 [-3.51, +7.02] |
| Breast Cancer | GPT-6 Astra → Jev | XGBoost | +5.26 [+0.00, +11.40] |
| Breast Cancer | Qwen3 4B → Jev | LightGBM | +0.88 [-5.26, +7.02] |
| Breast Cancer | Qwen3 4B → Jev | XGBoost | +4.39 [-1.75, +10.53] |
| Wine | GPT-6 Astra → Jev | LightGBM | +11.11 [-2.78, +25.00] |
| Wine | GPT-6 Astra → Jev | XGBoost | +22.22 [+5.56, +38.89] |
| Wine | Qwen3 4B → Jev | LightGBM | +5.56 [-8.33, +22.22] |
| Wine | Qwen3 4B → Jev | XGBoost | +16.67 [+0.00, +33.33] |

## Measured incremental cost

The new review made **299 Jev calls** and no new LLM-generation calls. Known API-reported charges total **US$0.047467518** across 298 calls; Calls with unknown reported charges: 1. Conservative review reservations are **US$0.803712000**, bringing cumulative study accounting to **US$19.325827700 / US$20**. Reservations are not an invoice. [Machine-readable cost reconciliation](COSTS.json).

## Limits

Only 114 Breast Cancer and 36 Wine test rows, one split, seed 42, one serialization, fixed untuned tree recipes, and already familiar public datasets. Four labels per class is an extreme low-data setting; it does not characterize normally trained boosting. Full-training trees use additional labels. Pretraining exposure cannot be ruled out. Model architecture and output protocol are not isolated. Jev probabilities may still be miscalibrated on this task.

The review's extra cost is incremental to the source LLM call. Reusing cached proposals avoids new LLM charges in this experiment; a live chain still pays and waits for both stages. Local amortized batch timings and hosted request timings are not controlled speed comparisons.
