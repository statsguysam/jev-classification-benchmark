# Numerical boosting supplement

This is an exploratory supplement added after earlier test results were viewed. It is not an independent or preregistered confirmation. No earlier run, prepared split, prediction, or frozen benchmark implementation was modified.

Only Breast Cancer Wisconsin (30 numerical measurements; binary) and Wine (13 numerical measurements; three classes) are included. Titanic and all text datasets are excluded. This supplement evaluates real XGBoost 3.4.1 and LightGBM 4.6.0 on native numeric columns; it does not measure an LLM + Jev composition.

| Dataset | Model | Training labels | Test rows | Accuracy [95% CI] | Macro-F1 [95% CI] |
|---|---|---:|---:|---|---|
| breast_cancer | xgboost | 8 (4/class) | 114 | 88.60% [82.46, 93.86] | 0.8725 [0.8006, 0.9337] |
| breast_cancer | lightgbm | 8 (4/class) | 114 | 92.11% [86.84, 96.49] | 0.9147 [0.8579, 0.9627] |
| breast_cancer | xgboost | 341 (full train) | 114 | 97.37% [93.86, 100.00] | 0.9719 [0.9360, 1.0000] |
| breast_cancer | lightgbm | 341 (full train) | 114 | 96.49% [92.98, 99.12] | 0.9619 [0.9210, 0.9909] |
| wine | xgboost | 12 (4/class) | 36 | 72.22% [58.33, 86.11] | 0.6923 [0.5085, 0.8384] |
| wine | lightgbm | 12 (4/class) | 36 | 83.33% [69.44, 94.44] | 0.8205 [0.6632, 0.9377] |
| wine | xgboost | 106 (full train) | 36 | 94.44% [86.11, 100.00] | 0.9410 [0.8472, 1.0000] |
| wine | lightgbm | 106 (full train) | 36 | 94.44% [86.11, 100.00] | 0.9381 [0.8413, 1.0000] |

All 600 predictions had valid labels and complete native probability distributions. Probabilities are uncalibrated. Accuracy, macro-F1, balanced accuracy, log loss, Brier score, ECE, per-class scores, and confusion matrices are saved in each run. Batch timing describes local prediction plus preprocessing; it is not interchangeable with individual hosted API latency.

## What is held constant

- Seed 42 and the original prepared test rows: 114 for Breast Cancer and 36 for Wine.
- The matched track uses exactly the same ordered four-per-class demonstration IDs as the saved Jev few-shot runs: 8 Breast Cancer labels and 12 Wine labels.
- Full training uses only the original training split: 341 and 106 labels. It is a separately labeled unequal-data reference, not a matched-label comparison.
- Native column order follows the prepared schema. Median imputation is fitted only on selected training rows; no scaling is applied. Validation labels, validation feature statistics, early stopping, and test-based parameter selection are excluded.
- Both algorithms use 100 trees, learning rate 0.05, maximum depth 3, L2 regularization 1, seed 42, and one CPU thread. Small leaf constraints permit fitting the 8/12-label track. All parameters, version pins, resolved booster settings, row IDs, input hashes, and source hashes are saved.

## Uncertainty and limits

Intervals use 2,000 unstratified percentile bootstrap replicates over whole groups of identical serialized numerical records, reusing the existing group-bootstrap implementation. They are conditional on this fixed split and training selection; they do not quantify variability across dataset splits, prompt choices, or training seeds. Separate confidence intervals do not establish a significant difference between methods.

These are two small, well-known datasets and one fixed configuration per algorithm. The full-training scores benefit from more task labels. Pretrained models have external pretraining that is not matched by the task-label budget. Performance on these examples is not evidence of a general Jev advantage or a general boosting advantage. The supplement was added after prior outcomes were visible, and this status is retained in configuration and every run.

## Reproduce and inspect

From the repository root:

```sh
.venv/bin/python -m pip install xgboost==3.4.1 lightgbm==4.6.0
.venv/bin/python -m pytest tests/test_numeric_boosting.py -q
.venv/bin/python scripts/run_numeric_boosting.py
```

For a fresh execution, provide a new `--output` directory; the default directory verifies and reuses completed artifacts. Input data must be prepared using the existing tabular preparation workflow.

- [Fixed experiment configuration](../../configs/numeric_boosting.json)
- [Supplement runner](../../scripts/run_numeric_boosting.py)
- [Protocol saved before supplementary fits](boosting/protocol.json)
- [Machine-readable summary with intervals](boosting/summary.json)

Fourteen focused tests passed with the actual libraries. Checks cover exact matched training IDs, binary and multiclass class-probability order, train-only imputation including all-missing columns, invariance to validation/test label changes, and categorical/sidecar rejection. A separate read-only audit recomputed all eight score records and intervals, checked artifact/source hashes, and verified exact equality of test manifests and matched training IDs with inherited Jev runs.

Constructor parameters were checked against the official [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier), [XGBoost parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html), and [LightGBM 4.6.0 API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html).
