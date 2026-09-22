# Classical text extension

All 16 planned conditions completed on the frozen 200-row SST-2 and 200-row TREC test sets: 3,200 predictions, zero failed predictions, and valid probabilities for every row. `summary.json` contains metrics and per-run hashes. `AUDIT.json` records an independent reread of exact data, provenance, training IDs, feature matrices, prediction alignment, and recomputed metrics.

| Dataset | Algorithm | Four labels per class | Full prepared training |
|---|---|---:|---:|
| SST-2 | XGBoost | 44.5% | 69.0% |
| SST-2 | LightGBM | 47.5% | 67.5% |
| SST-2 | Logistic regression | 49.5% | 79.0% |
| SST-2 | Random forest | 53.0% | 76.0% |
| TREC | XGBoost | 35.0% | 70.5% |
| TREC | LightGBM | 37.5% | 72.5% |
| TREC | Logistic regression | 50.0% | 86.5% |
| TREC | Random forest | 49.0% | 81.0% |

Matched training uses the exact seed-42 prompting demonstration IDs: eight SST-2 labels and 24 TREC labels. Full prepared training means 10,000 SST-2 rows and 4,886 TREC rows. It excludes validation labels and is a separately labeled reference with more supervised training data. This is not the entire original SST-2 source training set.

Every estimator uses the same sparse word-plus-character TF-IDF representation. Vocabulary and IDF are fit exclusively on its selected training rows. No validation examples enter fitting, no test results select hyperparameters, and there is one fixed configuration per algorithm. Source and configuration hashes were saved before fitting. Prior text-pilot results had already been seen, so this extension is exploratory rather than a preregistered confirmatory study. The fixed tree configurations are not an optimized text leaderboard.

The four models share exact train and test feature-matrix hashes within each dataset/training budget. Native XGBoost treats unstored sparse entries as missing, while the other estimators interpret sparse absence as zero; this model-specific sparse behavior is recorded in each run. Predictions use the maximum native probability in frozen class order, with the lowest class index on ties. Probabilities are not calibrated.

Runtime audit: LightGBM emitted one sklearn feature-name warning per condition because its wrapper assigns feature names during fitting and receives unnamed sparse CSR input at prediction. The same fitted vectorizer supplies both matrices, and their dimensions and column ordering are unchanged. These four warnings are saved rather than discarded. There were no convergence warnings, invalid probabilities, or inference failures.

Bootstrap metadata clarification: the shared immutable bootstrap helper labels its groups as raw serialized-text hashes, whereas this extension supplies normalized-text hashes. Both frozen test sets contain 200 distinct groups under either definition. Their grouping, estimates and interval distribution therefore agree. The original source and run records are preserved, with the descriptive correction recorded in `AUDIT.json`. Confidence intervals are conditional on this split, example selection and fitted models, and are not adjusted for multiple comparisons.

Reproduction from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_text_classical_extension.py
```

The runner requires the pinned versions in `configs/text_classical_extension.json` and the frozen `data/pilot/{sst2,trec}` files. An unchanged existing output is verified and reused. Changed source/configuration/data requires a new output directory so historical evidence remains intact. Thirty isolated tests cover binary/six-class execution, train-only vocabularies, exact training budgets, sparse-only operations, validation independence, probability alignment, corruption rejection and explicit failed-row preservation.
