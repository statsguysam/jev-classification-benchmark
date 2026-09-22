# Executed classical pilot

This report records 40 completed local CPU runs: five datasets, four classical baselines, and two training tracks, all at seed 42. Every run evaluates the same 200 held-out rows for its dataset. All 8,000 predictions completed without errors. These are pilot results; the broader Jev, frontier-model, open-model, and LoRA comparison is not established by these runs.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python -u scripts/run_classical_matrix.py \
  --data-root data/pilot --output results/pilot --seeds 42 --budgets 4 --include-full --bootstrap-samples 1000
```

Started: 2026-09-22T09:45:21.271690+00:00. Completed: 2026-09-22T09:51:01.981837+00:00.

All text inputs use a 2,000-character prefix. Source revisions, file hashes, overlap-removal audits, selected training IDs, exact test IDs, predictions, hyperparameter trials, environment versions, and 1,000-replicate bootstrap intervals are saved in each run directory. The source corpus files remain in ignored `data/` directories.

**Training tracks:** `4/class` fits four labeled training examples per class, shared exactly across all models; C=1 or alpha=1 is fixed a priori and no validation labels are used. `Full-prepared` uses every available row in the prepared pilot training split, with validation macro-F1 selecting among three predeclared hyperparameters for each non-dummy estimator. This is not full-corpus training for datasets capped at 10,000 rows.

| Dataset | Classes | 4/class train | Full-prepared train | Validation for full-prepared | Test |
|---|---:|---:|---:|---:|---:|
| sst2 | 2 | 8 | 10000 | 1000 | 200 |
| imdb | 2 | 8 | 10000 | 1000 | 200 |
| ag_news | 4 | 16 | 10000 | 1000 | 200 |
| trec | 6 | 24 | 4886 | 545 | 200 |
| banking77 | 77 | 308 | 8995 | 1000 | 200 |

| Dataset | Training track | Model | Accuracy | Macro-F1 | Run artifact |
|---|---|---|---:|---:|---|
| sst2 | 4/class | majority | 0.4900 | 0.3289 | [record](pilot/sst2__majority__face6e86e516/run.json) |
| sst2 | 4/class | logistic_regression | 0.4950 | 0.4826 | [record](pilot/sst2__logistic_regression__81d2ff14c2be/run.json) |
| sst2 | 4/class | linear_svc | 0.5000 | 0.4869 | [record](pilot/sst2__linear_svc__3e7e9ec774d5/run.json) |
| sst2 | 4/class | multinomial_nb | 0.5400 | 0.5388 | [record](pilot/sst2__multinomial_nb__a094038879f9/run.json) |
| sst2 | Full-prepared | majority | 0.5100 | 0.3377 | [record](pilot/sst2__majority__58d1470456af/run.json) |
| sst2 | Full-prepared | logistic_regression | 0.7950 | 0.7949 | [record](pilot/sst2__logistic_regression__e2833fa6fdfe/run.json) |
| sst2 | Full-prepared | linear_svc | 0.7950 | 0.7949 | [record](pilot/sst2__linear_svc__2f0742f75b09/run.json) |
| sst2 | Full-prepared | multinomial_nb | 0.8450 | 0.8443 | [record](pilot/sst2__multinomial_nb__36d0a632e5d2/run.json) |
| imdb | 4/class | majority | 0.5000 | 0.3333 | [record](pilot/imdb__majority__562de0d0de73/run.json) |
| imdb | 4/class | logistic_regression | 0.4950 | 0.4950 | [record](pilot/imdb__logistic_regression__1ef646fb8e67/run.json) |
| imdb | 4/class | linear_svc | 0.4950 | 0.4950 | [record](pilot/imdb__linear_svc__2ad929b3cc6a/run.json) |
| imdb | 4/class | multinomial_nb | 0.5400 | 0.5370 | [record](pilot/imdb__multinomial_nb__0adf93da38dc/run.json) |
| imdb | Full-prepared | majority | 0.5000 | 0.3333 | [record](pilot/imdb__majority__dc4e8ab26aa0/run.json) |
| imdb | Full-prepared | logistic_regression | 0.9050 | 0.9050 | [record](pilot/imdb__logistic_regression__94bf6d77405e/run.json) |
| imdb | Full-prepared | linear_svc | 0.9100 | 0.9100 | [record](pilot/imdb__linear_svc__030c7a7bcda5/run.json) |
| imdb | Full-prepared | multinomial_nb | 0.8450 | 0.8450 | [record](pilot/imdb__multinomial_nb__5a5cb96d35ef/run.json) |
| ag_news | 4/class | majority | 0.2500 | 0.1000 | [record](pilot/ag_news__majority__a8a0d0ea7ac8/run.json) |
| ag_news | 4/class | logistic_regression | 0.4850 | 0.4782 | [record](pilot/ag_news__logistic_regression__f3e487f9ab2a/run.json) |
| ag_news | 4/class | linear_svc | 0.4800 | 0.4732 | [record](pilot/ag_news__linear_svc__bb553fae9f66/run.json) |
| ag_news | 4/class | multinomial_nb | 0.4700 | 0.4640 | [record](pilot/ag_news__multinomial_nb__2822ae8d78ea/run.json) |
| ag_news | Full-prepared | majority | 0.2500 | 0.1000 | [record](pilot/ag_news__majority__52d1c07fc801/run.json) |
| ag_news | Full-prepared | logistic_regression | 0.9100 | 0.9094 | [record](pilot/ag_news__logistic_regression__138af83bc1e3/run.json) |
| ag_news | Full-prepared | linear_svc | 0.9050 | 0.9044 | [record](pilot/ag_news__linear_svc__a23e5c995947/run.json) |
| ag_news | Full-prepared | multinomial_nb | 0.9150 | 0.9146 | [record](pilot/ag_news__multinomial_nb__c27fb667c2c5/run.json) |
| trec | 4/class | majority | 0.0200 | 0.0065 | [record](pilot/trec__majority__d3cfb0a0827d/run.json) |
| trec | 4/class | logistic_regression | 0.5000 | 0.4857 | [record](pilot/trec__logistic_regression__8bc76aab4183/run.json) |
| trec | 4/class | linear_svc | 0.4950 | 0.4757 | [record](pilot/trec__linear_svc__d2b4ae115ac0/run.json) |
| trec | 4/class | multinomial_nb | 0.5000 | 0.4857 | [record](pilot/trec__multinomial_nb__db3012a1e2fb/run.json) |
| trec | Full-prepared | majority | 0.1900 | 0.0532 | [record](pilot/trec__majority__f28a22e53edd/run.json) |
| trec | Full-prepared | logistic_regression | 0.8700 | 0.8679 | [record](pilot/trec__logistic_regression__8d9deee58454/run.json) |
| trec | Full-prepared | linear_svc | 0.9000 | 0.8980 | [record](pilot/trec__linear_svc__6829a061a4ce/run.json) |
| trec | Full-prepared | multinomial_nb | 0.7750 | 0.7914 | [record](pilot/trec__multinomial_nb__5d3313ad7372/run.json) |
| banking77 | 4/class | majority | 0.0150 | 0.0004 | [record](pilot/banking77__majority__17d71238e242/run.json) |
| banking77 | 4/class | logistic_regression | 0.5600 | 0.5336 | [record](pilot/banking77__logistic_regression__9f8fca344d2e/run.json) |
| banking77 | 4/class | linear_svc | 0.5450 | 0.5083 | [record](pilot/banking77__linear_svc__6b1940bc8028/run.json) |
| banking77 | 4/class | multinomial_nb | 0.5300 | 0.5016 | [record](pilot/banking77__multinomial_nb__ce01e834b93c/run.json) |
| banking77 | Full-prepared | majority | 0.0150 | 0.0004 | [record](pilot/banking77__majority__3f1cdb7677fd/run.json) |
| banking77 | Full-prepared | logistic_regression | 0.9200 | 0.9159 | [record](pilot/banking77__logistic_regression__475793a0cfcb/run.json) |
| banking77 | Full-prepared | linear_svc | 0.9250 | 0.9238 | [record](pilot/banking77__linear_svc__3764b058c93c/run.json) |
| banking77 | Full-prepared | multinomial_nb | 0.8850 | 0.8709 | [record](pilot/banking77__multinomial_nb__a1d4030505c7/run.json) |

Interpretation and checks:

- One seed and 200 held-out examples per dataset are insufficient for final model rankings. Banking77 has only two or three held-out examples per class in this pilot, so its class-level estimates are especially uncertain. Bootstrap intervals are exploratory and conditional on these prepared splits.
- The two training tracks use very different amounts of supervision; compare model families within a track. Full-prepared also spends validation labels on hyperparameter selection.
- The majority dummy chooses the lowest class ID when the training counts tie. Balanced 4/class training therefore predicts class 0, including the rare ABBR class in TREC; its low TREC score is expected.
- Native logistic-regression and naive-Bayes probabilities are uncalibrated. LinearSVC returns no probabilities, and its artifacts have zero probability coverage.
- Classical latency is amortized batch inference including TF-IDF transformation; it must not be compared directly to individual hosted-request latency.
- Artifact audit passed: all 40 runs are complete, prediction IDs match held-out manifests, no prediction errors occurred, training IDs are identical across the four models within each dataset/track, and fixed-budget runs used no validation labels.

All 40 runs recorded implementation SHA-256 `d5547ccde4224e182653315d81c0a631c789bbe294ad7bcf95ef0cddd25ce608`. Python 3.12.4; scikit-learn 1.9.1.

[Machine-readable matrix summary](pilot/classical_matrix_summary.json).
