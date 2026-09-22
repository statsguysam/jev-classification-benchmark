# Matched-label classical analysis

All 180 matched-label runs completed: five datasets, four classical baselines, budgets of 1, 4, and 8 training examples per class, and selection seeds 13, 42, and 87. Each run evaluates the same 200 held-out examples for its dataset; all 36,000 predictions completed without errors.

The table reports macro-F1 mean ± sample standard deviation across the three selection seeds (sample SD uses n−1). These seeds change the labeled training subset and model randomness. They do not create independent test replications: dataset preparation and held-out examples remain fixed. The standard deviation describes training-subset sensitivity and is not a confidence interval or a significance test.

Each budget uses fixed C=1 or alpha=1, with no validation-based tuning. The exact same labeled rows are used by all four models for a dataset, seed, and budget; the 1/class subset is contained in 4/class, which is contained in 8/class. Dataset preprocessing, caveats, and the separate 40-run pilot are documented in [EXECUTION.md](EXECUTION.md).

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python -u scripts/run_classical_matrix.py \
  --data-root data/pilot --output results/matched --budgets 1 4 8 --seeds 13 42 87 --bootstrap-samples 1000
```

Started: 2026-09-22T09:54:00.488860+00:00. Completed: 2026-09-22T10:01:37.158036+00:00.

| Dataset | Model | 1/class macro-F1 | 4/class macro-F1 | 8/class macro-F1 |
|---|---|---:|---:|---:|
| sst2 | majority | 0.3289 ± 0.0000 | 0.3289 ± 0.0000 | 0.3289 ± 0.0000 |
| sst2 | logistic_regression | 0.4285 ± 0.0503 | 0.5057 ± 0.0261 | 0.5244 ± 0.0252 |
| sst2 | linear_svc | 0.4285 ± 0.0503 | 0.5040 ± 0.0223 | 0.5270 ± 0.0262 |
| sst2 | multinomial_nb | 0.4840 ± 0.0575 | 0.5126 ± 0.0275 | 0.5275 ± 0.0202 |
| imdb | majority | 0.3333 ± 0.0000 | 0.3333 ± 0.0000 | 0.3333 ± 0.0000 |
| imdb | logistic_regression | 0.5013 ± 0.0706 | 0.5424 ± 0.0587 | 0.5929 ± 0.0537 |
| imdb | linear_svc | 0.5013 ± 0.0706 | 0.5393 ± 0.0568 | 0.6006 ± 0.0494 |
| imdb | multinomial_nb | 0.4449 ± 0.0902 | 0.5337 ± 0.0486 | 0.5623 ± 0.0371 |
| ag_news | majority | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 |
| ag_news | logistic_regression | 0.3108 ± 0.0426 | 0.4756 ± 0.0031 | 0.5586 ± 0.0525 |
| ag_news | linear_svc | 0.3108 ± 0.0426 | 0.4737 ± 0.0102 | 0.5523 ± 0.0500 |
| ag_news | multinomial_nb | 0.3141 ± 0.0323 | 0.4729 ± 0.0210 | 0.5349 ± 0.0365 |
| trec | majority | 0.0065 ± 0.0000 | 0.0065 ± 0.0000 | 0.0065 ± 0.0000 |
| trec | logistic_regression | 0.2680 ± 0.0939 | 0.4027 ± 0.1732 | 0.5059 ± 0.0696 |
| trec | linear_svc | 0.2614 ± 0.0863 | 0.4062 ± 0.1618 | 0.5066 ± 0.0563 |
| trec | multinomial_nb | 0.2530 ± 0.1065 | 0.3621 ± 0.1880 | 0.4474 ± 0.1081 |
| banking77 | majority | 0.0004 ± 0.0000 | 0.0004 ± 0.0000 | 0.0004 ± 0.0000 |
| banking77 | logistic_regression | 0.2984 ± 0.0459 | 0.5482 ± 0.0241 | 0.6875 ± 0.0185 |
| banking77 | linear_svc | 0.2943 ± 0.0490 | 0.5247 ± 0.0327 | 0.6782 ± 0.0164 |
| banking77 | multinomial_nb | 0.3003 ± 0.0429 | 0.5167 ± 0.0139 | 0.6620 ± 0.0102 |

Data/classical regression tests also passed: `13 passed in 15.07s` using `.venv/bin/python -m pytest tests/test_data.py tests/test_classical.py -q` (with plugin autoload disabled). No benchmark source changes were made for this extension.

Artifact checks passed:

- All 180 run records are complete, and every prediction ID matches its held-out manifest.
- Every class has exactly its requested training budget, with identical rows across models and strictly nested 1/4/8 subsets within each seed.
- All runs use one fixed candidate and no validation labels; LinearSVC exposes no probability distributions.
- All 20 overlapping seed-42, 4/class runs reproduce the earlier pilot class predictions, accuracy, and macro-F1 exactly.
- Majority-baseline ties choose class 0, explaining its constant scores across budgets and seeds.

Banking77 still has only two or three test rows per class, so its macro-F1 estimates are exploratory. Non-monotonic changes as the labeled budget increases can occur on these small training subsets; three seeds do not resolve a final ranking.

Implementation SHA-256 values recorded by these runs: `d5547ccde4224e182653315d81c0a631c789bbe294ad7bcf95ef0cddd25ce608`.

[Per-run matrix summary](matched/classical_matrix_summary.json) · [Aggregated values and run IDs](matched/aggregate_macro_f1.json).
