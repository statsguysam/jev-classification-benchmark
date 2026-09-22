# Jev Benchmark Observatory

A static dashboard for the numerical classification experiment: six LLMs at zero-shot and four examples per class, their Jev-reviewed decisions, direct Jev and four native classical estimators. The default page reads the audited 68-condition aggregate snapshot; the earlier broad study is preserved at `historical.html`.

The hosted site is private to the owning account. It makes no model API calls. Source links point to the private benchmark repository.

## Numerical view

Filter the binary Breast Cancer or multiclass Wine task, supplied examples, source LLM and accuracy/macro-F1. Compare the reviewed pipeline with its source or with Jev alone. The paired table gives exact 95% bootstrap intervals, predictions corrected/harmed, and pipeline failures including inherited source failures. Native XGBoost, LightGBM, logistic regression and random forest have a separate matched/full label-budget selector. No cross-dataset pooled score or controlled latency ranking is shown.

The CSV exports the selected paired comparison, with its metric, endpoints and separate source/reviewer failure counts. Filters are retained in the page URL.

## Reproduce locally

From the benchmark repository, generate the aggregate assets:

```bash
python scripts/summarize_expanded_numeric.py --bootstrap-samples 2000
python scripts/build_expanded_numeric_dashboard.py --allow-incomplete
python scripts/build_dashboard_data.py --output dashboard/dist/data.json
```

The numeric builder revalidates predictions, manifests, protocol and accounting. Its default mode refuses incomplete reports; the explicit `--allow-incomplete` flag exports the current billing-paused snapshot with 61/68 completed conditions, null pending scores and visible status. Both modes reject stale reports. Remove that flag to require a completed experiment. The output allowlist excludes raw feature rows, prompts, credentials and local paths. Copy `dashboard/dist/numeric-data.json` into this site's `dist` directory when deploying from its separate checkout.

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory dist
node tests/dashboard.smoke.cjs
node tests/numeric-dashboard.smoke.cjs
```

The historical view retains all 318 broad-study executions (290 distinct conditions). Its measurements do not replace the later numerical study. Both views keep unavailable values distinct from zero and retain failures in accuracy. The numerical experiment uses two familiar public datasets, one split/seed and fixed inference recipes; pretraining exposure cannot be excluded. Full-training ML uses more labels. No new LoRA training is included in the expansion.
