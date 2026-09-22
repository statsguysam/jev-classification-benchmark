# Jev Benchmark Observatory

A static metrics dashboard for the measured Jev classification study. It reads the sanitized `dist/data.json` snapshot and makes no model API calls.

**[Open the hosted dashboard](https://jev-benchmark-observatory.statsguysalim.chatgpt.site)**. The site is private to the owning account. Deployment provenance is recorded in [DEPLOYMENT.json](DEPLOYMENT.json); its four deployed static files are identical to this directory's `dist` files.

The dashboard includes data type, task, dataset, family, model, method, label-budget, seed, probability-coverage and replication filters; per-dataset charts; a Jev-focused view; run search; confidence intervals; per-class metrics and confusion matrices; probability quality and execution context; whole-study cost accounting; filtered CSV and SVG exports. Filters are preserved in the page URL.

All 318 execution records are retained. The default hides 28 classical replication records and selects seed 42 with zero or four examples per class. The full study contains 290 distinct conditions across eight datasets. Missing measurements stay absent. Recorded failures remain in scores. Text and tabular confidence intervals retain their different bootstrap definitions.

## Local preview

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:8766/`. A web server is required because the page fetches its JSON asset. The site uses native browser controls and SVG; no package installation or build step is needed. Optional Google Fonts fall back to system fonts when unavailable.

## Refresh the measured data

From the benchmark repository, regenerate the sanitized snapshot using `scripts/build_dashboard_data.py`, then copy the generated asset to `dist/data.json`. The builder verifies saved metrics against predictions, attaches audited tabular group intervals and checks source hashes. It excludes raw feature/text rows, prompts, credentials and machine paths.

```bash
python scripts/build_dashboard_data.py --output dashboard/dist/data.json
node dashboard/tests/dashboard.smoke.cjs
```

The dashboard can be served from `dashboard/dist` when copied into the benchmark repository. Source links require access to the private GitHub repository. The Sites deployment is owner-private unless its audience is explicitly changed.

## Verification

The data builder has 14 semantic tests. The browser checks cover combined numeric/multiclass/Jev filters, canonical versus replica counts, Jev views, search, run details, cost coverage and phone-sized layout. Two optional WebMCP tools read the current view or configure filters; both valid and invalid calls were checked against visible state. `node tests/dashboard.smoke.cjs` verifies filters, missing values, failure denominators, CSV search scope/interval columns and chart scores.

Hosted accounting covers the entire study and deliberately does not follow sidebar filters. No controlled speed ranking or pooled cross-dataset accuracy is shown. Intervals condition on the fixed public-data experiment and do not rule out pretraining contamination.
