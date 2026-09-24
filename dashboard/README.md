# Jev Benchmark Observatory

A dashboard for comparing an LLM's original answer, its Jev-reviewed answer, Jev alone and classical classifiers. Six LLMs run zero-shot and with four examples per class. The landing page covers the 68 numerical conditions; `text.html` covers the separate 68-condition SST-2/TREC study. Earlier experiments are available at `historical.html`.

The hosted site is private to the owning account. It makes no model API calls. Source links point to the private benchmark repository.

## Numerical view

Choose Breast Cancer (binary) or Wine (three classes), then filter by examples supplied, source LLM and accuracy/macro-F1. Compare the reviewed pipeline with its source or with Jev alone. XGBoost, LightGBM, logistic regression and random forest have a separate selector for matched or full-training label budgets.

All 68 numerical conditions, including 24 reviews, and all 72 contrasts are audited. The paired table shows the 95% bootstrap interval, corrected and harmed predictions, and pipeline failures, including failures inherited from the source. These exploratory intervals describe the recorded split and are unadjusted across comparisons. The page keeps datasets separate and makes no controlled latency ranking.

The CSV exports the selected paired comparison, with its metric, endpoints and separate source/reviewer failure counts. Filters are retained in the page URL.

## Text extension view

[Open the text extension](https://jev-benchmark-observatory.statsguysalim.chatgpt.site/text.html), private to the owning account. SST-2 is binary sentiment and TREC is six-class question type; each has the same 200 frozen test rows across all methods. Six source LLMs at zero/four examples per class are compared alone and after Jev review. Jev sees the original prepared text, the same demonstrations and only the source's cached class proposal. Four direct-Jev references and sixteen classical conditions complete the matrix.

The [saved text report](../results/text_extension/COMPARISON.json) contains **68 completed conditions and 72 contrasts**, including all twenty-four Jev reviews and 4,800 new review calls. Its 48 review errors count as incorrect in the original scores. Both the local `text-data.json` and private hosted page contain this complete matrix. Missing measurements are shown as unavailable.

Together, the numerical and text matrices contain 136 conditions, including 48 source-to-review conditions. [Final conservative accounting](../results/completion_20260923/COSTS.md) totals US$24.110484775, leaving US$0.889515225 under the US$25 ceiling. These totals include retained reservations and are not a provider invoice.

The four classical classifiers fit word/character TF-IDF on training data only. They use these label budgets:

| Dataset | Matched training | Full prepared training |
|---|---:|---:|
| SST-2 | 8 | 10,000 |
| TREC | 24 | 4,886 |

XGBoost treats unstored sparse entries as missing; the other estimators use zero. There is no validation-label fitting or tuning. The 72 contrasts compare review with source, review with Jev alone, and few-shot with zero-shot. The last comparison changes the label budget. The intervals are unadjusted and describe one split and seed. An allowed class can still be wrong. See the [protocol](../docs/TEXT_EXTENSION_PROTOCOL.md) and [reproduction guide](../docs/TEXT_EXTENSION_REPRODUCTION.md).

## Reproduce locally

From the benchmark repository, generate the aggregate assets:

```bash
python scripts/summarize_expanded_numeric.py --bootstrap-samples 2000
python scripts/build_expanded_numeric_dashboard.py
python scripts/summarize_text_extension.py --bootstrap-samples 2000
python scripts/build_text_extension_dashboard.py
python scripts/build_dashboard_data.py --output dashboard/dist/data.json
```

Run the aggregate audit/export sequence after inference workers have stopped; a changing evidence file must fail verification rather than become a published snapshot. The numeric builder revalidates predictions, manifests, protocol and accounting. Its default mode requires all 68 completed conditions. The explicit `--allow-incomplete` flag supports a visibly unfinished reproduction with null pending scores. Both modes reject stale reports. The output allowlist excludes raw feature rows, prompts, credentials and local paths. Copy `dashboard/dist/numeric-data.json` into this site's `dist` directory when deploying from its separate checkout.

The text builder likewise reaudits the evidence and requires complete conditions by default. Its explicit partial mode writes `dashboard/dist/text-data.json` with all 68 planned rows and 72 contrasts, retaining missing values and source links. Copy that aggregate asset to the separate site checkout when updating its text view. Source prompts, raw dataset text, credentials and local paths are excluded. Rebuilding either JSON asset does not publish the site.

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory dashboard/dist
node dashboard/tests/dashboard.smoke.cjs
node dashboard/tests/numeric-dashboard.smoke.cjs dashboard
node dashboard/tests/text-dashboard.smoke.cjs dashboard
node dashboard/tests/review-dashboard.smoke.cjs dashboard
```

The historical view retains all 318 broad-study executions (290 distinct conditions). These earlier measurements remain separate from the numerical and text pipeline studies. All pages distinguish unavailable values from zero and retain failures in accuracy. Both extensions use familiar public datasets, one split and seed, and fixed inference recipes. Pretraining exposure cannot be excluded, full-training ML uses more labels, and neither extension adds LoRA training. The dashboard does not pool the datasets into one model ranking.


## Value of review

[The review-value view](https://jev-benchmark-observatory.statsguysalim.chatgpt.site/review.html) shows source, Jev-alone and pipeline metrics for all 24 numerical review conditions, along with corrected and harmed decisions. It includes all five eligible selective-review curves. The linked sensitivity appendix covers all sixteen local conditions with probability scores, including eleven additions that predicted a constant class. Sequence likelihoods are not calibrated confidence, and the curves are retrospective simulations rather than measured deployment or cost savings.

The page also contains the [completed matched controls](../results/review_controls/FINDINGS.md): 1,714 requests, 12 primary arms and eight paired contrasts. Separate recovery views preserve the original scores and repeat diagnostic. [Historical recovery](../results/completion_20260923/RECOVERY_FINDINGS.md) resolved 65 of 66 failed paid requests in 71 calls and supplied one dependent first call after its source recovered. [Control recovery](../results/completion_20260923/CONTROL_RECOVERY_FINDINGS.md) resolved all fifteen failed control/repeat requests in fifteen calls.

Both the local asset and private hosted page contain these completed results. Partial conditions are never scored. The [output-validation notes](../docs/OUTPUT_VALIDATION_NOTES.md) explain the distinction between an accepted wrong class and a rejected response. Original accuracy counts both as incorrect, while the breakdown keeps them separate; recovery results do not overwrite either category.

Use the audited report/export order in the [completion reproduction guide](../docs/COMPLETION_REPRODUCTION.md), then rebuild with `python scripts/build_review_value_dashboard.py` and verify with `node dashboard/tests/review-dashboard.smoke.cjs dashboard`. Regenerate the sensitivity appendix after this asset because it records the asset hash. Only explicit aggregate fields are exported. See the [story](../results/review_value/STORY.md), [sharing figure](../results/review_value/figures/selective_review.png), and [draft post](../results/review_value/LINKEDIN_DRAFT.md). The deployment remains private.
