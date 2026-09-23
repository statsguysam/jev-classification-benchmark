# Jev Benchmark Observatory

A static dashboard for the numerical classification experiment: six LLMs at zero-shot and four examples per class, their Jev-reviewed decisions, direct Jev and four native classical estimators. The default page reads the audited 68-condition numerical aggregate; `text.html` applies the same pipeline comparison to a separate 68-condition SST-2/TREC extension. The earlier broad study is preserved at `historical.html`.

The hosted site is private to the owning account. It makes no model API calls. Source links point to the private benchmark repository.

## Numerical view

Filter the binary Breast Cancer or multiclass Wine task, supplied examples, source LLM and accuracy/macro-F1. Compare the reviewed pipeline with its source or with Jev alone. All 68 numerical conditions, including 24 reviews, and all 72 paired contrasts are audited. The paired table displays the saved exploratory 95% bootstrap intervals, predictions corrected/harmed, and pipeline failures including inherited source failures. Intervals are conditional on the recorded split and unadjusted across comparisons. Native XGBoost, LightGBM, logistic regression and random forest have a separate matched/full label-budget selector. No cross-dataset pooled score or controlled latency ranking is shown.

The CSV exports the selected paired comparison, with its metric, endpoints and separate source/reviewer failure counts. Filters are retained in the page URL.

## Text extension view

[Open the text extension](https://jev-benchmark-observatory.statsguysalim.chatgpt.site/text.html), private to the owning account. SST-2 is binary sentiment and TREC is six-class question type; each has the same 200 frozen test rows across all methods. Six source LLMs at zero/four examples per class are compared alone and after Jev review. Jev sees the original prepared text, the same demonstrations and only the source's cached class proposal. Four direct-Jev references and sixteen classical conditions complete the matrix.

The [saved text report](../results/text_extension/COMPARISON.json) is complete: **68/68 conditions and 72/72 contrasts**, including all twenty-four Jev reviews and 4,800 new review calls. Its 48 review errors remain incorrect in the full first-attempt denominators. The rebuilt local `text-data.json` contains the complete matrix; the private hosted text view has been refreshed with that same snapshot. Missing values remain unavailable rather than zero. Both original matrices together contain 136 conditions, including 48 source-to-review conditions. [Final conservative accounting](../results/completion_20260923/COSTS.md) totals US$24.110484775, leaving US$0.889515225 under the US$25 ceiling; it is not a provider invoice.

Classical XGBoost, LightGBM, logistic regression and random forest use train-only word/character TF-IDF. Matched training uses 8/24 labels and full-prepared-training references use 10,000/4,886 labels for SST-2/TREC. Full and matched references must remain distinct. XGBoost's unstored sparse entries mean missing; other estimators use zero. There is no validation-label fitting or tuning. The 72 audited contrasts comprise review-minus-source, review-minus-direct-Jev and descriptive few-minus-zero comparisons; 95% intervals are unadjusted and conditional on one split/seed. Bounded output validity is separate from accuracy. See the [protocol](../docs/TEXT_EXTENSION_PROTOCOL.md) and [reproduction guide](../docs/TEXT_EXTENSION_REPRODUCTION.md).

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

The historical view retains all 318 broad-study executions (290 distinct conditions). Its measurements do not replace the later numerical or text pipeline studies. All views keep unavailable values distinct from zero and retain failures in accuracy. Both extensions use familiar public datasets, one split/seed and fixed inference recipes; pretraining exposure cannot be excluded. Full-training ML uses more labels. No new LoRA training is included in either extension, and there is no pooled cross-domain model ranking.


## Value of review

[The review-value view](https://jev-benchmark-observatory.statsguysalim.chatgpt.site/review.html) adds source / direct-Jev / pipeline metrics, corrected and harmed decisions, and every eligible fixed-coverage curve. It shows all 24 complete numerical review conditions. All five eligible primary curves are included; the linked sensitivity appendix covers all sixteen local probability-bearing conditions, including eleven constant-label additions. The local asset includes [completed matched controls](../results/review_controls/FINDINGS.md): 1,714 requests, 12 primary arms and eight paired contrasts. [Historical recovery](../results/completion_20260923/RECOVERY_FINDINGS.md) resolved 65/66 failed paid requests in 71 new calls and supplied one dependent first call after its source recovered. [Post-control recovery](../results/completion_20260923/CONTROL_RECOVERY_FINDINGS.md) resolved 15/15 failed control/repeat requests in 15 calls. These are separate views; original scores and the original repeat diagnostic stay intact. The private hosted review page now serves these completed assets. Unavailable metrics remain distinct from zero, and partial arms are never scored. Sequence likelihoods are not calibrated confidence; curves are retrospective simulations, not deployment or dollar-saving results. [Output-validation notes](../docs/OUTPUT_VALIDATION_NOTES.md) distinguish an accepted wrong class from a rejected response. First-attempt accuracy counts both as incorrect while preserving their separate categories; recovery results cannot overwrite those original outcomes.

Use the audited report/export order in the [completion reproduction guide](../docs/COMPLETION_REPRODUCTION.md), then rebuild with `python scripts/build_review_value_dashboard.py` and verify with `node dashboard/tests/review-dashboard.smoke.cjs dashboard`. Regenerate the sensitivity appendix after this asset because it records the asset hash. Only explicit aggregate fields are exported. See the [story](../results/review_value/STORY.md), [sharing figure](../results/review_value/figures/selective_review.png), and [draft post](../results/review_value/LINKEDIN_DRAFT.md). The deployment remains private.
