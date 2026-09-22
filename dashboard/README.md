# Jev Benchmark Observatory

A static dashboard for the numerical classification experiment: six LLMs at zero-shot and four examples per class, their Jev-reviewed decisions, direct Jev and four native classical estimators. The default page reads the audited 68-condition numerical aggregate; `text.html` applies the same pipeline comparison to a separate 68-condition SST-2/TREC extension. The earlier broad study is preserved at `historical.html`.

The hosted site is private to the owning account. It makes no model API calls. Source links point to the private benchmark repository.

## Numerical view

Filter the binary Breast Cancer or multiclass Wine task, supplied examples, source LLM and accuracy/macro-F1. Compare the reviewed pipeline with its source or with Jev alone. The paired table gives exact 95% bootstrap intervals, predictions corrected/harmed, and pipeline failures including inherited source failures. Native XGBoost, LightGBM, logistic regression and random forest have a separate matched/full label-budget selector. No cross-dataset pooled score or controlled latency ranking is shown.

The CSV exports the selected paired comparison, with its metric, endpoints and separate source/reviewer failure counts. Filters are retained in the page URL.

## Text extension view

[Open the text extension](https://jev-benchmark-observatory.statsguysalim.chatgpt.site/text.html), private to the owning account. SST-2 is binary sentiment and TREC is six-class question type; each has the same 200 frozen test rows across all methods. Six source LLMs at zero/four examples per class are compared alone and after Jev review. Jev sees the original prepared text, the same demonstrations and only the source's cached class proposal. Four direct-Jev references and sixteen classical conditions complete the matrix.

The current aggregate has **44/68 complete conditions**: all twenty-four source runs, four direct-Jev references and sixteen classical runs. Twenty-four Jev reviews remain pending because the provider account has no available credit. No text-review calls have started. The current audited asset controls the displayed status; no partial-checkpoint scores are shown. Pending accuracy, macro-F1, intervals, failure counts and transitions remain unavailable rather than zero. Source models and review conditions stay linked even when their results are pending.

Classical XGBoost, LightGBM, logistic regression and random forest use train-only word/character TF-IDF. Matched training uses 8/24 labels and full-prepared-training references use 10,000/4,886 labels for SST-2/TREC. Full and matched references must remain distinct. XGBoost's unstored sparse entries mean missing; other estimators use zero. There is no validation-label fitting or tuning. The planned 72 contrasts comprise review-minus-source, review-minus-direct-Jev and descriptive few-minus-zero comparisons; 95% intervals are unadjusted and conditional on one split/seed. Bounded output validity is separate from accuracy. See the [protocol](../docs/TEXT_EXTENSION_PROTOCOL.md) and [reproduction guide](../docs/TEXT_EXTENSION_REPRODUCTION.md).

## Reproduce locally

From the benchmark repository, generate the aggregate assets:

```bash
python scripts/summarize_expanded_numeric.py --bootstrap-samples 2000
python scripts/build_expanded_numeric_dashboard.py --allow-incomplete
python scripts/summarize_text_extension.py --bootstrap-samples 2000
python scripts/build_text_extension_dashboard.py --allow-incomplete
python scripts/build_dashboard_data.py --output dashboard/dist/data.json
```

The numeric builder revalidates predictions, manifests, protocol and accounting. Its default mode refuses incomplete reports; the explicit `--allow-incomplete` flag exports the current billing-paused snapshot with 61/68 completed conditions, null pending scores and visible status. Both modes reject stale reports. Remove that flag to require a completed experiment. The output allowlist excludes raw feature rows, prompts, credentials and local paths. Copy `dashboard/dist/numeric-data.json` into this site's `dist` directory when deploying from its separate checkout.

The text builder likewise reaudits the evidence and requires complete conditions by default. Its explicit partial mode writes `dashboard/dist/text-data.json` with all 68 planned rows and 72 contrasts, retaining missing values and source links. Copy that aggregate asset to the separate site checkout when updating its text view. Source prompts, raw dataset text, credentials and local paths are excluded. Rebuilding either JSON asset does not publish the site.

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory dist
node tests/dashboard.smoke.cjs
node tests/numeric-dashboard.smoke.cjs
```

The historical view retains all 318 broad-study executions (290 distinct conditions). Its measurements do not replace the later numerical or text pipeline studies. All views keep unavailable values distinct from zero and retain failures in accuracy. Both extensions use familiar public datasets, one split/seed and fixed inference recipes; pretraining exposure cannot be excluded. Full-training ML uses more labels. No new LoRA training is included in either extension, and there is no pooled cross-domain model ranking.


## Value of review

[The review-value view](https://jev-benchmark-observatory.statsguysalim.chatgpt.site/review.html) adds source / direct-Jev / pipeline metrics, corrected and harmed decisions, and every eligible fixed-coverage curve. It shows 17/24 complete numerical review conditions and keeps the 1,714-request matched follow-up visibly pending. Sequence likelihoods are not calibrated confidence; curves are retrospective simulations, not deployment or dollar-saving results.

Rebuild with `python scripts/build_review_value_dashboard.py`; verify with `node dashboard/tests/review-dashboard.smoke.cjs dashboard`. Only explicit aggregate fields are exported. See the [story](../results/review_value/STORY.md), [sharing figure](../results/review_value/figures/selective_review.png), and [draft post](../results/review_value/LINKEDIN_DRAFT.md). The deployment remains private.
