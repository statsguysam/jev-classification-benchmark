# Numeric evidence review · 23 September 2026

**The strongest numeric story is to assess review against both its source and direct Jev, with correction/harm counts and matched-label classical references.** All 68 conditions, 24 review conditions and 72 contrasts are complete. This historical memo audits the completed first-attempt numerical snapshot as it stood on 23 September, before text/control completion and operational recovery. Its dated checks and original report pins below are retained. For current complete evidence, read the [full story](../review_value/STORY.md), [first-pass review across both studies](FIRST_PASS_EVIDENCE_REVIEW.md), [recovery findings](RECOVERY_FINDINGS.md) and [current sensitivity appendix](../review_value_sensitivity/FINDINGS.md).

Paths below use zero-based JSON array indices. `C` = [numeric_expansion/COMPARISON.json](../numeric_expansion/COMPARISON.json), `A` = [review_value/ANALYSIS.json](../review_value/ANALYSIS.json), `S` = [review_value_sensitivity/ANALYSIS.json](../review_value_sensitivity/ANALYSIS.json). Scope: 114 Breast Cancer and 36 Wine test rows, repeatedly reused across model conditions (`A.unique_test_rows`).

## Strongest checked findings

| Observation | Paired evidence and qualification | Exact JSON paths |
|---|---|---|
| Wine, Astra, zero-shot: 36/36 → 16/36 (100% → 44.4%). | 0 fixed, 20 harmed, all 20 wrong-label replacements; no source/review failures. Review-minus-source −55.56 percentage points, exploratory 95% interval [−72.22, −38.89]. Direct Jev was 12/36, so beating direct Jev still concealed substantial damage to the source. | `A.conditions[22].metrics`, `.review_minus_base`, `.failures`; `C.comparisons[66].paired_bootstrap.metrics.accuracy` |
| Wine, Qwen2.5 0.5B, four-shot: 12/36 → 33/36 (33.3% → 91.7%). | 21 fixed, 0 harmed; source contrast +58.33 points [41.67, 75.00]. All 36 reviewed labels equal the separately called direct-Jev labels. Direct Jev also achieved 33/36. This is zero observed incremental accuracy over direct Jev on these cases, not general equivalence. | `A.conditions[13].metrics`, `.review_minus_base`, `.review_minus_direct.changed_predictions` (=0); `C.comparisons[38:40]` |
| Breast Cancer, Qwen2.5 0.5B, zero-shot: 43/114 → 97/114. | 54 fixed, 0 harmed, but direct Jev was already 96/114. Review-minus-direct +0.88 points [−4.39, 6.14]. This prevents presenting a large source gain as demonstrated value over direct classification. | `A.conditions[0]`; `C.comparisons[0:2]` |
| Native-feature models remain strong with the matched label budget. | Breast Cancer random forest: 109/114 (95.6%) with 8 labels; Wine logistic regression: 35/36 (97.2%) with 12 labels. Wine random forest is 34/36 (94.4%). These are fixed recipes; full-training references use more labels and must be distinguished. | `C.runs[28]`, `C.runs[60]`, `C.runs[62]`: `.accuracy`, `.n_test`, `.train_labels`, `.train_per_class` |

I independently re-read aligned raw predictions/test manifests for the two headline Wine cases and the two named classical examples; counts and the 36/36 direct-label agreement match. All 204 run/prediction/test-manifest hashes in `C.runs[*].artifact_sha256` match the saved files.

Counterexamples remain important: zero-shot Wine Qwen3 fixes 12 cases but harms 13 (net −1; `A.conditions[14].review_minus_base`); Luna fixes 7 but harms 10 (net −3; condition 20). Astra review reduces correct counts in all four numerical conditions, with no corrections (`A.conditions[10,11,22,23].review_minus_base`). This is a description of four reused-case conditions, not four independent replications. Breast Cancer Qwen3 zero-shot is a favorable comparison against direct Jev: +10 correct, +8.77 points, interval [0.00, 17.54] (`C.comparisons[7]`). No universal reviewer benefit follows.

## Selective review: report all five primary curves and the broader sensitivity

At the predetermined 50% point, the entire primary inventory is:

| Dataset / source / examples per class | Selected review requests | Accuracy at 50% | Random expectation | Full review accuracy | Exact A condition |
|---|---:|---:|---:|---:|---:|
| Breast Cancer / Qwen2.5 / 4 | 57/114 | 70.2% | 77.6% | 93.9% | 1 |
| Breast Cancer / Qwen3 / 0 | 57/114 | 76.3% | 77.2% | 93.0% | 2 |
| Breast Cancer / Qwen3 / 4 | 57/114 | 93.0% | 89.5% | 93.0% | 3 |
| Breast Cancer / Granite / 4 | 57/114 | 86.8% | 76.8% | 92.1% | 7 |
| Wine / Qwen3 / 4 | 18/36 | 88.9% | 84.7% | 88.9% | 15 |

For each indexed condition, use `.selective_review.points[3].metrics.micro_accuracy`, `.random_matched_rate.micro_accuracy`, `.required_inferences.review_api_requests`; the full-review endpoint is `.points[5]`. The two Qwen3 four-shot half/full matches survive completion. Their review requests have no observed failures; the reduction is in simulated required requests from cached outcomes, not an experiment that actually executed a cheaper gate. The source still needs an inference for every row. No dollar, latency or deployable-threshold saving is established.

`S.counters` records 16 local probability-bearing conditions: the five primary conditions plus 11 constant-label conditions, all 11 with varying maximum scores, yielding 96 curve points. Among the additions, accuracy exceeds random selection at all four intermediate coverages in 2 conditions, falls below throughout in 4, is mixed in 3, and ties throughout in 2 (`S.added_condition_accuracy_vs_random_patterns`). The primary constant-label exclusion was post-hoc; the broader appendix prevents conflating constant labels with useless probability ranks. Both analyses are exploratory. All six coverages, balanced accuracy and macro-F1 remain in the machine reports; random macro-F1 Monte Carlo errors are not sampling confidence intervals.

## Failure, uncertainty and measurement limits

`C.runs[*]` contains one source inference failure, two direct-Jev failures, and ten review-pipeline failures. Nine review failures followed actual Jev calls; one Wine Astra four-shot row inherited its failed source proposal and received no Jev call (`A.conditions[23].observed_skipped_source_failure_rows`). Six historical billing failures remain in the completed Breast Cancer SmolLM2 four-shot score. First-attempt failures count as incorrect; the subsequently completed retries retain originals and report [operational recovery separately](RECOVERY_FINDINGS.md). A completed condition is not necessarily failure-free.

The saved paired intervals use 2,000 group-bootstrap resamples, seed 42, with paired rows and 114/36 distinct groups. They condition on this fixed split, examples, prompts, fitted models and saved responses; intervals are exploratory and unadjusted across many comparisons (`C.comparisons[*].paired_bootstrap`, `.interpretation`). They do not measure training/serving variation. No gate confidence intervals are reported, and a zero-spanning interval does not establish equivalence.

Local source scores are normalized likelihoods of numeric class ID plus EOS, not calibrated probabilities of being correct; 11/16 local conditions predict a single label. Source renderings differ across model/provider families. Reviewer prompts omit source identity, confidence and rationale. Across 60 overlapping source pairs, 2,331 same-label cases have identical prompt hashes; 77/2,308 both-valid comparisons disagree, with 23 comparisons containing a failure (`A.same_label_prompt_pairs[*]`). These historical, dependent comparisons span serving times and are not designed replicates or an anchoring estimate. Familiar public datasets, one split, arbitrary zero-shot Wine cultivar IDs, possible pretraining exposure and mixed runtimes further constrain generalization.

## Historical accounting and pre-execution figure check

The expansion-only ledger has 1,500 new calls, US$0.124087194 known reported charges and 8 unknown-charge calls; all US$4.032000000 reserved remains retained (`C.costs`, equal to [COSTS.json](../numeric_expansion/COSTS.json)). Its cumulative subtotal is US$23.357827700, or US$23.360515700 including the separate health-probe reserve, before text completion/retries/controls. This is not the changing global total, an invoice or a source-cost comparison.

Static review of `scripts/plot_review_controls.py` found no unsupported result or misleading selection/axis encoding: it refuses partial/mismatched reports, requires all 12 primary arms and 8 contrasts, shows all four datasets, uses a zero-reference for paired differences, and labels conditional, unadjusted intervals and first-attempt failures. At this memo's original audit, the script had not been run and no control result was asserted. The [completed control report](../review_controls/FINDINGS.md) and [full story](../review_value/STORY.md) now supply the later evidence; the scoped accounting above is not the [final cumulative total](COSTS.md).

Suggested skeptical framing: “A reviewer can repair a weak source while adding nothing over direct classification, and it can also damage a strong source. This pilot makes that trade-off visible by retaining errors and comparing the chain with both stages alone and with matched-label classical baselines.”

## Original dated snapshot pins

These hashes identify the reports checked for this historical memo; they are not a claim that every currently regenerated report has the same bytes. In particular, the sensitivity report records the dashboard asset hash and is regenerated after final dashboard updates. Use its current provenance for the latest snapshot. The numerical observations above remain first-attempt findings.

- `results/numeric_expansion/COMPARISON.json`: `b6b33421c51ee1cf9809659589f0724bb2ea883d9d1bb094b9f973d5ea654256`
- `results/review_value/ANALYSIS.json`: `8f7b1dd14045750d08975f5539ca108d614dbaa7f7d5c4e0ad7c7c632e72bc48`
- `results/review_value_sensitivity/ANALYSIS.json`: `c4564fc6fb55fac1cd58507b5166fbe3523a68ab12e006056fa64d1882cbed6f`
