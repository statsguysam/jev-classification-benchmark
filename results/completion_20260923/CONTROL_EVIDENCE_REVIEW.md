# Independent evidence review: matched proposal controls

This audit covers the complete original 1,714-call control study (12 primary arms; 550 cases × 3 arms, plus 64 repeat calls) and its finalized 15-call failure-recovery overlay. All 12 original arms, eight paired contrasts and 64 repeat pairs are retained. No partial conditions or selected winners are reported. Original outcomes remain the primary result.

**Finding:** providing the cached Qwen3 class did not show a consistent accuracy advantage over the same reviewer prompt with the proposal slot set to “not provided.” Original accuracy differences were 0.00 pp on Breast Cancer, 0.00 on Wine, +0.50 on SST-2 and −0.50 on TREC. Every paired accuracy interval included zero. After the separate failure-only recovery, those differences were +0.88, −2.78, +0.50 and −0.50 pp; all four intervals still included zero. This is an absence of a consistent observed advantage in this fixed pilot, not evidence of equivalence or proof that proposals cannot help.

## What was checked

The audit rebuilt the original aggregate report in memory from immutable saved responses and the frozen request bundle, using the published pure metric/bootstrap helpers. It matched every primary report field produced by pure aggregation, including all metrics, interval endpoints, transitions, failure lists and repeat diagnostics. It separately counted correct answers and correction/wrong-label/failure transitions from row-level labels. Request order and metadata, exact prompt SHA256 values, source prediction pins, input-text hashes, manifest truths, and proposal/donor mapping were checked. Across each case, the prompt prefix and suffix were identical and only the proposal-slot value differed. No service or live whole-repository collector was invoked.

The secondary overlay was independently reconstructed from all 15 finalized attempts: every attempt targeted an original failure, matched its original prompt and prediction identity, passed the frozen output contract, and was the first additional attempt for its row. Original successful predictions were unchanged. Full denominators, accuracy, balanced accuracy, macro-F1 and all primary paired intervals were recomputed for both views. Both finalized control ledger hash chains and anchors reconciled to their saved counts and retained amounts. All 31 scoped artifact hashes were unchanged on reread.

## Every primary arm

All percentages below use the entire test fold, including original failures as incorrect. “Recovered” is the separate earliest-valid failure-only overlay. BA = balanced accuracy; F1 = macro-F1. Original arm accuracy intervals and all full-precision metric fields remain in `results/review_controls/COMPARISON.json` → `runs[]`. Recovery counterparts are in `results/completion_20260923/CONTROL_RECOVERY_COMPARISON.json` → `runs[]`.

| Dataset | Arm | Original correct / N | Original failures | Original BA / F1 (%) | Recovered correct / N | Recovered BA / F1 (%) |
|---|---|---:|---:|---:|---:|---:|
| Breast Cancer | Actual Qwen3 proposal | 106/114 | 1 | 93.45 / 93.00 | 107/114 | 94.61 / 93.60 |
| Breast Cancer | No proposal | 106/114 | 0 | 92.53 / 92.53 | 106/114 | 92.53 / 92.53 |
| Breast Cancer | Shuffled proposal | 106/114 | 2 | 92.99 / 93.28 | 108/114 | 95.32 / 94.49 |
| Wine | Actual Qwen3 proposal | 31/36 | 1 | 87.70 / 87.54 | 32/36 | 90.08 / 89.07 |
| Wine | No proposal | 31/36 | 2 | 87.70 / 88.64 | 33/36 | 92.86 / 91.64 |
| Wine | Shuffled proposal | 32/36 | 0 | 90.08 / 89.07 | 32/36 | 90.08 / 89.07 |
| SST-2 | Actual Qwen3 proposal | 192/200 | 1 | 96.04 / 96.24 | 193/200 | 96.55 / 96.50 |
| SST-2 | No proposal | 191/200 | 1 | 95.55 / 95.74 | 192/200 | 96.06 / 96.00 |
| SST-2 | Shuffled proposal | 188/200 | 2 | 94.06 / 94.48 | 190/200 | 95.08 / 95.00 |
| TREC | Actual Qwen3 proposal | 172/200 | 1 | 89.90 / 88.06 | 173/200 | 90.20 / 88.27 |
| TREC | No proposal | 173/200 | 1 | 89.86 / 88.29 | 174/200 | 90.17 / 88.50 |
| TREC | Shuffled proposal | 176/200 | 2 | 91.33 / 89.68 | 176/200 | 91.33 / 89.39 |

## Every primary accuracy contrast

Each row is **actual proposal minus comparator**. Differences and 95% intervals are in percentage points; intervals are unadjusted paired group percentile bootstraps (2,000 draws, seed 42). “Fixed / wrong / failure harms” means comparator incorrect → actual correct, comparator correct → actual wrong valid label, and comparator correct → actual failed response. A “fixed” row can therefore include a comparator failure. These are arm comparisons, not source-model review transitions.

| Dataset | Comparator | Original Δ [95% CI], pp | Original fixed / wrong / failure harms | Recovered Δ [95% CI], pp | Recovered fixed / wrong / failure harms |
|---|---|---|---:|---|---:|
| Breast Cancer | No proposal | +0.00 [-4.39, +4.39] | 3 / 2 / 1 | +0.88 [-2.63, +4.39] | 3 / 2 / 0 |
| Breast Cancer | Shuffled proposal | +0.00 [-3.51, +3.51] | 2 / 1 / 1 | -0.88 [-2.63, +0.00] | 0 / 1 / 0 |
| Wine | No proposal | +0.00 [-13.89, +13.89] | 3 / 2 / 1 | -2.78 [-11.11, +5.56] | 1 / 2 / 0 |
| Wine | Shuffled proposal | -2.78 [-8.33, +0.00] | 0 / 0 / 1 | +0.00 [+0.00, +0.00] | 0 / 0 / 0 |
| SST-2 | No proposal | +0.50 [-1.00, +2.50] | 2 / 0 / 1 | +0.50 [+0.00, +1.50] | 1 / 0 / 0 |
| SST-2 | Shuffled proposal | +2.00 [+0.00, +4.50] | 5 / 0 / 1 | +1.50 [+0.00, +3.50] | 3 / 0 / 0 |
| TREC | No proposal | -0.50 [-3.00, +1.50] | 2 / 2 / 1 | -0.50 [-2.50, +1.00] | 1 / 2 / 0 |
| TREC | Shuffled proposal | -2.00 [-4.00, -0.50] | 0 / 3 / 1 | -1.50 [-3.50, +0.00] | 0 / 3 / 0 |

All full-precision endpoints are at `comparisons[] → paired_group_bootstrap.metrics.accuracy` in the original report and `comparisons[] → {first_attempt,recovered}.paired_group_bootstrap.metrics.accuracy` in the recovery report. The original TREC actual-minus-shuffled interval excluded zero, but the secondary recovered interval reaches zero (−3.50 to 0.00 pp). Neither result justifies claiming a general benefit from shuffled proposals.

| Dataset | Comparator | Original Δ macro-F1 [95% CI], ×100 | Recovered Δ macro-F1 [95% CI], ×100 |
|---|---|---|---|
| Breast Cancer | No proposal | +0.47 [-3.81, +4.93] | +1.07 [-2.80, +5.22] |
| Breast Cancer | Shuffled proposal | -0.28 [-3.17, +2.43] | -0.89 [-2.80, +0.00] |
| Wine | No proposal | -1.10 [-12.76, +10.58] | -2.56 [-11.92, +6.99] |
| Wine | Shuffled proposal | -1.54 [-5.69, +0.00] | +0.00 [+0.00, +0.00] |
| SST-2 | No proposal | +0.50 [-0.53, +1.86] | +0.50 [+0.00, +1.52] |
| SST-2 | Shuffled proposal | +1.76 [+0.00, +3.80] | +1.50 [+0.00, +3.50] |
| TREC | No proposal | -0.23 [-1.74, +1.24] | -0.22 [-1.53, +1.13] |
| TREC | Shuffled proposal | -1.61 [-3.23, -0.48] | -1.11 [-2.54, +0.00] |

The same paired-bootstrap paths under `macro_f1` hold the exact F1 endpoints. The original BC accuracy tie concealed a +0.47-point macro-F1 difference; the Wine accuracy tie concealed −1.10 points. Accuracy equality does not imply identical predictions or equal class-wise performance.

## Failures, shuffle and repeat checks

The original 15 failed requests comprise 14 primary-arm calls and one repeat call: 10 HTTP 529 errors, two connection/timeout errors and three rejected probability-sum outputs. These are protocol/service failures, not observed wrong class decisions. Parsed choices/vectors were discarded for validation failures, so their latent class correctness and any rounding explanation cannot be inferred. All 15 received one separately recorded retry and all returned valid outputs; this means response recovery, not necessarily a correct answer. Two recovered TREC shuffled predictions were still wrong. The original first-attempt scores and failure records are unchanged. See `saved_failure_records` in the primary report and `recovery_counts` / `new_call_outcomes` in the recovery report.

The original control ledger contains 1,714 reservations and 1,699 settlements; its retained amount is $0.192398804. The recovery ledger contains 15 reservations and no settlements; it retains the full $0.040320000 upper bound. These are scoped accounting amounts, not model-reported invoice totals or cost-saving estimates.

The deterministic shuffle preserved proposal-class frequencies and had no donor-row fixed points. It still retained the same proposed class for 64/114 Breast Cancer, 12/36 Wine, 86/200 SST-2 and 54/200 TREC cases. “Shuffled” therefore does not mean “deliberately wrong.” There was one predeclared label-blind permutation per dataset, not repeated random permutations. Fields: frozen protocol `datasets[dataset].{donor_mapping,actual_proposal_counts,shuffled_proposal_counts,unchanged_proposal_labels}`.

The 64 predeclared byte-identical no-proposal repeat pairs yielded 62 pairs with two valid responses, and all 62 agreed on the chosen label. One pair had only the reference fail, and one had both responses fail; no pair had only the repeat fail. Thus valid-pair agreement is 62/62; agreement with two valid responses across every declared pair is 62/64. Both-valid counts were 16 Breast Cancer, 15 Wine, 15 SST-2 and 16 TREC. Every valid pair used the same resolved model alias, with none unknown. This is a small descriptive serving diagnostic, not 64/64 success, statistical independence or proof of deterministic serving. The recovery report correctly retains the original repeat diagnostic without substituting recovered outputs. Exact fields: `serving_repeat_diagnostic` and `first_attempt_serving_repeat_diagnostic`.

## Interpretation and proposed post wording

“I then held the reviewer prompt and examples fixed and varied only the proposal slot: actual Qwen3 class, no class, or a shuffled class. Across the four fixed holdouts, the actual proposal showed no consistent accuracy advantage over no proposal; the four paired intervals included zero, including in a separate failure-recovery analysis.”

This supports the practical question “What does each stage add over both stages alone?” It does not identify the reviewer’s internal reasoning or prove anchoring. The no-proposal arm retains the review wording and an explicit “not provided” slot; it is the matched control, distinct from historical Jev-alone prompts. Only one source (Qwen3-4B-Instruct-2507), one reviewer route, four examples per class and one prompt template were tested here. The 550 public cases and four holdouts had already been examined in the earlier pilot. Arms and contrasts share cases; they are not independent replications. The CIs condition on the observed rows, serialization, demonstrations, model fits and served responses; they do not measure training variability or future serving variability. Wine has only 36 rows and its original actual-versus-none interval spans ±13.89 pp. Local class-ID-plus-EOS scoring and possible public-dataset pretraining exposure continue to limit broader model-capability claims.

No strong general claim that the two-model pipeline is better, that proposals are useless, that shuffled labels help, or that architecture is causally responsible follows. The strongest story remains the combination of selected rescue/harm examples, every first-pass condition, and this controlled check of the proposal’s added value.

## Scoped SHA256 receipt

Only the finalized files below were pinned and reread. This memo writes no raw data, producer, protocol, prediction, ledger, or primary/secondary report. Metric sources are the two report paths in this receipt; underlying row evidence is the original and recovery JSONL plus frozen requests and source manifests.

```text
d1ee99e24e4d2850f5c83e8afe6b22b341b46f8d8ed57c8323c4c38d611b2f06  results/colab/sst2__Qwen3-4B-Instruct-2507__5823bb43da99/predictions.jsonl
52356557a98032673cf65b2576d8e97834cbade77e02370745128d5320e5d085  results/colab/sst2__Qwen3-4B-Instruct-2507__5823bb43da99/run.json
9964be2fb433e67602760b7385499df94718ece310cf44d5895a9942ffb92ba0  results/colab/sst2__Qwen3-4B-Instruct-2507__5823bb43da99/test_manifest.json
075d2c324b92542fe85089be9f310e979568d0574887bfcc59e6c0ef971adf80  results/colab/trec__Qwen3-4B-Instruct-2507__c0b1cc9b3d0f/predictions.jsonl
7c1a7106a7dd43fa0856a8cefdb2986a0590052d9391ca84586966869e4d0362  results/colab/trec__Qwen3-4B-Instruct-2507__c0b1cc9b3d0f/run.json
95ea0e430c5c11744de599ad7d264384f004be0c39d767329aa96a32e22109fb  results/colab/trec__Qwen3-4B-Instruct-2507__c0b1cc9b3d0f/test_manifest.json
452389af3ffb627a33eb7ffb7ede6b030df0fa59902c5a9ff6bb0c105bb5056c  results/completion_20260923/CONTROL_RECOVERY_COMPARISON.json
f7a1ce87fa27c5dedc94cf1f024070605049b544af585749ba0abd1a6cd1fea4  results/completion_20260923/control_retries/attempts.jsonl
b5e53e46c92fded484db81b7692b708e7eb6d7d07e4990501f4334cc0a05e8a0  results/completion_20260923/control_retries/budget.jsonl
0d53b5554b506df0c294f5c981fa258901407ed021c381c080df441aa2b9345e  results/completion_20260923/control_retries/budget.jsonl.lock
0951f39a9f480e68fc27822ba5e2b696bff27c574b65dfc07c06f91c34599779  results/completion_20260923/control_retries/plan.json
22a858ef8ad0146ad1cc5daba8e58d86f2df9b65d08b9198f3cef6f457e9dba8  results/completion_20260923/control_retries/run.json
b7eedd2a8c45a50edfd2393a367e4dea20db626af3a8bba5c583d5e76af3078d  results/review_controls/COMPARISON.json
cf4b29351ce7d387f6a50b04da17a4035b67884c09efd4a93a50e28167cb2e8e  results/review_controls/execution/budget.jsonl
bb462a59959eeddbf8b41f023a8c8fc9adc64576ce77ed670d94bad1c35618a1  results/review_controls/execution/budget.jsonl.lock
0175c3a99498ad02cf6990d633b8331611d19e7169d757e7c2e6078b8dc7f945  results/review_controls/execution/predictions.jsonl
2cc5336ba2189e3d11f7ef91e3892a5209644445e4a87776f5f117320cc6d7be  results/review_controls/execution/run.json
786c68ac439b400310ef4a2fcc77369bb8508e981d6d7372a6395a9d814f8ef4  results/review_controls/prepared_full/manifest.json
c1c94d336ad8700c1bb39b306069ca426ac22e392f588d5f967aeb03d2783992  results/review_controls/prepared_full/protocol.json
ae24a2a13a952bddac30f591a5d9b34559a96119c355e73e26e3c5fb5a327d60  results/review_controls/prepared_full/requests.jsonl
e5c1b86c06a40de1724e263df2849e46ff24c8957df163d1424b4008a1c58917  results/tabular/colab/breast_cancer__Qwen3-4B-Instruct-2507__cd3baa813fe3/predictions.jsonl
a5d870f25fb6c13fd66bd686e77b5ca60cf5c7f1c43991b21297a325eea2146e  results/tabular/colab/breast_cancer__Qwen3-4B-Instruct-2507__cd3baa813fe3/run.json
d3445c1ee998cbebc299755c9c4284295416ad8a1f4165cfad80ddf85b987685  results/tabular/colab/breast_cancer__Qwen3-4B-Instruct-2507__cd3baa813fe3/test_manifest.json
b9b11638881c31a9853035c1803a4daf61cbf82f83c136ba57494463f16607d4  results/tabular/colab/wine__Qwen3-4B-Instruct-2507__cea3e01ea8d0/predictions.jsonl
91378022c1e86e476bd4f3b59a301d411d804d50bed3ca8ccc7ba744c25bfb43  results/tabular/colab/wine__Qwen3-4B-Instruct-2507__cea3e01ea8d0/run.json
edbfa3c757d7142e13a35ba40b53c590587e3e591508d70161dd433f881a5c76  results/tabular/colab/wine__Qwen3-4B-Instruct-2507__cea3e01ea8d0/test_manifest.json
d048bb4f17a5fa023ed54ee81d7152da1dc4b55814f9d343ef8e27f802afd455  scripts/run_review_controls.py
081514afa07e5c3296bb4bcdcbe0b743980317cdf6a3a1decb6da871ead020e9  scripts/summarize_control_failed_retries.py
e656c4793bf26670cdbb6c946d6ed6f25788efa0d4591c74f1c7936dca4bb5be  scripts/summarize_review_controls.py
422e973822e97e928e2128f604dabe84fa3675e57fdcb86305a75473c55b7157  scripts/summarize_tabular.py
551146253f47dedc9354e6e1b1ed81fdab57c07b0c77b8471a31eb4e2b6528c9  src/jevbench/metrics.py
```
