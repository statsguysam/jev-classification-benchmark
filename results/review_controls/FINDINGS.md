# Matched proposal-value controls

**12/12 primary arms complete; 1714/1714 requests saved.** Every incomplete arm remains unscored. This report performs no API calls.

[Audited machine report](COMPARISON.json) · [Frozen protocol](prepared_full/protocol.json)

Each frozen case receives an actual Qwen3 four-shot proposal, an explicit no-proposal slot, and a shuffled proposal. Input, examples, choices and all other prompt wording are identical. The primary contrasts are actual minus no-proposal and actual minus shuffled, separately for each dataset. Existing direct Jev results provide context and are not substituted for this matched no-proposal arm.

| Dataset | Arm | Saved/expected | Status | Accuracy | Balanced accuracy | Macro-F1 | Failures |
|---|---|---:|---|---:|---:|---:|---:|
| breast_cancer | actual | 114/114 | complete | 92.98% | 93.45% | 93.00% | 1 |
| breast_cancer | no_proposal | 114/114 | complete | 92.98% | 92.53% | 92.53% | 0 |
| breast_cancer | shuffled | 114/114 | complete | 92.98% | 92.99% | 93.28% | 2 |
| sst2 | actual | 200/200 | complete | 96.00% | 96.04% | 96.24% | 1 |
| sst2 | no_proposal | 200/200 | complete | 95.50% | 95.55% | 95.74% | 1 |
| sst2 | shuffled | 200/200 | complete | 94.00% | 94.06% | 94.48% | 2 |
| trec | actual | 200/200 | complete | 86.00% | 89.90% | 88.06% | 1 |
| trec | no_proposal | 200/200 | complete | 86.50% | 89.86% | 88.29% | 1 |
| trec | shuffled | 200/200 | complete | 88.00% | 91.33% | 89.68% | 2 |
| wine | actual | 36/36 | complete | 86.11% | 87.70% | 87.54% | 1 |
| wine | no_proposal | 36/36 | complete | 86.11% | 87.70% | 88.64% | 2 |
| wine | shuffled | 36/36 | complete | 88.89% | 90.08% | 89.07% | 0 |

| Dataset | Contrast | Accuracy difference, pp [95% group bootstrap] | Fixed | Harmed |
|---|---|---|---:|---:|
| breast_cancer | actual_minus_no_proposal | +0.00 [-4.39, +4.39] | 3 | 3 |
| breast_cancer | actual_minus_shuffled | +0.00 [-3.51, +3.51] | 2 | 2 |
| sst2 | actual_minus_no_proposal | +0.50 [-1.00, +2.50] | 2 | 1 |
| sst2 | actual_minus_shuffled | +2.00 [+0.00, +4.50] | 5 | 1 |
| trec | actual_minus_no_proposal | -0.50 [-3.00, +1.50] | 2 | 3 |
| trec | actual_minus_shuffled | -2.00 [-4.00, -0.50] | 0 | 4 |
| wine | actual_minus_no_proposal | +0.00 [-13.89, +13.89] | 3 | 3 |
| wine | actual_minus_shuffled | -2.78 [-8.33, +0.00] | 0 | 1 |

**Serving-repeat diagnostic: complete (64/64 pairs available).** Agreement metrics are withheld until every declared repeat and its reference is present.

Among 62 pairs with two valid responses, 62 agree and 0 disagree. Reference-only failures: 1; repeat-only failures: 0; both failed: 1. Both failures are not counted as label agreement. Resolved-model differences and per-dataset counts are in the JSON.

All failures remain incorrect in primary metrics, including unknown transport outcomes; missing costs stay unknown. The repeat sample is predefined and serves a separate descriptive diagnostic. Repeated prompts and primary arms share test cases, so these observations are not independent architecture trials.

These holdouts were already observed. Intervals are exploratory, unadjusted group-bootstrap intervals conditional on the frozen cases, demonstrations, prompts and observed responses; they do not establish generalization to new data or deterministic serving. Full matched arms are required for comparisons, and no partial result is promoted to a headline. No paid request, budget allocation, or historical producer/report modification occurs in this summarizer.
