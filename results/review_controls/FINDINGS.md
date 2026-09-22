# Matched proposal-value controls

**0/12 primary arms complete; 0/1714 requests saved.** Every incomplete arm remains unscored. This report performs no API calls.

[Audited machine report](COMPARISON.json) · [Frozen protocol](prepared_full/protocol.json)

Each frozen case receives an actual Qwen3 four-shot proposal, an explicit no-proposal slot, and a shuffled proposal. Input, examples, choices and all other prompt wording are identical. The primary contrasts are actual minus no-proposal and actual minus shuffled, separately for each dataset. Existing direct Jev results provide context and are not substituted for this matched no-proposal arm.

| Dataset | Arm | Saved/expected | Status | Accuracy | Balanced accuracy | Macro-F1 | Failures |
|---|---|---:|---|---:|---:|---:|---:|
| breast_cancer | actual | 0/114 | pending | pending | pending | pending | pending |
| breast_cancer | no_proposal | 0/114 | pending | pending | pending | pending | pending |
| breast_cancer | shuffled | 0/114 | pending | pending | pending | pending | pending |
| sst2 | actual | 0/200 | pending | pending | pending | pending | pending |
| sst2 | no_proposal | 0/200 | pending | pending | pending | pending | pending |
| sst2 | shuffled | 0/200 | pending | pending | pending | pending | pending |
| trec | actual | 0/200 | pending | pending | pending | pending | pending |
| trec | no_proposal | 0/200 | pending | pending | pending | pending | pending |
| trec | shuffled | 0/200 | pending | pending | pending | pending | pending |
| wine | actual | 0/36 | pending | pending | pending | pending | pending |
| wine | no_proposal | 0/36 | pending | pending | pending | pending | pending |
| wine | shuffled | 0/36 | pending | pending | pending | pending | pending |

| Dataset | Contrast | Accuracy difference, pp [95% group bootstrap] | Fixed | Harmed |
|---|---|---|---:|---:|
| breast_cancer | actual_minus_no_proposal | pending | pending | pending |
| breast_cancer | actual_minus_shuffled | pending | pending | pending |
| sst2 | actual_minus_no_proposal | pending | pending | pending |
| sst2 | actual_minus_shuffled | pending | pending | pending |
| trec | actual_minus_no_proposal | pending | pending | pending |
| trec | actual_minus_shuffled | pending | pending | pending |
| wine | actual_minus_no_proposal | pending | pending | pending |
| wine | actual_minus_shuffled | pending | pending | pending |

**Serving-repeat diagnostic: pending (0/64 pairs available).** Agreement metrics are withheld until every declared repeat and its reference is present.

All failures remain incorrect in primary metrics, including unknown transport outcomes; missing costs stay unknown. The repeat sample is predefined and serves a separate descriptive diagnostic. Repeated prompts and primary arms share test cases, so these observations are not independent architecture trials.

These holdouts were already observed. Intervals are exploratory, unadjusted group-bootstrap intervals conditional on the frozen cases, demonstrations, prompts and observed responses; they do not establish generalization to new data or deterministic serving. Full matched arms are required for comparisons, and no partial result is promoted to a headline. No paid request, budget allocation, or historical producer/report modification occurs in this summarizer.
