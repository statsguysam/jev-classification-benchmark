# Original first-pass evidence review

Both primary studies are complete: **68/68 numerical conditions and 68/68 text conditions, with 72/72 contrasts in each**. This scoped audit covers all 136 original runs and all 48 source→Jev review conditions. No recovery or matched-control outcome is included. Original failures remain in full test denominators.

## What the complete condition inventory shows

Direction means a difference in the number of accepted-correct decisions on the same full test fold. It does not mean statistical significance. Each dataset contributes twelve review conditions: six sources at zero and four training examples per class.

| Dataset | N | Review vs source: improved / tied / degraded | Review vs Jev alone: improved / tied / degraded |
|---|---:|---:|---:|
| Breast Cancer | 114 | 10 / 0 / 2 | 8 / 1 / 3 |
| Wine | 36 | 6 / 2 / 4 | 4 / 6 / 2 |
| SST-2 | 200 | 10 / 0 / 2 | 2 / 8 / 2 |
| TREC | 200 | 6 / 1 / 5 | 7 / 1 / 4 |

Across the 48 condition entries: **32 / 3 / 13 versus source** and **21 / 16 / 11 versus Jev alone**. These are inventory counts, not 48 independent trials or a pooled accuracy estimate. The four test sets contain 550 distinct case IDs reused across conditions; each direct-Jev condition is reused as a reference for six sources.

## Selected observations, checked against original predictions

- **Wine, Qwen2.5 0.5B, four examples/class:** 12/36 → 33/36 (33.3% → 91.7%); 21 corrected, no harmful overrides and no source/review/direct failures. All 36/36 review labels equal direct Jev, which also scores 33/36. Source-to-review change: +58.33 percentage points; saved exploratory paired 95% interval [41.67, 75.00]. This is no observed incremental accuracy over direct Jev on these cases, not general equivalence.
- **TREC, Astra, four examples/class:** 194/200 → 173/200 (97.0% → 86.5%); 3 corrected, 24 accepted wrong-label harms, zero failures at both stages. Change: −10.50 points; saved paired interval [−15.50, −6.00]. The reviewed pipeline still exceeds direct Jev's 171/200 by two correct cases; that comparison alone conceals damage to the source.
- **Wine, Astra, zero-shot:** 36/36 → 16/36 (100.0% → 44.4%), with 20 wrong-label harms, no corrections and no failures. The selected Wine gain and harm examples are unaffected by rejected-response handling.
- **Matched-label native references:** Wine logistic regression 35/36 with 12 training labels; Wine random forest 34/36 with 12; Breast Cancer random forest 109/114 with 8. Their exact training-example IDs match the corresponding four-per-class prompt examples. Full-training results use a different label budget.

## All eight Astra directions

| Dataset | Examples/class | Source correct | Review correct | Corrected | Wrong-label harms | Failure harms | All review failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Breast Cancer | 0 | 113/114 | 109/114 | 0 | 4 | 0 | 0 |
| Breast Cancer | 4 | 112/114 | 107/114 | 0 | 4 | 1 | 1 |
| Wine | 0 | 36/36 | 16/36 | 0 | 20 | 0 | 0 |
| Wine | 4 | 35/36 | 34/36 | 0 | 1 | 0 | 1 |
| SST-2 | 0 | 194/200 | 187/200 | 2 | 8 | 1 | 1 |
| SST-2 | 4 | 195/200 | 193/200 | 1 | 3 | 0 | 0 |
| TREC | 0 | 194/200 | 84/200 | 0 | 100 | 10 | 10 |
| TREC | 4 | 194/200 | 173/200 | 3 | 24 | 0 | 0 |

All eight decrease, but they reuse four test sets and are not eight independent replications. Granting every failed review-pipeline row a correct result, while holding accepted choices fixed, still leaves seven conditions below their source. Wine four-shot only ties after granting its never-called upstream skip a hypothetical correct result. Astra TREC zero-shot has 84 accepted-correct outputs plus ten failures: its optimistic bound is 94/200 (47%), versus the source’s 194/200 (97%). This is a descriptive response-validity bound, not a population confidence interval or an estimate of retry performance. See [the separate Astra audit](ASTRA_REVIEW_AUDIT.md).

## Four examples per class versus zero examples

| Dataset | Source: improved / tied / degraded | Reviewed pipeline: improved / tied / degraded | Direct Jev correct: zero → four | New labels at four/class |
|---|---:|---:|---:|---:|
| Breast Cancer | 4 / 1 / 1 | 4 / 1 / 1 | 96/114 → 106/114 | 8 |
| Wine | 3 / 2 / 1 | 6 / 0 / 0 | 12/36 → 33/36 | 12 |
| SST-2 | 6 / 0 / 0 | 6 / 0 / 0 | 187/200 → 193/200 | 8 |
| TREC | 5 / 1 / 0 | 6 / 0 / 0 | 67/200 → 171/200 | 24 |

These contrasts are descriptive comparisons of different supplied-label budgets. Both the source and reviewer receive demonstrations, and source proposals may change; the contrasts do not isolate why a change occurred. No test-selected example set or pooled winner is inferred.

## Failure inventory within these two primary matrices

| Dataset | Source failures | Direct-Jev failures | Review-pipeline failures | Of review failures: called / upstream skip |
|---|---:|---:|---:|---:|
| Breast Cancer | 0 | 2 | 8 | 8 / 0 |
| Wine | 1 | 0 | 2 | 1 / 1 |
| SST-2 | 0 | 2 | 3 | 3 / 0 |
| TREC | 0 | 3 | 45 | 45 / 0 |

There are 58 review-pipeline failures: 57 recorded failed review calls and one inherited no-call skip. Numeric reviews contain ten; text reviews contain 48 (three SST-2, 45 TREC). The separate source and direct-reference failures above are counted once per original run, not multiplied by their repeated appearances as comparison references. Call failure does not imply a known provider charge.

| Exact review error | Count |
|---|---:|
| `http_error: status=402; no automatic retry` | 6 |
| `http_error: status=520; no automatic retry` | 5 |
| `invalid_output: Jev choice disagrees with maximum probability` | 3 |
| `invalid_output: probabilities do not sum to one` | 39 |
| `network_error: connection failed or timed out; no automatic retry` | 4 |
| `source_proposal_failed: Jev review not called` | 1 |

All 39 review sum-to-one failures and all three review choice-versus-maximum failures occur on TREC. They are rejected responses, not established wrong-class decisions. Their saved labels/vectors are null; no rounding cause, unobserved returned class or hypothetical correctness can be recovered. The frozen metrics retain them as incorrect under the original acceptance policy. [Output-validation notes](../../docs/OUTPUT_VALIDATION_NOTES.md) explain the distinction and the subsequent collector compatibility fix.

The scoped audit found five accepted non-first-argmax labels, all exact tied maxima: Wine Granite zero-shot and the four disclosed TREC rows. No accepted selected probability in this inventory was below the maximum. Original choices and probability vectors were preserved. These accepted cases are not failed requests and must not enter failure-only recovery.

## All 48 original review conditions

Counts below retain each full N. “Wrong” is harm to an originally correct source through an accepted wrong label; “failed” is harm through an unusable review response. Other failed rows remain incorrect without necessarily being new harm. All review-failure totals, source IDs and artifact pins are retained in the two machine reports.

| Dataset | Source | Examples/class | Source → review correct | Direct Jev correct | Corrected / wrong / failed | Net vs source | Net vs direct |
|---|---|---:|---:|---:|---:|---:|---:|
| Breast Cancer | Qwen2.5 0.5B | 0 | 43 → 97 / 114 | 96/114 | 54 / 0 / 0 | +54 | +1 |
| Breast Cancer | Qwen2.5 0.5B | 4 | 70 → 107 / 114 | 106/114 | 41 / 4 / 0 | +37 | +1 |
| Breast Cancer | Qwen3 4B | 0 | 70 → 106 / 114 | 96/114 | 38 / 2 / 0 | +36 | +10 |
| Breast Cancer | Qwen3 4B | 4 | 98 → 106 / 114 | 106/114 | 10 / 2 / 0 | +8 | +0 |
| Breast Cancer | SmolLM2 1.7B | 0 | 43 → 99 / 114 | 96/114 | 57 / 1 / 0 | +56 | +3 |
| Breast Cancer | SmolLM2 1.7B | 4 | 43 → 102 / 114 | 106/114 | 62 / 1 / 2 | +59 | -4 |
| Breast Cancer | Granite 3.3 2B | 0 | 43 → 98 / 114 | 96/114 | 57 / 2 / 0 | +55 | +2 |
| Breast Cancer | Granite 3.3 2B | 4 | 70 → 105 / 114 | 106/114 | 38 / 3 / 0 | +35 | -1 |
| Breast Cancer | GPT-5.6 Luna | 0 | 86 → 100 / 114 | 96/114 | 15 / 1 / 0 | +14 | +4 |
| Breast Cancer | GPT-5.6 Luna | 4 | 104 → 105 / 114 | 106/114 | 3 / 1 / 1 | +1 | -1 |
| Breast Cancer | GPT-6 Astra | 0 | 113 → 109 / 114 | 96/114 | 0 / 4 / 0 | -4 | +13 |
| Breast Cancer | GPT-6 Astra | 4 | 112 → 107 / 114 | 106/114 | 0 / 4 / 1 | -5 | +1 |
| Wine | Qwen2.5 0.5B | 0 | 12 → 12 / 36 | 12/36 | 0 / 0 / 0 | +0 | +0 |
| Wine | Qwen2.5 0.5B | 4 | 12 → 33 / 36 | 33/36 | 21 / 0 / 0 | +21 | +0 |
| Wine | Qwen3 4B | 0 | 14 → 13 / 36 | 12/36 | 12 / 13 / 0 | -1 | +1 |
| Wine | Qwen3 4B | 4 | 29 → 32 / 36 | 33/36 | 3 / 0 / 0 | +3 | -1 |
| Wine | SmolLM2 1.7B | 0 | 12 → 12 / 36 | 12/36 | 0 / 0 / 0 | +0 | +0 |
| Wine | SmolLM2 1.7B | 4 | 12 → 32 / 36 | 33/36 | 20 / 0 / 0 | +20 | -1 |
| Wine | Granite 3.3 2B | 0 | 10 → 12 / 36 | 12/36 | 12 / 10 / 0 | +2 | +0 |
| Wine | Granite 3.3 2B | 4 | 12 → 33 / 36 | 33/36 | 21 / 0 / 0 | +21 | +0 |
| Wine | GPT-5.6 Luna | 0 | 17 → 14 / 36 | 12/36 | 7 / 10 / 0 | -3 | +2 |
| Wine | GPT-5.6 Luna | 4 | 32 → 33 / 36 | 33/36 | 2 / 1 / 0 | +1 | +0 |
| Wine | GPT-6 Astra | 0 | 36 → 16 / 36 | 12/36 | 0 / 20 / 0 | -20 | +4 |
| Wine | GPT-6 Astra | 4 | 35 → 34 / 36 | 33/36 | 0 / 1 / 0 | -1 | +1 |
| SST-2 | Qwen2.5 0.5B | 0 | 100 → 187 / 200 | 187/200 | 88 / 1 / 0 | +87 | +0 |
| SST-2 | Qwen2.5 0.5B | 4 | 170 → 193 / 200 | 193/200 | 26 / 3 / 0 | +23 | +0 |
| SST-2 | Qwen3 4B | 0 | 183 → 188 / 200 | 187/200 | 12 / 6 / 1 | +5 | +1 |
| SST-2 | Qwen3 4B | 4 | 191 → 192 / 200 | 193/200 | 6 / 4 / 1 | +1 | -1 |
| SST-2 | SmolLM2 1.7B | 0 | 98 → 187 / 200 | 187/200 | 90 / 1 / 0 | +89 | +0 |
| SST-2 | SmolLM2 1.7B | 4 | 140 → 193 / 200 | 193/200 | 57 / 4 / 0 | +53 | +0 |
| SST-2 | Granite 3.3 2B | 0 | 160 → 187 / 200 | 187/200 | 29 / 2 / 0 | +27 | +0 |
| SST-2 | Granite 3.3 2B | 4 | 188 → 192 / 200 | 193/200 | 6 / 2 / 0 | +4 | -1 |
| SST-2 | GPT-5.6 Luna | 0 | 185 → 189 / 200 | 187/200 | 5 / 1 / 0 | +4 | +2 |
| SST-2 | GPT-5.6 Luna | 4 | 191 → 193 / 200 | 193/200 | 4 / 2 / 0 | +2 | +0 |
| SST-2 | GPT-6 Astra | 0 | 194 → 187 / 200 | 187/200 | 2 / 8 / 1 | -7 | +0 |
| SST-2 | GPT-6 Astra | 4 | 195 → 193 / 200 | 193/200 | 1 / 3 / 0 | -2 | +0 |
| TREC | Qwen2.5 0.5B | 0 | 31 → 65 / 200 | 67/200 | 60 / 25 / 1 | +34 | -2 |
| TREC | Qwen2.5 0.5B | 4 | 35 → 175 / 200 | 171/200 | 143 / 3 / 0 | +140 | +4 |
| TREC | Qwen3 4B | 0 | 105 → 83 / 200 | 67/200 | 6 / 24 / 4 | -22 | +16 |
| TREC | Qwen3 4B | 4 | 166 → 172 / 200 | 171/200 | 23 / 17 / 0 | +6 | +1 |
| TREC | SmolLM2 1.7B | 0 | 4 → 65 / 200 | 67/200 | 61 / 0 / 0 | +61 | -2 |
| TREC | SmolLM2 1.7B | 4 | 55 → 169 / 200 | 171/200 | 133 / 19 / 0 | +114 | -2 |
| TREC | Granite 3.3 2B | 0 | 92 → 79 / 200 | 67/200 | 7 / 19 / 1 | -13 | +12 |
| TREC | Granite 3.3 2B | 4 | 127 → 171 / 200 | 171/200 | 57 / 11 / 2 | +44 | +0 |
| TREC | GPT-5.6 Luna | 0 | 105 → 79 / 200 | 67/200 | 4 / 27 / 3 | -26 | +12 |
| TREC | GPT-5.6 Luna | 4 | 170 → 170 / 200 | 171/200 | 20 / 20 / 0 | +0 | -1 |
| TREC | GPT-6 Astra | 0 | 194 → 84 / 200 | 67/200 | 0 / 100 / 10 | -110 | +17 |
| TREC | GPT-6 Astra | 4 | 194 → 173 / 200 | 171/200 | 3 / 24 / 0 | -21 | +2 |

## Scope and independent checks

- Recomputed all saved metric fields for 136 complete original runs from their ordered predictions and label-only test manifests, using the unchanged metric function. Checked every value against both the original run and aggregate report. Matched dataset/class order, full denominators and paired test rows; equal-budget contrasts use identical training-example IDs. Recomputed all 48 correction/harm partitions and verified accuracy and macro-F1 point estimates for all 144 contrasts. The quoted bootstrap intervals come from the audited saved reports; this scoped audit did not resample them again.
- Verified every report-listed SHA for 408 original run/prediction/test-manifest files, and re-read all 415 scoped files after aggregation. No global ledger inventory, mutable recovery folder, inference API or whole-repository collector was accessed. Only this memo is added to results; original records and reports are unchanged.
- The numerical local scorer is constant-label in 11/16 conditions; the text local scorer in 3/16. Its normalized numeric-ID-plus-EOS sequence likelihoods are not calibrated correctness probabilities. Source renderings and serving protocols differ. Large improvements over weak recorded sources do not establish general model capability or added value over direct Jev.
- The historical direct-Jev and review prompts differ. Repeated model conditions share cases and direct references; no direction count is a significance test. Familiar public datasets, possible pretraining exposure, one split, tiny Wine N and changing supplied-label budgets limit generalization. Intervals are exploratory, conditional and unadjusted. No latency, dollar-saving, matched-control or recovery finding is asserted here.

## Scoped evidence hashes

Each report’s `runs[*].artifact_sha256` maps the original run directory to its verified `run.json`, `predictions.jsonl` and `test_manifest.json` SHA-256 values. These report hashes bind all 408 raw-file pins without repeating that inventory in this memo.

| File | SHA-256 |
|---|---|
| `results/completion_sessions/20260923T151304Z-text-54e746b7eeb9/completion.json` | `31fd4c038d811b7edc7897c78ee0049afb97d2ba07502975fb8424b4d6b5f356` |
| `results/numeric_expansion/COMPARISON.json` | `b6b33421c51ee1cf9809659589f0724bb2ea883d9d1bb094b9f973d5ea654256` |
| `results/text_extension/COMPARISON.json` | `ef4e08cc9e44f384a86093db8e874a49bfd42577c1f1f97f5a5a8d176b55a92a` |
| `src/jevbench/metrics.py` | `551146253f47dedc9354e6e1b1ed81fdab57c07b0c77b8471a31eb4e2b6528c9` |
| `src/jevbench/providers.py` | `fbeb2156be56246639bd87da309931e0f62928617ab23a4a6b3da60353b1c149` |
| `scripts/summarize_expanded_numeric.py` | `3910031e6c27ae157b636903323225849e51d8901968ed10a9f5dfc3838f0d41` |
| `scripts/summarize_text_extension.py` | `bef6c994570a8bef1cbf9d647049bd1831d8bfa25260a5f3d5026b78f74dfb99` |

Exact fact locations: each report’s `comparisons[*]` entries with `kind == "review_minus_source"`, `"review_minus_jev_alone"` or `"few_minus_zero_descriptive"`; `.transitions` on source contrasts; `.paired_bootstrap.metrics` for intervals; and `runs[*]` for per-condition metrics, training budgets, failures, source paths and raw hashes.
