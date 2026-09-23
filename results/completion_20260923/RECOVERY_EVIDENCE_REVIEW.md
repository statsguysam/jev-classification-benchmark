# Historical recovery evidence review

The finite historical failure-recovery phase is complete. This secondary audit uses only its finalized report, plan, attempts and ledger, plus original evidence. Matched controls and any later control recovery are outside scope. No original prediction, primary report or producer was changed.

## Operational result

- **71 additional service calls: 70 Jev and one Astra.** There were 66 valid returns and five rejected attempts, all five reporting `invalid_output: probabilities do not sum to one`.
- The frozen inventory held 66 originally failed paid requests and one inherited upstream skip. **65/66 originally failed requests obtained a valid response** within the two-additional-attempt limit. The dependent Wine review also obtained a valid first actual response, giving 66 recovered inventory entries and one unresolved entry. A valid response need not be correct.
- The secondary report covers all rows in **23 affected original conditions**, including one historical Titanic direct-Jev condition outside the current four-dataset review matrix. It does not score only the failed subset. Original successes, including wrong labels, were never retried or replaced; selection is the earliest valid response, without reference to truth.
- The finalized recovery ledger contains 71 reservations, 71 result events and one eligible Astra settlement. Independently reconstructed retained accounting is **US$0.215772500**, matching the saved run and report. Jev recovery reservations remain fully retained. This is a stage-specific conservative amount, not actual provider billing or the current global total.

## Selected examples are unchanged

| Selected condition | Source correct | Reviewed correct | Direct Jev correct | Recovery effect |
|---|---:|---:|---:|---|
| Wine / Qwen2.5 0.5B / four per class | 12/36 | 33/36 | 33/36 | None; every original source/review/direct prediction remains selected |
| Wine / Astra / zero-shot | 36/36 | 16/36 | 12/36 | None; all 20 harms remain accepted wrong-label overrides |
| TREC / Astra / four per class | 194/200 | 173/200 | 171/200 | None; three corrections and 24 wrong-label harms, zero failures |

Wine Qwen2.5’s all-36 label agreement with direct Jev remains intact because neither arm changed. Its matched twelve-label logistic-regression reference remains 35/36. These observations retain their original denominators and exploratory interpretation; recovery adds no new independent replication.

## Recovered Astra pipelines compared with recovered sources

| Dataset | Examples/class | Recovered source correct | Recovered review correct | Corrected | Wrong-label harms | Failure harms | Remaining review failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Breast Cancer | 0 | 113/114 | 109/114 | 0 | 4 | 0 | 0 |
| Breast Cancer | 4 | 112/114 | 108/114 | 0 | 4 | 0 | 0 |
| Wine | 0 | 36/36 | 16/36 | 0 | 20 | 0 | 0 |
| Wine | 4 | 36/36 | 35/36 | 0 | 1 | 0 | 0 |
| SST-2 | 0 | 194/200 | 188/200 | 2 | 8 | 0 | 0 |
| SST-2 | 4 | 195/200 | 193/200 | 1 | 3 | 0 | 0 |
| TREC | 0 | 194/200 | 89/200 | 0 | 104 | 1 | 1 |
| TREC | 4 | 194/200 | 173/200 | 3 | 24 | 0 | 0 |

**All eight recovered pipelines still have fewer correct decisions than their corresponding recovered sources.** These conditions reuse four test sets and are not eight independent trials or a significance test. The large TREC zero-shot deficit persists: 89/200 accepted correct versus 194/200 for the source. Even granting the one remaining failure a correct result gives only 90/200 (45%). This is an optimistic completed-sample response-validity bound, not a confidence interval or a forecast of another retry.

Wine four-shot requires special care. Its source improves from 35/36 to 36/36 when the failed Astra request returns a valid correct class. That recovered class then enables the previously unsent Jev review, taking its pipeline from 34/36 to 35/36. The final comparison is therefore **36/36 source versus 35/36 review**, not a comparison with the old 35/36 source snapshot. The review is an explicitly versioned dependent first call with a new proposal; its original prompt hash is null because no earlier Jev request existed. The saved dependency binds the exact recovered source-prediction hash.

## One unresolved request

- Original condition: `trec__astra__k0__jev-review` (TREC, Astra proposal, zero-shot).
- Row ID: `trec:test:455:bbe89303faa2ae74`; original prediction line 186.
- Original failure and both additional attempts: `invalid_output: probabilities do not sum to one`.
- Frozen failure identity: `763a39d71bd8e9f2edad9a7ce994dff7e2d58cc62e074eb444cbd9d5c2bf455f`.
- The finite policy is exhausted for this row. Its saved class and probability vector remain null. The record establishes a protocol rejection, not the returned class’s correctness or rounding as the cause. No additional attempt is assumed or authorized by this memo.

## Scientific interpretation

Recovery improves the availability of usable outputs and changes some secondary full-condition scores, but it does not erase first-attempt operational failures or reveal what the original rejected classes were. The selected rescue-versus-direct comparison and strong-source harm examples survive. So does the all-eight Astra direction pattern after using recovered sources consistently. These data still do not isolate the causal value of a proposal: the matched control experiment has its own primary outcomes and is not analyzed here.

Keep the [first-pass evidence review](FIRST_PASS_EVIDENCE_REVIEW.md) and its direction counts labeled first attempt. Recovery outcomes are later serving observations selected only from operationally failed cases, not fresh test data, improved prompting, or a general estimate of semantic model improvement. Source-scoring limitations and reused public holdouts remain unchanged.

## Audit scope and hashes

Checked the finalized recovery artifact hashes against its report and run, all original artifact pins in its provenance, every attempt’s original prediction and request-prompt identity, the maximum two-attempt policy and stopping at the first valid return. Verified the Wine dependency’s recovered-source hash. Recomputed all three full-condition metric views for all 23 affected conditions and checked their saved paired accuracy point estimates. Reconstructed the finalized recovery ledger hash chain, reservation/result correspondence, single settlement, durable anchor and retained stage amount without accessing a live control ledger.

Also rechecked all 408 original primary run/prediction/test-manifest files and both primary-report hashes against the earlier scoped first-pass audit. They remain unchanged. All scoped files were re-read after aggregation. No global ledger inventory or live control/recovery collector was invoked.

| Scoped file | SHA-256 |
|---|---|
| `results/completion_20260923/RECOVERY_COMPARISON.json` | `576adcd44d703c7e42e01dcec22faa178a6986c0f85bb0c99af65e0f06802cac` |
| `results/completion_20260923/RECOVERY_FINDINGS.md` | `b5957fa145a6a8fd80fce3e6f63dd5879357bf5c4e37a4ebb0b4bf9695a9755c` |
| `results/completion_20260923/retries/run.json` | `1d76a620ed4d8103a12f54cd83a646b9b618a12be5418686fcde04fb61132b2f` |
| `results/completion_20260923/retries/plan.json` | `0f0f6674a8befdd291f6cb3101743f75e19f62b5d484a368ccd7ab7bc39bb8fb` |
| `results/completion_20260923/retries/attempts.jsonl` | `e314764968c53f527f322065180ec2e74eb6b4cc3092286221b5d9a9508376bc` |
| `results/completion_20260923/retries/budget.jsonl` | `e346bcf3fdb1f6002cbdb389e02e80947fb5382bb210f9b7696f4f30266c8918` |
| `results/completion_20260923/retries/budget.jsonl.lock` | `e68083a173e62e9dccbd91f251aa1594ce18fc2d4c0ee64e2b30d0e994bb5a6b` |
| `results/numeric_expansion/COMPARISON.json` | `b6b33421c51ee1cf9809659589f0724bb2ea883d9d1bb094b9f973d5ea654256` |
| `results/text_extension/COMPARISON.json` | `ef4e08cc9e44f384a86093db8e874a49bfd42577c1f1f97f5a5a8d176b55a92a` |
| `scripts/summarize_failed_retries.py` | `3e51a3f127cc6d608b6eb0bff4828ae486279bf23a0bcead846798241031c811` |
| `scripts/analyze_jev_recovery.py` | `c6a606113d79dc8a1038058a45a2f191c2eb2f42d19fd3b7627f246aed136df8` |

The primary reports’ `runs[*].artifact_sha256` and recovery report’s `provenance.original_artifact_sha256` retain the full original-file hash mappings. Exact secondary facts are in `api_outcomes`, `budget`, and `conditions[*].{original_snapshot,first_attempt,recovered}`. The additional Astra comparison table here was recomputed by overlaying only the earliest valid recovered response onto each complete original source/review condition.
