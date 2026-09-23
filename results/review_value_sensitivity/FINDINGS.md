# Constant-label sensitivity appendix

**All 16 complete local-model conditions with eligible probability scores are included:** 5 conditions shared with the primary analysis and 11 constant-label conditions added only here. The 0 incomplete review conditions receive no metrics, and hosted sources remain outside this probability-ranking analysis.

[Machine report, complete curves and source pins](ANALYSIS.json) · [Unchanged primary findings](../review_value/FINDINGS.md)

The primary analysis excluded sources that predicted one class throughout their test fold. That was a post-hoc scope restriction, not proof that their confidence ranks were unusable. This appendix removes only that restriction and reuses precisely the same source maximum-probability ranking, row-hash tie-break, six coverage rates, rounding, cached review outcomes and random reference. No threshold, model or coverage is selected using these test outcomes.

All 11/11 newly included constant-label conditions have varying maximum-probability scores. Output-label collapse and score variation are different properties. Every unchanged primary curve matches the earlier machine report exactly; source/full-review endpoints and request counts are checked against the same audited records.

## All included conditions

The 50% column is the same predetermined midpoint for every condition. It is descriptive, not a recommendation. Primary membership is explicit; all other coverages follow in the next table.

| Dataset | Source | Examples/class | Membership | Constant label? | Unique max scores | No review accuracy | 50% review accuracy | Random 50% expectation | Full review accuracy |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| Breast Cancer | Qwen2.5 0.5B | 0 | sensitivity only | yes | 114 | 37.7% | 53.5% | 61.4% | 85.1% |
| Breast Cancer | Qwen2.5 0.5B | 4 | primary + sensitivity | no | 114 | 61.4% | 70.2% | 77.6% | 93.9% |
| Breast Cancer | Qwen3 4B | 0 | primary + sensitivity | no | 111 | 61.4% | 76.3% | 77.2% | 93.0% |
| Breast Cancer | Qwen3 4B | 4 | primary + sensitivity | no | 108 | 86.0% | 93.0% | 89.5% | 93.0% |
| Breast Cancer | SmolLM2 1.7B | 0 | sensitivity only | yes | 114 | 37.7% | 58.8% | 62.3% | 86.8% |
| Breast Cancer | SmolLM2 1.7B | 4 | sensitivity only | yes | 114 | 37.7% | 63.2% | 63.6% | 89.5% |
| Breast Cancer | Granite 3.3 2B | 0 | sensitivity only | yes | 114 | 37.7% | 65.8% | 61.8% | 86.0% |
| Breast Cancer | Granite 3.3 2B | 4 | primary + sensitivity | no | 114 | 61.4% | 86.8% | 76.8% | 92.1% |
| Wine | Qwen2.5 0.5B | 0 | sensitivity only | yes | 36 | 33.3% | 33.3% | 33.3% | 33.3% |
| Wine | Qwen2.5 0.5B | 4 | sensitivity only | yes | 36 | 33.3% | 58.3% | 62.5% | 91.7% |
| Wine | Qwen3 4B | 0 | sensitivity only | yes | 36 | 38.9% | 41.7% | 37.5% | 36.1% |
| Wine | Qwen3 4B | 4 | primary + sensitivity | no | 36 | 80.6% | 88.9% | 84.7% | 88.9% |
| Wine | SmolLM2 1.7B | 0 | sensitivity only | yes | 36 | 33.3% | 33.3% | 33.3% | 33.3% |
| Wine | SmolLM2 1.7B | 4 | sensitivity only | yes | 36 | 33.3% | 50.0% | 61.1% | 88.9% |
| Wine | Granite 3.3 2B | 0 | sensitivity only | yes | 36 | 27.8% | 19.4% | 30.6% | 33.3% |
| Wine | Granite 3.3 2B | 4 | sensitivity only | yes | 36 | 33.3% | 75.0% | 62.5% | 91.7% |

## Every fixed coverage

Accuracy is within-condition micro accuracy. Random accuracy and balanced accuracy are exact expectations for a uniformly selected subset of the same size. Random macro-F1, its Monte Carlo standard error, exact selection IDs, full direct-Jev references, and all failure details remain in the JSON.

| Dataset/source/shots | Membership | Reviewed rows | Accuracy | Balanced accuracy | Macro-F1 | Random accuracy | Fixed | Harmed | Source inferences + review requests |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Breast Cancer / Qwen2.5 0.5B / 0 | sensitivity only | 0/114 | 37.7% | 50.0% | 0.274 | 37.7% | 0 | 0 | 114 + 0 |
| Breast Cancer / Qwen2.5 0.5B / 0 | sensitivity only | 11/114 | 40.4% | 52.1% | 0.320 | 42.3% | 3 | 0 | 114 + 11 |
| Breast Cancer / Qwen2.5 0.5B / 0 | sensitivity only | 29/114 | 43.0% | 54.2% | 0.363 | 49.8% | 6 | 0 | 114 + 29 |
| Breast Cancer / Qwen2.5 0.5B / 0 | sensitivity only | 57/114 | 53.5% | 62.7% | 0.512 | 61.4% | 18 | 0 | 114 + 57 |
| Breast Cancer / Qwen2.5 0.5B / 0 | sensitivity only | 86/114 | 67.5% | 73.9% | 0.673 | 73.5% | 34 | 0 | 114 + 86 |
| Breast Cancer / Qwen2.5 0.5B / 0 | sensitivity only | 114/114 | 85.1% | 88.0% | 0.849 | 85.1% | 54 | 0 | 114 + 114 |
| Breast Cancer / Qwen2.5 0.5B / 4 | primary + sensitivity | 0/114 | 61.4% | 49.3% | 0.380 | 61.4% | 0 | 0 | 114 + 0 |
| Breast Cancer / Qwen2.5 0.5B / 4 | primary + sensitivity | 11/114 | 62.3% | 50.0% | 0.384 | 64.5% | 1 | 0 | 114 + 11 |
| Breast Cancer / Qwen2.5 0.5B / 4 | primary + sensitivity | 29/114 | 63.2% | 51.2% | 0.409 | 69.7% | 2 | 0 | 114 + 29 |
| Breast Cancer / Qwen2.5 0.5B / 4 | primary + sensitivity | 57/114 | 70.2% | 60.5% | 0.576 | 77.6% | 10 | 0 | 114 + 57 |
| Breast Cancer / Qwen2.5 0.5B / 4 | primary + sensitivity | 86/114 | 80.7% | 75.3% | 0.769 | 85.9% | 24 | 2 | 114 + 86 |
| Breast Cancer / Qwen2.5 0.5B / 4 | primary + sensitivity | 114/114 | 93.9% | 93.7% | 0.935 | 93.9% | 41 | 4 | 114 + 114 |
| Breast Cancer / Qwen3 4B / 0 | primary + sensitivity | 0/114 | 61.4% | 52.0% | 0.479 | 61.4% | 0 | 0 | 114 + 0 |
| Breast Cancer / Qwen3 4B / 0 | primary + sensitivity | 11/114 | 64.9% | 54.9% | 0.502 | 64.5% | 5 | 1 | 114 + 11 |
| Breast Cancer / Qwen3 4B / 0 | primary + sensitivity | 29/114 | 68.4% | 58.6% | 0.552 | 69.4% | 9 | 1 | 114 + 29 |
| Breast Cancer / Qwen3 4B / 0 | primary + sensitivity | 57/114 | 76.3% | 68.6% | 0.691 | 77.2% | 18 | 1 | 114 + 57 |
| Breast Cancer / Qwen3 4B / 0 | primary + sensitivity | 86/114 | 85.1% | 80.2% | 0.823 | 85.2% | 28 | 1 | 114 + 86 |
| Breast Cancer / Qwen3 4B / 0 | primary + sensitivity | 114/114 | 93.0% | 91.2% | 0.923 | 93.0% | 38 | 2 | 114 + 114 |
| Breast Cancer / Qwen3 4B / 4 | primary + sensitivity | 0/114 | 86.0% | 88.7% | 0.858 | 86.0% | 0 | 0 | 114 + 0 |
| Breast Cancer / Qwen3 4B / 4 | primary + sensitivity | 11/114 | 89.5% | 91.1% | 0.892 | 86.6% | 5 | 1 | 114 + 11 |
| Breast Cancer / Qwen3 4B / 4 | primary + sensitivity | 29/114 | 91.2% | 92.0% | 0.909 | 87.8% | 8 | 2 | 114 + 29 |
| Breast Cancer / Qwen3 4B / 4 | primary + sensitivity | 57/114 | 93.0% | 93.4% | 0.927 | 89.5% | 10 | 2 | 114 + 57 |
| Breast Cancer / Qwen3 4B / 4 | primary + sensitivity | 86/114 | 93.0% | 93.4% | 0.927 | 91.3% | 10 | 2 | 114 + 86 |
| Breast Cancer / Qwen3 4B / 4 | primary + sensitivity | 114/114 | 93.0% | 93.4% | 0.927 | 93.0% | 10 | 2 | 114 + 114 |
| Breast Cancer / SmolLM2 1.7B / 0 | sensitivity only | 0/114 | 37.7% | 50.0% | 0.274 | 37.7% | 0 | 0 | 114 + 0 |
| Breast Cancer / SmolLM2 1.7B / 0 | sensitivity only | 11/114 | 43.9% | 54.9% | 0.376 | 42.5% | 7 | 0 | 114 + 11 |
| Breast Cancer / SmolLM2 1.7B / 0 | sensitivity only | 29/114 | 50.0% | 59.9% | 0.465 | 50.2% | 14 | 0 | 114 + 29 |
| Breast Cancer / SmolLM2 1.7B / 0 | sensitivity only | 57/114 | 58.8% | 66.4% | 0.578 | 62.3% | 25 | 1 | 114 + 57 |
| Breast Cancer / SmolLM2 1.7B / 0 | sensitivity only | 86/114 | 71.1% | 76.3% | 0.710 | 74.8% | 39 | 1 | 114 + 86 |
| Breast Cancer / SmolLM2 1.7B / 0 | sensitivity only | 114/114 | 86.8% | 89.0% | 0.866 | 86.8% | 57 | 1 | 114 + 114 |
| Breast Cancer / SmolLM2 1.7B / 4 | sensitivity only | 0/114 | 37.7% | 50.0% | 0.274 | 37.7% | 0 | 0 | 114 + 0 |
| Breast Cancer / SmolLM2 1.7B / 4 | sensitivity only | 11/114 | 40.4% | 52.1% | 0.323 | 42.7% | 3 | 0 | 114 + 11 |
| Breast Cancer / SmolLM2 1.7B / 4 | sensitivity only | 29/114 | 51.8% | 61.3% | 0.493 | 50.9% | 16 | 0 | 114 + 29 |
| Breast Cancer / SmolLM2 1.7B / 4 | sensitivity only | 57/114 | 63.2% | 69.5% | 0.637 | 63.6% | 31 | 2 | 114 + 57 |
| Breast Cancer / SmolLM2 1.7B / 4 | sensitivity only | 86/114 | 77.2% | 80.8% | 0.785 | 76.8% | 47 | 2 | 114 + 86 |
| Breast Cancer / SmolLM2 1.7B / 4 | sensitivity only | 114/114 | 89.5% | 90.2% | 0.917 | 89.5% | 62 | 3 | 114 + 114 |
| Breast Cancer / Granite 3.3 2B / 0 | sensitivity only | 0/114 | 37.7% | 50.0% | 0.274 | 37.7% | 0 | 0 | 114 + 0 |
| Breast Cancer / Granite 3.3 2B / 0 | sensitivity only | 11/114 | 45.6% | 56.3% | 0.403 | 42.4% | 9 | 0 | 114 + 11 |
| Breast Cancer / Granite 3.3 2B / 0 | sensitivity only | 29/114 | 57.0% | 65.5% | 0.555 | 50.0% | 22 | 0 | 114 + 29 |
| Breast Cancer / Granite 3.3 2B / 0 | sensitivity only | 57/114 | 65.8% | 72.5% | 0.655 | 61.8% | 32 | 0 | 114 + 57 |
| Breast Cancer / Granite 3.3 2B / 0 | sensitivity only | 86/114 | 72.8% | 77.7% | 0.728 | 74.1% | 41 | 1 | 114 + 86 |
| Breast Cancer / Granite 3.3 2B / 0 | sensitivity only | 114/114 | 86.0% | 87.8% | 0.857 | 86.0% | 57 | 2 | 114 + 114 |
| Breast Cancer / Granite 3.3 2B / 4 | primary + sensitivity | 0/114 | 61.4% | 69.0% | 0.606 | 61.4% | 0 | 0 | 114 + 0 |
| Breast Cancer / Granite 3.3 2B / 4 | primary + sensitivity | 11/114 | 67.5% | 73.9% | 0.673 | 64.4% | 7 | 0 | 114 + 11 |
| Breast Cancer / Granite 3.3 2B / 4 | primary + sensitivity | 29/114 | 76.3% | 81.0% | 0.763 | 69.2% | 17 | 0 | 114 + 29 |
| Breast Cancer / Granite 3.3 2B / 4 | primary + sensitivity | 57/114 | 86.8% | 89.4% | 0.867 | 76.8% | 29 | 0 | 114 + 57 |
| Breast Cancer / Granite 3.3 2B / 4 | primary + sensitivity | 86/114 | 92.1% | 92.3% | 0.917 | 84.6% | 38 | 3 | 114 + 86 |
| Breast Cancer / Granite 3.3 2B / 4 | primary + sensitivity | 114/114 | 92.1% | 92.3% | 0.917 | 92.1% | 38 | 3 | 114 + 114 |
| Wine / Qwen2.5 0.5B / 0 | sensitivity only | 0/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 0 |
| Wine / Qwen2.5 0.5B / 0 | sensitivity only | 4/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 4 |
| Wine / Qwen2.5 0.5B / 0 | sensitivity only | 9/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 9 |
| Wine / Qwen2.5 0.5B / 0 | sensitivity only | 18/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 18 |
| Wine / Qwen2.5 0.5B / 0 | sensitivity only | 27/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 27 |
| Wine / Qwen2.5 0.5B / 0 | sensitivity only | 36/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 36 |
| Wine / Qwen2.5 0.5B / 4 | sensitivity only | 0/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 0 |
| Wine / Qwen2.5 0.5B / 4 | sensitivity only | 4/36 | 33.3% | 33.3% | 0.167 | 39.8% | 0 | 0 | 36 + 4 |
| Wine / Qwen2.5 0.5B / 4 | sensitivity only | 9/36 | 38.9% | 40.0% | 0.285 | 47.9% | 2 | 0 | 36 + 9 |
| Wine / Qwen2.5 0.5B / 4 | sensitivity only | 18/36 | 58.3% | 61.4% | 0.563 | 62.5% | 9 | 0 | 36 + 18 |
| Wine / Qwen2.5 0.5B / 4 | sensitivity only | 27/36 | 75.0% | 78.6% | 0.743 | 77.1% | 15 | 0 | 36 + 27 |
| Wine / Qwen2.5 0.5B / 4 | sensitivity only | 36/36 | 91.7% | 92.9% | 0.916 | 91.7% | 21 | 0 | 36 + 36 |
| Wine / Qwen3 4B / 0 | sensitivity only | 0/36 | 38.9% | 33.3% | 0.187 | 38.9% | 0 | 0 | 36 + 0 |
| Wine / Qwen3 4B / 0 | sensitivity only | 4/36 | 41.7% | 36.1% | 0.245 | 38.6% | 1 | 0 | 36 + 4 |
| Wine / Qwen3 4B / 0 | sensitivity only | 9/36 | 41.7% | 36.9% | 0.290 | 38.2% | 3 | 2 | 36 + 9 |
| Wine / Qwen3 4B / 0 | sensitivity only | 18/36 | 41.7% | 38.1% | 0.320 | 37.5% | 6 | 5 | 36 + 18 |
| Wine / Qwen3 4B / 0 | sensitivity only | 27/36 | 38.9% | 37.3% | 0.287 | 36.8% | 10 | 10 | 36 + 27 |
| Wine / Qwen3 4B / 0 | sensitivity only | 36/36 | 36.1% | 35.7% | 0.217 | 36.1% | 12 | 13 | 36 + 36 |
| Wine / Qwen3 4B / 4 | primary + sensitivity | 0/36 | 80.6% | 82.1% | 0.809 | 80.6% | 0 | 0 | 36 + 0 |
| Wine / Qwen3 4B / 4 | primary + sensitivity | 4/36 | 83.3% | 84.9% | 0.836 | 81.5% | 1 | 0 | 36 + 4 |
| Wine / Qwen3 4B / 4 | primary + sensitivity | 9/36 | 83.3% | 84.9% | 0.836 | 82.6% | 1 | 0 | 36 + 9 |
| Wine / Qwen3 4B / 4 | primary + sensitivity | 18/36 | 88.9% | 90.1% | 0.891 | 84.7% | 3 | 0 | 36 + 18 |
| Wine / Qwen3 4B / 4 | primary + sensitivity | 27/36 | 88.9% | 90.1% | 0.891 | 86.8% | 3 | 0 | 36 + 27 |
| Wine / Qwen3 4B / 4 | primary + sensitivity | 36/36 | 88.9% | 90.1% | 0.891 | 88.9% | 3 | 0 | 36 + 36 |
| Wine / SmolLM2 1.7B / 0 | sensitivity only | 0/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 0 |
| Wine / SmolLM2 1.7B / 0 | sensitivity only | 4/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 4 |
| Wine / SmolLM2 1.7B / 0 | sensitivity only | 9/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 9 |
| Wine / SmolLM2 1.7B / 0 | sensitivity only | 18/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 18 |
| Wine / SmolLM2 1.7B / 0 | sensitivity only | 27/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 27 |
| Wine / SmolLM2 1.7B / 0 | sensitivity only | 36/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 36 |
| Wine / SmolLM2 1.7B / 4 | sensitivity only | 0/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 0 |
| Wine / SmolLM2 1.7B / 4 | sensitivity only | 4/36 | 33.3% | 33.3% | 0.167 | 39.5% | 0 | 0 | 36 + 4 |
| Wine / SmolLM2 1.7B / 4 | sensitivity only | 9/36 | 38.9% | 40.0% | 0.285 | 47.2% | 2 | 0 | 36 + 9 |
| Wine / SmolLM2 1.7B / 4 | sensitivity only | 18/36 | 50.0% | 52.4% | 0.457 | 61.1% | 6 | 0 | 36 + 18 |
| Wine / SmolLM2 1.7B / 4 | sensitivity only | 27/36 | 72.2% | 75.2% | 0.727 | 75.0% | 14 | 0 | 36 + 27 |
| Wine / SmolLM2 1.7B / 4 | sensitivity only | 36/36 | 88.9% | 90.5% | 0.888 | 88.9% | 20 | 0 | 36 + 36 |
| Wine / Granite 3.3 2B / 0 | sensitivity only | 0/36 | 27.8% | 33.3% | 0.145 | 27.8% | 0 | 0 | 36 + 0 |
| Wine / Granite 3.3 2B / 0 | sensitivity only | 4/36 | 22.2% | 26.1% | 0.153 | 28.4% | 1 | 3 | 36 + 4 |
| Wine / Granite 3.3 2B / 0 | sensitivity only | 9/36 | 25.0% | 28.3% | 0.203 | 29.2% | 3 | 4 | 36 + 9 |
| Wine / Granite 3.3 2B / 0 | sensitivity only | 18/36 | 19.4% | 21.1% | 0.167 | 30.6% | 4 | 7 | 36 + 18 |
| Wine / Granite 3.3 2B / 0 | sensitivity only | 27/36 | 25.0% | 25.6% | 0.179 | 31.9% | 8 | 9 | 36 + 27 |
| Wine / Granite 3.3 2B / 0 | sensitivity only | 36/36 | 33.3% | 32.9% | 0.208 | 33.3% | 12 | 10 | 36 + 36 |
| Wine / Granite 3.3 2B / 4 | sensitivity only | 0/36 | 33.3% | 33.3% | 0.167 | 33.3% | 0 | 0 | 36 + 0 |
| Wine / Granite 3.3 2B / 4 | sensitivity only | 4/36 | 44.4% | 45.7% | 0.380 | 39.8% | 4 | 0 | 36 + 4 |
| Wine / Granite 3.3 2B / 4 | sensitivity only | 9/36 | 55.6% | 55.2% | 0.523 | 47.9% | 8 | 0 | 36 + 9 |
| Wine / Granite 3.3 2B / 4 | sensitivity only | 18/36 | 75.0% | 73.8% | 0.736 | 62.5% | 15 | 0 | 36 + 18 |
| Wine / Granite 3.3 2B / 4 | sensitivity only | 27/36 | 86.1% | 87.1% | 0.860 | 77.1% | 19 | 0 | 36 + 27 |
| Wine / Granite 3.3 2B / 4 | sensitivity only | 36/36 | 91.7% | 92.9% | 0.916 | 91.7% | 21 | 0 | 36 + 36 |

## What survives the broader scope

Across the 11 added constant-label conditions, 2 are above the random-selection accuracy expectation at all four intermediate coverages, 4 are below it throughout, 3 have mixed differences, and 2 tie throughout. These are descriptive signs of differences, not statistical tests; every condition and all six coverages are displayed above.

The two Qwen3 four-shot observations remain the same because their inputs, ranking and cached outcomes have not changed: 57/114 Breast Cancer reviews and 18/36 Wine reviews match their respective full-review accuracies. The broader appendix does not turn that arithmetic observation into general evidence for a confidence gate. Constant-label conditions can still have variable ranks, and their results must be considered alongside the 5 currently eligible primary conditions.

The primary eligibility rule was chosen after earlier outcomes existed. This appendix is also post-hoc and does not repair that limitation by adding more curves. Model conditions share cases, and source sequence probabilities include class ID plus EOS; they are uncalibrated. Report all conditions rather than selecting successful routes or a best test-set coverage. This analysis command makes no new inference or paid request, changes no primary report/dashboard, and establishes no dollar or latency saving.
