# Added value of Jev review: offline numeric evidence

**24/24 complete numeric review conditions are scored.** No partial-run accuracy is used. Breast Cancer has 114 test cases, Wine 36, each on one frozen split. This report covers numerical reviews; the text extension is reported separately.

This analysis reconstructs raw predictions through the unchanged provenance/metric auditors. Source, direct Jev and review share the same held-out rows and matching zero/four examples per class. All failures count as incorrect. The machine report preserves each failure and its unknown reported charge; it does not convert unknown charges to zero.

[Machine-readable report and exact artifact pins](ANALYSIS.json)

## Does review add value over both available alternatives?

Accuracy below is micro accuracy within each condition. Balanced accuracy and macro-F1 are in the JSON for all three arms. Fixed/harmed counts compare review with its own source, including failures; net versus direct compares with separately called Jev alone.

| Dataset | Source | Examples/class | Source accuracy | Review accuracy | Direct Jev accuracy | Fixed | Harmed | Net vs source | Net vs direct |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Breast Cancer | Qwen2.5 0.5B | 0 | 37.7% | 85.1% | 84.2% | 54 | 0 | +54 | +1 |
| Breast Cancer | Qwen2.5 0.5B | 4 | 61.4% | 93.9% | 93.0% | 41 | 4 | +37 | +1 |
| Breast Cancer | Qwen3 4B | 0 | 61.4% | 93.0% | 84.2% | 38 | 2 | +36 | +10 |
| Breast Cancer | Qwen3 4B | 4 | 86.0% | 93.0% | 93.0% | 10 | 2 | +8 | +0 |
| Breast Cancer | SmolLM2 1.7B | 0 | 37.7% | 86.8% | 84.2% | 57 | 1 | +56 | +3 |
| Breast Cancer | SmolLM2 1.7B | 4 | 37.7% | 89.5% | 93.0% | 62 | 3 | +59 | -4 |
| Breast Cancer | Granite 3.3 2B | 0 | 37.7% | 86.0% | 84.2% | 57 | 2 | +55 | +2 |
| Breast Cancer | Granite 3.3 2B | 4 | 61.4% | 92.1% | 93.0% | 38 | 3 | +35 | -1 |
| Breast Cancer | GPT-5.6 Luna | 0 | 75.4% | 87.7% | 84.2% | 15 | 1 | +14 | +4 |
| Breast Cancer | GPT-5.6 Luna | 4 | 91.2% | 92.1% | 93.0% | 3 | 2 | +1 | -1 |
| Breast Cancer | GPT-6 Astra | 0 | 99.1% | 95.6% | 84.2% | 0 | 4 | -4 | +13 |
| Breast Cancer | GPT-6 Astra | 4 | 98.2% | 93.9% | 93.0% | 0 | 5 | -5 | +1 |
| Wine | Qwen2.5 0.5B | 0 | 33.3% | 33.3% | 33.3% | 0 | 0 | +0 | +0 |
| Wine | Qwen2.5 0.5B | 4 | 33.3% | 91.7% | 91.7% | 21 | 0 | +21 | +0 |
| Wine | Qwen3 4B | 0 | 38.9% | 36.1% | 33.3% | 12 | 13 | -1 | +1 |
| Wine | Qwen3 4B | 4 | 80.6% | 88.9% | 91.7% | 3 | 0 | +3 | -1 |
| Wine | SmolLM2 1.7B | 0 | 33.3% | 33.3% | 33.3% | 0 | 0 | +0 | +0 |
| Wine | SmolLM2 1.7B | 4 | 33.3% | 88.9% | 91.7% | 20 | 0 | +20 | -1 |
| Wine | Granite 3.3 2B | 0 | 27.8% | 33.3% | 33.3% | 12 | 10 | +2 | +0 |
| Wine | Granite 3.3 2B | 4 | 33.3% | 91.7% | 91.7% | 21 | 0 | +21 | +0 |
| Wine | GPT-5.6 Luna | 0 | 47.2% | 38.9% | 33.3% | 7 | 10 | -3 | +2 |
| Wine | GPT-5.6 Luna | 4 | 88.9% | 91.7% | 91.7% | 2 | 1 | +1 | +0 |
| Wine | GPT-6 Astra | 0 | 100.0% | 44.4% | 33.3% | 0 | 20 | -20 | +4 |
| Wine | GPT-6 Astra | 4 | 97.2% | 94.4% | 91.7% | 0 | 1 | -1 | +1 |

The JSON supplies all eight joint correctness combinations for source/direct/review, so a correction can be distinguished from a gain already achieved by direct Jev. These conditions reuse the same cases and direct references; no pooled average or count is presented as independent evidence.

## Same proposed label means the same reviewer prompt

Across 60 within-dataset/shot source pairs, 2331 row-pair comparisons had the same valid proposed label. All had exactly matching audited review-prompt hashes. Of 2308 comparisons with two valid review responses, 77 produced different final labels; 23 comparisons had at least one review failure. These are overlapping pairwise comparisons, not independent trials.

The both-valid comparisons contain 0 resolved-model-ID mismatches and 0 unknown resolved-model comparisons. Per-run resolved IDs, start/completion times and execution sessions are preserved in the artifact inventory. Matching returned IDs do not establish an unchanged serving backend.

Source identity, confidence and rationale are absent from the reviewer prompt. Therefore, when two sources propose the same class for the same row and examples, the reviewer gets the same prompt. Cross-source differences on those matched prompts cannot be attributed to source identity being visible. The observed variability includes historical calls at different serving times and errors; these dependent comparisons were not designed as replicates and are not a repeatability estimate or causal anchoring test.

## Fixed-coverage selective review

Eligible conditions require a complete review, a local HF source, complete normalized class-sequence probabilities and at least two distinct source labels. Constant-label sources and hosted sources without probability scores are excluded. This leaves 5 conditions in the current snapshot. Eligibility depends on source outputs and completion, not test correctness; it defines the scope of this descriptive analysis.

Rank rows by ascending source maximum probability, with SHA-256(row ID) breaking ties. Select the nearest integer number of rows at predetermined 0/10/25/50/75/100% coverage (.5 rounds upward). No labels choose the rank, coverage, or threshold. This is a fixed-coverage batch simulation, not a validated deployable gate. Sequence likelihoods include numeric label plus EOS and are not calibrated probabilities of correctness.

Random matched-rate accuracy and balanced accuracy are exact expectations over uniform fixed-size subsets. Random macro-F1 is a 10,000-draw estimate with fixed seed and reported Monte Carlo standard error, since macro-F1 is nonlinear. These are reference expectations, not confidence intervals.

| Dataset/source/shots | Reviewed rows | Accuracy | Balanced accuracy | Macro-F1 | Random expected accuracy | Fixed | Harmed | Source inferences + review requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Breast Cancer / Qwen2.5 0.5B / 4 | 0/114 | 61.4% | 49.3% | 0.380 | 61.4% | 0 | 0 | 114 + 0 |
| Breast Cancer / Qwen2.5 0.5B / 4 | 11/114 | 62.3% | 50.0% | 0.384 | 64.5% | 1 | 0 | 114 + 11 |
| Breast Cancer / Qwen2.5 0.5B / 4 | 29/114 | 63.2% | 51.2% | 0.409 | 69.7% | 2 | 0 | 114 + 29 |
| Breast Cancer / Qwen2.5 0.5B / 4 | 57/114 | 70.2% | 60.5% | 0.576 | 77.6% | 10 | 0 | 114 + 57 |
| Breast Cancer / Qwen2.5 0.5B / 4 | 86/114 | 80.7% | 75.3% | 0.769 | 85.9% | 24 | 2 | 114 + 86 |
| Breast Cancer / Qwen2.5 0.5B / 4 | 114/114 | 93.9% | 93.7% | 0.935 | 93.9% | 41 | 4 | 114 + 114 |
| Breast Cancer / Qwen3 4B / 0 | 0/114 | 61.4% | 52.0% | 0.479 | 61.4% | 0 | 0 | 114 + 0 |
| Breast Cancer / Qwen3 4B / 0 | 11/114 | 64.9% | 54.9% | 0.502 | 64.5% | 5 | 1 | 114 + 11 |
| Breast Cancer / Qwen3 4B / 0 | 29/114 | 68.4% | 58.6% | 0.552 | 69.4% | 9 | 1 | 114 + 29 |
| Breast Cancer / Qwen3 4B / 0 | 57/114 | 76.3% | 68.6% | 0.691 | 77.2% | 18 | 1 | 114 + 57 |
| Breast Cancer / Qwen3 4B / 0 | 86/114 | 85.1% | 80.2% | 0.823 | 85.2% | 28 | 1 | 114 + 86 |
| Breast Cancer / Qwen3 4B / 0 | 114/114 | 93.0% | 91.2% | 0.923 | 93.0% | 38 | 2 | 114 + 114 |
| Breast Cancer / Qwen3 4B / 4 | 0/114 | 86.0% | 88.7% | 0.858 | 86.0% | 0 | 0 | 114 + 0 |
| Breast Cancer / Qwen3 4B / 4 | 11/114 | 89.5% | 91.1% | 0.892 | 86.6% | 5 | 1 | 114 + 11 |
| Breast Cancer / Qwen3 4B / 4 | 29/114 | 91.2% | 92.0% | 0.909 | 87.8% | 8 | 2 | 114 + 29 |
| Breast Cancer / Qwen3 4B / 4 | 57/114 | 93.0% | 93.4% | 0.927 | 89.5% | 10 | 2 | 114 + 57 |
| Breast Cancer / Qwen3 4B / 4 | 86/114 | 93.0% | 93.4% | 0.927 | 91.3% | 10 | 2 | 114 + 86 |
| Breast Cancer / Qwen3 4B / 4 | 114/114 | 93.0% | 93.4% | 0.927 | 93.0% | 10 | 2 | 114 + 114 |
| Breast Cancer / Granite 3.3 2B / 4 | 0/114 | 61.4% | 69.0% | 0.606 | 61.4% | 0 | 0 | 114 + 0 |
| Breast Cancer / Granite 3.3 2B / 4 | 11/114 | 67.5% | 73.9% | 0.673 | 64.4% | 7 | 0 | 114 + 11 |
| Breast Cancer / Granite 3.3 2B / 4 | 29/114 | 76.3% | 81.0% | 0.763 | 69.2% | 17 | 0 | 114 + 29 |
| Breast Cancer / Granite 3.3 2B / 4 | 57/114 | 86.8% | 89.4% | 0.867 | 76.8% | 29 | 0 | 114 + 57 |
| Breast Cancer / Granite 3.3 2B / 4 | 86/114 | 92.1% | 92.3% | 0.917 | 84.6% | 38 | 3 | 114 + 86 |
| Breast Cancer / Granite 3.3 2B / 4 | 114/114 | 92.1% | 92.3% | 0.917 | 92.1% | 38 | 3 | 114 + 114 |
| Wine / Qwen3 4B / 4 | 0/36 | 80.6% | 82.1% | 0.809 | 80.6% | 0 | 0 | 36 + 0 |
| Wine / Qwen3 4B / 4 | 4/36 | 83.3% | 84.9% | 0.836 | 81.5% | 1 | 0 | 36 + 4 |
| Wine / Qwen3 4B / 4 | 9/36 | 83.3% | 84.9% | 0.836 | 82.6% | 1 | 0 | 36 + 9 |
| Wine / Qwen3 4B / 4 | 18/36 | 88.9% | 90.1% | 0.891 | 84.7% | 3 | 0 | 36 + 18 |
| Wine / Qwen3 4B / 4 | 27/36 | 88.9% | 90.1% | 0.891 | 86.8% | 3 | 0 | 36 + 27 |
| Wine / Qwen3 4B / 4 | 36/36 | 88.9% | 90.1% | 0.891 | 88.9% | 3 | 0 | 36 + 36 |

Never-review needs one source inference per row. Always-review additionally requests review for every valid source proposal; a failed proposal remains a pipeline failure with no fallback call. Direct Jev needs only its own request per row. Observed review reservation records are checked against valid proposals, so failed requests count and skipped source failures do not become requests. The JSON includes these counts and selection IDs. Curve counts are simulated logical inference/request requirements under the frozen one-attempt behavior, not new paid calls, measured wall-clock time, training compute, or dollar savings; source costs remain incomplete.

## Limits and next evidence needed

These are exploratory analyses chosen after earlier test results were seen. The coverage grid is fixed before this analysis is calculated, but was not preregistered before the original outcomes existed. Do not select a winning coverage on these test cases and call it validated. Unknown serving outcomes remain recorded failures. 0 incomplete conditions and all text reviews remain outside scored comparisons.

Review versus existing direct Jev changes prompt wording as well as adding a proposal, and uses separate serving occasions. The separate [matched proposal-control experiment](../review_controls/FINDINGS.md) changes only the proposal slot and reports its own execution status and comparisons; the historical direct-Jev arm is not that control. The familiar public datasets, one split, tiny Wine sample, restricted-label source scoring, repeated cases and mixed runtimes limit generalization. No architecture-wide or universal improvement claim follows.
