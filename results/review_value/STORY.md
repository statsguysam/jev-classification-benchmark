# Does a second model add a useful second opinion?

A large gain after review is only the start of the evaluation. We need to ask both **whether the reviewer improves its source and whether the source improves the reviewer**. This pilot makes those comparisons visible alongside the individual corrections, harmful overrides and failed responses.

Both original matrices are complete: **68 numerical conditions and 68 text conditions**, with 72 paired contrasts in each. Together they include 48 source-to-Jev review conditions across six source LLMs, four datasets and zero or four training examples per class. The same 114 Breast Cancer, 36 Wine, 200 SST-2 and 200 TREC cases recur across their respective conditions. These are 550 held-out rows on one fixed split per dataset, not thousands of independent test cases. Original results remain visible; operational recovery is reported separately.

Across the 48 review conditions, accuracy improved over the source in 32, tied in three and declined in 13. Compared with historical Jev-alone calls, the counts were 21 improvements, 16 ties and 11 declines. These are descriptive counts of dependent conditions, not a significance test or a general model ranking. The [complete first-pass audit](../completion_20260923/FIRST_PASS_EVIDENCE_REVIEW.md) publishes every condition and verifies the original prediction files.

## A large rescue did not demonstrate a benefit from the chain

On Wine with four examples per class, Qwen2.5 0.5B scored **12/36 (33.3%)**. After Jev review it scored **33/36 (91.7%)**: 21 corrections, no harmful overrides and no failures. The gain was 58.3 percentage points, with an exploratory paired 95% bootstrap interval of [41.7, 75.0].

But **all 36 reviewed labels matched the labels from Jev alone**, which also scored 33/36. The chain delivered no observed incremental accuracy over direct Jev in this condition. Its source had predicted one class throughout the test fold. This is a useful rescue of that recorded source output, but it does not establish that using both models added value beyond direct classification. The historical direct and review prompts also differ, so that comparison alone does not isolate the proposal's effect. See the [complete numerical findings](FINDINGS.md) and [numeric evidence audit](../completion_20260923/NUMERIC_EVIDENCE_REVIEW.md).

The classical references make the resource comparison more concrete. Using the same selected training rows and held-out cases, Wine logistic regression achieved **35/36 (97.2%) with twelve labels**, and Breast Cancer random forest achieved **109/114 (95.6%) with eight labels**. These fixed native-feature recipes use the same newly supplied labels as the four-examples-per-class LLM conditions. They are descriptive references, not a test-selected deployment winner; full-training models use a separate, larger label budget.

## The reviewer can also undo correct decisions

Wine Astra zero-shot moved in the opposite direction: **36/36 became 16/36**, with 20 correct source labels replaced by accepted wrong labels, no corrections and no source or review failures. The paired change was −55.6 percentage points, with an exploratory 95% interval of [−72.2, −38.9]. The reviewed pipeline still beat direct Jev's 12/36, illustrating why comparison against only one of the two stages can conceal substantial damage.

This is not the only completed Astra counterexample. Across all eight Astra conditions—four datasets at two example settings—review reduced accepted-correct counts in every condition. These are **four reused test sets, not eight independent replications**, and the direction count is not a significance test. The [Astra review audit](../completion_20260923/ASTRA_REVIEW_AUDIT.md) publishes all eight rows, correction/harm partitions and source-file hashes.

TREC with four examples per class provides a larger, failure-free example: Astra's **194/200 (97.0%) became 173/200 (86.5%)**. Review corrected three cases but replaced 24 correct labels with accepted wrong labels. Both stages had zero failures. The adverse net change of 21 decisions follows from the recorded class decisions themselves.

Some other conditions include rejected responses, which require a different description. In Astra TREC zero-shot, review produced 84 accepted-correct decisions and ten probability-validation failures. Even granting every failed row a correct result gives **94/200 (47.0%)**, below the source's 194/200 (97.0%); another 100 source-correct rows became accepted wrong labels. This optimistic response-validity bound holds accepted decisions fixed. It is neither a confidence interval nor a prediction of retry recovery. The same bound remains below the source in seven Astra conditions; Wine four-shot ties only after hypothetically making its inherited, never-called upstream skip correct.

The [output-validation notes](../../docs/OUTPUT_VALIDATION_NOTES.md) distinguish wrong-label harms from failure harms. A rejected probability vector is not an established wrong class decision: saved errors retain neither the returned class nor the vector, so they cannot establish correctness, discrepancy size or rounding as the cause. First-attempt pipeline accuracy keeps those failures in the full denominator. Accepted choices belong to the declared class set; response validity and classification correctness still require separate evaluation.

## Recovery did not remove the adverse Astra pattern

The fixed recovery policy made 71 additional calls: 70 Jev calls and one Astra call. It recovered 65 of 66 originally failed paid requests and supplied the first Jev decision for a previously skipped Wine review. One Astra TREC zero-shot review still failed probability validation after both permitted retries. Every original valid response remained unchanged, including wrong answers; retries stopped at the first valid response without consulting truth labels.

After recovery, **all eight Astra reviewed pipelines still scored below their corresponding recovered source**. For Wine four-shot, the correct comparison is the recovered source's 36/36 against the newly completed pipeline's 35/36. TREC zero-shot rose from 84/200 to 89/200 but remained below its source's 194/200. The failure-free Wine and TREC examples above did not change. The [recovery findings](../completion_20260923/RECOVERY_FINDINGS.md) and [independent recovery audit](../completion_20260923/RECOVERY_EVIDENCE_REVIEW.md) preserve all affected full-condition views. This is a sensitivity analysis of response availability, not an improvement achieved by retrying valid wrong answers.

## Selective review is a secondary, exploratory result

The original first-attempt numerical outcomes, without recovery overlays, also permit fixed-coverage simulations: rank rows by ascending source maximum probability, break ties by row hash, and review predetermined fractions. All **five primary curves** are reported. At the same 50% coverage point:

| Dataset / source / examples per class | Review requests | Accuracy at 50% | Random-selection expectation | Full-review accuracy |
|---|---:|---:|---:|---:|
| Breast Cancer / Qwen2.5 0.5B / 4 | 57/114 | 70.2% | 77.6% | 93.9% |
| Breast Cancer / Qwen3 4B / 0 | 57/114 | 76.3% | 77.2% | 93.0% |
| Breast Cancer / Qwen3 4B / 4 | 57/114 | 93.0% | 89.5% | 93.0% |
| Breast Cancer / Granite 3.3 2B / 4 | 57/114 | 86.8% | 76.8% | 92.1% |
| Wine / Qwen3 4B / 4 | 18/36 | 88.9% | 84.7% | 88.9% |

The two Qwen3 few-shot conditions match full-review accuracy at half the review requests. Qwen2.5 Breast Cancer ranks perform below the random-selection expectation at every intermediate coverage. These are retrospective batch simulations on saved responses, not a validated confidence threshold or measured financial saving. Source inference is still needed for every row. Failure and skipped-call accounting remains in the machine report; simulated request counts are not additional paid calls.

Excluding constant-label sources from the primary analysis was a post-hoc scope restriction. The [sensitivity appendix](../review_value_sensitivity/FINDINGS.md) therefore includes **all 16 eligible local probability-bearing conditions**: the five primary conditions and eleven constant-label additions. All eleven have varying maximum scores. Among those additions, two exceed the random accuracy expectation at every intermediate coverage, four fall below throughout, three are mixed and two tie throughout. All six coverages are published; these dependent descriptive signs do not establish a general routing rule. The two Qwen3 half/full equalities survive this broader scope unchanged.

## Measurement and scope constrain the interpretation

The source audit found constant predictions in 11/16 numerical open-model conditions and 3/16 text source conditions. Local scores normalize the joint likelihood of a numeric class ID and an end-of-sequence token; they are not calibrated probabilities of correctness. These results describe that scoring recipe, not each model's general capability.

The completed [validation-only diagnostic](validation_scoring/AUDITED_FINDINGS.md) compared class-ID-only and class-ID-plus-EOS scoring for SmolLM2 and Granite. Removing EOS changed **0 of 136 decisions** on 34 distinct validation cases reused across models and example settings; seven of eight conditions remained constant-class. This does not support EOS alone as the explanation on these examples. It does not establish that other scoring formats, prompts or free generation are interchangeable. The balanced validation samples have different class prevalence from the test sets, add 34 diagnostic labels, and provide no measured latency comparison. No historical test prediction or gate was changed.

The reviewer receives the original row, the same demonstrations and a cached class ID, without source identity, confidence or rationale. Equal proposed labels on the same row therefore create identical review prompts. Historical differences among such calls span serving times and overlapping source pairs; they are not designed independent replicates or a repeatability estimate.

All quoted accuracies use full within-condition denominators; balanced accuracy and macro-F1 are available in the reports. Numerical paired intervals condition on the saved split, examples, prompts and responses and are unadjusted across comparisons. Familiar public datasets, possible pretraining exposure, arbitrary zero-shot Wine cultivar IDs, tiny test folds and mixed runtimes limit generalization. Neither apparent equality nor a zero-spanning interval establishes equivalence.

## Matched prompts did not show a consistent benefit from the proposal

The completed [frozen controls](../review_controls/FINDINGS.md) compare otherwise identical Jev prompts with actual Qwen3 4B proposals, no proposal, or proposals shuffled between rows. All use four examples per class. They cover 550 cases across four datasets: 1,650 primary requests plus 64 exact repeated no-proposal calls, totaling **1,714 completed requests**. Historical direct-Jev calls are not substituted for the matched no-proposal arm. The [independent control audit](../completion_20260923/CONTROL_EVIDENCE_REVIEW.md) verifies all prompt identities, primary metrics, intervals, repeats and recovery lineage.

| Dataset | Real proposal | No proposal | Shuffled proposal | Failures: real / none / shuffled |
|---|---:|---:|---:|---:|
| Breast Cancer (114) | 106/114 (93.0%) | 106/114 (93.0%) | 106/114 (93.0%) | 1 / 0 / 2 |
| Wine (36) | 31/36 (86.1%) | 31/36 (86.1%) | 32/36 (88.9%) | 1 / 2 / 0 |
| SST-2 (200) | 192/200 (96.0%) | 191/200 (95.5%) | 188/200 (94.0%) | 1 / 1 / 2 |
| TREC (200) | 172/200 (86.0%) | 173/200 (86.5%) | 176/200 (88.0%) | 1 / 1 / 2 |

| Dataset | Real minus none, pp [95% interval] | Real minus shuffled, pp [95% interval] |
|---|---:|---:|
| Breast Cancer | 0.00 [−4.39, +4.39] | 0.00 [−3.51, +3.51] |
| Wine | 0.00 [−13.89, +13.89] | −2.78 [−8.33, 0.00] |
| SST-2 | +0.50 [−1.00, +2.50] | +2.00 [0.00, +4.50] |
| TREC | −0.50 [−3.00, +1.50] | −2.00 [−4.00, −0.50] |

Adding the real proposal produced observed accuracy differences of **0, 0, +0.5 and −0.5 percentage points versus no proposal**. All four paired intervals include zero. This pilot therefore does not establish a reliable accuracy benefit from adding this source proposal. It does not prove equivalence, especially with Wine's small sample and wide interval. The actual-versus-shuffled results vary in direction; the TREC interval lies below zero, but these eight exploratory intervals are unadjusted and conditional on one shuffle. They do not establish that shuffling is a better general strategy.

Shuffling preserves class counts and does not guarantee a wrong proposal: the assigned class remains unchanged on 64/114 Breast Cancer, 12/36 Wine, 86/200 SST-2 and 54/200 TREC rows. The experiment changes proposal alignment, not the underlying input or examples, and cannot identify an internal anchoring mechanism.

In the predefined repeat diagnostic, **62/62 pairs with two valid responses returned the same class**. Of the other two pairs, one had a failed reference and one had both calls fail; these are not counted as agreement. All 64 pairs are accounted for. This is a small serving check, not proof of deterministic behavior. Primary scoring retains all 14 primary failures as incorrect; one additional failure occurred among the 64 repeat calls.

The [separate recovery sensitivity](../completion_20260923/CONTROL_RECOVERY_FINDINGS.md) resolved all 15 failed control calls with 15 additional calls. Recovered real-minus-none accuracy differences were +0.88 points on Breast Cancer, −2.78 on Wine, +0.50 on SST-2 and −0.50 on TREC; all four paired intervals still include zero. The original TREC real-minus-shuffled interval excluded zero, while its recovered interval [−3.50, 0.00] reaches zero. That sensitivity further limits any claim that shuffled proposals outperform real ones. All twelve recovered arms and eight contrasts are published alongside the originals. Neither the primary results nor the original repeat diagnostic is overwritten.

These holdouts had already been inspected, and the experiment uses one split and one shuffle with Qwen3 four-shot proposals. It remains an exploratory test of this pipeline, not a general verdict on Jev or multi-model systems. The complete study's [conservative accounting](../completion_20260923/COSTS.md) is **US$24.110484775 against a US$25 ceiling**, including retained failed-call reservations; it is not a provider invoice and does not price local compute.

## Position relative to prior work

[QuicqDev/Jev-vs-ML](https://github.com/QuicqDev/Jev-vs-ML/tree/6c261e8d6c7a46b26dd582d00d6da23430975bcc) already compares Jev with classical methods across text and tabular tasks. Its completed V3 study is useful context; its V4 plans are not completed evidence. Our contribution is the explicit source → reviewer pipeline, comparisons against both stages alone, correction/harm accounting and completed matched proposal controls. We make no first-ever benchmark claim or direct leaderboard comparison across different splits, prompts, routes and label budgets.
