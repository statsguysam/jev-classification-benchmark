# The story: does a second model add a second useful opinion?

The defensible contribution is an evaluation question: **does a reviewer improve its source, and does the source improve the reviewer?** Those are different comparisons. A bounded output guarantees membership in a category set, not correctness, complementary reasoning or a better pipeline.

## What the current evidence supports

| Observation | Evidence | Interpretation and limit |
|---|---|---|
| A dramatic rescue can be replacement | Wine, Qwen2.5 0.5B, four examples/class: source 33.3%, review 91.7%, direct Jev 91.7%; all 36 review labels equal direct Jev's labels | The pipeline's gain over this constant-label source does not establish added value beyond direct Jev. Existing direct and review prompts differ. |
| Review can damage a strong source | Wine, Astra, zero-shot: 36/36 correct before review, 16/36 after; 20 harmful overrides and no review API failures | A striking counterexample on 36 cases, not an architecture-wide verdict or an estimate across new datasets. |
| Selectivity may matter | Qwen3, four examples/class: reviewing the least-confident half retained full-review accuracy—93.0% on Breast Cancer, 88.9% on Wine | Cached, post-hoc simulation with 57/114 and 18/36 reviews. Source inference still runs for every row. Neither a validated gate nor measured financial savings. |
| Confidence routing is source-dependent | Qwen2.5 Breast Cancer few-shot ranking underperformed uniform random selection at every intermediate coverage | Publish all four eligible curves. The positive Qwen3 few-shot result does not generalize even to every source in this pilot. |
| Classical ML remains essential | Matched training: Breast Cancer random forest 95.6% with eight labels; Wine logistic regression 97.2% with twelve labels. Qwen3+Jev few-shot: 93.0% / 88.9% | Same test rows and training-example counts. Separate fixed recipes, no test-selected tuning. These named results are descriptive; choosing a winner after inspecting all models is not a selection protocol. |

All accuracy values are within-condition micro accuracy. Breast Cancer has 114 test cases and Wine 36, from one split. Examples are selected from training only. The machine reports also provide balanced accuracy and macro-F1. The existing numerical dashboard reports paired intervals; these repeated model comparisons are not independent replications. Numerical review is currently complete for 17/24 conditions. Incomplete conditions are not scored. Text has source, direct-Jev and classical results, but no completed source-to-Jev reviews.

## Sensitivity to the eligibility rule

The main four-curve analysis excludes sources with constant predicted labels. That is a post-hoc scope choice, not a mathematical reason their scores cannot rank rows. The [sensitivity appendix](../review_value_sensitivity/FINDINGS.md) includes all nine completed local-model conditions with probability vectors, adding all five constant-label conditions and publishing every coverage point. Their maximum scores vary despite constant predicted classes. Among those five added conditions, one is above the matched random accuracy expectation at every intermediate coverage, two are below, one is mixed and one is tied. These are dependent descriptive comparisons, not independent wins and losses or evidence for a universal gate. The original two Qwen3 few-shot half/full equalities are unchanged.

## The experiment that can strengthen the explanation

The follow-up freezes 550 cases and compares three **otherwise identical** reviewer prompts: the real Qwen3 few-shot proposal, no proposal, and a shuffled proposal. Shuffling preserves the marginal class counts and uses no test labels. Sixty-four exact repeated no-proposal calls assess serving variability separately. Together: 1,714 planned requests, **zero executed** in this snapshot.

This isolates proposal content more closely than the earlier direct-Jev comparison, which changed wording too. The analysis specifies paired differences and uncertainty before these new calls, but earlier results on the same public holdouts have already been inspected: this remains exploratory, not a fresh confirmatory test.

Interpretation is conditional on the eventual result:

- If real proposals reliably outperform both controls, the source contributes information in this setup.
- If all three perform similarly, Jev may mostly classify from the row and demonstrations; absence of a significant difference alone is not proof of equivalence.
- If shuffled proposals degrade performance, irrelevant proposals can influence decisions. Serving repeats help quantify one alternative explanation, but do not by themselves identify an internal mechanism.
- If real proposals harm performance, the pipeline needs a better decision rule; a second model is not automatically beneficial.

No paid outcomes are assumed. Existing unfinished reviews take priority and all new calls remain within the US$25 authorization, subject to actual remaining conservative headroom and provider funding.

## Measurement deserves its own check

The source audit found constant predictions in 11/16 numerical open-model conditions and 3/16 text conditions. Local source scores normalize the joint likelihood of a numeric class ID and an end-of-sequence token. These are not calibrated probabilities of correctness. High score concentration therefore cannot be treated as proof of a confident, capable classifier.

The completed [validation-only diagnostic](validation_scoring/AUDITED_FINDINGS.md) compares class-ID-only and class-ID-plus-EOS scores from the same forward pass per candidate for SmolLM2 and Granite. Removing EOS changed **0 of 136 class decisions** on 34 distinct validation cases reused across two models and two shot settings. Seven of eight conditions remained constant-class. This does not support EOS alone as the explanation for collapse on these selected examples; it does not prove that scoring format, prompts or free generation are interchangeable. Probability vectors can still differ even when the selected class does not.

These validation samples are balanced (16 Breast Cancer, 18 Wine), so their absolute accuracies must not be compared directly with earlier test accuracies under different class prevalence. The 34 labels add a diagnostic development budget. No historical test prediction, scoring method or gate was selected or changed. The audited report excludes unmeasured latency fields; the raw evaluator's zero timing placeholders are preserved and explicitly identified as unmeasured.

## Position relative to prior work

[QuicqDev/Jev-vs-ML](https://github.com/QuicqDev/Jev-vs-ML/tree/6c261e8d6c7a46b26dd582d00d6da23430975bcc) already compares Jev with classical methods across text and tabular tasks. Its completed V3 study is useful context; its V4 plans are not completed evidence. Our contribution is the explicit source → reviewer pipeline, comparisons against both stages alone, correction/harm accounting and planned proposal controls. Do not claim the first Jev-versus-ML benchmark or directly compare leaderboard numbers across different splits, prompts, routes and label budgets.

## Sharing package

Use the [LinkedIn draft](LINKEDIN_DRAFT.md), [all-four-curves graphic](figures/selective_review.png), [detailed findings](FINDINGS.md) and interactive review-value dashboard. The GitHub repository and hosted dashboard currently remain private; a reader needs access to follow those links. The graphic contains no raw inputs. Publish measured observations with sample sizes, and describe the controlled follow-up as pending.
