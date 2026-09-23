A 58-point accuracy gain made me ask whether we needed two models.

I tested Jev 1.13 as a bounded classifier and as a reviewer of an LLM’s proposed class, alongside LLMs alone and classical ML.

The setup: six source LLMs, zero-shot and four examples per class, across numerical Breast Cancer and Wine data, plus SST-2 sentiment and TREC question classification. Each method saw the same held-out rows within its dataset.

Two results stood out:

• On Wine with four examples per class, Qwen2.5 0.5B rose from 12/36 correct to 33/36 after Jev review: 33.3% → 91.7%. Yet all 36 final labels were identical to Jev alone. Logistic regression reached 35/36 using the same twelve task-specific training labels.

• On TREC, GPT-6 Astra’s few-shot result fell from 194/200 to 173/200: 97.0% → 86.5%. Jev corrected 3 mistakes and overrode 24 correct answers with wrong ones. Neither stage had a failed response in that condition.

Across 48 original review conditions, accuracy improved in 32, tied in 3 and declined in 13. All eight Astra conditions declined, including after failed-call recovery. These reuse four test sets, not independent replications.

Then I tested the proposal itself: the same row, examples and Jev prompt, with a real Qwen3 proposal, no proposal, or a shuffled one. Across 550 rows, adding the real proposal changed accuracy versus no proposal by 0 points on Breast Cancer and Wine, +0.5 on SST-2, and −0.5 on TREC. All four uncertainty intervals included zero. This pilot did not establish a reliable benefit from the proposal; it does not prove equivalence.

For a two-model pipeline, I now want two comparisons: does it beat the source alone, and does it beat the reviewer alone?

Returning an allowed category does not guarantee a better decision. Count the corrections, count the harmful overrides, and include the simpler baselines.

This is an exploratory pilot on 550 rows from familiar public datasets, one split per dataset and fixed prompt/scoring recipes. Several open-model runs collapsed to one class under our class-ID scoring recipe, so this is not a general ranking of those models. Pretraining exposure is not matched by giving methods the same task-specific labels.

How do you decide when a reviewer should be allowed to override the first model?

#MachineLearning #LLM #ModelEvaluation #TabularData

---

Publication draft, not posted. The control numbers above are original first-attempt results. See the full story for every real/no/shuffled arm, paired interval, repeat check and separate failure-recovery analysis.

Attach [the selected-example graphic](../completion_20260923/figures/selected_examples/when_second_model_helps.png). Use [the proposal-control graphic](../review_controls/figures/proposal_value.png) as a second slide. [All 48 review conditions](../completion_20260923/figures/review_outcomes.png) and [the full evidence story](STORY.md) provide context. The examples above remain unchanged by operational recovery.

The repository and dashboard are currently private. Add a repository link for readers only after access has been arranged. Do not imply that the code is already publicly accessible.
