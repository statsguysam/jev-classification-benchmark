A 58-point accuracy gain made me ask whether we needed two models.

Shirin Khosravi Jam’s recent Jev discussion asked “compared to what?” It resonated with a benchmark I’d been running.

Jev makes it easier to start classifying without training a model. But we still need representative evaluation data, calibration checks for confidence scores, and comparisons with smaller classifiers.

I compared an LLM alone, Jev alone, and the LLM’s proposed class reviewed by Jev, alongside XGBoost, LightGBM, logistic regression and random forest.

Six source LLMs. Zero-shot and four examples per class. Numerical Breast Cancer and Wine data, plus SST-2 sentiment and TREC question classification. Same held-out rows within each dataset, with matched-label and full-training ML references kept separate.

Three findings stood out:

• A large improvement didn’t demonstrate added value from the chain. On Wine, few-shot Qwen2.5 0.5B went from 12/36 correct to 33/36 after Jev review: 33.3% → 91.7%. Yet every final label matched Jev alone. Logistic regression reached 35/36 using the same twelve training labels.

• Review could undo good decisions. On TREC, few-shot GPT-6 Astra fell from 194/200 to 173/200: 97.0% → 86.5%. Jev corrected 3 mistakes and replaced 24 correct answers with wrong ones. Neither stage had a failed response.

• The proposed class didn’t show a consistent benefit. With the same row, examples and Jev prompt, I tested real Qwen3 proposals, no proposal, and shuffled proposals. Real versus no proposal changed first-attempt accuracy by −0.5 to +0.5 percentage points across the four datasets. All four paired 95% intervals included zero, including after separate failure recovery. This does not prove equivalence.

My takeaway: evaluate the chain against BOTH stages alone, and keep classical ML in the comparison. Count corrections and harmful overrides separately.

This was an exploratory pilot: 550 distinct test rows, one split per dataset, familiar public data and fixed recipes. Several small-model runs predicted one class throughout under our scoring method, so these results are not a general model ranking. Equal task labels don’t equalize pretraining. These findings concern accuracy, not proof of calibrated confidence or production cost savings.

When does a second model earn the right to override the first?

Credit to Shirin for the discussion:
https://www.linkedin.com/posts/shirin-khosravi-jam_some-thoughts-on-jev-as-a-data-scientist-activity-7508196268746211328-oMNj

#MachineLearning #LLM #ModelEvaluation #TabularData

---

Publication draft, not posted. The experiment predates reading Shirin’s discussion; this draft credits the related framing without claiming her post initiated the study. The linked source includes her written explanation and an automated transcript; the Instagram caption was accessible, but exact cross-platform video identity was not independently verified.

The control point estimates above are original first-attempt results. After recovery the four real-minus-none differences were +0.88 pp (Breast Cancer), −2.78 pp (Wine), +0.50 pp (SST-2), and −0.50 pp (TREC); all four paired intervals still include zero. See the full story for every real/no/shuffled arm, paired interval, repeat check and separate failure-recovery analysis. The selected Wine and TREC examples remain unchanged by operational recovery. The cost/calibration discussion motivates evaluation; it is not a claim that the focused experiment established either property.

Attach [the selected-example graphic](../completion_20260923/figures/selected_examples/when_second_model_helps.png). Use [the proposal-control graphic](../review_controls/figures/proposal_value.png) as a second slide. [All 48 review conditions](../completion_20260923/figures/review_outcomes.png) and [the full evidence story](STORY.md) provide context.

The repository and dashboard remain private. Add a repository link for readers only after access has been arranged. Do not imply that the code is already publicly accessible.
