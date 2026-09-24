I tested Jev to understand where it helps—and where a simpler classifier is enough.

The question was practical: how well does it handle bounded decisions, such as yes/no or a fixed category, compared with LLMs and classical ML?

Zero-shot classification makes experimentation easier. Choosing a model still requires labeled evaluation data, checks on its confidence scores, and a fair comparison with practical alternatives. A small trained classifier belongs in that comparison too.

So I tested:
• LLMs alone — open-weight models: Qwen2.5 0.5B, Qwen3 4B, SmolLM2 1.7B and IBM Granite 3.3 2B; hosted OpenAI models: GPT-5.6 Luna and GPT-6 Astra.
• Jev alone — TypeSafe’s Jev 1.13.
• LLM → Jev review — each LLM’s proposed class reviewed by Jev 1.13.
• Classical ML — XGBoost, LightGBM, logistic regression and random forest.

Six source LLMs, zero-shot and four examples per class, across numerical Breast Cancer and Wine data, plus SST-2 sentiment and TREC question classification. Each method saw the same test rows within its dataset; matched-label and full-training ML references stayed separate.

Three findings stood out:

1. A big rescue didn’t demonstrate a benefit from using both models.

On Wine, few-shot Qwen2.5 0.5B improved from 33.3% to 91.7% after Jev review: 12/36 → 33/36 correct.

But every final label matched Jev alone. Logistic regression reached 35/36 using the same twelve training labels.

2. Review could make a strong result worse.

On TREC, few-shot GPT-6 Astra fell from 97.0% to 86.5%: 194/200 → 173/200.

Jev corrected 3 mistakes and replaced 24 correct answers with wrong ones. Neither stage had a failed response.

3. The proposal itself needed testing.

Keeping the row, examples and Jev prompt wording fixed, I compared real Qwen3 proposals, no proposal, and shuffled proposals.

Real versus no proposal changed first-attempt accuracy by −0.5 to +0.5 percentage points across the four datasets. All four paired 95% intervals included zero. After separate failure recovery, all four intervals still included zero. This does not prove equivalence.

My takeaway: assess a two-model pipeline against both stages alone and credible classical baselines. Count what the reviewer fixes—and what it breaks.

An allowed label can still be wrong. Confidence needs calibration checks. Cost and latency need their own measurements. Extra complexity should earn its place.

This is an exploratory pilot: 550 distinct test rows, one split per dataset, familiar public data and fixed recipes. Several small-model runs predicted one class throughout under our scoring method; this is not a general model ranking. Equal task labels don’t equalize pretraining. We haven’t established calibration or production cost savings here.

When does adding Jev as a reviewer improve the final decision enough to justify the extra step?

#MachineLearning #LLM #ModelEvaluation #TabularData

---

Publication draft, not posted. General evaluation principles are expressed in original wording alongside the independently measured benchmark findings.

The control point estimates above are original first-attempt results. After recovery the four real-minus-none differences were +0.88 pp (Breast Cancer), −2.78 pp (Wine), +0.50 pp (SST-2), and −0.50 pp (TREC); all four paired intervals still include zero. See the full story for every real/no/shuffled arm, paired interval, repeat check and separate failure-recovery analysis. The selected Wine and TREC examples remain unchanged by operational recovery. The cost/calibration discussion motivates evaluation; it is not a claim that the focused experiment established either property.

Attach [the selected-example graphic](../completion_20260923/figures/selected_examples/when_second_model_helps.png). Use [the proposal-control graphic](../review_controls/figures/proposal_value.png) as a second slide. [All 48 review conditions](../completion_20260923/figures/review_outcomes.png) and [the full evidence story](STORY.md) provide context.

The repository and dashboard remain private. Add a repository link for readers only after access has been arranged. Do not imply that the code is already publicly accessible.
