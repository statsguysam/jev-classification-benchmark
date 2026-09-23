# What the numerical classification experiment shows

**Status, 23 September 2026 UTC:** all 68/68 conditions and 72/72 contrasts are audited, including all 24 source-model and 24 review conditions. Historical failures remain in the first-attempt scores, including six Breast Cancer SmolLM2 four-shot billing failures. Any subsequent recovery results are separate; no original outcome is replaced and no partial-test score is reported.

The expansion compares six source LLMs—Qwen2.5 0.5B, Qwen3 4B, SmolLM2 1.7B, Granite 3.3 2B, GPT-5.6 Luna and GPT-6 Astra—on the same serialized numerical rows from Breast Cancer Wisconsin Diagnostic and Wine. Each model is evaluated with zero examples and four examples per class. Jev then reviews its saved proposed label using the original row and the same examples. Direct Jev and native-feature logistic regression, random forest, XGBoost and LightGBM provide references. Four examples per class mean eight labeled training rows for Breast Cancer and twelve for Wine. This expansion adds no LoRA training.

The most useful finding is that review is a separate decision stage whose value must be measured. It can correct weak proposals, reproduce what Jev would already predict alone, or replace a correct source prediction with a wrong one. Producing a valid choice from a bounded class set does not guarantee that the choice is correct.

## All six models: zero-shot and few-shot

Each entry shows **LLM alone → LLM followed by Jev**, as accuracy. Few-shot means four training examples per class for both stages. Full metrics, paired intervals and failure counts are in [FINDINGS.md](FINDINGS.md).

### Breast Cancer · 114 test rows · eight few-shot labels

| Source model | Zero-shot | Few-shot |
|---|---:|---:|
| Qwen2.5 0.5B | 37.7% → 85.1% | 61.4% → 93.9% |
| SmolLM2 1.7B | 37.7% → 86.8% | 37.7% → 89.5% |
| Granite 3.3 2B | 37.7% → 86.0% | 61.4% → 92.1% |
| Qwen3 4B | 61.4% → 93.0% | 86.0% → 93.0% |
| GPT-5.6 Luna | 75.4% → 87.7% | 91.2% → 92.1% |
| GPT-6 Astra | 99.1% → 95.6% | 98.2% → 93.9% |
| Jev alone | 84.2% | 93.0% |

### Wine · 36 test rows · twelve few-shot labels

| Source model | Zero-shot | Few-shot |
|---|---:|---:|
| Qwen2.5 0.5B | 33.3% → 33.3% | 33.3% → 91.7% |
| SmolLM2 1.7B | 33.3% → 33.3% | 33.3% → 88.9% |
| Granite 3.3 2B | 27.8% → 33.3% | 33.3% → 91.7% |
| Qwen3 4B | 38.9% → 36.1% | 80.6% → 88.9% |
| GPT-5.6 Luna | 47.2% → 38.9% | 88.9% → 91.7% |
| GPT-6 Astra | 100.0% → 44.4% | 97.2% → 94.4% |
| Jev alone | 33.3% | 91.7% |

Seven of the eight new SmolLM2/Granite source conditions predicted one class. Only Granite Breast Cancer few-shot varied (87 class-0 and 27 class-1 predictions). All 600 probability vectors were finite, normalized, and matched their argmax labels; there were no inference failures or truncated prompts. These weak results are retained.

At zero-shot Breast Cancer, Qwen 0.5B, SmolLM2 and Granite supplied identical proposal vectors. At zero-shot Wine, Qwen 0.5B and SmolLM2 matched; at few-shot Wine, Qwen 0.5B, SmolLM2 and Granite matched. Their Jev prompts are identical within each group. Those historical calls occurred at different serving times. Their differences cannot be attributed to a different visible source proposal; these overlapping comparisons were not designed as independent replicates or a repeatability estimate.

The small open-model results describe the recorded class-label likelihood recipe, not the models’ general capabilities. Their source prompts and serving protocols are documented explicitly.

## A bounded review can damage correct answers

On Wine with zero examples, Astra answered all 36 held-out rows correctly. After Jev review, 16 remained correct: **100% became 44.4%**. The reviewer changed 20 correct labels into wrong labels and corrected no cases. There were no source or reviewer failures in this condition, and the ordered test rows and label mapping were identical. Ten true cultivar-2 rows and all ten true cultivar-3 rows were reassigned to cultivar 1.

The paired accuracy change was **−55.56 percentage points**, with an exploratory 95% bootstrap interval of **[−72.22, −38.89]**. This is an observed failure case for this review setup; it does not establish a universal rule about Jev or review systems.

The reviewed pipeline still exceeded direct Jev's 12/36 correct predictions. That comparison alone would conceal the substantial loss from its 36/36 source. Both comparisons are necessary: review versus the source, and review versus Jev alone.

In all four Astra conditions, review reduced the correct counts from 113 to 109 and 112 to 107 on Breast Cancer, and from 36 to 16 and 35 to 34 on Wine, for zero and four examples per class respectively. No case was corrected in those four conditions. Wine's few-shot pipeline inherited one source failure; the additional loss was one actual wrong-label replacement.

## A large improvement over a weak source may add nothing over direct Jev

Qwen2.5 0.5B on Wine with four examples per class improved from **12/36 to 33/36**, or **33.3% to 91.7%**, after review. Jev corrected 21 cases and harmed none. The paired change was **+58.33 percentage points**, with an exploratory 95% bootstrap interval of **[+41.67, +75.00]**.

However, all 36 reviewed labels were identical to direct Jev's labels at the same shot count. The pipeline's improvement over Qwen therefore provides no observed accuracy gain over using Jev directly in this condition. A deployed chain would also require the source stage's cost and latency, even though this experiment reused saved proposals.

A similar distinction appears on Breast Cancer zero-shot: Qwen2.5 0.5B improved from 43/114 to 97/114, correcting 54 cases without harming any, while direct Jev already achieved 96/114. The review-minus-direct-Jev change was only **+0.88 percentage points**, with interval **[−4.39, +6.14]**. That interval does not establish either an improvement or equivalence beyond the observed rows.

Adding labeled examples also changes the question. Direct Jev on Wine moved from 12/36 at zero-shot to 33/36 with twelve examples. These are different newly supplied label budgets. They should not be pooled as interchangeable estimates of model performance.

## Small native-feature baselines remain strong

All native methods use the same training-row selection and held-out records, with preprocessing fitted only on the permitted training rows. Matched-label results are distinct from full-training results:

| Dataset | Training labels | Logistic regression | Random forest | XGBoost | LightGBM |
|---|---:|---:|---:|---:|---:|
| Breast Cancer | 8 | 108/114 (94.7%) | 109/114 (95.6%) | 101/114 (88.6%) | 105/114 (92.1%) |
| Wine | 12 | 35/36 (97.2%) | 34/36 (94.4%) | 26/36 (72.2%) | 30/36 (83.3%) |
| Breast Cancer, full training | 341 | 113/114 (99.1%) | 112/114 (98.2%) | 111/114 (97.4%) | 110/114 (96.5%) |
| Wine, full training | 106 | 35/36 (97.2%) | 36/36 (100%) | 34/36 (94.4%) | 34/36 (94.4%) |

These are fixed recipes, not a search for the strongest possible classical classifier. The matched logistic-regression and random-forest results prevent a broad claim that Jev or an LLM pipeline outperforms classical ML here. Full-training references use more labels and answer a different resource-budget question.

## What the comparisons do not identify

The reviewer receives the row, examples and proposed class ID, but no source-model name, confidence or rationale. At the same row and shot count, identical proposed labels produce identical review prompts. Their separate API calls may still differ. Small differences between such pipelines cannot be assigned to source architecture, and equal overall accuracies alone do not establish that the underlying proposals are identical.

Review and direct Jev differ in prompt wording, the presence of a proposal and separate calls. This is not a controlled experiment isolating bounded decoding or the proposal's causal effect. Original Qwen/OpenAI source prompts retain their earlier model/provider rendering; the new SmolLM2/Granite sources use a fixed explicit system message. This is also not a same-prompt architecture comparison. Scores from label likelihoods, Jev Choice and classical estimators are not assumed to be calibrated or interchangeable. Mixed local, Colab and hosted timings do not support a controlled speed ranking.

All failures remain incorrect in the full denominators of 114 Breast Cancer and 36 Wine rows. Paired intervals use 2,000 resamples, seed 42, and the same sampled held-out groups for both conditions. These numeric test folds have no duplicate feature groups. Intervals condition on this split, selected examples, prompts and fitted models, and are exploratory without adjustment for multiple comparisons. An interval containing zero is not evidence of equivalence.

The expansion was chosen after earlier results were observed. It covers two familiar public datasets and one split/seed; pretraining exposure cannot be excluded. Wine cultivar IDs have arbitrary meaning without examples, so zero-shot Wine performance chiefly probes familiarity and inference from this representation, not supervised learnability. Even 36/36 correct predictions do not establish general numerical-classification ability. The useful takeaway is to evaluate the reviewer against both its source and a direct-classifier reference, while keeping simple native-feature baselines in view.

## Reproduction and accounting

See the [protocol](../../docs/EXPANDED_NUMERIC_PROTOCOL.md), [Colab guide](../../docs/EXPANDED_NUMERIC_REPRODUCTION.md), [dashboard](https://jev-benchmark-observatory.statsguysalim.chatgpt.site), and [numeric LinkedIn draft](LINKEDIN_DRAFT.md). The [main publication draft](../review_value/LINKEDIN_DRAFT.md) integrates the broader study; this numeric report makes no text or proposal-control claim. Verify public link access before publication.

The 1,500 new Jev requests have US$0.124087194 of known reported charges and 8 unknown-cost calls. The expansion retains US$4.032000000 in conservative reservations. Its report's cumulative subtotal is US$23.357827700; adding the separate US$0.002688000 health probe gives US$23.360515700 before text completion, retries and controls. This is a scoped subtotal, not the live global spend; those phases share the unchanged US$25.00 ceiling. Unknown charges are not zero, these figures are not a provider invoice, and local compute is unpriced.

The [selective-review analysis](../review_value/FINDINGS.md) now contains all five eligible primary curves. Its [sensitivity appendix](../review_value_sensitivity/FINDINGS.md) includes all 16 probability-bearing local conditions, including 11 constant-label sources. It retains the two Qwen3 four-shot half-review/full-review accuracy matches, alongside counterexamples that perform below random selection. Both analyses are post-hoc fixed-coverage simulations on cached outcomes, not validated confidence gates or evidence of dollar savings.
