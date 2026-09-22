# Expanded numerical classification and Jev review protocol

This is an exploratory expansion selected after the tabular and first numerical-supplement results were observed. Model selection, new conditions and contrasts are post-hoc; none is described as preregistered. Earlier measurements remain unchanged and are reused by their exact saved artifact hashes. There is no new LoRA or QLoRA training in this expansion.

## Frozen data and representation

The two tasks are Breast Cancer Wisconsin Diagnostic, a binary task with 30 numerical features, and Wine, a three-class task with 13 numerical features. They use the exact prepared native/serialized bundles from [TABULAR_PROTOCOL.md](TABULAR_PROTOCOL.md). The held-out sets contain 114 and 36 records, respectively. Train/validation/test assignments, labels, feature order, missing-value representation, numeric precision, fixed task descriptions and text serialization are unchanged. No test row is resampled or discarded to fit a new budget.

Zero-shot supplies no new labeled examples. Four-per-class conditions use the exact frozen `select_examples` selector with seed 42: eight labels for Breast Cancer and 12 for Wine. Source LLM and Jev review use the same shot count, ordered demonstration IDs and full held-out rows. No validation labels enter this expansion. Full-training classical references use 341 and 106 labels, respectively, and remain separate from the matched-label comparisons.

## Inventory: 68 conditions

| Family | Conditions | Count |
|---|---|---:|
| Source LLM | Six models × two datasets × zero/four-per-class | 24 |
| Source proposal → Jev review | Matching 24 source conditions | 24 |
| Jev direct reference | Two datasets × zero/four-per-class | 4 |
| Native classical reference | Logistic regression, random forest, XGBoost, LightGBM × two datasets × four-per-class/full training | 16 |

The six source models are Qwen2.5 0.5B Instruct, Qwen3 4B Instruct 2507, SmolLM2 1.7B Instruct, Granite 3.3 2B Instruct, GPT-5.6 Luna and GPT-6 Astra. The 16 original Qwen/OpenAI source runs are reused; eight SmolLM2/Granite runs are new. Four original Qwen3 4B/Astra few-shot reviews are reused, with 20 new review conditions. Native references and direct Jev measurements are reused.

The original tabular report is pinned at SHA256 `7e3df4dc6a04aa0a88068e9639e3d04d95f86bcae38d3086ad12d23b4f11a7e2`; the first numeric-supplement report is pinned at `881dfb668ea8fca161de42fcba8ca8f0635a0d314507afc542567be11c471e76`. Their referenced prediction/run/test-manifest files must match the recorded hashes. Those report pins are evidence of unchanged source artifacts, not cryptographic attestation of the machines that generated them.

## Source-model protocol differences

Original source predictions retain their original provider/model prompt rendering. Qwen uses the earlier frozen chat-template behavior and numeric-label-plus-EOS likelihood scoring; OpenAI returns generated numeric labels. New SmolLM2 and Granite runs use the pinned revisions and tokenizer files in [expanded_numeric_models.json](../configs/expanded_numeric_models.json): an explicit `You are a helpful assistant.` system message, thinking disabled, original named-feature row text, and the same frozen task prompt/demonstrations. The fixed system message avoids Granite's date-dependent default system text. The new wrapper records rendered prompt/token hashes and rejects context overflow rather than truncating input.

This is not a controlled same-prompt architecture ablation. Original and new model renderers differ, although tabular serialization, demonstrations and held-out identities match. Device, precision, tokenizer/template identity and resolved model revision remain recorded. Mixed local MPS, Colab CUDA and hosted execution make latency figures unsuitable for a controlled speed ranking.

## Jev review

Each review receives the original numerical row, its matching-shot demonstrations and the exact cached source class proposal. No new source rationale is generated. The proposal is advisory; Jev may preserve or change it. Jev uses its native finite-class Choice interface through the pinned OpenRouter route. A valid bounded class output can still be wrong; boundedness is not an accuracy guarantee.

Failed or invalid source proposals are carried forward as failures and trigger no review call. A failed reviewer response counts as an incorrect pipeline result; the source label is not silently restored as a fallback. Raw predictions retain source provenance, prompt hashes, request/reservation identity and response metadata. Reused source predictions avoid new source-model calls in this experiment, but a deployed pipeline incurs both stages' cost and latency.

The reviewer prompt includes the proposed class but not the source model name, probabilities or rationale. Two models that propose the same class for a row therefore produce identical review prompts at the same shot count. Their separate review calls can still differ; small between-pipeline differences must not be attributed to source architecture alone.

Review-versus-Jev-direct comparisons are matched on supplied labels and held-out data. They differ in proposal availability, review prompt wording and separate calls, so effects cannot be assigned solely to the proposal or to bounded decoding. This is not a same-base decoding experiment. Restricted-label likelihood scores, Jev Choice probabilities and native classifier probabilities have different meanings and are not assumed calibrated.

## Scoring and contrasts

[summarize_expanded_numeric.py](../scripts/summarize_expanded_numeric.py) checks the exact historical pins, revalidates model/review/native provenance, verifies ordered train/test identities and recomputes every displayed score from predictions. Missing or incomplete runs remain among all 68 planned rows with null scores. A final status requires all requested conditions; partial results do not establish an aggregate conclusion.

Accuracy and macro-F1 include failures as incorrect. Probability metrics are conditional on valid predictions with complete class distributions, and coverage remains explicit. Review transitions separately count wrong-to-correct, correct-to-wrong-label, correct-to-failure, both-correct, both-wrong, source failures and reviewer failures. Net corrected minus harmed must equal the change in correct test predictions.

The planned report contains 48 main exploratory paired contrasts: each review minus its own source and each review minus direct Jev at the same shot count. Another 24 few-shot-minus-zero-shot contrasts, for base and review pipelines, are explicitly descriptive comparisons with unequal new-label budgets. Matched and full-training native methods appear in separate reference tables; the report does not select a winning native estimator by held-out performance.

Intervals use 2,000 unstratified whole-group percentile bootstrap resamples and seed 42. Groups are identical serialized held-out rows; both models receive the same sampled groups in a paired contrast. There are no exact duplicate test feature vectors in these two frozen datasets, so their groups are single rows. Failures remain in the resamples. Intervals are conditional on one split, training/example selection, prompt/serialization and fitted models, with no adjustment for multiple comparisons. They do not cover cross-seed, prompt, dataset or serving changes. A percentile interval need not bracket its point estimate; the exact interval endpoints are retained.

## Interpretation limits

Both tasks are canonical public datasets with possible pretraining exposure. Wine cultivar IDs have arbitrary class meaning without labeled examples. Zero-shot Wine performance primarily probes pretrained familiarity and inference from the chosen representation, not supervised learnability; perfect observed accuracy on 36 rows does not establish general numerical-classification ability. The study does not identify boundedness as the causal explanation for any result.

These narrow binary/three-class tasks, tiny matched training budgets and fixed serialization do not support general claims about all numeric/tabular data, all LLMs or all review systems. The Breast Cancer dataset is a historical benchmark, not clinical validation. Native full-training references use additional labels. Few-shot review success or harm is reported condition by condition, including failures, without hiding negative results.

A follow-up should compare generated-label and label-token-only scoring with the frozen label-plus-EOS recipe, using training/validation data to select protocols, then evaluate new held-out data. This expansion does not perform that ablation.

## Rebuilding the report

```bash
.venv/bin/python scripts/summarize_expanded_numeric.py --bootstrap-samples 2000
```

This writes `results/numeric_expansion/COMPARISON.json`, `COMPARISON.csv` and `FINDINGS.md`. It performs no inference, model fitting, paid API calls or modification of earlier measurements. Hosted review execution and its cumulative budget use the separate guarded runner and cost auditor; this analysis does not create a new allowance.
