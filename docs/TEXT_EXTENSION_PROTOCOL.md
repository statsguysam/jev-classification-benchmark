# Text extension: the same bounded-decision pipeline

This extension asks whether Jev changes the accuracy of a language model's proposed class on **text classification**, using the same source → Jev review design as the [numerical study](EXPANDED_NUMERIC_PROTOCOL.md). Numerical tabular classification remains the primary study. The original text/LoRA pilot and its artifacts remain unchanged; the new extension adds no adapter training.

**All 68 conditions and 72 contrasts are complete and audited.** These comprise twenty-four source runs, twenty-four Jev reviews, four historical direct-Jev runs and sixteen classical conditions. The eight new SmolLM2/Granite conditions produced 1,600 predictions on a Colab T4. The Jev reviews made 4,800 requests; their 48 errors remain in the original scores. See the [comparison](../results/text_extension/COMPARISON.json) and [findings](../results/text_extension/FINDINGS.md). Later [failure recovery](../results/completion_20260923/RECOVERY_FINDINGS.md) is reported separately. This status update does not change the experimental design below.

## Fixed data and label budgets

| Dataset | Task | Classes | Frozen test rows | Four examples per class | Full prepared training |
|---|---|---:|---:|---:|---:|
| SST-2 | Sentiment | 2 | 200 | 8 labels | 10,000 labels |
| TREC | Question type | 6 | 200 | 24 labels | 4,886 labels |

Every condition within a dataset uses exactly the original pilot's ordered 200 test IDs, label mapping and prepared text bytes. SST-2 uses its labeled validation source as final test because official test labels are hidden. Both datasets retain the pilot's first-2,000-character policy; this is a capped pilot rather than a full official-test evaluation. Exact prepared file/content hashes and original source artifacts are pinned in [text_extension_sources.json](../configs/text_extension_sources.json). Sources, revisions, attribution and license notes remain in the original manifests and [dataset configuration](../configs/datasets.json).

Four-per-class conditions use the unchanged training-only selector and seed 42. Source and reviewer receive **the same** demonstration IDs, so a two-stage few-shot pipeline uses eight or twenty-four unique training labels, not twice that number. Zero-shot uses no labeled demonstrations. The extension uses no validation labels for fitting or selection and performs no prompt, model or hyperparameter search against test outcomes.

Historical Qwen3 Colab manifests contain additional metadata and therefore have different manifest hashes from the local preparation. Those hashes are preserved. Independent checks of the full prepared content, ordered test rows and selected training IDs establish eligibility for separately labeled matched comparisons; the audit does not assert that unequal manifests are identical.

## Sixty-eight conditions

The source models are Qwen2.5 0.5B, SmolLM2 1.7B, Granite 3.3 2B, Qwen3 4B, GPT-5.6 Luna and GPT-6 Astra. Each has zero-shot and four-per-class conditions on both datasets: **24 source conditions**. Each source condition has a corresponding Jev review: **24 pipeline conditions**. Four direct-Jev references and sixteen classical references complete the matrix.

Existing Qwen/Luna/Astra predictions and direct-Jev predictions are reused after artifact and content audits. Their original model settings, failures and provenance remain intact. SmolLM2 and Granite add eight source conditions (1,600 predictions) through the [new local wrapper](../scripts/run_text_extension_local.py), which reuses the unchanged numerical extension's renderer, preflight and scorer:

| New source | Immutable revision |
|---|---|
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` |
| `ibm-granite/granite-3.3-2b-instruct` | `707f574c62054322f6b5b04b6d075f0a8f05e0f0` |

Both use float16, an explicitly selected CUDA/MPS device, the system message `You are a helpful assistant.`, and thinking disabled. All prompts plus candidate labels are checked against 8,192 tokens with no truncation or device/precision fallback. Model, tokenizer, template, rendered-prompt and wrapper identities are recorded. The source bundle excludes credentials, prior results and model weights; public snapshot downloads explicitly disable implicit Hugging Face authentication.

Local classification sums the log likelihood of each numeric class ID **plus EOS**, then normalizes over these complete candidate sequences. This yields bounded class predictions but is a specific likelihood-scoring protocol, not free-form chat generation. Historical hosted sources use their preserved generated-label protocol. Model family, rendering, output protocol and runtime can all affect comparisons; these are not isolated architecture effects.

## What Jev sees

The [review wrapper](../scripts/run_text_jev_review.py) supplies the original prepared text, allowed classes, the same zero/four-per-class examples, and the source's **cached proposed class ID only**. It tells Jev that the proposal may be wrong and permits keeping or overriding it. It supplies no source rationale, confidence vector, hidden test label or new source-model generation. Jev uses the native Choice route through OpenRouter with the frozen route/model settings.

This tests a second classification decision given a proposal; it does not test attaching a formatting constraint to another model's internal logits. Identical original text, examples and proposed classes produce identical review prompts regardless of which source model produced the class. Differences between repeated reviews of identical prompts cannot be attributed to source-model identity.

A failed source prediction remains a failed pipeline row and triggers no Jev request. Reviewer errors also remain in the full test denominator. Accuracy, macro-F1, source failures, reviewer failures, decisions corrected and decisions harmed are reported separately. A valid allowed class is not evidence that it is the correct class. Provider scores and restricted likelihood probabilities are not assumed calibrated or interchangeable.

## Classical references

Native XGBoost, LightGBM, logistic regression and random forest each run at matched four-per-class and full-prepared-training budgets: **16 conditions**. All use the same sparse word `(1,2)` and `char_wb (3,5)` TF-IDF representation. Vocabulary and IDF are fit only on the selected training rows; validation/test text does not fit preprocessing. There is no dense conversion, feature selection, validation-label fitting, tuning or early stopping. Fixed configurations and package versions were saved before fitting in [text_classical_extension.json](../configs/text_classical_extension.json).

Within each dataset and training budget, all four estimators receive matrices with identical feature hashes. **XGBoost treats unstored sparse entries as missing; the other estimators interpret sparse absence as zero.** This native sparse behavior remains part of the comparison. The modest fixed tree recipes are not an optimized text-classification leaderboard. Full-training references use substantially more labels and must remain distinct from matched training and zero-shot.

See the [classical results and audit notes](../results/text_extension/classical/README.md). No conclusions about Jev review follow from those completed baselines alone.

## Metrics, contrasts and interpretation

The aggregate recomputes metrics from aligned raw predictions, retaining failures as incorrect. It reports 95% percentile bootstrap intervals for accuracy and macro-F1, using 2,000 unstratified resamples of normalized-text groups and seed 42. Both frozen test sets contain 200 distinct groups. Paired comparisons resample the same groups jointly across both methods.

The **72 planned contrasts** comprise 24 pipeline-minus-source, 24 pipeline-minus-direct-Jev, and 24 four-per-class-minus-zero-shot comparisons (12 source and 12 pipeline). The latter change the labeled-data budget and are descriptive contrasts. Intervals are conditional on this split, example selection and fixed fitted models; they are unadjusted for multiple comparisons. They do not establish broad significance or a universal ranking.

This is an exploratory extension designed after earlier test results were viewed. Familiar public datasets may have appeared in pretraining; there is one split and one demonstration seed. Small test sets, protocol differences, uncalibrated scores, unequal full-training label budgets, cached historical runs and provider nondeterminism constrain interpretation. No pooled text-plus-numerical leaderboard, controlled latency claim, calibration claim or novel-discovery claim is justified by this design alone.

Hosted execution uses a separate **US$1.60** text-review ledger while protecting the full numerical allowance inside the existing US$25 cumulative ceiling. It requires funded account credit and a current verified transport declaration. Successful new calls may settle conservatively; unknown/failed calls retain their reservation. No old ledger is repriced. See [budget controls](BUDGET.md) and [reproduction](TEXT_EXTENSION_REPRODUCTION.md).
