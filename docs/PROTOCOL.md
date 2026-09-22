# Jev classification benchmark protocol

Protocol date: **2026-09-22**. This document distinguishes the implemented pilot from proposed research extensions. It does not assert that the full matrix has run. Only completed run artifacts supply measurements; [SOURCES.md](SOURCES.md) records the primary-source audit.

## Scope and implementation status

Compare Jev, open-weight instruction models, hosted frontier models, and classical classifiers on English closed-set, single-label text classification. Macro-F1 is the primary quality endpoint, with accuracy co-reported. A valid typed answer can still be wrong: schema validity does not prove factual correctness or “zero hallucinations.” Pretraining data and compute differ across model families and cannot be equalized here.

| Available in the implementation | Research extensions, not completed analyses |
|---|---|
| Pinned datasets, stratified preparation, exact-overlap audits, manifests and hashes | Semantic/phrase deduplication, sentence-group splitting for SST-2, contamination detection |
| Classical baselines; Jev/OpenAI/Anthropic/Gemini/local-compatible adapters; local Hugging Face scoring | Every provider/model authenticated and benchmarked; uniform/embedding baselines |
| Zero-shot, shared k/class examples, matched-label classical fitting, response-only LoRA/QLoRA entry point | Equal tuning budgets across all families; calibrated SVM; post-hoc probability calibration |
| Failure-aware classification/probability metrics, individual and paired bootstrap, reports | Calibration plots, selective-risk curves, grouped bootstrap, multi-seed statistical pooling |
| Per-row timing/usage, training metadata, resumable model predictions | Complete billing reconciliation, training amortization, comparable serving benchmarks |

`configs/experiment.json` is a **design inventory**, not an automatically executed configuration. CLI arguments and resolved `run.json` settings define each actual run. The classical matrix script executes explicitly selected baselines. The model matrix script prints a dry-run plan by default; execution requires `--execute` and a matrix-wide request cap, plus `--allow-paid` for hosted models. It runs zero-shot once at seed 42 and few-shot at the requested seeds. Dataset-specific adapters use the single-run CLI, not the matrix driver. A supported adapter, plan, or training entry point is not evidence that the experiment succeeded.

## Data and input track

| Dataset | Task | Training source | Final heldout source |
|---|---|---|---|
| SST-2 | 2-class sentiment | `stanfordnlp/sst2`, train | Public validation: 872 original rows; public test labels are hidden |
| IMDb | 2-class sentiment | `stanfordnlp/imdb`, 25,000 training reviews | 25,000 test reviews |
| AG News | 4-class topic | `fancyzhx/ag_news`, 120,000 training rows | 7,600 test rows |
| TREC-6 | 6-class question type | Pinned script-free representation of CogComp training data | Original 500-question heldout source, represented in pinned files |
| Banking77, optional | 77-class intent | Pinned PolyAI training CSV | Pinned upstream test CSV |

The [dataset sources](SOURCES.md#datasets) support these split semantics. Call SST-2 results **validation-as-test**, not official test results. Use TREC coarse labels only. Exclude IMDb's unlabeled split. Banking77 is a separate extension to the four-dataset core.

Preparation uses **split/subset seed 42**. Remove training examples with exact normalized overlap against the complete original heldout source, then draw a stratified development split from 10% of remaining training rows. Remove remaining training rows overlapping that development split. Every split must contain every class. Development examples never come from the final heldout source, even when that source is named “validation” upstream.

Exact-match normalization uses Unicode **NFKC**, casefolding, and whitespace collapsing, for overlap detection only. Within-split duplicates and conflicting labels are retained and counted. This is not sentence-group splitting: related SST phrases can still cross training/development. Near-duplicates and pretraining contamination remain unmeasured.

The **pilot** explicitly caps preparation at **10,000 training rows, 1,000 development rows, and 200 heldout rows**, or available size if smaller. Selection is deterministic and approximately proportional by class. These are pilot arguments, not implicit limits in every CLI invocation. All compared systems must consume the same saved prepared dataset. Never select evaluation IDs based on predictions or successful completion.

The default input policy is the named **shared 2,000-character prefix track**: retain the first 2,000 Unicode characters of every training, development, and test text for every family. This feasibility constraint changes the task, particularly for IMDb. The manifest records the policy, truncation counts, and transformed-content hash. A second overlap filter after transformation gives heldout data priority. No model-specific silent truncation is allowed. Full-text evaluation is a distinct future track whose results must not be pooled with this track.

Manifests preserve source repository revisions, file checksums, split parameters, exclusions, labels, row IDs, and prepared-file hashes. IDs contain dataset name, source split, row index, and a text/label digest; combine them with the manifest's source revision. Data files are local ignored artifacts. Publication runs should remove pilot row caps, declare the input policy, and rerun every compared system against the same new manifest.

## Downstream label budgets

Use selection/training seeds **13, 42, 87**. Fixed per-class permutations provide nested prefixes of **k = 1, 4, 8 examples per class**. Supervision is **k × C labels**, not k total labels. Every family receives identical selected IDs for a given seed and k; demonstration order is seeded and shared. No pseudo-labels, augmentation, retrieval corpus, or synthetic examples are included.

| Track | Methods | Supervision and selection |
|---|---|---|
| Zero-shot | Jev and LLM adapters | Fixed class names/common instruction; no dataset examples or development-label optimization |
| Matched k/class | In-context learning; classical fitting; separate LoRA fitting | Same k × C training IDs; no development-label selection, calibration, or early stopping |
| All prepared training reference | Classical; optional LoRA | Every prepared training row; disclose preparation cap and actual count |

“All prepared training” is **not necessarily all original training data**: pilot caps and development exclusions still apply. Majority learns from its declared training labels and is a supervised reference, not zero-shot.

Matched-label classical runs fix `C=1` or `alpha=1` in advance. All-prepared-training logistic regression/SVM try `C ∈ {0.1,1,10}`; Naive Bayes tries `alpha ∈ {0.1,1,10}`. Select the first highest-development-macro-F1 candidate using **all prepared development rows**. Majority has one candidate. Vocabulary/IDF are fit on training text only. Selected estimators are not refit on development labels. Reports must disclose training and selection-label counts.

This three-candidate classical reference is not an equal tuning budget comparison against untuned LLMs. Such a study is an extension requiring the same development IDs, candidate cap, and selection rule across families. Calibration needs a disjoint calibration partition and extra disclosed labels. Current LoRA CLI runs use a fixed final checkpoint, without development tuning.

Do not present all-prepared-training models as matched counterparts of 1/4/8-shot models. Matched-budget comparisons require compatible seed and training IDs; unequal-training contrasts are exploratory. Adapter training dataset, labels, seed, and allowed training IDs must be verified. Any training overlap with evaluation IDs invalidates the run. Cross-dataset transfer is a distinct intervention.

Zero-shot has no demonstration-selection variation. Run it once per model/dataset unless measuring service nondeterminism; cached predictions under different seed labels are not independent observations. Per-seed reports are available; mean/standard-deviation synthesis is a separate reporting step.

## Models and fixed recipes

See [the verified inventory](SOURCES.md#model-selection-audit). The direct TypeSafe configuration pins Jev to **`jev-1.13.0`**. The measured OpenRouter route requests **`typesafe/jev-1.13`**, with the served identifier recorded per response and checked against the verified snapshot allowlist. Main practical open model: `Qwen/Qwen3-4B-Instruct-2507`; the smaller Qwen configuration is a technical smoke baseline. Fix open-model/tokenizer revision, precision, adapter, and scoring method. Resolve missing/moving revisions before publication and do not mix resolved versions when resuming.

Candidate hosted panel: `gpt-5.6-luna`, `claude-sonnet-5`, and stable `gemini-3.8-flash`; `gpt-6-astra` is a higher-cost reference. Other audited identifiers are alternatives, not executed cells. The runner requires explicit hosted-call authorization and caps new requests. A request cap is **not a dollar budget**: establish credentials, access, output limits, and an acceptable spending ceiling before paid experiments.

Jev uses one **Choice** over all labels, with clearly separated examples and final item. Official documentation allows 255 options, so Banking77 meets the option-count limit; context must still fit. Jev documents 64k total request tokens, 32k state plus longest question, and no customer LoRA/fine-tuning. These are vendor specifications, not measured capabilities. See [models](https://docs.typesafe.ai/models), [Choice](https://docs.typesafe.ai/primitives/choice), and [failure modes](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

OpenRouter serves this native protocol through `/api/v1/systemone`; the route changes authentication and transport, while the frozen core builds the same shared prompt and Choice criteria. Its catalog advertises 32k context. Record native probabilities, exact returned model, provider, usage and reported cost. The separate wrapper adds durable cost reservations without changing the benchmark core. Hosted latency includes routing/network and reservation overhead and is not a direct TypeSafe latency measurement. See the [official SDK guide](https://openrouter.ai/docs/guides/community/typesafe-sdk) and [public model snapshot](../results/JEV_MODEL_SNAPSHOT.json).

Classical features combine word 1–2-gram TF-IDF (60,000-feature cap) with character-within-word 3–5-gram TF-IDF (80,000 cap), sublinear TF, and default `min_df=1`. Models are logistic regression, linear SVM, multinomial Naive Bayes, and majority. SVM margins are not converted to probabilities. Other native probabilities receive no post-hoc calibration.

The LoRA **CLI** recipe is **3 fixed epochs, rank 8, alpha 16, all-linear targets, dropout 0, learning rate 2e-4, batch size 1, gradient accumulation 8, maximum sequence length 2,048**. Base weights stay frozen; loss covers numeric label plus EOS tokens. A `max_steps` override defines a distinct recorded recipe, primarily for smoke tests. Optional CUDA QLoRA uses 4-bit NF4 with double quantization. Record trainable parameters, resolved revision, optimizer steps, precision, and training metadata. These are practical defaults, not established optimal hyperparameters. See [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), and [PEFT](https://huggingface.co/docs/peft/main/en/developer_guides/quantization).

Evaluate untuned and adapted models with the same scorer and inference precision. Different quantization, targets, or context policies define different conditions. Full prompts exceeding configured context/training length fail rather than silently losing texts or demonstrations. Local/Colab support does not guarantee GPU memory fit, successful large-model training, or a particular runtime.

## Prompt and probability semantics

Backends share a frozen prompt mapping numeric IDs to class names, quoting text as data, listing examples, and marking the final item. Output must be a numeric class ID only. Current prompts use configured class names; more detailed rubrics or prompt search change the protocol. No browsing, tools, chain-of-thought request, or repair model is used.

Label mappings must match requests and metric arrays: negative/positive for SST-2 and IMDb; configured AG News topics and TREC coarse order; pinned Banking77 categories. Native chat templates/API wrappers differ and must be identified. Hosted sampling/reasoning options are those actually sent; omitted options use provider defaults. Asking for one label does not make every hosted run greedy or deterministic.

| Backend | Output protocol |
|---|---|
| Jev | Native Choice and returned full class distribution |
| Hosted generative/local-compatible API | Strictly parsed generated numeric label; class probabilities unavailable |
| Local Hugging Face, before/after LoRA | Teacher-forced log likelihood of each complete numeric label plus EOS, normalized across candidates |
| Classical | Native prediction; native probabilities where supported |

Closed-label likelihood is conditional on candidate strings and termination, not automatically a calibrated semantic posterior. All candidates are scored, including multi-token IDs; this does not measure free-generation compliance. Protocol differences can be compared operationally but must not be attributed solely to model architecture.

For Jev score `probabilities`, **not its separate `confidence`**. Confidence summarizes distribution shape; see [TypeSafe confidence](https://docs.typesafe.ai/confidence). Noul outputs, LLM self-reported confidence, and truncated top-k token lists are not substitutes. Missing distributions stay unavailable, never imputed as one-hot vectors.

Vectors need C finite entries in [0,1], with sum within absolute tolerance 1e-6 of one. The scorer renormalizes accepted vectors and records the count adjusted; malformed distributions are failures, not silently accepted vectors. Preserve auditable metadata locally without publishing dataset texts or credentials.

## Metrics and statistical comparison

Implemented metrics include evaluation count, accuracy, macro-F1 over every true class, balanced accuracy, per-class precision/recall/F1/support, valid-response accuracy, failure count/rate, and a confusion matrix with a failure column. Invalid predictions count as incorrect and as false negatives for their true class, retaining the fixed test denominator. Refusals, malformed outputs, HTTP/network errors, and context failures remain error rows. There are no automatic retries; three consecutive provider errors stop a run for inspection. Resume reuses recorded rows rather than silently replacing failures.

Valid distributions supply coverage, NLL/log loss, **unscaled multiclass Brier** `mean(sum_c((p_ic - 1[y_i=c])^2))`, and top-label ECE with 15 equal-width bins. This Brier convention ranges [0,2] and is twice the positive-class-only binary convention. NLL uses natural logs and **1e-15 clipping followed by renormalization**, recording zero true-class probabilities. ECE uses the maximum probability and correctness of its argmax, weighting each bin's absolute confidence-minus-accuracy gap by frequency. The last bin includes confidence 1. Bin counts/means and disagreements between the returned label and probability argmax are recorded. Binary ROC AUC/average precision require both target classes among probability rows. ECE depends on binning/sample size; proper scores also reflect properties beyond calibration. See [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) and [scikit-learn](https://scikit-learn.org/stable/modules/calibration.html).

Individual pilot intervals default to **1,000 stratified bootstrap resamples**, using the run seed. The paired comparison defaults to **2,000**, seed 42. For publication explicitly request **10,000**. Sample with replacement within each true class, preserving class count; apply identical draws to both models and recompute A minus B accuracy/macro-F1. Use 2.5/97.5 percentiles. Validate identical heldout IDs, labels, and text hashes. A matched-budget interpretation additionally requires compatible seed/training metadata; unequal-training comparisons are explicitly exploratory.

Intervals condition on the chosen models, class composition, and seeds. They do not estimate variation over every training sample or provider update. Current bootstrap treats rows as independent: retained duplicates or correlated sources can make intervals too narrow. Grouped bootstrap is needed before stronger claims in that setting. Separate interval overlap is not a paired significance test.

**Research extensions**, not automatically generated: same-valid-intersection probability contrasts, graphical reliability diagrams, post-hoc calibration, selective risk/coverage, paired multi-seed pooling, McNemar tests, Holm-corrected superiority claims, and aggregate dataset intervals. Multi-seed pooling must resample the same test IDs across all seeds, not treat repeated predictions as independent cases. See [Dror et al.](https://aclanthology.org/P18-1128/). A development-only prompt/order sensitivity study is another extension; see [Lu et al.](https://aclanthology.org/2022.acl-long.556/).

## Timing, cost, and publication

Hosted timing currently measures sequential end-to-end requests. Classical timing is batch feature transformation/prediction divided by row count; resulting p50/p95 are not measured individual-request quantiles. Local likelihood scoring evaluates all candidates rather than generating one label. These describe implemented pipelines and cannot substantiate architecture-only speed ratios. Usage and training duration are metadata, not a complete cost comparison.

A publication latency/cost study must separately measure synchronized local timing, load/warmup behavior, hardware/software, region/network, concurrency, caching, failures/retries, and reasoning-token billing. Reconcile actual usage with a dated price snapshot. Include training/tuning/calibration costs, then amortize as `setup_cost/N + inference_cost_per_item` for N=1,000/10,000/100,000. Free credits mean zero out-of-pocket spending for that run, not zero compute. Vendor marketing ratios are not benchmark measurements.

Artifacts must retain resolved settings, manifest/implementation hashes, seeds, model IDs, training/selection counts, prediction status, and probability semantics. Preserve resume provenance and reject mixed model versions or unexpected prediction IDs. Cached predictions are not fresh timing repetitions. Record deviations in run/report notes.

Publish code, commands/notebooks, environment specifications, sources, ID/hash manifests, label-only predictions, measured tables, and limitations. Raw corpora, full-text request bodies, credentials, and private logs stay untracked. Public access is not redistribution permission: SST-2/AG News/TREC cards mark licenses unknown, IMDb lacks clear redistribution terms, Banking77 advertises CC BY 4.0, and AG News describes research/non-commercial use. Check terms before releasing text or adapters; see [dataset sources](SOURCES.md#datasets).

Historical benchmark contamination is plausible for every foundation model and is not tested by exact overlap filtering. The small, familiar core cannot establish universal superiority or fresh-task generalization. New licensed heldout tasks, full evaluation sets, additional models, and the listed extensions can strengthen a later study. Unrun cells and unsupported measurements must remain explicitly unrun or unavailable.
