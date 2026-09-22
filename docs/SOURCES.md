# Primary-source audit

Accessed **2026-09-22**. These links document model interfaces and methodological choices; they are not benchmark results. Search snippets and unofficial Jev look-alike domains were excluded from the evidence used below. Recheck service availability/prices immediately before spending money and preserve a dated snapshot in the run manifest.

## Jev

| Primary source | Verified fact and consequence |
|---|---|
| [TypeSafe introduction](https://docs.typesafe.ai/introduction) | Jev takes state plus typed questions. Its typed interface does not itself establish correct answers on this benchmark. |
| [Current models](https://docs.typesafe.ai/models) | Explicit ID `jev-1.13.0`; `jev-latest`/`jev-preview` are moving aliases. $0.042 per million input tokens, free output tokens as of access. Context: 64k total request, 32k state plus longest question. Text input only. Published limits are dynamic. Customer fine-tuning/LoRA is unavailable. |
| [Choice primitive](https://docs.typesafe.ai/primitives/choice) | Fixed criteria map, up to 255 options, winning choice plus full probability distribution. This permits 77-way classification by option count. |
| [Confidence](https://docs.typesafe.ai/confidence) | `confidence` summarizes the distribution's shape. It must not be substituted for the predicted class probability in calibration metrics. |
| [API reference](https://docs.typesafe.ai/api) | `POST https://api.typesafe.ai/v1/systemone`; bearer authentication; response carries model, answers and input/output usage. Choice probabilities cover all requested options and sum to one. Retryable throttling/overload and validation errors are documented. |
| [Quick start](https://docs.typesafe.ai/introduction/quickstart) | `typesafe-sdk`, `TypeSafeClient`, `Choice`, and `TYPESAFE_API_KEY` are documented integration names. |
| [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13) | Vendor documents errors involving literal interpretation, numeric precision, indirection, irrelevant context, adversarial content, and non-guaranteed identities between separate primitives. |

The benchmark must independently test the vendor's calibration and reliability claims. No paper or public weights establishing Jev's complete architecture/training recipe was verified during this audit.

## Model-selection audit

These are usable candidate identifiers documented on the access date, not proof of access through a particular account. “Recommended” here means a feasible experimental panel, not a measured quality ranking.

| Role | Verified identifier | Pinning and source |
|---|---|---|
| Jev | `jev-1.13.0` | Explicit version on [TypeSafe models](https://docs.typesafe.ai/models). Log returned ID. |
| Open-weight primary | `Qwen/Qwen3-4B-Instruct-2507` | Official [Qwen card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), Apache 2.0. Resolve repository commit and tokenizer revision before running. |
| Open-weight diversity | `meta-llama/Llama-3.1-8B-Instruct` | Official [Meta card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), Llama 3.1 Community License and access conditions. This is a historical practical baseline. |
| Open-weight diversity | `google/gemma-3-4b-it` | Official [Google card](https://huggingface.co/google/gemma-3-4b-it), Gemma terms and gated agreement. Text-only benchmark usage. |
| Cost-conscious hosted | `gpt-5.6-luna` | [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna): $0.20/M input and $1.20/M output at access; structured outputs; no fine-tuning. Snapshot section lists the dateless ID only. No dated suffix was verified. |
| High-capability hosted | `gpt-6-astra` | [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-6-astra): $10/M input and $50/M output at access; reasoning effort low through max; structured outputs; no fine-tuning. No distinct dated snapshot was verified. |
| Balanced hosted | `claude-sonnet-5` | Official [Anthropic overview](https://platform.claude.com/docs/en/models/overview), $2/M input and $10/M output at access. |
| High-capability hosted | `claude-opus-5` | Same [overview](https://platform.claude.com/docs/en/models/overview), $5/M input and $25/M output at access. |
| Optional high-end hosted | `claude-fable-5-1` | Same [overview](https://platform.claude.com/docs/en/models/overview), $10/M input and $50/M output at access. |
| Smaller hosted alternative | `claude-haiku-4-5-20251001` | Same [overview](https://platform.claude.com/docs/en/models/overview), dated snapshot. |
| Stable Google panel | `gemini-3.8-flash` | Official [Gemini model catalog](https://ai.google.dev/gemini-api/docs/models), stable text-capable model. |
| Optional Google reasoning | `gemini-3.1-pro-preview` | Same [catalog](https://ai.google.dev/gemini-api/docs/models), explicitly preview; must not be described as an immutable stable snapshot. |

[Anthropic versioning](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions) explicitly says newer dateless model IDs refer to pinned snapshots, while serving infrastructure may still change behavior. Google's catalog distinguishes stable, preview, and moving `latest` names; “usually unchanged” is not an absolute infrastructure guarantee. For every provider record request date, requested ID, returned ID, API version, endpoint, reasoning/sampling options, and account-specific capability checks. Do not manufacture date-stamped IDs.

The open-model recommendation prioritizes a manageable QLoRA run and diversity. It does not claim these are the newest or highest-scoring open models available in September 2026. Execution authorization and allocations are recorded in [BUDGET.md](BUDGET.md); this source inventory does not authorize additional spending.

## Datasets

| Dataset | Author/maintainer sources | Split or license observations |
|---|---|---|
| SST-2 | [Stanford dataset card](https://huggingface.co/datasets/stanfordnlp/sst2), [original Stanford resource](https://nlp.stanford.edu/sentiment/) | Card: 67,349 train, 872 validation, 1,821 test; test labels hidden. License marked unknown. Training includes sentiment phrases, requiring overlap awareness. |
| IMDb | [Author's dataset page](https://ai.stanford.edu/~amaas/data/sentiment/), [Stanford card](https://huggingface.co/datasets/stanfordnlp/imdb) | 25,000 train/25,000 test and an unused unlabeled split. Card license tag is “other”; licensing section asks for more information. Do not infer permission to redistribute reviews. |
| AG News | [Maintainer card](https://huggingface.co/datasets/fancyzhx/ag_news), [dataset paper](https://arxiv.org/abs/1509.01626) | Four topics, 120,000 train/7,600 test. License tag is unknown; its corpus description limits the stated uses to research and other non-commercial activities. Underlying news articles may carry separate rights. |
| TREC-6 | [Original CogComp data page](https://cogcomp.seas.upenn.edu/Data/QA/QC/), [CogComp card](https://huggingface.co/datasets/CogComp/trec) | Six coarse labels and 500 test questions; the original training file is authoritative. Card license unknown; old scripted loader is not a reliable modern Datasets integration. |
| Banking77 | [PolyAI repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), [PolyAI card](https://huggingface.co/datasets/PolyAI/banking77) | 77 intents, card marks CC BY 4.0. Pin upstream CSV and category-list revision, preserve attribution. Card's legacy Python loader currently reports that dataset scripts are unsupported. |

Repository publication should distribute download/parsing instructions, pinned source identifiers, hashes, labels/predictions, and aggregate metrics. The code's license does not relicense upstream datasets. Raw corpora and request bodies containing those texts stay untracked by default. This audit records what the retrieved pages say; it does not resolve unspecified licenses or confer new rights.

### Tabular extension

The [tabular protocol](TABULAR_PROTOCOL.md) records exact source byte hashes, feature units, exclusions, serialization and split construction. It uses Vanderbilt's 1,309-row Titanic3 dataset and the scikit-learn bundled numerical datasets. The prepared feature projections, targets and hashes are included in the separate tabular Colab release bundle with attribution; the original Titanic personal-description columns are excluded.

| Dataset | Primary sources | Attribution and redistribution |
|---|---|---|
| Titanic3 | [Vanderbilt Biostatistics data collection](https://hbiostat.org/data/), [Titanic3 dictionary](https://hbiostat.org/data/repo/ctitanic3) | Data obtained from https://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics; use permitted with attribution. |
| Breast Cancer Wisconsin Diagnostic | [UCI dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [DOI 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) | Wolberg, Mangasarian, Street and Street (1993), CC BY 4.0. Feature names are serialized without changing the measured values; records are partitioned into benchmark splits. |
| Wine | [UCI dataset](https://archive.ics.uci.edu/dataset/109/wine), [DOI 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J) | Aeberhard and Forina (1992), CC BY 4.0. Feature names are serialized without changing the measured values; records are partitioned into benchmark splits. |

The native baselines use scikit-learn's [mixed-column preprocessing](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data) and [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html). All fitted preprocessing is trained on the declared training rows. The report's whole-group bootstrap is an explicitly implemented project analysis, with regression tests and its assumptions stated in the protocol.

## Methods

| Primary source | Use in this project |
|---|---|
| [Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) | Calibration, reliability diagrams, ECE, and heldout temperature scaling. The protocol fixes its own binning and clipping conventions. |
| [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | Distinguishes probability calibration from discrimination and explains calibration-data separation. Proper scores combine calibration and other predictive properties; lower Brier alone is not pure evidence of better calibration. |
| [Zhao et al., Calibrate Before Use, ICML 2021](https://proceedings.mlr.press/v139/zhao21c.html) | Prompt/label bias and contextual calibration motivate reporting raw and adjusted probabilities separately. Contextual calibration is not silently applied in the primary track. |
| [Lu et al., Fantastically Ordered Prompts, ACL 2022](https://aclanthology.org/2022.acl-long.556/) | Motivates shared seeded example order and a separately budgeted sensitivity analysis. |
| [Hu et al., LoRA](https://arxiv.org/abs/2106.09685) | Frozen base weights plus trainable low-rank adaptation. |
| [Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314) | Quantized base-model adaptation and its compute/memory considerations. |
| [Hugging Face PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization) | Practical quantized adapter training; record the installed library version instead of treating mutable `main` documentation as an environment lock. |
| [Dror et al., The Hitchhiker's Guide to Testing Statistical Significance in NLP, ACL 2018](https://aclanthology.org/P18-1128/) | Paired evaluation and appropriate significance-test selection. The exact stratified bootstrap and multiple-comparison choices are documented project decisions. Reported contrasts are exploratory and were not prospectively registered. |

No source above supplies the benchmark's eventual results. Unrun model/dataset cells must remain explicitly unrun.

## Jev access through OpenRouter

Verified on 22 September 2026. [OpenRouter's official TypeSafe integration](https://openrouter.ai/docs/guides/community/typesafe-sdk) documents the stable `/api/v1/systemone` endpoint and native TypeSafe request/response shapes, including OpenRouter's returned provider, request ID and `usage.cost`. The [live endpoint catalog](https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints) lists `typesafe/jev-1.13`, TypeSafe as provider, a 32,000-token context and $0.042/M input / free output pricing. The response's served snapshot is recorded per prediction; an alias or provider label alone is not an immutable identity guarantee. A public [metadata snapshot](../results/JEV_MODEL_SNAPSHOT.json) preserves the retrieved catalog and retrieval hash.

OpenRouter credentials authenticate to OpenRouter only. Route-level latency and reported charges must be described as OpenRouter-routed Jev measurements, separately from direct TypeSafe serving.
