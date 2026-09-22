# Independent audit of the additional open models

All eight imported SmolLM2 1.7B and Granite 3.3 2B conditions passed the independent read-only source audit: **1,600 ordered test predictions, zero inference failures**. Dataset contents, class order, held-out IDs, selected training examples, model revisions, renderer fingerprints, probability vectors, class decisions and recorded metrics matched their frozen evidence.

The imported result ZIP has SHA-256 `01e80b38f8ddc8498c5d2ccf3a1c404ef1473330a901a3944aa1c7c54fdcae61`.

| Dataset | Model | Zero-shot accuracy | Few-shot accuracy | Zero-shot macro-F1 | Few-shot macro-F1 |
|---|---|---:|---:|---:|---:|
| SST-2 | SmolLM2 1.7B | 49.0% | 70.0% | 0.3289 | 0.6703 |
| SST-2 | Granite 3.3 2B | 80.0% | 94.0% | 0.7940 | 0.9399 |
| TREC | SmolLM2 1.7B | 2.0% | 27.5% | 0.0065 | 0.0719 |
| TREC | Granite 3.3 2B | 46.0% | 63.5% | 0.4058 | 0.5977 |

Few-shot means four examples per class: eight labels for SST-2 and 24 for TREC. These are source-model results; **they do not measure an LLM → Jev review effect**.

## Predicted class counts

SST-2 class IDs are **0: negative**, **1: positive**. Its held-out class counts are 98 negative and 102 positive.

| Model | Examples/class | Negative (0) | Positive (1) |
|---|---:|---:|---:|
| SmolLM2 1.7B | 0 | 200 | 0 |
| SmolLM2 1.7B | 4 | 42 | 158 |
| Granite 3.3 2B | 0 | 136 | 64 |
| Granite 3.3 2B | 4 | 108 | 92 |

TREC class IDs are **0: abbreviation**, **1: entity**, **2: description or abstract concept**, **3: human being**, **4: location**, **5: numeric value**. Its held-out class counts in that order are **4, 38, 55, 26, 32, 45**.

| Model | Examples/class | Abbreviation (0) | Entity (1) | Description (2) | Human (3) | Location (4) | Numeric (5) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2 1.7B | 0 | 200 | 0 | 0 | 0 | 0 | 0 |
| SmolLM2 1.7B | 4 | 0 | 0 | 200 | 0 | 0 | 0 |
| Granite 3.3 2B | 0 | 5 | 0 | 124 | 37 | 34 | 0 |
| Granite 3.3 2B | 4 | 7 | 2 | 73 | 47 | 33 | 38 |

**Three of the eight conditions predict a single class for every test row.** SmolLM2's TREC increase from 2.0% to 27.5% is entirely a switch from always predicting abbreviation to always predicting description or abstract concept. It does not establish successful discrimination between individual questions. SmolLM2 also predicts negative for every zero-shot SST-2 row. These observations concern this fixed inference recipe; they do not establish that the model is generally incapable of the tasks.

Granite produces different labels across rows in all four conditions. No complete source prediction vector is identical to another model's vector within the same dataset and shot condition across the six source models in this extension.

## Protocol and limits

Both additional models ran on CUDA in float16 with pinned revisions, tokenizer assets and chat templates. The system message was `You are a helpful assistant.` Classification scored each permitted **numeric class ID followed by EOS**, summed its token log-likelihood and normalized the candidate scores; the selected class matched the maximum probability in every saved row. This is restricted-label scoring, not free-form generation, and the resulting probabilities are not demonstrated to be calibrated.

Recorded prompts ranged from 80 to 796 tokens; all eight conditions passed the full-prompt preflight without truncation. The source audit also verified the expected ordered few-shot examples against the frozen training data.

Each dataset has one frozen 200-row test set, one preparation/selection seed (42), and one example selection. Familiar public datasets may have appeared in pretraining. Four-example and zero-example conditions use different amounts of newly supplied supervision. Original Qwen/OpenAI runs retain their historical rendering, so comparisons with these new models do not isolate architecture alone. Investigating the constant outputs would require a separately labeled scoring or generation ablation; no such ablation is claimed here.

Evidence: [source registry](../../configs/text_extension_sources.json), [source validator](../../scripts/text_extension_sources.py), [local inference wrapper](../../scripts/run_text_extension_local.py), and the [comparison report](COMPARISON.json). Imported raw run, prediction and test-manifest artifacts are under [local](local/).
