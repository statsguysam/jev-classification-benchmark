# Model access and training

The adapters implement `build_provider(config).predict(row, labels, prompt)`.
Every adapter receives the same complete task prompt, examples, and final text.
Prediction code does not read `row.label`. Optional neural libraries are imported
only when a local model is loaded. Constructing a provider makes no API request.

## Credentials and explicit model selection

Set credentials in your shell or Colab secrets; never put key values in JSON,
notebooks, Git, or result files. Provider configs contain an environment-variable
**name**, not a key value. Example configs:

```json
{"provider":"jev","model":"jev-1.13.0","api_key_env":"TYPESAFE_API_KEY","timeout":120}
```

```json
{"provider":"openai","model":"YOUR_AVAILABLE_PINNED_MODEL","api_key_env":"OPENAI_API_KEY","max_output_tokens":4096,"timeout":120}
```

```json
{"provider":"anthropic","model":"YOUR_AVAILABLE_PINNED_MODEL","api_key_env":"ANTHROPIC_API_KEY","max_output_tokens":256,"timeout":120}
```

```json
{"provider":"gemini","model":"YOUR_AVAILABLE_PINNED_MODEL","api_key_env":"GEMINI_API_KEY","max_output_tokens":4096,"timeout":120}
```

Use current account-available model IDs and immutable versions where offered.
An alias can change during a study: record both requested and returned model IDs.
The code records the server's returned model/snapshot fields; their absence means
the server did not identify an immutable snapshot. A provider's name alone is not
evidence of model identity or reproducibility.

`max_output_tokens` is required to budget generation meaningfully. For reasoning
models this budget can include internal reasoning; too small a limit may prevent
any answer. Such truncations count as failed predictions. Do not change the budget
after inspecting test outcomes. Sampling fields are only sent when configured,
because some model families reject `temperature`, `seed`, or `reasoning_effort`.
OpenAI supports these three config fields. Anthropic supports `temperature`.
Gemini supports `temperature` and provider-native `thinking_config`.

HTTP requests have a timeout and no automatic retries. A timeout may still be
billed by the server, and retrying could incur another charge. The runner should
record failures and enforce its paid-call authorization before invoking adapters.
Returned token usage is preserved even for invalid labels or truncated answers;
network failures with no usage response retain unknown usage, never assumed zero.
HTTP errors expose status only, not response bodies or secrets. Redirects are
refused. Remote endpoints must use HTTPS; HTTP is allowed only for localhost.

## Jev

Jev receives the shared prompt as `state`, with a `Choice` question mapping the
numeric class IDs to the same class descriptions. The adapter uses
`POST /v1/systemone`, retains the complete returned class distribution, verifies
it sums to one, and rejects a selected choice inconsistent with its maximum.
The separate vendor confidence score is not substituted for class probabilities.
Binary tasks also use Choice, so binary and multiclass experiments use the same
interface. This introduces an unavoidable interface difference from text models;
it must be disclosed in the comparison.

Sources: [TypeSafe quickstart](https://docs.typesafe.ai/introduction/quickstart),
[Choice request and response](https://docs.typesafe.ai/primitives/choice).

Jev 1.13 documents a 32k-token limit for state plus longest question, and 64k for
the combined request. Its tokenizer is not available in this harness, so no exact
local token preflight is claimed; oversized request errors count as failures.
Keep shared few-shot prompts within the smallest model's context window.
[TypeSafe model limits](https://docs.typesafe.ai/models).

## Frontier APIs and local servers

OpenAI, Anthropic, and Gemini adapters request a numeric ID and parse the entire
answer strictly. Extra explanation, refusal, malformed output, and truncated
generation are failures; the adapter never extracts a convenient digit from an
invalid response. Hosted generative models return `probabilities=None`: generated
confidence claims and partial top-token log probabilities are not complete class
distributions. Probability-dependent metrics must therefore be unavailable for
these runs. An accuracy comparison remains possible.

For a local OpenAI-compatible server, including vLLM:

```json
{"provider":"openai_compatible","model":"YOUR_SERVED_MODEL","base_url":"http://localhost:8000/v1","api_key_env":null,"token_limit_parameter":"max_tokens","temperature":0,"max_output_tokens":32}
```

This server path uses generated labels. To compare with conditional likelihood
scoring, use the Hugging Face adapter instead and disclose the decoding difference.
Only `api_key_env: null` disables authentication. For authenticated local servers,
provide the appropriate environment-variable name.

Sources: [OpenAI Chat API](https://developers.openai.com/api/reference/resources/chat),
[Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create),
[Gemini generateContent](https://ai.google.dev/api/generate-content).

## Local open-weight models

Install with `pip install -e '.[neural]'`. Example:

```json
{"provider":"huggingface","model":"Qwen/Qwen2.5-0.5B-Instruct","revision":"REPLACE_WITH_COMMIT_SHA","device":"auto","max_context_tokens":4096}
```

The 0.5B model is a technical smoke baseline, not a representative current frontier
open model. Use stronger models in the main study, subject to available memory.
`device: auto` chooses CUDA, MPS, then CPU. Default dtype is float32 on CPU and
float16 elsewhere; `dtype` can explicitly select float32, float16, or bfloat16.
Remote model code is disabled. The resolved Hugging Face commit hash, device,
dtype, and probability semantics are recorded in each prediction.

The local adapter scores each complete numeric class ID followed by the
tokenizer's EOS token. It sums conditional token log likelihoods and applies
softmax **only over the allowed label sequences**. This is a restricted
distribution, not a guarantee of calibrated class probabilities or equivalence
to Jev's distribution. EOS makes label strings such as `1` and `10` disjoint.
The model's native chat template formats the shared prompt; raw models without
a chat template use their normal tokenizer. Labels and EOS use the same encoding
for fine-tuning and evaluation. There is no length normalization.

The simple scorer performs one full forward pass per class, without KV-cache
reuse or batching. Its latency is an implementation baseline and should not be
interpreted as optimized inference throughput. Metadata records input tokens,
candidate tokens scored, and forward-pass count; output tokens are zero because
no free-form text is generated. Weight-loading time is excluded from per-row
latency. Full prompts exceeding the configured or model context window fail;
the backend does not silently truncate them.

Source: [Qwen 0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

## LoRA and Colab

`train_lora(dataset, model_id, output_dir, ...)` applies PEFT LoRA to a downloadable
causal model. It uses train rows only, with loss masked over the entire prompt and
padding. Targets are the numeric ID plus EOS. Fixed upfront epochs and optional
optimizer-step limit select a single final checkpoint. There is no early stopping
or validation-based checkpoint selection. Validation loss can be requested as a
final diagnostic in full-training runs; matched-label-budget training forbids it.

`train_per_class=k` uses exactly the same seeded row selector as few-shot prompting,
so the matched experiment has identical labeled training examples. Full-training
LoRA is a separate condition. The training function never accesses the test split.

Defaults are rank 8, alpha 16, all linear layers, zero LoRA dropout, AdamW learning
rate 0.0002, batch size 1, gradient accumulation 8, and one epoch in the Python function.
The CLI/notebook explicitly default to **three epochs** and sequence length 2,048. These are starter
settings, not claimed optima. Fix hyperparameters before testing; use training and
validation data only for any separate tuning study. Runtime depends substantially
on base-model size and prompt length, even when the dataset is small.

Training defaults to bfloat16 on supporting CUDA GPUs, float16 with CUDA AMP and
gradient scaling on other CUDA GPUs, and float32 on CPU/MPS. An explicit `dtype`
overrides this choice. AMP configuration and any skipped optimizer updates are
recorded in metadata; inspect skipped updates when assessing numerical stability.

Colab GPU and local devices use the same entry point. For CUDA QLoRA install
`pip install -e '.[neural,qlora]'` and set `load_in_4bit=True`; NF4 quantization and
double quantization are used. Four-bit training is rejected on CPU/MPS. Free Colab
GPU type, availability, and session duration are not guaranteed. Check memory with
a smoke run before choosing the full matrix. Ordinary LoRA on the 0.5B baseline
can be used to verify the pipeline locally.

The output directory must be empty. Saved adapters include
`jevbench_training.json` with base revision, labels, exact training row IDs,
prompt hash, optimizer settings, seed, package versions, device, training duration,
and validation usage. Keep this metadata with every adapter. Evaluation example:

```json
{"provider":"huggingface","model":"Qwen/Qwen2.5-0.5B-Instruct","adapter_path":"artifacts/lora/run-name","device":"auto","max_context_tokens":4096}
```

The loader reuses the trained base revision and rejects a conflicting revision,
base model, or label order. Fixed seeds improve repeatability but do not establish
bitwise reproducibility across GPU models, drivers, and backend implementations.
LoRA is not supported for Jev or proprietary hosted model weights.

Sources: [PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora),
[PEFT quantization](https://huggingface.co/docs/peft/main/en/developer_guides/quantization).
