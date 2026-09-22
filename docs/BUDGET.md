# Hosted API budget and resumption

The approved cumulative total is now **US$25**: the original US$10 ceiling, US$10 for the tabular extension, and US$5 for the expanded numerical zero/few-shot review experiment. These ledgers enforce reservations under the documented pricing assumptions; they are not provider invoices or account-wide caps.

## Expanded numerical review stage

All four earlier ledgers and their lock files remain frozen. Their cumulative conservative amount is **US$19.325827700**, including the first four numerical review runs. The new `results/numeric_expansion/review-budget.jsonl` has a **US$5 allocation** and a **1,500-call limit**. Each Jev call reserves US$0.002688, so the complete expansion can reserve at most **US$4.032000**, bringing cumulative conservative accounting to at most **US$23.357827700**. Reported API charges are recorded separately; reservations are never released in this stage.

Six source models are compared at zero and four examples per class on Breast Cancer and Wine. Existing source predictions and the four original Qwen3 4B/Astra few-shot reviews are reused. SmolLM2 and Granite inference runs locally or on Colab with public weights. No new paid source-LLM calls are required.

### Current halt and guarded recovery

The audited snapshot is **61/68 conditions complete**: 24 source runs, 17 reviews, four direct Jev references and 16 native references. Seven reviews remain unfinished, and 54/72 comparisons are complete. See [the current status](../results/numeric_expansion/STATUS.md) for the exact remaining conditions. The ledger contains **1,099 new Jev requests**, including seven calls with unknown reported cost. Known reported charges total **US$0.084934668**; these are not the complete expense. Retained reservations total **US$2.954112000** for this stage and **US$22.279939700 of US$25** cumulatively.

The Breast Cancer SmolLM2 four-per-class checkpoint has **85/114 saved rows**, including **six HTTP 402 failures**. The current halt follows three consecutive billing failures. Those failures stay in the evidence and count as incorrect when the full condition completes. Partial predictions are not scored as a completed test. **401 next-unseen rows remain across seven conditions.**

Recovery requires funded provider credits as well as the existing benchmark budget headroom. The ordinary runner alone cannot clear the current billing halt. Do not delete or reset the ledger, lock, failed predictions, source manifests or halted checkpoint, and do not retry already recorded failed rows. Stop any previous worker before recovery; the wrapper also acquires the same execution lock.

The [billing-recovery wrapper](../scripts/recover_expanded_numeric_billing.py) defaults to a read-only audit and does not read credentials in this mode:

```bash
python scripts/recover_expanded_numeric_billing.py \
  --dataset breast_cancer --model-key smollm2 --shots 4 --dry-run
```

After funding is available, the following command checks authenticated available account credit, preserves the halted checkpoint and recovery evidence, and attempts **one next-unseen row** through the frozen inference producer:

```bash
python scripts/recover_expanded_numeric_billing.py \
  --dataset breast_cancer --model-key smollm2 --shots 4 \
  --execute --prompt-api-key --stop-after-new-requests 1
```

The credit check must be fresh and positive; a key's configured quota is not proof of funded account credit. A positive balance does not guarantee that every remaining request will be funded. The wrapper records an immutable halted-run snapshot, the credit-check receipt and before/after evidence hashes under `results/numeric_expansion/billing_recovery/`. It permits one reservation past the exact acknowledged HTTP-402 tail, for the next unseen row only. Original ledger events and all 85 saved predictions remain unchanged. No failed row is retried. If that first resumed request fails, the producer stops again immediately.

**The dated transport guard still applies.** The frozen route/pricing declaration is verified for **22 September 2026 UTC**, regardless of the local calendar date. After that UTC date, recovery refuses to authenticate or mutate the checkpoint until the route and prices have been explicitly reverified and a compatible recovery path reviewed. Do not merely change the date or edit frozen producers to bypass this check. Funding alone does not clear a stale-date guard.

Only after the guarded next-row request succeeds and its checkpoint/ledger reconcile can the normal runner continue the remaining SmolLM2/Granite conditions, skipping completed arms and saved rows:

```bash
python scripts/run_expanded_numeric_review.py \
  --datasets breast_cancer wine --model-keys smollm2 granite --shots 0 4 \
  --execute --prompt-api-key
```

The expansion ledger is already initialized; never add `--init-ledger` to recovery or resumption. All 24 source conditions are complete. Source manifests freeze each individual model/dataset/shot artifact, and changing model, prompt, source code or provenance is not a resume. Unknown costs and interrupted calls retain their reservation; no automatic retries occur. Reconcile with `python scripts/audit_expanded_numeric_costs.py` after execution, then regenerate the numerical summary before updating completion claims. The remaining 401 planned calls would retain another US$1.077888, bringing cumulative conservative accounting to US$23.357827700 if all are executed under the unchanged per-request reservation.

## Historical tabular allocation

Before this expansion the approved ceiling was US$20. The completed text study conservatively used **US$5.293007**. Its two ledgers were frozen before the tabular stage received one shared **US$14.70** ledger, making the prior retained amount plus that allocation **US$19.993007**.

## Completed tabular stage

All three hosted providers share `results/tabular/api-budget.jsonl` and its `.lock` file. The driver checks SHA-256 identities of both old ledgers and their lock files before every reservation. Do not resume the old text workers while running this stage: changing an old ledger makes the new driver stop. Do not delete, reset, duplicate or independently replace any budget ledger.

The fixed matrix covers Jev, Luna and Astra at zero and four examples per class on all prepared test rows: Titanic 262, Breast Cancer 114, Wine 36. This is **18 runs and 2,472 requests**. It retains the earlier model settings and prices. Successful OpenAI usage can settle reservations conservatively; Jev, failed requests and unknown usage retain their full reservations. All providers share the same POSIX-locked reservation ceiling.

Dry-run planning sends no paid requests:

```bash
python scripts/run_tabular_hosted.py \
  --data data/tabular-full/titanic data/tabular-full/breast_cancer data/tabular-full/wine
```

The ledger has already been initialized. To resume **after all previous tabular workers have stopped**, keep every recorded file and run:

```bash
python scripts/run_tabular_hosted.py \
  --data data/tabular-full/titanic data/tabular-full/breast_cancer data/tabular-full/wine \
  --workers 3 --execute --prompt-api-key
```

This launches one isolated process per dataset. It asks for `OPENAI_API_KEY` and `OPENROUTER_API_KEY` through hidden terminal prompts, then keeps them in memory. Each process uses the same ledger; changing dataset, prompt, driver or model provenance is not a resume. Errors are recorded, not retried automatically. Do not add `--init-ledger` to a resume command. A fresh reservation can be larger than its later settlement, so execution may stop before the ceiling is fully spent.

Run `python scripts/summarize_tabular_costs.py` to reconcile the tabular ledger and retain the distinction between reported usage, estimates, unknown costs and conservative reservations. See the [tabular protocol](TABULAR_PROTOCOL.md) for dataset and label-budget details.

## Historical text-stage allocation

The original approved total was **US$10**, allocated **US$7.50 to OpenAI** and **US$2.50 to Jev**. OpenAI execution reconciles to US$3.142607 in conservative settlements and US$2.568174 in reported-token standard-rate estimates. Jev retained US$2.1504 in reservations. See [OpenAI accounting](../results/API_COSTS.md) and [Jev/combined accounting](../results/JEV_COSTS.md) for measured totals and cost coverage. The commands below document that completed stage; its unused allocations are not additional current spending authority.

| Allocation | Ledger | Scope as of 22 September 2026 |
|---|---|---|
| OpenAI: US$7.50 | `results/openai-budget.jsonl` and `results/openai-budget.jsonl.lock` | SST-2 and TREC, Luna and Astra, zero-shot and four-shot-per-class, seed 42, 200 test rows per run; eight completed runs, 1,600 requests |
| Jev: US$2.50 | `results/jev-budget.jsonl` and `results/jev-budget.jsonl.lock` | OpenRouter native Choice on SST-2/TREC, zero/four-shot-per-class, seed 42, 200 rows per run; four-condition, 800-request matrix |

The OpenAI design contains eight runs and at most 1,600 new test requests. It is not a promise that every planned run will finish within its allocation. The original [6,400-request inventory](../results/HOSTED_PLAN.json) covers a broader design and is not the currently funded execution plan. No completion or accuracy result is inferred from a plan or ledger.

## Guarded execution

Use [run_budgeted_hosted.py](../scripts/run_budgeted_hosted.py) with [hosted_budget.json](../configs/hosted_budget.json). Luna uses `gpt-5.6-luna`, reasoning effort `none`, and a 256-token output ceiling. Astra uses `gpt-6-astra`, effort `low`, and a 1,024-token output ceiling. The output cap includes reasoning and visible completion tokens. These settings are fixed for zero-shot and few-shot and must not be changed in response to test scores.

From an activated project environment, this command **only plans** the OpenAI matrix. It sends no requests and does not initialize or mutate a ledger:

```bash
python scripts/run_budgeted_hosted.py --config configs/hosted_budget.json \
  --model-keys openai_economical openai_frontier \
  --data data/pilot/sst2 data/pilot/trec --shots 0 4 --seed 42 \
  --output results/hosted --max-requests 200 \
  --budget-usd 7.50 --ledger results/openai-budget.jsonl
```

The current execution uses two workers, one for each dataset, sharing the same OpenAI ledger. For a **first execution with no existing ledger**, the SST-2 worker is initialized once as follows. The current study has already performed this initialization; do not repeat it to resume:

```bash
python scripts/run_budgeted_hosted.py --config configs/hosted_budget.json \
  --model-keys openai_economical openai_frontier \
  --data data/pilot/sst2 --shots 0 4 --seed 42 \
  --output results/hosted --max-requests 200 \
  --budget-usd 7.50 --ledger results/openai-budget.jsonl \
  --execute --init-ledger --prompt-api-key
```

The TREC worker uses that same ledger and omits `--init-ledger`:

```bash
python scripts/run_budgeted_hosted.py --config configs/hosted_budget.json \
  --model-keys openai_economical openai_frontier \
  --data data/pilot/trec --shots 0 4 --seed 42 \
  --output results/hosted --max-requests 200 \
  --budget-usd 7.50 --ledger results/openai-budget.jsonl \
  --execute --prompt-api-key
```

`--prompt-api-key` reads the selected credential through a hidden controlling-terminal prompt. It is held in the process environment, not included in command arguments, configuration, results or notebook cells. It requires a terminal; do not pipe credentials into the command. Already configured environment credentials can be used by omitting that flag.

To **resume after the original workers have stopped**, retain both ledger files and the original result directories, then use the combined command below. Do not run a second worker for an already active dataset/model/shot job. The ledger synchronizes money reservations across processes; it is not a lock for duplicate experiment jobs.

```bash
python scripts/run_budgeted_hosted.py --config configs/hosted_budget.json \
  --model-keys openai_economical openai_frontier \
  --data data/pilot/sst2 data/pilot/trec --shots 0 4 --seed 42 \
  --output results/hosted --max-requests 200 \
  --budget-usd 7.50 --ledger results/openai-budget.jsonl \
  --execute --prompt-api-key
```

The resume command deliberately omits `--init-ledger`. The original US$7.50 budget, data, model settings, driver provenance and result paths must remain consistent. Saved predictions are checkpointed and reused; changing provenance can produce a different run rather than resuming. There are no automatic retries. `--max-requests 200` caps new requests per run and rejects prepared test sets larger than 200 rather than shortening the test set. It is not the monetary control.

Jev through OpenRouter has a separate wrapper that preserves the frozen classification core and ledger implementation. This command plans the four-run matrix without requesting credentials or sending a call:

```bash
python scripts/run_openrouter_jev.py --config configs/jev_openrouter.json \
  --model-key jev_openrouter --data data/pilot/sst2 data/pilot/trec \
  --shots 0 4 --seed 42 --output results/jev --max-requests 200 \
  --budget-usd 2.50 --ledger results/jev-budget.jsonl
```

For a first execution with no ledger, add `--execute --init-ledger --prompt-api-key`. To resume an initialized study, add only `--execute --prompt-api-key`, retaining the ledger, lock anchor and result folders. The hidden prompt requests `OPENROUTER_API_KEY`; a direct TypeSafe key cannot authenticate to OpenRouter. The wrapper uses the official `/api/v1/systemone` native Choice route and records its own source hash separately. `--stop-after-new-requests 1` supports a first-row transport check: the row is checkpointed and reused when the identical study resumes, without an extra request. This operational pause is not a prompt-tuning step.

There are 800 requests in the full Jev matrix. Each reserves **US$0.002688**, so the complete matrix retains **US$2.1504** of the US$2.50 allocation. Jev reservations are never released, even when reported usage costs less. No automatic retries or model fallbacks are used. Unexpected model/provider identity, usage or cost above the reserved assumptions halts the ledger for review. Errors still retain their reservation. All request bodies use the existing frozen shared prompt and native Choice criteria.

## What the ledger bounds

The driver reserves a conservative amount **before each HTTP request**, with an append-only, hash-linked ledger, durable writes and a persistent POSIX file lock. All workers in an allocation must use the same ledger and lock on the same local filesystem. Both files are required to resume. Never delete, truncate, replace or move one without the other, or create a fresh ledger to bypass accumulated reservations. The allocation stored in the ledger is immutable. A missing/inconsistent ledger, stale price declaration or exhausted allocation stops execution before the next request; unfinished runs remain unfinished.

OpenAI reservations use the prompt's UTF-8 byte length plus **2,048 assumed chat-overhead tokens**, a 1.25 multiplier on the input price, and the configured maximum output tokens. The byte-length/overhead calculation is a conservative protocol assumption, not a provider-certified token count. Inputs above the declared 272,000-token pricing tier are rejected. The driver forces standard service and allows only the expected text-only request shape. Jev reservations use 64,000 input tokens per request, with free output tokens. This is deliberately larger than OpenRouter's cataloged 32k context; it is a cost reservation, not a claim that the route accepts 64k prompts.

For a successful OpenAI response with positive, complete input/output usage and a matching returned model identifier, the reservation is settled once using reported tokens at **1.25 times the standard input rate plus the standard output rate**. Cache discounts are not credited. The released difference becomes available for later requests within the same ledger. Invalid predictions, errors, timeouts, unknown usage, unrecognized model identifiers and interrupted calls retain their full reservation. A timeout may still have been billed. Reported usage above the reserved assumptions halts the ledger for reconciliation.

`reported_usage_estimate` uses reported tokens at uncached standard rates. It can be lower than `settled_conservative_usd`, which includes the input multiplier. Neither field is an invoice. Ledger totals combine conservative settled charges and unresolved reservations; they are not necessarily actual provider spend. The bound depends on correct current prices, text-only requests, token/accounting assumptions and standard service. It excludes unrelated account spending, taxes and any unmodeled billing adjustment. Reconcile final provider billing before describing actual dollar cost. Local/Colab compute cost is separate from this hosted API authorization.

## Price sources and date

Verified against official documentation on **22 September 2026**. USD prices below are per million tokens for the standard short-context tier; no cache discount is assumed by the guard.

| Model | Standard input | Guard input rate (including multiplier) | Output | Official source |
|---|---:|---:|---:|---|
| `gpt-5.6-luna` | $0.20 | $0.25 | $1.20 | [OpenAI Luna model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna) |
| `gpt-6-astra` | $10.00 | $12.50 | $50.00 | [OpenAI Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra) |
| `jev-1.13.0` | $0.042 | $0.042 | Free | [TypeSafe model pricing and limits](https://docs.typesafe.ai/models) |
| `typesafe/jev-1.13` through OpenRouter | $0.042 | $0.042 | Free | [OpenRouter Jev catalog](https://openrouter.ai/typesafe/jev-1.13) |

OpenAI documents 1.25-times input rates for cache writes and higher rates beyond 272k input tokens; the guard accounts for the former and rejects the latter. [Luna pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna), [Astra pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).

Both drivers require prices verified on the current UTC date. For a later OpenAI execution, review the official source and supply a `--prices-json` declaration with the required fields and explicit verification statement documented in `select_price`. The separate OpenRouter wrapper instead fixes its reviewed date, route, price and response-model allowlist in source. Review the official sources before making a dated replacement; do not merely change the date without rechecking prices. Keep existing ledger history and the same allocation. Updating driver or price provenance can change run identity and must be treated explicitly when reconciling a resumed study.
