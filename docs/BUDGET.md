# Hosted API budget and resumption

The approved total is **US$10**. Completed execution reconciles to US$3.142607 in conservative settlements and US$2.568174 in reported-token standard-rate estimates; neither is an invoice. See [final accounting](../results/API_COSTS.md). The current study allocates **US$7.50 to OpenAI** and retains **US$2.50 for Jev**. The OpenAI ceiling remains US$7.50 even while the Jev allocation is unused. Jev API access is unavailable, so Jev has **not been measured**; absence of a run is not a zero benchmark score.

| Allocation | Ledger | Scope as of 22 September 2026 |
|---|---|---|
| OpenAI: US$7.50 | `results/openai-budget.jsonl` and `results/openai-budget.jsonl.lock` | SST-2 and TREC, Luna and Astra, zero-shot and four-shot-per-class, seed 42, 200 test rows per run; eight completed runs, 1,600 requests |
| Jev: US$2.50 | `results/jev-budget.jsonl` and `results/jev-budget.jsonl.lock`, to be created only when execution becomes possible | Reserved allocation, unspent; API access unavailable |

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

If Jev access becomes available, its separate allocation can be planned without requesting credentials or making a call:

```bash
python scripts/run_budgeted_hosted.py --config configs/hosted_budget.json \
  --model-keys jev --data data/pilot/sst2 data/pilot/trec \
  --shots 0 4 --seed 42 --output results/hosted --max-requests 200 \
  --budget-usd 2.50 --ledger results/jev-budget.jsonl
```

Only its first actual execution would add `--execute --init-ledger --prompt-api-key`; subsequent resumptions would omit initialization. No Jev execution is currently possible.

## What the ledger bounds

The driver reserves a conservative amount **before each HTTP request**, with an append-only, hash-linked ledger, durable writes and a persistent POSIX file lock. All workers in an allocation must use the same ledger and lock on the same local filesystem. Both files are required to resume. Never delete, truncate, replace or move one without the other, or create a fresh ledger to bypass accumulated reservations. The allocation stored in the ledger is immutable. A missing/inconsistent ledger, stale price declaration or exhausted allocation stops execution before the next request; unfinished runs remain unfinished.

OpenAI reservations use the prompt's UTF-8 byte length plus **2,048 assumed chat-overhead tokens**, a 1.25 multiplier on the input price, and the configured maximum output tokens. The byte-length/overhead calculation is a conservative protocol assumption, not a provider-certified token count. Inputs above the declared 272,000-token pricing tier are rejected. The driver forces standard service and allows only the expected text-only request shape. Jev reservations use its entire declared 64,000-token request maximum, with free output tokens.

For a successful OpenAI response with positive, complete input/output usage and a matching returned model identifier, the reservation is settled once using reported tokens at **1.25 times the standard input rate plus the standard output rate**. Cache discounts are not credited. The released difference becomes available for later requests within the same ledger. Invalid predictions, errors, timeouts, unknown usage, unrecognized model identifiers and interrupted calls retain their full reservation. A timeout may still have been billed. Reported usage above the reserved assumptions halts the ledger for reconciliation.

`reported_usage_estimate` uses reported tokens at uncached standard rates. It can be lower than `settled_conservative_usd`, which includes the input multiplier. Neither field is an invoice. Ledger totals combine conservative settled charges and unresolved reservations; they are not necessarily actual provider spend. The bound depends on correct current prices, text-only requests, token/accounting assumptions and standard service. It excludes unrelated account spending, taxes and any unmodeled billing adjustment. Reconcile final provider billing before describing actual dollar cost. Local/Colab compute cost is separate from this hosted API authorization.

## Price sources and date

Verified against official documentation on **22 September 2026**. USD prices below are per million tokens for the standard short-context tier; no cache discount is assumed by the guard.

| Model | Standard input | Guard input rate (including multiplier) | Output | Official source |
|---|---:|---:|---:|---|
| `gpt-5.6-luna` | $0.20 | $0.25 | $1.20 | [OpenAI Luna model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna) |
| `gpt-6-astra` | $10.00 | $12.50 | $50.00 | [OpenAI Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra) |
| `jev-1.13.0` | $0.042 | $0.042 | Free | [TypeSafe model pricing and limits](https://docs.typesafe.ai/models) |

OpenAI documents 1.25-times input rates for cache writes and higher rates beyond 272k input tokens; the guard accounts for the former and rejects the latter. [Luna pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna), [Astra pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).

The driver requires prices verified on the current UTC date. On a later date, review the official source and supply a `--prices-json` declaration with the required fields and explicit verification statement documented in `select_price`; do not merely change the date without rechecking prices. Keep existing ledger history and the same allocation. Updating driver or price provenance can change run identity and must be treated explicitly when reconciling a resumed study.
