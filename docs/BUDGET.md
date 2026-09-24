# Hosted API budget and resumption

The approved cumulative total is **US$25**: the original US$10 ceiling, US$10 for the tabular extension, and US$5 for the expanded numerical zero/few-shot review experiment. The text extension, explicit failure recovery and matched proposal controls share remaining headroom inside this ceiling; they add no spending authorization. These ledgers enforce reservations under the documented pricing assumptions; they are not provider invoices or account-wide caps.

## Completed execution and final reconciliation

The [final audited reconciliation](../results/completion_20260923/COSTS.md) counts **US$24.110484775** against the **US$25** authorization, leaving **US$0.889515225**. It includes all ten ledgers, the full health-probe reservation, historical failure recovery, completed proposal controls and post-control recovery. These are retained conservative charges/reservations, not a provider invoice; local and Colab compute are unpriced. Allocation ceilings must not be added together as spending.

All 1,500 numerical expansion requests, all 68
numerical conditions (including 24 reviews), and all 72 contrasts are complete
and audited. The stage reports US$0.124087194 in known provider charges and
eight unknown-charge calls; unknown charges are not zero. Its ledger retains
US$4.032000000. Together
with the fixed earlier US$19.325827700 and the US$0.002688000 health allowance,
the conservative subtotal before text reviews is **US$23.360515700**. This is a
scoped subtotal, not the live cumulative amount after text, recovery or controls.

The text first pass is complete: all 4,800 requests, 24 reviews, 68 total
conditions and 72 contrasts are audited. Its stage retains **US$0.301477771**
in conservative charges/reservations, with **US$0.138994044** in known reported
charges for 4,794 calls and six unknown charges. The 48 saved review errors
remain incorrect in first-attempt metrics. Together the original numerical and
text matrices contain **136 complete conditions, including 48 reviews**.

[Historical recovery](../results/completion_20260923/RECOVERY_FINDINGS.md) used
71 new calls to recover 65 of 66 failed paid requests and supply one previously
skipped dependent review; one invalid response remained after the finite policy.
The [matched controls](../results/review_controls/FINDINGS.md) completed all
**1,714 requests, 12 primary arms and eight contrasts**. A separate
[post-control recovery](../results/completion_20260923/CONTROL_RECOVERY_FINDINGS.md)
recovered all 15 failed control/repeat requests in 15 new calls. Original valid
responses, first-attempt metrics and serving-repeat diagnostics remain unchanged.
No old failed reservation was released.

| Completed phase | New calls | Retained conservative USD |
|---|---:|---:|
| Historical failure recovery, including dependent first call | 71 | 0.215772500 |
| Matched controls, including exact-prompt repeats | 1,714 | 0.192398804 |
| Post-control failure recovery | 15 | 0.040320000 |

The [completion protocol](COMPLETION_RETRY_PROTOCOL.md) and
[reproduction guide](COMPLETION_REPRODUCTION.md) describe the current path.
September 22 execution commands below document the original drivers and their
historical guard; the separately reviewed September 23 compatibility wrapper
records fresh verification without changing those frozen sources.

## Operational check on 23 September 2026

The isolated [Jev health probe](../scripts/probe_jev_health.py) has a single-request allowance of **US$0.002688**, recorded separately in `results/health_checks/jev-20260923/budget.jsonl` and its anchor. It uses a synthetic input and is excluded from benchmark metrics. Its full allowance remains reserved even if the provider reports a lower charge or the request fails. It does not retry or replace a historical prediction.

The completed numerical reservations, full text allocation and probe allowance total **US$24.960515700**, within the US$25 authorization. This was the allocation ceiling at that point, not a spending total. Later recovery and control allocations used the remaining audited budget. The controls allocator protects the probe ledger and anchor and subtracts the full probe allowance before assigning funds. Historical study reports keep their study-only totals; the probe allowance is additional. The check's route/pricing verification does not renew the historical producers' dated execution guards.

The [saved check](../results/health_checks/jev-20260923/run.json) completed successfully at **2026-09-23 14:48:43 UTC**. TypeSafe served `typesafe/jev-1.13-20260917`, returned the expected bounded choice in approximately 0.55 seconds, and reported US$0.000014658 for this request. The ledger still retains the full US$0.002688 allowance. Historical evidence was unchanged; the synthetic check contributes no benchmark accuracy or speed claim.

## Text pipeline extension allocation

**The text-review first pass is complete and audited: 4,800 calls across all twenty-four source → Jev conditions.** Its ledger records 4,752 eligible successful settlements and 48 retained failed-call reservations, with no missing result records or partial checkpoints. Four accepted exact-tie Choices remain successful responses under the frozen adapter contract; the [collector correction](OUTPUT_VALIDATION_NOTES.md) changes no saved prediction or charge. Existing Qwen/Luna/Astra and direct-Jev predictions are reused; SmolLM2/Granite used Colab public weights. Reusing paid source predictions creates no new source-model charge in this stage, but it does not make a fresh two-stage pipeline free.

The [completed text report](../results/text_extension/COMPARISON.json) reconciles **US$0.301477771** retained by this stage against its US$1.60 allocation. Known reported charges total **US$0.138994044**; six of the 4,800 charges are unknown. Its **US$23.659305471** cumulative study subtotal includes the fixed earlier work and full numerical expansion, but excludes the separate health-probe allowance and subsequent recovery/controls. It is a scoped subtotal, not a provider invoice. The [final cumulative reconciliation](../results/completion_20260923/COSTS.md) is US$24.110484775 after the health probe, recovery and controls.

The new `results/text_extension/review-budget.jsonl` has a **US$1.60 allocation**, separate from all previous ledgers. Its envelope protects the entire 1,500-request numerical expansion, including the rows that were unfinished at allocation:

| Protected amount | USD |
|---|---:|
| Fixed conservative amount before the numerical expansion | 19.325827700 |
| Full numerical expansion allowance: 1,500 × 0.002688 | 4.032000000 |
| New text-review allocation | 1.600000000 |
| Maximum combined conservative accounting | **24.957827700** |
| Existing cumulative authorization | **25.000000000** |

Earlier text, tabular and initial numerical ledgers remain frozen. The now-complete numerical expansion ledger retains its original reservation policy and all US$4.032000000 reserved for its 1,500 calls. The new wrapper verifies those identities and the combined envelope before reservations; it does not release, rewrite or reprice previous events.

### New-call settlement policy

Each new text-review HTTP request first reserves the unchanged **US$0.002688**. Only a successful new response with verified TypeSafe/Jev identity, request ID, complete valid input/output usage, known nonnegative reported cost and no reservation-assumption overrun may settle once. Its retained charge is:

```text
min(original reservation,
    round upward to the next 0.000000001 USD(
        1.25 × max(reported cost, input tokens × 0.042 / 1,000,000)))
```

The 1.25 multiplier applies to the larger of reported cost and uncached input-token cost; it is not a 0.125 multiplier. Output tokens are free under the frozen route declaration. Any released difference becomes available only within this new text ledger. Failed predictions, unknown costs/usage, timeouts and interrupted calls retain the full reservation. Reported charges, usage-derived estimates and conservative charges remain distinct; none is represented as an invoice.

Keeping every request at its original reservation would have required US$12.9024, exceeding this allocation. Eligible conservative settlements allowed the actual 4,800-call first pass to finish within US$1.60; that allocation alone was not a guarantee of completion. The first-pass runner did not retry failures. Explicit recovery remains a separate experiment, and any partial condition has no final accuracy score.

### Planning, first execution and resumption

The [text review wrapper](../scripts/run_text_jev_review.py) defaults to read-only planning: no model calls, credential reads or ledger initialization. `--freeze-sources` explicitly writes immutable source manifests after auditing the saved predictions. It does not make paid calls by itself.

```bash
python scripts/run_text_jev_review.py \
  --datasets sst2 trec \
  --model-keys qwen_small qwen_main luna astra smollm2 granite --shots 0 4
```

Execution requires complete, frozen selected sources and a fresh authenticated **positive available-credit** check before creating a ledger or sending a model request. Account balance totals are not persisted in public evidence. A configured API-key quota is not proof that the provider account is funded. The dated transport guard still requires the frozen declaration verified on **22 September 2026 UTC**; after that UTC date, route/pricing reverification and a reviewed compatible path are needed. Funding alone does not bypass this guard. Do not edit frozen producers or merely change their date.

Only for a first execution with no existing text ledger, after those prerequisites are satisfied:

```bash
python scripts/run_text_jev_review.py \
  --datasets sst2 trec \
  --model-keys qwen_small qwen_main luna astra smollm2 granite --shots 0 4 \
  --freeze-sources --execute --init-ledger --prompt-api-key
```

The hidden terminal prompt reads `OPENROUTER_API_KEY`; do not place credentials in notebook cells, command arguments or committed files. `--stop-after-new-requests 1` can checkpoint one paid request without retrying it on resume. The runner has one execution lock, a 4,800-request cap, duplicate-row checks and global checkpoint/ledger reconciliation. A billing error stops execution; so do three consecutive other errors. A halted checkpoint requires investigation, not deletion or a replacement ledger.

To resume an initialized, non-halted text study after its previous worker has stopped, keep the same configuration, source manifests, ledger and lock, and use the same command **without `--init-ledger`**. Saved rows are retained. Do not change producer, prompt, source or budget identities and call it a resume. Rebuild the [text comparison](../results/text_extension/COMPARISON.json) after audited execution before changing completion or cost claims. See [protocol](TEXT_EXTENSION_PROTOCOL.md) and [reproduction](TEXT_EXTENSION_REPRODUCTION.md).

## Expanded numerical review stage

All four earlier ledgers and their lock files remain frozen. Their cumulative conservative amount is **US$19.325827700**, including the first four numerical review runs. The new `results/numeric_expansion/review-budget.jsonl` has a **US$5 allocation** and a **1,500-call limit**. Each Jev call reserves US$0.002688, so the complete expansion can reserve at most **US$4.032000**, bringing cumulative conservative accounting to at most **US$23.357827700**. Reported API charges are recorded separately; reservations are never released in this stage.

Six source models are compared at zero and four examples per class on Breast Cancer and Wine. Existing source predictions and the four original Qwen3 4B/Astra few-shot reviews are reused. SmolLM2 and Granite inference runs locally or on Colab with public weights. No new paid source-LLM calls are required.

### Historical September 22 halt and guarded recovery

The September 22 audited snapshot was **61/68 conditions complete**: 24 source runs, 17 reviews, four direct Jev references and 16 native references. Seven reviews were unfinished, and 54/72 comparisons were complete. See [the current status](../results/numeric_expansion/STATUS.md) for the completed September 23 snapshot. The following counts and commands describe that historical halt and its next-unseen-row recovery procedure, not current unfinished work. At that point the ledger contained **1,099 new Jev requests**, including seven calls with unknown reported cost. Known reported charges were **US$0.084934668**; these were not the complete expense. Retained reservations were **US$2.954112000** for this stage and **US$22.279939700 of US$25** cumulatively.

The historical Breast Cancer SmolLM2 four-per-class checkpoint had **85/114 saved rows**, including **six HTTP 402 failures**, and halted after three consecutive billing failures. Those six failures remain in its now-complete first-attempt score. At the historical halt, **401 next-unseen rows remained across seven conditions**; all have since been attempted. No partial checkpoint was scored as a completed test.

That historical recovery required funded provider credits as well as benchmark budget headroom. The ordinary runner alone could not clear the billing halt. Do not delete or reset the ledger, lock, failed predictions, source manifests or halted checkpoint, and do not retry already recorded failed rows. Stop any previous worker before recovery; the wrapper also acquires the same execution lock.

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

The expansion ledger is already initialized; never add `--init-ledger` to recovery or resumption. All 24 source conditions are complete. Source manifests freeze each individual model/dataset/shot artifact, and changing model, prompt, source code or provenance is not a resume. Unknown costs and interrupted calls retain their reservation; no automatic retries occur. Reconcile with `python scripts/audit_expanded_numeric_costs.py` after execution, then regenerate the numerical summary before updating completion claims. The 401 then-remaining calls have now retained another US$1.077888 under the unchanged policy, producing the completed numerical subtotal of US$23.357827700 before the separate health, text, recovery and control phases.

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

## Proposal-value controls: sequenced within the same US$25

The frozen follow-up completed **1,650 primary requests plus 64 identical-prompt repeats** across 550 cases, with all 12 primary arms and eight contrasts audited. Preparation and saved-results analysis made no model calls. The [control runner](../scripts/run_review_controls.py) requires both numerical and text matrices to reach 68/68 audited conditions and the separate historical failure-recovery run to be finalized and audited. It then pins every prior ledger and lock file, including recovery evidence and the full health-probe allowance, and may allocate only `25.00 − cumulative conservative accounting` to a new control ledger. No earlier reservation is released or changed. The control ledger retains US$0.192398804; [audited first-attempt results](../results/review_controls/FINDINGS.md) retain its 15 failed calls (14 primary, one repeat). The separate recovery ledger retains US$0.040320000 for 15 successful recovery calls; neither the primary results nor the original repeat diagnostic is overwritten.

The control ledger reused the text stage's verified settlement rule. Completion did not relax its guards: each call required its complete reservation before dispatch, unknown/failed charges remained reserved, and automatic retries or source-label fallback were prohibited. The separate post-control recovery policy ran only after the controls completed. Funding the provider account did not raise the authorized ceiling. Planning and readiness are read-only; the original allocation was permitted only after both matrices and historical failure recovery passed their audits:

```bash
python scripts/restore_review_controls_payload.py
python scripts/run_review_controls.py
```

After funding, current route/pricing verification and successful readiness, first execution uses `--execute --init-ledger --prompt-api-key`; resumption omits `--init-ledger`. Both historical execution locks and the new control lock are held during paid execution. A changed earlier ledger, partial append, orphan request, changed producer or request bundle stops the runner before a subsequent request. See the [frozen design](REVIEW_CONTROLS_PROTOCOL.md) and [reproduction guide](REVIEW_VALUE_REPRODUCIBILITY.md).
