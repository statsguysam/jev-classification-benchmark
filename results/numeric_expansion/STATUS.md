# Numerical expansion status

**Snapshot: 61 of 68 conditions complete and audited; paid review is halted by billing errors.** This describes the saved evidence checked on 22 September 2026 UTC. The full six-model experiment is not complete.

| Condition family | Complete | Planned |
|---|---:|---:|
| Source LLMs | 24 | 24 |
| Source proposal → Jev review | 17 | 24 |
| Direct Jev references | 4 | 4 |
| Native-feature classical references | 16 | 16 |
| Total | 61 | 68 |

The report retains all 68 conditions and all 72 comparisons. **54 comparisons are complete.** Unfinished reviews and their dependent comparisons have null scores and intervals, with no inferred transition or failure counts. Their source links remain available. Saved partial predictions are evidence of progress, not a completed held-out evaluation.

The 17 completed reviews include four historical reviews reused without new requests and 13 completed reviews from this expansion. All Qwen, SmolLM2, Granite, Luna and Astra source conditions are complete. This expansion adds no LoRA training.

## What remains

| Dataset | Source model | Examples per class | Saved review rows | Next-unseen rows remaining |
|---|---|---:|---:|---:|
| Breast Cancer | SmolLM2 1.7B | 4 | 85 / 114 | 29 |
| Breast Cancer | Granite 3.3 2B | 0 | 0 / 114 | 114 |
| Breast Cancer | Granite 3.3 2B | 4 | 0 / 114 | 114 |
| Wine | SmolLM2 1.7B | 0 | 0 / 36 | 36 |
| Wine | SmolLM2 1.7B | 4 | 0 / 36 | 36 |
| Wine | Granite 3.3 2B | 0 | 0 / 36 | 36 |
| Wine | Granite 3.3 2B | 4 | 0 / 36 | 36 |
| Total | | | 85 saved in unfinished conditions | **401** |

The Breast Cancer SmolLM2 four-per-class checkpoint contains **six recorded HTTP 402 failures among its 85 saved rows**. Its last three predictions triggered the billing halt. Recovery must preserve those failures and continue from the next unseen row; it must not replace them with successful retries. They will count as incorrect in the full 114-row denominator if the condition completes.

## Accounting

The current expansion ledger reconciles **1,099 requests with 1,099 result events and saved predictions**. Seven calls have unknown reported cost; six are the recorded billing failures in the unfinished condition. No reserved request is missing its result or saved prediction in this snapshot.

| Quantity | USD |
|---|---:|
| Known reported charges for the new reviews | 0.084934668 |
| Conservative retained amount for the new reviews | 2.954112000 |
| Prior conservative accounting | 19.325827700 |
| Cumulative conservative accounting | **22.279939700** |
| Approved cumulative ceiling | **25.00** |

Known reported charges are a partial subtotal, not an invoice or the total expense; unknown costs are not treated as zero. Every new review request retains its original reservation. The remaining 401 planned calls would retain another US$1.077888 under the current reservation assumptions. Provider funding is needed despite remaining benchmark budget headroom. Local compute cost is unmeasured.

## Recovery

The recovery helper is prepared; this snapshot does not claim that funded recovery has executed. Preserve the original ledger, locks, source manifests, checkpoints and failures. Once provider credits are funded and the dated UTC transport guard is valid, use the dedicated wrapper for a single next-unseen request:

```bash
python scripts/recover_expanded_numeric_billing.py \
  --dataset breast_cancer --model-key smollm2 --shots 4 \
  --execute --prompt-api-key --stop-after-new-requests 1
```

The wrapper requires a fresh authenticated positive-credit check, audits the exact billing halt, preserves recovery evidence, and retains the original request and monetary limits. A positive balance does not promise enough funding for all remaining calls. The ordinary runner alone does not clear this halt. If the first resumed request fails, execution stops again. No saved failed row is retried.

The frozen route/pricing guard is dated **22 September 2026 UTC**. After that UTC date, recovery stops before authentication or mutation pending explicit route/price reverification and review of a compatible recovery path. Do not change the date or frozen code merely to bypass it. See [budget and recovery instructions](../../docs/BUDGET.md#current-halt-and-guarded-recovery) for the read-only audit, successful-probe continuation and reconciliation sequence.

The current audited evidence is in [COMPARISON.json](COMPARISON.json), [FINDINGS.md](FINDINGS.md) and [COSTS.json](COSTS.json). Completion and published metrics must be regenerated from those preserved raw artifacts after any further execution.
