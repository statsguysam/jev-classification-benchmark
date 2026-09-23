# Numerical expansion status

**23 September 2026 UTC: all 68/68 conditions and 72/72 contrasts are complete and audited.** All 24 numerical review conditions now have their full held-out denominator. These are first-attempt results; historical failures remain unchanged and count as incorrect.

| Condition family | Complete | Planned |
|---|---:|---:|
| Source LLMs | 24 | 24 |
| Source proposal → Jev review | 24 | 24 |
| Direct Jev references | 4 | 4 |
| Native-feature classical references | 16 | 16 |
| Total | 68 | 68 |

Twenty review conditions were executed by the numerical expansion, and four historical reviews were reused. There are no unfinished numerical review conditions or partial-test scores. This expansion adds no LoRA training.

The completed reviews retain nine actual Jev call failures: six Breast Cancer SmolLM2 four-shot billing failures, one Breast Cancer Luna four-shot failure, one Breast Cancer Astra four-shot failure, and one Wine Granite zero-shot failure. Wine Astra four-shot additionally retains one upstream source failure: Jev was not called for that row. The source matrix has that one failed inference; direct Jev has two failures. The completed [recovery analysis](../completion_20260923/RECOVERY_FINDINGS.md) records additional attempts separately and preserves these original outcomes.

## Accounting at numerical first-pass completion

The expansion ledger reconciles **1,500 requests, 1,500 result events and 1,500 saved predictions**. Eight calls have unknown reported charges. There are no orphan reservations/results or partial checkpoints.

| Quantity | USD |
|---|---:|
| Known reported charges for the 1,500 new reviews | 0.124087194 |
| Conservative retained amount for the new reviews | 4.032000000 |
| Prior conservative accounting in the numerical report | 19.325827700 |
| Numerical report's cumulative subtotal | 23.357827700 |
| Separate 23 September health-probe reservation | 0.002688000 |
| Protected subtotal including the probe, before text completion/retries/controls | **23.360515700** |
| Approved aggregate ceiling | **25.00** |

These are scoped accounting figures, not a live global total or a provider invoice. Known reported charges cover 1,492 of the 1,500 new reviews; unknown charges are not zero. All numerical expansion reservations remain retained in full. Four reused review conditions belong to prior accounting. Source inference costs and local compute remain incompletely measured. Text, recovery and controls are now complete. Their separate accounting is included in the [final reconciliation](../completion_20260923/COSTS.md): US$24.110484775 retained conservatively, with US$0.889515225 remaining under the same ceiling. The table above remains the numerical first-pass snapshot.

The completed first-pass evidence is in [COMPARISON.json](COMPARISON.json), [FINDINGS.md](FINDINGS.md) and [COSTS.json](COSTS.json). The [recovery protocol](../../docs/COMPLETION_RETRY_PROTOCOL.md) preserves original failures and records each additional attempt separately. See the [independent numeric evidence review](../completion_20260923/NUMERIC_EVIDENCE_REVIEW.md) for checked claims and limitations.
