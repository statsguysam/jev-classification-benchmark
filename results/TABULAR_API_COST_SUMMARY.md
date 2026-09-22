# Tabular API cost reconciliation

Snapshot: 2026-09-22T13:45:11.227251+00:00. **18/18 hosted runs complete**, with 2,472/2,472 persisted prediction rows.

The tabular ledger conservatively accounts for **US$13.229108700**. Including the unchanged text-study US$5.293007000, cumulative accounting is **US$18.522115700**, within the approved **US$20.00** ceiling. The tabular allocation is US$14.70.

All 2,472 captured reservations are included: 1,647 validated settlements and 825 full retained reservations. 0 requests lack a final outcome; 0 reservations lack a persisted prediction. These can be active or interrupted attempts and are never discarded on resume.

API-reported dollar costs are available only for some Jev responses. OpenAI amounts are independently recomputed standard-rate token estimates. Missing costs/usage remain unknown; their known subtotals do not represent a complete bill. Conservative accounting retains unresolved reserves. None of these figures is a provider invoice.

| Model | Reservations / settled | Known API-cost requests | Known API-cost subtotal USD | Known token-estimate requests | Token-estimate subtotal USD | Conservative accounted USD |
|---|---:|---:|---:|---:|---:|---:|
| typesafe/jev-1.13 | 824 / 0 | 821/824 | 0.048789090 | 821/824 | 0.048789090 | 2.214912000 |
| gpt-5.6-luna | 824 / 824 | 0/824 | 0.000000000 | 824/824 | 0.157638400 | 0.196059200 |
| gpt-6-astra | 824 / 823 | 0/824 | 0.000000000 | 823/824 | 8.745930000 | 10.818137500 |

Jev response IDs are checked against ledger outcomes and must be unique. The frozen OpenAI adapter did not retain provider request IDs; its durable reservation IDs, usage, model and settlements are reconciled instead.

Ledger and prediction files captured while holding the existing ledger lock. Outcomes without persisted predictions, including interrupted or currently active requests, remain included at their full reservation or validated settlement.

[Machine-readable snapshot](tabular/API_COST_SUMMARY.json). Preserve the canonical ledger and its `.lock` anchor together. This report performs no API calls or settlements. Refresh with `python scripts/summarize_tabular_costs.py`; a snapshot taken during execution is deliberately incomplete.
