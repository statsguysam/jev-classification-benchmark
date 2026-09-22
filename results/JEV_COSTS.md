# Jev cost reconciliation and combined API accounting

All **800 Jev request reservations** reconcile one-to-one with completed prediction artifacts and final ledger results, including **5 recorded errors**. Known OpenRouter-reported costs sum to **US$0.021164346**, covering **797/800 requests**; **3 request costs remain unknown**. Missing costs are never counted as zero. These values are not a provider invoice.

Independently applying $0.042 per million known input tokens and zero output cost gives a known token-rate subtotal of **US$0.021164346**, covering **797/800 input-usage records**. Complete-cost/usage totals are null in JSON when their coverage is incomplete.

All US$0.002688000 per-request reservations remain retained: **US$2.150400** against the separate US$2.50 Jev allocation. Known reported cost plus full reservations for unknown-cost requests is **US$0.029228346**: a conditional accounted upper estimate, distinct from both an invoice and the all-request conservative ledger total.

| Dataset | Method | Requests / errors | Known input / output tokens | Cost coverage | Known API-cost subtotal USD | Known token-rate subtotal USD | Retained reservation USD |
|---|---|---:|---:|---:|---:|---:|---:|
| sst2 | zero_shot | 200 / 2 | 78806 / 6138 | 198/200 | 0.003309852 | 0.003309852 | 0.537600 |
| sst2 | few_shot | 200 / 0 | 123597 / 6200 | 200/200 | 0.005191074 | 0.005191074 | 0.537600 |
| trec | few_shot | 200 / 0 | 207691 / 11800 | 200/200 | 0.008723022 | 0.008723022 | 0.537600 |
| trec | zero_shot | 200 / 3 | 93819 / 11741 | 199/200 | 0.003940398 | 0.003940398 | 0.537600 |

## Combined accounting under US$10

| Component / basis | USD |
|---|---:|
| OpenAI: reported-token standard-rate estimate | 2.568173600 |
| Jev: known API-cost subtotal (797/800 requests) | 0.021164346 |
| OpenAI estimate + Jev known API-cost subtotal | 2.589337946 |
| OpenAI estimate + Jev conditional accounted upper estimate | 2.597401946 |
| Both providers: known token-rate subtotals | 2.589337946 |
| OpenAI conservative settlement + Jev retained reservations | 5.293007000 |
| Remaining across both separate allocations | 4.706993000 |

OpenAI API-reported dollar cost is not available here: its number is derived from reported tokens at dated standard rates. OpenAI conservative settlements include a 1.25 input multiplier; Jev retains its entire reservation without settlement. The two ledgers retain their separate US$7.50/US$2.50 limits. Remaining allocations are not automatically transferable.

Reported API cost, known token-rate subtotals and retained reservations are distinct. A timeout/error may still be billed; unknown cost/usage is preserved and its full reservation remains. The conditional accounted upper estimate assumes each unknown request fits its reserved limit. Caching, taxes, unrelated account activity, billing adjustments and Colab compute can make actual invoiced totals differ. Known usage and API cost are checked against each corresponding ledger result with exact Decimal strings. Returned model/provider/request IDs must match when present; unknown response metadata is allowed only for recorded errors. Every reserved endpoint is checked. Source/ledger/artifact hashes are retained in the JSON.

The rate was frozen for the execution at [OpenRouter Jev pricing](https://openrouter.ai/typesafe/jev-1.13), with the [native System One route](https://openrouter.ai/docs/guides/community/typesafe-sdk). This reconciliation does not update prices or make requests.

[Machine-readable reconciliation](JEV_COSTS.json) · [OpenAI-only accounting](API_COSTS.md) · [Budget assumptions](../docs/BUDGET.md)

Rebuild with `python scripts/summarize_jev_costs.py` only after all jobs finish. Keep each ledger and its `.lock` anchor together.
