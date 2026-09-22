# OpenAI-only API cost accounting

All 1,600 OpenAI request reservations reconcile one-to-one with completed prediction records and settlements. The conservative ledger total is **US$3.142607**, within the US$7.50 OpenAI allocation. The reported-token estimate at uncached standard rates is **US$2.568174**. These are accounting estimates, not a provider invoice.

This report covers **OpenAI only**. The separately allocated US$2.50 Jev arm and combined totals are reconciled in [JEV_COSTS.md](JEV_COSTS.md). The approved combined ceiling is US$10. Colab compute and unrelated account charges are outside these API calculations.

| Dataset | Model | Method | Requests | Input tokens | Output tokens, including reasoning | Standard-rate estimate USD | Conservative ledger USD |
|---|---|---|---:|---:|---:|---:|---:|
| sst2 | gpt-5.6-luna | zero_shot | 200 | 16199 | 800 | 0.004200 | 0.005010 |
| sst2 | gpt-5.6-luna | few_shot | 200 | 58599 | 800 | 0.012680 | 0.015610 |
| sst2 | gpt-6-astra | zero_shot | 200 | 16199 | 1297 | 0.226840 | 0.267338 |
| sst2 | gpt-6-astra | few_shot | 200 | 58599 | 1105 | 0.641240 | 0.787738 |
| trec | gpt-5.6-luna | zero_shot | 200 | 19435 | 800 | 0.004847 | 0.005819 |
| trec | gpt-5.6-luna | few_shot | 200 | 131035 | 800 | 0.027167 | 0.033719 |
| trec | gpt-6-astra | zero_shot | 200 | 19435 | 2010 | 0.294850 | 0.343438 |
| trec | gpt-6-astra | few_shot | 200 | 131035 | 920 | 1.356350 | 1.683938 |

The ledger includes a conservative input-price multiplier. Token pricing assumptions, caches, taxes and unrelated account activity can make the provider invoice differ. Per-row metadata retains both the initial reservation and final settlement. Preserve the ledger and its `.lock` anchor together. Re-running this script sends no API requests.

[Machine-readable reconciliation](API_COSTS.json) · [Budget assumptions and resume procedure](../docs/BUDGET.md)

Rebuild with `python scripts/summarize_api_costs.py` after workers have finished.
