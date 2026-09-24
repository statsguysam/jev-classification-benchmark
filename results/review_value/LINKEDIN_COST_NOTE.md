# API cost comparison supporting the LinkedIn draft

This is a retrospective comparison of one pass over the same 200 TREC test questions, with four labeled examples per class (24 total). The three saved runs have identical ordered test row IDs and training example IDs. All 200 records in each run have the cost fields used below. No new model calls were made for this analysis.

| Method | Correct | Accuracy | API cost for 200 questions, USD | Cost basis |
|---|---:|---:|---:|---|
| GPT-6 Astra alone | 194/200 | 97.0% | 1.356350000 | Token-based standard-rate estimate |
| TypeSafe Jev 1.13 alone | 171/200 | 85.5% | 0.008723022 | OpenRouter-reported charges |
| GPT-6 Astra followed by Jev | 173/200 | 86.5% | 1.365568622 | Original source estimate plus reported review charges |

The added Jev review cost was $0.009218622, approximately 0.68% of the source estimate. In this condition it corrected 3 source mistakes and overturned 24 correct source decisions, reducing accuracy by 10.5 percentage points. The added API charge was small; the decision quality still worsened. Jev alone offered a lower-cost, lower-accuracy operating point than Astra alone in this experiment.

## Derivation and evidence

Sum these fields across every prediction record, without selecting by correctness:

- Astra: `metadata.budget.reported_usage_estimate.usd` in [source predictions](../hosted/trec__gpt-6-astra__f4c124cbaa8e/predictions.jsonl), with [run metadata](../hosted/trec__gpt-6-astra__f4c124cbaa8e/run.json).
- Jev alone: `metadata.openrouter.reported_cost_usd` in [standalone predictions](../jev/trec__jev-1.13__bc52fb6a4b0f/predictions.jsonl), with [run metadata](../jev/trec__jev-1.13__bc52fb6a4b0f/run.json).
- Jev review: `metadata.openrouter.reported_cost_usd` in [review predictions](../text_extension/review/trec__astra__k4__jev-review/predictions.jsonl), with [run metadata](../text_extension/review/trec__astra__k4__jev-review/run.json).

The source estimate is 131,035 input tokens at $10 per million plus 920 output tokens, including reasoning, at $50 per million: $1.356350000. These are the frozen rates recorded for the September 22, 2026 source run, not a current price quote. No cache discount is assumed.

The reported Jev totals also reconcile to the recorded input-token rate of $0.042 per million and zero output-token charge: 207,691 input tokens for standalone classification ($0.008723022), and 219,491 for review ($0.009218622). Both runs resolved `typesafe/jev-1.13-20260917`.

The reconstructed pipeline total is $1.356350000 + $0.009218622 = $1.365568622. Source predictions were cached and reused during review, so this total is not newly incurred review-phase spending. Existing source generation is counted once when reconstructing the complete pipeline.

## Interpretation limits

- OpenAI values are estimates from recorded usage and dated standard rates, not invoice totals. Cache billing details, taxes and other charges were not reconciled.
- Jev values are provider-reported API charges. The displayed rounded post values are $0.009, $1.36 and $1.37.
- This compares saved operating points. Jev alone used its standalone prompt; the comparison does not isolate the causal effect of adding a proposal. The separate matched-prompt experiment addresses that question.
- Local and Colab compute, classical-model training and inference, and engineering work were not priced. No classical-ML dollar ranking or production cost-saving claim follows from this table.
- Conservative budget reservations are not substituted for token-based estimates or reported charges. The study-wide budget reconciliation is a different accounting measure.

[Historical OpenAI cost summary](../API_COSTS.md) · [Historical Jev cost summary](../JEV_COSTS.md) · [Frozen budget assumptions](../../docs/BUDGET.md) · [Full evidence story](STORY.md)
