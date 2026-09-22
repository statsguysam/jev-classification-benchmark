# Audited validation scoring diagnostic

The completed diagnostic covers 136 model contexts on 34 distinct validation cases: 16 Breast Cancer cases (8 per class) and 18 Wine cases (6 per class), evaluated with SmolLM2 and Granite at zero and four training examples per class. Each context compares numeric ID scoring with numeric ID plus EOS scoring from the same forward pass per candidate.

[AUDITED_SUMMARY.json](AUDITED_SUMMARY.json) contains the verified classification metrics. All 136 ordered records, frozen source and protocol identities, token-score decomposition, class counts, and fixed/harmed counts were checked against the unchanged [raw summary](SUMMARY.json) and predictions. Tokenization was bound by the independently pinned original preflight; this offline audit did not load tokenizers or models.

The raw evaluator's zero latency values are placeholders, not measurements. The audited summary omits numeric timing fields and marks latency as not measured. No inference-time or speed comparison is supported by this diagnostic.

| Dataset | Model | Examples/class | Validation cases | ID+EOS accuracy | ID-only accuracy | Changed | Fixed / harmed by ID-only |
|---|---|---:|---:|---:|---:|---:|---:|
| breast_cancer | smollm2 | 0 | 16 | 50.0% | 50.0% | 0 | 0 / 0 |
| breast_cancer | smollm2 | 4 | 16 | 50.0% | 50.0% | 0 | 0 / 0 |
| wine | smollm2 | 0 | 18 | 33.3% | 33.3% | 0 | 0 / 0 |
| wine | smollm2 | 4 | 18 | 33.3% | 33.3% | 0 | 0 / 0 |
| breast_cancer | granite | 0 | 16 | 50.0% | 50.0% | 0 | 0 / 0 |
| breast_cancer | granite | 4 | 16 | 68.8% | 68.8% | 0 | 0 / 0 |
| wine | granite | 0 | 18 | 33.3% | 33.3% | 0 | 0 / 0 |
| wine | granite | 4 | 18 | 33.3% | 33.3% | 0 | 0 / 0 |

Compare the scoring rules only within identical selected validation rows. These small balanced samples have a different class prevalence from the earlier test sets; their absolute accuracies must not be compared directly with historical test accuracy. Reusing the same 34 cases across models and shot counts does not create 136 independent cases.

This is exploratory development evidence after earlier test behavior was observed. The 34 validation labels add a diagnostic label budget. No test results were rescored, no winning method was selected, and no confidence gate was fitted. This isolates two scoring rules within each frozen model context, not architecture differences. Free generation, adapters, and claims about performance on new held-out data remain outside this diagnostic.
