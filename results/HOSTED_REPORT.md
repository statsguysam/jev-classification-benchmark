# Recorded benchmark results

Only completed, measured runs are listed. Missing models have not been benchmarked; no result is inferred.

Compare rows only when test-manifest hashes, label budgets, preprocessing, and protocol match. These are dataset-specific results, not an overall model ranking.

| Dataset | Model | Method / output protocol | Labels per class | Total train / dev labels | Seed | N | Accuracy | Macro F1 | 95% F1 CI | Failures | Prob. coverage |
|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|
| sst2 | gpt-5.6-luna | zero_shot / label generation | 0 | 0 / 0 | 42 | 200 | 0.9250 | 0.9249 | [0.8845, 0.9600] | 0 | 0% |
| sst2 | gpt-5.6-luna | few_shot / label generation | 4 | 8 / 0 | 42 | 200 | 0.9550 | 0.9550 | [0.9248, 0.9800] | 0 | 0% |
| sst2 | gpt-6-astra | zero_shot / label generation | 0 | 0 / 0 | 42 | 200 | 0.9700 | 0.9700 | [0.9450, 0.9900] | 0 | 0% |
| sst2 | gpt-6-astra | few_shot / label generation | 4 | 8 / 0 | 42 | 200 | 0.9750 | 0.9750 | [0.9500, 0.9950] | 0 | 0% |
| trec | gpt-5.6-luna | zero_shot / label generation | 0 | 0 / 0 | 42 | 200 | 0.5250 | 0.5592 | [0.4933, 0.6171] | 0 | 0% |
| trec | gpt-5.6-luna | few_shot / label generation | 4 | 24 / 0 | 42 | 200 | 0.8500 | 0.8479 | [0.7933, 0.9013] | 0 | 0% |
| trec | gpt-6-astra | zero_shot / label generation | 0 | 0 / 0 | 42 | 200 | 0.9700 | 0.9580 | [0.9173, 0.9915] | 0 | 0% |
| trec | gpt-6-astra | few_shot / label generation | 4 | 24 / 0 | 42 | 200 | 0.9700 | 0.9566 | [0.9167, 0.9908] | 0 | 0% |

Probability metrics are computed only where actual class probabilities are available. Do not compare their quality without checking coverage and probability semantics.

Classical prediction timings are amortized batch timings; hosted latency measures sequential end-to-end requests. They are not direct serving-speed comparisons.

Incomplete runs: 0.
