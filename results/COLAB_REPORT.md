# Recorded benchmark results

Only completed, measured runs are listed. Missing models have not been benchmarked; no result is inferred.

Compare rows only when test-manifest hashes, label budgets, preprocessing, and protocol match. These are dataset-specific results, not an overall model ranking.

| Dataset | Model | Method / output protocol | Labels per class | Total train / dev labels | Seed | N | Accuracy | Macro F1 | 95% F1 CI | Failures | Prob. coverage |
|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|
| sst2 | Qwen/Qwen3-4B-Instruct-2507 | zero_shot / closed-label likelihood | 0 | 0 / 0 | 42 | 200 | 0.9150 | 0.9147 | [0.8695, 0.9500] | 0 | 100% |
| sst2 | Qwen/Qwen3-4B-Instruct-2507 | lora / closed-label likelihood | 4 | 8 / 0 | 42 | 200 | 0.9350 | 0.9350 | [0.8950, 0.9650] | 0 | 100% |
| sst2 | Qwen/Qwen3-4B-Instruct-2507 | few_shot / closed-label likelihood | 4 | 8 / 0 | 42 | 200 | 0.9550 | 0.9550 | [0.9200, 0.9800] | 0 | 100% |
| sst2 | linear_svc | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.5000 | 0.4869 | [0.4172, 0.5528] | 0 | 0% |
| sst2 | logistic_regression | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.4950 | 0.4826 | [0.4143, 0.5469] | 0 | 100% |
| sst2 | majority | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.4900 | 0.3289 | [0.3289, 0.3289] | 0 | 100% |
| sst2 | multinomial_nb | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.5400 | 0.5388 | [0.4674, 0.6021] | 0 | 100% |
| trec | Qwen/Qwen3-4B-Instruct-2507 | zero_shot / closed-label likelihood | 0 | 0 / 0 | 42 | 200 | 0.5250 | 0.4820 | [0.3756, 0.5552] | 0 | 100% |
| trec | Qwen/Qwen3-4B-Instruct-2507 | lora / closed-label likelihood | 4 | 24 / 0 | 42 | 200 | 0.8000 | 0.7510 | [0.6728, 0.8171] | 0 | 100% |
| trec | Qwen/Qwen3-4B-Instruct-2507 | few_shot / closed-label likelihood | 4 | 24 / 0 | 42 | 200 | 0.8300 | 0.8518 | [0.8089, 0.8913] | 0 | 100% |
| trec | linear_svc | classical / classical native | 4 | 24 / 0 | 42 | 200 | 0.4950 | 0.4757 | [0.4061, 0.5403] | 0 | 0% |
| trec | logistic_regression | classical / classical native | 4 | 24 / 0 | 42 | 200 | 0.5000 | 0.4857 | [0.4173, 0.5552] | 0 | 100% |
| trec | majority | classical / classical native | 4 | 24 / 0 | 42 | 200 | 0.0200 | 0.0065 | [0.0065, 0.0065] | 0 | 100% |
| trec | multinomial_nb | classical / classical native | 4 | 24 / 0 | 42 | 200 | 0.5000 | 0.4857 | [0.4194, 0.5466] | 0 | 100% |

Probability metrics are computed only where actual class probabilities are available. Do not compare their quality without checking coverage and probability semantics.

Classical prediction timings are amortized batch timings; hosted latency measures sequential end-to-end requests. They are not direct serving-speed comparisons.

Incomplete runs: 0.
