# Recorded benchmark results

Only completed, measured runs are listed. Missing models have not been benchmarked; no result is inferred.

Compare rows only when test-manifest hashes, label budgets, preprocessing, and protocol match. These are dataset-specific results, not an overall model ranking.

| Dataset | Model | Method / output protocol | Labels per class | Total train / dev labels | Seed | N | Accuracy | Macro F1 | 95% F1 CI | Failures | Prob. coverage |
|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|
| ag_news | linear_svc | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.9050 | 0.9044 | [0.8588, 0.9401] | 0 | 0% |
| ag_news | linear_svc | classical / classical native | 4 | 16 / 0 | 42 | 200 | 0.4800 | 0.4732 | [0.4048, 0.5438] | 0 | 0% |
| ag_news | logistic_regression | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.9100 | 0.9094 | [0.8654, 0.9449] | 0 | 100% |
| ag_news | logistic_regression | classical / classical native | 4 | 16 / 0 | 42 | 200 | 0.4850 | 0.4782 | [0.4109, 0.5467] | 0 | 100% |
| ag_news | majority | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.2500 | 0.1000 | [0.1000, 0.1000] | 0 | 100% |
| ag_news | majority | classical / classical native | 4 | 16 / 0 | 42 | 200 | 0.2500 | 0.1000 | [0.1000, 0.1000] | 0 | 100% |
| ag_news | multinomial_nb | classical / classical native | 4 | 16 / 0 | 42 | 200 | 0.4700 | 0.4640 | [0.3939, 0.5350] | 0 | 100% |
| ag_news | multinomial_nb | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.9150 | 0.9146 | [0.8739, 0.9546] | 0 | 100% |
| banking77 | linear_svc | classical / classical native | full prepared train | 8995 / 1000 | 42 | 200 | 0.9250 | 0.9238 | [0.8887, 0.9474] | 0 | 0% |
| banking77 | linear_svc | classical / classical native | 4 | 308 / 0 | 42 | 200 | 0.5450 | 0.5083 | [0.4522, 0.5432] | 0 | 0% |
| banking77 | logistic_regression | classical / classical native | full prepared train | 8995 / 1000 | 42 | 200 | 0.9200 | 0.9159 | [0.8782, 0.9408] | 0 | 100% |
| banking77 | logistic_regression | classical / classical native | 4 | 308 / 0 | 42 | 200 | 0.5600 | 0.5336 | [0.4692, 0.5692] | 0 | 100% |
| banking77 | majority | classical / classical native | 4 | 308 / 0 | 42 | 200 | 0.0150 | 0.0004 | [0.0004, 0.0004] | 0 | 100% |
| banking77 | majority | classical / classical native | full prepared train | 8995 / 1000 | 42 | 200 | 0.0150 | 0.0004 | [0.0004, 0.0004] | 0 | 100% |
| banking77 | multinomial_nb | classical / classical native | full prepared train | 8995 / 1000 | 42 | 200 | 0.8850 | 0.8709 | [0.8245, 0.9012] | 0 | 100% |
| banking77 | multinomial_nb | classical / classical native | 4 | 308 / 0 | 42 | 200 | 0.5300 | 0.5016 | [0.4362, 0.5339] | 0 | 100% |
| imdb | linear_svc | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.9100 | 0.9100 | [0.8693, 0.9500] | 0 | 0% |
| imdb | linear_svc | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.4950 | 0.4950 | [0.4237, 0.5600] | 0 | 0% |
| imdb | logistic_regression | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.4950 | 0.4950 | [0.4237, 0.5600] | 0 | 100% |
| imdb | logistic_regression | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.9050 | 0.9050 | [0.8599, 0.9450] | 0 | 100% |
| imdb | majority | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.5000 | 0.3333 | [0.3333, 0.3333] | 0 | 100% |
| imdb | majority | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.5000 | 0.3333 | [0.3333, 0.3333] | 0 | 100% |
| imdb | multinomial_nb | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.5400 | 0.5370 | [0.4646, 0.6068] | 0 | 100% |
| imdb | multinomial_nb | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.8450 | 0.8450 | [0.7949, 0.8949] | 0 | 100% |
| sst2 | Qwen/Qwen2.5-0.5B-Instruct | zero_shot / closed-label likelihood | 0 | 0 / 0 | 42 | 200 | 0.5000 | 0.3503 | [0.3289, 0.3812] | 0 | 100% |
| sst2 | Qwen/Qwen2.5-0.5B-Instruct | few_shot / closed-label likelihood | 4 | 8 / 0 | 42 | 200 | 0.8500 | 0.8499 | [0.8000, 0.8950] | 0 | 100% |
| sst2 | Qwen/Qwen2.5-0.5B-Instruct | lora / closed-label likelihood | 4 | 8 / 0 | 42 | 200 | 0.8950 | 0.8944 | [0.8488, 0.9348] | 0 | 100% |
| sst2 | linear_svc | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.7950 | 0.7949 | [0.7349, 0.8451] | 0 | 0% |
| sst2 | linear_svc | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.5000 | 0.4869 | [0.4172, 0.5528] | 0 | 0% |
| sst2 | logistic_regression | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.4950 | 0.4826 | [0.4143, 0.5469] | 0 | 100% |
| sst2 | logistic_regression | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.7950 | 0.7949 | [0.7348, 0.8499] | 0 | 100% |
| sst2 | majority | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.5100 | 0.3377 | [0.3377, 0.3377] | 0 | 100% |
| sst2 | majority | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.4900 | 0.3289 | [0.3289, 0.3289] | 0 | 100% |
| sst2 | multinomial_nb | classical / classical native | full prepared train | 10000 / 1000 | 42 | 200 | 0.8450 | 0.8443 | [0.7886, 0.8896] | 0 | 100% |
| sst2 | multinomial_nb | classical / classical native | 4 | 8 / 0 | 42 | 200 | 0.5400 | 0.5388 | [0.4674, 0.6021] | 0 | 100% |
| trec | linear_svc | classical / classical native | full prepared train | 4886 / 545 | 42 | 200 | 0.9000 | 0.8980 | [0.8177, 0.9462] | 0 | 0% |
| trec | linear_svc | classical / classical native | 4 | 24 / 0 | 42 | 200 | 0.4950 | 0.4757 | [0.4061, 0.5403] | 0 | 0% |
| trec | logistic_regression | classical / classical native | 4 | 24 / 0 | 42 | 200 | 0.5000 | 0.4857 | [0.4173, 0.5552] | 0 | 100% |
| trec | logistic_regression | classical / classical native | full prepared train | 4886 / 545 | 42 | 200 | 0.8700 | 0.8679 | [0.7868, 0.9192] | 0 | 100% |
| trec | majority | classical / classical native | 4 | 24 / 0 | 42 | 200 | 0.0200 | 0.0065 | [0.0065, 0.0065] | 0 | 100% |
| trec | majority | classical / classical native | full prepared train | 4886 / 545 | 42 | 200 | 0.1900 | 0.0532 | [0.0532, 0.0532] | 0 | 100% |
| trec | multinomial_nb | classical / classical native | full prepared train | 4886 / 545 | 42 | 200 | 0.7750 | 0.7914 | [0.7052, 0.8476] | 0 | 100% |
| trec | multinomial_nb | classical / classical native | 4 | 24 / 0 | 42 | 200 | 0.5000 | 0.4857 | [0.4194, 0.5466] | 0 | 100% |

Probability metrics are computed only where actual class probabilities are available. Do not compare their quality without checking coverage and probability semantics.

Classical prediction timings are amortized batch timings; hosted latency measures sequential end-to-end requests. They are not direct serving-speed comparisons.

Incomplete runs: 1.
