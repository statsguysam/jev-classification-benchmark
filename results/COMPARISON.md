# Combined SST-2 / TREC pilot

**28 completed model/reference rows** are summarized below. 0 planned or unfinished rows have no score. Only completed imported run artifacts supply numbers; UI observations and partial predictions are not scored.

This is a **200-test-row, one-selection-seed (42) pilot per dataset**, using the shared 2,000-character prefix. Zero-shot uses no new task examples. Four-per-class prompting, adaptation and the fixed Naive Bayes reference use eight labeled examples for SST-2 or 24 for TREC, with no development labels. Pretraining data/compute are not matched.

The 0.5B adapter uses ordinary LoRA; the 4B adapter uses QLoRA when its recorded metadata confirms four-bit training. Both use normal configured precision for inference. Hosted LoRA, including Jev LoRA, is unavailable. Jev runs use the exact requested OpenRouter model `typesafe/jev-1.13` with native Choice probabilities; absent or incomplete runs remain pending and are never assigned a zero score.

## Completed runs by exact manifest group

Different complete manifest hashes are kept in separate groups. Membership alone is not proof that two environments prepared equivalent data. Cross-environment equivalence requires the separately linked audit of ordered IDs, labels, text/content hashes and training provenance; this summarizer does not infer paired contrasts across hashes.

### SST2 · sst2-M1

| Model | Method | Train / dev labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Errors | Artifact |
|---|---|---:|---|---|---:|---|
| TF-IDF + Multinomial NB | classical 4/class | 8 / 0 | 0.5400 [0.4700, 0.6050] | 0.5388 [0.4674, 0.6021] | 0 | [run](pilot/sst2__multinomial_nb__a094038879f9/run.json) |
| Qwen2.5 0.5B | zero-shot | 0 / 0 | 0.5000 [0.4900, 0.5150] | 0.3503 [0.3289, 0.3812] | 0 | [run](pilot/sst2__Qwen2.5-0.5B-Instruct__594a827ceab2/run.json) |
| Qwen2.5 0.5B | few-shot 4/class | 8 / 0 | 0.8500 [0.8000, 0.8950] | 0.8499 [0.8000, 0.8950] | 0 | [run](pilot/sst2__Qwen2.5-0.5B-Instruct__9800b4c889cc/run.json) |
| Qwen2.5 0.5B | LoRA 4/class | 8 / 0 | 0.8950 [0.8500, 0.9350] | 0.8944 [0.8488, 0.9348] | 0 | [run](pilot/sst2__Qwen2.5-0.5B-Instruct__c29d9f2be13f/run.json) |
| GPT-5.6 Luna | zero-shot | 0 / 0 | 0.9250 [0.8850, 0.9600] | 0.9249 [0.8845, 0.9600] | 0 | [run](hosted/sst2__gpt-5.6-luna__31e207fb95d3/run.json) |
| GPT-5.6 Luna | few-shot 4/class | 8 / 0 | 0.9550 [0.9250, 0.9800] | 0.9550 [0.9248, 0.9800] | 0 | [run](hosted/sst2__gpt-5.6-luna__af6e2721cfa0/run.json) |
| GPT-6 Astra | zero-shot | 0 / 0 | 0.9700 [0.9450, 0.9900] | 0.9700 [0.9450, 0.9900] | 0 | [run](hosted/sst2__gpt-6-astra__9279b61f6e7a/run.json) |
| GPT-6 Astra | few-shot 4/class | 8 / 0 | 0.9750 [0.9500, 0.9950] | 0.9750 [0.9500, 0.9950] | 0 | [run](hosted/sst2__gpt-6-astra__e77320ccd3cf/run.json) |
| Jev 1.13 (OpenRouter) | zero-shot | 0 / 0 | 0.9350 [0.9000, 0.9650] | 0.9395 [0.9040, 0.9698] | 2 | [run](jev/sst2__jev-1.13__9d5a851767fc/run.json) |
| Jev 1.13 (OpenRouter) | few-shot 4/class | 8 / 0 | 0.9650 [0.9350, 0.9900] | 0.9650 [0.9350, 0.9900] | 0 | [run](jev/sst2__jev-1.13__a33c2b36ded6/run.json) |

### SST2 · sst2-M2

| Model | Method | Train / dev labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Errors | Artifact |
|---|---|---:|---|---|---:|---|
| TF-IDF + Multinomial NB | classical 4/class | 8 / 0 | 0.5400 [0.4700, 0.6050] | 0.5388 [0.4674, 0.6021] | 0 | [run](colab/sst2__multinomial_nb__a0cf6a1ac8df/run.json) |
| Qwen3 4B | zero-shot | 0 / 0 | 0.9150 [0.8700, 0.9500] | 0.9147 [0.8695, 0.9500] | 0 | [run](colab/sst2__Qwen3-4B-Instruct-2507__0e4acd9aac6a/run.json) |
| Qwen3 4B | few-shot 4/class | 8 / 0 | 0.9550 [0.9200, 0.9800] | 0.9550 [0.9200, 0.9800] | 0 | [run](colab/sst2__Qwen3-4B-Instruct-2507__5823bb43da99/run.json) |
| Qwen3 4B | QLoRA 4/class | 8 / 0 | 0.9350 [0.8950, 0.9650] | 0.9350 [0.8950, 0.9650] | 0 | [run](colab/sst2__Qwen3-4B-Instruct-2507__22d330c63925/run.json) |

### TREC · trec-M1

| Model | Method | Train / dev labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Errors | Artifact |
|---|---|---:|---|---|---:|---|
| TF-IDF + Multinomial NB | classical 4/class | 24 / 0 | 0.5000 [0.4350, 0.5650] | 0.4857 [0.4194, 0.5466] | 0 | [run](colab/trec__multinomial_nb__9794e79f5e77/run.json) |
| Qwen3 4B | zero-shot | 0 / 0 | 0.5250 [0.4850, 0.5650] | 0.4820 [0.3756, 0.5552] | 0 | [run](colab/trec__Qwen3-4B-Instruct-2507__2f1ff41bd14e/run.json) |
| Qwen3 4B | few-shot 4/class | 24 / 0 | 0.8300 [0.7850, 0.8750] | 0.8518 [0.8089, 0.8913] | 0 | [run](colab/trec__Qwen3-4B-Instruct-2507__c0b1cc9b3d0f/run.json) |
| Qwen3 4B | QLoRA 4/class | 24 / 0 | 0.8000 [0.7600, 0.8450] | 0.7510 [0.6728, 0.8171] | 0 | [run](colab/trec__Qwen3-4B-Instruct-2507__70bf91598825/run.json) |

### TREC · trec-M2

| Model | Method | Train / dev labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Errors | Artifact |
|---|---|---:|---|---|---:|---|
| TF-IDF + Multinomial NB | classical 4/class | 24 / 0 | 0.5000 [0.4350, 0.5650] | 0.4857 [0.4194, 0.5466] | 0 | [run](pilot/trec__multinomial_nb__db3012a1e2fb/run.json) |
| Qwen2.5 0.5B | zero-shot | 0 / 0 | 0.1550 [0.1250, 0.1800] | 0.0817 [0.0677, 0.0954] | 0 | [run](pilot/trec__Qwen2.5-0.5B-Instruct__67b435872393/run.json) |
| Qwen2.5 0.5B | few-shot 4/class | 24 / 0 | 0.1750 [0.1400, 0.2150] | 0.0881 [0.0638, 0.1153] | 0 | [run](pilot/trec__Qwen2.5-0.5B-Instruct__5d00e79e1d60/run.json) |
| Qwen2.5 0.5B | LoRA 4/class | 24 / 0 | 0.0400 [0.0250, 0.0600] | 0.0445 [0.0158, 0.0799] | 0 | [run](pilot/trec__Qwen2.5-0.5B-Instruct__23d9b42b5b4e/run.json) |
| GPT-5.6 Luna | zero-shot | 0 / 0 | 0.5250 [0.4700, 0.5750] | 0.5592 [0.4933, 0.6171] | 0 | [run](hosted/trec__gpt-5.6-luna__0434e88259a6/run.json) |
| GPT-5.6 Luna | few-shot 4/class | 24 / 0 | 0.8500 [0.8049, 0.8950] | 0.8479 [0.7933, 0.9013] | 0 | [run](hosted/trec__gpt-5.6-luna__a183fc22ddc8/run.json) |
| GPT-6 Astra | zero-shot | 0 / 0 | 0.9700 [0.9450, 0.9900] | 0.9580 [0.9173, 0.9915] | 0 | [run](hosted/trec__gpt-6-astra__4788a3f7e01d/run.json) |
| GPT-6 Astra | few-shot 4/class | 24 / 0 | 0.9700 [0.9450, 0.9900] | 0.9566 [0.9167, 0.9908] | 0 | [run](hosted/trec__gpt-6-astra__f4c124cbaa8e/run.json) |
| Jev 1.13 (OpenRouter) | zero-shot | 0 / 0 | 0.3350 [0.2900, 0.3850] | 0.3622 [0.3065, 0.4174] | 3 | [run](jev/trec__jev-1.13__ce2f8d23bf30/run.json) |
| Jev 1.13 (OpenRouter) | few-shot 4/class | 24 / 0 | 0.8550 [0.8100, 0.9000] | 0.8740 [0.8288, 0.9191] | 0 | [run](jev/trec__jev-1.13__bc52fb6a4b0f/run.json) |

## Pending and unavailable conditions

| Dataset | Model | Method | State | Existing artifact |
|---|---|---|---|---|
| — | — | — | No pending conditions in this inventory | — |

## Larger supervised reference: extra training and development labels

The observed test-best classical result is included only as a **post-hoc descriptive reference**, not a prespecified winner. Its ordinary bootstrap interval does not adjust for choosing the best test score. The highest-validation-score estimator is also shown when it differs. Neither is an equal-label comparison with the k=4 arms. Candidate families are logistic regression, linear SVM and Multinomial NB; each uses its recorded three-candidate development search.

| Dataset / group | Model | Reference selection | Train / dev labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Artifact |
|---|---|---|---:|---|---|---|
| sst2-M1 | TF-IDF + Multinomial NB | observed test-best; post-hoc descriptive | 10000 / 1000 | 0.8450 [0.7900, 0.8900] | 0.8443 [0.7886, 0.8896] | [run](pilot/sst2__multinomial_nb__36d0a632e5d2/run.json) |
| sst2-M1 | TF-IDF + linear SVM | highest recorded validation macro-F1 | 10000 / 1000 | 0.7950 [0.7350, 0.8451] | 0.7949 [0.7349, 0.8451] | [run](pilot/sst2__linear_svc__2f0742f75b09/run.json) |
| trec-M2 | TF-IDF + linear SVM | observed test-best; post-hoc descriptive; also highest validation macro-F1 | 4886 / 545 | 0.9000 [0.8550, 0.9350] | 0.8980 [0.8177, 0.9462] | [run](pilot/trec__linear_svc__6829a061a4ce/run.json) |

## Recorded latency and output protocol

Seconds below use each run's own timing convention. Hosted end-to-end requests, local sequential candidate scoring and amortized classical batch timings are not equal-hardware throughput measurements. Model loading is excluded from per-row local timing; training time and all underlying fields are retained in the CSV/run artifacts.

| Dataset | Model / method | Output protocol | Timing basis | p50 seconds | p95 seconds |
|---|---|---|---|---:|---:|
| sst2 | TF-IDF + Multinomial NB / classical 4/class | classical native | amortized batch per row | 0.000138 | 0.000138 |
| sst2 | TF-IDF + Multinomial NB / full prepared | classical native | amortized batch per row | 0.000083 | 0.000083 |
| sst2 | TF-IDF + Multinomial NB / classical 4/class | classical native | amortized batch per row | 0.000053 | 0.000053 |
| sst2 | Qwen2.5 0.5B / zero-shot | restricted-label likelihood | local sequential class scoring | 0.599204 | 1.755371 |
| sst2 | Qwen2.5 0.5B / few-shot 4/class | restricted-label likelihood | local sequential class scoring | 0.974741 | 3.698362 |
| sst2 | Qwen2.5 0.5B / LoRA 4/class | restricted-label likelihood | local sequential class scoring | 0.657982 | 1.029594 |
| sst2 | Qwen3 4B / zero-shot | restricted-label likelihood | local sequential class scoring | 0.141004 | 0.177170 |
| sst2 | Qwen3 4B / few-shot 4/class | restricted-label likelihood | local sequential class scoring | 0.510362 | 0.615791 |
| sst2 | Qwen3 4B / QLoRA 4/class | restricted-label likelihood | local sequential class scoring | 0.215243 | 0.291680 |
| sst2 | GPT-5.6 Luna / zero-shot | generated label | hosted end-to-end request | 0.977493 | 1.681450 |
| sst2 | GPT-5.6 Luna / few-shot 4/class | generated label | hosted end-to-end request | 0.996753 | 1.693659 |
| sst2 | GPT-6 Astra / zero-shot | generated label | hosted end-to-end request | 1.717712 | 3.041394 |
| sst2 | GPT-6 Astra / few-shot 4/class | generated label | hosted end-to-end request | 1.688494 | 2.855524 |
| sst2 | Jev 1.13 (OpenRouter) / zero-shot | native Choice | hosted end-to-end request | 0.613026 | 0.827266 |
| sst2 | Jev 1.13 (OpenRouter) / few-shot 4/class | native Choice | hosted end-to-end request | 0.614601 | 0.750080 |
| sst2 | TF-IDF + linear SVM / full prepared | classical native | amortized batch per row | 0.000078 | 0.000078 |
| trec | TF-IDF + Multinomial NB / classical 4/class | classical native | amortized batch per row | 0.000061 | 0.000061 |
| trec | TF-IDF + Multinomial NB / classical 4/class | classical native | amortized batch per row | 0.000200 | 0.000200 |
| trec | Qwen2.5 0.5B / zero-shot | restricted-label likelihood | local sequential class scoring | 1.364238 | 1.650557 |
| trec | Qwen2.5 0.5B / few-shot 4/class | restricted-label likelihood | local sequential class scoring | 2.959726 | 6.270093 |
| trec | Qwen2.5 0.5B / LoRA 4/class | restricted-label likelihood | local sequential class scoring | 0.544274 | 0.573042 |
| trec | Qwen3 4B / zero-shot | restricted-label likelihood | local sequential class scoring | 0.584771 | 0.628192 |
| trec | Qwen3 4B / few-shot 4/class | restricted-label likelihood | local sequential class scoring | 4.183219 | 4.269964 |
| trec | Qwen3 4B / QLoRA 4/class | restricted-label likelihood | local sequential class scoring | 0.687900 | 0.844441 |
| trec | GPT-5.6 Luna / zero-shot | generated label | hosted end-to-end request | 0.980717 | 1.418755 |
| trec | GPT-5.6 Luna / few-shot 4/class | generated label | hosted end-to-end request | 0.997648 | 1.545682 |
| trec | GPT-6 Astra / zero-shot | generated label | hosted end-to-end request | 1.703971 | 3.552805 |
| trec | GPT-6 Astra / few-shot 4/class | generated label | hosted end-to-end request | 1.663998 | 2.997414 |
| trec | Jev 1.13 (OpenRouter) / zero-shot | native Choice | hosted end-to-end request | 0.599320 | 0.741663 |
| trec | Jev 1.13 (OpenRouter) / few-shot 4/class | native Choice | hosted end-to-end request | 0.601250 | 0.765208 |
| trec | TF-IDF + linear SVM / full prepared | classical native | amortized batch per row | 0.000200 | 0.000200 |

## Provenance and separate audit evidence

* `sst2-M1`: complete prepared-manifest SHA-256 `8c12945f6b673f72452096793b4777a94f2455ea55c51f4dc8c9dba9bb8a8201`.
* `sst2-M2`: complete prepared-manifest SHA-256 `c608dce4ff40a3a5022e447c05bdf92a922f4cb102adb4c3130876f4b69be86f`.
* `trec-M1`: complete prepared-manifest SHA-256 `82ffabbb8da285a63645b2d21dc15f8cbba5d150c2d30117a286f937f3e185fe`.
* `trec-M2`: complete prepared-manifest SHA-256 `d44f342fb000a58a037309a667697f2cf9c3720d52437298b5c50cbc239e333a`.
* [Colab import audit](COLAB_AUDIT.json): recorded pass `True`, 14 imported runs verified, 8 classical replications checked. Original manifest hashes remain distinct; see the audit for content/training identity evidence and scope.
* [Separately computed paired contrasts and content audits](comparisons/combined/index.md). These are exploratory contrasts with unadjusted intervals; review each named pair and audit before interpreting its difference.

Displayed intervals come from recorded stratified test-item bootstrap results (resamples present: 1000). They condition on the fixed train/test split and exclude training-seed, prompt-selection and public-pretraining contamination uncertainty. Individual intervals are not paired difference tests. No global model ranking or hallucination-free claim follows.

The fixed small-model findings are retained, including TREC LoRA collapse. SST-2 LoRA-minus-few-shot uncertainty and TREC pairing diagnostics are documented in [NEURAL_PILOT.md](NEURAL_PILOT.md). Probability metrics from native classifiers, Jev and restricted-label likelihoods have different semantics; hosted label generation does not supply a complete class distribution.

Table model names identify the requested model. The CSV retains returned model identifiers and resolved open-weight revisions when recorded; a missing returned snapshot does not establish immutability. Local numeric-ID-plus-EOS likelihood scoring and hosted generated-label parsing are different protocols. Their reported accuracies describe those exact configurations, not each model's best possible result.

Token usage in the CSV is reported usage with its coverage, not provider billing. Known totals do not make missing usage zero. Budget reservations and conservative settlement also differ from an actual invoice; see [BUDGET.md](../docs/BUDGET.md). No cost ranking is inferred here.

Rebuild with `python scripts/summarize_combined_pilot.py`. The CSV includes pending rows with blank score fields, full source paths, exact manifest/test hashes, label budgets, output protocols, latency definitions and selection notes. It does not contain corpus text or credentials.

## Reproducible comparison figure

![Measured model/method macro-F1 and conditional intervals](figures/combined-pilot.png)

[SVG](figures/combined-pilot.svg) · [Plotted values and cross-environment audit evidence](figures/combined-pilot.json). Rebuild with `python scripts/plot_combined_pilot.py` in an environment with Matplotlib.
