# Astra review audit · completed first attempts · 23 September 2026

All eight Astra source→Jev conditions have complete, aligned original evidence. Accepted-correct counts decrease in all eight: **8 decreases, 0 increases, 0 ties**. These conditions reuse **four test sets** at zero and four examples per class; they are not eight independent replications, and this direction count is not a significance test.

Each score retains the entire original test denominator. A rejected response is an operational/protocol failure, not an established wrong class decision. Separately recorded retries are excluded.

| Dataset | Examples/class | Source correct | Review correct | Corrections | Correct → accepted wrong label | Correct → failed review | Net correct change | All review failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Breast Cancer | 0 | 113/114 (99.1%) | 109/114 (95.6%) | 0 | 4 | 0 | -4 | 0 |
| Breast Cancer | 4 | 112/114 (98.2%) | 107/114 (93.9%) | 0 | 4 | 1 | -5 | 1 |
| Wine | 0 | 36/36 (100.0%) | 16/36 (44.4%) | 0 | 20 | 0 | -20 | 0 |
| Wine | 4 | 35/36 (97.2%) | 34/36 (94.4%) | 0 | 1 | 0 | -1 | 1 |
| SST-2 | 0 | 194/200 (97.0%) | 187/200 (93.5%) | 2 | 8 | 1 | -7 | 1 |
| SST-2 | 4 | 195/200 (97.5%) | 193/200 (96.5%) | 1 | 3 | 0 | -2 | 0 |
| TREC | 0 | 194/200 (97.0%) | 84/200 (42.0%) | 0 | 100 | 10 | -110 | 10 |
| TREC | 4 | 194/200 (97.0%) | 173/200 (86.5%) | 3 | 24 | 0 | -21 | 0 |

Corrections and harm categories reconcile exactly with the net change in every row. “All review failures” includes failures on already-wrong source cases and upstream skips, so it need not equal “correct → failed review.” Wine four-shot has one source failure and an inherited no-call skip; all other listed source conditions have zero failures.

## Descriptive response-validity bound

Hold every accepted review decision fixed and grant every failed pipeline row a correct result. If C is accepted-correct review count, F is all review-pipeline failures and N is the full test set, (C + F) / N is an optimistic upper bound under that assignment. This is a completed-sample sensitivity bound, not a population confidence interval, estimate of retry performance, or change to the frozen scoring policy.

| Dataset | Examples/class | Accepted correct + all failures | Optimistic review bound | Source accepted accuracy | Source count above bound |
|---|---:|---:|---:|---:|---:|
| Breast Cancer | 0 | 109 + 0 = 109/114 | 95.6% | 99.1% | 4 |
| Breast Cancer | 4 | 107 + 1 = 108/114 | 94.7% | 98.2% | 4 |
| Wine | 0 | 16 + 0 = 16/36 | 44.4% | 100.0% | 20 |
| Wine | 4 | 34 + 1 = 35/36 | 97.2% | 97.2% | 0 |
| SST-2 | 0 | 187 + 1 = 188/200 | 94.0% | 97.0% | 6 |
| SST-2 | 4 | 193 + 0 = 193/200 | 96.5% | 97.5% | 2 |
| TREC | 0 | 84 + 10 = 94/200 | 47.0% | 97.0% | 100 |
| TREC | 4 | 173 + 0 = 173/200 | 86.5% | 97.0% | 21 |

The bound remains below the source in seven conditions and ties it in Wine four-shot. Wine four-shot is an especially generous hypothetical assignment: its failed row was never sent to Jev, so there is no rejected reviewer class to reconstruct. Its other observed wrong-label override remains present.

TREC zero-shot has 84 accepted-correct decisions and ten probability-validation failures. Even making all ten failed rows correct gives 94/200 (47.0%), below the source’s 194/200 (97.0%) by 100 decisions, or 50 percentage points. There are independently 100 source-correct → accepted-wrong review decisions. The validation failures therefore cannot account for the full drop.

TREC four-shot is now complete at 173/200 (86.5%) with zero review failures; its difference from the source is entirely attributable to the recorded accepted class decisions, without invoking missing class outputs.

## Exact failure types

- Breast Cancer / 4 examples per class: 1 × `http_error: status=520; no automatic retry`.
- Wine / 4 examples per class: 1 × `source_proposal_failed: Jev review not called`.
- SST-2 / 0 examples per class: 1 × `http_error: status=520; no automatic retry`.
- TREC / 0 examples per class: 10 × `invalid_output: probabilities do not sum to one`.

The probability sum-to-one validator uses absolute tolerance 1e-6 and relative tolerance zero. On rejection the saved Prediction discards the returned class and vector. These files cannot establish the returned class correctness, discrepancy magnitude, rounding, or another cause of that validation error. Accepted wrong labels and failures are deliberately separated above.

## Scope and verification

- Read only the eight completed Astra source/review pairs. Matched ordered test IDs, truth labels, class mapping, source-run linkage and full row counts; recomputed accepted-correct and error counts and checked them against each completed run’s saved metrics.
- Verified transition arithmetic independently from raw predictions. Re-read all 48 scoped run/prediction/test-manifest files after collection to confirm their bytes had not changed. No whole-repository collector, global budget audit, API call, frozen-file edit or retry overlay was used.
- Original source and reviewer prompts differ; this is not the matched no-proposal experiment and does not identify an internal anchoring mechanism or isolate bounded decoding. Fixed prompts, one split, repeated cases, different new-label budgets and possible pretraining exposure limit generalization.
- Direction counts do not establish statistical significance or model-wide superiority. Response-validity bounds leave accepted outputs fixed and cannot predict how later calls will behave. Separate recovery reports must preserve these originals.

## Source and review records

| Dataset | Examples/class | Source run | Review run |
|---|---:|---|---|
| Breast Cancer | 0 | `results/tabular/hosted/breast_cancer__gpt-6-astra__c2361622445b/run.json` | `results/numeric_expansion/review/breast_cancer__astra__k0__jev-review/run.json` |
| Breast Cancer | 4 | `results/tabular/hosted/breast_cancer__gpt-6-astra__6be368014170/run.json` | `results/numeric_decisions/review/breast_cancer__astra__jev-review/run.json` |
| Wine | 0 | `results/tabular/hosted/wine__gpt-6-astra__bb75b85e6193/run.json` | `results/numeric_expansion/review/wine__astra__k0__jev-review/run.json` |
| Wine | 4 | `results/tabular/hosted/wine__gpt-6-astra__d5c2863bd80f/run.json` | `results/numeric_decisions/review/wine__astra__jev-review/run.json` |
| SST-2 | 0 | `results/hosted/sst2__gpt-6-astra__9279b61f6e7a/run.json` | `results/text_extension/review/sst2__astra__k0__jev-review/run.json` |
| SST-2 | 4 | `results/hosted/sst2__gpt-6-astra__e77320ccd3cf/run.json` | `results/text_extension/review/sst2__astra__k4__jev-review/run.json` |
| TREC | 0 | `results/hosted/trec__gpt-6-astra__4788a3f7e01d/run.json` | `results/text_extension/review/trec__astra__k0__jev-review/run.json` |
| TREC | 4 | `results/hosted/trec__gpt-6-astra__f4c124cbaa8e/run.json` | `results/text_extension/review/trec__astra__k4__jev-review/run.json` |

## Scoped artifact SHA-256 pins

Paths are relative to the repository root; these pins identify the original first-attempt evidence, not recovery attempts.

- `results/hosted/sst2__gpt-6-astra__9279b61f6e7a/predictions.jsonl`: `bcfdc54dd9671520e44f48eaa1198066206ce2f462ea3375f6706f2e21f796d4`
- `results/hosted/sst2__gpt-6-astra__9279b61f6e7a/run.json`: `5d5e5188b49fbe7ba354d6e833f04dedafe585f04510f9477bc2ef694323ee91`
- `results/hosted/sst2__gpt-6-astra__9279b61f6e7a/test_manifest.json`: `9fcb2492f1eb55dca24bfb3524cd8f1578101c6f5732c959eee1eacc67f6d898`
- `results/hosted/sst2__gpt-6-astra__e77320ccd3cf/predictions.jsonl`: `e2d7d4ba0f39220fe2fdf6ab107bf5364aafae961b3440386d247c674d55357d`
- `results/hosted/sst2__gpt-6-astra__e77320ccd3cf/run.json`: `86c3428b816b6dcd9e366c4eb2e8c804829ef0fc331e7e90509901b90b357f31`
- `results/hosted/sst2__gpt-6-astra__e77320ccd3cf/test_manifest.json`: `9fcb2492f1eb55dca24bfb3524cd8f1578101c6f5732c959eee1eacc67f6d898`
- `results/hosted/trec__gpt-6-astra__4788a3f7e01d/predictions.jsonl`: `190e7aa8751b370215d952f52714a4f7dd34c7b8e9f7b60fbff7a118f99d6615`
- `results/hosted/trec__gpt-6-astra__4788a3f7e01d/run.json`: `66ee96053a56f3604d580dc29f2138349dbf89b2b1ccc2b392d262d37844d5ed`
- `results/hosted/trec__gpt-6-astra__4788a3f7e01d/test_manifest.json`: `462a00705aed5a3f6e2846483cac278e870c58a74f248c3fd265820018b33a89`
- `results/hosted/trec__gpt-6-astra__f4c124cbaa8e/predictions.jsonl`: `6690440f7d1f8d02baab7f59270677c653a424c83f21d37e15adeb0bf25c076f`
- `results/hosted/trec__gpt-6-astra__f4c124cbaa8e/run.json`: `5542881d56ffeb439c275c82ecd8d3b8166c33143e6d9e6a4fb212fb9973f715`
- `results/hosted/trec__gpt-6-astra__f4c124cbaa8e/test_manifest.json`: `462a00705aed5a3f6e2846483cac278e870c58a74f248c3fd265820018b33a89`
- `results/numeric_decisions/review/breast_cancer__astra__jev-review/predictions.jsonl`: `ebdb5f9455c2401b48d79c58e2539747acf76f3dd53d81a81ec7f3575abef2b1`
- `results/numeric_decisions/review/breast_cancer__astra__jev-review/run.json`: `4c9f63f0de765395ce0c3914e5c06860651f1116091a7c1cb5a076778ae223a7`
- `results/numeric_decisions/review/breast_cancer__astra__jev-review/test_manifest.json`: `d3445c1ee998cbebc299755c9c4284295416ad8a1f4165cfad80ddf85b987685`
- `results/numeric_decisions/review/wine__astra__jev-review/predictions.jsonl`: `90427f92c0442400abb54c60a27445c46421f0492d88dd7eba78fd110a4fce3c`
- `results/numeric_decisions/review/wine__astra__jev-review/run.json`: `e85950e389894748e7abdda557df992147ee6b5e533201c31ac3e1c9cc5ab814`
- `results/numeric_decisions/review/wine__astra__jev-review/test_manifest.json`: `edbfa3c757d7142e13a35ba40b53c590587e3e591508d70161dd433f881a5c76`
- `results/numeric_expansion/review/breast_cancer__astra__k0__jev-review/predictions.jsonl`: `1a4bdff8b852bd963311ab58c913ffe3bc4e5d7dd5306e16a0ff3db803931978`
- `results/numeric_expansion/review/breast_cancer__astra__k0__jev-review/run.json`: `a0d78a56d1da5d00121f6ffb4ed0c69710bf157be309c294658afe1eae53d362`
- `results/numeric_expansion/review/breast_cancer__astra__k0__jev-review/test_manifest.json`: `d3445c1ee998cbebc299755c9c4284295416ad8a1f4165cfad80ddf85b987685`
- `results/numeric_expansion/review/wine__astra__k0__jev-review/predictions.jsonl`: `663e1deadc4ad1fd595fcfd505187b0ec51628d3c87645404a88585acd651dfc`
- `results/numeric_expansion/review/wine__astra__k0__jev-review/run.json`: `215263d9557d8063e3a03a5698943e21a1736049ea6d66a700fa9de7f734c32c`
- `results/numeric_expansion/review/wine__astra__k0__jev-review/test_manifest.json`: `edbfa3c757d7142e13a35ba40b53c590587e3e591508d70161dd433f881a5c76`
- `results/tabular/hosted/breast_cancer__gpt-6-astra__6be368014170/predictions.jsonl`: `872826ba524654e0e4bdabccca79dc69e9403e884bf5ac9a29407d600705b177`
- `results/tabular/hosted/breast_cancer__gpt-6-astra__6be368014170/run.json`: `bb1c82612d9c4787d057d78be96ea225ebe592738ee8090a27b08d08963a1176`
- `results/tabular/hosted/breast_cancer__gpt-6-astra__6be368014170/test_manifest.json`: `d3445c1ee998cbebc299755c9c4284295416ad8a1f4165cfad80ddf85b987685`
- `results/tabular/hosted/breast_cancer__gpt-6-astra__c2361622445b/predictions.jsonl`: `2923aaf59fe68f0b1827db265e4580e018174a830d132b0a5cd60efd7e1d08ad`
- `results/tabular/hosted/breast_cancer__gpt-6-astra__c2361622445b/run.json`: `b65c1ffb1f8ff8472acc4ede9b1202d6426fb0f99783e491f683247032390a9e`
- `results/tabular/hosted/breast_cancer__gpt-6-astra__c2361622445b/test_manifest.json`: `d3445c1ee998cbebc299755c9c4284295416ad8a1f4165cfad80ddf85b987685`
- `results/tabular/hosted/wine__gpt-6-astra__bb75b85e6193/predictions.jsonl`: `c27d4c883920ed3b763786fbc88ba4bcbaf08ea84040e576ca1d1f3311340be0`
- `results/tabular/hosted/wine__gpt-6-astra__bb75b85e6193/run.json`: `fb9a2322140095f404184003409537acb2fed918166e456e9b0772885fc46f92`
- `results/tabular/hosted/wine__gpt-6-astra__bb75b85e6193/test_manifest.json`: `edbfa3c757d7142e13a35ba40b53c590587e3e591508d70161dd433f881a5c76`
- `results/tabular/hosted/wine__gpt-6-astra__d5c2863bd80f/predictions.jsonl`: `dbe5ff17012aec89bf5588be1493217451e25607eb725b6a422fedda986156ab`
- `results/tabular/hosted/wine__gpt-6-astra__d5c2863bd80f/run.json`: `c3162e3b5defb51b2d0fe24d9d05a5e131c67f2ff7ce1f0dfa02077256999622`
- `results/tabular/hosted/wine__gpt-6-astra__d5c2863bd80f/test_manifest.json`: `edbfa3c757d7142e13a35ba40b53c590587e3e591508d70161dd433f881a5c76`
- `results/text_extension/review/sst2__astra__k0__jev-review/predictions.jsonl`: `6ecfcba74ad71f058f965a1cd47e6e10529bcabe61b0a90fdbebb5eec143f1d3`
- `results/text_extension/review/sst2__astra__k0__jev-review/run.json`: `2010cd1d33aaa489992343eef6dd658a497a80036c67edaa63167e0998e4c4d5`
- `results/text_extension/review/sst2__astra__k0__jev-review/test_manifest.json`: `9fcb2492f1eb55dca24bfb3524cd8f1578101c6f5732c959eee1eacc67f6d898`
- `results/text_extension/review/sst2__astra__k4__jev-review/predictions.jsonl`: `645f0a94aedd641f212eafec711dfcb8baf54b9ffa8a82f434eab5a58b0a01c7`
- `results/text_extension/review/sst2__astra__k4__jev-review/run.json`: `cf3b705490589a5cccdbf260bcf3c181c3b9d2357673f07cd073990808445553`
- `results/text_extension/review/sst2__astra__k4__jev-review/test_manifest.json`: `9fcb2492f1eb55dca24bfb3524cd8f1578101c6f5732c959eee1eacc67f6d898`
- `results/text_extension/review/trec__astra__k0__jev-review/predictions.jsonl`: `1f509b2dc4a51dcb0c08410678fe69ca8f5185664c2f907eca1b04d23aa13a13`
- `results/text_extension/review/trec__astra__k0__jev-review/run.json`: `8fe91713ebc44f876438f808661bdd0c31b76fb3e21a209e021433448c3e4117`
- `results/text_extension/review/trec__astra__k0__jev-review/test_manifest.json`: `462a00705aed5a3f6e2846483cac278e870c58a74f248c3fd265820018b33a89`
- `results/text_extension/review/trec__astra__k4__jev-review/predictions.jsonl`: `d256221fd373a5338f22ab2d420639fd0e90c03f5b8ebb4af64ee9e425e8631a`
- `results/text_extension/review/trec__astra__k4__jev-review/run.json`: `8e1ecb69548d3c6386112b5a1b0415ec2c3878999d589bf37e45b72c4e6e78d6`
- `results/text_extension/review/trec__astra__k4__jev-review/test_manifest.json`: `462a00705aed5a3f6e2846483cac278e870c58a74f248c3fd265820018b33a89`
