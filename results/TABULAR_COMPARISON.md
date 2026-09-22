# Serialized tabular classification benchmark

**66 of 66 planned runs are complete and audited.** Pending runs have no score. The matrix consists of 30 native classical, 18 open-model and 18 hosted-model conditions.

All methods use the same held-out rows. Four-per-class prompting, LoRA/QLoRA and matched classical models use the identical eight labels for Titanic/Breast Cancer or 12 labels for Wine. Zero-shot uses zero new task labels. The full-training classical track uses extra labels and is reported separately. No validation labels are used in this fixed-recipe extension.

Intervals below are **unstratified group percentile bootstrap intervals**: sample identical serialized-test-text groups with replacement, retaining every member. This accounts for exact repeated feature vectors, including conflicting targets, but not every possible dependency. These exploratory, unadjusted intervals condition on one fixed split, seed, prompt, serialization and fitted model; they do not measure variation across training seeds or establish general superiority.

Each interval uses 2000 replicates and seed 42. Original per-row bootstrap fields in run artifacts are preserved but are not the intervals displayed here.

## Data and preparation audit

| Dataset | Train / dev / test rows | Test feature groups | Maximum serialized characters |
|---|---:|---:|---:|
| titanic | 785 / 262 / 262 | 218 | 307 |
| breast_cancer | 341 / 114 / 114 | 114 | 929 |
| wine | 106 / 36 / 36 | 36 | 444 |

Prepared manifests, serialized/native file hashes, ordered training/test IDs and unchanged inference-core identities are checked against the local frozen bundle. Completed metrics are recomputed from raw predictions. Imported Colab runs must pass the same checks; no cross-environment hash exceptions are allowed.

## titanic: zero-shot and matched-label methods

| Model | Method | Train labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Failures | Probability coverage | Run |
|---|---|---:|---|---|---:|---:|---|
| Majority | 4/class | 8 | 0.6183 [0.5439, 0.6908] | 0.3821 [0.3523, 0.4086] | 0 | 100.0% | [artifact](tabular/classical/titanic__native_majority__05aea2d028ea/run.json) |
| Logistic regression | 4/class | 8 | 0.6412 [0.5708, 0.7152] | 0.6185 [0.5493, 0.6839] | 0 | 100.0% | [artifact](tabular/classical/titanic__native_logistic_regression__93afc6c9bc34/run.json) |
| RBF SVM | 4/class | 8 | 0.7023 [0.6330, 0.7683] | 0.6440 [0.5749, 0.7080] | 0 | 0.0% | [artifact](tabular/classical/titanic__native_rbf_svc__b05e1156e319/run.json) |
| Random forest | 4/class | 8 | 0.5802 [0.4929, 0.6705] | 0.5766 [0.4897, 0.6652] | 0 | 100.0% | [artifact](tabular/classical/titanic__native_random_forest__1e813c4a1d10/run.json) |
| Hist. gradient boosting | 4/class | 8 | 0.5649 [0.4852, 0.6471] | 0.5425 [0.4657, 0.6122] | 0 | 100.0% | [artifact](tabular/classical/titanic__native_hist_gradient_boosting__c0a4aad9177f/run.json) |
| Qwen2.5 0.5B | zero-shot | 0 | 0.6183 [0.5439, 0.6908] | 0.3821 [0.3523, 0.4086] | 0 | 100.0% | [artifact](tabular/local/titanic__Qwen2.5-0.5B-Instruct__df1e72182230/run.json) |
| Qwen2.5 0.5B | few-shot 4/class | 8 | 0.6031 [0.5284, 0.6767] | 0.3850 [0.3486, 0.4216] | 0 | 100.0% | [artifact](tabular/local/titanic__Qwen2.5-0.5B-Instruct__e4a9d0e76a35/run.json) |
| Qwen2.5 0.5B | LoRA 4/class | 8 | 0.6183 [0.5439, 0.6908] | 0.3821 [0.3523, 0.4086] | 0 | 100.0% | [artifact](tabular/local/titanic__Qwen2.5-0.5B-Instruct__aef5dfcf4960/run.json) |
| Qwen3 4B | zero-shot | 0 | 0.6450 [0.5731, 0.7213] | 0.6450 [0.5693, 0.7169] | 0 | 100.0% | [artifact](tabular/colab/titanic__Qwen3-4B-Instruct-2507__bc3feaefd6e3/run.json) |
| Qwen3 4B | few-shot 4/class | 8 | 0.5382 [0.4478, 0.6219] | 0.5307 [0.4393, 0.6136] | 0 | 100.0% | [artifact](tabular/colab/titanic__Qwen3-4B-Instruct-2507__1d84babc1c9c/run.json) |
| Qwen3 4B | QLoRA 4/class | 8 | 0.6336 [0.5572, 0.7121] | 0.6336 [0.5545, 0.7084] | 0 | 100.0% | [artifact](tabular/colab/titanic__Qwen3-4B-Instruct-2507__47dd7e7dd3cc/run.json) |
| Jev 1.13 | zero-shot | 0 | 0.7710 [0.7119, 0.8283] | 0.7643 [0.7061, 0.8196] | 0 | 100.0% | [artifact](tabular/hosted/titanic__jev-1.13__0a3eba74ea4b/run.json) |
| Jev 1.13 | few-shot 4/class | 8 | 0.7443 [0.6905, 0.7992] | 0.7358 [0.6793, 0.7881] | 1 | 99.6% | [artifact](tabular/hosted/titanic__jev-1.13__ca2172a48634/run.json) |
| GPT-5.6 Luna | zero-shot | 0 | 0.7672 [0.7066, 0.8251] | 0.7556 [0.6936, 0.8116] | 0 | 0.0% | [artifact](tabular/hosted/titanic__gpt-5.6-luna__b5122ec4e50d/run.json) |
| GPT-5.6 Luna | few-shot 4/class | 8 | 0.8130 [0.7570, 0.8631] | 0.7963 [0.7405, 0.8472] | 0 | 0.0% | [artifact](tabular/hosted/titanic__gpt-5.6-luna__5241c331369c/run.json) |
| GPT-6 Astra | zero-shot | 0 | 0.8282 [0.7731, 0.8763] | 0.8138 [0.7562, 0.8628] | 0 | 0.0% | [artifact](tabular/hosted/titanic__gpt-6-astra__3dcf7666b968/run.json) |
| GPT-6 Astra | few-shot 4/class | 8 | 0.8588 [0.8099, 0.9034] | 0.8462 [0.7966, 0.8918] | 0 | 0.0% | [artifact](tabular/hosted/titanic__gpt-6-astra__6422a6aaa735/run.json) |

## breast_cancer: zero-shot and matched-label methods

| Model | Method | Train labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Failures | Probability coverage | Run |
|---|---|---:|---|---|---:|---:|---|
| Majority | 4/class | 8 | 0.3772 [0.2895, 0.4649] | 0.2739 [0.2245, 0.3174] | 0 | 100.0% | [artifact](tabular/classical/breast_cancer__native_majority__601690bd324b/run.json) |
| Logistic regression | 4/class | 8 | 0.9474 [0.9035, 0.9825] | 0.9435 [0.8935, 0.9816] | 0 | 100.0% | [artifact](tabular/classical/breast_cancer__native_logistic_regression__55222f2fa251/run.json) |
| RBF SVM | 4/class | 8 | 0.9386 [0.8947, 0.9825] | 0.9349 [0.8820, 0.9800] | 0 | 0.0% | [artifact](tabular/classical/breast_cancer__native_rbf_svc__a26c1965de9e/run.json) |
| Random forest | 4/class | 8 | 0.9561 [0.9123, 0.9912] | 0.9521 [0.9074, 0.9903] | 0 | 100.0% | [artifact](tabular/classical/breast_cancer__native_random_forest__a42a202bf624/run.json) |
| Hist. gradient boosting | 4/class | 8 | 0.9211 [0.8684, 0.9649] | 0.9147 [0.8579, 0.9627] | 0 | 100.0% | [artifact](tabular/classical/breast_cancer__native_hist_gradient_boosting__3380de7bd318/run.json) |
| Qwen2.5 0.5B | zero-shot | 0 | 0.3772 [0.2895, 0.4649] | 0.2739 [0.2245, 0.3174] | 0 | 100.0% | [artifact](tabular/local/breast_cancer__Qwen2.5-0.5B-Instruct__dda8e08a1643/run.json) |
| Qwen2.5 0.5B | few-shot 4/class | 8 | 0.6140 [0.5263, 0.7018] | 0.3804 [0.3448, 0.4124] | 0 | 100.0% | [artifact](tabular/local/breast_cancer__Qwen2.5-0.5B-Instruct__4f3b067694b6/run.json) |
| Qwen2.5 0.5B | LoRA 4/class | 8 | 0.6228 [0.5351, 0.7105] | 0.3838 [0.3486, 0.4154] | 0 | 100.0% | [artifact](tabular/local/breast_cancer__Qwen2.5-0.5B-Instruct__cc629425dc09/run.json) |
| Qwen3 4B | zero-shot | 0 | 0.6140 [0.5263, 0.7018] | 0.4792 [0.3965, 0.5643] | 0 | 100.0% | [artifact](tabular/colab/breast_cancer__Qwen3-4B-Instruct-2507__1939b7a5ad23/run.json) |
| Qwen3 4B | few-shot 4/class | 8 | 0.8596 [0.7895, 0.9211] | 0.8581 [0.7884, 0.9183] | 0 | 100.0% | [artifact](tabular/colab/breast_cancer__Qwen3-4B-Instruct-2507__cd3baa813fe3/run.json) |
| Qwen3 4B | QLoRA 4/class | 8 | 0.3772 [0.2895, 0.4649] | 0.2739 [0.2245, 0.3174] | 0 | 100.0% | [artifact](tabular/colab/breast_cancer__Qwen3-4B-Instruct-2507__3bd9e18dc27d/run.json) |
| Jev 1.13 | zero-shot | 0 | 0.8421 [0.7719, 0.9123] | 0.8449 [0.7726, 0.9115] | 1 | 99.1% | [artifact](tabular/hosted/breast_cancer__jev-1.13__e30c52b19b9e/run.json) |
| Jev 1.13 | few-shot 4/class | 8 | 0.9298 [0.8772, 0.9737] | 0.9287 [0.8777, 0.9726] | 1 | 99.1% | [artifact](tabular/hosted/breast_cancer__jev-1.13__3d1cba817d2a/run.json) |
| GPT-5.6 Luna | zero-shot | 0 | 0.7544 [0.6754, 0.8246] | 0.7543 [0.6742, 0.8243] | 0 | 0.0% | [artifact](tabular/hosted/breast_cancer__gpt-5.6-luna__b8612044e1e5/run.json) |
| GPT-5.6 Luna | few-shot 4/class | 8 | 0.9123 [0.8596, 0.9561] | 0.9089 [0.8492, 0.9559] | 0 | 0.0% | [artifact](tabular/hosted/breast_cancer__gpt-5.6-luna__6874301c84d2/run.json) |
| GPT-6 Astra | zero-shot | 0 | 0.9912 [0.9737, 1.0000] | 0.9906 [0.9693, 1.0000] | 0 | 0.0% | [artifact](tabular/hosted/breast_cancer__gpt-6-astra__c2361622445b/run.json) |
| GPT-6 Astra | few-shot 4/class | 8 | 0.9825 [0.9561, 1.0000] | 0.9812 [0.9510, 1.0000] | 0 | 0.0% | [artifact](tabular/hosted/breast_cancer__gpt-6-astra__6be368014170/run.json) |

## wine: zero-shot and matched-label methods

| Model | Method | Train labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Failures | Probability coverage | Run |
|---|---|---:|---|---|---:|---:|---|
| Majority | 4/class | 12 | 0.3333 [0.1944, 0.5000] | 0.1667 [0.1085, 0.2222] | 0 | 100.0% | [artifact](tabular/classical/wine__native_majority__becfabe82b76/run.json) |
| Logistic regression | 4/class | 12 | 0.9722 [0.9167, 1.0000] | 0.9740 [0.9103, 1.0000] | 0 | 100.0% | [artifact](tabular/classical/wine__native_logistic_regression__c517f9ce4b63/run.json) |
| RBF SVM | 4/class | 12 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0 | 0.0% | [artifact](tabular/classical/wine__native_rbf_svc__2de4260ee5e8/run.json) |
| Random forest | 4/class | 12 | 0.9444 [0.8611, 1.0000] | 0.9475 [0.8611, 1.0000] | 0 | 100.0% | [artifact](tabular/classical/wine__native_random_forest__b446bd3c19ff/run.json) |
| Hist. gradient boosting | 4/class | 12 | 0.7778 [0.6389, 0.9167] | 0.7422 [0.5729, 0.8836] | 0 | 100.0% | [artifact](tabular/classical/wine__native_hist_gradient_boosting__2313f9e6d88a/run.json) |
| Qwen2.5 0.5B | zero-shot | 0 | 0.3333 [0.1944, 0.5000] | 0.1667 [0.1085, 0.2222] | 0 | 100.0% | [artifact](tabular/local/wine__Qwen2.5-0.5B-Instruct__6caaa1e6b4e8/run.json) |
| Qwen2.5 0.5B | few-shot 4/class | 12 | 0.3333 [0.1944, 0.5000] | 0.1667 [0.1085, 0.2222] | 0 | 100.0% | [artifact](tabular/local/wine__Qwen2.5-0.5B-Instruct__76dea05d8b42/run.json) |
| Qwen2.5 0.5B | LoRA 4/class | 12 | 0.3333 [0.1944, 0.5000] | 0.1667 [0.1085, 0.2222] | 0 | 100.0% | [artifact](tabular/local/wine__Qwen2.5-0.5B-Instruct__4091defc6e25/run.json) |
| Qwen3 4B | zero-shot | 0 | 0.3889 [0.2222, 0.5556] | 0.1867 [0.1212, 0.2381] | 0 | 100.0% | [artifact](tabular/colab/wine__Qwen3-4B-Instruct-2507__c923604a4ba1/run.json) |
| Qwen3 4B | few-shot 4/class | 12 | 0.8056 [0.6667, 0.9167] | 0.8088 [0.6654, 0.9193] | 0 | 100.0% | [artifact](tabular/colab/wine__Qwen3-4B-Instruct-2507__cea3e01ea8d0/run.json) |
| Qwen3 4B | QLoRA 4/class | 12 | 0.2778 [0.1389, 0.4167] | 0.1449 [0.0813, 0.1961] | 0 | 100.0% | [artifact](tabular/colab/wine__Qwen3-4B-Instruct-2507__55b0160bc45a/run.json) |
| Jev 1.13 | zero-shot | 0 | 0.3333 [0.1944, 0.5000] | 0.1667 [0.1085, 0.2222] | 0 | 100.0% | [artifact](tabular/hosted/wine__jev-1.13__7daebee622e6/run.json) |
| Jev 1.13 | few-shot 4/class | 12 | 0.9167 [0.8326, 1.0000] | 0.9164 [0.8077, 1.0000] | 0 | 100.0% | [artifact](tabular/hosted/wine__jev-1.13__33047fa693a8/run.json) |
| GPT-5.6 Luna | zero-shot | 0 | 0.4722 [0.3056, 0.6389] | 0.3451 [0.2091, 0.4583] | 0 | 0.0% | [artifact](tabular/hosted/wine__gpt-5.6-luna__e3ff7ad8c4ad/run.json) |
| GPT-5.6 Luna | few-shot 4/class | 12 | 0.8889 [0.7778, 0.9722] | 0.8889 [0.7762, 0.9732] | 0 | 0.0% | [artifact](tabular/hosted/wine__gpt-5.6-luna__bf0ee5396886/run.json) |
| GPT-6 Astra | zero-shot | 0 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0 | 0.0% | [artifact](tabular/hosted/wine__gpt-6-astra__bb75b85e6193/run.json) |
| GPT-6 Astra | few-shot 4/class | 12 | 0.9722 [0.9167, 1.0000] | 0.9877 [0.9524, 1.0000] | 1 | 0.0% | [artifact](tabular/hosted/wine__gpt-6-astra__d5c2863bd80f/run.json) |

## Full-training classical references: extra labels

Every listed estimator uses fixed parameters; this report does not choose a winner using test results. Each full-training model fits preprocessing only on the full training split, still without validation labels. These rows do not have an equal-label budget with the four-per-class arms.

| Dataset | Model | Train labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Failures | Probability coverage |
|---|---|---:|---|---|---:|---:|
| titanic | Majority | 785 | 0.6183 [0.5439, 0.6908] | 0.3821 [0.3523, 0.4086] | 0 | 100.0% |
| titanic | Logistic regression | 785 | 0.7901 [0.7298, 0.8437] | 0.7734 [0.7128, 0.8264] | 0 | 100.0% |
| titanic | RBF SVM | 785 | 0.7939 [0.7343, 0.8487] | 0.7727 [0.7121, 0.8274] | 0 | 0.0% |
| titanic | Random forest | 785 | 0.7863 [0.7289, 0.8410] | 0.7698 [0.7136, 0.8227] | 0 | 100.0% |
| titanic | Hist. gradient boosting | 785 | 0.7824 [0.7184, 0.8396] | 0.7619 [0.6955, 0.8186] | 0 | 100.0% |
| breast_cancer | Majority | 341 | 0.6228 [0.5351, 0.7105] | 0.3838 [0.3486, 0.4154] | 0 | 100.0% |
| breast_cancer | Logistic regression | 341 | 0.9912 [0.9737, 1.0000] | 0.9907 [0.9706, 1.0000] | 0 | 100.0% |
| breast_cancer | RBF SVM | 341 | 0.9912 [0.9737, 1.0000] | 0.9906 [0.9698, 1.0000] | 0 | 0.0% |
| breast_cancer | Random forest | 341 | 0.9825 [0.9561, 1.0000] | 0.9813 [0.9521, 1.0000] | 0 | 100.0% |
| breast_cancer | Hist. gradient boosting | 341 | 0.9825 [0.9561, 1.0000] | 0.9812 [0.9521, 1.0000] | 0 | 100.0% |
| wine | Majority | 106 | 0.3889 [0.2222, 0.5556] | 0.1867 [0.1212, 0.2381] | 0 | 100.0% |
| wine | Logistic regression | 106 | 0.9722 [0.9167, 1.0000] | 0.9718 [0.9012, 1.0000] | 0 | 100.0% |
| wine | RBF SVM | 106 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0 | 0.0% |
| wine | Random forest | 106 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0 | 100.0% |
| wine | Hist. gradient boosting | 106 | 0.9722 [0.9167, 1.0000] | 0.9718 [0.9012, 1.0000] | 0 | 100.0% |

## Paired contrasts — main matched comparison

Differences are A minus B. Both predictions receive the same resampled groups. All intervals are exploratory and unadjusted for multiple comparisons.

| Dataset | A | B | Δ accuracy [95% CI] | Δ macro-F1 [95% CI] | Groups |
|---|---|---|---|---|---:|
| titanic | Jev 1.13 few shot | GPT-6 Astra few shot | -0.1145 [-0.1648, -0.0629] | -0.1103 [-0.1623, -0.0586] | 218 |
| titanic | Jev 1.13 few shot | Qwen3 4B few shot | +0.2061 [+0.1035, +0.3234] | +0.2051 [+0.1079, +0.3116] | 218 |
| titanic | Jev 1.13 few shot | Logistic regression classical tabular | +0.1031 [+0.0264, +0.1807] | +0.1174 [+0.0418, +0.1906] | 218 |
| titanic | Jev 1.13 few shot | Random forest classical tabular | +0.1641 [+0.0644, +0.2760] | +0.1592 [+0.0653, +0.2606] | 218 |
| titanic | Qwen2.5 0.5B lora | Qwen2.5 0.5B few shot | +0.0153 [+0.0000, +0.0340] | -0.0030 [-0.0276, +0.0122] | 218 |
| titanic | Qwen3 4B lora | Qwen3 4B few shot | +0.0954 [-0.0402, +0.2474] | +0.1028 [-0.0407, +0.2505] | 218 |
| breast_cancer | Jev 1.13 few shot | GPT-6 Astra few shot | -0.0526 [-0.0965, -0.0175] | -0.0525 [-0.0975, -0.0175] | 114 |
| breast_cancer | Jev 1.13 few shot | Qwen3 4B few shot | +0.0702 [+0.0000, +0.1404] | +0.0706 [+0.0062, +0.1415] | 114 |
| breast_cancer | Jev 1.13 few shot | Logistic regression classical tabular | -0.0175 [-0.0614, +0.0263] | -0.0148 [-0.0587, +0.0296] | 114 |
| breast_cancer | Jev 1.13 few shot | Random forest classical tabular | -0.0263 [-0.0789, +0.0263] | -0.0235 [-0.0766, +0.0296] | 114 |
| breast_cancer | Qwen2.5 0.5B lora | Qwen2.5 0.5B few shot | +0.0088 [+0.0000, +0.0263] | +0.0033 [+0.0000, +0.0106] | 114 |
| breast_cancer | Qwen3 4B lora | Qwen3 4B few shot | -0.4825 [-0.5789, -0.3860] | -0.5842 [-0.6478, -0.5190] | 114 |
| wine | Jev 1.13 few shot | GPT-6 Astra few shot | -0.0556 [-0.1667, +0.0556] | -0.0713 [-0.1856, +0.0187] | 36 |
| wine | Jev 1.13 few shot | Qwen3 4B few shot | +0.1111 [+0.0000, +0.2500] | +0.1076 [-0.0113, +0.2485] | 36 |
| wine | Jev 1.13 few shot | Logistic regression classical tabular | -0.0556 [-0.1667, +0.0556] | -0.0576 [-0.1721, +0.0439] | 36 |
| wine | Jev 1.13 few shot | Random forest classical tabular | -0.0278 [-0.1389, +0.0833] | -0.0311 [-0.1558, +0.0897] | 36 |
| wine | Qwen2.5 0.5B lora | Qwen2.5 0.5B few shot | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 36 |
| wine | Qwen3 4B lora | Qwen3 4B few shot | -0.5278 [-0.6944, -0.3611] | -0.6639 [-0.7716, -0.5232] | 36 |

## Paired contrasts — descriptive: few-shot versus zero-shot; unequal labels

Differences are A minus B. Both predictions receive the same resampled groups. All intervals are exploratory and unadjusted for multiple comparisons.

| Dataset | A | B | Δ accuracy [95% CI] | Δ macro-F1 [95% CI] | Groups |
|---|---|---|---|---|---:|
| titanic | Qwen2.5 0.5B few shot | Qwen2.5 0.5B zero shot | -0.0153 [-0.0340, +0.0000] | +0.0030 [-0.0122, +0.0276] | 218 |
| titanic | Qwen3 4B few shot | Qwen3 4B zero shot | -0.1069 [-0.2598, +0.0305] | -0.1143 [-0.2629, +0.0308] | 218 |
| titanic | Jev 1.13 few shot | Jev 1.13 zero shot | -0.0267 [-0.0856, +0.0350] | -0.0285 [-0.0870, +0.0348] | 218 |
| titanic | GPT-5.6 Luna few shot | GPT-5.6 Luna zero shot | +0.0458 [+0.0070, +0.0884] | +0.0407 [-0.0017, +0.0834] | 218 |
| titanic | GPT-6 Astra few shot | GPT-6 Astra zero shot | +0.0305 [+0.0000, +0.0643] | +0.0324 [-0.0007, +0.0689] | 218 |
| breast_cancer | Qwen2.5 0.5B few shot | Qwen2.5 0.5B zero shot | +0.2368 [+0.0614, +0.4123] | +0.1065 [+0.0275, +0.1879] | 114 |
| breast_cancer | Qwen3 4B few shot | Qwen3 4B zero shot | +0.2456 [+0.1228, +0.3596] | +0.3788 [+0.2609, +0.4861] | 114 |
| breast_cancer | Jev 1.13 few shot | Jev 1.13 zero shot | +0.0877 [+0.0088, +0.1667] | +0.0838 [+0.0067, +0.1623] | 114 |
| breast_cancer | GPT-5.6 Luna few shot | GPT-5.6 Luna zero shot | +0.1579 [+0.0877, +0.2283] | +0.1546 [+0.0856, +0.2287] | 114 |
| breast_cancer | GPT-6 Astra few shot | GPT-6 Astra zero shot | -0.0088 [-0.0263, +0.0000] | -0.0095 [-0.0302, +0.0000] | 114 |
| wine | Qwen2.5 0.5B few shot | Qwen2.5 0.5B zero shot | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 36 |
| wine | Qwen3 4B few shot | Qwen3 4B zero shot | +0.4167 [+0.1944, +0.6389] | +0.6221 [+0.4667, +0.7557] | 36 |
| wine | Jev 1.13 few shot | Jev 1.13 zero shot | +0.5833 [+0.4167, +0.7500] | +0.7497 [+0.6318, +0.8440] | 36 |
| wine | GPT-5.6 Luna few shot | GPT-5.6 Luna zero shot | +0.4167 [+0.1944, +0.6389] | +0.5438 [+0.3687, +0.7172] | 36 |
| wine | GPT-6 Astra few shot | GPT-6 Astra zero shot | -0.0278 [-0.0833, +0.0000] | -0.0123 [-0.0476, +0.0000] | 36 |

## Paired contrasts — descriptive: full versus 4/class; unequal labels

Differences are A minus B. Both predictions receive the same resampled groups. All intervals are exploratory and unadjusted for multiple comparisons.

| Dataset | A | B | Δ accuracy [95% CI] | Δ macro-F1 [95% CI] | Groups |
|---|---|---|---|---|---:|
| titanic | Majority full training | Majority classical tabular | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 218 |
| titanic | Logistic regression full training | Logistic regression classical tabular | +0.1489 [+0.0723, +0.2239] | +0.1550 [+0.0756, +0.2311] | 218 |
| titanic | RBF SVM full training | RBF SVM classical tabular | +0.0916 [+0.0290, +0.1527] | +0.1287 [+0.0563, +0.1994] | 218 |
| titanic | Random forest full training | Random forest classical tabular | +0.2061 [+0.0880, +0.3299] | +0.1932 [+0.0789, +0.3101] | 218 |
| titanic | Hist. gradient boosting full training | Hist. gradient boosting classical tabular | +0.2176 [+0.1402, +0.2959] | +0.2194 [+0.1379, +0.3005] | 218 |
| breast_cancer | Majority full training | Majority classical tabular | +0.2456 [+0.0702, +0.4211] | +0.1099 [+0.0312, +0.1909] | 114 |
| breast_cancer | Logistic regression full training | Logistic regression classical tabular | +0.0439 [+0.0000, +0.0877] | +0.0473 [+0.0008, +0.0996] | 114 |
| breast_cancer | RBF SVM full training | RBF SVM classical tabular | +0.0526 [+0.0088, +0.1053] | +0.0557 [+0.0088, +0.1120] | 114 |
| breast_cancer | Random forest full training | Random forest classical tabular | +0.0263 [-0.0088, +0.0614] | +0.0292 [-0.0094, +0.0706] | 114 |
| breast_cancer | Hist. gradient boosting full training | Hist. gradient boosting classical tabular | +0.0614 [+0.0088, +0.1228] | +0.0664 [+0.0088, +0.1307] | 114 |
| wine | Majority full training | Majority classical tabular | +0.0556 [-0.2222, +0.3333] | +0.0200 [-0.0805, +0.1186] | 36 |
| wine | Logistic regression full training | Logistic regression classical tabular | +0.0000 [-0.0833, +0.0833] | -0.0022 [-0.0835, +0.0757] | 36 |
| wine | RBF SVM full training | RBF SVM classical tabular | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 36 |
| wine | Random forest full training | Random forest classical tabular | +0.0556 [+0.0000, +0.1389] | +0.0525 [+0.0000, +0.1389] | 36 |
| wine | Hist. gradient boosting full training | Hist. gradient boosting classical tabular | +0.1944 [+0.0556, +0.3611] | +0.2296 [+0.0626, +0.4086] | 36 |

## Probability and execution differences

Classification failures count as incorrect predictions. Probability metrics cover only valid predictions with complete class distributions; coverage is always shown. Native classical probabilities are uncalibrated estimates (RBF SVM supplies none); Qwen values normalize numeric-label-plus-EOS likelihoods; Jev supplies native Choice distributions; OpenAI generated-label runs supply no class distribution. These are different probability-generating procedures.

| Dataset | Model / method | Probability kind | Coverage | Log loss | Brier sum | ECE (15 bins) |
|---|---|---|---:|---:|---:|---:|
| titanic | Majority / 4/class | native_uncalibrated | 100.0% | 13.1827 | 0.7634 | 0.3817 |
| titanic | Majority / full training | native_uncalibrated | 100.0% | 13.1827 | 0.7634 | 0.3817 |
| titanic | Logistic regression / 4/class | native_uncalibrated | 100.0% | 0.7849 | 0.5124 | 0.1215 |
| titanic | Logistic regression / full training | native_uncalibrated | 100.0% | 0.4602 | 0.2946 | 0.0359 |
| titanic | RBF SVM / 4/class | none | 0.0% | — | — | — |
| titanic | RBF SVM / full training | none | 0.0% | — | — | — |
| titanic | Random forest / 4/class | native_uncalibrated | 100.0% | 0.6993 | 0.5050 | 0.1509 |
| titanic | Random forest / full training | native_uncalibrated | 100.0% | 0.5110 | 0.3081 | 0.0618 |
| titanic | Hist. gradient boosting / 4/class | native_uncalibrated | 100.0% | 1.1382 | 0.6845 | 0.3080 |
| titanic | Hist. gradient boosting / full training | native_uncalibrated | 100.0% | 0.4809 | 0.3069 | 0.0893 |
| titanic | Qwen2.5 0.5B / zero-shot | label_sequence_likelihood_normalized | 100.0% | 0.7381 | 0.5285 | 0.1641 |
| titanic | Qwen2.5 0.5B / few-shot 4/class | label_sequence_likelihood_normalized | 100.0% | 0.6817 | 0.4878 | 0.0433 |
| titanic | Qwen2.5 0.5B / LoRA 4/class | label_sequence_likelihood_normalized | 100.0% | 0.7152 | 0.5129 | 0.1429 |
| titanic | Qwen3 4B / zero-shot | label_sequence_likelihood_normalized | 100.0% | 4.1877 | 0.7062 | 0.3521 |
| titanic | Qwen3 4B / few-shot 4/class | label_sequence_likelihood_normalized | 100.0% | 3.9256 | 0.8856 | 0.4458 |
| titanic | Qwen3 4B / QLoRA 4/class | label_sequence_likelihood_normalized | 100.0% | 2.2707 | 0.6808 | 0.3244 |
| titanic | Jev 1.13 / zero-shot | jev_choice_distribution | 100.0% | 0.5097 | 0.3235 | 0.0669 |
| titanic | Jev 1.13 / few-shot 4/class | jev_choice_distribution | 99.6% | 0.5405 | 0.3604 | 0.1026 |
| titanic | GPT-5.6 Luna / zero-shot | unavailable | 0.0% | — | — | — |
| titanic | GPT-5.6 Luna / few-shot 4/class | unavailable | 0.0% | — | — | — |
| titanic | GPT-6 Astra / zero-shot | unavailable | 0.0% | — | — | — |
| titanic | GPT-6 Astra / few-shot 4/class | unavailable | 0.0% | — | — | — |
| breast_cancer | Majority / 4/class | native_uncalibrated | 100.0% | 21.5110 | 1.2456 | 0.6228 |
| breast_cancer | Majority / full training | native_uncalibrated | 100.0% | 13.0278 | 0.7544 | 0.3772 |
| breast_cancer | Logistic regression / 4/class | native_uncalibrated | 100.0% | 0.1521 | 0.0800 | 0.0624 |
| breast_cancer | Logistic regression / full training | native_uncalibrated | 100.0% | 0.0431 | 0.0222 | 0.0340 |
| breast_cancer | RBF SVM / 4/class | none | 0.0% | — | — | — |
| breast_cancer | RBF SVM / full training | none | 0.0% | — | — | — |
| breast_cancer | Random forest / 4/class | native_uncalibrated | 100.0% | 0.1940 | 0.0933 | 0.1170 |
| breast_cancer | Random forest / full training | native_uncalibrated | 100.0% | 0.0726 | 0.0352 | 0.0397 |
| breast_cancer | Hist. gradient boosting / 4/class | native_uncalibrated | 100.0% | 0.3434 | 0.1535 | 0.0634 |
| breast_cancer | Hist. gradient boosting / full training | native_uncalibrated | 100.0% | 0.0462 | 0.0285 | 0.0267 |
| breast_cancer | Qwen2.5 0.5B / zero-shot | label_sequence_likelihood_normalized | 100.0% | 1.1564 | 0.8682 | 0.4371 |
| breast_cancer | Qwen2.5 0.5B / few-shot 4/class | label_sequence_likelihood_normalized | 100.0% | 0.7188 | 0.5225 | 0.1867 |
| breast_cancer | Qwen2.5 0.5B / LoRA 4/class | label_sequence_likelihood_normalized | 100.0% | 0.6699 | 0.4764 | 0.0788 |
| breast_cancer | Qwen3 4B / zero-shot | label_sequence_likelihood_normalized | 100.0% | 1.5616 | 0.6442 | 0.3065 |
| breast_cancer | Qwen3 4B / few-shot 4/class | label_sequence_likelihood_normalized | 100.0% | 1.1060 | 0.2711 | 0.1427 |
| breast_cancer | Qwen3 4B / QLoRA 4/class | label_sequence_likelihood_normalized | 100.0% | 6.4233 | 1.2454 | 0.6227 |
| breast_cancer | Jev 1.13 / zero-shot | jev_choice_distribution | 99.1% | 0.3984 | 0.2361 | 0.1865 |
| breast_cancer | Jev 1.13 / few-shot 4/class | jev_choice_distribution | 99.1% | 0.1622 | 0.0952 | 0.0524 |
| breast_cancer | GPT-5.6 Luna / zero-shot | unavailable | 0.0% | — | — | — |
| breast_cancer | GPT-5.6 Luna / few-shot 4/class | unavailable | 0.0% | — | — | — |
| breast_cancer | GPT-6 Astra / zero-shot | unavailable | 0.0% | — | — | — |
| breast_cancer | GPT-6 Astra / few-shot 4/class | unavailable | 0.0% | — | — | — |
| wine | Majority / 4/class | native_uncalibrated | 100.0% | 23.0259 | 1.3333 | 0.6667 |
| wine | Majority / full training | native_uncalibrated | 100.0% | 21.1070 | 1.2222 | 0.6111 |
| wine | Logistic regression / 4/class | native_uncalibrated | 100.0% | 0.1723 | 0.0690 | 0.1453 |
| wine | Logistic regression / full training | native_uncalibrated | 100.0% | 0.0622 | 0.0262 | 0.0519 |
| wine | RBF SVM / 4/class | none | 0.0% | — | — | — |
| wine | RBF SVM / full training | none | 0.0% | — | — | — |
| wine | Random forest / 4/class | native_uncalibrated | 100.0% | 0.3756 | 0.1789 | 0.2289 |
| wine | Random forest / full training | native_uncalibrated | 100.0% | 0.1264 | 0.0502 | 0.1075 |
| wine | Hist. gradient boosting / 4/class | native_uncalibrated | 100.0% | 0.5854 | 0.2859 | 0.1152 |
| wine | Hist. gradient boosting / full training | native_uncalibrated | 100.0% | 0.0876 | 0.0570 | 0.0360 |
| wine | Qwen2.5 0.5B / zero-shot | label_sequence_likelihood_normalized | 100.0% | 1.4994 | 0.7993 | 0.2687 |
| wine | Qwen2.5 0.5B / few-shot 4/class | label_sequence_likelihood_normalized | 100.0% | 1.5112 | 0.8555 | 0.3778 |
| wine | Qwen2.5 0.5B / LoRA 4/class | label_sequence_likelihood_normalized | 100.0% | 1.1220 | 0.6848 | 0.1205 |
| wine | Qwen3 4B / zero-shot | label_sequence_likelihood_normalized | 100.0% | 5.7532 | 1.0748 | 0.5450 |
| wine | Qwen3 4B / few-shot 4/class | label_sequence_likelihood_normalized | 100.0% | 2.1599 | 0.3837 | 0.2045 |
| wine | Qwen3 4B / QLoRA 4/class | label_sequence_likelihood_normalized | 100.0% | 1.2144 | 0.7219 | 0.1699 |
| wine | Jev 1.13 / zero-shot | jev_choice_distribution | 100.0% | 1.6428 | 1.0065 | 0.4725 |
| wine | Jev 1.13 / few-shot 4/class | jev_choice_distribution | 100.0% | 0.2211 | 0.1118 | 0.1183 |
| wine | GPT-5.6 Luna / zero-shot | unavailable | 0.0% | — | — | — |
| wine | GPT-5.6 Luna / few-shot 4/class | unavailable | 0.0% | — | — | — |
| wine | GPT-6 Astra / zero-shot | unavailable | 0.0% | — | — | — |
| wine | GPT-6 Astra / few-shot 4/class | unavailable | 0.0% | — | — | — |

## Model identity and numerical precision

The table aggregates only completed audited runs. Short revisions below are identifiers, not mutable model aliases; full values and wrapper provenance remain in the JSON/CSV. Missing hosted precision or hardware details are marked undisclosed.

| Requested model | Method | Returned model / revision | Inference dtype | Training dtype / quantization | Device |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | zero-shot | 7ae557604adf | float16 | not applicable | mps |
| `Qwen/Qwen2.5-0.5B-Instruct` | few-shot 4/class | 7ae557604adf | float16 | not applicable | mps |
| `Qwen/Qwen2.5-0.5B-Instruct` | LoRA 4/class | 7ae557604adf | float16 | float32 / ordinary LoRA | mps |
| `Qwen/Qwen3-4B-Instruct-2507` | zero-shot | cdbee75f17c0 | float16 | not applicable | cuda |
| `Qwen/Qwen3-4B-Instruct-2507` | few-shot 4/class | cdbee75f17c0 | float16 | not applicable | cuda |
| `Qwen/Qwen3-4B-Instruct-2507` | QLoRA 4/class | cdbee75f17c0 | float16 | bfloat16 / 4-bit base | cuda |
| `typesafe/jev-1.13` | zero-shot | typesafe/jev-1.13-20260917 | undisclosed | not applicable | undisclosed |
| `typesafe/jev-1.13` | few-shot 4/class | typesafe/jev-1.13-20260917 | undisclosed | not applicable | undisclosed |
| `gpt-5.6-luna` | zero-shot | gpt-5.6-luna | undisclosed | not applicable | undisclosed |
| `gpt-5.6-luna` | few-shot 4/class | gpt-5.6-luna | undisclosed | not applicable | undisclosed |
| `gpt-6-astra` | zero-shot | gpt-6-astra | undisclosed | not applicable | undisclosed |
| `gpt-6-astra` | few-shot 4/class | gpt-6-astra | undisclosed | not applicable | undisclosed |

Requested and returned model IDs/revisions, actual recorded inference precision/devices, training precision/quantization, wrapper hashes, latency and training time are retained in the JSON/CSV. Hosted hardware is not disclosed. Local sequential likelihood scoring, hosted end-to-end request latency and native amortized batch inference are different timing conventions.

## Scope and limitations

These are familiar public benchmarks; memorized rows or dataset-specific priors may influence pretrained models. Grouped splits prevent exact feature overlap in newly supplied data, but cannot remove pretraining contamination. Wine cultivar numbers are arbitrary identifiers, so zero-shot results are not evidence that cultivar names have transferable semantic meaning. The Breast Cancer task is a benchmark of a historical dataset, not a clinical validation or deployment recommendation.

This extension evaluates one fixed named-feature serialization, with source scales preserved and no truncation. It does not compare alternative serialization formats, feature-order permutations or missing-value wordings. Family relationships in Titanic may cause dependence beyond identical feature vectors. Tiny four-per-class adaptation is a narrow fixed-recipe test; a poor result does not establish a general limit of fine-tuning.

Hosted LoRA for Jev/OpenAI is unsupported by this study and is not assigned a score. LoRA and QLoRA apply only to the downloadable open models.

Protocol and source citations: [TABULAR_PROTOCOL.md](../docs/TABULAR_PROTOCOL.md). Machine-readable evidence: [TABULAR_COMPARISON.json](TABULAR_COMPARISON.json) and [TABULAR_COMPARISON.csv](TABULAR_COMPARISON.csv).
