# Jev pilot and probability quality

**4 completed Jev runs** are recorded below. Only completed artifacts supply measurements. Jev is accessed as `typesafe/jev-1.13` through OpenRouter; the frozen inference core remains unchanged and wrapper provenance is retained in the JSON evidence.

This is the shared SST-2/TREC pilot: 200 held-out rows per dataset, selection seed 42 and a 2,000-character input prefix. Zero-shot uses no training examples. Few-shot uses exactly four per class (eight SST-2 labels or 24 TREC labels), matched to the classical and other few-shot/adapted arms; no development labels are used. The different regimes remain labeled separately. Jev adapter training was not run.

| Dataset | Jev method | New labels | Accuracy | Macro F1 | Errors | Probability coverage | Artifact |
|---|---|---:|---:|---:|---:|---:|---|
| sst2 | zero-shot | 0 | 0.9350 | 0.9395 | 2 | 198/200 | [run](jev/sst2__jev-1.13__9d5a851767fc/run.json) |
| sst2 | few-shot 4/class | 8 | 0.9650 | 0.9650 | 0 | 200/200 | [run](jev/sst2__jev-1.13__a33c2b36ded6/run.json) |
| trec | zero-shot | 0 | 0.3350 | 0.3622 | 3 | 197/200 | [run](jev/trec__jev-1.13__ce2f8d23bf30/run.json) |
| trec | few-shot 4/class | 24 | 0.8550 | 0.8740 | 0 | 200/200 | [run](jev/trec__jev-1.13__bc52fb6a4b0f/run.json) |

## Recorded Jev failures

The fixed validator counts failed predictions as incorrect in accuracy and as missing predictions in macro F1. Failures include any recorded transport errors and output-validation errors; a choice-versus-maximum-probability rejection is an output-validation failure, not evidence of a timeout. The table below uses the original saved row error strings without reclassifying the failure or imputing a label.

The [official Choice contract](https://docs.typesafe.ai/primitives/choice) defines the chosen option as the one with the highest probability. The frozen validator rejects a response when `p(choice) < max(p) - 1e-6`. A rejection through OpenRouter does not isolate whether its cause lies in the underlying model or a serving layer. The retained artifacts do not support a claim about that cause or the numerical size of the rejected disagreement.

| Dataset | Method | Original recorded error | Rows |
|---|---|---|---:|
| sst2 | zero-shot | `network_error: connection failed or timed out; no automatic retry` | 2 |
| sst2 | few-shot 4/class | No recorded failures | 0 |
| trec | zero-shot | `invalid_output: Jev choice disagrees with maximum probability` | 1 |
| trec | zero-shot | `invalid_output: probabilities do not sum to one` | 1 |
| trec | zero-shot | `network_error: connection failed or timed out; no automatic retry` | 1 |
| trec | few-shot 4/class | No recorded failures | 0 |

Of 5 recorded Jev failures, 5 have no retained probability distribution. Probability scores exclude every failed prediction, including choice-versus-argmax rejections. The rejected raw distributions are unavailable when the saved field is null; their NLL, Brier and calibration errors cannot be reconstructed and are not imputed. Original prediction files retain available usage, request IDs, model and routing metadata. The valid-row `chosen_label_argmax_disagreement_rows` metric applies only to retained valid predictions: a zero value does not mean that no raw API response disagreed with its argmax.


## Probability metrics on the fixed datasets

Every probability metric below was recomputed from saved predictions and checked against the recorded score. Exact test content and matched training IDs were verified against frozen prepared files. Jev uses its API-returned Choice distribution; Qwen scores complete numeric class IDs plus EOS and normalizes over the permitted labels; Naive Bayes uses its native `predict_proba`. These distributions have different semantics. Their observed quality can be compared on the same labels and rows, but a score difference does not isolate architecture or establish a general calibration advantage.

Lower NLL, Brier sum and ECE are better within the same dataset. NLL clips probabilities at 1e-15 then renormalizes; a finite NLL does not remove the significance of an exact zero assigned to the true class. Brier is the sum of squared class-probability errors per row, without division by the class count. ECE uses 15 equal-width bins of the maximum predicted probability and the argmax class's correctness. Bin estimates are noisy at 200 rows; no post-hoc calibration, test-guided thresholds or probability-parameter tuning were performed.

Probability metrics are conditional on rows with a valid class prediction and distribution. Coverage and failures are reported so missing distributions cannot masquerade as better calibration. These are point estimates, not confidence intervals or multiplicity-adjusted claims. Accuracy and macro F1 include inference failures; the table does not pool different datasets or training-label budgets.

### SST2

| Model | Method | New labels | Coverage | NLL | Brier sum | ECE | True-class zeros |
|---|---|---:|---:|---:|---:|---:|---:|
| [Jev 1.13](jev/sst2__jev-1.13__9d5a851767fc/run.json) | zero-shot | 0 | 198/200 | 0.3026 | 0.0850 | 0.0408 | 1 |
| [Jev 1.13](jev/sst2__jev-1.13__a33c2b36ded6/run.json) | few-shot 4/class | 8 | 200/200 | 0.0853 | 0.0491 | 0.0121 | 0 |
| [Qwen2.5 0.5B](pilot/sst2__Qwen2.5-0.5B-Instruct__594a827ceab2/run.json) | zero-shot | 0 | 200/200 | 0.7051 | 0.5120 | 0.1849 | 0 |
| [Qwen2.5 0.5B](pilot/sst2__Qwen2.5-0.5B-Instruct__9800b4c889cc/run.json) | few-shot 4/class | 8 | 200/200 | 0.4122 | 0.2496 | 0.1802 | 0 |
| [Qwen2.5 0.5B](pilot/sst2__Qwen2.5-0.5B-Instruct__c29d9f2be13f/run.json) | LoRA 4/class | 8 | 200/200 | 0.3265 | 0.1925 | 0.1055 | 0 |
| [Qwen3 4B](colab/sst2__Qwen3-4B-Instruct-2507__0e4acd9aac6a/run.json) | zero-shot | 0 | 200/200 | 0.7770 | 0.1521 | 0.0772 | 0 |
| [Qwen3 4B](colab/sst2__Qwen3-4B-Instruct-2507__5823bb43da99/run.json) | few-shot 4/class | 8 | 200/200 | 0.7980 | 0.0889 | 0.0460 | 0 |
| [Qwen3 4B](colab/sst2__Qwen3-4B-Instruct-2507__22d330c63925/run.json) | QLoRA 4/class | 8 | 200/200 | 1.0703 | 0.1232 | 0.0633 | 0 |
| [TF-IDF MultinomialNB](pilot/sst2__multinomial_nb__a094038879f9/run.json) | classical 4/class | 8 | 200/200 | 0.6986 | 0.5050 | 0.0944 | 0 |

### TREC

| Model | Method | New labels | Coverage | NLL | Brier sum | ECE | True-class zeros |
|---|---|---:|---:|---:|---:|---:|---:|
| [Jev 1.13](jev/trec__jev-1.13__ce2f8d23bf30/run.json) | zero-shot | 0 | 197/200 | 1.7631 | 0.7966 | 0.3291 | 2 |
| [Jev 1.13](jev/trec__jev-1.13__bc52fb6a4b0f/run.json) | few-shot 4/class | 24 | 200/200 | 0.2819 | 0.1703 | 0.0538 | 0 |
| [Qwen2.5 0.5B](pilot/trec__Qwen2.5-0.5B-Instruct__67b435872393/run.json) | zero-shot | 0 | 200/200 | 2.7296 | 1.0899 | 0.3375 | 0 |
| [Qwen2.5 0.5B](pilot/trec__Qwen2.5-0.5B-Instruct__5d00e79e1d60/run.json) | few-shot 4/class | 24 | 200/200 | 2.4058 | 0.9237 | 0.2338 | 0 |
| [Qwen2.5 0.5B](pilot/trec__Qwen2.5-0.5B-Instruct__23d9b42b5b4e/run.json) | LoRA 4/class | 24 | 200/200 | 2.0089 | 0.9209 | 0.2912 | 0 |
| [Qwen3 4B](colab/trec__Qwen3-4B-Instruct-2507__2f1ff41bd14e/run.json) | zero-shot | 0 | 200/200 | 7.0131 | 0.9364 | 0.4754 | 0 |
| [Qwen3 4B](colab/trec__Qwen3-4B-Instruct-2507__c0b1cc9b3d0f/run.json) | few-shot 4/class | 24 | 200/200 | 2.2314 | 0.3187 | 0.1618 | 0 |
| [Qwen3 4B](colab/trec__Qwen3-4B-Instruct-2507__70bf91598825/run.json) | QLoRA 4/class | 24 | 200/200 | 0.8107 | 0.3116 | 0.1363 | 0 |
| [TF-IDF MultinomialNB](pilot/trec__multinomial_nb__db3012a1e2fb/run.json) | classical 4/class | 24 | 200/200 | 1.5382 | 0.7362 | 0.2277 | 0 |

## Matched-label paired comparisons

The separate [comparison index](comparisons/combined/index.md) includes Jev few-shot minus fixed TF-IDF Naive Bayes, Astra few-shot and Qwen 4B few-shot. Each comparison audits the exact same training IDs/seed and test items. These are additional exploratory, unadjusted contrasts selected after earlier pilot results were visible; they are not prospectively registered. No equal-training paired claim is made for Jev zero-shot versus few-shot. Cross-model/protocol contrasts remain descriptive.

OpenAI label-only runs do not expose a class-probability vector in this experiment, so they have no NLL, Brier or ECE entry. Jev probabilities alone do not establish an advantage over absent OpenAI probability measurements. [Combined accuracy report](COMPARISON.md) · [Colab content audit](COLAB_AUDIT.md).

[CSV](JEV_PILOT.csv) · [Full audit, reliability bins and provenance](JEV_PILOT.json)

Rebuild with `PYTHONPATH=src .venv/bin/python scripts/summarize_jev_pilot.py`.
