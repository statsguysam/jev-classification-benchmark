# Colab import audit

**All 14 completed imported runs passed.** The eight classical replications produced exactly the same ordered class predictions, accuracy, macro F1, balanced accuracy, confusion matrices and per-class scores as their local counterparts. Both environments used the same frozen data and selected training rows.

The current local inference core and every imported run record share source SHA-256 `d5547ccde4224e182653315d81c0a631c789bbe294ad7bcf95ef0cddd25ce608`. Colab used Python 3.13; local runs used Python 3.12. The audit checks realized rows and file hashes rather than inferring reproducibility from equal seeds.

Original manifest hashes differ and remain unchanged. Their only differing keys are `splits_before_text_transform` and `post_transform_overlap_removed_ids`, which were added as supplemental preparation audits. For every pair, all three frozen JSONL file hashes and the full prepared-content hash were recomputed from local files and matched both recorded manifests. Ordered test IDs, labels and text hashes; selected training IDs; selection seed; and validation-label use also matched.

The six neural checks compare the 4B and 0.5B runs only for data/training identity. They do not establish identical models, numerical execution, probabilities or predictions. Probability and timing equality is not claimed for the classical replications. Original `compare_runs` retains its exact-manifest requirement; separately labeled paired analyses retain both hashes and embed this content evidence.

| Dataset | Method / imported model | Local counterpart | Data/training audit | Classical class-label replication |
|---|---|---|---|---|
| sst2 | [zero_shot / Qwen/Qwen3-4B-Instruct-2507](colab/sst2__Qwen3-4B-Instruct-2507__0e4acd9aac6a/run.json) | [run](pilot/sst2__Qwen2.5-0.5B-Instruct__594a827ceab2/run.json) | passed | not applicable |
| sst2 | [lora / Qwen/Qwen3-4B-Instruct-2507](colab/sst2__Qwen3-4B-Instruct-2507__22d330c63925/run.json) | [run](pilot/sst2__Qwen2.5-0.5B-Instruct__c29d9f2be13f/run.json) | passed | not applicable |
| sst2 | [few_shot / Qwen/Qwen3-4B-Instruct-2507](colab/sst2__Qwen3-4B-Instruct-2507__5823bb43da99/run.json) | [run](pilot/sst2__Qwen2.5-0.5B-Instruct__9800b4c889cc/run.json) | passed | not applicable |
| sst2 | [classical / linear_svc](colab/sst2__linear_svc__d849105442ba/run.json) | [run](pilot/sst2__linear_svc__3e7e9ec774d5/run.json) | passed | exact |
| sst2 | [classical / logistic_regression](colab/sst2__logistic_regression__a2b87ec12b18/run.json) | [run](pilot/sst2__logistic_regression__81d2ff14c2be/run.json) | passed | exact |
| sst2 | [classical / majority](colab/sst2__majority__14e58b6b44f9/run.json) | [run](pilot/sst2__majority__face6e86e516/run.json) | passed | exact |
| sst2 | [classical / multinomial_nb](colab/sst2__multinomial_nb__a0cf6a1ac8df/run.json) | [run](pilot/sst2__multinomial_nb__a094038879f9/run.json) | passed | exact |
| trec | [zero_shot / Qwen/Qwen3-4B-Instruct-2507](colab/trec__Qwen3-4B-Instruct-2507__2f1ff41bd14e/run.json) | [run](pilot/trec__Qwen2.5-0.5B-Instruct__67b435872393/run.json) | passed | not applicable |
| trec | [lora / Qwen/Qwen3-4B-Instruct-2507](colab/trec__Qwen3-4B-Instruct-2507__70bf91598825/run.json) | [run](pilot/trec__Qwen2.5-0.5B-Instruct__23d9b42b5b4e/run.json) | passed | not applicable |
| trec | [few_shot / Qwen/Qwen3-4B-Instruct-2507](colab/trec__Qwen3-4B-Instruct-2507__c0b1cc9b3d0f/run.json) | [run](pilot/trec__Qwen2.5-0.5B-Instruct__5d00e79e1d60/run.json) | passed | not applicable |
| trec | [classical / linear_svc](colab/trec__linear_svc__a0eff755888f/run.json) | [run](pilot/trec__linear_svc__d2b4ae115ac0/run.json) | passed | exact |
| trec | [classical / logistic_regression](colab/trec__logistic_regression__061883d7eb3d/run.json) | [run](pilot/trec__logistic_regression__8bc76aab4183/run.json) | passed | exact |
| trec | [classical / majority](colab/trec__majority__4effad8e22e5/run.json) | [run](pilot/trec__majority__d3cfb0a0827d/run.json) | passed | exact |
| trec | [classical / multinomial_nb](colab/trec__multinomial_nb__9794e79f5e77/run.json) | [run](pilot/trec__multinomial_nb__db3012a1e2fb/run.json) | passed | exact |

The [complete JSON evidence](COLAB_AUDIT.json) preserves both full manifest hashes, content/file hashes, exact metadata differences and before/after hashes confirming original artifacts were unchanged. [Exploratory paired comparisons](comparisons/combined/index.md).

Rebuild with `PYTHONPATH=src .venv/bin/python scripts/audit_colab_import.py`.
