# Audited recovery: original outcomes remain visible

Finite policy complete: 71 new service calls across 23 affected original conditions.

Accuracy / failures below use the full original condition. First actual call and recovered views are separate from historical upstream skips.

| Dataset | Model / source | Examples/class | Original snapshot | First actual call | Recovered | New calls |
|---|---|---:|---|---|---|---:|
| breast_cancer | typesafe/jev-1.13 / proposal: gpt-6-astra | 4 | 93.86% / 1 | 93.86% / 1 | 94.74% / 0 | 1 |
| breast_cancer | typesafe/jev-1.13 | 4 | 92.98% / 1 | 92.98% / 1 | 93.86% / 0 | 1 |
| breast_cancer | typesafe/jev-1.13 | 0 | 84.21% / 1 | 84.21% / 1 | 85.09% / 0 | 1 |
| breast_cancer | typesafe/jev-1.13 / proposal: gpt-5.6-luna | 4 | 92.11% / 1 | 92.11% / 1 | 92.98% / 0 | 1 |
| breast_cancer | typesafe/jev-1.13 / proposal: HuggingFaceTB/SmolLM2-1.7B-Instruct | 4 | 89.47% / 6 | 89.47% / 6 | 93.86% / 0 | 6 |
| sst2 | typesafe/jev-1.13 / proposal: gpt-6-astra | 0 | 93.50% / 1 | 93.50% / 1 | 94.00% / 0 | 1 |
| sst2 | typesafe/jev-1.13 | 0 | 93.50% / 2 | 93.50% / 2 | 94.50% / 0 | 2 |
| sst2 | typesafe/jev-1.13 / proposal: Qwen/Qwen3-4B-Instruct-2507 | 0 | 94.00% / 1 | 94.00% / 1 | 94.50% / 0 | 1 |
| sst2 | typesafe/jev-1.13 / proposal: Qwen/Qwen3-4B-Instruct-2507 | 4 | 96.00% / 1 | 96.00% / 1 | 96.50% / 0 | 1 |
| titanic | typesafe/jev-1.13 | 4 | 74.43% / 1 | 74.43% / 1 | 74.81% / 0 | 1 |
| trec | typesafe/jev-1.13 / proposal: gpt-6-astra | 0 | 42.00% / 10 | 42.00% / 10 | 44.50% / 1 | 11 |
| trec | typesafe/jev-1.13 / proposal: ibm-granite/granite-3.3-2b-instruct | 0 | 39.50% / 3 | 39.50% / 3 | 39.50% / 0 | 3 |
| trec | typesafe/jev-1.13 / proposal: ibm-granite/granite-3.3-2b-instruct | 4 | 85.50% / 4 | 85.50% / 4 | 87.50% / 0 | 4 |
| trec | typesafe/jev-1.13 | 0 | 33.50% / 3 | 33.50% / 3 | 33.50% / 0 | 3 |
| trec | typesafe/jev-1.13 / proposal: gpt-5.6-luna | 0 | 39.50% / 4 | 39.50% / 4 | 41.00% / 0 | 6 |
| trec | typesafe/jev-1.13 / proposal: gpt-5.6-luna | 4 | 85.00% / 1 | 85.00% / 1 | 85.50% / 0 | 1 |
| trec | typesafe/jev-1.13 / proposal: Qwen/Qwen3-4B-Instruct-2507 | 0 | 41.50% / 7 | 41.50% / 7 | 42.00% / 0 | 7 |
| trec | typesafe/jev-1.13 / proposal: Qwen/Qwen2.5-0.5B-Instruct | 0 | 32.50% / 7 | 32.50% / 7 | 33.50% / 0 | 8 |
| trec | typesafe/jev-1.13 / proposal: HuggingFaceTB/SmolLM2-1.7B-Instruct | 0 | 32.50% / 7 | 32.50% / 7 | 34.50% / 0 | 7 |
| trec | typesafe/jev-1.13 / proposal: HuggingFaceTB/SmolLM2-1.7B-Instruct | 4 | 84.50% / 2 | 84.50% / 2 | 85.50% / 0 | 2 |
| wine | typesafe/jev-1.13 / proposal: gpt-6-astra | 4 | 94.44% / 1 | 97.22% / 0 | 97.22% / 0 | 1 |
| wine | gpt-6-astra | 4 | 97.22% / 1 | 97.22% / 1 | 100.00% / 0 | 1 |
| wine | typesafe/jev-1.13 / proposal: ibm-granite/granite-3.3-2b-instruct | 0 | 33.33% / 1 | 33.33% / 1 | 36.11% / 0 | 1 |

## API outcomes

{"dependent_first_calls": 1, "dependent_review_calls": 1, "new_calls": 71, "ordinary_paid_failure_retry_calls": 70, "outcomes": {"invalid_output": 5, "success": 66}, "stages": {"dependent_first_call": 1, "retry": 70}, "upstream_unresolved_not_called": 0}

The paired intervals in RECOVERY_COMPARISON.json compare recovered minus first actual call. They do not attribute semantic improvements to retrying a valid response.

- Original outcomes and reservations remain immutable; retries add calls and cost.
- An upstream skip is distinct from a paid failure. A dependent review begins only after its source has a valid recovered response.
- First attempt means the first actual service call; original_snapshot also includes historical upstream skips.
- Earliest valid success is selected without using truth labels. Original valid responses, including wrong ones, remain unchanged.
- Full-condition metrics include failures as incorrect. Unattempted dependent rows suppress full service-view metrics.
- Paired group bootstrap intervals are unadjusted and conditional on these cases and observed attempts; they measure recovery availability, not semantic model improvement.
- Affected conditions were selected for operational errors and are not a new independent benchmark. No overall accuracy winner or latency claim is reported.
