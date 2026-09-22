# Numeric dashboard data schema v1

By default, `dashboard/dist/numeric-data.json` is generated only after all 68 conditions and 72 contrasts are complete, with cost accounting reconciled. An explicit `--allow-incomplete` flag permits an operational snapshot containing all planned conditions and contrasts, with unfinished performance fields kept null. Both modes rerun the same read-only scientific audits and require exact agreement with the saved report. The exporter uses an aggregate-field allowlist. There are no row records, prompts, source paths, request IDs, credentials or configuration objects.

```bash
# Default: require the full completed matrix.
.venv/bin/python scripts/build_expanded_numeric_dashboard.py
# Explicit operational snapshot: preserve all pending/halted rows without scores.
.venv/bin/python scripts/build_expanded_numeric_dashboard.py --allow-incomplete
```

An incomplete snapshot preserves `completion.status = "in_progress_or_incomplete"` and actual completed run/contrast counts. Each run has `status`: `complete`, `running`, `stopped_after_three_consecutive_errors`, or `pending: no complete audited artifact`. Each comparison has `status`: `complete` or `pending`. Unfinished runs have null scores, CIs, CI metadata, probability coverage/loss, failure counts/rates, `n_test`, `train_labels`, and transitions. Their declared dataset/model/class/shot/budget descriptors remain available; use `datasets[].n_test` for the planned denominator. Never interpret null as zero or score the partial checkpoint.

Pending reviews retain `source_model` and a `source_run_id` that resolves to the base row. Missing review placeholders use the planned final run ID `dataset__model_key__k{0|4}__jev-review`; comparison references are remapped consistently. A halted run keeps its existing artifact ID. Pending comparison deltas, endpoints, CI metadata and transitions are all null. Their A/B references and label-budget descriptors remain usable. Incomplete-state notes are included in `limitations` and cost notes.

All accuracy/F1 scores and interval endpoints are fractions in `[0,1]`. Differences and paired interval endpoints are fractions in `[-1,1]`; multiply by 100 to display percentage points. USD values are decimal strings. Null means unavailable, never zero. Percentile intervals may exclude their point estimate, so use their exact endpoints.

Top-level keys:

```text
schema_version: 1
scope: string
completion: {status, complete_runs, expected_runs, complete_comparisons, expected_comparisons}
datasets: [{id, label, data_type, task, n_classes, n_features, n_test, full_training_labels,
            classes: [{id, label}]}]
models: [{id, label, family}]
runs: [Run]
comparisons: [Comparison]
costs: CostSummary
limitations: [string]
provenance: {source_report_sha256, report_builder_sha256, protocol_sha256, exporter_sha256,
             bootstrap_samples, seed}
```

`data_type` is `numeric`; `task` is `binary` or `multiclass`. Model `family` is `open_weight_llm`, `hosted_llm`, `jev`, or `classical`. Model IDs use exact repository/API/model names. For review runs, `model` includes `+jev_review`; `source_model` identifies the source model and matches an entry in `models`.

```json
{
  "run_id": "example-run-id",
  "status": "complete",
  "dataset": "breast_cancer",
  "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct+jev_review",
  "display_model": "SmolLM2 1.7B → Jev",
  "source_model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  "source_run_id": "example-source-run-id",
  "reviewer_model": "typesafe/jev-1.13",
  "model_family": "open_weight_llm",
  "arm": "review",
  "jev_mode": "review",
  "label_budget": "four_per_class",
  "shots_per_class": 4,
  "train_per_class": 4,
  "train_labels": 8,
  "n_test": 114,
  "n_classes": 2,
  "accuracy": 0.5,
  "accuracy_ci95": [0.4, 0.6],
  "macro_f1": 0.45,
  "macro_f1_ci95": [0.35, 0.55],
  "balanced_accuracy": 0.5,
  "n_failures": 0,
  "failure_rate": 0.0,
  "probability_coverage": 1.0,
  "log_loss": null,
  "brier_sum": null,
  "ci_method": "unstratified group percentile bootstrap",
  "ci_samples": 2000,
  "ci_n_groups": 114,
  "review_transitions": null
}
```

The values above are schema examples, **not measured results**. `arm` is `base`, `review`, `direct`, or `classical`; `jev_mode` is `none`, `review`, or `alone`. `label_budget` is `zero_shot`, `four_per_class`, or `full_training`. Classical `shots_per_class` is null because its labels train a model; `train_per_class` remains 4 or null. Full-training `train_per_class` is null. `source_model` is the base LLM for both base and review arms and null for direct Jev/classical arms. `source_run_id` and `reviewer_model` are nonnull only for review arms.

Each comparison contains:

```text
kind: review_minus_source | review_minus_jev_alone | few_minus_zero_descriptive
status: complete | pending
dataset, a, b: dataset ID and compared run IDs; every difference is A minus B
equal_new_label_budget: boolean
train_per_class_a, train_per_class_b: 0 or 4
accuracy_delta, accuracy_delta_ci95, macro_f1_delta, macro_f1_delta_ci95
ci_method, ci_samples, ci_n_groups
transitions: null or TransitionCounts
```

`TransitionCounts`, also used for a review run's `review_transitions`, has exactly: `wrong_to_correct`, `correct_to_wrong`, `correct_to_wrong_label`, `correct_to_failure`, `both_correct`, `both_wrong`, `changed_predictions`, `source_failure_rows`, `review_stage_failure_rows`, `net_correct_change`, and `accuracy_delta_pp`. `correct_to_wrong` includes label errors and failures; do not sum it again with its two components. `accuracy_delta_pp` is already in percentage points.

`CostSummary` has: `status`, `new_review_calls`, `known_reported_api_usd`, `unknown_cost_calls`, `new_review_reserved_usd`, `prior_reserved_usd`, `cumulative_reserved_usd`, `authorized_usd`, `new_source_api_calls`, `local_compute_usd` (null), `source_failure_rows_without_jev_call`, `reused_review_conditions`, `complete_review_conditions`, `expected_review_conditions`, `reservations_without_result`, `results_without_saved_prediction`, `partial_checkpoint_files`, `per_condition`, and `notes` (string array). Its status preserves the audited `complete`, `halted`, or `in_progress` state; actual calls, unknown charges and retained reservations include incomplete attempts. A `per_condition` entry has `dataset`, `source_model`, `shots_per_class`, `model_requests`, `known_reported_api_usd`, `unknown_cost_requests`, and `conservative_usd`. These are new-review-only costs. Four reused reviews and previously generated hosted proposals are accounted in prior spending; unknown API charges are not free calls.
