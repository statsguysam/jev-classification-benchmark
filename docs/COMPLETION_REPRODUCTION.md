# Reproduce the completion and failure-recovery analysis

This guide rebuilds the reports from saved predictions. The [completion
protocol](COMPLETION_RETRY_PROTOCOL.md) records how the remaining requests and
failed-call recovery were handled. It supplements the original numerical, text
and matched-control protocols; their predictions and pricing declarations remain
part of the evidence.

The active package now includes fixes made after this study. Use the verified
archived runtime for the historical commands below. The [runtime guide](RUNTIME_MAINTENANCE.md)
explains how current tests, original tests and saved evidence remain separate.

## Environment and frozen data

From a fresh checkout, use Python 3.12 and install the analysis dependencies.
The numerical library pins match the CI audit environment; Matplotlib supplies
the four figure commands below. No model weights or GPU are needed for this
analysis workflow.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,numeric]' \
  'transformers==4.57.6' 'huggingface-hub==0.36.2' \
  'scikit-learn==1.9.1' 'numpy==2.5.3' 'scipy==1.18.1' \
  'matplotlib>=3.9,<4'
```

On a fresh checkout, prepare the ignored public datasets with the archived code.
This also restores the exact historical text manifests. It downloads pinned
public data and makes no model requests. Skip this step when the prepared
datasets already exist; the exporter refuses to replace them.

```bash
python scripts/reproduce_frozen_study.py prepare-data --export-prepared data
```

For a complete report check, run `python scripts/reproduce_frozen_study.py audit`.
It rebuilds eight machine-readable reports in a temporary checkout and requires
each to match the saved bytes. No original file is changed.

To run the detailed commands below, create a persistent copy instead:

```bash
python scripts/reproduce_frozen_study.py stage \
  --workspace artifacts/frozen-study-324d633
cd artifacts/frozen-study-324d633
export PYTHONPATH="$PWD/src:$PWD/scripts"
```

Keep the project environment activated. The remaining commands run from this
copied checkout, which contains the original scripts and package, saved results
and prepared data. The launcher also restores the ignored control request
payload and verifies all 1,714 request identities against their frozen manifest.
It does not load models or allocate a budget.

## Offline analysis

With the environment and frozen data restored, these commands read existing
evidence and make no model requests:

```bash
python scripts/summarize_expanded_numeric.py --bootstrap-samples 2000
python scripts/summarize_text_extension.py --bootstrap-samples 2000
python scripts/analyze_review_value.py
python scripts/summarize_failed_retries.py --bootstrap-samples 2000
python scripts/summarize_review_controls.py
python scripts/summarize_control_failed_retries.py --bootstrap-samples 2000
python scripts/summarize_completion_costs.py
```

The recovery summarizer calls the runner's read-only `audit_run()` API to verify
the completed finite policy, protected originals and ledger before reporting.
The runner's dry-run CLI is for planning before controls are allocated; it
intentionally refuses to plan additional historical retries after controls exist.
A partial execution is never reported as completed merely because a checkpoint
exists. Recovered scores belong to the separate recovery report, not to the
historical source/review files.
The [output-validation notes](OUTPUT_VALIDATION_NOTES.md) explain why a rejected
probability vector is distinct from a known wrong class, and how both affect
first-attempt pipeline accuracy.

Once aggregate reports match their audited evidence, export the dashboard:

```bash
python scripts/build_expanded_numeric_dashboard.py
python scripts/build_text_extension_dashboard.py
python scripts/build_review_value_dashboard.py
python scripts/analyze_review_sensitivity.py
node dashboard/tests/numeric-dashboard.smoke.cjs dashboard
node dashboard/tests/text-dashboard.smoke.cjs dashboard
node dashboard/tests/review-dashboard.smoke.cjs dashboard
python scripts/plot_review_value.py
python scripts/plot_review_controls.py
python scripts/plot_review_outcomes.py
python scripts/plot_selected_review_examples.py
```

Generate the sensitivity appendix after the dashboard: its provenance includes
the exact `review-data.json` hash. Regenerating that asset later requires
regenerating the appendix to keep the recorded pin current.

The four figure commands additionally require Matplotlib. Their manifests
record the rendering version and hashes of the input evidence and output files.

The matched-control figure requires all 1,714 original calls to be present and
the saved report to equal a fresh audit. It shows every dataset and both planned
contrasts and refuses to plot incomplete evidence.
The review-outcomes figure likewise requires both complete 68-condition matrices.
It displays all 48 source-to-review conditions, separating corrections, wrong-label
harms and failure harms while retaining each full test denominator. The selected-example
figure uses two fixed named conditions for a readable social post; it is explicitly
not an overall model ranking and includes a matching-label classical reference for Wine.

## Execution provenance

The September 23 completion wrapper is `scripts/run_jev_completion.py`. Its
`numeric-recovery`, `numeric`, `text` and `controls` modes forward the unchanged
original requests through an independently reverified route check. Each paid
session writes a receipt under `results/completion_sessions/`, including current
public documentation hashes, source checksums and an audit of preserved evidence.
The wrapper sends no inference request without `--execute`.

The wrapper is deliberately date-bound. Do not edit a date or weaken a pin to
make old credentials, prices or source changes pass. A later execution date or
changed model route requires a newly reviewed compatibility path with its own
verification evidence. Offline analysis of the saved results does not need that
dated transport check.

Historical failure recovery uses `scripts/run_failed_retries.py` and its separate
`results/completion_20260923/retries/` directory. Its initial plan freezes the live
failure inventory, request identities, earlier ledgers and full conservative
allocation. `attempts.jsonl` records every new call. The final `run.json` binds the
plan, attempts, ledger and ledger anchor by hash.

For the originally skipped Wine review, the recovered Astra response creates an
explicitly versioned source proposal. Its first Jev call is a dependent first
call, not a retry of a nonexistent earlier request. The aggregate report keeps
that distinction visible.

## Credentials and accounting

Execution can read `OPENROUTER_API_KEY` and `OPENAI_API_KEY` from the process
environment or request them with hidden terminal prompts. Keys do not belong in
commands, source files, notebooks, reports or Git. A new ledger is initialized
once; resuming always omits `--init-ledger` and preserves both the ledger and its
`.lock` anchor.

The combined ceiling is US$25. Original failed-call reservations remain in the ledger.
Every new Jev recovery call also retains its full conservative reservation. Only
eligible, verified successful usage in the separately documented current drivers
can settle a new reservation. These records bound this study's accounted requests;
they are not a provider invoice or an account-wide billing limit.

The real/no/shuffled control experiment uses its frozen original calls for the
primary analysis. Any subsequent failed-control recovery is a separate secondary
analysis and cannot overwrite those primary metrics or the original serving-repeat
diagnostic.

The post-control runner is `scripts/run_control_failed_retries.py`; its evidence
is separate under `results/completion_20260923/control_retries/`. If the complete
control experiment has no failed calls, it writes an audited no-op with no API
requests or positive-budget ledger. Its secondary summary requires the linked
primary control report to match a fresh audit before publication.
