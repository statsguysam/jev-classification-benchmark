# Reproduce the completion and failure-recovery analysis

The [completion protocol](COMPLETION_RETRY_PROTOCOL.md) fixes the execution and
recovery policy. It supplements the original numerical, text and matched-control
protocols without replacing their predictions or pricing declarations.

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

Restore the ignored public datasets on a fresh checkout before running any
summary. The historical failure audit also needs Titanic, even though the
focused proposal-control study uses Breast Cancer, Wine, SST-2 and TREC.
These preparation commands may download public dataset files; they make no
model requests. Do not prepare over existing experiment evidence.

```bash
python scripts/tabular_data.py --datasets titanic breast_cancer wine \
  --output data/tabular-full --seed 42
jevbench prepare sst2 trec --cache-dir data --output data/pilot \
  --seed 42 --train-limit 10000 --validation-limit 1000 \
  --test-limit 200 --max-text-chars 2000
python scripts/restore_text_extension_data.py --candidate-root data/pilot
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/restore_review_controls_payload.py
```

The text restoration checks every frozen row-file hash before restoring the
historical manifest bytes. The control restoration reconstructs only the ignored
`results/review_controls/prepared_full/requests.jsonl` from those frozen datasets
and saved source predictions. It requires exact agreement with the versioned
protocol and manifest, including all 1,714 request identities. An existing exact
payload is verified; an existing changed or partial payload is never overwritten.
Neither restoration makes API calls or allocates a budget.

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
contrasts. A rendering command cannot turn incomplete evidence into a score.
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

The combined ceiling is US$25. Original failed-call reservations remain retained.
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
