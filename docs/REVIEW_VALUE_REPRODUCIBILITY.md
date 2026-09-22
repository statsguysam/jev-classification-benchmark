# Reproduce the review-value analysis and control plan

The existing-results analysis and the newly prepared controls are separate. The analysis reuses audited predictions. The full control plan contains **550 cases, 1,650 primary requests and 64 exact no-proposal repeats: 1,714 planned requests**, with no measured control outcomes yet.

Run these commands from the repository root with its Python environment. The frozen numerical and text datasets must already exist; preparation and exact-hash restoration are documented in [numerical reproduction](EXPANDED_NUMERIC_REPRODUCTION.md) and [text reproduction](TEXT_EXTENSION_REPRODUCTION.md). All commands below are offline with respect to model providers: they make no inference calls and need no API keys.

## Recompute the existing-results analysis

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/analyze_review_value.py
```

This audits prior evidence and writes [ANALYSIS.json](../results/review_value/ANALYSIS.json) and [FINDINGS.md](../results/review_value/FINDINGS.md). It does not fill missing reviews or create new measurements. The matched-prompt comparisons overlap and are not independent repeatability trials; selective-review curves are retrospective simulations rather than validated deployment policies.

## Restore the ignored request payload after cloning

The frozen [protocol](../results/review_controls/prepared_full/protocol.json) and [manifest](../results/review_controls/prepared_full/manifest.json) are versioned. Their 7.58 MB `requests.jsonl` repeats public inputs and training examples and is intentionally ignored by Git.

```sh
.venv/bin/python scripts/restore_review_controls_payload.py
```

The restoration helper audits the original Qwen3 source predictions and datasets, reconstructs the complete protocol and request sequence, and requires exact equality with the retained protocol and its pinned request hash. It stages the verified payload and publishes it with an exclusive atomic filesystem link. It never replaces an existing payload. An already valid file is audited without modification; partial files, unexpected directory contents, changed source data, changed hashes or symlinks cause an error.

There is no need to rerun a model or overwrite the frozen preparation directory. `--plan PATH` can select another separately frozen plan; it does not relax verification. A failed restoration should be investigated, not resolved by deleting or rewriting the protocol to fit different inputs.

## Verify the plan and current completion status

```sh
.venv/bin/python scripts/prepare_review_controls.py --verify results/review_controls/prepared_full
.venv/bin/python scripts/summarize_review_controls.py
.venv/bin/python scripts/run_review_controls.py
```

The first command rebuilds and compares every request, including the shuffled donor mapping, one-slot prompt controls, exact repeat pairs and label-blind execution order. The second writes the control study's [comparison](../results/review_controls/COMPARISON.json) and [findings](../results/review_controls/FINDINGS.md), retaining unavailable metrics for incomplete arms. The final command is the runner's **dry run** and reports readiness without creating a ledger or making model requests.

The request payload contains no per-case test truth. Source identities and evidence hashes remain in the protocol; only the original input, allowed classes, shared labeled training examples and proposal slot enter reviewer prompts. See the immutable [control protocol](REVIEW_CONTROLS_PROTOCOL.md) for the primary contrasts and repeatability diagnostic.

## Funding and execution remain separate

Restoration, verification and analysis allocate **US$0** and send **zero model requests**. Preparing 1,714 requests does not establish funding. Paid execution remains gated on available provider credit, completion and audit of the earlier review studies, and an explicit guarded allocation within the existing **US$25 cumulative authorization**. Historical reservations and protected allocations are not repriced by these commands.

Until the new controls execute and pass their audits, there is no measured actual-versus-no-proposal, actual-versus-shuffled, or serving-repeat result from this new experiment. Existing direct-Jev results are contextual references because their prompt wording differs from the exact no-proposal control.

## Aggregate dashboard and sharing figure

After regenerating and auditing the reports, run:

```bash
python scripts/build_review_value_dashboard.py
node dashboard/tests/review-dashboard.smoke.cjs dashboard
python -m pip install 'matplotlib==3.11.2'  # optional plotting dependency
python scripts/plot_review_value.py
```

The exporter copies explicit aggregate fields only: no prompts, row IDs, source inputs or API credentials. The figure includes every eligible condition, not only the two encouraging curves, and writes its source/data hashes alongside PNG and SVG outputs. Rendering uses the saved audited dashboard snapshot and makes no model calls. The report, figure and LinkedIn draft distinguish measured results, retrospective simulations and unexecuted controls.

## Constant-label sensitivity appendix

Run `python scripts/analyze_review_sensitivity.py` to regenerate the separate nine-condition appendix. It changes only the primary analysis's constant-label exclusion, preserves the same six coverage points and random reference, and verifies the four original curves and every full-review endpoint. It sends no model calls and does not modify the primary dashboard or report. All nine conditions are reported; no winning coverage is selected.

## Completed validation scoring diagnostic

The [audited findings](../results/review_value/validation_scoring/AUDITED_FINDINGS.md) compare ID-only and ID+EOS scoring on 34 distinct balanced validation cases, two models and two shot settings. All 136 contexts completed locally on MPS without APIs or downloads; no class decisions changed. This is a separate diagnostic development sample, not new test performance.

Reaudit the stored results without models, tokenizers or a GPU:

```bash
python scripts/audit_validation_scoring_ablation.py --output results/review_value/validation_scoring
```

The auditor verifies the complete ordered predictions, frozen protocol and producer hashes, numerical decomposition, metrics and raw summary before writing separate `AUDITED_SUMMARY.json` and `AUDITED_FINDINGS.md`. Inference package versions remain bound to the original protocol; the auditor can run on a different CPU environment and must still reproduce the saved classification metrics exactly. Numeric latency fields are omitted because they were not measured.

The original inference runner remains an MPS/cache-only frozen diagnostic: `python scripts/run_validation_scoring_ablation.py` shows a read-only plan; `--freeze` creates a new protocol only where absent; `--execute` audits the existing frozen protocol and uses the exact cached snapshots. Preserve the published completed folder. A different device or recipe requires a separately declared experiment, not an overwrite or a replacement test score. The broader text and free-generation ablations mentioned in the earlier source audit were proposals and were not executed here.
