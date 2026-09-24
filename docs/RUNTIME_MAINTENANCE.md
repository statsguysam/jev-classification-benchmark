# Current code and the saved study

The current `jevbench` package includes fixes made after the benchmark finished.
Those fixes should be used for new work. The saved study still needs its original
code: its run identities, audit records and request plans pin that code by hash.
Changing those pins would destroy the link between the records and what ran.

The original files are therefore archived at commit
`324d63329e6ee6f98bcf5c4e00ca0256b74e0e64` in
[`reproducibility/`](../reproducibility/README.md). The launcher verifies the
archive and runs it in a separate checkout and process. Its reported historical
core hash comes from the archived files actually loaded. The active package's
hash comes from the current files; the two identities stay separate.

New runs pin the actual test rows, selected training rows and any validation rows
used for model selection. A completed run is rescored before its cached result is
reused. Paired comparisons also verify the manifests, prediction IDs and saved
metrics, and check training content as well as row IDs. These checks reject an
incomplete or altered record instead of silently treating it as a valid result.

## Check the current package and the original study

Use the project environment and dependency versions in the
[completion guide](COMPLETION_REPRODUCTION.md). These commands need the prepared
datasets already present:

```bash
# Current implementation and regression tests.
python -m pytest -q tests/current

# Every original test, using the archived implementation.
python scripts/reproduce_frozen_study.py test

# Rebuild and compare eight saved machine-readable reports.
python scripts/reproduce_frozen_study.py audit
```

The default `python -m pytest` runs the current tests and a bridge to the complete
original test suite. CI runs the two suites as separate steps so both counts are
visible. No historical tests are skipped merely because the active code changed.
Optional-dependency skips in the original suite keep their original conditions.

The archived audit covers the numerical and text comparisons, review-value
analysis, proposal controls, both recovery analyses, final costs and validation
scoring diagnostic. Each rebuilt JSON must match the saved bytes. It writes only
inside its temporary checkout, which is removed afterward. It makes no model
calls and does not need API credentials or a GPU.

## Prepare data on a fresh checkout

The dataset files are ignored by Git. This command uses the archived preparation
code, downloads the pinned public sources and restores the exact historical
manifest bytes before exporting the five prepared datasets:

```bash
python scripts/reproduce_frozen_study.py prepare-data --export-prepared data
```

This is the only launcher mode that permits public dataset downloads. It makes
no model requests. It refuses to replace an existing prepared dataset. If the
prepared data already exists, use it directly for the test and audit commands.
The ignored proposal-control request payload is restored inside the audit
checkout from the prepared rows and saved source predictions.

## Run an individual historical command

For inspecting or extending an offline audit, create a persistent checkout:

```bash
python scripts/reproduce_frozen_study.py stage \
  --workspace artifacts/frozen-study-324d633
cd artifacts/frozen-study-324d633
export PYTHONPATH="$PWD/src:$PWD/scripts"
python scripts/summarize_review_controls.py
```

Keep the original project environment activated. The workspace contains copies
of the results and prepared data, so changing a copied file cannot change the
source evidence. `FROZEN_RUNTIME.json` records the archive, launcher and core
identities. Historical reproduction commands elsewhere in this repository must
be run from this workspace, with the import path above. Installing the archived
package into the active environment is unnecessary.

The launcher itself exposes only verification, tests, offline audits, dataset
preparation and dashboard export. It does not launch inference, retries, training
or paid workers. The archived dated execution guards are unchanged.

## Check current dashboard code against frozen evidence

Dashboard refactoring does not require changing the scientific analysis code.
A separate launcher mode copies the current numerical/text exporters, their
shared helper and tests, and the current frontend files into the frozen checkout.
The list of allowed overlay paths is explicit in the launcher. Its receipt
records the SHA-256 of each current file used.

```bash
python scripts/reproduce_frozen_study.py current-dashboard
```

This runs the exporter tests, rebuilds all four aggregate JSON assets, runs the
four current JavaScript smoke suites and compares the generated assets with
`dashboard/dist`. Original predictions and reports remain unchanged.

An exporter change can legitimately change the exporter hash in an aggregate
asset. To inspect the new derived files before replacing repository snapshots:

```bash
python scripts/reproduce_frozen_study.py current-dashboard \
  --export-dashboard /tmp/jev-dashboard-review
```

The destination must be new. It receives only four aggregate JSON files and a
runtime receipt. Nothing is copied back into the repository automatically. Check
the differences before updating `dashboard/dist`; changing a source hash is not
permission to change a measured value.

## Use the portable Colab source bundle

`python scripts/export_colab_bundle.py` packages the maintained implementation
and its standalone tests. Its exported pytest settings run only those tests.
The frozen runtime, historical test suite and saved-evidence regression tests
stay in the full repository because they need the archive or original records.
No data, model weights or raw result directories enter the portable bundle.

CI extracts the bundle into a separate directory, verifies its manifest, checks
that imports resolve to the extracted source and runs its default pytest command.
Notebook outputs and local metadata are removed during export. These checks
exercise the portable package; they do not replace the full study audit above.
