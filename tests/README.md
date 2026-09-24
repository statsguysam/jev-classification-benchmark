# Test layout

`current/` tests the maintained Python package, including provider responses,
checkpoint recovery, scoring and archive isolation. Run it with:

```bash
python -m pytest -q tests/current
```

The original study tests depend on exact historical source hashes. Their complete
version is stored in the verified source archive and runs through
`test_frozen_suite.py`, rather than importing a different implementation under an
old identity. The default `python -m pytest -q` runs both suites. CI reports each
suite separately.

The study-specific tests at this directory's top level remain available for
reading and targeted audit development. Run historical versions from a frozen
workspace, as described in the [runtime guide](../docs/RUNTIME_MAINTENANCE.md).
The current numerical/text exporter tests and shared aggregation tests run through
`python scripts/reproduce_frozen_study.py current-dashboard`, which combines the
maintained exporters with the original scientific analysis code.
