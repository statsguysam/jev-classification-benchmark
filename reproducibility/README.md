# Frozen study runtime

`study-324d633.zip` contains 216 files taken directly from Git commit
`324d63329e6ee6f98bcf5c4e00ca0256b74e0e64`. It preserves the code, tests,
configuration and supporting files used before the active package was repaired.
The `results/` and `data/` directories, model weights and credentials are excluded.
Original aggregate dashboard assets are included for the archived smoke tests.

The accompanying manifest records each file's SHA-256, Git blob ID, size and mode,
plus the archive SHA-256 and original core digest. ZIP entries are sorted, use a
fixed 1980 timestamp and DEFLATE level 9. The payloads are original Git blobs,
not files copied from a working directory.

The active package remains `src/jevbench`. Do not install the archived package
into the active environment or copy its code over current source files. Run it
through the launcher:

```bash
python scripts/reproduce_frozen_study.py verify
python scripts/reproduce_frozen_study.py test
python scripts/reproduce_frozen_study.py audit
```

The launcher verifies the pinned manifest and archive before extracting anything.
It rejects unsafe paths and non-regular files, checks every original Git blob,
and confirms the imported package's actual source digest. Saved results and
prepared data are copied into a disposable checkout. Audits run there in a new
Python process, so current and historical modules cannot share import state.
The existing historical guards remain active; no hash or saved prediction is
rewritten to make a check pass.

Use [the runtime guide](../docs/RUNTIME_MAINTENANCE.md) for fresh-checkout data
preparation, a persistent audit workspace and current dashboard verification.
