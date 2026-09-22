# Reproduce the two-model numerical expansion

[colab_numeric_expansion.ipynb](../notebooks/colab_numeric_expansion.ipynb) runs the eight new SmolLM2/Granite conditions on CUDA: two numerical datasets, zero-shot and four examples per class, 600 test predictions in total. It contains no saved outputs, credentials or hosted API calls. Use a fresh GPU runtime; the observed run used a Tesla T4. Model revisions, float16 precision, renderer and likelihood scorer are fixed. This notebook does not run Jev reviews or additional adaptation.

From the repository, install its dependencies and prepare the exact public data if the prepared directories are absent. Preparation is version-sensitive; the original manifests used scikit-learn 1.9.1.

```bash
python -m pip install -e '.[dev,neural]' 'scikit-learn==1.9.1' 'transformers==4.57.6' 'huggingface-hub==0.36.2'
python scripts/tabular_data.py --datasets breast_cancer wine --output data/tabular-full --seed 42
python scripts/export_expanded_numeric_colab.py
```

Open the notebook in Colab, select a CUDA GPU, and run its cells in order. Upload `artifacts/jev-numeric-expansion-colab.zip`. The notebook requires this source ZIP SHA256:

```text
f09e8181b5c6238cdcb13a842e57857b161446ff800377ab782d88fe405bd405
```

It verifies the entire archive and each manifest member before extracting into a fresh directory. A hash mismatch stops execution; inspect differences rather than bypassing the pin. Prepared split IDs, numerical serialization and demonstration selection must match the published evidence exactly.

The notebook installs Transformers 4.57.6 and huggingface-hub 0.36.2 and retains the runtime's compatible CUDA PyTorch. It records actual Python/package versions, GPU identity, source hashes and invocation in the exported runtime metadata. An optional incompatible Colab `torchao` installation is removed only if the model-class import smoke check identifies it as the failure; this experiment uses ordinary float16 inference. Tests use a tiny synthetic model, without downloading real weights.

The inference command is:

```bash
python scripts/run_expanded_numeric_local.py --device cuda \
  --model-keys smollm2 granite --shots 0 4 \
  --datasets breast_cancer wine --allow-download --execute
```

`--allow-download` caches only pinned public Hugging Face snapshots, explicitly passing `token=False`; no Hugging Face secret is needed. All full prompts plus candidate labels are checked before each model's inference. Context overflow and unavailable devices stop execution, and out-of-memory errors do not trigger an automatic precision/device change. The models run sequentially and release memory between families. Without `--execute` or `--preflight-only`, the helper prints a plan; downloads still occur if `--allow-download` is explicitly included.

The final cell audits all eight completed conditions, then downloads `jev-numeric-expansion-results.zip`. Only `run.json`, `predictions.jsonl`, `test_manifest.json` and local preflight/runtime metadata are included. Import it from the repository:

```bash
.venv/bin/python scripts/import_expanded_numeric_colab.py /path/to/jev-numeric-expansion-results.zip --execute
```

Omit `--execute` for verification only. The importer verifies all expected conditions, immutable model/config/data identities, rendered prompt hashes, probabilities and saved metrics. It refuses to replace differing evidence and stores the archive/member hashes in `results/numeric_expansion/imports/`. An early complete SmolLM2-only archive can be imported with `--model-keys smollm2`; it must contain that model's four complete conditions and no Granite run artifacts.

Within an uninterrupted Colab runtime, rerunning the inference cell resumes saved predictions. Rerunning the upload cell creates a fresh directory, so retain the existing project directory if resuming. CUDA kernels and different hardware/software versions need not produce bit-identical numerical results; provenance remains recorded. These runs are an exploratory model expansion on one fixed split, as described in the [protocol](EXPANDED_NUMERIC_PROTOCOL.md).
