# Reproduce the text extension

Read the [protocol](TEXT_EXTENSION_PROTOCOL.md) first. Reproduction uses the frozen SST-2/TREC pilot splits and saved source predictions. Keep those records intact, including unsuccessful outcomes. The completed extension adds eight local source conditions, sixteen classical conditions and twenty-four Jev review conditions; it adds no LoRA training.

## Environment and frozen data

From the repository root with Python 3.11–3.13:

```bash
python -m pip install -e '.[dev,numeric,neural]' \
  'transformers==4.57.6' 'huggingface-hub==0.36.2'
python -m pytest -q tests/test_text_extension_local.py
python scripts/run_text_extension_local.py
```

The last command only prints a plan: 2 datasets × 2 new models × 2 shot settings × 200 rows = **1,600 predictions**. It does not load weights, download files or call a hosted API. Exact classical package versions are separately pinned in [the classical configuration](../configs/text_classical_extension.json); use those versions when reproducing its fits.

The prepared `data/pilot/{sst2,trec}` files are ignored by Git and must come from the original audited preparation or a verified bundle containing those exact bytes. The text registry and runners reject changed content. A fresh download/preparation is not automatically equivalent. Historical Qwen/Luna/Astra/direct-Jev artifacts are verified against their saved hashes; new source artifacts are verified through the wrapper's read-only `audit_source_run(path, dataset, expected_key, shots)`.

### Fresh CI preparation without a release download

On a fresh checkout where `data/pilot` is absent, the pinned public sources can recreate the exact six training/validation/test JSONL files:

```bash
jevbench prepare sst2 trec --cache-dir data --output data/pilot \
  --seed 42 --train-limit 10000 --validation-limit 1000 \
  --test-limit 200 --max-text-chars 2000
python scripts/restore_text_extension_data.py --candidate-root data/pilot
```

The current preparation adds two manifest fields that were absent from the historical pilot: `post_transform_overlap_removed_ids` and `splits_before_text_transform`. The restoration helper accepts only empty removed-ID lists for train/validation and a before-transform split description exactly equal to the historical `splits`. It verifies **both datasets and all six JSONL byte hashes before writing**, checks the pinned historical Qwen-small zero-shot run, and requires its encoded manifest to match the original manifest's exact SHA256. Only then does it restore those historical manifest bytes under an exclusive directory lock and atomic per-file replacement. It downloads nothing and never edits source runs, predictions or row files.

Already frozen manifests are accepted without rewriting. Any row difference, additional metadata difference, nonempty overlap removal, changed source artifact or mismatching manifest encoding fails closed. For an independent verification use a new candidate directory in both commands; do not prepare over existing experiment evidence. After the strict check, the resulting dataset files have the original pinned bytes and can be used by CI and the normal source auditors without a GitHub release dependency.

## Colab source inference

Build the allowlisted source/data bundle:

```bash
python scripts/export_text_extension_colab.py
```

Open [colab_text_extension.ipynb](../notebooks/colab_text_extension.ipynb), select a CUDA GPU runtime, and upload `artifacts/jev-text-extension-colab.zip`. If the inline upload widget is unavailable, use Colab's **Files sidebar** to upload it to `/content/jev-text-extension-colab.zip`; the notebook prefers that existing file and applies the same verification. The notebook pins the archive SHA256, verifies every member before extraction into a fresh directory, installs Transformers 4.57.6 and Hugging Face Hub 0.36.2, and runs offline tests before inference. It never reads secrets or mounts Drive. Weights are public and downloaded with `token=False` and implicit Hugging Face authentication disabled.

The frozen bundle SHA256 is `30620cce39a27521551e76a73db30228ad9b4e116d37c87e86a678183884d80d`; its text runner SHA256 is `d3f6c06f6138da02a38e49ab220ed3a184c047b3b4a4729eb01908e513334526`. A changed bundle must be inspected and separately pinned before execution; do not bypass a mismatch. The notebook is deliberately excluded from the bundle to avoid a circular archive-hash dependency.

The actual inference command, run by the notebook, is:

```bash
python scripts/run_text_extension_local.py --device cuda \
  --model-keys smollm2 granite --datasets sst2 trec --shots 0 4 \
  --allow-download --execute
```

For authorized local inference, use `--device mps` on a supported Mac or `--device cuda` on a CUDA machine. Models run sequentially under the shared device lock. There is no fallback after memory/device/context errors. `--preflight-only` loads the selected models and verifies all prompt lengths without predicting. Omit `--allow-download` when the exact public snapshots are already cached. Full-precision recipes, adapters and prompt truncation are not alternatives within the same run identity.

The unchanged scorer computes numeric-ID-plus-EOS likelihoods. Saved artifacts include the exact revision, renderer/tokenizer/template hashes, per-row rendered prompt identities, class probabilities, package/runtime information and all failures. An interrupted condition resumes its original saved rows when the identical command is repeated in the same project. Do not rerun the notebook upload cell when retaining that directory.

After all eight conditions complete, the notebook audits and downloads a checksummed result ZIP. Import it locally using the SHA256 printed by Colab:

```bash
python scripts/import_text_extension_colab.py /path/to/jev-text-extension-results.zip \
  --expected-sha256 REPLACE_WITH_PRINTED_ARCHIVE_SHA256 --execute
```

The completed eight-condition result archive has SHA256 `01e80b38f8ddc8498c5d2ccf3a1c404ef1473330a901a3944aa1c7c54fdcae61`. Its [import receipt](../results/text_extension/imports/01e80b38f8ddc8498c5d2ccf3a1c404ef1473330a901a3944aa1c7c54fdcae61.json) records all 28 member hashes and 1,600 imported predictions. The recorded runtime used a Tesla T4, float16, PyTorch 2.11.0+cu128, Transformers 4.57.6 and Hugging Face Hub 0.36.2. No hosted requests or adapter training were used.

Omit `--execute` for verification only. The importer checks the embedded result manifest, source/data pins, complete condition inventory, every file hash and the raw predictions; it refuses to replace differing evidence. Results go to `results/text_extension/local` with an import receipt. To transfer one fully completed model first, export with `python scripts/export_text_extension_colab.py --results --model-keys smollm2` (or `granite`) and use the matching importer flag. Partial conditions are never exported as complete.

## Classical references

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/run_text_classical_extension.py
```

This fixed runner executes or audits/reuses all sixteen conditions. Its train-only word/character TF-IDF pipelines use matched 8/24 labels and separate full-prepared-training references of 10,000/4,886 labels. Validation labels are unused. Preserve the protocol, configuration, source hashes, matrices and saved warnings; do not change hyperparameters after examining test results. See [classical audit notes](../results/text_extension/classical/README.md), including XGBoost's treatment of unstored sparse entries as missing.

## Jev review and aggregate reports

The first-attempt matrix is complete: **68/68 conditions and 72/72 contrasts**, including all twenty-four Jev review conditions. Completed conditions retain failed predictions in the full test denominator; completion does not mean every request succeeded. Subsequent failed-call recovery has separate evidence and metrics in the [completion analysis](COMPLETION_REPRODUCTION.md), and does not replace these first-attempt scores.

The [budget guide](BUDGET.md) records the original US$1.60 stage allocation and US$25 cumulative ceiling; the [completion protocol](COMPLETION_RETRY_PROTOCOL.md) documents the later execution and recovery policy. Historical source calls were reused, not charged again by this stage. Their recorded costs are not zero-cost estimates for running the complete pipeline afresh. Rebuilding the saved analysis below needs neither API credentials nor provider credit.

After restoring the frozen datasets, refresh the report before the dashboard asset:

```bash
python scripts/summarize_text_extension.py --bootstrap-samples 2000
python scripts/build_text_extension_dashboard.py
node dashboard/tests/text-dashboard.smoke.cjs dashboard
```

The dashboard builder independently reaudits the report and requires the completed matrix. Missing scores/intervals remain unavailable, and metrics are never calculated from partial checkpoints. `--allow-incomplete` is reserved for explicitly labeled interim snapshots; it is unnecessary for this completed study. Source and inference producers remain frozen when regenerating summaries. For a fresh checkout covering the numerical results, proposal controls and recovery as well, follow the exact restoration and refresh order in the [completion reproduction guide](COMPLETION_REPRODUCTION.md).

Serve the repository mirror with `python3 -m http.server 8766 --bind 127.0.0.1 --directory dashboard/dist` and open `/text.html`. The [hosted text view](https://jev-benchmark-observatory.statsguysalim.chatgpt.site/text.html) and linked repository remain private to their owning account. Building an asset or notebook does not publish the site or make these results public.
