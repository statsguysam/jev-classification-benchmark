# Release adapter packages

Adapter ZIPs contain trained LoRA weights, the PEFT configuration, optional saved tokenizer files,
training provenance, per-file SHA-256 hashes, upstream license provenance and
downstream modification notices. Pretrained base weights and raw datasets are
excluded. These are benchmark adaptations, not official Qwen releases. Evaluation
reloads the tokenizer from the pinned base snapshot; the tabular 4B packages
contain only adapter weights, configuration and training provenance, plus release notices.

The packager supports only these immutable Apache-2.0 base snapshots:

| Model | Base revision |
|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Qwen/Qwen3-4B-Instruct-2507 | `cdbee75f17c01a7cc42f958dc650907174af0554` |

The pinned [Qwen2.5 license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE)
and [Qwen3 license](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/LICENSE)
were verified on September 22, 2026. Both have SHA-256
`832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`.
Neither pinned snapshot lists a separate upstream NOTICE file. The packager
supports copying such notices for any future explicitly reviewed snapshot.

## Evaluate downloaded 4B adapters

Extract the individual adapter folders beneath the repository's `models/`
directory. Preserve their names, including the dataset suffix. Verify every
payload against `MANIFEST.json` before use. Prepare the identical dataset snapshot
and settings used in the original run; do not edit provenance to make an
incompatible split pass validation. See the protocol for the exact 200-row pilot
split and character-prefix settings.

```bash
python -m jevbench.cli model --config configs/colab_adapters.json \
  --model-key qwen_main_sst2_lora --data data/pilot/sst2 \
  --output results/reproduced --shots 0 --seed 42 --max-requests 200

python -m jevbench.cli model --config configs/colab_adapters.json \
  --model-key qwen_main_trec_lora --data data/pilot/trec \
  --output results/reproduced --shots 0 --seed 42 --max-requests 200
```

The loader checks the base revision, dataset, label order, seed, selected training
rows and prompt hash. QLoRA training does not imply quantized evaluation: this
evaluation path loads the base model at its configured dtype and can require
substantial GPU memory. Results across devices and package versions can differ.

## Package adapters after downloading Colab artifacts

Pinned upstream LICENSE files must already exist at
`artifacts/upstream/Qwen2.5-0.5B-Instruct/LICENSE` and
`artifacts/upstream/Qwen3-4B-Instruct-2507/LICENSE`. These local source licenses
and their verification records are ignored Git artifacts. The packager makes no
network requests and checks their expected hashes.

```bash
python scripts/package_adapters.py \
  models/qwen05-sst2-k4-s42 models/qwen05-trec-k4-s42 \
  models/Qwen3-4B-Instruct-2507-sst2-k4-s42 \
  models/Qwen3-4B-Instruct-2507-trec-k4-s42 \
  --upstream-root artifacts/upstream \
  --output artifacts/qwen-pilot-adapters.zip
```

The command fails if a requested adapter has not been downloaded. It also rejects
unexpected files, links, unsupported bases/revisions, full-module weights and
safetensors containing anything other than expected LoRA A/B tensors.
Recognizable credential fields and key patterns in metadata are rejected; review
manually edited prose before publication. No automated scanner can establish the
absence of every possible secret. The manifest hashes all payload files except
the manifest itself, and publication of the ZIP is a separate step.

## Tabular adapters

The tabular extension has six adaptations: Titanic, Breast Cancer and Wine for
both base models. They use four training examples per class, three fixed epochs,
and the identical serialized features documented in [TABULAR_PROTOCOL.md](TABULAR_PROTOCOL.md).
The 0.5B model uses ordinary LoRA on MPS; the 4B model uses QLoRA training on a
Colab T4 and normal base precision for evaluation. No validation labels select
these checkpoints.

The `qwen-tabular-adapters.zip` release asset uses distinct `*-lora` and `*-qlora`
folder names. Verify its SHA-256 and internal `MANIFEST.json`, then extract it
under `models/tabular-release/`. [tabular_adapters.json](../configs/tabular_adapters.json)
maps all six checkpoints to their immutable base revisions. For example:

```bash
python scripts/tabular_data.py --output data/tabular-full
python -m jevbench.cli model --config configs/tabular_adapters.json \
  --model-key qwen_main_titanic_qlora --data data/tabular-full/titanic \
  --output results/reproduced-tabular --shots 0 --seed 42 --max-requests 262
```

The tabular source/data bundle already contains the exact prepared splits used
for the release. Consuming those verified splits avoids preparation-version
differences; regenerate them with the recorded scikit-learn version if needed.
The clean [tabular notebook](../notebooks/colab_tabular_benchmark.ipynb) documents
training and result export. Its transfer helper preserves original Colab paths
inside provenance rather than rewriting run identities during import.

Package newly trained tabular adapters with the same license-aware packager:

```bash
python scripts/package_adapters.py \
  models/tabular/Qwen2.5-0.5B-Instruct/{titanic,breast_cancer,wine}-k4-s42-lora \
  models/tabular/Qwen3-4B-Instruct-2507/{titanic,breast_cancer,wine}-k4-s42-qlora \
  --upstream-root artifacts/upstream --output artifacts/qwen-tabular-adapters.zip
```
