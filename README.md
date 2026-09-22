# Jev classification benchmark

A reproducible comparison of Jev, open-weight language models, hosted frontier models, LoRA adaptation, and classical machine learning on public text-classification datasets.

**Status:** runnable benchmark with measured local pilot results; the full Jev/frontier matrix requires API credentials and a spending limit. The repository never substitutes invented scores for missing experiments. See [recorded results](results/REPORT.md), the [experimental protocol](docs/PROTOCOL.md), and [primary sources](docs/SOURCES.md).

## Study design

| Dataset | Task | Classes |
|---|---|---:|
| SST-2 | Movie-review sentiment | 2 |
| IMDb | Movie-review sentiment | 2 |
| AG News | News topic | 4 |
| TREC | Question type | 6 |
| Banking77 | Banking intent | 77 |

SST-2's labeled validation set is reserved as final test; its published test labels are hidden. Development data comes only from training data. Every method uses the same frozen held-out IDs, label mapping and text preprocessing. Dataset snapshots and prepared splits are hashed. Exact cross-split text overlaps are removed from the training/development side, preserving test rows.

The default pilot uses up to 10,000 training rows, 1,000 development rows and 200 test rows, with the first 2,000 characters of each text for **all** families. These caps are declared pilot choices; results do not represent a full official-test-set evaluation. Banking77's 200-row pilot is especially noisy.

Two separate comparisons are necessary:

* **Equal labeled data:** 1, 4 or 8 examples **per class**, selected using seeds 13, 42 and 87. Few-shot prompting, LoRA and classical classifiers get exactly the same examples. No development labels are used in this track.
* **Larger supervised training:** classical ML and LoRA use the full prepared training split. Classical hyperparameters are selected on development data; that extra label use is reported explicitly. These results are kept distinct from few-shot performance.

Zero-shot is an additional reference. LoRA applies to downloadable open weights; Jev and proprietary hosted models have no LoRA arm here. Local LLMs use closed-label likelihood scoring, while hosted LLMs generate a label and Jev returns a native Choice. Reports label these different output protocols.

## Install and verify

Python 3.11–3.13 is required. From this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

The base installation skips tests that require optional neural libraries. Install
the neural extra and rerun the tests to include the local tensor/adapter checks;
those tests use tiny models and do not download weights or make hosted calls.

Optional neural training/inference dependencies:

```bash
python -m pip install -e '.[neural]'
# CUDA QLoRA only:
python -m pip install -e '.[qlora]'
```

[requirements-macos.lock.txt](requirements-macos.lock.txt) records the exact local environment used for the pilot. It is an environment capture, not a promise that these binary packages fit every Colab/CUDA runtime. Every run records its installed versions and implementation hash.

## Prepare shared pilot data

```bash
jevbench prepare sst2 imdb ag_news trec banking77 \
  --output data/pilot --seed 42 --train-limit 10000 --validation-limit 1000 \
  --test-limit 200 --max-text-chars 2000
```

For all public labeled held-out rows, prepare a **separate** directory and omit the three size limits. SST-2 still uses validation-as-test. Do not overwrite an existing experiment's prepared splits. Raw text, model caches, credentials and adapters stay out of Git.

## Run the classical baselines

The baselines are majority class, word+character TF-IDF with logistic regression, linear SVM, and multinomial Naive Bayes. Vocabulary fitting never sees development/test text. SVM margins are not misrepresented as probabilities.

```bash
python scripts/run_classical_matrix.py --data-root data/pilot \
  --output results/pilot --budgets 4 --seeds 42 --include-full

# Planned larger matched-label study:
python scripts/run_classical_matrix.py --data-root data/pilot \
  --output results/matched --budgets 1 4 8 --seeds 13 42 87
```

## Run zero-shot and few-shot models

[configs/models.json](configs/models.json) contains provider configurations and pinned open-model revisions. Provider access is described in [MODEL_ACCESS.md](docs/MODEL_ACCESS.md). Set secrets in environment variables or Colab secrets; `.env.example` is only a template and is not loaded automatically.

```bash
jevbench model --data data/pilot/sst2 --config configs/models.json \
  --model-key qwen_small --shots 0 --output results/pilot --max-requests 200

jevbench model --data data/pilot/sst2 --config configs/models.json \
  --model-key qwen_small --shots 4 --output results/pilot --max-requests 200

# Hosted requests are opt-in and may incur charges:
jevbench model --data data/pilot/sst2 --config configs/models.json \
  --model-key jev --shots 4 --output results/hosted \
  --allow-paid --max-requests 200
```

Hosted runs have no automatic retries and stop after three consecutive errors. `--max-requests` limits the number of new test requests in that invocation, **not dollars**. Check provider pricing and authorize a total budget before executing a matrix. Incomplete rows are checkpointed; repeating the identical command resumes. Model input/context errors are failures, not reasons to drop inconvenient test examples.

Plan a multi-model matrix without sending model requests:

```bash
python scripts/run_model_matrix.py --data-root data/pilot \
  --model-keys jev openai_economical anthropic gemini \
  --shots 0 4 --seeds 42 --output results/hosted
```

With four core datasets and 200 heldout rows each, that explicit configuration plans **32 runs and at most 6,400 test requests**. A saved [hosted plan](results/HOSTED_PLAN.json) is a request inventory, not measured results; hosted experiments remain pending. Zero-shot runs once at seed 42; few-shot runs use each requested seed. The dry-run script does not load model weights or call model APIs. To execute after approving spending, add `--execute --allow-paid --max-total-requests 6400`. This cap counts requests, not dollars, and conservatively includes cached jobs in its upper bound. Use the single-run CLI for dataset-specific adapters. `configs/experiment.json` is a design inventory and is not automatically applied by either driver.

## LoRA and Colab

Open [notebooks/colab_benchmark.ipynb](notebooks/colab_benchmark.ipynb) in Colab and select a GPU runtime. For this private repository, run `python scripts/export_colab_bundle.py` locally and upload `artifacts/jev-classification-benchmark-colab.zip` through the notebook. It verifies the source checksums, extracts into a fresh directory, and installs the same package/CLI. The source bundle excludes datasets, results, weights, environment files, and saved notebook outputs; review source code for manually pasted secrets before sharing it. Included README links to measured results become usable only after you generate or separately supply those results.

The notebook runs local weights only, with no hosted API calls. Colab compute quotas and charges depend on your account. Start with `qwen_small` to check the pipeline if GPU memory or session time is limited; this is a technical smoke model, not the main 4B comparison. Download the notebook's allowlisted results ZIP before the runtime expires.

```bash
jevbench train-lora --data data/pilot/sst2 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --train-per-class 4 --seed 42 --epochs 3 --load-in-4bit \
  --output models/qwen3-4b-sst2-k4-s42

jevbench model --data data/pilot/sst2 --config configs/models.json \
  --model-key qwen_lora --seed 42 --output results/lora --max-requests 200
```

`--load-in-4bit` requires CUDA. Ordinary LoRA supports local CPU/MPS. Training supervises only the class-ID response and EOS, masks prompt/padding tokens, uses a fixed final checkpoint, and does not read test labels. Adapter metadata records selected training IDs, revision, prompt hash, hyperparameters and training time. Evaluation checks this provenance.

Four-bit training is **QLoRA**. Evaluation loads the base at its normal configured inference precision and applies the adapter; quantization of training and inference therefore differ. Untuned and adapted evaluations use the same scorer and inference precision. A 4B full-weight evaluation may still exceed a small GPU's memory even when QLoRA training fits. The checkpoint directory must be empty for new training; use a new path for a changed recipe.

## Reports and statistical comparisons

```bash
jevbench report --results results/pilot --output results/REPORT.md
jevbench compare results/pilot/RUN_A results/pilot/RUN_B --samples 2000
```

The report and its underlying run JSON include accuracy, macro-F1, balanced accuracy, per-class metrics, failure rate, bootstrap intervals, timing and token coverage. NLL, multiclass Brier, ROC-AUC and ECE require a valid class distribution: native Jev/classical probabilities or the local model's normalized restricted-label likelihoods. Their semantics differ. Self-reported LLM confidence is never substituted. Per-class details and reliability-bin counts are in each run JSON.

Paired bootstrap compares exactly aligned test examples. Unequal training budgets/seeds require `--allow-unequal-training` and remain descriptive. Probability methods differ across native Jev, classical classifiers and restricted-label LLM likelihoods. Hosted request latency and classical amortized batch latency are recorded separately; neither establishes equal-hardware throughput.

Each run saves `run.json`, per-example `predictions.jsonl` and a text-free `test_manifest.json`. Public-corpus contamination of pretrained models cannot be ruled out, so this study does not establish performance on unseen future data. See the protocol for the full limitations and remaining publication-scale experiments.

## Publication

Publish code, configurations, notebook, dependency capture, dataset provenance and measured predictions/metrics. Do not publish credentials, dataset text, downloaded base weights or unreviewed licensed artifacts. The dataset/model licenses are independent of this repository; citations and source terms are in [SOURCES.md](docs/SOURCES.md).
