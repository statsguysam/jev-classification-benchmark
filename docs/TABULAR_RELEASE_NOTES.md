# Completed serialized tabular benchmark

This release adds **Titanic3, Breast Cancer Wisconsin Diagnostic and Wine** to the existing text study. All **66 planned conditions, 9,064 predictions and 48 paired contrasts** are complete and audited. Four transport failures remain in the scores. Earlier text results and release assets are preserved.

The comparison includes Jev 1.13, GPT-5.6 Luna, GPT-6 Astra, Qwen2.5-0.5B, Qwen3-4B and five native classical estimators. Open models have zero-shot, four-examples-per-class and LoRA/QLoRA arms; hosted models have zero/few-shot arms. Native classifiers use matched labels and separate full-training references. All methods share held-out IDs; matched arms share exactly the same training rows. Numerical features are serialized without loss for LLMs.

Jev few-shot accuracy is **74.4% on Titanic, 93.0% on Breast Cancer and 91.7% on Wine**. Astra few-shot reaches 85.9%, 98.2% and 97.2%; matched native logistic regression reaches 64.1%, 94.7% and 97.2%. The fixed Qwen3 4B QLoRA recipe performs poorly on Breast Cancer and Wine. Paired group-bootstrap intervals, all unsuccessful outcomes and caveats are retained. This is one split, seed and serialization on familiar public datasets, not a general model ranking.

Read the [findings](https://github.com/statsguysam/jev-classification-benchmark/blob/v0.3.0-tabular/results/TABULAR_FINDINGS.md), [full comparison](https://github.com/statsguysam/jev-classification-benchmark/blob/v0.3.0-tabular/results/TABULAR_COMPARISON.md), [protocol](https://github.com/statsguysam/jev-classification-benchmark/blob/v0.3.0-tabular/docs/TABULAR_PROTOCOL.md) and [cost audit](https://github.com/statsguysam/jev-classification-benchmark/blob/v0.3.0-tabular/results/TABULAR_API_COST_SUMMARY.md).

## Assets

* `jev-tabular-colab.zip`: frozen source and prepared public feature/label splits with source attribution and an internal SHA-256 manifest.
* `tabular_colab_artifacts.py`: standalone verified transfer helper. Upload this file **and** the source/data ZIP when running the notebook.
* `colab_tabular_benchmark.ipynb`: clean Colab notebook for the pinned Qwen3 4B zero/few/QLoRA workflow.
* `qwen-tabular-adapters.zip`: all six trained adapters with provenance, upstream Apache-2.0 licenses, modification notices and file hashes. Pretrained base weights are not included. See [adapter instructions](https://github.com/statsguysam/jev-classification-benchmark/blob/v0.3.0-tabular/docs/ADAPTERS.md).
* `jev-tabular-measured-results.zip`: all 66 run artifacts, per-row predictions, paired comparisons, charts, cost ledger with its lock anchor, import audit and documentation. Its manifest records the source commit and hashes every included payload.
* `TABULAR_SHA256SUMS.txt`: checksums for the five deliverable files above.

Conservative tabular API accounting is **US$13.229108700**; the unchanged text study brings cumulative accounting to **US$18.522115700**, below the authorized US$20. Known provider costs and token estimates are reported separately; these accounting totals are not invoices.

Validation: **300 tests passed, one skipped**; both notebooks validate and contain no saved outputs; the Colab import verifies nine runs and three adapters against the frozen local source/data; all 66 metric records recompute from predictions; all 2,472 hosted attempts reconcile with the durable ledger. Published files pass credential-pattern scanning, and adapter payloads pass weight/license/hash checks.
