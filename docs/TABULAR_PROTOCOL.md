# Numerical and mixed-feature tabular extension

This extension compares Jev, two open language models, two OpenAI models, and native classical estimators on the same serialized tabular classification tasks. It extends the text pilot with numerical inputs and mixed numerical/categorical inputs. The inference core and prompt template from the earlier release remain frozen. Preparation, training and hosted execution use separately hashed extension scripts.

## Datasets and immutable sources

| Dataset | Task | Original rows | Input features | Train / validation / test |
|---|---|---:|---:|---:|
| Titanic3 | Binary passenger survival | 1,309 | 4 numerical + 3 categorical | 785 / 262 / 262 |
| Breast Cancer Wisconsin Diagnostic | Binary malignant/benign classification | 569 | 30 numerical | 341 / 114 / 114 |
| Wine | Three-class cultivar classification | 178 | 13 numerical | 106 / 36 / 36 |

Titanic3 comes from the [Vanderbilt Biostatistics dataset collection](https://hbiostat.org/data/), using its [Titanic3 data dictionary](https://hbiostat.org/data/repo/ctitanic3). Data obtained from https://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics. The collection permits use with attribution. This is the 1,309-passenger Titanic3 dataset, not the 891-row Kaggle competition training partition.

The numerical datasets use scikit-learn's bundled copies of [Breast Cancer Wisconsin Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) and [Wine](https://archive.ics.uci.edu/dataset/109/wine). UCI lists both under CC BY 4.0. Cite Wolberg, Mangasarian, Street and Street (1993), DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), and Aeberhard and Forina (1992), DOI [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J), respectively. These loaders preserve the documented feature order and labels; their APIs are described by [scikit-learn's Breast Cancer loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) and [Wine loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html).

The exact source byte hashes are pinned in [tabular_datasets.json](../configs/tabular_datasets.json):

| Source file | SHA256 |
|---|---|
| `titanic3.csv` | `db6df9666818c69a753cd85d743e01502c8518b00579b6aada0d4fc5a66ccb9d` |
| `breast_cancer.csv` | `fed3eb72d0575ef6192293f5093c6e801b1476b577d0386bf4455504522172ed` |
| `wine_data.csv` | `10e8a802908b34f86e5da8ce962f3c806694bc98450a18f61851af59f324bede` |

A source hash mismatch fails preparation. The prepared manifests also record the scikit-learn version, selected native-record hash, serialized file hashes, native sidecar hashes, ordered row-ID hashes, preparation-script hash and final prepared-content hash. Preparation used scikit-learn 1.9.1. Exact reproduction should use that version and verify the resulting prepared hashes; Colab consumes the already prepared bundle without regenerating splits.

## Feature selection and serialization

Titanic allows only `pclass`, `sex`, `age_years`, `sibsp`, `parch`, `fare_gbp`, and `embarked`. Passenger class is categorical. Age is in years; fare is in British pounds; `sibsp` and `parch` are family-member counts. Embarkation codes become full port names. The target and excluded columns—name, ticket, cabin, boat, body, and home/destination—never enter model input. Boat and body can reveal outcomes; identifiers and unnecessary personal descriptions are also omitted.

Breast Cancer uses all 30 supplied measurements: the mean, standard error and worst summaries of cell-nucleus features. Wine uses all 13 supplied chemical measurements. Numerical values retain their original source scales. Units are stated when documented, otherwise the schema says that the original source scale is used; the benchmark does not invent physical units.

Every LLM receives the same deterministic `name-value-v1` representation: a short fixed task description followed by `feature=value` pairs in declared order. The serializer uses 12 significant decimal digits, JSON-quoted categorical strings and the unquoted missing marker `NA`. The current source values round-trip exactly at this precision. Example format, using illustrative values:

```text
Titanic passenger survival. pclass: 1=first, 2=second, 3=third class. age_years is age; fare_gbp is fare in British pounds; sibsp counts siblings/spouses and parch counts parents/children aboard. Missing=NA.
pclass="3"; sex="male"; age_years=NA; sibsp=0; parch=0; fare_gbp=7.25; embarked="Southampton"
```

No row ID or target appears among the serialized features. There is no learned imputation, scaling, binning or feature selection in the serializer, and no input truncation. Maximum serialized row lengths are 307 characters for Titanic, 929 for Breast Cancer and 444 for Wine. The frozen prompt places class names outside the feature text and adds only the selected training demonstrations for few-shot runs.

This is one fixed serialization, not a search over alternative formats, feature orders or missing-value descriptions. Native classical models consume the original numerical/categorical values, rather than TF-IDF of serialized strings. `load_native_prepared` verifies that every native record reserializes exactly to its corresponding LLM input with identical row ID, target and ordering.

## Splits and label budgets

Source records are ordered by stable IDs. Five folds are produced with `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. Groups are hashes of normalized serialized features, excluding the target and row ID. Fold 0 is the test split, fold 1 is validation, and folds 2–4 form training. All source rows are retained. No exact feature group crosses a split, including groups containing conflicting target labels.

The primary extension evaluates the **full held-out fold**, not a sampled test subset: 262 Titanic, 114 Breast Cancer and 36 Wine rows. The preparation helper optionally supports a deterministic proportional stratified test limit for future budget-limited pilots; excluded test IDs are audited and never reassigned to training. That option is unused in the primary full-fold matrix.

Titanic's seven-feature projection contains natural duplicates. Its test set has 218 distinct feature vectors among 262 rows, including six groups with conflicting targets. The train and validation splits retain their own duplicates without feature overlap across splits. Breast Cancer and Wine have no exact duplicate feature vectors in these prepared splits. The split policy does not guarantee that related Titanic families fall in different groups.

The shared frozen example selector draws exactly four examples per class with seed 42. Few-shot prompting, LoRA/QLoRA and matched classical fitting receive the same ordered training IDs: eight labels for each binary task and 12 for Wine. Zero-shot receives no new labeled examples. Full-training native baselines use 785, 341 or 106 training labels respectively. No extension condition uses validation labels, performs a development-set hyperparameter search or selects a checkpoint using test performance. Pretraining data and computation are not matched across model families.

## Model and method matrix

The intended inventory is **66 runs**:

| Family | Conditions | Runs across three datasets |
|---|---|---:|
| Native classical | Majority, logistic regression, RBF SVM, random forest, histogram gradient boosting; each with 4/class and full training | 30 |
| Open LLM | Qwen2.5 0.5B and Qwen3 4B; zero-shot, 4/class few-shot, and 4/class adaptation | 18 |
| Hosted | Jev 1.13, GPT-5.6 Luna, GPT-6 Astra; zero-shot and 4/class few-shot | 18 |

Open-model revisions and hosted settings are fixed in the run configuration. Results retain requested and returned model identifiers, resolved revisions, observed inference precision/device, adaptation precision and quantization, source hashes, and wrapper provenance. Hosted hardware is not disclosed. The report labels ordinary LoRA versus QLoRA from recorded training metadata. Jev/OpenAI LoRA is unsupported by this study and is left unscored.

The open-model recipe uses three epochs, learning rate `2e-4`, batch size 1, gradient accumulation 8, LoRA rank 8, alpha 16, all linear target modules, zero LoRA dropout, response-only numeric-class-plus-EOS supervision, and a fixed final checkpoint. Training accepts complete prompts up to 2,048 tokens and rejects oversized inputs. Evaluation uses the configured context limit, with complete prompt checks. Training failures and configuration changes require separately identified artifacts, not silent replacement of an observed result.

Native classical hyperparameters are fixed in [run_tabular_classical.py](../scripts/run_tabular_classical.py). Its training-only pipeline fits numerical median imputation and standardization, categorical constant missing imputation, and one-hot encoding with unknown categories ignored. Entirely missing numerical training columns become zero. The same pipeline recipe applies in matched and full-training tracks; validation data do not enter fitting. RBF SVM uses `probability=False` to avoid extra calibration fitting on the tiny label budget. Other classical probabilities are uncalibrated.

## Metrics, audits and uncertainty

Only complete prediction sets are scored. Failures count as incorrect predictions for accuracy and macro-F1. The report retains failure counts, probability coverage, log loss, Brier sum, ECE, latency and training time where available. Probability metrics apply only to valid predictions with full class distributions; missing probabilities are not replaced with fabricated confidence values.

Different output interfaces are explicit: Qwen uses normalized numeric-class-plus-EOS sequence likelihoods, Jev provides native Choice distributions, OpenAI returns a generated class label, and classical estimators provide their native predictions/probabilities. These procedures are not interchangeable calibrated probability estimators. Hosted end-to-end latency, local sequential candidate scoring and classical amortized batch latency also measure different workloads on different hardware.

[summarize_tabular.py](../scripts/summarize_tabular.py) checks each completed run against `load_native_prepared`: exact prepared-manifest equality, serialized/native hashes, ordered held-out rows, training IDs and frozen core identity. It checks approved open-model revisions, context limits, devices and adaptation quantization against the configured presets. Hosted presets and wrapper provenance must match the frozen execution source, canonical ledger identity, cumulative caps and earlier ledger anchors. The analysis reads those declarations without importing request code. It verifies native extension hashes and fixed parameters, then independently refits only preprocessing on the audited selected training rows to check the recorded medians, scales and category levels. It recomputes metrics from raw predictions and fails if saved values disagree. Colab results must consume the identical prepared bundle; there is no exception allowing merely similar manifests. Incomplete and missing conditions remain pending without scores.

The tabular analysis adds **unstratified group percentile bootstrap intervals** with 2,000 replicates and seed 42. A replicate samples the observed number of distinct test-text SHA256 groups with replacement, keeping every row in each drawn group. Conflicting targets and failure predictions remain included. Replicate row counts may vary because groups have different sizes. Paired contrasts use the same sampled groups for both models, and the reported difference is A minus B. Original per-row bootstrap fields in run artifacts remain unchanged; the tabular report displays the newly computed group intervals.

The 18 primary matched comparisons are, for each dataset: Jev few-shot minus Astra few-shot, Qwen 4B few-shot, matched logistic regression and matched random forest; plus adapted-minus-few-shot for each open model. Fifteen few-shot-minus-zero-shot contrasts and 15 full-training-minus-four-per-class native contrasts are separately labeled descriptive comparisons with unequal new-label budgets. These are exploratory analyses with unadjusted intervals, not a preregistered significance-testing program or evidence of general model superiority.

Intervals condition on one split, one demonstration/training selection seed, one prompt and one serialization. They do not capture variability across training seeds, prompt changes, dataset shifts or undisclosed provider changes. Identical-feature groups address one known dependency, not every possible family or source dependence.

## Interpretation limits

All three datasets are widely circulated public benchmarks. Grouped held-out splits prevent exact feature leakage from newly supplied training rows but cannot remove pretraining memorization or dataset-specific prior knowledge. Wine cultivar IDs are arbitrary source identifiers: zero-shot accuracy should not be interpreted as evidence that the class names provide transferable semantic information. The Breast Cancer experiment evaluates a historical machine-learning dataset and is not clinical validation or a deployment recommendation.

Tiny four-per-class fine-tuning is a specific fixed recipe. A weak adapted result does not establish a general limit of LoRA, nor justify retuning after inspecting the same test fold. The small 36-row Wine test fold particularly limits the precision and scope of any comparison.

## Reproduction

Prepare the full frozen data and audit native/serialized parity:

```bash
.venv/bin/python scripts/tabular_data.py --output data/tabular-full
```

Run the native baselines, using the same IDs as the language models:

```bash
.venv/bin/python scripts/run_tabular_classical.py --data-root data/tabular-full --include-full
```

Open-model execution is planned and run by [run_tabular_local.py](../scripts/run_tabular_local.py). Hosted execution is planned and budget-guarded by [run_tabular_hosted.py](../scripts/run_tabular_hosted.py); credentials belong in the environment or a secure hidden prompt, never in a configuration or result file. Earlier text-study ledgers are preserved and included in cumulative budget accounting.

After importing the completed Colab artifacts, rebuild the audited report:

```bash
.venv/bin/python scripts/summarize_tabular.py --bootstrap-samples 2000
```

The report writes `results/TABULAR_COMPARISON.md`, `.json` and `.csv`, leaving raw predictions and original run metrics untouched. Missing runs stay visible as pending conditions.

Generate the two static figures from that audited JSON (matplotlib runtime shown for the local environment):

```bash
MPLCONFIGDIR=artifacts/matplotlib /opt/anaconda3/bin/python scripts/plot_tabular.py
```

`tabular-matched` shows 17 zero-shot or matched-label conditions per dataset. `tabular-native-label-budgets` separately compares five native estimators under four-per-class and full-training budgets. Both produce PNG, SVG and plotted-value JSON under `results/figures`, including the source-summary hash. Model order is fixed rather than ranked by observed score; pending conditions have no plotted point.
