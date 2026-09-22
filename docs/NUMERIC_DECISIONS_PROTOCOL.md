# Numerical decisions: corrected research question

## Question and scope

On numerical tabular classification, how does Jev compare with native XGBoost and LightGBM, and does reviewing an LLM's proposed class with Jev improve the final decision?

This supplement corrects the scope of the earlier broad text/tabular study. The existing dashboard and releases describe that earlier study; they do not establish that adding Jev improves an LLM. Only Breast Cancer Wisconsin Diagnostic and Wine are included here. Titanic contains categorical fields, and all text datasets are excluded.

This is an exploratory extension designed after inspecting the earlier results. It is not preregistered or a publication-scale benchmark. The protocol is written before the new Jev review calls, and the boosted-tree configuration is fixed before fitting those models. Neither is tuned against the test outcomes.

## What Jev does

[TypeSafe describes Jev](https://docs.typesafe.ai/concepts/system-one) as a separate model that evaluates a state and answers typed questions with probabilities. A Choice question defines the allowed categories; Noul evaluates a yes/no statement. Jev does not modify another LLM's weights or expose a switch that turns arbitrary LLMs into Jev. An application can compose the two models by passing an LLM's output into Jev's state.

Restricting an answer to allowed categories ensures output validity, not prediction correctness. XGBoost, LightGBM, and the existing Qwen restricted-label scorer already choose among a fixed set of labels. The scientific question is therefore the quality and cost of those decisions, including which errors a second model corrects or introduces.

## Systems

| Arm | Input and decision procedure | Status of evidence |
|---|---|---|
| Jev alone | Serialized numeric row and zero or four labeled examples per class → native Choice | Reuse immutable measured runs |
| Qwen3 4B alone | Same row/examples → restricted class-label likelihood scoring | Reuse immutable four-per-class run |
| GPT-6 Astra alone | Same row/examples → generated class ID, strictly parsed | Reuse immutable four-per-class run |
| Qwen3 4B proposal → Jev | Original row/examples plus the cached Qwen proposed class → Jev Choice | New review arm |
| GPT-6 Astra proposal → Jev | Original row/examples plus the cached Astra proposed class → Jev Choice | New review arm |
| XGBoost | Native numeric columns → supervised classification | New fixed-recipe runs |
| LightGBM | Native numeric columns → supervised classification | New fixed-recipe runs |
| Logistic regression, random forest | Native numeric columns → supervised classification | Reuse reference runs |

The new arm is specifically a **cached-label proposal + Jev review**. It does not generate an LLM explanation, invent reasoning that was never saved, or claim to test all possible LLM–Jev architectures. Reusing the exact proposed labels makes it possible to count corrections and regressions without changes from another LLM sampling pass. It also avoids spending on duplicate LLM calls.

Jev receives the original features and the same labeled training examples in both direct and review conditions. The proposed class is advisory, not ground truth. No test target or identifying row ID enters the review state. A failed source proposal is a pipeline failure; it is not silently repaired by routing to a different arm. A Jev error is also retained as a failed final prediction. No automatic retries or test-row deletion.

The comparison estimates the effect of this particular composition, not the causal effect of boundedness alone. The models and output mechanisms differ. A future schema-constrained LLM arm would help distinguish output-format reliability from a second model's classification ability.

## Data and label budgets

All source hashes, feature definitions, native/serialized equality checks, and grouped split identities are inherited from [TABULAR_PROTOCOL.md](TABULAR_PROTOCOL.md).

| Dataset | Task | Numeric features | Train / validation / test | Matched training labels |
|---|---|---:|---|---:|
| Breast Cancer Wisconsin Diagnostic | Binary | 30 | 341 / 114 / 114 | 8 |
| Wine | Three classes | 13 | 106 / 36 / 36 | 12 |

The matched comparison gives each system the exact same four training records per class selected with seed 42. The full-training tree baselines separately use all 341 or 106 training labels. These full-training results describe the practical supervised alternative; they are not an equal-label comparison. No validation labels are used. Pretraining exposure and compute remain unequal.

Native models see numerical columns, with any imputation fitted only on their selected training records. Jev and LLMs see the existing deterministic, lossless named-feature serialization. Serializing numerical data does not turn the underlying task into a text-classification dataset.

## Analysis fixed before new review results

- Report accuracy, macro-F1, balanced accuracy, failures, and available probability metrics with coverage.
- For each LLM, compare reviewed versus original predictions on exactly aligned test rows. Count **wrong → correct**, **correct → wrong**, **both correct**, and **both wrong**. The net change in correct predictions equals corrections minus regressions.
- Calculate paired 95% percentile bootstrap intervals for accuracy and macro-F1 differences using 2,000 whole-feature-group resamples, seed 42. These datasets have no exact duplicate feature vectors, so their groups each contain one row.
- Compare Jev alone and the review arms with both boosted-tree baselines on matched labels. Show full-training baselines in a separate table.
- Retain all outcomes, including deterioration, failures, and intervals crossing zero. No model selection by test performance and no headline selected solely because it favors Jev.
- Treat probability quality as descriptive. Native Jev distributions, tree probabilities, and restricted-label LLM likelihoods are not automatically calibrated or interchangeable. Do not use an LLM's self-reported confidence as a probability estimate.
- Report the review's incremental API cost separately from the earlier LLM generation cost. Cached proposals do not make a live two-model pipeline free. Recorded local batch time and hosted request time do not establish a controlled speed ranking.

## Spending and reproducibility

The completed earlier study accounts conservatively for US$18.522115700 under the user's cumulative US$20 ceiling. The new review allocation is capped at US$0.90, anchored to all three existing ledgers and lock files. At most 300 Jev calls reserve US$0.806400000 at the verified US$0.002688 per-request bound. No new OpenAI calls are required. API-reported charges and conservative reservations remain separate.

New artifacts live under `results/numeric_decisions/`; frozen earlier source files and run records remain unchanged. The review stores source-run identities and prediction hashes so the proposed labels can be traced back to the exact original runs. The supplemental tree runs record true library versions, full parameters, selected training IDs, preprocessing, native probabilities, and source hashes.

## What a LinkedIn post can claim

An honest post can describe a reproducible **two-dataset numerical pilot**, report the exact corrections/regressions and label budgets, and explain what it taught us about combining decision models. It cannot establish general superiority, novel architecture, production suitability, or a result nobody has reported before.

Wine has only 36 held-out rows: one changed prediction is 2.78 percentage points. Both datasets are widely known, so pretraining exposure cannot be excluded. One split and one example-selection seed do not measure robustness. A larger follow-up should add less familiar numeric datasets, repeat training/example selections, evaluate anonymized feature names and permuted class IDs, and reserve untouched test data for confirmation. Those checks are future work, not completed evidence in this pilot.
