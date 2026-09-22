# Proposal-value controls: prepared, not executed

This is a **new exploratory experiment on already-seen holdouts**, designed to test whether a cached source proposal adds value to a Jev decision. It is not a confirmatory study or a claim of novel general capability. A larger numerical benchmark with genuinely fresh evaluation data would be a separate study.

The primary experiment uses Qwen3 4B at four examples per class, on every frozen test row in four existing datasets. Model identity, revision, original source artifacts, source manifest hashes and independent text content-equivalence proofs remain unchanged.

| Dataset | Input | Test rows | Classes | Training examples shared by all arms | Primary Jev requests |
|---|---|---:|---:|---:|---:|
| Breast Cancer | Serialized numerical features | 114 | 2 | 8 | 342 |
| Wine | Serialized numerical features | 36 | 3 | 12 | 108 |
| SST-2 | Text | 200 | 2 | 8 | 600 |
| TREC | Text | 200 | 6 | 24 | 600 |
| Total | Separate dataset-level analyses | 550 | — | — | 1,650 |

The primary contrasts require **1,650 requests**. A separately declared serving-repeatability diagnostic adds **64 exact repeated no-proposal requests**, for **1,714 planned requests total**.

**Preparation makes no API calls and allocates no money.** Provider funding remains a prerequisite. Existing numerical and text-review allocations remain protected inside the previously authorized cumulative budget. A future guarded runner must record its own explicit allocation and refuse to exceed it. Preparing all 1,714 inputs does not imply the full experiment is funded or that it will finish.

## Three exact prompt controls

Every selected case has three independent Jev requests:

1. **Actual proposal:** the source model's cached class ID for that same row.
2. **No proposal:** the literal JSON string `"not provided"` occupies the proposal slot.
3. **Shuffled proposal:** a cached class ID from a different selected row in the same dataset.

All three use identical wording, original input, allowed classes, training examples and example order. The **only prompt difference is the JSON value in one proposal slot**. The shared wording tells Jev that a supplied proposal can be kept or overridden and that an absent proposal requires an independent decision. The prompt never identifies Qwen, supplies its probabilities, or generates a rationale.

Conceptually, the shared suffix is:

```text
A cached proposal may be available in the slot below. It may be incorrect.
Classify the original final item using its content and the labeled examples.
If a proposal is provided, you may keep or override it;
if it is not provided, decide independently.
Cached proposal class ID: <integer or "not provided">
Return only the numeric class ID.
Class ID:
```

The executable module contains the exact whitespace and text; frozen prompt hashes bind each request. Historical direct-Jev predictions remain contextual references because their prompts differ. They are **not substituted for the newly measured no-proposal control**.

This studies the value of proposal availability and proposal alignment for a separate reviewer under fixed wording. It does not isolate bounded decoding applied to the same LLM. Separate provider calls can still vary; identical prompts are not independent architecture treatments.

## Serving-repeatability diagnostic

Before any new review outcomes, select sixteen case IDs per dataset by a separate SHA256 domain over seed 42, dataset and case ID. For each, add an independent request named `no_proposal_repeat` with **exactly the same prompt bytes, class choices and prompt hash** as its `no_proposal` reference. The new request has its own identity and records `repeat_of_request_id`. There are 64 repeat calls; they do not change the two primary proposal-value contrasts.

Repeat selection never uses truth, source correctness or review outcomes. The diagnostic reports label agreement/disagreement and success/failure combinations for every selected pair. It helps interpret possible serving variability; these repeats are not independent architecture treatments. A historical audit found disagreements among identical prompts, but those observations are not new measurements in this experiment.

All repeat and primary calls share the globally randomized order. A request named “repeat” may occur earlier than its named reference: the designation identifies a predeclared pair, not a temporal treatment. Both must be independent fresh calls.

## Label-blind selection, shuffling and execution order

The primary study includes all 550 rows. Cases are ordered by SHA256 of a domain separator, seed 42, dataset name and opaque frozen row ID. No target labels, text content, source correctness or new Jev outcomes determine selection or order.

Within each dataset, selected row IDs receive a separately seeded SHA256 ranking. Each row's donor is the next row in that ranking, wrapping at the end. This creates a single-cycle derangement: **no row donates its own proposal**, and the multiset of proposed classes is preserved exactly. It is one deterministic permutation, not a uniform sample from all possible derangements. Different rows can have the same proposed class, so each dataset records both the zero donor fixed-point count and the number of **unchanged proposal labels**. The latter can be substantial or even equal the whole dataset if a source collapses to one class.

All case/arm requests are then globally ordered by a third SHA256 domain over seed, dataset, row ID and arm. This interleaves treatments without looking at truth or predictions. A runner must follow the frozen order with fresh independent requests and no shared chat history. Execution order, request IDs and prompt hashes are recorded; no automatic retry or prompt substitution is permitted by preparation.

An optional **16-row-per-dataset operational pilot** uses the same row-ID ranking and three-arm design: 64 cases and 192 planned requests. The optional pilot has only the three primary arms and no serving-repeat diagnostic. Its role is testing the workflow, and it must use a separate directory. It is not the primary publication result and must not be promoted into the full study after seeing its results. The label-blind subset is not stratified and may omit classes. No replacement row is selected to improve label balance or source accuracy. If a selected source proposal is failed or invalid, preparation fails rather than dropping or replacing the case; a separate documented failure policy would be required.

## Predeclared comparisons and reporting

The two primary contrasts, separately for each dataset, are:

- **Actual minus no proposal:** does supplying the source proposal improve Jev over the otherwise identical independent-decision prompt?
- **Actual minus shuffled:** does correct row-to-proposal alignment add value beyond supplying a proposal with the same aggregate class frequencies?

Future analysis must use the identical selected rows in each contrast, keep failed decisions in the accuracy denominator, and distinguish corrected predictions, harmed predictions, wrong labels and inference failures. No silent fallback to the source proposal is allowed. An incomplete arm remains unscored as a completed test.

Report accuracy and macro-F1 with paired normalized-text group percentile bootstrap intervals using 2,000 resamples and seed 42. These are exploratory intervals conditional on the selected rows, source predictions, one shuffle and the serving environment, without multiple-comparison adjustment. They do not measure variation across alternative splits, example sets, shuffles or repeated provider calls.

Balanced accuracy is reportable only if every declared true class is represented in the evaluated subset; otherwise it must be null with an explicit missing-class reason. This matters especially for the operational TREC subset. Do not pool all four datasets into one overall accuracy or claim a model ranking from operational partial completion.

Test truth remains in the separately audited frozen data and may be read by the eventual reporter. **Request files contain no per-case test labels.** They do contain the allowed class names, training demonstrations and the cached proposed class needed by each treatment. Training labels are intentionally shared across all three arms.

## Offline artifacts and reproduction

The preparation module is [prepare_review_controls.py](../scripts/prepare_review_controls.py). Its default is a read-only dry run:

```sh
.venv/bin/python scripts/prepare_review_controls.py --all-rows
```

After verifying the plan, explicitly freeze it:

```sh
.venv/bin/python scripts/prepare_review_controls.py --all-rows --freeze \
  --output results/review_controls/prepared_full
```

The bundle contains:

- `protocol.json`: study scope, source hashes, source IDs, selected row IDs, donor mapping, proposal counts, execution order, predeclared repeat pairs and the request-file hash. No raw input or demonstration text is embedded.
- `requests.jsonl`: local execution payloads with case identity, arm, proposal value, donor identity, class choices, exact prompt, prompt hash, execution index and optional repeat-reference request ID. This repeated raw-input file is ignored by Git; rebuild it from the frozen data.
- `manifest.json`: hashes of the protocol and request file.

Freezing uses exclusive directory/file creation and refuses to overwrite complete or partial artifacts. Verification reads the bundle, checks hashes and helper identities, and reconstructs every request from freshly audited original sources:

```sh
.venv/bin/python scripts/prepare_review_controls.py \
  --verify results/review_controls/prepared_full
```

The Python integration API is `load_jobs()`, `build_plan(jobs, limit_per_dataset=None, seed=42)`, `freeze_plan(plan, output)`, and `load_frozen_plan(output, revalidate_sources=True)`. A runner should always revalidate sources and send **only `prompt` and `choices`** through the bounded reviewer interface; all other request fields are audit metadata. The source model's name, confidence and provenance belong in the protocol/logs, not the reviewer prompt.

No new response, score, API credential or paid ledger is created by this preparatory work.
