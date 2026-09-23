# Response validity and class correctness

The frozen adapter accepts numeric class IDs from the declared label set. For
Jev, it also validates the returned probability vector before accepting the
prediction. Each class must be present, every probability must be finite and
within `[0, 1]`, the sum must be within an absolute tolerance of `1e-6` of one,
and the selected class must agree with a maximum probability within `1e-6`.
See [the adapter](../src/jevbench/providers.py).

These checks are part of the original execution protocol. They are not relaxed
after inspecting outcomes, including during exact-request recovery.

## Exact ties in accepted Choice responses

The completed text-review matrix contains four accepted TREC responses whose
saved Choice is a tied maximum but is not the first maximum in class-ID order:

| Source proposal | Examples per class | Prediction line | Saved Choice | Tied class IDs | Probability of each tied class |
|---|---:|---:|---:|---|---:|
| Qwen3 4B | 0 | 104 | 2 | 1, 2 | 0.39 |
| Qwen2.5 0.5B | 0 | 189 | 4 | 1, 4 | 0.42 |
| SmolLM2 1.7B | 0 | 104 | 2 | 1, 2 | 0.34 |
| SmolLM2 1.7B | 4 | 23 | 2 | 1, 2 | 0.50 |

The text-report collector incorrectly imposed a first-index argmax tie break on
these provider-selected Choices. Its audit-only correction matches the frozen
adapter's rule: the selected probability must be at least the maximum minus
`1e-6`. It retains first-index argmax for the other recipes and rejects Jev
Choices outside that tolerance. This does not change the frozen provider,
returned labels, probability vectors, saved metrics or ledgers. These four
accepted responses are not failures and do not enter failure recovery. Their
correctness is assessed using the saved Choice, without selecting among tied
classes using the true label. The existing argmax-disagreement diagnostic may
count these exact ties; that count alone does not establish an invalid response.

An independent scan of all 4,800 completed text-review rows found 4,752 accepted
responses and 48 saved errors. These four exact ties were the only accepted
responses that differed from first-index argmax; none selected a strictly lower
probability. This observation does not establish why the provider returned ties.

## Recovery audit compatibility before execution

Before the first paid recovery plan was created, the pure recovery-analysis
helper was aligned with the same frozen Jev decision rule. Its former check
already accepted any exact tied maximum, but rejected a selected probability
strictly below the maximum even when it was within the adapter's `1e-6`
tolerance. The Jev branch now requires `p[label] >= max(p) - 1e-6`.

Only `scripts/analyze_jev_recovery.py` and its focused tests changed for this
correction. The optional-probability branch used for other providers retains its
exact-maximum check. Probability normalization, range, shape and finite-value
checks are unchanged. The helper preserves the saved class and probability
vector, and still selects the earliest valid response without consulting truth.
No original response, metric, request, provider, budget or retry policy was
rewritten, and no paid request was made during this change.

- Before helper SHA-256: `b83e496cf67bcbb1bc454c2bb714b4dd15551e35e496b1e725d81dace3daa7e3`.
- After helper SHA-256: `c6a606113d79dc8a1038058a45a2f191c2eb2f42d19fd3b7627f246aed136df8`.
- Validation: **31 tests passed** in `tests/test_analyze_jev_recovery.py`,
  including a non-first exact tie, choices within and at the existing tolerance,
  rejection beyond it, unchanged normalization and other-provider checks, and
  earliest-success selection without relabeling.

## What a failed check means

A probability-validation failure is not an established wrong-class decision.
The adapter's saved error prediction has a null label and probability vector.
The original returned class, the size of a normalization discrepancy and its
cause therefore cannot be recovered from that record. In particular, the saved
error does not establish that rounding caused the failure. Reported usage and
provider cost can still exist: an invalid response can be a paid request.

First-attempt pipeline accuracy counts every failure as incorrect, retaining the
full held-out denominator. This measures accepted correct outputs under the
frozen protocol, rather than the unobserved correctness of rejected choices.
Any valid-response-only statistic is a separate conditional diagnostic.

## Corrections, harms and recovery

- A **wrong-label harm** replaces a correct source label with a valid but
  incorrect reviewer class.
- A **failure harm** replaces a correct source decision with an unusable review
  result under the validation policy.
- A successful retry is an additional observation and cost. It does not reveal
  what the original rejected response predicted and does not erase that failure.

The review-outcomes graphic separates wrong-label harms from failure harms.
Primary reports retain the original attempts; recovery reports expose additional
attempts separately. Accepted choices belong to the declared class set, while
response validity and classification correctness remain distinct measurements.
