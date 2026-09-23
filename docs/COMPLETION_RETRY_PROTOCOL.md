# Completion and explicit failure recovery

On 23 September 2026 the user explicitly requested completion of pending work,
including failed runs. The cumulative hosted-API authorization remains US$25.
This extension changes the earlier no-retry operational policy prospectively;
it never deletes original failures, releases old reservations, or reruns valid
predictions because their labels were wrong.

## Scope and sequence

1. Finish the 401 unseen numerical-review requests and 4,800 unseen text-review
   requests with their original inputs, cached proposals, demonstration IDs,
   prompt bytes, model settings, source hashes, and ledgers.
2. Audit and freeze a complete failure inventory. At the start there were 16
   failed paid Jev requests (including one historical Titanic request), one
   failed paid Astra request, and one Wine review not called because its Astra
   proposal failed. Include new first-pass failures in this inventory.
3. Retry only failed inference requests. Allow at most two additional attempts
   per failed row, stop at its first valid response, and never consult test truth
   to decide whether to retry or which response to keep. Use stable request order
   within each attempt round. Every new request has its own reservation and
   durable attempt record, including failures.
4. If the failed Astra proposal recovers, make the previously blocked Wine review
   using that explicitly versioned new proposal. This is a new dependent request,
   not a byte-identical retry of a request that never occurred. It has the same
   maximum two-attempt operational policy and separate provenance.
5. Execute all 1,714 previously frozen proposal-control requests in their original
   order after the numerical/text matrices are complete. Their published primary
   comparisons retain first-attempt failures in the denominator. Any later retry
   of a failed control is separate recovery evidence with the same finite policy.

Partial records, orphan reservations, changed prompt identities, changed sources,
and route or usage violations stop execution for reconciliation. A positive
funding check does not clear these errors. A billing failure or three consecutive
transport failures pauses the worker; it does not silently skip to successful
rows. Source-model collapse to one class is a measured outcome, not an execution
failure.

## Price and budget continuity

The frozen producers retain their original price declarations and source hashes.
A separately recorded completion session verifies current official route,
snapshot and prices and supplies an equivalent dated transport check. All
non-temporal request, source, ledger, accounting and model checks remain active.
The session receipt distinguishes historical declarations from fresh verification.

Retries receive a separately initialized allowance only after the existing
numerical/text first passes finish. The allowance is bounded by the unspent
cumulative authorization. The controls allocator must count this recovery ledger
before assigning remaining funds. Any control-recovery allocation is created
only after control execution is finalized and its ledger is frozen. Unknown
usage and failed requests keep their full reservations. No provider balance or
bank mandate increases the authorized cap.

A full attempt to complete every request is authorized; successful completion
is not guaranteed if the budget is exhausted or the provider fails repeatedly.
Report any remaining failure or unexecuted request explicitly.

## Reporting

Publish first-attempt and operationally recovered scores separately. Recovery
uses the earliest valid success in recorded order; it cannot select the most
accurate response. Keep the original prediction hash, request/prompt hash,
source hash, attempt index and selected-attempt identity for every changed row.
Retain original and recovered failure counts, and distinguish not-attempted
rows from upstream-blocked reviews. An incomplete evaluation has no final score.

Primary proposal-control claims use the frozen first-attempt experiment.
Recovered results are a sensitivity analysis. No claim of improved model
intelligence follows solely from replacing an API error with a valid response.
The LinkedIn post must disclose exploratory public-data evaluation and sample
sizes and must agree with the completed, audited reports.
