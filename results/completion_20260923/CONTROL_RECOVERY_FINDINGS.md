# Control failure-recovery sensitivity

[Audited secondary report](CONTROL_RECOVERY_COMPARISON.json) · [Original primary controls](../review_controls/COMPARISON.json)

This separate overlay retains all 12 full primary arms and eight paired contrasts. Original first attempts and the original serving-repeat diagnostic remain unchanged.

**15 additional calls.** Primary failures: 14 → 0; repeat failures: 1 → 0. Recovered repeat agreement is not substituted for the original diagnostic.

| Dataset | Arm | Full N | First accuracy | Recovered accuracy | First / recovered failures |
|---|---|---:|---:|---:|---:|
| breast_cancer | actual | 114 | 92.98% | 93.86% | 1 / 0 |
| breast_cancer | no_proposal | 114 | 92.98% | 92.98% | 0 / 0 |
| breast_cancer | shuffled | 114 | 92.98% | 94.74% | 2 / 0 |
| sst2 | actual | 200 | 96.00% | 96.50% | 1 / 0 |
| sst2 | no_proposal | 200 | 95.50% | 96.00% | 1 / 0 |
| sst2 | shuffled | 200 | 94.00% | 95.00% | 2 / 0 |
| trec | actual | 200 | 86.00% | 86.50% | 1 / 0 |
| trec | no_proposal | 200 | 86.50% | 87.00% | 1 / 0 |
| trec | shuffled | 200 | 88.00% | 88.00% | 2 / 0 |
| wine | actual | 36 | 86.11% | 88.89% | 1 / 0 |
| wine | no_proposal | 36 | 86.11% | 91.67% | 2 / 0 |
| wine | shuffled | 36 | 88.89% | 88.89% | 0 / 0 |

| Dataset | Contrast | First difference, pp [95% CI] | Recovered difference, pp [95% CI] |
|---|---|---|---|
| breast_cancer | actual_minus_no_proposal | +0.00 [-4.39, +4.39] | +0.88 [-2.63, +4.39] |
| breast_cancer | actual_minus_shuffled | +0.00 [-3.51, +3.51] | -0.88 [-2.63, +0.00] |
| sst2 | actual_minus_no_proposal | +0.50 [-1.00, +2.50] | +0.50 [+0.00, +1.50] |
| sst2 | actual_minus_shuffled | +2.00 [+0.00, +4.50] | +1.50 [+0.00, +3.50] |
| trec | actual_minus_no_proposal | -0.50 [-3.00, +1.50] | -0.50 [-2.50, +1.00] |
| trec | actual_minus_shuffled | -2.00 [-4.00, -0.50] | -1.50 [-3.50, +0.00] |
| wine | actual_minus_no_proposal | +0.00 [-13.89, +13.89] | -2.78 [-11.11, +5.56] |
| wine | actual_minus_shuffled | -2.78 [-8.33, +0.00] | +0.00 [+0.00, +0.00] |

Conservative added charge/reservation: $0.040320000; allocation $0.080640000. Every reservation is retained.

- This is a secondary availability sensitivity view on the same frozen cases, not a new test set.
- Primary first-attempt controls and the original serving-repeat diagnostic remain unchanged.
- An incorrect but valid recovered label is retained; true labels never select a retry.
- Every error remaining after the finite policy counts as incorrect with the full original denominator.
- Intervals are exploratory unadjusted group bootstraps conditional on frozen cases and served responses; balanced-accuracy deltas have no interval.
- Retries use additional calls and retain their full conservative reservation; this does not estimate deployment savings.
