# Numeric LinkedIn draft

Draft only; not posted. This numeric-only version describes completed first-attempt results: 68/68 conditions, 24/24 reviews and 72/72 contrasts. Original failures remain included; [completed recovery](../completion_20260923/RECOVERY_FINDINGS.md) is a separate view. The [main publication draft](../review_value/LINKEDIN_DRAFT.md) incorporates the completed text, matched-control and recovery evidence. Verify public access to linked evidence before sharing.

---

I tested whether adding a decision reviewer improved numerical classification. Sometimes it corrected the source. Sometimes it damaged answers that were already right.

Six LLMs, two public datasets, zero examples or four examples per class. Jev reviewed each saved class proposal using the same row and examples. I also measured Jev alone and native-feature classical models.

All 24 review settings are complete. Three observations stand out:

- Astra's zero-shot Wine accuracy fell from 36/36 to 16/36: **100% → 44.4%**. Twenty correct answers became wrong labels. This condition had no API failures.
- Qwen2.5 0.5B's four-shot Wine accuracy rose from 12/36 to 33/36: **33.3% → 91.7%**. Yet every reviewed label matched Jev alone. That large gain over the source added no observed accuracy over direct Jev.
- With the same twelve Wine training examples, native-feature logistic regression reached **35/36 (97.2%)**, and random forest **34/36 (94.4%)**.

That changes how I evaluate a review stage: measure both what it fixes and what it breaks, compare it with its source and with the reviewer alone, and keep simple baselines in the experiment.

These are exploratory results from 150 unique held-out rows across two familiar public datasets, one split and fixed recipes. The small models use a particular restricted-label scoring setup; Wine cultivar IDs are arbitrary without examples. The results do not establish general model superiority or the causal value of a proposal. Original inference failures remain in the denominators; separate retries must not erase them.

Code, predictions and protocol: [jev-classification-benchmark](https://github.com/statsguysam/jev-classification-benchmark).
