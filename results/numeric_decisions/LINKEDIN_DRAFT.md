# LinkedIn draft: numerical Jev experiment

Owner note: this draft has **not been posted**. The GitHub repository is currently private. Make a reviewed public repository/release available before presenting its link as open-source access. Suggested image: [review-corrections.png](review-corrections.png); optional second image: [numeric-comparison.png](numeric-comparison.png). Results and limitations are documented in [INTERPRETATION.md](INTERPRETATION.md).

---

Does adding a decision model after an LLM actually improve its decisions?

I explored this with Jev on numerical tabular data: Breast Cancer Wisconsin (binary) and Wine (three classes). No text-classification datasets.

The setup: an LLM proposes a class, then Jev receives that proposal plus the original numerical features and makes the final bounded choice. I also tested Jev alone and native XGBoost, LightGBM and simpler ML baselines.

In the matched-label comparison, each system received the same four labeled examples per class and the same held-out rows. I reused the exact saved LLM proposals to measure what Jev changed.

The result depended on the first model:

• Qwen3 4B → Jev: accuracy rose from 86.0% to 93.0% on Breast Cancer, and 80.6% to 88.9% on Wine.

• GPT-6 Astra → Jev: accuracy fell from 98.2% to 93.9%, and 97.2% to 94.4%.

On Breast Cancer, Jev fixed 10 Qwen errors and introduced 2. For Astra, it fixed none, introduced 4 wrong labels, and had 1 transport failure. Failures stayed in the score.

The classical models gave me another reference. Jev's few-shot point estimates exceeded our fixed XGBoost/LightGBM recipes at this tiny label budget. But logistic regression did better than Jev with the same labels. Both boosted-tree models also exceeded Jev's few-shot accuracy when trained on the full training split, which used more labels.

And Jev alone already scored 93.0% / 91.7%, compared with 93.0% / 88.9% for Qwen → Jev. Adding an LLM stage did not improve those point estimates.

My takeaway: bounded outputs make the answer space predictable. They do not guarantee a correct answer, and another model in the pipeline is not automatically an upgrade. Always test the components, the chain, and a simple classical baseline.

This is an exploratory pilot: 114 and 36 test rows, one split/seed, fixed recipes, possible pretraining exposure. Paired bootstrap intervals are in the report; the Wine changes remain uncertain. I would want larger, less familiar datasets and repeated splits before making a general claim.

Code, predictions and methodology: https://github.com/statsguysam/jev-classification-benchmark/tree/main/results/numeric_decisions

#MachineLearning #TabularData #LLM #Jev #AIResearch
