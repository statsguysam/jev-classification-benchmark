# Validation scoring diagnostic: numeric ID versus ID + EOS

Completed 136 model contexts on 34 distinct validation rows. Both arms use the same forward pass per candidate. The recipe, row IDs, demonstration IDs, code, tokenizers and rendered contexts were frozen before inference. No test predictions were read or changed, no adapter was trained, and no API or download was used.

| Dataset | Model | Examples/class | Validation n | ID+EOS accuracy | ID-only accuracy | Class changes | Fixed / harmed by ID-only |
|---|---|---:|---:|---:|---:|---:|---:|
| breast_cancer | smollm2 | 0 | 16 | 50.0% | 50.0% | 0 | 0 / 0 |
| breast_cancer | smollm2 | 4 | 16 | 50.0% | 50.0% | 0 | 0 / 0 |
| wine | smollm2 | 0 | 18 | 33.3% | 33.3% | 0 | 0 / 0 |
| wine | smollm2 | 4 | 18 | 33.3% | 33.3% | 0 | 0 / 0 |
| breast_cancer | granite | 0 | 16 | 50.0% | 50.0% | 0 | 0 / 0 |
| breast_cancer | granite | 4 | 16 | 68.8% | 68.8% | 0 | 0 / 0 |
| wine | granite | 0 | 18 | 33.3% | 33.3% | 0 | 0 / 0 |
| wine | granite | 4 | 18 | 33.3% | 33.3% | 0 | 0 / 0 |

These small balanced samples were selected only from the existing validation split using the unchanged selector and seed42 (8/class Breast Cancer, 6/class Wine). The original training-only four-per-class examples are unchanged. Zero-shot has no examples. Model families are SmolLM2 and Granite, each with the exact frozen fixed-system, thinking-disabled chat renderer, cached pinned revision, float16 MPS and 8,192-token ceiling.

Each record saves the numeric-ID log probability, conditional EOS log probability, joint log probability and both normalized vectors. The joint score is the frozen core's float32 sum; small floating-point differences between its rounded sum and adding recorded Python floats are audited within declared tolerance. ID-only omits the EOS contribution and is therefore a distinct output-scoring event. No architecture comparison is isolated by this within-context diagnostic.

This is exploratory development evidence after earlier test behavior was observed. No confidence gate or winning method is selected, and no historical score is replaced. The 34 validation labels are an added diagnostic label budget; repeating them across models/shots does not make them independent test examples. The balanced samples change natural class prevalence, so their accuracies are not directly comparable to previous test accuracy. Confirm any recipe on new held-out data before deployment or general claims. Greedy free generation was explicitly omitted before execution and remains a separately planned ablation.
