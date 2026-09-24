# Medium figure captions

## 01_wine_rescue

On the 36-row Wine test set, adding Jev improved Qwen2.5 from 12 to 33 correct answers. Jev alone made exactly the same final predictions. Logistic regression scored 35/36 with the same 12 training labels.

## 02_trec_overrides

On TREC, Jev corrected three Astra mistakes but changed 24 correct answers to wrong ones. The net effect was 21 fewer correct answers out of 200, a drop of 10.5 percentage points.

## 03_proposal_value

Including Qwen3's proposed answer changed Jev's observed accuracy by no more than half a percentage point in either direction in the original primary comparison. The 95% intervals all include zero. The Wine interval is wide because its test set contains only 36 rows. This does not prove equivalence.

## 04_trec_cost

For the same 200 TREC questions, Jev alone cost about $0.0087 and scored 85.5%; Astra cost an estimated $1.3564 and scored 97.0%. Combining them cost about $1.3656 and scored 86.5%. The comparison counts both pipeline stages once and excludes local compute, training and engineering.

## 05_matched_label_context

With only four training examples per class, classical methods were competitive on these numerical tasks but much weaker on the text tasks. Text baselines used TF-IDF. Equal numbers of task-specific labels do not equalize the prior pretraining available to Jev and Astra. These are fixed-recipe, one-split results.

## 06_jev_few_shot

Jev's standalone accuracy improved after adding four labeled examples per class to the prompt on each dataset, with the largest observed gains on Wine and TREC. This was in-context learning, not fine-tuning.

## 07_all_review_conditions

Across all 48 original source-model and prompting conditions, Jev review improved accuracy in 32, tied in three, and reduced it in 13. Cells show percentage-point changes. The conditions reuse four test sets and are not independent replications.
