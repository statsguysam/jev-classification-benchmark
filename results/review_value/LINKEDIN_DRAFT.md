I tested Jev to understand where it helps and where a simpler classifier is enough.

The question was practical: how well does it handle bounded decisions, such as yes/no or a fixed category, compared with LLMs and classical ML?

So I tested:

• LLMs alone: Qwen2.5 0.5B, Qwen3 4B, SmolLM2 1.7B and IBM Granite 3.3 2B (open-weight); GPT-5.6 Luna and GPT-6 Astra (OpenAI).
• Jev alone: TypeSafe’s Jev 1.13.
• Each LLM’s proposed class reviewed by Jev.
• Classical ML: XGBoost, LightGBM, logistic regression and random forest.

Datasets:

• Numerical, binary: Breast Cancer.
• Numerical, multiclass: Wine (3 classes).
• Text, binary: SST-2 sentiment.
• Text, multiclass: TREC questions (6 classes).

LLMs and Jev used zero-shot prompts (no labeled examples) and few-shot prompts (4 examples per class).

All methods saw the same test rows within each dataset. Classical ML trained on either the same few-shot examples or the full training set, with results reported separately.

Three findings stood out:

1. A large improvement did not mean both models were needed.

On Wine, few-shot Qwen2.5 0.5B improved from 33.3% to 91.7% after Jev review: 12/36 to 33/36 correct.

But Jev alone produced exactly the same final labels. Logistic regression reached 35/36 (97.2%) using the same twelve training examples.

2. A reviewer could undo good decisions.

On TREC, few-shot GPT-6 Astra scored 97.0%. Adding Jev reduced this to 86.5%. It corrected 3 mistakes but changed 24 correct answers to wrong ones.

For those 200 questions, API cost per pass and accuracy were:

• Jev alone: $0.009, 85.5%.
• Astra alone: $1.36, 97.0%.
• Astra + Jev: $1.37, 86.5%.

OpenAI costs are estimated from token usage; Jev costs are provider-reported. The pipeline total includes both stages. Local compute and training were not priced. Jev was inexpensive, but adding it to Astra reduced accuracy.

3. Did Jev need the LLM’s suggestion?

I gave Jev the same rows and examples, with and without Qwen3’s suggested answer. Across these four datasets, the comparison did not show a clear, consistent accuracy benefit from including that suggestion. A larger experiment could still find a benefit.

My takeaway: compare a two-model pipeline with each model alone and a trained classical baseline. Count both the mistakes a reviewer fixes and the correct answers it overturns.

This was a small study: 550 test examples and one train/test split per dataset. Some small models repeatedly chose a single class with our prompts and scoring. Public datasets may have appeared in model training. These results describe this setup; broader conclusions need more datasets and prompt variations.

When does adding Jev as a reviewer improve the final decision enough to justify the extra step?

#MachineLearning #LLM #ModelEvaluation #TabularData

---

Publication draft, not posted. General evaluation principles are expressed in original wording alongside the independently measured benchmark findings.

The plain-language proposal-control conclusion is supported by both primary and secondary analyses. It does not claim equivalence or that Qwen3 proposals are universally unhelpful. Full condition-level results, uncertainty intervals and operational accounting remain in [the full evidence story](STORY.md).

The TREC cost comparison is a retrospective per-pass reconstruction from recorded token usage and Jev-reported charges, not a new execution or an invoice. Source predictions were reused in the review experiment. See [cost derivation and scope](LINKEDIN_COST_NOTE.md).

Attach [the selected-example graphic](../completion_20260923/figures/selected_examples/when_second_model_helps.png). [All 48 review conditions](../completion_20260923/figures/review_outcomes.png) and [the full evidence story](STORY.md) provide context.

The repository and dashboard remain private. Add a repository link for readers only after access has been arranged. Do not imply that the code is already publicly accessible.
