I started with a simple question: does adding Jev improve an LLM's classification decisions on numerical tables?

The more useful question turned out to be: **does the first LLM add anything?**

I compared LLMs alone, Jev alone, LLM proposals reviewed by Jev, and classical ML on the same held-out rows, using fixed prompts and scoring rules.

One result made me rethink the comparison:

On Wine, Qwen2.5 0.5B improved from 33.3% to 91.7% after Jev review, using four examples per class. A huge gain—but all 36 final labels were identical to Jev alone.

That demonstrates a rescue of a weak first-stage result. It doesn't demonstrate that we needed two models.

Review could also hurt: a zero-shot frontier-model run went from 36/36 correct to 16/36 after Jev review, with no review API failures.

There was an encouraging lead too. In a simulation using saved Qwen3 few-shot predictions, reviewing the least-confident half of the rows matched full-review accuracy on both numerical datasets:

• Breast Cancer: 93.0%, with 57 rather than 114 reviews.
• Wine: 88.9%, with 18 rather than 36 reviews.

But this routing strategy underperformed random selection for another source. And classical ML stayed competitive: logistic regression reached 97.2% on Wine using the same twelve training labels.

My takeaway: a bounded answer is a valid category, not a guarantee of a better decision. Evaluate what a reviewer corrects, what it damages, and whether the first stage earns its place.

This is an exploratory pilot: one split, 114 Breast Cancer cases and 36 Wine cases, familiar public datasets, and some review conditions still unfinished. The selective-review result is a hypothesis—not a validated production policy or a measured cost saving.

I've prepared a matched real-proposal / no-proposal / shuffled-proposal follow-up to test the explanation more directly. Those calls have not run yet.

What would you require before trusting a second model to override the first?

#MachineLearning #LLM #TabularData #ModelEvaluation

---

Draft only. Attach `figures/selective_review.png`. Add a repository URL only once readers have access; the current repository and dashboard are private. Full evidence and limitations: `FINDINGS.md` and `STORY.md`. The scoring diagnostic is separate and should not be described as a test-set result.
