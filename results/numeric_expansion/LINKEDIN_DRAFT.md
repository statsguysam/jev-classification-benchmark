# LinkedIn draft

Draft only; not posted. The repository and dashboard are currently private, so arrange public access or a public evidence copy before sharing their links. The current completion status must remain visible when sharing. Use the dashboard to inspect each condition; final publication figures require the completed review matrix.

---

I tested whether a bounded decision reviewer could improve numerical classification.

I tested six LLMs (Qwen 0.5B/4B, SmolLM2, Granite, GPT Luna and Astra) on serialized Breast Cancer and Wine data, with zero examples and four examples per class. Jev reviews each saved class proposal; direct Jev and classical ML provide references.

The source runs are complete; Jev review is complete in 17 of 24 settings, with the remainder paused by provider billing. These examples come from completed settings.

Three results made me pause:

• Astra's zero-shot Wine accuracy fell from 100% to 44.4% after Jev review. Twenty correct answers became wrong labels. There were no API failures in that condition.

• Qwen2.5 0.5B's few-shot Wine accuracy rose from 33.3% to 91.7%. But every reviewed prediction matched direct Jev. The large improvement over Qwen added no accuracy over Jev alone here.

• With the same twelve Wine training examples, native-feature logistic regression reached 97.2% and random forest 94.4%. Those simple baselines deserve a place in the comparison.

I'm learning to ask two questions: did review improve the source, and did the source add anything beyond the reviewer alone? Bounded outputs can still be wrong.

This is exploratory: two small public datasets, one split, and possible pretraining exposure. Wine's cultivar IDs are arbitrary without examples. These results don't establish general model superiority.

Code, predictions and protocol: https://github.com/statsguysam/jev-classification-benchmark

How would you test whether a reviewer adds value to a classifier?
