# Medium story package

[Open the saved Medium draft](https://medium.com/p/829da0026f3e/edit)

**What Happened When I Added Jev to a Classification Pipeline**

Prepared for Salim Shaikh and saved in his Medium account on September 24, 2026. The article is an unpublished draft. The existing GitHub repository and dashboard retain their private access settings.

- [Article in Markdown](ARTICLE.md)
- [Formatted local preview](ARTICLE.html)
- [Seven chart captions](figures/CAPTIONS.md)
- [Chart inputs and file hashes](figures/MANIFEST.json)
- [Medium save and verification metadata](MEDIUM_DRAFT.json)

The seven PNG charts are embedded in Medium in this order: Wine result, TREC overrides, all 48 review conditions, proposal controls, Jev zero/few-shot, matched classical context, and API cost/accuracy. Each has a native caption and an alt description. SVG companions support further editing or export.

`MEDIUM_PASTE.html` is an editing template with image placeholders; it is not the finished article. `MEDIUM_IMAGES.json` maps those placeholders to figure files. `MEDIUM_SAVED_TEXT.txt` was read from the editor after all chart and caption edits. Its text exactly matches the authored article after whitespace normalization and exclusion of image alt attributes. The draft was reloaded to verify persistent text, all seven Medium-hosted images, all seven captions and all seven alt descriptions.

## Evidence and editorial decisions

The story uses the frozen primary numerical/text results and the original matched proposal comparison. Headline counts and method descriptions were independently reviewed against [the evidence story](../review_value/STORY.md), the [numeric report](../numeric_expansion/COMPARISON.json), the [text report](../text_extension/COMPARISON.json), and [controlled comparisons](../review_controls/COMPARISON.json). The [cost note](../review_value/LINKEDIN_COST_NOTE.md) distinguishes OpenAI token-based estimates, Jev-reported charges and reconstructed two-stage cost.

The proposal-control follow-up used already-inspected test rows and is explicitly exploratory. The uncertainty ranges describe the saved test examples, not variation across new prompts or splits. Matching newly supplied labels does not equalize pretraining. One-class local predictions are disclosed. Earlier LoRA work remains separate from this focused reviewer comparison.

The public prose omits operational-response discussion as requested. The original complete reports and full-denominator scoring remain unchanged. In particular, the all-48 accuracy-change heatmap must not be described as a chart of accepted wrong-label overrides: it reflects the full original metrics. The selected TREC four-examples-per-class example independently supports the three-correction/twenty-four-overturn finding.

Official TypeSafe and dataset sources were checked while preparing the article. The article makes no first-ever-study claim, no production cost-saving claim, and no statement that the code or dashboard is publicly accessible. No new model calls, training or billing actions were needed.

## Reproduce the figures

From the repository root:

```bash
.venv/bin/python scripts/plot_medium_story.py
```

The script checks saved prediction identities, row/example alignment and exact TREC cost sums before plotting. It writes PNG/SVG charts, captions, chart data and SHA-256 provenance. All generated charts were visually inspected. The saved figure and evidence hashes were checked again before this package was committed.
