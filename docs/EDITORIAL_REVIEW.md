# Repository wording review, September 24, 2026

The review covered the README, documentation, result narratives, article and post drafts, notebook instructions, and dashboard copy. It also checked tracked text for em dashes, boilerplate phrases and broken local documentation links.

The README now leads with the research question, three concrete findings, and a dataset table. Longer explanations were shortened or split into paragraphs. Dashboard tables use `N/A` for unavailable values, keeping them distinct from zero. The text-extension protocol's old pending status was corrected to 68 completed conditions.

The measurements, saved predictions, statistical caveats, source credits and original experiment records remain intact. The saved Medium article already contained no em dashes and was preserved with its text and chart checksums.

## Frozen files

Four files retain their original punctuation because their exact bytes are part of the experiment's recorded provenance:

- `docs/REVIEW_CONTROLS_PROTOCOL.md`: two empty cells in the total row of its design table.
- `scripts/summarize_tabular.py`: historical table headings and unavailable-value placeholders.
- `scripts/summarize_expanded_numeric.py`: unavailable-value placeholders.
- `scripts/summarize_text_extension.py`: an unavailable-value placeholder.

Changing these files would invalidate recorded checksums. The remaining em dashes are technical formatting in these frozen files, not narrative asides. The report text, dashboard pages and publication drafts use the revised punctuation.

## Rebuilding the older tabular report

`results/TABULAR_COMPARISON.md` has a typography-only edit. Its frozen producer still writes the original headings and placeholders. After rebuilding that report, apply this display cleanup to reproduce the checked-in Markdown:

```bash
python - <<'PY'
from pathlib import Path

path = Path('results/TABULAR_COMPARISON.md')
text = path.read_text()
dash = chr(0x2014)
headings = {
    f'Paired contrasts {dash} main matched comparison':
        'Paired contrasts with matched labels',
    f'Paired contrasts {dash} descriptive: few-shot versus zero-shot; unequal labels':
        'Few-shot versus zero-shot (descriptive; unequal labels)',
    f'Paired contrasts {dash} descriptive: full versus 4/class; unequal labels':
        'Full training versus four examples per class (descriptive; unequal labels)',
}
for original, revised in headings.items():
    text = text.replace(original, revised)
path.write_text(text.replace(dash, 'N/A'))
PY
```

This changes only Markdown typography. The JSON and CSV remain the numerical record.
