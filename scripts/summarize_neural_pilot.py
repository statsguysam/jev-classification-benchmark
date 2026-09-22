"""Regenerate the small-model pilot narrative from recorded, completed runs."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
METHOD_ORDER = {'zero_shot': 0, 'few_shot': 1, 'lora': 2}


def load(path):
    return json.loads(path.read_text())


def main():
    runs = [load(p) for p in sorted((ROOT / 'results/pilot').glob('*/run.json'))]
    runs = [r for r in runs if r.get('status') == 'complete']
    neural = sorted([r for r in runs if r['config'].get('provider') in {'hf', 'huggingface'} and r['config']['model'] == MODEL],
                    key=lambda r: (r['dataset'], METHOD_ORDER[r['method']], r['seed']))
    if not neural:
        raise ValueError('No completed small-model runs found')
    by_key = {(r['dataset'], r['method']): r for r in neural}
    if len(by_key) != len(neural) or {r['seed'] for r in neural} != {42}:
        raise ValueError('Update the single-seed pilot description before adding more seeds')
    revisions = {r['config']['revision'] for r in neural}
    if len(revisions) != 1:
        raise ValueError('The pilot contains multiple model revisions')
    predictions = {}
    for r in neural:
        directory = ROOT / 'results/pilot' / r['run_id']
        rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
        manifest = load(directory / 'test_manifest.json')
        if [p['row_id'] for p in rows] != [p['id'] for p in manifest['rows']]:
            raise ValueError(f"Prediction alignment failed: {r['run_id']}")
        if len(rows) != r['metrics']['n_test'] or len(rows) != 200:
            raise ValueError('Update the fixed 200-example pilot description')
        predictions[r['run_id']] = rows
    for dataset in sorted({r['dataset'] for r in neural}):
        group = [r for r in neural if r['dataset'] == dataset]
        if len({r['test_ids_sha256'] for r in group}) != 1:
            raise ValueError(f'Test sets differ for {dataset}')
        few, lora = by_key.get((dataset, 'few_shot')), by_key.get((dataset, 'lora'))
        if few and lora and few['training_example_ids'] != lora['training_example_ids']:
            raise ValueError(f'Training examples differ for {dataset}')
        if lora:
            t = lora['adapter_training']
            recipe = (t['epochs'], t['r'], t['lora_alpha'], t['learning_rate'], t['train_per_class'])
            if recipe != (3, 8, 16, 0.0002, 4) or t['validation_rows'] or t['test_accessed']:
                raise ValueError('Update the fixed, no-development-label LoRA description')
    total_predictions = sum(len(rows) for rows in predictions.values())
    failures = sum(r['metrics']['n_failures'] for r in neural)
    probability_rows = sum(r['metrics']['n_probability_rows'] for r in neural)
    revision = next(iter(revisions))
    lines = [
        '# Open-model pilot: measured results', '',
        f'**{len(neural)} completed runs and {total_predictions:,} recorded predictions** for **Qwen2.5-0.5B-Instruct**, pinned revision `{revision}`. These are exploratory local-MPS results. This document covers only the small-model arm; see [COMPARISON.md](COMPARISON.md) for Colab, OpenAI and Jev measurements; this report does not summarize those arms.', '',
        'Every row uses the same 200 held-out examples per dataset and the shared 2,000-character input policy. Selection seed is 42. Local inference scores the complete numeric class ID plus EOS and normalizes likelihood over the permitted labels. This is a specific verbalizer/scoring protocol, not a claim about the best achievable performance of this model.', '',
        'Few-shot and LoRA both use exactly four training examples per class: eight for SST-2 and 24 for TREC. Matched classical references use those same examples, with no development labels. LoRA uses a fixed three epochs, rank 8, alpha 16, learning rate 0.0002 and response-only supervision. Both adapters use their final checkpoint; no test-guided retuning or checkpoint selection was performed.', '',
        '| Dataset | Method | New training labels | Accuracy | Macro F1 | 95% macro-F1 CI | NLL | Brier sum | ECE |',
        '|---|---|---:|---:|---:|---|---:|---:|---:|',
    ]
    for r in neural:
        m = r['metrics']; ci = m['bootstrap']['metrics']['macro_f1']['ci95']
        lines.append(f"| {r['dataset']} | {r['method']} | {len(r['training_example_ids'])} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {m['log_loss']:.4f} | {m['brier_sum']:.4f} | {m['ece_15_equal_width']:.4f} |")
    lines += ['', 'Intervals above use 1,000 stratified test-item bootstrap resamples. They are conditional on these fixed test sets, training subsets and one selection seed. NLL, Brier and 15-bin ECE describe normalized restricted-label likelihoods; these are not Jev-native probabilities.', '',
              '## Fixed classical references', '',
              'These references use the same test items. The larger supervised track spends extra training and development labels and is not an equal-label comparison.', '',
              '| Dataset | Model | Training labels | Development labels | Accuracy | Macro F1 |', '|---|---|---:|---:|---:|---:|']
    for dataset in sorted({r['dataset'] for r in neural}):
        for model in ['logistic_regression', 'linear_svc', 'multinomial_nb']:
            for budget in [4, None]:
                records = [r for r in runs if r['dataset'] == dataset and r['method'] == 'classical' and r['config']['model'] == model and r['seed'] == 42 and r['config']['train_per_class'] == budget]
                for r in records:
                    m, t = r['metrics'], r['training']
                    lines.append(f"| {dataset} | {model} | {t['training_rows']} | {t['validation_rows']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
    lines += ['', '## Interpretation and limits', '']
    comparison_path = ROOT / 'results/comparisons/sst2-qwen-lora-k4-vs-few4.json'
    if ('sst2', 'lora') in by_key and comparison_path.exists():
        comparison = load(comparison_path)
        if comparison['run_a'] != by_key[('sst2', 'lora')]['run_id'] or comparison['run_b'] != by_key[('sst2', 'few_shot')]['run_id'] or not comparison['equal_training_ids_and_seed']:
            raise ValueError('Paired comparison does not identify the matched SST-2 runs')
        contrast = comparison['metrics']['macro_f1']; low, high = contrast['ci95']
        caveat = 'includes zero' if low <= 0 <= high else 'excludes zero'
        lines.append(f"* SST-2 LoRA minus few-shot has an observed macro-F1 difference of {contrast['estimate']:+.4f}. The paired {comparison['samples']:,}-resample 95% interval is [{low:+.4f}, {high:+.4f}] and {caveat}. This small, single-seed pilot does not establish a clear LoRA advantage.")
    if ('trec', 'lora') in by_key:
        r = by_key[('trec', 'lora')]; m = r['metrics']
        label, count = Counter(p['label'] for p in predictions[r['run_id']]).most_common(1)[0]
        lines.append(f"* TREC LoRA collapsed toward **{r['labels'][label]}**, choosing it on **{count}/{m['n_test']} examples ({count / m['n_test']:.0%})**. Its accuracy is {m['accuracy']:.4f} and macro-F1 is {m['macro_f1']:.4f}. This unfavorable result is retained. No prompt, verbalizer, training recipe or checkpoint was retuned against the observed test scores; its cause is not established.")
        comparison_path = ROOT / 'results/comparisons/trec_lora_k4_minus_few4_seed42.json'
        if comparison_path.exists():
            comparison = load(comparison_path)
            if comparison['run_a'] != r['run_id'] or comparison['run_b'] != by_key[('trec', 'few_shot')]['run_id'] or not comparison['equal_training_ids_and_seed']:
                raise ValueError('Paired comparison does not identify the matched TREC runs')
            contrast = comparison['metrics']['macro_f1']; low, high = contrast['ci95']
            lines.append(f"* TREC LoRA minus few-shot macro-F1 is {contrast['estimate']:+.4f}, with paired {comparison['samples']:,}-resample 95% interval [{low:+.4f}, {high:+.4f}]. This describes a decline for these fixed runs; it does not identify the cause or incorporate training-seed/prompt-selection uncertainty. [Paired comparison](comparisons/trec_lora_k4_minus_few4_seed42.json).")
        lines.append(f"* TREC LoRA used {r['adapter_training']['optimizer_steps']} optimizer steps on 24 rows. Three epochs and falling training loss do not establish convergence. The [pairing audit](comparisons/trec_pairing_audit_seed42.json) verifies shared test/training IDs, four examples per class, no development-label use, no train/test text overlap and the training prompt hash.")
    lines += [
        '* The weak zero-shot and TREC results are specific to the fixed numeric-label/EOS likelihood protocol. Native generation, other verbalizers, prompt sensitivity and contextual calibration are separate, unrun experiments.',
        f'* There are {failures} inference errors across {total_predictions:,} predictions and {probability_rows:,}/{total_predictions:,} valid class distributions. Valid output structure does not imply correct classification.',
        '* Local training uses float32 MPS; inference uses float16 MPS. Inference recomputes the context for each candidate class. Some neural execution overlapped local CPU benchmark/analysis work. Timing is exploratory and not an isolated serving or architecture-level speed comparison. Model download/loading is outside per-row timing.',
        '* LLM pretraining data and compute are not matched to classical models. Widely known public datasets may have appeared in pretraining. Equal new task labels do not imply equal total learning resources.',
        '* Classical seed sensitivity is reported separately in [MATCHED_ANALYSIS.md](MATCHED_ANALYSIS.md). Neural pilots use one seed; full-test-set and multi-seed neural results remain pending.', '',
        '## Recorded neural artifacts', '',
    ]
    for r in neural:
        lines.append(f"* {r['dataset']} / {r['method']}: [run metadata and metrics](pilot/{r['run_id']}/run.json), [per-example predictions](pilot/{r['run_id']}/predictions.jsonl).")
    lines += ['', 'Rebuild this document with `python scripts/summarize_neural_pilot.py`.', '',
              '[Execution status](STATUS.md) · [SST-2 paired comparison](comparisons/sst2-qwen-lora-k4-vs-few4.json) · [Full pilot table](REPORT.md) · [Protocol](../docs/PROTOCOL.md)', '']
    output = ROOT / 'results/NEURAL_PILOT.md'
    output.write_text('\n'.join(lines))
    print(output)


if __name__ == '__main__':
    main()
