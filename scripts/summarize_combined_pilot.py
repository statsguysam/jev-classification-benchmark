#!/usr/bin/env python3
"""Summarize recorded SST-2/TREC pilot runs without scoring unfinished work.

No API calls, raw datasets or model weights are loaded. Run this again after
importing Colab artifacts or finishing hosted jobs. Different manifest hashes
remain separate table groups; cross-environment audit evidence is linked apart.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('sst2', 'trec')
MODEL_ORDER = (
    'multinomial_nb', 'Qwen/Qwen2.5-0.5B-Instruct',
    'Qwen/Qwen3-4B-Instruct-2507', 'gpt-5.6-luna', 'gpt-6-astra', 'typesafe/jev-1.13',
)
DISPLAY = {
    'multinomial_nb': 'TF-IDF + Multinomial NB',
    'linear_svc': 'TF-IDF + linear SVM',
    'logistic_regression': 'TF-IDF + logistic regression',
    'Qwen/Qwen2.5-0.5B-Instruct': 'Qwen2.5 0.5B',
    'Qwen/Qwen3-4B-Instruct-2507': 'Qwen3 4B',
    'gpt-5.6-luna': 'GPT-5.6 Luna', 'gpt-6-astra': 'GPT-6 Astra', 'typesafe/jev-1.13': 'Jev 1.13 (OpenRouter)',
}
METHOD_ORDER = {'classical': 0, 'zero_shot': 1, 'few_shot': 2, 'lora': 3}
SCORE_FIELDS = ('accuracy', 'accuracy_ci_low', 'accuracy_ci_high', 'macro_f1',
                'macro_f1_ci_low', 'macro_f1_ci_high', 'bootstrap_samples',
                'n_test', 'n_failures', 'failure_rate', 'latency_p50_s', 'latency_p95_s',
                'probability_coverage', 'n_probability_rows', 'log_loss', 'brier_sum', 'ece_15_equal_width')


def read_json(path):
    return json.loads(path.read_text())


def row_from_run(path, record, root, track='matched/reference', selection='configuration fixed'):
    config = record['config']
    method, model = record['method'], config['model']
    training = record.get('adapter_training') or record.get('training') or {}
    budget = (config.get('train_per_class') if method == 'classical' else
              training.get('train_per_class') if method == 'lora' else config.get('shots_per_class', 0))
    complete = record.get('status') == 'complete'
    expected = 8 if record['dataset'] == 'sst2' else 24
    train_count = len(record['training_example_ids'])
    if track == 'matched/reference':
        if (method == 'zero_shot' and train_count != 0) or (method != 'zero_shot' and (budget != 4 or train_count != expected)):
            raise ValueError(f'Unexpected labeled budget in {path}')
        if training.get('validation_rows', 0):
            raise ValueError(f'Matched-label run uses development labels: {path}')
    if method == 'lora':
        method_label = 'QLoRA 4/class' if training.get('load_in_4bit') else 'LoRA 4/class'
    else:
        method_label = {'zero_shot': 'zero-shot', 'few_shot': 'few-shot 4/class',
                        'classical': 'full prepared' if budget is None else 'classical 4/class'}[method]
    provider = config.get('provider', 'classical')
    protocol = ('classical native' if method == 'classical' else
                'restricted-label likelihood' if provider in {'hf', 'huggingface'} else
                'native Choice' if provider in {'jev', 'typesafe'} else 'generated label')
    timing = ('amortized batch per row' if method == 'classical' else
              'local sequential class scoring' if provider in {'hf', 'huggingface'} else
              'hosted end-to-end request')
    row = dict(dataset=record['dataset'], model=model, display_model=DISPLAY.get(model, model),
               method=method, method_label=method_label, track=track,
               status='complete' if complete else f"pending ({record.get('status', 'unknown')})",
               seed=record['seed'], labels_per_class='full prepared' if budget is None else budget,
               train_labels=train_count, dev_labels=training.get('validation_rows', 0),
               label_count_basis='recorded', provider=provider, output_protocol=protocol, latency_basis=timing,
               run_id=record['run_id'], source_path=path.relative_to(root).as_posix(),
               manifest_sha256=record['manifest_sha256'], test_ids_sha256=record.get('test_ids_sha256', ''),
               selection=selection, selected_validation_macro_f1=training.get('selected_validation_macro_f1'),
               base_revision=config.get('revision', ''), reasoning_effort=config.get('reasoning_effort', ''),
               max_output_tokens=config.get('max_output_tokens'), training_quantized=training.get('load_in_4bit'),
               training_s=training.get('training_s', training.get('fit_and_selection_s')))
    row.update({field: None for field in SCORE_FIELDS})
    if complete:
        if not path.with_name('test_manifest.json').is_file() or not path.with_name('predictions.jsonl').is_file():
            row['status'] = 'pending (incomplete imported artifacts)'
            return row
        metrics = record['metrics']
        if metrics['n_test'] != 200:
            raise ValueError(f'This pilot report requires 200 completed test rows: {path}')
        manifest = read_json(path.with_name('test_manifest.json'))
        predictions = [json.loads(line) for line in path.with_name('predictions.jsonl').read_text().splitlines()]
        if len(predictions) != 200 or [p['row_id'] for p in predictions] != [p['id'] for p in manifest['rows']]:
            raise ValueError(f'Completed prediction/manifest alignment failed: {path}')
        if manifest['manifest_sha256'] != record['manifest_sha256'] or manifest['labels'] != record['labels']:
            raise ValueError(f'Completed manifest provenance differs: {path}')
        row['resolved_models'] = ';'.join(sorted({p.get('metadata', {}).get('resolved_model') for p in predictions if p.get('metadata', {}).get('resolved_model')}))
        row['resolved_revisions'] = ';'.join(sorted({p.get('metadata', {}).get('resolved_revision') for p in predictions if p.get('metadata', {}).get('resolved_revision')}))
        row['scoring_details'] = ';'.join(sorted({p.get('metadata', {}).get('scoring') for p in predictions if p.get('metadata', {}).get('scoring')}))
        row['probability_kind'] = ';'.join(sorted({p.get('metadata', {}).get('probability_kind') for p in predictions if p.get('metadata', {}).get('probability_kind')}))
        for field in ('input_tokens', 'output_tokens', 'input_tokens_coverage', 'output_tokens_coverage',
                      'input_tokens_known_total', 'output_tokens_known_total'):
            row[field] = metrics.get(field)
        for field in ('accuracy', 'macro_f1', 'n_test', 'n_failures', 'failure_rate', 'latency_p50_s', 'latency_p95_s',
                      'probability_coverage', 'n_probability_rows', 'log_loss', 'brier_sum', 'ece_15_equal_width'):
            row[field] = metrics.get(field)
        row['bootstrap_samples'] = metrics.get('bootstrap', {}).get('samples')
        for metric, prefix in (('accuracy', 'accuracy'), ('macro_f1', 'macro_f1')):
            interval = metrics.get('bootstrap', {}).get('metrics', {}).get(metric, {}).get('ci95')
            if interval:
                row[f'{prefix}_ci_low'], row[f'{prefix}_ci_high'] = interval
    return row


def sort_key(row):
    model = row['model']
    return (DATASETS.index(row['dataset']), MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER),
            model, METHOD_ORDER.get(row['method'], 9), row.get('source_path', ''))


def collect(root):
    rows, records, full_candidates = [], [], []
    for folder in ('pilot', 'colab', 'hosted', 'jev'):
        for path in sorted((root / 'results' / folder).glob('*/run.json')):
            record = read_json(path)
            if record.get('dataset') not in DATASETS or record.get('seed') != 42:
                continue
            config, method = record['config'], record['method']
            if method == 'classical':
                # Keep the declared fixed reference in each local environment.
                if folder not in {'pilot', 'colab'}:
                    continue
                if config.get('train_per_class') is None and config['model'] != 'majority':
                    if folder == 'pilot':
                        full_candidates.append((path, record))
                    continue
                if config.get('model') != 'multinomial_nb' or config.get('train_per_class') != 4:
                    continue
            elif method not in {'zero_shot', 'few_shot', 'lora'}:
                continue
            elif method == 'few_shot' and config.get('shots_per_class') != 4:
                continue
            elif method == 'lora' and record.get('adapter_training', {}).get('train_per_class') != 4:
                continue
            elif method == 'zero_shot' and config.get('shots_per_class', 0) != 0:
                continue
            row = row_from_run(path, record, root)
            rows.append(row)
            records.append((path, record))
    # Explicitly labeled descriptive test-best and independently validation-selected reference.
    for dataset in DATASETS:
        candidates = [(p, r) for p, r in full_candidates if r['dataset'] == dataset and r.get('status') == 'complete']
        if not candidates:
            continue
        test_best = max(candidates, key=lambda pr: (pr[1]['metrics']['macro_f1'], pr[1]['config']['model']))
        validation_candidates = [(p, r) for p, r in candidates if r['training'].get('selected_validation_macro_f1') is not None]
        validation_best = max(validation_candidates, key=lambda pr: (pr[1]['training']['selected_validation_macro_f1'], pr[1]['config']['model'])) if validation_candidates else None
        chosen = [test_best] + ([validation_best] if validation_best and validation_best[0] != test_best[0] else [])
        for path, record in chosen:
            selection = ('observed test-best; post-hoc descriptive' if path == test_best[0] else 'highest recorded validation macro-F1')
            if validation_best and path == test_best[0] == validation_best[0]:
                selection += '; also highest validation macro-F1'
            rows.append(row_from_run(path, record, root, 'full prepared (extra labels)', selection))
    # Missing runs stay visible as unscored inventory entries.
    keys = {(r['dataset'], r['model'], r['method']) for r in rows}
    for dataset in DATASETS:
        for model in MODEL_ORDER:
            methods = ('classical',) if model == 'multinomial_nb' else ('zero_shot', 'few_shot', 'lora') if model.startswith('Qwen/') else ('zero_shot', 'few_shot')
            for method in methods:
                if (dataset, model, method) in keys:
                    continue
                status = 'pending (no imported run artifact)'
                label = {'classical': 'classical 4/class', 'zero_shot': 'zero-shot', 'few_shot': 'few-shot 4/class', 'lora': 'QLoRA 4/class' if model.endswith('4B-Instruct-2507') else 'LoRA 4/class'}[method]
                rows.append(dict(dataset=dataset, model=model, display_model=DISPLAY[model], method=method,
                                 method_label=label, track='matched/reference', status=status, seed=42,
                                 labels_per_class=0 if method == 'zero_shot' else 4,
                                 train_labels=0 if method == 'zero_shot' else 8 if dataset == 'sst2' else 24,
                                 dev_labels=0, label_count_basis='planned', source_path='', run_id='',
                                 manifest_sha256='', test_ids_sha256='', selection='planned condition',
                                 **{field: None for field in SCORE_FIELDS}))
    return sorted(rows, key=sort_key), records


def fmt(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def score(row, metric):
    value = fmt(row.get(metric))
    low, high = row.get(f'{metric}_ci_low'), row.get(f'{metric}_ci_high')
    return value if low is None or high is None else f'{value} [{low:.4f}, {high:.4f}]'


def link(path, output):
    return os.path.relpath(path, output.parent).replace(os.sep, '/')


def summarize(root, output):
    rows, records = collect(root)
    primary = [r for r in rows if r['track'] == 'matched/reference']
    complete = [r for r in primary if r['status'] == 'complete']
    pending = [r for r in primary if r['status'] != 'complete']
    # Within an identical manifest, verify matched training IDs rather than infer
    # equality just from a count. Cross-manifest equivalence remains an audit task.
    matched_ids = {}
    for path, record in records:
        if record['method'] == 'zero_shot':
            continue
        key = (record['dataset'], record['manifest_sha256'])
        ids = record['training_example_ids']
        if key in matched_ids and matched_ids[key] != ids:
            raise ValueError(f'Matched training IDs differ within one manifest group: {path}')
        matched_ids[key] = ids
    groups = {}
    for dataset in DATASETS:
        hashes = sorted({r['manifest_sha256'] for r in rows if r['dataset'] == dataset and r.get('manifest_sha256')})
        groups.update({(dataset, digest): f'{dataset}-M{i+1}' for i, digest in enumerate(hashes)})
    for row in rows:
        row['manifest_group'] = groups.get((row['dataset'], row.get('manifest_sha256')), '')
    lines = ['# Combined SST-2 / TREC pilot', '',
             f'This report contains **{len(complete)} completed model/reference rows**. Scores come from completed saved runs. Planned or unfinished rows without scores: {len(pending)}.', '',
             'Each dataset has **200 test rows**, with training examples selected using seed 42. Every method sees the same first 2,000 characters of each input. Zero-shot uses no new task examples. Four-per-class prompting, adaptation and the fixed Naive Bayes reference use eight labeled examples for SST-2 or 24 for TREC, with no development labels. Pretraining data and compute differ between models.', '',
             'The 0.5B adapter uses ordinary LoRA; the 4B adapter uses QLoRA when its recorded metadata confirms four-bit training. Both use normal configured precision for inference. Hosted LoRA, including Jev LoRA, is unavailable. Jev runs use the exact requested OpenRouter model `typesafe/jev-1.13` with native Choice probabilities; absent or incomplete runs remain pending and are never assigned a zero score.', '',
             '## Completed runs by exact manifest group', '',
             'Different complete manifest hashes are kept in separate groups. Membership alone is not proof that two environments prepared equivalent data. Cross-environment equivalence requires the separately linked audit of ordered IDs, labels, text/content hashes and training provenance; this summarizer does not infer paired contrasts across hashes.', '']
    for dataset in DATASETS:
        for group in sorted({r['manifest_group'] for r in complete if r['dataset'] == dataset}):
            selected = [r for r in complete if r['dataset'] == dataset and r['manifest_group'] == group]
            lines += [f'### {dataset.upper()} · {group}', '',
                      '| Model | Method | Train / dev labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Errors | Artifact |',
                      '|---|---|---:|---|---|---:|---|']
            for r in selected:
                lines.append(f"| {r['display_model']} | {r['method_label']} | {r['train_labels']} / {r['dev_labels']} | {score(r, 'accuracy')} | {score(r, 'macro_f1')} | {r['n_failures']} | [run]({link(root/r['source_path'], output)}) |")
            lines.append('')
    lines += ['## Pending and unavailable conditions', '',
              '| Dataset | Model | Method | State | Existing artifact |', '|---|---|---|---|---|']
    for r in pending:
        artifact = f"[run]({link(root/r['source_path'], output)})" if r['source_path'] else 'N/A'
        lines.append(f"| {r['dataset']} | {r['display_model']} | {r['method_label']} | {r['status']} | {artifact} |")
    if not pending:
        lines.append('| N/A | N/A | N/A | No pending conditions in this inventory | N/A |')
    lines += ['', '## Larger supervised reference: extra training and development labels', '',
              'The observed test-best classical result is included only as a **post-hoc descriptive reference**, not a prespecified winner. Its ordinary bootstrap interval does not adjust for choosing the best test score. The highest-validation-score estimator is also shown when it differs. Neither is an equal-label comparison with the k=4 arms. Candidate families are logistic regression, linear SVM and Multinomial NB; each uses its recorded three-candidate development search.', '',
              '| Dataset / group | Model | Reference selection | Train / dev labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Artifact |',
              '|---|---|---|---:|---|---|---|']
    for r in [r for r in rows if r['track'] != 'matched/reference']:
        lines.append(f"| {r['manifest_group']} | {r['display_model']} | {r['selection']} | {r['train_labels']} / {r['dev_labels']} | {score(r, 'accuracy')} | {score(r, 'macro_f1')} | [run]({link(root/r['source_path'], output)}) |")
    lines += ['', '## Recorded latency and output protocol', '',
              'Seconds below use each run\'s own timing convention. Hosted end-to-end requests, local sequential candidate scoring and amortized classical batch timings are not equal-hardware throughput measurements. Model loading is excluded from per-row local timing; training time and all underlying fields are retained in the CSV/run artifacts.', '',
              '| Dataset | Model / method | Output protocol | Timing basis | p50 seconds | p95 seconds |',
              '|---|---|---|---|---:|---:|']
    for r in [r for r in rows if r['status'] == 'complete']:
        lines.append(f"| {r['dataset']} | {r['display_model']} / {r['method_label']} | {r['output_protocol']} | {r['latency_basis']} | {fmt(r['latency_p50_s'], 6)} | {fmt(r['latency_p95_s'], 6)} |")
    lines += ['', '## Provenance and separate audit evidence', '']
    for (dataset, digest), group in sorted(groups.items()):
        lines.append(f'* `{group}`: complete prepared-manifest SHA-256 `{digest}`.')
    colab_audit_path = root/'results/COLAB_AUDIT.json'
    if colab_audit_path.is_file():
        colab_audit = read_json(colab_audit_path)
        lines.append(f"* [Colab import audit]({link(colab_audit_path, output)}): recorded pass `{colab_audit.get('passed')}`, {colab_audit.get('runs_verified')} imported runs verified, {colab_audit.get('classical_replications_verified')} classical replications checked. Original manifest hashes remain distinct; see the audit for content/training identity evidence and scope.")
    audits = sorted((root/'results/comparisons').glob('cross_environment_*.json'))
    if audits:
        for path in audits:
            audit = read_json(path)
            eligible = audit.get('eligible_for_separately_labeled_matched_pairing')
            lines.append(f"* [Cross-environment audit: {path.stem}]({link(path, output)}): recorded pairing eligibility `{eligible}`. Review the audit's exact run IDs and checks; its evidence applies only to the named pair.")
    elif not colab_audit_path.is_file():
        lines.append('* No cross-environment audit artifact has been imported yet. Different manifest groups remain unpaired in this report.')
    paired_index = root/'results/comparisons/combined/index.md'
    if paired_index.is_file():
        lines.append(f'* [Separately computed paired contrasts and content audits]({link(paired_index, output)}). These are exploratory contrasts with unadjusted intervals; review each named pair and audit before interpreting its difference.')
    samples = sorted({r['bootstrap_samples'] for r in rows if r.get('bootstrap_samples') is not None})
    lines += ['', f"Displayed intervals come from recorded stratified test-item bootstrap results (resamples present: {', '.join(map(str, samples)) or 'none'}). They condition on the fixed train/test split and exclude training-seed, prompt-selection and public-pretraining contamination uncertainty. Individual intervals are not paired difference tests. No global model ranking or hallucination-free claim follows.", '',
              'The fixed small-model findings are retained, including TREC LoRA collapse. SST-2 LoRA-minus-few-shot uncertainty and TREC pairing diagnostics are documented in [NEURAL_PILOT.md](NEURAL_PILOT.md). Probability metrics from native classifiers, Jev and restricted-label likelihoods have different semantics; hosted label generation does not supply a complete class distribution.', '',
              'Table model names identify the requested model. The CSV retains returned model identifiers and resolved open-weight revisions when recorded; a missing returned snapshot does not establish immutability. Local numeric-ID-plus-EOS likelihood scoring and hosted generated-label parsing are different protocols. Their reported accuracies describe those exact configurations, not each model\'s best possible result.', '',
              'Token usage in the CSV is reported usage with its coverage, not provider billing. Known totals do not make missing usage zero. Budget reservations and conservative settlement also differ from an actual invoice; see [BUDGET.md](../docs/BUDGET.md). No cost ranking is inferred here.', '',
              'Rebuild with `python scripts/summarize_combined_pilot.py`. The CSV includes pending rows with blank score fields, full source paths, exact manifest/test hashes, label budgets, output protocols, latency definitions and selection notes. It does not contain corpus text or credentials.', '']
    figure = root/'results/figures/combined-pilot.png'
    figure_data = figure.with_suffix('.json')
    if figure.is_file() and figure_data.is_file():
        plotted = [r for panel in read_json(figure_data)['panels'].values() for r in panel['rows']]
        actual = {r['run_id']: r for r in complete}
        if all(r['status'] == 'complete' and r['run_id'] in actual
               and all(r[k] == actual[r['run_id']][k] for k in ('macro_f1', 'macro_f1_ci_low', 'macro_f1_ci_high', 'manifest_sha256'))
               for r in plotted):
            lines += ['## Reproducible comparison figure', '',
                      f'![Measured model/method macro-F1 and conditional intervals]({link(figure, output)})', '',
                      f'[SVG]({link(figure.with_suffix(".svg"), output)}) · [Plotted values and cross-environment audit evidence]({link(figure_data, output)}). Rebuild with `python scripts/plot_combined_pilot.py` in an environment with Matplotlib.', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.with_suffix('.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return {'complete_primary_rows': len(complete), 'unscored_primary_rows': len(pending), 'full_prepared_reference_rows': sum(r['track'] != 'matched/reference' for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root/'results/COMPARISON.md'
    print(json.dumps(summarize(root, output), sort_keys=True))


if __name__ == '__main__':
    main()
