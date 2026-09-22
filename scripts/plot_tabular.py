#!/usr/bin/env python3
"""Static tabular benchmark figures from the audited summary JSON only.

MPLCONFIGDIR=artifacts/matplotlib /opt/anaconda3/bin/python scripts/plot_tabular.py
No project runtime, model weights, network calls or raw datasets are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('titanic', 'breast_cancer', 'wine')
TITLES = {'titanic': 'Titanic · survival', 'breast_cancer': 'Breast Cancer · diagnosis', 'wine': 'Wine · cultivar'}
MODELS = (
    ('typesafe/jev-1.13', 'Jev 1.13', ('zero_shot', 'few_shot')),
    ('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen2.5 0.5B', ('zero_shot', 'few_shot', 'lora')),
    ('Qwen/Qwen3-4B-Instruct-2507', 'Qwen3 4B', ('zero_shot', 'few_shot', 'lora')),
    ('gpt-5.6-luna', 'GPT-5.6 Luna', ('zero_shot', 'few_shot')),
    ('gpt-6-astra', 'GPT-6 Astra', ('zero_shot', 'few_shot')),
)
CLASSICAL = (
    ('majority', 'Majority'), ('logistic_regression', 'Logistic regression'),
    ('rbf_svc', 'RBF SVM'), ('random_forest', 'Random forest'),
    ('hist_gradient_boosting', 'Hist. gradient boosting'),
)
COLORS = {'zero_shot': '#657589', 'few_shot': '#087E8B', 'lora': '#C56820',
          'classical_tabular': '#7654A2', 'full': '#234F86'}
MARKERS = {'zero_shot': 'o', 'few_shot': 's', 'lora': '^', 'classical_tabular': 'D', 'full': 'o'}
CI_LABEL = '95% unstratified group percentile bootstrap; fixed split, prompt, training selection and fitted model'


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return row['dataset'], row['model'], row['method'], row['train_per_class']


def load_evidence(path):
    summary = json.loads(Path(path).read_text())
    require(summary.get('schema_version') == 1 and summary.get('expected_runs') == 66, 'Expected audited 66-condition tabular summary')
    require(summary.get('seed') == 42 and summary.get('bootstrap_samples', 0) >= 100, 'Unexpected summary seed/bootstrap settings')
    require(set(summary['datasets']) == set(DATASETS), 'Expected exactly the three tabular datasets')
    expected = set()
    for dataset in DATASETS:
        for model, _, methods in MODELS:
            expected.update((dataset, model, method, 0 if method == 'zero_shot' else 4) for method in methods)
        expected.update((dataset, model, 'classical_tabular', budget) for model, _ in CLASSICAL for budget in (4, None))
    rows = {}
    for row in summary['runs']:
        identity = key(row)
        require(identity in expected and identity not in rows, 'Unexpected or duplicate summary condition')
        rows[identity] = row
        if row['status'] != 'complete':
            require(row.get('macro_f1') is None and row.get('group_bootstrap') is None, 'Pending rows must be unscored')
            continue
        info, bootstrap = summary['datasets'][row['dataset']], row['group_bootstrap']
        require(row['audit']['prepared_manifest_sha256'] == info['manifest_sha256'] == row['manifest_sha256'], 'Plotted row has no matching prepared audit')
        require(row['audit']['ordered_training_ids_verified'] and row['audit']['ordered_test_ids_verified'], 'Plotted row lacks ordered-ID audits')
        require(bootstrap['method'] == 'unstratified group percentile bootstrap' and bootstrap['group_definition'] == 'identical serialized test text SHA256', 'Expected whole-text-group confidence intervals')
        require(bootstrap['n_groups'] == info['test_text_clusters'] and bootstrap['n_rows'] == info['splits']['test']['rows'] == row['n_test'], 'Group/row count differs from summary audit')
        require(bootstrap['samples'] == summary['bootstrap_samples'] and bootstrap['seed'] == summary['seed'], 'Confidence settings differ')
        value = row['macro_f1']
        lo, hi = bootstrap['metrics']['macro_f1']['ci95']
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in (value, lo, hi)), 'Invalid plotted macro-F1 or interval')
        require(lo <= hi and math.isclose(value, bootstrap['metrics']['macro_f1']['estimate'], abs_tol=1e-12), 'Invalid interval or point-estimate mismatch')
        # A percentile CI need not contain the original estimate; draw endpoints
        # directly rather than forcing positive error-bar offsets around the dot.
    require(set(rows) == expected and len(rows) == 66, 'Missing inventory rows; rebuild the audit summary')
    require(sum(row['status'] == 'complete' for row in rows.values()) == summary['complete_runs'], 'Completed-run count differs')
    return summary, rows


def plot_row(row):
    result = {field: row.get(field) for field in ('dataset', 'model', 'display_model', 'method', 'method_label',
        'train_per_class', 'train_labels', 'status', 'n_test', 'n_failures', 'macro_f1', 'run_id', 'source_path', 'manifest_sha256')}
    result['ci95'] = row['group_bootstrap']['metrics']['macro_f1']['ci95'] if row['status'] == 'complete' else None
    result['n_groups'] = row['group_bootstrap']['n_groups'] if row['status'] == 'complete' else None
    result['artifact_sha256'] = row.get('artifact_sha256')
    return result


def style_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(['0', '.25', '.50', '.75', '1.00'])
    ax.set_xlabel('Macro-F1', fontsize=11, labelpad=9, color='#334155')
    ax.grid(axis='x', color='#DCE3EB', linewidth=.75, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=0, pad=8, labelcolor='#334155')
    for spine in ax.spines.values():
        spine.set_visible(False)


def dot_interval(ax, y, row, color, marker, *, value_column=True, pending_label='pending'):
    if row['status'] != 'complete':
        ax.text(.025, y, pending_label, transform=ax.get_yaxis_transform(), fontsize=9.3,
                color='#7A8797', va='center', fontstyle='italic')
        return
    lo, hi = row['group_bootstrap']['metrics']['macro_f1']['ci95']
    value = row['macro_f1']
    ax.hlines(y, lo, hi, color=color, linewidth=2, alpha=.8, zorder=3)
    ax.vlines([lo, hi], y-.075, y+.075, color=color, linewidth=1.2, zorder=3)
    ax.scatter([value], [y], marker=marker, s=56, color=color, edgecolors='white', linewidths=.85, zorder=4, clip_on=False)
    if value_column:
        ax.text(1.035, y, f'{value:.3f}', transform=ax.get_yaxis_transform(), fontsize=9.8,
                color=color, va='center', ha='left', fontweight='bold')


def main_inventory(rows):
    inventory = []
    for model, display, methods in MODELS:
        for method in methods:
            if method == 'lora':
                completed = [rows[(dataset, model, method, 4)] for dataset in DATASETS
                             if rows[(dataset, model, method, 4)]['status'] == 'complete']
                recorded = {row['method_label'].split(' 4/class')[0] for row in completed}
                suffix = next(iter(recorded)) if len(recorded) == 1 else 'adapter'
            else:
                suffix = 'zero-shot' if method == 'zero_shot' else 'few-shot'
            inventory.append({'model': model, 'method': method, 'budget': 0 if method == 'zero_shot' else 4,
                              'label': f'{display} · {suffix}', 'group': model})
    for model, display in CLASSICAL:
        inventory.append({'model': model, 'method': 'classical_tabular', 'budget': 4,
                          'label': f'{display} · 4/class', 'group': 'native classical'})
    return inventory


def make_main(summary, rows):
    inventory = main_inventory(rows)
    positions = list(reversed(range(len(inventory))))
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 10.5), sharey=True)
    fig.subplots_adjust(left=.255, right=.945, top=.79, bottom=.245, wspace=.28)
    fig.text(.04, .955, 'Tabular classification: identical held-out rows, matched label budgets',
             fontsize=20, fontweight='bold', color='#172738')
    fig.text(.04, .915, 'Named-feature serialization · fixed model order, not a ranking · failures remain in the scores',
             fontsize=11.5, color='#4C5B6D')
    complete = sum(rows[(dataset, item['model'], item['method'], item['budget'])]['status'] == 'complete'
                   for dataset in DATASETS for item in inventory)
    fig.text(.04, .88, f'{complete}/51 zero-shot or matched-label conditions complete; full-training native references are in the companion figure.',
             fontsize=10.8, color='#4C5B6D')
    for ax, dataset in zip(axes, DATASETS):
        info = summary['datasets'][dataset]
        labels = 4 * len(info['labels'])
        ax.set_title(TITLES[dataset], loc='left', fontsize=13.5, fontweight='bold', pad=35, color='#172738')
        ax.text(0, 1.032, f"{info['splits']['test']['rows']} rows · {info['test_text_clusters']} groups · {labels} matched labels",
                transform=ax.transAxes, fontsize=9.4, color='#536377')
        last_group = None
        start = positions[0]
        bands = []
        for position, item in zip(positions, inventory):
            if last_group is not None and item['group'] != last_group:
                bands.append((position+.5, start+.5))
                start = position
            last_group = item['group']
        bands.append((positions[-1]-.5, start+.5))
        for index, (bottom, top) in enumerate(bands):
            if index % 2 == 0:
                ax.axhspan(bottom, top, color='#EFF3F7', alpha=.7, zorder=0)
        for position, item in zip(positions, inventory):
            row = rows[(dataset, item['model'], item['method'], item['budget'])]
            dot_interval(ax, position, row, COLORS[item['method']], MARKERS[item['method']])
        ax.axhline(4.5, color='#CCD5E0', linewidth=1, zorder=1)
        ax.set_yticks(positions, [item['label'] for item in inventory], fontsize=10.3)
        ax.set_ylim(-.6, positions[0]+.6)
        style_axis(ax)
    legend = [Line2D([0], [0], marker=MARKERS[method], color='none', markerfacecolor=COLORS[method],
                    markeredgecolor='white', markersize=8, label=label) for method, label in
              [('zero_shot', 'Zero-shot: 0 new labels'), ('few_shot', 'Few-shot: 4/class'),
               ('lora', 'LoRA / QLoRA: 4/class'), ('classical_tabular', 'Native classical: 4/class')]]
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(.039, .153), ncol=4,
               frameon=False, fontsize=10.6, columnspacing=2.5)
    notes = [
        f"Whiskers: 95% whole-feature-group percentile bootstrap ({summary['bootstrap_samples']:,} resamples, seed {summary['seed']}); all rows in a sampled group stay together.",
        'Intervals condition on this split, serialization, prompt and fitted model; they are exploratory and unadjusted. Shared training IDs are audited.',
        'Qwen scores numeric class ID + EOS likelihoods; Jev uses native Choice; OpenAI generates labels. These output interfaces differ.',
        'No validation labels are used. Hosted LoRA is unavailable. Pending conditions have no dot; familiar public datasets may be present in pretraining.',
    ]
    for y, note in zip((.131, .102, .073, .044), notes):
        fig.text(.04, y, note, fontsize=10.1, color='#536377')
    evidence = {'chart': 'tabular-matched', 'metric': 'macro_f1', 'order': inventory,
                'ci_interpretation': CI_LABEL, 'complete_conditions': complete, 'expected_conditions': 51,
                'panels': {dataset: [plot_row(rows[(dataset, item['model'], item['method'], item['budget'])])
                                     for item in inventory] for dataset in DATASETS}}
    return fig, evidence


def make_native(summary, rows):
    positions = list(reversed(range(len(CLASSICAL))))
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.7), sharey=True)
    fig.subplots_adjust(left=.19, right=.955, top=.725, bottom=.345, wspace=.19)
    fig.text(.045, .94, 'Native tabular baselines: four labels per class versus full training',
             fontsize=19, fontweight='bold', color='#172738')
    fig.text(.045, .882, 'Additional training labels change the comparison · fixed estimator order and parameters · same test rows',
             fontsize=11.2, color='#4C5B6D')
    for ax, dataset in zip(axes, DATASETS):
        info = summary['datasets'][dataset]
        matched, full = 4 * len(info['labels']), info['splits']['train']['rows']
        ax.set_title(TITLES[dataset], loc='left', fontsize=13.2, fontweight='bold', pad=33, color='#172738')
        ax.text(0, 1.045, f'{matched} versus {full} training labels · {info["splits"]["test"]["rows"]} test rows',
                transform=ax.transAxes, fontsize=9.5, color='#536377')
        for position, (model, _) in zip(positions, CLASSICAL):
            if position % 2 == 0:
                ax.axhspan(position-.5, position+.5, color='#EFF3F7', alpha=.75, zorder=0)
            row4 = rows[(dataset, model, 'classical_tabular', 4)]
            rowfull = rows[(dataset, model, 'classical_tabular', None)]
            dot_interval(ax, position+.14, row4, COLORS['classical_tabular'], 'D', value_column=False, pending_label='4/class pending')
            dot_interval(ax, position-.14, rowfull, COLORS['full'], 'o', value_column=False, pending_label='full pending')
        ax.set_yticks(positions, [display for _, display in CLASSICAL], fontsize=11.1)
        ax.set_ylim(-.55, positions[0]+.55)
        style_axis(ax)
    legend = [Line2D([0], [0], marker=marker, color=color, linewidth=1.5, markerfacecolor=color,
                     markeredgecolor='white', markersize=8, label=label) for marker, color, label in
              [('D', COLORS['classical_tabular'], 'Matched: 4 labels/class'), ('o', COLORS['full'], 'Full training: additional labels')]]
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(.044, .165), ncol=2,
               frameon=False, fontsize=11, columnspacing=3)
    notes = [
        f"Whiskers: 95% unstratified whole-feature-group percentile bootstrap; {summary['bootstrap_samples']:,} resamples, seed {summary['seed']}.",
        'All preprocessing is fitted on the selected training rows. No validation labels, hyperparameter search or test-selected winner.',
        'Intervals are conditional and exploratory. Full-training results do not have the same label budget as the language-model few-shot/adapted arms.',
    ]
    for y, note in zip((.135, .098, .061), notes):
        fig.text(.045, y, note, fontsize=10.1, color='#536377')
    evidence = {'chart': 'tabular-native-label-budgets', 'metric': 'macro_f1',
                'order': [model for model, _ in CLASSICAL], 'ci_interpretation': CI_LABEL,
                'comparison_scope': 'Unequal training-label budgets; descriptive',
                'panels': {dataset: [plot_row(rows[(dataset, model, 'classical_tabular', budget)])
                                     for model, _ in CLASSICAL for budget in (4, None)] for dataset in DATASETS}}
    return fig, evidence


def save_figure(fig, evidence, stem, source, summary):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    evidence.update(summary_file=Path(source).name, summary_sha256=sha(source),
                    analysis_sha256=summary['analysis_sha256'], plot_script_sha256=sha(__file__),
                    bootstrap_samples=summary['bootstrap_samples'], seed=summary['seed'],
                    datasets=summary['datasets'])
    stem.with_suffix('.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    for suffix in ('png', 'svg'):
        metadata = {'Date': None} if suffix == 'svg' else {'Software': 'jevbench plot_tabular.py'}
        fig.savefig(stem.with_suffix('.' + suffix), dpi=220, bbox_inches='tight', pad_inches=.18, metadata=metadata)
        print(stem.with_suffix('.' + suffix))
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT / 'results/TABULAR_COMPARISON.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/figures')
    args = parser.parse_args(argv)
    summary, rows = load_evidence(args.summary)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'jevbench-tabular-v1',
                         'savefig.facecolor': 'white', 'axes.facecolor': 'white'})
    for maker, name in ((make_main, 'tabular-matched'), (make_native, 'tabular-native-label-budgets')):
        fig, evidence = maker(summary, rows)
        save_figure(fig, evidence, args.output_dir / name, args.summary, summary)


if __name__ == '__main__':
    main()
