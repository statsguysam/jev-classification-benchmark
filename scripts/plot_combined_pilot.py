#!/usr/bin/env python3
"""Plot the completed SST-2/TREC model panel in fixed, non-ranked order.

MPLCONFIGDIR=artifacts/matplotlib /opt/anaconda3/bin/python scripts/plot_combined_pilot.py

The plot uses recorded intervals. It verifies the separate local/Colab data
content audit before sharing one fixed NB reference, and writes that proof and
all plotted values to a companion JSON. It never changes any run artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from audit_cross_environment import audit, read_run, require
from summarize_combined_pilot import ROOT, collect

MODELS = (
    ('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen2.5 0.5B', '#147D92'),
    ('Qwen/Qwen3-4B-Instruct-2507', 'Qwen3 4B', '#3565B4'),
    ('gpt-5.6-luna', 'GPT-5.6 Luna', '#8A58A6'),
    ('gpt-6-astra', 'GPT-6 Astra', '#C36A24'),
    ('typesafe/jev-1.13', 'Jev 1.13', '#187B52'),
)
METHODS = ('zero_shot', 'few_shot', 'lora')
MARKERS = {'zero_shot': 'o', 'few_shot': 's', 'lora': '^'}


def panel_data(root):
    rows, _ = collect(root)
    rows = [row for row in rows if row['track'] == 'matched/reference']
    panels, proofs = {}, {}
    for dataset in ('sst2', 'trec'):
        references = [row for row in rows if row['dataset'] == dataset and row['model'] == 'multinomial_nb' and row['status'] == 'complete']
        by_folder = {Path(row['source_path']).parts[1]: row for row in references}
        require(len(references) == len(by_folder), f'Ambiguous NB reference for {dataset}')
        require(set(by_folder) == {'pilot', 'colab'}, f'Both local and Colab NB references are required for {dataset}')
        local, colab = by_folder['pilot'], by_folder['colab']
        proof = audit((root/local['source_path']).parent, (root/colab['source_path']).parent, root/'data/pilot'/dataset)
        require(proof['eligible_for_separately_labeled_matched_pairing'] and proof['raw_content_recomputed'], f'Cross-environment content audit failed: {dataset}')
        require(local['macro_f1'] == colab['macro_f1'], f'Repeated NB result differs across environments: {dataset}')
        proofs[dataset] = proof
        references_by_hash = {r['manifest_sha256']: read_run((root/r['source_path']).parent)[0] for r in references}
        selected = []
        for model, display, color in MODELS:
            for method in METHODS:
                if method == 'lora' and not model.startswith('Qwen/'):
                    continue
                candidates = [row for row in rows if row['dataset'] == dataset and row['model'] == model and row['method'] == method]
                require(len(candidates) == 1, f'Ambiguous/missing inventory entry: {dataset}/{model}/{method}')
                row = dict(candidates[0])
                row.update(display_model=display, color=color)
                if row['status'] == 'complete':
                    record, _ = read_run((root/row['source_path']).parent)
                    require(record['manifest_sha256'] in references_by_hash, 'Model manifest has no audited environment reference')
                    reference = references_by_hash[record['manifest_sha256']]
                    require(record['labels'] == reference['labels'] and record['test_ids_sha256'] == reference['test_ids_sha256'], 'Model test/label mapping differs from its audited reference')
                    if method == 'zero_shot':
                        require(not record['training_example_ids'], 'Zero-shot has training examples')
                    else:
                        require(record['training_example_ids'] == reference['training_example_ids'], 'Matched model training IDs differ from reference')
                    if model.endswith('4B-Instruct-2507') and method == 'lora':
                        require(record['adapter_training']['load_in_4bit'], '4B adapter must be labeled according to actual quantization')
                selected.append(row)
        panels[dataset] = {'reference': local, 'rows': selected}
    return panels, proofs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-stem', type=Path, default=Path('results/figures/combined-pilot'))
    args = parser.parse_args()
    root = args.root.resolve()
    stem = args.output_stem if args.output_stem.is_absolute() else root/args.output_stem
    panels, proofs = panel_data(root)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'jevbench-combined-pilot-v1', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 10.8), sharey=True)
    fig.subplots_adjust(left=.265, right=.94, bottom=.285, top=.795, wspace=.25)
    fig.text(.055, .95, 'SST-2 and TREC: measured classification pilot', fontsize=22, fontweight='bold', color='#182333')
    fig.text(.055, .91, 'Five models · fixed model/method order, not a ranking · 200 held-out examples per task · seed 42', fontsize=11.7, color='#475569')
    positions, bands = [], []
    cursor = len(panels['sst2']['rows']) + .5*(len(MODELS)-1) - .3
    for model, _, color in MODELS:
        count = sum(row['model'] == model for row in panels['sst2']['rows'])
        group_positions = [cursor-index for index in range(count)]
        positions.extend(group_positions)
        bands.append((min(group_positions)-.5, max(group_positions)+.5, color))
        cursor -= count+.5
    labels = [f"{row['display_model']} · {row['method_label']}" for row in panels['sst2']['rows']]
    for ax, dataset, title, budget in zip(axes, ('sst2', 'trec'), ('SST-2 · sentiment', 'TREC · question type'), (8, 24)):
        panel = panels[dataset]
        ax.set_title(title, loc='left', fontsize=14, fontweight='bold', pad=32, color='#182333')
        ax.text(0, 1.035, f'{budget} new labels per adapted/few-shot arm; zero-shot uses 0', transform=ax.transAxes, fontsize=10.2, color='#475569')
        reference = panel['reference']['macro_f1']
        ax.axvline(reference, color='#334155', linestyle=(0, (4, 4)), linewidth=1.4, alpha=.7, zorder=1)
        ax.text(reference + .012, positions[0]+.83, f'NB: {reference:.3f}', fontsize=9.7, color='#334155', va='center')
        for low, high, color in bands:
            ax.axhspan(low, high, color=color, alpha=.04, zorder=0)
        for position, row in zip(positions, panel['rows']):
            if row['status'] != 'complete':
                ax.text(.025, position, 'pending', fontsize=10, color='#64748B', va='center')
                continue
            value, lo, hi = row['macro_f1'], row['macro_f1_ci_low'], row['macro_f1_ci_high']
            color = row['color']
            require(lo is not None and hi is not None, 'Completed run lacks a recorded confidence interval')
            ax.hlines(position, lo, hi, color=color, linewidth=2.8, alpha=.7, zorder=2)
            ax.vlines([lo, hi], position-.07, position+.07, color=color, linewidth=1.25, zorder=2)
            ax.scatter([value], [position], marker=MARKERS[row['method']], s=76, color=color, edgecolors='white', linewidths=1, zorder=3)
            ax.text(1.045, position, f'{value:.3f}', transform=ax.get_yaxis_transform(), fontsize=10.7, color=color, va='center', fontweight='bold')
        ax.set_yticks(positions, labels)
        ax.set_ylim(-.05, positions[0]+1.15)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .2, .4, .6, .8, 1])
        ax.set_xlabel('Macro-F1', labelpad=10)
        ax.grid(axis='x', color='#E2E8F0', linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=0, pad=9, labelcolor='#334155')
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [Line2D([0], [0], marker=MARKERS[method], color='none', markerfacecolor='#475569', markeredgecolor='white', markersize=8, label=label)
               for method, label in [('zero_shot', 'Zero-shot'), ('few_shot', 'Few-shot: 4/class'), ('lora', 'LoRA / QLoRA: 4/class')]]
    handles.append(Line2D([0], [0], color='#334155', linestyle='--', label='Fixed TF-IDF + NB: 4/class'))
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.053, .182), ncol=4, frameon=False, fontsize=10.7, columnspacing=1.8)
    notes = [
        'Intervals: recorded 95% stratified test-item bootstrap, 1,000 resamples; conditional on this split, seed and prompt.',
        'Qwen2.5 0.5B uses LoRA; Qwen3 4B uses QLoRA. Local models rank numeric ID + EOS; OpenAI generates labels.',
        'Jev uses native Choice via OpenRouter (typesafe/jev-1.13); it has no LoRA arm. Incomplete conditions have no dot.',
        'The dashed NB reference is fixed by configuration; its interval is in the report. Extra-label full-prepared baselines are omitted.',
        'Cross-environment data/training alignment is audited separately; original differing manifest hashes are retained.',
    ]
    for y, text in zip((.155, .126, .097, .068, .039), notes):
        fig.text(.055, y, text, fontsize=10.3, color='#475569')
    stem.parent.mkdir(parents=True, exist_ok=True)
    evidence = {'figure': 'combined-pilot', 'ordering': 'fixed model/method order, not ranked',
                'panels': panels, 'cross_environment_reference_audits': proofs,
                'ci_interpretation': 'recorded conditional 95% bootstrap, not paired contrast or multiple-selection adjustment'}
    stem.with_suffix('.json').write_text(json.dumps(evidence, indent=2, sort_keys=True)+'\n')
    for suffix in ('png', 'svg'):
        metadata = {'Date': None} if suffix == 'svg' else {'Software': 'jevbench plot_combined_pilot.py'}
        fig.savefig(stem.with_suffix('.'+suffix), dpi=200, bbox_inches='tight', pad_inches=.22, metadata=metadata)
        print(stem.with_suffix('.'+suffix))
    plt.close(fig)


if __name__ == '__main__':
    main()
