"""Export only audited aggregate review-value evidence; never prompts or row IDs."""
from __future__ import annotations
import json
from pathlib import Path
import analyze_review_value as analysis
import summarize_review_controls as controls

ROOT = Path(__file__).resolve().parents[1]

METRICS = ('micro_accuracy', 'accuracy', 'balanced_accuracy', 'macro_f1', 'n_rows', 'n_failures')
TRANSITIONS = ('wrong_to_correct', 'correct_to_wrong', 'both_correct', 'both_wrong',
               'changed_predictions', 'correct_to_failure', 'correct_to_wrong_label',
               'source_failure_rows', 'review_stage_failure_rows', 'net_correct_change', 'accuracy_delta_pp')
INFERENCES = ('source', 'review_api_requests', 'direct_jev')
RANDOM = ('micro_accuracy', 'balanced_accuracy', 'expected_fixed', 'expected_harmed',
          'expected_net_fixed', 'expected_failures', 'linear_metrics_method', 'macro_f1',
          'macro_f1_method', 'macro_f1_monte_carlo_se', 'samples')

def scalars(value, keys):
    out = {}
    for key in keys:
        if key in value:
            item = value[key]
            if item is not None and type(item) not in (str, int, float, bool):
                raise ValueError(f'Non-scalar aggregate: {key}')
            out[key] = item
    return out

def project(report, control_report):
    conditions = []
    for c in report['conditions']:
        out = scalars(c, ('dataset', 'source_model', 'shots_per_class', 'n_rows'))
        out['metrics'] = {arm: scalars(c['metrics'][arm], METRICS)
                          for arm in ('never_review','always_review','direct_jev')}
        for key in ('review_minus_base','review_minus_direct'):
            out[key] = scalars(c[key], TRANSITIONS)
        out['required_inferences'] = {arm: scalars(c['required_inferences'][arm], INFERENCES)
                          for arm in ('never_review','always_review','direct_jev')}
        out['selective_review_eligibility'] = scalars(c['selective_review_eligibility'], ('eligible','probability_semantics'))
        out['curve'] = None
        if c['selective_review']:
            out['curve'] = []
            for p in c['selective_review']['points']:
                point = scalars(p, ('requested_review_percent','selected_rows','actual_review_fraction'))
                point.update(metrics=scalars(p['metrics'],METRICS),
                    transitions_from_never_review=scalars(p['transitions_from_never_review'],TRANSITIONS),
                    required_inferences=scalars(p['required_inferences'],INFERENCES),
                    random_matched_rate=scalars(p['random_matched_rate'],RANDOM))
                out['curve'].append(point)
        conditions.append(out)
    pairs = report['same_label_prompt_pairs']
    return {'schema_version': 1, 'complete_review_conditions': report['complete_review_conditions'],
        'expected_review_conditions': report['expected_review_conditions'],
        'interpretation': report['interpretation'], 'conditions': conditions,
        'pending_conditions': [{k:c[k] for k in ('dataset','source_model','shots_per_class')} for c in report['excluded_conditions']],
        'identical_prompts': {key: sum(p[key] for p in pairs) for key in ('same_label_identical_prompt_rows',
            'both_valid_rows','valid_output_disagreements','rows_with_either_review_failure')},
        'controls': {'planned_cases':550, 'planned_primary_requests':1650, 'planned_repeats':64,
                     'runs': [{**scalars(r, ('dataset','arm','status')), 'metrics': None if r['metrics'] is None else scalars(r['metrics'], METRICS)} for r in control_report['runs']]}}

def main():
    report = analysis.collect()
    saved = json.loads((ROOT/'results/review_value/ANALYSIS.json').read_text())
    if report != saved:
        raise RuntimeError('Review-value report is stale; regenerate it before publishing')
    controlled = controls.collect()
    target = ROOT/'dashboard/dist/review-data.json'
    target.write_text(json.dumps(project(report, controlled), indent=2, allow_nan=False)+'\n')
    print(f'Wrote audited aggregate snapshot: {target}')

if __name__ == '__main__': main()
