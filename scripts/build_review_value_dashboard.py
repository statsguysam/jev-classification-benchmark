"""Export only audited aggregate review-value evidence; never prompts or row IDs."""
from __future__ import annotations
import json
import math
from pathlib import Path
import analyze_review_value as analysis
import summarize_review_controls as controls
import summarize_failed_retries as recovery
import summarize_control_failed_retries as control_recovery

ROOT = Path(__file__).resolve().parents[1]

METRICS = ('micro_accuracy', 'accuracy', 'balanced_accuracy', 'macro_f1', 'n_rows', 'n_failures')
TRANSITIONS = ('wrong_to_correct', 'correct_to_wrong', 'both_correct', 'both_wrong',
               'changed_predictions', 'correct_to_failure', 'correct_to_wrong_label',
               'source_failure_rows', 'review_stage_failure_rows', 'net_correct_change', 'accuracy_delta_pp')
INFERENCES = ('source', 'review_api_requests', 'direct_jev')
RANDOM = ('micro_accuracy', 'balanced_accuracy', 'expected_fixed', 'expected_harmed',
          'expected_net_fixed', 'expected_failures', 'linear_metrics_method', 'macro_f1',
          'macro_f1_method', 'macro_f1_monte_carlo_se', 'samples')
REPEAT_COUNTS = ('both_valid', 'valid_label_agreement', 'valid_label_disagreement',
    'reference_failed_only', 'repeat_failed_only', 'both_failed',
    'resolved_model_mismatch_among_valid', 'unknown_resolved_model_among_valid')
CONTROL_DATASETS = ('breast_cancer', 'wine', 'sst2', 'trec')
CONTROL_ARMS = ('actual', 'no_proposal', 'shuffled')

def scalars(value, keys):
    out = {}
    for key in keys:
        if key in value:
            item = value[key]
            if item is not None and type(item) not in (str, int, float, bool):
                raise ValueError(f'Non-scalar aggregate: {key}')
            if type(item) is float and not math.isfinite(item):
                raise ValueError(f'Nonfinite aggregate: {key}')
            out[key] = item
    return out

def uncertainty(value):
    if value is None:
        return None
    out = scalars(value, ('method', 'group_definition', 'samples', 'seed', 'n_rows', 'n_groups'))
    out['metrics'] = {}
    for key in ('accuracy', 'macro_f1'):
        item = value['metrics'][key]
        interval = item['ci95']
        if (type(item['estimate']) not in (int, float) or not math.isfinite(item['estimate'])
                or not isinstance(interval, list) or len(interval) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in interval)
                or interval[0] > interval[1]):
            raise ValueError('Malformed aggregate confidence interval')
        out['metrics'][key] = {'estimate': item['estimate'], 'ci95': list(interval)}
    return out

def project_controls(report):
    runs = []
    for run in report['runs']:
        complete = run['status'] == 'complete'
        if run['status'] not in ('complete', 'pending'):
            raise ValueError('Unknown control arm status')
        if not complete and any(run.get(k) is not None for k in ('metrics', 'group_bootstrap', 'failures')):
            raise ValueError('Incomplete control arm must not carry scores')
        if complete and (run.get('metrics') is None or run.get('group_bootstrap') is None):
            raise ValueError('Complete control arm requires metrics and uncertainty')
        item = scalars(run, ('dataset', 'arm', 'status', 'expected_requests', 'saved_requests', 'training_examples'))
        item['metrics'] = scalars(run['metrics'], METRICS) if complete else None
        if complete:
            item['metrics']['n_rows'] = run['metrics'].get('n_test', run['metrics'].get('n_rows'))
        item['group_bootstrap'] = uncertainty(run['group_bootstrap']) if complete else None
        runs.append(item)
    expected = {(d, a) for d in CONTROL_DATASETS for a in CONTROL_ARMS}
    if len(runs) != 12 or {(r['dataset'], r['arm']) for r in runs} != expected:
        raise ValueError('Matched control inventory differs')
    comparisons = []
    for row in report['comparisons']:
        complete = row['status'] == 'complete'
        if row['status'] not in ('complete', 'pending'):
            raise ValueError('Unknown control contrast status')
        if not complete and any(row[k] is not None for k in ('paired_group_bootstrap', 'balanced_accuracy_delta', 'transitions_B_to_A')):
            raise ValueError('Pending control contrast must not carry scores')
        if complete and (row['paired_group_bootstrap'] is None or row['transitions_B_to_A'] is None):
            raise ValueError('Complete contrast lacks paired evidence')
        item = scalars(row, ('dataset', 'contrast', 'a', 'b', 'status', 'balanced_accuracy_delta'))
        item['paired_group_bootstrap'] = uncertainty(row['paired_group_bootstrap']) if complete else None
        item['transitions_B_to_A'] = scalars(row['transitions_B_to_A'], TRANSITIONS) if complete else None
        comparisons.append(item)
    if (len(comparisons) != 8 or {(c['dataset'], c['a'], c['b']) for c in comparisons} !=
            {(d, 'actual', b) for d in CONTROL_DATASETS for b in ('no_proposal', 'shuffled')}):
        raise ValueError('Matched contrast inventory differs')
    repeated = report['serving_repeat_diagnostic']
    repeat = scalars(repeated, ('status', 'expected_pairs', 'available_complete_pairs'))
    repeat.update(counts=None, by_dataset=None, valid_pair_agreement=None,
                  valid_agreement_fraction_of_all_pairs=None)
    if repeated['status'] == 'complete':
        if repeated['available_complete_pairs'] != repeated['expected_pairs'] or repeated['counts'] is None:
            raise ValueError('Repeat diagnostic is not complete')
        repeat['counts'] = scalars(repeated['counts'], REPEAT_COUNTS)
        repeat['by_dataset'] = {name: scalars(counts, REPEAT_COUNTS)
            for name, counts in repeated['by_dataset'].items() if name in CONTROL_DATASETS}
        repeat.update(scalars(repeated, ('valid_pair_agreement', 'valid_agreement_fraction_of_all_pairs')))
    elif repeated.get('counts') is not None or repeated.get('by_dataset') is not None:
        raise ValueError('Pending repeat diagnostic must not expose partial agreement')
    result = scalars(report, ('status', 'expected_requests', 'saved_requests', 'expected_primary_arms',
        'complete_primary_arms', 'expected_primary_contrasts', 'complete_primary_contrasts', 'bootstrap_samples', 'bootstrap_seed'))
    result.update(planned_cases=sum(r['expected_requests'] for r in runs if r['arm'] == 'actual'),
        planned_primary_requests=sum(r['expected_requests'] for r in runs),
        planned_repeats=repeated['expected_pairs'], runs=runs, comparisons=comparisons,
        serving_repeat_diagnostic=repeat,
        interval_note='95% paired group bootstrap intervals are exploratory, unadjusted, and conditional on the fixed cases and responses. Balanced-accuracy contrasts have no interval.',
        scope_note='Only the proposal slot changes. The actual proposal is cached Qwen3 four-shot output; no-proposal explicitly supplies no class. These previously examined holdouts are not a fresh confirmatory test.')
    return result

def project_recovery(report):
    if report is None:
        return {'status':'not_available','conditions':[],
            'note':'An audited recovery report is not available. Existing comparisons retain original failures and upstream skips.'}
    if report.get('status') != 'complete' or report.get('execution_status') != 'complete':
        raise ValueError('Only completed audited recovery execution can be exported')
    conditions=[]
    for row in report['conditions']:
        out=scalars(row,('dataset','model','provider','source_model','method','shots_per_class','scope','n_rows',
            'status','analysis_status','condition_variant','n_new_calls','dependent_review_first_calls',
            'same_request_as_original_snapshot','unresolved_upstream_rows'))
        if row['status'] not in ('complete','incomplete'):
            raise ValueError('Unknown recovery view status')
        for key in ('original_snapshot','first_attempt','recovered'):
            view=row[key]
            if view['status']=='incomplete' and view['metrics'] is not None:
                raise ValueError('Incomplete recovery view must not carry metrics')
            out[key]=None if view['metrics'] is None else scalars(view['metrics'],METRICS)
        paired=row['paired_first_to_recovered']
        if row['status']=='incomplete' and (paired is not None or row['paired_group_bootstrap'] is not None
                or out['first_attempt'] is not None or out['recovered'] is not None):
            raise ValueError('Incomplete full recovery condition must not expose selected-row scores')
        out.update(corrected=None if paired is None else paired['corrected'],
            harmed=None if paired is None else paired['harmed'],
            paired_group_bootstrap=uncertainty(row['paired_group_bootstrap']),
            balanced_accuracy_delta=None)
        if out['first_attempt'] is not None and out['recovered'] is not None:
            a,b=out['recovered'].get('balanced_accuracy'),out['first_attempt'].get('balanced_accuracy')
            out['balanced_accuracy_delta']=None if a is None or b is None else a-b
        conditions.append(out)
    return {'status':'complete','conditions':conditions,
        'api_outcomes':scalars(report['api_outcomes'],('new_calls','ordinary_paid_failure_retry_calls',
            'dependent_review_calls','dependent_first_calls','upstream_unresolved_not_called')),
        'budget':scalars(report['budget'],('budget_usd','charged_or_reserved_usd','remaining_reservation_usd','reservations','settlements')),
        'note':'The finite recovery policy is audited and complete. Original snapshots retain every historical failure. First actual calls and recovered outcomes are separate full-condition views; unresolved upstream rows keep service-view scores unavailable.',
        'interval_note':'Recovered minus first actual call: exploratory paired 95% group bootstrap intervals. Balanced-accuracy differences have no interval. These are availability effects on selected failed conditions, not evidence of better semantic decisions.'}

def project_control_recovery(report):
    if report is None:
        return {'status':'not_available','runs':[],'comparisons':[],
            'note':'Post-control recovery is not complete. Primary controls and serving-repeat results retain original failures.'}
    if report.get('status') != 'complete' or report.get('study_role') != 'secondary_failure_recovery_sensitivity':
        raise ValueError('Only complete secondary control recovery may be published')
    runs=[]
    for row in report['runs']:
        if row['status'] != 'complete':
            raise ValueError('Control recovery requires every full arm')
        out=scalars(row,('dataset','arm','status','n_rows','resolved_failures'))
        for view in ('first_attempt','recovered'):
            out[view]={'metrics':scalars(row[view]['metrics'],METRICS),
                'group_bootstrap':uncertainty(row[view]['group_bootstrap'])}
        out['paired_first_to_recovered']=uncertainty(row['paired_first_to_recovered'])
        runs.append(out)
    if len(runs)!=12 or {(r['dataset'],r['arm']) for r in runs}!={(d,a) for d in CONTROL_DATASETS for a in CONTROL_ARMS}:
        raise ValueError('Control recovery arm inventory differs')
    comparisons=[]
    for row in report['comparisons']:
        if row['status'] != 'complete':
            raise ValueError('Control recovery requires every paired contrast')
        out=scalars(row,('dataset','a','b','contrast','status'))
        for view in ('first_attempt','recovered'):
            out[view]={'paired_group_bootstrap':uncertainty(row[view]['paired_group_bootstrap']),
                'balanced_accuracy_delta':row[view]['balanced_accuracy_delta'],
                'transitions_B_to_A':scalars(row[view]['transitions_B_to_A'],TRANSITIONS)}
            scalars(out[view],('balanced_accuracy_delta',))
        comparisons.append(out)
    if len(comparisons)!=8 or {(c['dataset'],c['a'],c['b']) for c in comparisons}!={(d,'actual',b) for d in CONTROL_DATASETS for b in ('no_proposal','shuffled')}:
        raise ValueError('Control recovery contrast inventory differs')
    return {'status':'complete','no_op':report['no_op'],'runs':runs,'comparisons':comparisons,
        'recovery_counts':{scope:scalars(report['recovery_counts'][scope],
            ('original_failures','remaining_failures','resolved_failures','new_calls')) for scope in ('primary','repeat')},
        'new_calls':report['new_call_outcomes']['n_calls'],
        'budget':scalars(report['budget'],('budget_usd','charged_or_reserved_usd','remaining_reservation_usd','reservations','settlements')),
        'note':'Separate secondary sensitivity: original control scores and serving-repeat agreement remain unchanged. The earliest valid recovery is selected regardless of correctness; unresolved failures stay incorrect with the full denominator.',
        'interval_note':'A minus B within each view. These are exploratory unadjusted paired 95% group-bootstrap intervals on the same cases; balanced-accuracy differences have no interval.'}

def project(report, control_report, recovery_report=None, control_recovery_report=None):
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
    return {'schema_version': 2, 'complete_review_conditions': report['complete_review_conditions'],
        'expected_review_conditions': report['expected_review_conditions'],
        'interpretation': report['interpretation'], 'conditions': conditions,
        'pending_conditions': [{k:c[k] for k in ('dataset','source_model','shots_per_class')} for c in report['excluded_conditions']],
        'identical_prompts': {key: sum(p[key] for p in pairs) for key in ('same_label_identical_prompt_rows',
            'both_valid_rows','valid_output_disagreements','rows_with_either_review_failure')},
        'controls': project_controls(control_report),
        'recovery': project_recovery(recovery_report),
        'control_recovery':project_control_recovery(control_recovery_report)}

def main():
    report = analysis.collect()
    saved = json.loads((ROOT/'results/review_value/ANALYSIS.json').read_text())
    if report != saved:
        raise RuntimeError('Review-value report is stale; regenerate it before publishing')
    controlled = controls.collect()
    recovered = recovery.optional_collect()
    recovered_controls = control_recovery.optional_collect()
    target = ROOT/'dashboard/dist/review-data.json'
    target.write_text(json.dumps(project(report, controlled, recovered, recovered_controls), indent=2, allow_nan=False)+'\n')
    print(f'Wrote audited aggregate snapshot: {target}')

if __name__ == '__main__': main()
