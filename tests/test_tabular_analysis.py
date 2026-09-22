import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('tabular_analysis', ROOT / 'scripts/summarize_tabular.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
from tabular_data import prepare_from_native, save_native_prepared
from jevbench.runner import _identity, environment, finish_run
from jevbench.types import Prediction
from run_tabular_classical import run_native_classical


def prepared_fixture(name='titanic', n_classes=2):
    spec = {'labels': [f'class {i}' for i in range(n_classes)], 'target_column': 'target',
            'task_description': 'Fixture.', 'features': [{'name': 'x', 'source_name': 'x', 'kind': 'numeric'}]}
    records = [{'id': f'{name}:{i:04d}', 'label': i % n_classes, 'features': {'x': float(i)}} for i in range(60)]
    return prepare_from_native(name, spec, records, {'observed_sha256': 'fixture-source'})


def native_run(tmp_path):
    dataset, native = prepared_fixture()
    save_native_prepared(dataset, native, tmp_path / 'data')
    record = run_native_classical(dataset, native, 'majority', tmp_path / 'results', bootstrap_samples=100)
    path = tmp_path / 'results' / record['run_id'] / 'run.json'
    return dataset, native, path


def test_expected_matrix_and_comparison_inventory():
    keys = analysis.expected_conditions()
    assert len(keys) == len(set(keys)) == 66
    assert sum(key[2] == 'classical_tabular' for key in keys) == 30
    assert sum(key[1] in analysis.OPEN_MODELS for key in keys) == 18
    assert sum(key[1] in analysis.HOSTED for key in keys) == 18
    comparisons = analysis.comparison_specs()
    assert len(comparisons) == 48
    assert sum(item[0] == 'main matched comparison' for item in comparisons) == 18


def test_group_bootstrap_matches_explicit_whole_group_resampling():
    y = [0, 1, 1, 0, 1, 0]
    a = [0, 0, 1, -1, 1, 0]
    b = [1, 1, 0, 0, 1, -1]
    groups = ['a', 'a', 'b', 'c', 'c', 'c']
    actual = analysis.group_bootstrap(y, a, groups, b, n_classes=2, samples=100, seed=19)
    rng = np.random.default_rng(19)
    indexes = [np.flatnonzero(np.asarray(groups) == group) for group in sorted(set(groups))]
    accuracies, f1s, counts = [], [], []
    for draw in rng.integers(0, 3, size=(100, 3)):
        idx = np.concatenate([indexes[index] for index in draw])
        yy, aa, bb = np.asarray(y)[idx], np.asarray(a)[idx], np.asarray(b)[idx]
        counts.append(len(idx))
        accuracies.append(np.mean(yy == aa) - np.mean(yy == bb))
        f1s.append(f1_score(yy, aa, labels=[0, 1], average='macro', zero_division=0) -
                   f1_score(yy, bb, labels=[0, 1], average='macro', zero_division=0))
    assert actual['n_groups'] == 3 and actual['n_rows'] == 6
    assert actual['resampled_row_count_range'] == [min(counts), max(counts)]
    assert actual['resampled_row_count_range'][0] != actual['resampled_row_count_range'][1]
    assert actual['metrics']['accuracy']['ci95'] == pytest.approx(np.quantile(accuracies, [.025, .975]))
    assert actual['metrics']['macro_f1']['ci95'] == pytest.approx(np.quantile(f1s, [.025, .975]))


def test_group_bootstrap_keeps_conflicting_truths_together_and_counts_failures():
    result = analysis.group_bootstrap([0, 1], [0, -1], ['one', 'one'], n_classes=2, samples=100)
    assert result['n_groups'] == 1
    assert result['metrics']['accuracy'] == {'estimate': .5, 'ci95': [.5, .5]}
    assert result['metrics']['macro_f1']['estimate'] == pytest.approx(.5)
    paired = analysis.group_bootstrap([0, 1], [0, -1], ['one', 'one'], [0, -1], n_classes=2, samples=100)
    assert paired['metrics']['macro_f1'] == {'estimate': 0., 'ci95': [0., 0.]}


@pytest.mark.parametrize('y,a,groups', [([], [], []), ([0, 1], [0], ['a', 'b']), ([0, 2], [0, 1], ['a', 'b']), ([0, 1], [0, 1], ['a', '']), ([0., 1.], [0, 1], ['a', 'b'])])
def test_invalid_bootstrap_inputs_rejected(y, a, groups):
    with pytest.raises(ValueError):
        analysis.group_bootstrap(y, a, groups, n_classes=2, samples=100)


def test_complete_native_run_audits_recomputed_metrics_and_no_mutation(tmp_path):
    dataset, native, path = native_run(tmp_path)
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    row, payload = analysis.audit_run(path, dataset, native, tmp_path, samples=100)
    assert row['status'] == 'complete'
    assert row['n_test'] == len(dataset.test)
    assert row['train_labels'] == 8
    assert row['group_bootstrap']['n_groups'] == len(dataset.test)
    assert payload['training_example_ids'] == analysis._expected_ids(dataset, 4)
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}


def test_tampered_manifest_and_training_ids_rejected(tmp_path):
    dataset, native, path = native_run(tmp_path)
    record = analysis.read_json(path)
    original = copy.deepcopy(record)
    record['dataset_manifest']['seed'] = 43
    with pytest.raises(ValueError, match='manifest differs'):
        analysis.audit_record(record, dataset, native)
    record = copy.deepcopy(original)
    record['training_example_ids'] = list(reversed(record['training_example_ids']))
    with pytest.raises(ValueError, match='training/example IDs differ'):
        analysis.audit_record(record, dataset, native)


def test_completed_predictions_and_metrics_tampering_rejected(tmp_path):
    dataset, native, path = native_run(tmp_path)
    record = analysis.read_json(path)
    record['metrics']['accuracy'] = .123
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Saved metric differs'):
        analysis.audit_run(path, dataset, native, tmp_path, samples=100)


def test_incomplete_runs_remain_unscored(tmp_path):
    dataset, native, path = native_run(tmp_path)
    record = analysis.read_json(path)
    record['status'] = 'running'
    record.pop('training')
    path.write_text(json.dumps(record))
    path.with_name('predictions.jsonl').write_text('partial bad JSON')
    row, payload = analysis.audit_run(path, dataset, native, tmp_path, samples=100)
    assert row['accuracy'] is None and row['group_bootstrap'] is None and payload is None
    assert row['status'] == 'running'


def test_all_pending_report_has_66_unscored_rows_and_48_unscored_contrasts(tmp_path):
    for name in analysis.DATASETS:
        dataset, native = prepared_fixture(name, 3 if name == 'wine' else 2)
        save_native_prepared(dataset, native, tmp_path / 'data/tabular-full' / name)
    report = analysis.collect(tmp_path, samples=100)
    assert report['expected_runs'] == 66 and report['complete_runs'] == 0
    assert len(report['comparisons']) == 48
    assert all(item['group_bootstrap'] is None for item in report['comparisons'])
    output = tmp_path / 'results/TABULAR_COMPARISON.md'
    analysis.write_report(report, output, tmp_path)
    assert output.is_file() and output.with_suffix('.json').is_file() and output.with_suffix('.csv').is_file()
    assert '0 of 66' in output.read_text()


def test_reordered_predictions_rejected_even_when_all_rows_present(tmp_path):
    dataset, native, path = native_run(tmp_path)
    predictions_path = path.with_name('predictions.jsonl')
    predictions_path.write_text('\n'.join(reversed(predictions_path.read_text().splitlines())) + '\n')
    with pytest.raises(ValueError, match='Prediction order or coverage differs'):
        analysis.audit_run(path, dataset, native, tmp_path, samples=100)


def test_main_paired_contrast_rejects_unequal_training_ids():
    rows = {key: {'status': 'pending'} for key in analysis.expected_conditions()}
    ka = ('titanic', analysis.HOSTED[0], 'few_shot', 4)
    kb = ('titanic', analysis.HOSTED[2], 'few_shot', 4)
    for key in (ka, kb):
        rows[key] = {'status': 'complete', 'train_labels': 8, 'run_id': str(key)}
    shared = {'y': [0, 1], 'groups': ['a', 'b'], 'manifest_sha256': 'same', 'n_classes': 2,
              'predictions': [0, 1], 'training_example_ids': ['train1']}
    payloads = {ka: copy.deepcopy(shared), kb: copy.deepcopy(shared)}
    payloads[kb]['training_example_ids'] = ['train2']
    with pytest.raises(ValueError, match='unequal training IDs'):
        analysis.paired_contrasts(rows, payloads, samples=100)
    payloads[kb]['training_example_ids'] = ['train1']
    contrasts = analysis.paired_contrasts(rows, payloads, samples=100)
    complete = [item for item in contrasts if item['status'] == 'complete']
    assert len(complete) == 1
    assert complete[0]['equal_training_ids']
    assert complete[0]['group_bootstrap']['metrics']['accuracy']['estimate'] == 0.


@pytest.mark.parametrize('change,match', [
    ('code_hash', 'extension implementation'),
    ('data_hash', 'extension implementation'),
    ('parameters', 'fixed recipe'),
    ('preprocessing', 'training-only preprocessing'),
    ('fitted_scale', 'training-only preprocessing'),
])
def test_native_extension_provenance_contradictions_rejected(tmp_path, change, match):
    dataset, native, path = native_run(tmp_path)
    record = analysis.read_json(path)
    if change == 'code_hash':
        record['config']['tabular_classical_sha256'] = '0' * 64
    elif change == 'data_hash':
        record['config']['tabular_data_sha256'] = '0' * 64
    elif change == 'parameters':
        record['training']['selected_parameters'] = {'strategy': 'stratified'}
    elif change == 'preprocessing':
        record['training']['features']['fit_split'] = 'train plus validation plus test'
    elif change == 'fitted_scale':
        record['training']['features']['numeric_training_scales'][0] *= 2
    with pytest.raises(ValueError, match=match):
        analysis.audit_record(record, dataset, native)


def open_config_record(model_index=0, method='zero_shot'):
    model = analysis.OPEN_MODELS[model_index]
    config = dict(analysis.read_json(ROOT / 'configs/models.json')['qwen_small' if model_index == 0 else 'qwen_main'])
    config.update(device='mps' if model_index == 0 else 'cuda', shots_per_class=4 if method == 'few_shot' else 0)
    record = {'dataset': 'titanic', 'method': method, 'config': config}
    if method == 'lora':
        config['adapter_path'] = '/fixture/adapter'
        record['adapter_training'] = {'load_in_4bit': model_index == 1,
            'device': config['device'], 'dtype': 'float32' if model_index == 0 else 'float16'}
    return record


@pytest.mark.parametrize('field,value', [('revision', '0' * 40), ('max_context_tokens', 4096), ('device', 'cpu')])
def test_open_pinned_preset_mutations_rejected(field, value):
    record = open_config_record()
    analysis.audit_model_config(record)
    record['config'][field] = value
    with pytest.raises(ValueError, match='pinned preset/revision/context/device'):
        analysis.audit_model_config(record)


def test_adapter_quantization_is_specific_to_approved_open_arm():
    for model_index in (0, 1):
        record = open_config_record(model_index, 'lora')
        analysis.audit_model_config(record)
        record['adapter_training']['load_in_4bit'] = not record['adapter_training']['load_in_4bit']
        with pytest.raises(ValueError, match='adaptation precision/device/quantization'):
            analysis.audit_model_config(record)


def hosted_config_record(monkeypatch, model_key='openai_frontier'):
    config = dict(analysis.read_json(ROOT / 'configs/tabular_hosted.json')[model_key])
    config['shots_per_class'] = 0
    record = {'dataset': 'titanic', 'method': 'zero_shot', 'config': config}
    monkeypatch.setattr(analysis, 'canonical_tabular_ledger_header', lambda: {'ledger_id': 'fixture-canonical-ledger'})
    config['budget_guard'] = analysis.expected_hosted_guard(record, model_key)
    return record


@pytest.mark.parametrize('mutation', ['output_cap', 'wrapper_hash', 'ledger_id', 'stage_cap', 'prior_anchor', 'reasoning'])
def test_hosted_fixed_config_and_budget_provenance_mutations_rejected(monkeypatch, mutation):
    record = hosted_config_record(monkeypatch)
    analysis.audit_model_config(record)
    if mutation == 'output_cap':
        record['config']['max_output_tokens'] = 64
    elif mutation == 'wrapper_hash':
        record['config']['budget_guard']['tabular_wrapper_sha256'] = '0' * 64
    elif mutation == 'ledger_id':
        record['config']['budget_guard']['ledger_id'] = 'other-ledger'
    elif mutation == 'stage_cap':
        record['config']['budget_guard']['stage_cap_usd'] = '20.00'
    elif mutation == 'prior_anchor':
        record['config']['budget_guard']['prior_ledgers']['jev-budget.jsonl']['sha256'] = '0' * 64
    else:
        record['config']['reasoning_effort'] = 'high'
    with pytest.raises(ValueError, match='Hosted'):
        analysis.audit_model_config(record)
