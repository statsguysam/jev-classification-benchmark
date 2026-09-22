"""Offline checks for the sequential, local-only tabular neural driver."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_tabular_local as helper
from jevbench.data import save_prepared
from jevbench.types import PreparedDataset, Row


@pytest.fixture
def prepared(tmp_path):
    for name, classes in [('titanic', 2), ('breast_cancer', 2), ('wine', 3)]:
        rows = lambda split, n: [Row(f'{name}:{split}:{c}:{i}', f'Task: {name}; feature={split}-{c}-{i}', c)
                                 for c in range(classes) for i in range(n)]
        dataset = PreparedDataset(name, [f'class {i}' for i in range(classes)], rows('train', 7), rows('validation', 1), rows('test', 2))
        save_prepared(dataset, tmp_path/'data'/name)
    config = tmp_path/'models.json'
    config.write_text(json.dumps({'small': {'provider': 'hf', 'model': 'test/small', 'revision': 'a'*40,
                                          'device': 'auto', 'max_context_tokens': 8192}}))
    return helper.parser().parse_args(['--data-root', str(tmp_path/'data'), '--config', str(config),
         '--model-key', 'small', '--output', str(tmp_path/'results'), '--adapter-root', str(tmp_path/'adapters'),
         '--device', 'cpu'])


def write_adapter(item):
    path = Path(item['adapter_path']); path.mkdir(parents=True)
    (path/'jevbench_training.json').write_text(json.dumps(item['training_expectations']))
    (path/'adapter_config.json').write_text(json.dumps({'r': 8, 'lora_alpha': 16, 'lora_dropout': 0,
          'base_model_name_or_path': item['training_expectations']['model_id']}))
    (path/'adapter_model.safetensors').write_bytes(b'test fixture only')
    return path


def test_plan_is_read_only_and_has_three_arms_fixed_training_budget(prepared):
    plan = helper.make_plan(prepared)
    assert not prepared.output.exists() and not prepared.adapter_root.exists()
    assert len(plan['jobs']) == 12 and plan['evaluation_decisions_upper_bound'] == 42
    assert [d['training_labels'] for d in plan['datasets']] == [8, 8, 12]
    assert all(d['validation_labels_used'] == 0 for d in plan['datasets'])
    assert all(j['command'][:3] == [sys.executable, '-m', 'jevbench.cli'] for j in plan['jobs'])
    assert all('--allow-paid' not in j['command'] for j in plan['jobs'])
    assert plan['execution_policy']['hosted_calls'] is False
    assert plan['execution_policy']['network_downloads'] is False
    for item in plan['datasets']:
        assert all(':train:' in rid for rid in item['matched_training_row_ids'])
        assert item['training_expectations']['epochs'] == 3


@pytest.mark.parametrize('phase,phases', [('base', ['zero', 'few']), ('train', ['train']), ('eval', ['adapter'])])
def test_phase_is_bounded(prepared, phase, phases):
    prepared.phase = phase
    plan = helper.make_plan(prepared)
    assert [j['phase'] for j in plan['jobs']] == phases*3
    assert all(j['command'][3] == ('train-lora' if phase == 'train' else 'model') for j in plan['jobs'])


@pytest.mark.parametrize('change,match', [
    ({'provider': 'openai'}, 'hosted'), ({'revision': 'main'}, 'immutable'),
    ({'api_key': 'synthetic'}, 'credentials'), ({'adapter_path': 'somewhere'}, 'unadapted')])
def test_unsupported_model_configs_fail_closed(prepared, change, match):
    config = json.loads(prepared.config.read_text()); config['small'].update(change)
    prepared.config.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        helper.make_plan(prepared)


def test_quantization_requires_explicit_cuda(prepared):
    prepared.load_in_4bit = True
    with pytest.raises(ValueError, match='CUDA'):
        helper.make_plan(prepared)
    prepared.device = 'cuda'
    plan = helper.make_plan(prepared)
    assert all('--load-in-4bit' in j['command'] for j in plan['jobs'] if j['phase'] == 'train')
    assert all('--load-in-4bit' not in j['command'] for j in plan['jobs'] if j['phase'] != 'train')


def test_reuse_checks_recipe_labels_ids_and_prompt(prepared):
    item = helper.make_plan(prepared)['datasets'][0]
    path = write_adapter(item)
    helper.validate_adapter(path, item['training_expectations'])
    for key, value in [('epochs', 4), ('training_row_ids', []), ('prompt_sha256', 'different'), ('validation_rows', 1)]:
        changed = {**item['training_expectations'], key: value}
        (path/'jevbench_training.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=key):
            helper.validate_adapter(path, item['training_expectations'])


def test_partial_adapter_not_overwritten(prepared):
    item = helper.make_plan(prepared)['datasets'][0]
    path = Path(item['adapter_path']); path.mkdir(parents=True)
    (path/'unfinished').write_text('keep')
    with pytest.raises(ValueError, match='Incomplete'):
        helper.validate_adapter(path, item['training_expectations'])
    assert (path/'unfinished').read_text() == 'keep'


def test_device_lock_prevents_concurrent_helpers(tmp_path):
    with helper.exclusive_device_lock(tmp_path/'device.lock'):
        with pytest.raises(RuntimeError, match='device lock'):
            with helper.exclusive_device_lock(tmp_path/'device.lock'):
                raise AssertionError('must not acquire')


def test_execute_sequential_and_offline_then_stops_on_failure(prepared, monkeypatch, tmp_path):
    prepared.phase = 'base'
    plan = helper.make_plan(prepared)
    monkeypatch.setattr(helper, 'ROOT', tmp_path)
    called = []
    def fake_run(command, **kwargs):
        called.append((command, kwargs))
        return SimpleNamespace(returncode=0 if len(called) == 1 else 3)
    monkeypatch.setattr(helper.subprocess, 'run', fake_run)
    with pytest.raises(RuntimeError, match='stopped without retry'):
        helper.execute_plan(plan, prepared)
    assert len(called) == 2
    assert all(k['env']['HF_HUB_OFFLINE'] == '1' and k['env']['TRANSFORMERS_OFFLINE'] == '1' for _, k in called)
    events = next((prepared.output/'plans').glob('*.execution.jsonl')).read_text().splitlines()
    assert [json.loads(e)['status'] for e in events] == ['complete', 'failed']


def test_changed_manifest_stops_before_subprocess(prepared, monkeypatch, tmp_path):
    prepared.phase = 'base'; plan = helper.make_plan(prepared)
    monkeypatch.setattr(helper, 'ROOT', tmp_path)
    path = prepared.data_root/'titanic/manifest.json'
    manifest = json.loads(path.read_text()); manifest['changed'] = True; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(helper.subprocess, 'run', lambda *a, **kw: pytest.fail('must not execute'))
    with pytest.raises(ValueError, match='changed after planning'):
        helper.execute_plan(plan, prepared)


def test_verified_training_reuse_does_not_call_training(prepared, monkeypatch, tmp_path):
    prepared.phase = 'train'; plan = helper.make_plan(prepared)
    monkeypatch.setattr(helper, 'ROOT', tmp_path)
    for item in plan['datasets']:
        write_adapter(item)
    monkeypatch.setattr(helper.subprocess, 'run', lambda *a, **kw: pytest.fail('must reuse verified adapter'))
    helper.execute_plan(plan, prepared)
    events = next((prepared.output/'plans').glob('*.execution.jsonl')).read_text().splitlines()
    assert len(events) == 3 and all(json.loads(e)['status'] == 'reused_verified_adapter' for e in events)
