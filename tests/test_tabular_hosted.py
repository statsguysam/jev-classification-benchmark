"""Offline cumulative-budget, transport-isolation and resume checks for tabular jobs."""
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_tabular_hosted as hosted
import tabular_data
from jevbench import providers
from jevbench.types import PreparedDataset, Row


@pytest.fixture
def stage(tmp_path, monkeypatch):
    original_root = hosted.ROOT
    config_dir = tmp_path/'configs'; config_dir.mkdir()
    for name in ('tabular_hosted.json', 'hosted_budget.json'):
        (config_dir/name).write_bytes((original_root/'configs'/name).read_bytes())
    priors, expected = {}, {}
    for name, cap, spent in [('openai-budget.jsonl', '7.50', '0.10'), ('jev-budget.jsonl', '2.50', '0.20')]:
        ledger = hosted.budget.Ledger(tmp_path/'results'/name, cap, initialize=True)
        ledger.reserve(hosted.budget.usd_nano(spent), {'fixture': True})
        priors[name] = ledger
        expected[name] = {'cap': cap, 'sha256': hosted.file_sha(ledger.path),
                          'lock_sha256': hosted.file_sha(ledger.lock_path)}
    monkeypatch.setattr(hosted, 'ROOT', tmp_path)
    monkeypatch.setattr(hosted, 'PRIOR', expected)
    monkeypatch.setattr(hosted, 'PRIOR_TOTAL', '0.30')
    # Defaults capture ROOT at import time; bind them to the isolated fixture too.
    monkeypatch.setattr(hosted.verify_prior, '__defaults__', (tmp_path,))
    monkeypatch.setattr(hosted.AnchoredLedger.__init__, '__defaults__', (tmp_path, None))
    monkeypatch.setattr(hosted.jev_route, 'VERIFIED_ON', datetime.now(timezone.utc).date().isoformat())
    monkeypatch.setenv('OPENROUTER_API_KEY', 'DUMMY_TABULAR_TEST_ONLY')
    monkeypatch.setenv('OPENAI_API_KEY', 'DUMMY_OPENAI_TEST_ONLY')
    monkeypatch.setattr(hosted.jev_route, 'post_json_exact_cost', lambda *a, **kw: pytest.fail('Unexpected HTTP attempt'))
    monkeypatch.setattr(providers, '_post_json', lambda *a, **kw: pytest.fail('Unexpected provider transport'))
    def dataset(name):
        return PreparedDataset(name, ['negative', 'positive'],
             [Row(f'{name}:train:{label}:{i}', f'Feature: train-{label}-{i}', label)
              for label in range(2) for i in range(4)], [],
             [Row(f'{name}:test:{i}', f'Feature: test-{i}', i) for i in range(2)], {})
    monkeypatch.setattr(tabular_data, 'load_native_prepared', lambda path: (dataset(Path(path).name), {}))
    return SimpleNamespace(root=tmp_path, priors=priors, ledger=tmp_path/'results/tabular/api-budget.jsonl',
                           data=tmp_path/'data/titanic', output=tmp_path/'results/tabular/hosted')


def args(stage, *extra):
    return ['--data', str(stage.data), '--model-keys', 'jev_openrouter', '--shots', '0',
            '--bootstrap-samples', '100', *extra]


def response(index):
    return {'id': f'gen-tabular-test-{index}', 'model': 'typesafe/jev-1.13-20260917', 'provider': 'TypeSafe',
            'answers': {'classification': {'type': 'choice', 'choice': '1',
                        'probabilities': {'0': 0.25, '1': 0.75}, 'confidence': 0.75}},
            'usage': {'input_tokens': 476, 'output_tokens': 70, 'cost': '0.000019992'}}


def test_prior_hashes_sum_and_dry_run_leave_ledgers_unchanged(stage):
    before = [(p.path.read_bytes(), p.lock_path.read_bytes()) for p in stage.priors.values()]
    snapshots = hosted.verify_prior(stage.root)
    assert sum(Decimal(s['charged_or_reserved_usd']) for s in snapshots.values()) == Decimal('.30')
    plan = hosted.main(args(stage))
    assert plan['fresh_matrix_requests'] == 2 and plan['new_stage_cap_usd'] == '14.70'
    assert plan['approved_cumulative_cap_usd'] == '20.00'
    assert not stage.ledger.exists() and not stage.output.exists()
    assert before == [(p.path.read_bytes(), p.lock_path.read_bytes()) for p in stage.priors.values()]


@pytest.mark.parametrize('declared_total', ['0.31', '6.00'])
def test_incorrect_prior_total_or_excess_cumulative_allocation_rejected(stage, monkeypatch, declared_total):
    if declared_total == '6.00':
        ledger = stage.priors['openai-budget.jsonl']
        ledger.reserve(hosted.budget.usd_nano('5.70'), {})
        hosted.PRIOR['openai-budget.jsonl'].update(sha256=hosted.file_sha(ledger.path),
                                                lock_sha256=hosted.file_sha(ledger.lock_path))
    monkeypatch.setattr(hosted, 'PRIOR_TOTAL', declared_total)
    with pytest.raises(hosted.budget.GuardError, match='Cumulative'):
        hosted.verify_prior(stage.root)


@pytest.mark.parametrize('change', ['append', 'missing_anchor'])
def test_prior_change_stops_before_new_stage_reservation(stage, change):
    inner = hosted.budget.Ledger(stage.ledger, hosted.STAGE_CAP, initialize=True)
    anchored = hosted.AnchoredLedger(inner, stage.root)
    ancestor = stage.priors['openai-budget.jsonl']
    if change == 'append':
        ancestor.reserve(hosted.budget.usd_nano('.01'), {})
    else:
        ancestor.lock_path.unlink()
    # A missing file also fails closed before any reservation/transport.
    with pytest.raises((hosted.budget.GuardError, FileNotFoundError)):
        anchored.reserve(hosted.jev_route.RESERVE_NANO, {})
    assert inner.snapshot()['reservations'] == 0 and anchored.new_requests == 0


def test_stage_cap_prevents_http_and_does_not_create_reservation(stage):
    ledger = hosted.budget.Ledger(stage.ledger, hosted.STAGE_CAP, initialize=True)
    ledger.reserve(hosted.budget.usd_nano('14.699'), {'fixture': 'nearly spent'})
    anchored = hosted.AnchoredLedger(ledger, stage.root)
    config = hosted.validate_models(stage.root/'configs/tabular_hosted.json', ['jev_openrouter'])['jev_openrouter']
    classifier = hosted.jev_route.GuardedOpenRouterJev(config, anchored, {})
    with pytest.raises(hosted.budget.BudgetStop):
        classifier.predict(Row('test', 'feature: 1', 0), ['negative', 'positive'], 'prompt')
    assert ledger.snapshot()['reservations'] == 1 and anchored.new_requests == 0


def test_alternate_stage_ledger_cannot_create_fresh_allowance(stage):
    alternate = stage.root/'new-budget.jsonl'
    with pytest.raises(hosted.budget.GuardError, match='canonical'):
        hosted.main(args(stage, '--ledger', str(alternate), '--execute', '--init-ledger'))
    assert not alternate.exists() and not stage.ledger.exists()


def test_fixed_presets_cannot_change_output_budget_or_duplicate_keys(stage):
    path = stage.root/'configs/tabular_hosted.json'
    original = json.loads(path.read_text())
    assert set(hosted.validate_models(path, list(original))) == set(original)
    changed = json.loads(path.read_text()); changed['openai_frontier']['max_output_tokens'] = 4096
    path.write_text(json.dumps(changed))
    with pytest.raises(hosted.budget.GuardError, match='presets'):
        hosted.validate_models(path, ['openai_frontier'])
    with pytest.raises(hosted.budget.GuardError, match='unique'):
        hosted.validate_models(path, ['jev_openrouter', 'jev_openrouter'])


def test_invalid_bootstrap_count_rejected_before_initializing_budget(stage):
    with pytest.raises(hosted.budget.GuardError, match='Invalid'):
        hosted.main(args(stage, '--bootstrap-samples', '0', '--execute', '--init-ledger'))
    assert not stage.ledger.exists()


def test_checkpoint_resumes_real_runner_without_repeated_http(stage, monkeypatch):
    calls = []
    def fake_post(url, body, headers, timeout):
        calls.append(body)
        assert url == hosted.jev_route.ENDPOINT
        # A durable reservation must exist before the mocked HTTP boundary.
        ledger = hosted.budget.Ledger(stage.ledger, hosted.STAGE_CAP)
        assert ledger.snapshot()['reservations'] == len(calls)
        return response(len(calls))
    monkeypatch.setattr(hosted.jev_route, 'post_json_exact_cost', fake_post)
    first = hosted.main(args(stage, '--execute', '--init-ledger', '--stop-after-new-requests', '1'))
    assert first['status'] == 'checkpoint' and first['new_requests'] == 1
    assert len(calls) == 1
    run_paths = list(stage.output.glob('*/run.json')); assert len(run_paths) == 1
    assert len(run_paths[0].with_name('predictions.jsonl').read_text().splitlines()) == 1
    final = hosted.main(args(stage, '--execute'))
    assert final['completed'] == [run_paths[0].parent.name] and len(calls) == 2
    again = hosted.main(args(stage, '--execute'))
    assert again['completed'] == final['completed'] and len(calls) == 2
    rows = [json.loads(line) for line in run_paths[0].with_name('predictions.jsonl').read_text().splitlines()]
    assert [row['row_id'] for row in rows] == ['titanic:test:0', 'titanic:test:1']
    assert len({row['metadata']['budget']['reservation_id'] for row in rows}) == 2
    assert final['budget']['reservations'] == 2
    assert final['budget']['charged_or_reserved_usd'] == '0.005376000'
    record = json.loads(run_paths[0].read_text())
    assert record['status'] == 'complete' and record['metrics']['n_failures'] == 0
    assert 'DUMMY_' not in stage.ledger.read_text() and 'Feature:' not in stage.ledger.read_text()


def test_worker_mode_launches_one_process_per_unique_dataset_with_same_ledger(stage, monkeypatch):
    commands = []
    def child(command, **kwargs):
        commands.append(command)
        assert kwargs == {'check': False}
        assert '--init-ledger' not in command and '--prompt-api-key' not in command and '--workers' not in command
        assert command[command.index('--ledger')+1] == str(stage.ledger.resolve())
        assert not any('DUMMY_' in part for part in command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(hosted.subprocess, 'run', child)
    second = stage.root/'data/wine'
    result = hosted.main(args(stage, '--data', str(stage.data), str(second), '--workers', '2', '--execute', '--init-ledger'))
    assert result['worker_exit_codes'] == [0, 0] and result['budget']['reservations'] == 0
    assert len(commands) == 2
    assert {c[c.index('--data')+1] for c in commands} == {str(stage.data.resolve()), str(second.resolve())}


def test_repeated_dataset_fails_before_workers_or_ledger(stage, monkeypatch):
    monkeypatch.setattr(hosted.subprocess, 'run', lambda *a, **kw: pytest.fail('No worker may start'))
    with pytest.raises(hosted.budget.GuardError, match='Repeated dataset'):
        hosted.main(args(stage, '--data', str(stage.data), str(stage.data), '--workers', '2', '--execute', '--init-ledger'))
    assert not stage.ledger.exists()
