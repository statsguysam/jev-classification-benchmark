"""Offline ledger/result reconciliation failures must not become cheap-looking totals."""
import json
from decimal import Decimal
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from run_budgeted_hosted import Ledger, usd_nano
from summarize_jev_costs import ENDPOINT, MODEL, PER_REQUEST, reconcile_jev, sha_file


@pytest.fixture
def completed_fixture(tmp_path, request):
    case = getattr(request, 'param', '0.000042000')
    failed = isinstance(case, dict) and case.get('failed', False)
    reported_cost = case.get('cost') if isinstance(case, dict) else case
    inputs, outputs = (None, None) if failed else (1000, 31)
    scripts = tmp_path / 'scripts'; scripts.mkdir()
    for name in ('run_openrouter_jev.py', 'run_budgeted_hosted.py'):
        (scripts/name).write_bytes((SCRIPTS/name).read_bytes())
    ledger = Ledger(tmp_path/'results/jev-budget.jsonl', '2.50', initialize=True)
    paths, reservation_ids = [], []
    for dataset in ('sst2', 'trec'):
        for method in ('zero_shot', 'few_shot'):
            shots = 0 if method == 'zero_shot' else 4
            run_id = dataset+'_'+method
            path = tmp_path/'results/jev'/run_id/'run.json'; path.parent.mkdir(parents=True)
            row_id = dataset+':test:1'
            guard = {'wrapper_sha256': sha_file(scripts/'run_openrouter_jev.py'),
                     'ledger_driver_sha256': sha_file(scripts/'run_budgeted_hosted.py'),
                     'ledger_id': ledger.identity['ledger_id'], 'endpoint': ENDPOINT,
                     'route': 'OpenRouter', 'dataset': dataset, 'shots_per_class': shots, 'seed': 42}
            pricing = {'input_usd_per_million': '0.042', 'output_usd_per_million': '0',
                       'reserve_input_tokens': 64000, 'per_request_reserved_usd': str(PER_REQUEST)}
            rid = ledger.reserve(usd_nano(str(PER_REQUEST)), {'row_id': row_id, 'provider': 'jev',
                'model': MODEL, 'pricing': pricing, **guard})
            reservation_ids.append(rid)
            outcome = {'outcome': 'prediction_error' if failed else 'returned', 'usage': {'input_tokens': inputs, 'output_tokens': outputs},
                       'route': 'OpenRouter', 'request_id': None if failed else run_id, 'provider': None if failed else 'TypeSafe',
                       'resolved_model': None if failed else 'typesafe/jev-1.13-20260917', 'reported_cost_usd': reported_cost,
                       'reservation_released': False, 'usage_exceeded_reservation_assumptions': False,
                       'guard_reasons': {'reported_usage_or_cost_overrun': False, 'unexpected_route_or_snapshot': False, 'invalid_reported_cost': False}}
            ledger.result(rid, outcome)
            row = {'row_id': row_id, 'label': None if failed else 0, 'error': 'network_error: timeout' if failed else None, 'input_tokens': inputs, 'output_tokens': outputs,
                   'metadata': {'resolved_model': outcome['resolved_model'],
                    'openrouter': {'id': outcome['request_id'], 'provider': outcome['provider'], 'reported_cost_usd': reported_cost, 'endpoint': ENDPOINT},
                    'budget': {'ledger_id': ledger.identity['ledger_id'], 'reservation_id': rid,
                               'reserved_upper_bound_usd': str(PER_REQUEST), 'reservation_released': False}}}
            config = {'provider': 'jev', 'model': MODEL, 'base_url': 'https://openrouter.ai/api/v1',
                      'shots_per_class': shots, 'budget_guard': guard}
            run = {'run_id': run_id, 'dataset': dataset, 'method': method, 'seed': 42, 'status': 'complete',
                   'labels': ['negative', 'positive'], 'manifest_sha256': 'a'*64, 'config': config,
                   'metrics': {'n_test': 1, 'n_failures': int(failed)}}
            path.write_text(json.dumps(run))
            path.with_name('predictions.jsonl').write_text(json.dumps(row)+'\n')
            path.with_name('test_manifest.json').write_text(json.dumps({'labels': run['labels'],
                'manifest_sha256': run['manifest_sha256'], 'rows': [{'id': row_id}]}))
            paths.append(path)
    return tmp_path, ledger, paths, reservation_ids


def mutate_prediction(path, callback):
    p = path.with_name('predictions.jsonl')
    row = json.loads(p.read_text()); callback(row); p.write_text(json.dumps(row)+'\n')


def test_exact_decimal_accounting_without_ledger_mutation(completed_fixture):
    root, ledger, _, _ = completed_fixture
    before = (ledger.path.read_bytes(), ledger.lock_path.read_bytes())
    result = reconcile_jev(root, rows_per_run=1)
    assert result['requests'] == 4
    assert Decimal(result['reported_api_cost_usd']) == Decimal('0.000168000')
    assert Decimal(result['token_rate_estimate_usd']) == Decimal('0.000168')
    assert Decimal(result['retained_reservation_usd']) == Decimal('0.010752000')
    assert before == (ledger.path.read_bytes(), ledger.lock_path.read_bytes())


@pytest.mark.parametrize('change,match', [
    (lambda row: row['metadata']['openrouter'].update(reported_cost_usd=None), 'reported cost differs'),
    (lambda row: row['metadata']['openrouter'].update(reported_cost_usd='0.000043'), 'reported cost differs'),
    (lambda row: row.update(input_tokens=999), 'token usage differs'),
    (lambda row: row['metadata'].update(resolved_model='other/model'), 'model differs'),
    (lambda row: row['metadata']['openrouter'].update(endpoint='https://example.invalid'), 'endpoint differs'),
    (lambda row: row.update(error='network_error'), 'outcome differs'),
    (lambda row: row['metadata']['budget'].update(reservation_released=True), 'released reservation'),
])
def test_inconsistent_row_fails_closed(completed_fixture, change, match):
    root, _, paths, _ = completed_fixture
    mutate_prediction(paths[0], change)
    with pytest.raises(ValueError, match=match):
        reconcile_jev(root, rows_per_run=1)


def test_same_reservation_cannot_pay_for_two_predictions(completed_fixture):
    root, _, paths, ids = completed_fixture
    mutate_prediction(paths[0], lambda row: row['metadata']['budget'].update(reservation_id=ids[1]))
    with pytest.raises(ValueError, match='Duplicate or unmatched'):
        reconcile_jev(root, rows_per_run=1)


def test_multiple_results_for_one_reservation_rejected(completed_fixture):
    root, ledger, _, ids = completed_fixture
    ledger.result(ids[0], {'outcome': 'raised', 'usage_unknown': True})
    with pytest.raises(ValueError, match='Duplicate result'):
        reconcile_jev(root, rows_per_run=1)


def test_unrepresented_request_rejected(completed_fixture):
    root, ledger, _, _ = completed_fixture
    ledger.reserve(usd_nano(str(PER_REQUEST)), {'row_id': 'unrepresented'})
    with pytest.raises(ValueError, match='exactly one final result'):
        reconcile_jev(root, rows_per_run=1)


def test_no_reports_for_incomplete_matrix(completed_fixture):
    root, _, paths, _ = completed_fixture
    run = json.loads(paths[0].read_text()); run['status'] = 'running'; paths[0].write_text(json.dumps(run))
    with pytest.raises(ValueError, match='incomplete Jev runs'):
        reconcile_jev(root, rows_per_run=1)


def test_missing_durable_anchor_rejected(completed_fixture):
    root, ledger, _, _ = completed_fixture
    ledger.lock_path.unlink()
    with pytest.raises(Exception, match='ledger AND .lock required'):
        reconcile_jev(root, rows_per_run=1)


@pytest.mark.parametrize('completed_fixture', ['0'], indirect=True)
def test_explicit_zero_reported_cost_is_distinct_from_missing(completed_fixture):
    root, _, _, _ = completed_fixture
    result = reconcile_jev(root, rows_per_run=1)
    assert Decimal(result['reported_api_cost_usd']) == 0
    assert Decimal(result['token_rate_estimate_usd']) > 0
    assert Decimal(result['retained_reservation_usd']) == Decimal('0.010752000')


@pytest.mark.parametrize('completed_fixture', [{'cost': None, 'failed': True}], indirect=True)
def test_timeout_rows_preserve_unknown_cost_and_usage_with_full_reserves(completed_fixture):
    root, _, _, _ = completed_fixture
    result = reconcile_jev(root, rows_per_run=1)
    assert result['n_errors'] == 4
    assert result['reported_api_cost_usd'] is None and result['token_rate_estimate_usd'] is None
    assert result['reported_cost_unknown_requests'] == 4
    assert result['reported_cost_coverage'] == result['input_usage_coverage'] == '0'
    assert Decimal(result['reported_api_cost_known_subtotal_usd']) == 0
    assert Decimal(result['accounted_api_cost_conditional_upper_usd']) == Decimal('0.010752000')
    assert all(run['input_tokens'] is None and run['output_tokens'] is None for run in result['runs'])


@pytest.mark.parametrize('completed_fixture', [{'cost': None, 'failed': False}], indirect=True)
def test_successful_response_with_missing_cost_is_not_free(completed_fixture):
    root, _, _, _ = completed_fixture
    result = reconcile_jev(root, rows_per_run=1)
    assert result['n_errors'] == 0 and result['reported_cost_unknown_requests'] == 4
    assert result['reported_api_cost_usd'] is None
    assert Decimal(result['token_rate_estimate_usd']) == Decimal('0.000168')
    assert Decimal(result['accounted_api_cost_conditional_upper_usd']) == Decimal(result['retained_reservation_usd'])
