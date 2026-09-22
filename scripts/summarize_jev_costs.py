#!/usr/bin/env python3
"""Independently reconcile completed Jev/OpenRouter costs and both API allocations.

Read-only reconciliation; report generation sends no API requests. Missing costs
and usage retain explicit unknowns. Repeated reservations and incomplete matrices
fail closed. Run only after
all four Jev workers/jobs and the OpenAI jobs have finished.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from run_budgeted_hosted import Ledger
from summarize_api_costs import reconcile as reconcile_openai

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'typesafe/jev-1.13'
ALLOWED_MODELS = {MODEL, 'typesafe/jev-1.13-20260917'}
ENDPOINT = 'https://openrouter.ai/api/v1/systemone'
INPUT_RATE = Decimal('0.042')
PER_REQUEST = Decimal('0.002688000')
MILLION = Decimal(1_000_000)
NANO = Decimal(1_000_000_000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def money(value, name):
    require(isinstance(value, str), f'{name} must be an explicit decimal string; missing is not zero')
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise ValueError(f'Invalid decimal {name}') from None
    require(result.is_finite() and result >= 0, f'Invalid nonnegative finite {name}')
    return result


def load(path):
    return json.loads(path.read_text())


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional_money(value, name):
    return None if value is None else money(value, name)


def unique_events(events, kind):
    selected = [event for event in events if event['type'] == kind]
    by_id = {event['reservation_id']: event for event in selected}
    require(len(by_id) == len(selected), f'Duplicate {kind} event for a reservation')
    return by_id


def reconcile_jev(root=ROOT, *, rows_per_run=200):
    ledger_path = root/'results/jev-budget.jsonl'
    ledger = Ledger(ledger_path, '2.50')
    with ledger._locked():
        events, reserved_nano = ledger._read()
    reserves, results = unique_events(events, 'reserve'), unique_events(events, 'result')
    require(not any(event['type'] == 'settle' for event in events), 'Jev reservations must remain retained; settlement found')
    require(set(reserves) == set(results), 'Each Jev reservation needs exactly one final result event')
    snapshot = ledger.snapshot()
    require(not snapshot['halted'], 'Jev ledger halted after a guard violation')
    require(Decimal(reserved_nano)/NANO == money(snapshot['charged_or_reserved_usd'], 'ledger total'), 'Ledger changed during reconciliation')
    wrapper = root/'scripts/run_openrouter_jev.py'
    driver = root/'scripts/run_budgeted_hosted.py'
    wrapper_sha, driver_sha = sha_file(wrapper), sha_file(driver)
    seen, request_ids, conditions, runs = set(), set(), set(), []
    file_hashes = {p.relative_to(root).as_posix(): sha_file(p) for p in (ledger_path, ledger.lock_path, wrapper, driver)}
    for path in sorted((root/'results/jev').glob('*/run.json')):
        run = load(path)
        require(run['status'] == 'complete', 'Finish or separately account for incomplete Jev runs before final cost reporting')
        config = run['config']; guard = config['budget_guard']
        condition = (run['dataset'], run['method'])
        require(condition not in conditions, 'Duplicate Jev dataset/method condition')
        conditions.add(condition)
        require(config['provider'] == 'jev' and config['model'] == MODEL, 'Unexpected requested Jev model/provider')
        require(config['base_url'] == 'https://openrouter.ai/api/v1', 'Unexpected Jev base endpoint')
        require(run['seed'] == 42 and config['shots_per_class'] == (0 if run['method'] == 'zero_shot' else 4), 'Unexpected seed or shot budget')
        require(guard['wrapper_sha256'] == wrapper_sha and guard['ledger_driver_sha256'] == driver_sha, 'Wrapper/ledger-driver source hashes differ from run provenance')
        require(guard['ledger_id'] == snapshot['ledger_id'] and guard['endpoint'] == ENDPOINT and guard['route'] == 'OpenRouter', 'Run budget provenance does not match ledger/route')
        require(guard['dataset'] == run['dataset'] and guard['seed'] == 42 and guard['shots_per_class'] == config['shots_per_class'], 'Run/guard experimental provenance differs')
        require(run['metrics']['n_test'] == rows_per_run, 'Expected complete Jev evaluation')
        predictions_path, manifest_path = path.with_name('predictions.jsonl'), path.with_name('test_manifest.json')
        rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
        manifest = load(manifest_path)
        require(len(rows) == rows_per_run, 'Jev prediction count differs from expected test size')
        require([row['row_id'] for row in rows] == [row['id'] for row in manifest['rows']], 'Prediction/test row IDs or order differ')
        require(len({row['row_id'] for row in rows}) == len(rows), 'Duplicate prediction row IDs inside a run')
        require(manifest['manifest_sha256'] == run['manifest_sha256'] and manifest['labels'] == run['labels'], 'Run/test manifest provenance differs')
        reported, estimated, reserved = Decimal(0), Decimal(0), Decimal(0)
        input_tokens, output_tokens, resolved_models = 0, 0, set()
        cost_known, input_known, output_known, errors = 0, 0, 0, 0
        for row in rows:
            failed = row.get('error') is not None
            if failed:
                require(isinstance(row['error'], str) and bool(row['error']), 'Invalid error record')
                errors += 1
            else:
                require(type(row.get('label')) is int and 0 <= row['label'] < len(run['labels']), 'A successful Jev prediction has an invalid label')
            metadata = row['metadata']; budget = metadata['budget']; routed = metadata['openrouter']
            rid = budget['reservation_id']
            require(rid not in seen and rid in reserves and rid in results, 'Duplicate or unmatched row reservation ID')
            seen.add(rid)
            reserve, result = reserves[rid], results[rid]
            details, outcome = reserve['details'], result['details']
            require(budget['ledger_id'] == snapshot['ledger_id'] and budget['reservation_released'] is False, 'Prediction has a different ledger or released reservation')
            require(money(budget['reserved_upper_bound_usd'], 'prediction reservation') == PER_REQUEST and Decimal(reserve['reserved_nano'])/NANO == PER_REQUEST, 'Per-request reservation differs from $0.002688')
            require(details['row_id'] == row['row_id'] and details['provider'] == 'jev' and details['model'] == MODEL, 'Reserved row/model/provider differs')
            require(details['endpoint'] == routed['endpoint'] == ENDPOINT and details['route'] == 'OpenRouter', 'Reserved/predicted endpoint differs')
            require(all(details.get(key) == value for key, value in guard.items()), 'Reservation provenance differs from run guard')
            pricing = details['pricing']
            require(money(pricing['input_usd_per_million'], 'input rate') == INPUT_RATE and money(pricing['output_usd_per_million'], 'output rate') == 0, 'Reservation token pricing differs')
            require(pricing['reserve_input_tokens'] == 64000 and money(pricing['per_request_reserved_usd'], 'declared reservation') == PER_REQUEST, 'Reservation pricing bound differs')
            require(outcome['outcome'] == ('prediction_error' if failed else 'returned'), 'Prediction/ledger outcome differs')
            require(outcome['reservation_released'] is False and outcome['usage_exceeded_reservation_assumptions'] is False, 'Ledger result is released or halted')
            require(isinstance(outcome.get('guard_reasons'), dict) and not any(outcome['guard_reasons'].values()), 'Ledger contains a guard violation')
            actual_model = metadata.get('resolved_model')
            require(actual_model == outcome['resolved_model'] and (actual_model in ALLOWED_MODELS or (failed and actual_model is None)), 'Returned Jev model differs or is not allowed')
            require(routed['provider'] == outcome['provider'] and (routed['provider'] == 'TypeSafe' or (failed and routed['provider'] is None)) and outcome['route'] == 'OpenRouter', 'Returned Jev provider/route differs')
            request_id = routed['id']
            require(request_id == outcome['request_id'], 'Prediction/ledger API request ID differs')
            if request_id is not None:
                require(isinstance(request_id, str) and bool(request_id) and request_id not in request_ids, 'Repeated or invalid API request ID')
                request_ids.add(request_id)
            else:
                require(failed, 'Successful prediction lacks API request ID')
            cost = optional_money(routed.get('reported_cost_usd'), 'reported OpenRouter cost')
            require(cost == optional_money(outcome.get('reported_cost_usd'), 'ledger reported cost'), 'Prediction/ledger reported cost differs')
            require(cost is None or cost <= PER_REQUEST, 'Reported API cost exceeds retained reservation')
            inputs, outputs = row['input_tokens'], row['output_tokens']
            require(inputs is None or (type(inputs) is int and 0 <= inputs <= 64000), 'Invalid input token usage')
            require(outputs is None or (type(outputs) is int and outputs >= 0), 'Invalid output token usage')
            require(outcome['usage'] == {'input_tokens': inputs, 'output_tokens': outputs}, 'Prediction/ledger token usage differs')
            if actual_model is not None:
                resolved_models.add(actual_model)
            if inputs is not None:
                input_tokens += inputs; input_known += 1
                estimated += Decimal(inputs)*INPUT_RATE/MILLION
            if outputs is not None:
                output_tokens += outputs; output_known += 1
            if cost is not None:
                reported += cost; cost_known += 1
            reserved += PER_REQUEST
        require(errors == run['metrics']['n_failures'], 'Recorded failure metric differs from prediction errors')
        runs.append({'run_id': run['run_id'], 'source_path': path.relative_to(root).as_posix(),
                     'dataset': run['dataset'], 'method': run['method'], 'model': MODEL,
                     'resolved_models': sorted(resolved_models), 'endpoint': ENDPOINT,
                     'requests': len(rows), 'n_errors': errors,
                     'input_tokens': input_tokens if input_known == len(rows) else None,
                     'output_tokens': output_tokens if output_known == len(rows) else None,
                     'input_tokens_known_total': input_tokens, 'output_tokens_known_total': output_tokens,
                     'input_usage_known_requests': input_known, 'output_usage_known_requests': output_known,
                     'reported_cost_known_requests': cost_known,
                     'reported_api_cost_usd': str(reported) if cost_known == len(rows) else None,
                     'reported_api_cost_known_subtotal_usd': str(reported),
                     'token_rate_estimate_usd': str(estimated) if input_known == len(rows) else None,
                     'token_rate_estimate_known_subtotal_usd': str(estimated),
                     'retained_reservation_usd': str(reserved)})
        for artifact in (path, predictions_path, manifest_path):
            file_hashes[artifact.relative_to(root).as_posix()] = sha_file(artifact)
    expected = {(dataset, method) for dataset in ('sst2', 'trec') for method in ('zero_shot', 'few_shot')}
    require(conditions == expected, 'Final Jev accounting requires exactly SST2/TREC zero/few-shot conditions')
    require(seen == set(reserves) == set(results), 'Ledger requests are not exactly represented by completed prediction artifacts')
    require(len(seen) == 4*rows_per_run, 'Unexpected Jev total request count')
    totals = {field: sum((Decimal(run[field]) for run in runs), Decimal(0))
              for field in ('reported_api_cost_known_subtotal_usd', 'token_rate_estimate_known_subtotal_usd', 'retained_reservation_usd')}
    require(totals['retained_reservation_usd'] == money(snapshot['charged_or_reserved_usd'], 'snapshot reservation'), 'Per-run reservations do not reconcile with ledger')
    require(totals['retained_reservation_usd'] <= Decimal('2.50'), 'Jev allocation exceeded')
    # Reject a concurrent append instead of publishing a mixed-time reconciliation.
    require(all(sha_file(root/name) == digest for name, digest in file_hashes.items()), 'An input artifact changed during reconciliation')
    counts = {field: sum(run[field] for run in runs) for field in
              ('n_errors', 'reported_cost_known_requests', 'input_usage_known_requests', 'output_usage_known_requests')}
    unknown_costs = len(seen)-counts['reported_cost_known_requests']
    return {'scope': 'Jev native Choice via OpenRouter', 'model': MODEL, 'endpoint': ENDPOINT,
            'allocation_usd': '2.50', 'per_request_retained_usd': str(PER_REQUEST),
            'reported_cost_coverage': str(Decimal(counts['reported_cost_known_requests'])/Decimal(len(seen))),
            'reported_cost_unknown_requests': unknown_costs,
            'input_usage_coverage': str(Decimal(counts['input_usage_known_requests'])/Decimal(len(seen))),
            'output_usage_coverage': str(Decimal(counts['output_usage_known_requests'])/Decimal(len(seen))),
            'accounted_api_cost_conditional_upper_usd': str(totals['reported_api_cost_known_subtotal_usd']+PER_REQUEST*unknown_costs),
            'reported_api_cost_usd': str(totals['reported_api_cost_known_subtotal_usd']) if counts['reported_cost_known_requests'] == len(seen) else None,
            'token_rate_estimate_usd': str(totals['token_rate_estimate_known_subtotal_usd']) if counts['input_usage_known_requests'] == len(seen) else None,
            'requests': len(seen), 'ledger': snapshot, **counts,
            'wrapper_sha256': wrapper_sha, 'ledger_driver_sha256': driver_sha,
            'runs': runs, **{key: str(value) for key, value in totals.items()}, 'files_sha256': file_hashes}


def build_report(root=ROOT):
    jev = reconcile_jev(root)
    openai = reconcile_openai(root)
    require(openai['ledger']['reservations'] == 1600 and len(openai['runs']) == 8,
            'Combined final accounting requires all eight OpenAI pilot runs and 1,600 requests')
    openai_estimate = money(openai['reported_usage_standard_rate_estimate_usd'], 'OpenAI reported-token estimate')
    openai_conservative = money(openai['ledger']['charged_or_reserved_usd'], 'OpenAI conservative ledger total')
    combined_conservative = openai_conservative + Decimal(jev['retained_reservation_usd'])
    require(combined_conservative <= Decimal('10.00'), 'Combined conservative allocations exceed the approved ceiling')
    return {'basis': 'OpenRouter-reported Jev API cost and separately recomputed input-token estimate; combined OpenAI value is a reported-token standard-rate estimate, not API-reported dollars. None is a provider invoice.',
            'approved_total_usd': '10.00', 'openai_allocation_usd': '7.50', 'jev_allocation_usd': '2.50',
            'jev': jev, 'openai': {'requests': openai['ledger']['reservations'],
                'reported_usage_standard_rate_estimate_usd': str(openai_estimate),
                'conservative_settlement_usd': str(openai_conservative),
                'ledger': openai['ledger'], 'files_sha256': openai['files_sha256']},
            'combined_openai_token_estimate_plus_jev_reported_api_cost_usd': str(openai_estimate+Decimal(jev['reported_api_cost_usd'])) if jev['reported_api_cost_usd'] is not None else None,
            'combined_openai_estimate_plus_jev_reported_known_subtotal_usd': str(openai_estimate+Decimal(jev['reported_api_cost_known_subtotal_usd'])),
            'combined_openai_estimate_plus_jev_accounted_conditional_upper_usd': str(openai_estimate+Decimal(jev['accounted_api_cost_conditional_upper_usd'])),
            'combined_token_rate_estimate_usd': str(openai_estimate+Decimal(jev['token_rate_estimate_usd'])) if jev['token_rate_estimate_usd'] is not None else None,
            'combined_token_rate_estimate_known_subtotal_usd': str(openai_estimate+Decimal(jev['token_rate_estimate_known_subtotal_usd'])),
            'combined_conservative_charged_or_reserved_usd': str(combined_conservative),
            'combined_remaining_allocation_usd': str(Decimal('10.00')-combined_conservative)}


def main():
    report = build_report(); jev = report['jev']; openai = report['openai']
    lines = ['# Jev cost reconciliation and combined API accounting', '',
             f"All **{jev['requests']:,} Jev request reservations** reconcile one-to-one with completed prediction artifacts and final ledger results, including **{jev['n_errors']} recorded errors**. Known OpenRouter-reported costs sum to **US${Decimal(jev['reported_api_cost_known_subtotal_usd']):.9f}**, covering **{jev['reported_cost_known_requests']}/{jev['requests']} requests**; **{jev['reported_cost_unknown_requests']} request costs remain unknown**. Missing costs are never counted as zero. These values are not a provider invoice.", '',
             f"Independently applying $0.042 per million known input tokens and zero output cost gives a known token-rate subtotal of **US${Decimal(jev['token_rate_estimate_known_subtotal_usd']):.9f}**, covering **{jev['input_usage_known_requests']}/{jev['requests']} input-usage records**. Complete-cost/usage totals are null in JSON when their coverage is incomplete.", '',
             f"All US${PER_REQUEST} per-request reservations remain retained: **US${Decimal(jev['retained_reservation_usd']):.6f}** against the separate US$2.50 Jev allocation. Known reported cost plus full reservations for unknown-cost requests is **US${Decimal(jev['accounted_api_cost_conditional_upper_usd']):.9f}**: a conditional accounted upper estimate, distinct from both an invoice and the all-request conservative ledger total.", '',
             '| Dataset | Method | Requests / errors | Known input / output tokens | Cost coverage | Known API-cost subtotal USD | Known token-rate subtotal USD | Retained reservation USD |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for run in jev['runs']:
        lines.append(f"| {run['dataset']} | {run['method']} | {run['requests']} / {run['n_errors']} | {run['input_tokens_known_total']} / {run['output_tokens_known_total']} | {run['reported_cost_known_requests']}/{run['requests']} | {Decimal(run['reported_api_cost_known_subtotal_usd']):.9f} | {Decimal(run['token_rate_estimate_known_subtotal_usd']):.9f} | {Decimal(run['retained_reservation_usd']):.6f} |")
    lines += ['', '## Combined accounting under US$10', '',
              '| Component / basis | USD |', '|---|---:|',
              f"| OpenAI: reported-token standard-rate estimate | {Decimal(openai['reported_usage_standard_rate_estimate_usd']):.9f} |",
              f"| Jev: known API-cost subtotal ({jev['reported_cost_known_requests']}/{jev['requests']} requests) | {Decimal(jev['reported_api_cost_known_subtotal_usd']):.9f} |",
              f"| OpenAI estimate + Jev known API-cost subtotal | {Decimal(report['combined_openai_estimate_plus_jev_reported_known_subtotal_usd']):.9f} |",
              f"| OpenAI estimate + Jev conditional accounted upper estimate | {Decimal(report['combined_openai_estimate_plus_jev_accounted_conditional_upper_usd']):.9f} |",
              f"| Both providers: known token-rate subtotals | {Decimal(report['combined_token_rate_estimate_known_subtotal_usd']):.9f} |",
              f"| OpenAI conservative settlement + Jev retained reservations | {Decimal(report['combined_conservative_charged_or_reserved_usd']):.9f} |",
              f"| Remaining across both separate allocations | {Decimal(report['combined_remaining_allocation_usd']):.9f} |", '',
              'OpenAI API-reported dollar cost is not available here: its number is derived from reported tokens at dated standard rates. OpenAI conservative settlements include a 1.25 input multiplier; Jev retains its entire reservation without settlement. The two ledgers retain their separate US$7.50/US$2.50 limits. Remaining allocations are not automatically transferable.', '',
              'Reported API cost, known token-rate subtotals and retained reservations are distinct. A timeout/error may still be billed; unknown cost/usage is preserved and its full reservation remains. The conditional accounted upper estimate assumes each unknown request fits its reserved limit. Caching, taxes, unrelated account activity, billing adjustments and Colab compute can make actual invoiced totals differ. Known usage and API cost are checked against each corresponding ledger result with exact Decimal strings. Returned model/provider/request IDs must match when present; unknown response metadata is allowed only for recorded errors. Every reserved endpoint is checked. Source/ledger/artifact hashes are retained in the JSON.', '',
              'The rate was frozen for the execution at [OpenRouter Jev pricing](https://openrouter.ai/typesafe/jev-1.13), with the [native System One route](https://openrouter.ai/docs/guides/community/typesafe-sdk). This reconciliation does not update prices or make requests.', '',
              '[Machine-readable reconciliation](JEV_COSTS.json) · [OpenAI-only accounting](API_COSTS.md) · [Budget assumptions](../docs/BUDGET.md)', '',
              'Rebuild with `python scripts/summarize_jev_costs.py` only after all jobs finish. Keep each ledger and its `.lock` anchor together.', '']
    (ROOT/'results/JEV_COSTS.json').write_text(json.dumps(report, indent=2)+'\n')
    (ROOT/'results/JEV_COSTS.md').write_text('\n'.join(lines))
    print(json.dumps({'jev_requests': jev['requests'], 'jev_reported_api_cost_usd': jev['reported_api_cost_usd'],
                      'combined_conservative_usd': report['combined_conservative_charged_or_reserved_usd']}))


if __name__ == '__main__':
    main()
