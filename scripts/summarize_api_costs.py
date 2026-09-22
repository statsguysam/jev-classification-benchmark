"""Reconcile completed hosted prediction metadata with the preserved budget ledger."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from run_budgeted_hosted import Ledger

ROOT = Path(__file__).resolve().parents[1]


def main():
    ledger_path = ROOT / 'results/openai-budget.jsonl'
    snapshot = Ledger(ledger_path, '7.50').snapshot()
    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    reserves = {e['reservation_id']: e for e in events if e['type'] == 'reserve'}
    settlements = {e['reservation_id']: e for e in events if e['type'] == 'settle'}
    seen = set()
    runs = []
    for path in sorted((ROOT / 'results/hosted').glob('*/run.json')):
        run = json.loads(path.read_text())
        if run['status'] != 'complete':
            raise ValueError('Finish or separately account for incomplete hosted runs before final cost reporting')
        rows = [json.loads(line) for line in path.with_name('predictions.jsonl').read_text().splitlines()]
        reported, conservative = Decimal(0), Decimal(0)
        for row in rows:
            budget = row['metadata']['budget']
            rid = budget['reservation_id']
            if rid in seen or rid not in reserves or rid not in settlements:
                raise ValueError('Duplicate, absent or unsettled reservation; preserve full ledger and investigate')
            if budget['ledger_id'] != snapshot['ledger_id']:
                raise ValueError('Prediction belongs to a different ledger')
            charge = Decimal(settlements[rid]['charged_nano']) / Decimal(10**9)
            if charge != Decimal(budget['settled_conservative_usd']):
                raise ValueError('Prediction/ledger settlement mismatch')
            seen.add(rid)
            reported += Decimal(budget['reported_usage_estimate']['usd'])
            conservative += charge
        runs.append({'run_id': run['run_id'], 'dataset': run['dataset'],
                     'model': run['config']['model'], 'method': run['method'],
                     'requests': len(rows), 'input_tokens': sum(r['input_tokens'] for r in rows),
                     'output_tokens_including_reasoning': sum(r['output_tokens'] for r in rows),
                     'reported_usage_standard_rate_estimate_usd': str(reported),
                     'conservative_settlement_usd': str(conservative)})
    if seen != set(reserves) or seen != set(settlements):
        raise ValueError('Ledger contains requests not fully represented by completed prediction artifacts')
    total = sum((Decimal(r['conservative_settlement_usd']) for r in runs), Decimal(0))
    if total != Decimal(snapshot['charged_or_reserved_usd']):
        raise ValueError('Per-run totals do not reconcile with ledger')
    report = {'basis': 'Reported token usage at dated standard rates; not a provider invoice. Conservative settlements add a 1.25 input multiplier.',
              'approved_total_usd': '10.00', 'openai_allocation_usd': '7.50',
              'jev_allocation_usd': '2.50', 'jev_spent_usd': '0.00',
              'ledger': snapshot, 'runs': runs,
              'reported_usage_standard_rate_estimate_usd': str(sum((Decimal(r['reported_usage_standard_rate_estimate_usd']) for r in runs), Decimal(0))),
              'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ledger_path, ledger_path.with_name(ledger_path.name+'.lock')]}}
    (ROOT / 'results/API_COSTS.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Hosted API cost accounting', '',
             f"All {len(seen):,} request reservations reconcile one-to-one with completed prediction records and settlements. The conservative ledger total is **US${total:.6f}**, within the US$7.50 OpenAI allocation. The reported-token estimate at uncached standard rates is **US${Decimal(report['reported_usage_standard_rate_estimate_usd']):.6f}**. These are accounting estimates, not a provider invoice.", '',
             '**US$2.50 remains reserved and unspent for Jev**, whose API access is unavailable. The approved combined ceiling is US$10. No paid Colab upgrade was purchased for this experiment; account-level compute charges, if any, are not inferred from API tokens.', '',
             '| Dataset | Model | Method | Requests | Input tokens | Output tokens, including reasoning | Standard-rate estimate USD | Conservative ledger USD |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['method']} | {r['requests']} | {r['input_tokens']} | {r['output_tokens_including_reasoning']} | {Decimal(r['reported_usage_standard_rate_estimate_usd']):.6f} | {Decimal(r['conservative_settlement_usd']):.6f} |")
    lines += ['', 'The ledger includes a conservative input-price multiplier. Token pricing assumptions, caches, taxes and unrelated account activity can make the provider invoice differ. Per-row metadata retains both the initial reservation and final settlement. Preserve the ledger and its `.lock` anchor together. Re-running this script sends no API requests.', '',
              '[Machine-readable reconciliation](API_COSTS.json) · [Budget assumptions and resume procedure](../docs/BUDGET.md)', '',
              'Rebuild with `python scripts/summarize_api_costs.py` after workers have finished.', '']
    (ROOT / 'results/API_COSTS.md').write_text('\n'.join(lines))
    print(json.dumps({'requests': len(seen), 'conservative_usd': str(total), 'reported_estimate_usd': report['reported_usage_standard_rate_estimate_usd']}))


if __name__ == '__main__':
    main()
