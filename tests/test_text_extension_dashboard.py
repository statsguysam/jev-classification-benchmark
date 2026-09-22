"""Offline aggregate safety, consistency and stale-report export checks."""
import copy
import json
from pathlib import Path

import pytest


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import build_text_extension_dashboard
    return build_text_extension_dashboard


@pytest.fixture
def report():
    return json.loads((Path(__file__).parents[1]/'results/text_extension/COMPARISON.json').read_text())


def test_partial_export_needs_explicit_flag_and_keeps_nulls(exporter,report):
    if report['status']=='complete':pytest.skip('Real report is now complete')
    with pytest.raises(ValueError,match='all 68'):
        exporter.sanitize(report)
    result=exporter.sanitize(report,allow_incomplete=True)
    assert len(result['runs'])==68 and len(result['comparisons'])==72
    assert result['completion']['complete_runs']==report['complete_runs']
    for row in result['runs']:
        if row['status']!='complete':
            for key in ['accuracy','macro_f1','accuracy_ci95','review_transitions','n_failures','n_test']:
                assert row[key] is None
    assert result['datasets'][0]['n_features'] is None
    assert result['costs']['expected_review_conditions']==24
    assert result['costs']['reused_review_conditions']==0


def test_allowlist_omits_raw_content_paths_and_configuration(exporter,report):
    report['untrusted_extra']='TOP_LEVEL_DO_NOT_EXPORT'
    report['runs'][0].update(raw_text='ROW_TEXT_DO_NOT_EXPORT',config={'secret':'CONFIG_DO_NOT_EXPORT'},source_path='PATH_DO_NOT_EXPORT')
    encoded=json.dumps(exporter.sanitize(report,allow_incomplete=True))
    assert all(value not in encoded for value in ['TOP_LEVEL_DO_NOT_EXPORT','ROW_TEXT_DO_NOT_EXPORT','CONFIG_DO_NOT_EXPORT','PATH_DO_NOT_EXPORT'])


@pytest.mark.parametrize('mutation',['invented_pending_score','duplicate_condition','source_link','cost_total','unknown_count','contrast_endpoint'])
def test_export_rejects_inconsistent_evidence(exporter,report,mutation):
    if mutation=='invented_pending_score':
        row=next((r for r in report['runs'] if r['status']!='complete'),None)
        if row is None:pytest.skip('No pending conditions remain')
        row['accuracy']=0
    elif mutation=='duplicate_condition':report['runs'][-1]=copy.deepcopy(report['runs'][0])
    elif mutation=='source_link':next(r for r in report['runs'] if r['arm']=='review')['source_run_id']='invented'
    elif mutation=='cost_total':report['costs']['cumulative_conservative_usd']='24.99'
    elif mutation=='unknown_count':report['costs']['unknown_cost_requests']+=1
    elif mutation=='contrast_endpoint':report['comparisons'][0]['b']='invented'
    with pytest.raises(ValueError):exporter.sanitize(report,allow_incomplete=True)


def test_zero_accuracy_is_preserved(exporter,report):
    row=next(r for r in report['runs'] if r['arm']=='classical')
    row['accuracy']=0
    row['group_bootstrap']['metrics']['accuracy']={'estimate':0,'ci95':[0,.1]}
    result=exporter.sanitize(report,allow_incomplete=True)
    assert next(r for r in result['runs'] if r['run_id']==row['run_id'])['accuracy']==0


def test_stale_report_never_replaces_output(exporter,report,tmp_path,monkeypatch):
    folder=tmp_path/'results/text_extension';folder.mkdir(parents=True)
    (folder/'COMPARISON.json').write_text(json.dumps(report))
    output=tmp_path/'public.json';output.write_text('KEEP_EXISTING')
    changed=copy.deepcopy(report);changed['source_sha256']='changed'
    monkeypatch.setattr(exporter.scientific,'collect',lambda *a,**k:changed)
    with pytest.raises(ValueError,match='independently revalidated'):
        exporter.build(tmp_path,output,allow_incomplete=True)
    assert output.read_text()=='KEEP_EXISTING'


def test_verified_report_export_is_deterministic(exporter,report,tmp_path,monkeypatch):
    folder=tmp_path/'results/text_extension';folder.mkdir(parents=True)
    (folder/'COMPARISON.json').write_text(json.dumps(report))
    monkeypatch.setattr(exporter.scientific,'collect',lambda *a,**k:copy.deepcopy(report))
    first=exporter.build(tmp_path,tmp_path/'one.json',allow_incomplete=True)
    second=exporter.build(tmp_path,tmp_path/'two.json',allow_incomplete=True)
    assert first==second
    assert (tmp_path/'one.json').read_bytes()==(tmp_path/'two.json').read_bytes()
