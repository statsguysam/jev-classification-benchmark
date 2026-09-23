"""Offline reporting checks; fixtures never become measured evidence."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from jevbench.metrics import evaluate
from jevbench.prompts import select_examples
from jevbench.runner import digest
from jevbench.types import Prediction, PreparedDataset, Row


@pytest.fixture
def report_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import summarize_text_extension
    return summarize_text_extension


@pytest.fixture
def source_fixture(tmp_path):
    dataset=PreparedDataset('sst2',['negative','positive'],
        [Row(f'train-{i}',f'unique training text {i}',i%2) for i in range(10)],
        [Row(f'val-{i}',f'validation {i}',i%2) for i in range(2)],
        [Row(f'test-{i}',f'heldout {i}',i%2) for i in range(4)], {'fixture':True})
    predictions=[Prediction(row.id,row.label,[.9,.1] if row.label==0 else [.1,.9]) for row in dataset.test]
    selected=select_examples(dataset.train,dataset.labels,4,42)
    record={'status':'complete','dataset':'sst2','labels':dataset.labels,'seed':42,
        'training_example_ids':[r.id for r in selected],'metrics':evaluate(dataset.test,predictions,2),
        'method':'few_shot','run_id':'fixture-run','manifest_sha256':'retained-original-manifest','config':{}}
    path=tmp_path/'run.json'
    path.write_text(json.dumps(record))
    path.with_name('predictions.jsonl').write_text(''.join(json.dumps(asdict(p))+'\n' for p in predictions))
    path.with_name('test_manifest.json').write_text('{}')
    return dataset,record,predictions,path


def measured(module,fixture):
    ds,r,ps,path=fixture
    return module.audited_row(ds,r,ps,model='gpt-6-astra',arm='base',budget=4,path=path,samples=100,root=path.parent)


def test_complete_inventory_keeps_every_pending_condition_unscored(report_module):
    r=report_module.assemble({}, {}, samples=100)
    assert len(r['runs'])==68 and len(r['comparisons'])==72
    assert r['complete_runs']==r['complete_comparisons']==0
    assert r['expected_new_review_runs']==24 and r['expected_reused_review_runs']==0
    byid={row['run_id']:row for row in r['runs']}
    for row in r['runs']:
        assert row['accuracy'] is row['macro_f1'] is row['n_test'] is row['n_failures'] is None
        if row['arm']=='review':
            base=byid[row['source_run_id']]
            assert base['dataset']==row['dataset'] and base['train_per_class']==row['train_per_class']
    assert all(c['paired_bootstrap'] is c['transitions'] is None for c in r['comparisons'])


def test_exact_comparison_design(report_module):
    specs=report_module.comparison_specs()
    for kind in ('review_minus_source','review_minus_jev_alone','few_minus_zero_descriptive'):
        subset=[s for s in specs if s[0]==kind]
        assert len(subset)==24
        assert all(s[3] is (kind!='few_minus_zero_descriptive') for s in subset)
    assert len(set((kind,a,b) for kind,a,b,_ in specs))==72


def test_audited_row_preserves_original_manifest_while_pairing_frozen_content(report_module,source_fixture):
    row,payload=measured(report_module,source_fixture)
    ds,_,_,_=source_fixture
    assert row['original_source_manifest_sha256']=='retained-original-manifest'
    assert payload['manifest_sha256']==digest(ds.manifest)
    assert row['train_labels']==8 and row['n_test']==4 and row['accuracy']==1
    assert row['group_bootstrap']['group_definition'].startswith('Unicode NFKC')


def test_jev_tied_maximum_keeps_saved_choice_and_metrics(report_module,source_fixture):
    ds,r,ps,path=source_fixture
    r['config']['provider']='jev'
    ps[0].label,ps[0].probabilities=1,[.5,.5]
    ps[0].metadata['probability_kind']='jev_choice_distribution'
    r['metrics']=evaluate(ds.test,ps,2)
    original=copy.deepcopy((r,ps))
    row,payload=measured(report_module,source_fixture)
    assert (r,ps)==original
    assert payload['predictions'][0]==1
    assert row['accuracy']==.75 and row['n_failures']==0
    assert row['chosen_label_argmax_disagreement_rows']==1


def test_jev_within_frozen_maximum_tolerance_keeps_saved_choice(report_module,source_fixture):
    ds,r,ps,path=source_fixture
    r['config']['provider']='jev'
    ps[0].label,ps[0].probabilities=1,[.5000001,.4999999]
    r['metrics']=evaluate(ds.test,ps,2)
    row,payload=measured(report_module,source_fixture)
    assert payload['predictions'][0]==1 and row['accuracy']==.75 and row['n_failures']==0


@pytest.mark.parametrize('provider,probabilities',[
    ('hf',[.5,.5]),('classical',[.5,.5]),('jev',[.6,.4]),('jev',[.500002,.499998])])
def test_tie_fix_preserves_other_recipes_and_rejects_beyond_tolerance(report_module,source_fixture,provider,probabilities):
    ds,r,ps,path=source_fixture
    r['config']['provider']=provider
    ps[0].label,ps[0].probabilities=1,probabilities
    r['metrics']=evaluate(ds.test,ps,2)
    with pytest.raises(ValueError,match='Invalid probabilities or decision rule'):
        measured(report_module,source_fixture)


@pytest.mark.parametrize('mutation',['order','training','metric','probability','failed_label'])
def test_rejects_misalignment_leakage_or_invalid_measurements(report_module,source_fixture,mutation):
    ds,r,ps,path=copy.deepcopy(source_fixture)
    if mutation=='order':ps.reverse()
    elif mutation=='training':r['training_example_ids'][0]=ds.test[0].id
    elif mutation=='metric':r['metrics']['accuracy']=.5
    elif mutation=='probability':ps[0].probabilities=[.2,.2]
    elif mutation=='failed_label':ps[0].error='failed'
    with pytest.raises(ValueError):measured(report_module,(ds,r,ps,path))


def test_complete_review_requires_its_completed_source(report_module):
    key=('sst2','gpt-6-astra','review',4)
    row=report_module.placeholder(key)
    row.update(status='complete',source_run_id='unknown')
    with pytest.raises(ValueError,match='unavailable/different source'):
        report_module.assemble({key:row},{},samples=100)


def test_pairing_rejects_unequal_test_rows_and_training_ids(report_module):
    keys=[('sst2','gpt-6-astra','review',4),('sst2','gpt-6-astra','base',4)]
    rows={key:report_module.placeholder(key) for key in report_module.expected_conditions()}
    for key in keys:rows[key].update(status='complete')
    rows[keys[0]]['source_run_id']=rows[keys[1]]['run_id']
    payload={'manifest_sha256':'same','y':[0,1],'predictions':[0,1],'groups':['a','b'],'n_classes':2,'test_row_ids':['a','b'],'training_example_ids':['train']}
    payloads={key:copy.deepcopy(payload) for key in keys}
    payloads[keys[0]]['test_row_ids']=['b','a']
    with pytest.raises(ValueError,match='unequal held-out'):
        report_module.compare(rows,payloads,samples=100)
    payloads[keys[0]]=copy.deepcopy(payload)
    payloads[keys[0]]['training_example_ids']=['test']
    with pytest.raises(ValueError,match='unequal training'):
        report_module.compare(rows,payloads,samples=100)


def test_cost_projection_preserves_unknown_reservations_and_verified_settlements(report_module,tmp_path,monkeypatch):
    import run_text_jev_review
    root=tmp_path
    (root/'results/text_extension').mkdir(parents=True)
    (root/'results/text_extension/review-budget.jsonl').write_text('fixture ledger audited separately')
    numeric_path=root/'results/numeric_expansion/review-budget.jsonl'
    numeric_path.parent.mkdir(parents=True);numeric_path.write_text('unchanged numeric fixture')
    monkeypatch.setattr(run_text_jev_review,'verify_envelope',
                        lambda root:{'charged_or_reserved_usd':'2.954112000'})
    events=[
        {'type':'reserve','reservation_id':'success','reserved_nano':2688000,'details':{'dataset':'sst2','proposal_key':'astra_k4'}},
        {'type':'result','reservation_id':'success','details':{'reported_cost_usd':'0.00005'}},
        {'type':'settle','reservation_id':'success','charged_nano':62500},
        {'type':'reserve','reservation_id':'failed','reserved_nano':2688000,'details':{'dataset':'sst2','proposal_key':'astra_k4'}},
        {'type':'result','reservation_id':'failed','details':{'reported_cost_usd':None}},
    ]
    audited={'events':events,'charged_nano':2750500,'ledger_sha256':'fixture-hash',
        'checkpoints':[(Path('fixture'),{'run_id':'partial','status':'stopped_after_billing_error'},[])]}
    result=report_module.cost_snapshot([],root,audited)
    assert result['status']=='halted'
    assert result['new_model_requests']==2 and result['reported_cost_requests']==1 and result['unknown_cost_requests']==1
    assert result['known_reported_api_usd']=='0.00005'
    assert result['review_conservative_usd']=='0.0027505'
    assert result['cumulative_conservative_usd']=='22.282690200'
    assert result['settled_requests']==1 and result['partial_checkpoint_files']==1
    relevant=next(r for r in result['condition_costs'] if (r['dataset'],r['model_key'],r['shots_per_class'])==('sst2','astra',4))
    assert relevant['model_requests']==2 and relevant['unknown_cost_requests']==1
    assert relevant['conservative_usd']=='0.0027505'
    audited['charged_nano']-=1
    with pytest.raises(ValueError,match='accounting differs'):
        report_module.cost_snapshot([],root,audited)


@pytest.mark.parametrize('relative',['condition/predictions.jsonl','condition/predictions.jsonl.partial',
                                    'condition/run.json.tmp','condition/run.json','nested/other/fragment'])
def test_missing_text_ledger_rejects_every_orphan_file(report_module,tmp_path,relative):
    path=tmp_path/'results/text_extension/review'/relative
    path.parent.mkdir(parents=True);path.write_text('unreconciled evidence')
    with pytest.raises(ValueError,match='artifacts or partial files exist without a ledger'):
        report_module.cost_snapshot([],tmp_path)


def test_cost_reporting_requires_pinned_numeric_prefix(report_module,tmp_path,monkeypatch):
    import run_text_jev_review as review
    numeric_path=tmp_path/'results/numeric_expansion/review-budget.jsonl'
    numeric_path.parent.mkdir(parents=True);numeric_path.write_text('replacement ledger')
    called=[]
    def reject(root):
        called.append(root)
        raise review.budget.GuardError('Earlier numeric ledger prefix changed or was replaced')
    monkeypatch.setattr(review,'verify_envelope',reject)
    with pytest.raises(review.budget.GuardError,match='numeric ledger prefix changed'):
        report_module.cost_snapshot([],tmp_path)
    assert called==[tmp_path]


def test_cost_reporting_rejects_numeric_ledger_change_during_audit(report_module,tmp_path,monkeypatch):
    import run_text_jev_review as review
    numeric_path=tmp_path/'results/numeric_expansion/review-budget.jsonl'
    numeric_path.parent.mkdir(parents=True);numeric_path.write_text('stable before')
    def mutate(root):
        numeric_path.write_text('appended after snapshot')
        return {'charged_or_reserved_usd':'2.954112000'}
    monkeypatch.setattr(review,'verify_envelope',mutate)
    with pytest.raises(ValueError,match='Numeric ledger changed during reporting'):
        report_module.cost_snapshot([],tmp_path)
