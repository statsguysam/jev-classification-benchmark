"""Aggregate-only publication checks; no audits, model calls or network needed."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_review_value_dashboard as dashboard


@pytest.fixture
def reports():
    return (json.loads((ROOT / "results/review_value/ANALYSIS.json").read_text()),
            json.loads((ROOT / "results/review_controls/COMPARISON.json").read_text()))


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def test_real_export_has_exact_complete_inventory_and_full_coverage_curves(reports):
    report, controls = reports
    output = dashboard.project(report, controls)
    assert output["complete_review_conditions"] == len(output["conditions"]) == report['complete_review_conditions']
    assert output["expected_review_conditions"] == 24
    assert len(output["pending_conditions"]) == 24-len(output['conditions'])
    keys = [(r["dataset"], r["source_model"], r["shots_per_class"]) for r in output["conditions"]]
    assert len(set(keys)) == len(keys)
    gates = [r for r in output["conditions"] if r["curve"] is not None]
    assert len(gates) == sum(bool(c['selective_review']) for c in report['conditions'])
    assert all(c["selective_review_eligibility"]["eligible"] == (c["curve"] is not None) for c in output["conditions"])
    for gate in gates:
        assert [p["requested_review_percent"] for p in gate["curve"]] == [0, 10, 25, 50, 75, 100]
        assert gate["curve"][0]["metrics"] == gate["metrics"]["never_review"]
        assert gate["curve"][-1]["metrics"] == gate["metrics"]["always_review"]
        assert all(p["required_inferences"]["review_api_requests"] == p["selected_rows"] for p in gate["curve"])
    for exported, original in zip(output["conditions"], report["conditions"]):
        for arm in ("never_review", "always_review", "direct_jev"):
            for metric in ("micro_accuracy", "balanced_accuracy", "macro_f1", "n_rows", "n_failures"):
                assert exported["metrics"][arm][metric] == original["metrics"][arm][metric]
    assert output["identical_prompts"] == {key:sum(p[key] for p in report['same_label_prompt_pairs'])
        for key in ('same_label_identical_prompt_rows','both_valid_rows','valid_output_disagreements','rows_with_either_review_failure')}
    # Wine Astra's inherited source failure must not become a review call.
    skipped = next(c for c in output["conditions"] if c["dataset"] == "wine" and c["source_model"] == "gpt-6-astra" and c["shots_per_class"] == 4)
    assert skipped["required_inferences"]["always_review"]["review_api_requests"] == 35
    assert skipped["metrics"]["always_review"]["n_failures"] == 1


def test_all_twelve_pending_controls_keep_null_metrics(reports):
    report, controls = copy.deepcopy(reports)
    for row in controls['runs']:
        row.update(status='pending', saved_requests=0, metrics=None, group_bootstrap=None, failures=None)
    for row in controls['comparisons']:
        row.update(status='pending', paired_group_bootstrap=None, balanced_accuracy_delta=None, transitions_B_to_A=None)
    controls['serving_repeat_diagnostic'].update(status='pending', available_complete_pairs=0, counts=None, by_dataset=None)
    output = dashboard.project(report, controls)
    assert len(output["controls"]["runs"]) == 12
    assert len({(r["dataset"], r["arm"]) for r in output["controls"]["runs"]}) == 12
    assert all(r["status"] == "pending" and r["metrics"] is None for r in output["controls"]["runs"])
    assert output["controls"]["planned_primary_requests"] == 1650
    assert output["controls"]["planned_repeats"] == 64
    assert all(r['paired_group_bootstrap'] is r['transitions_B_to_A'] is None for r in output['controls']['comparisons'])
    assert output['controls']['serving_repeat_diagnostic']['counts'] is None
    assert output['recovery']['status'] == 'not_available'


def test_export_never_contains_current_raw_identifiers_or_prompt_fields(reports):
    output = dashboard.project(*reports)
    forbidden = {"row_id", "row_ids", "test_row_ids", "selected_row_ids", "ranked_row_ids", "prompt",
        "source_max_probabilities", "source_run_id", "review_run_id", "request_id", "source_path",
        "config", "api_key", "artifacts", "files_sha256", "disagreement_row_ids", "prompt_hashes"}
    assert not any(forbidden & node.keys() for node in walk(output))
    blob = json.dumps(output)
    assert "source:00" not in blob and "Cached proposal class ID" not in blob


def test_nested_allowlist_omits_unexpected_sensitive_fields(reports):
    report, controlled = copy.deepcopy(reports)
    sentinel = "PRIVATE_PROMPT_ROW_SENTINEL"
    report["private_metadata"] = sentinel
    condition = report["conditions"][0]
    condition.update(prompt=sentinel, row_ids=[sentinel])
    condition["metrics"]["never_review"].update(prompt=sentinel, row_ids=[sentinel])
    condition["review_minus_base"].update(prompt=sentinel, row_id=sentinel)
    condition["selective_review_eligibility"]["raw_rows"] = [sentinel]
    point = next(c for c in report["conditions"] if c["selective_review"])["selective_review"]["points"][0]
    point.update(prompt=sentinel, selected_row_ids=[sentinel])
    point["metrics"].update(prompt=sentinel, row_id=sentinel)
    point["random_matched_rate"].update(prompt=sentinel, selected_row_ids=[sentinel])
    point["transitions_from_never_review"]["row_ids"] = [sentinel]
    point["required_inferences"]["request_ids"] = [sentinel]
    controlled["runs"][0].update(status="complete", group_bootstrap=bootstrap(.5), metrics={"accuracy": .5, "balanced_accuracy": .5,
        "macro_f1": .5, "n_test": 114, "n_failures": 0, "prompt": sentinel, "row_ids": [sentinel]})
    output = dashboard.project(report, controlled)
    assert sentinel not in json.dumps(output)
    assert output["controls"]["runs"][0]["metrics"]["accuracy"] == .5


def test_projection_does_not_mutate_reports(reports):
    before = copy.deepcopy(reports)
    dashboard.project(*reports)
    assert reports == before


def bootstrap(estimate, paired=False):
    return {'method':'paired group percentile bootstrap' if paired else 'group percentile bootstrap',
        'samples':2000,'seed':42,'n_rows':114,'n_groups':114,
        'metrics':{key:{'estimate':estimate,'ci95':[estimate-.1,estimate+.1]} for key in ('accuracy','macro_f1')}}


def test_complete_controls_project_paired_intervals_and_repeat_denominators(reports):
    report, controlled = copy.deepcopy(reports)
    for row in controlled['runs']:
        n=row['expected_requests']
        row.update(status='complete',saved_requests=n,metrics={'accuracy':.75,'balanced_accuracy':.7,'macro_f1':.6,'n_test':n,'n_failures':2},
                   group_bootstrap=bootstrap(.75),failures=[{'row_id':'PRIVATE','error':'PRIVATE'}])
    for row in controlled['comparisons']:
        row.update(status='complete',paired_group_bootstrap=bootstrap(.125,True),balanced_accuracy_delta=.05,
            transitions_B_to_A={'wrong_to_correct':15,'correct_to_wrong':3,'net_correct_change':12,'row_ids':['PRIVATE']})
    counts=dict(both_valid=61,valid_label_agreement=60,valid_label_disagreement=1,
        reference_failed_only=1,repeat_failed_only=1,both_failed=1,
        resolved_model_mismatch_among_valid=0,unknown_resolved_model_among_valid=0)
    controlled['serving_repeat_diagnostic'].update(status='complete',available_complete_pairs=64,counts=counts,
        by_dataset={'breast_cancer':counts},valid_pair_agreement=60/61,valid_agreement_fraction_of_all_pairs=60/64)
    output=dashboard.project(report,controlled)['controls']
    assert len(output['runs'])==12 and len(output['comparisons'])==8
    assert output['runs'][0]['metrics']['n_rows']==114
    contrast=output['comparisons'][0]
    assert contrast['paired_group_bootstrap']['metrics']['accuracy']=={'estimate':.125,'ci95':[.024999999999999994,.225]}
    assert contrast['balanced_accuracy_delta']==.05
    repeated=output['serving_repeat_diagnostic']
    assert repeated['counts']['both_failed']==1
    assert repeated['valid_pair_agreement']==60/61
    assert repeated['valid_agreement_fraction_of_all_pairs']==60/64
    assert 'PRIVATE' not in json.dumps(output)


@pytest.mark.parametrize('location', ['arm','contrast','repeat'])
def test_partial_control_outputs_cannot_publish_metrics(reports,location):
    report, controlled=copy.deepcopy(reports)
    if location=='arm':
        controlled['runs'][0].update(status='pending',metrics={'accuracy':.5})
    elif location=='contrast':
        controlled['comparisons'][0].update(status='pending',balanced_accuracy_delta=.1)
    else:
        controlled['serving_repeat_diagnostic'].update(status='pending',counts={'both_valid':1})
    with pytest.raises(ValueError,match='Incomplete|Pending'):
        dashboard.project(report,controlled)


def test_nonfinite_or_malformed_intervals_are_not_exported():
    value=bootstrap(.5)
    value['metrics']['accuracy']['ci95']=[float('nan'),.6]
    with pytest.raises(ValueError,match='confidence interval'):
        dashboard.uncertainty(value)


def recovery_fixture():
    metrics={'accuracy':.75,'balanced_accuracy':.7,'macro_f1':.6,'n_rows':36,'n_failures':1}
    view={'status':'complete','metrics':metrics}
    return {'status':'complete','execution_status':'complete','conditions':[{
        'dataset':'wine','model':'typesafe/jev-1.13','provider':'jev','source_model':'gpt-6-astra',
        'method':'cached_label_jev_review','shots_per_class':4,'scope':'current_four_dataset_study',
        'n_rows':36,'status':'complete','analysis_status':'audited_full_condition',
        'condition_variant':'dependent_source_recovery_overlay','n_new_calls':1,
        'dependent_review_first_calls':1,'same_request_as_original_snapshot':False,'unresolved_upstream_rows':0,
        'original_snapshot':copy.deepcopy(view),'first_attempt':copy.deepcopy(view),'recovered':copy.deepcopy(view),
        'paired_first_to_recovered':{'corrected':0,'harmed':0},'paired_group_bootstrap':bootstrap(0,True),
        'identity_provenance':{'prompt':'PRIVATE','row_id':'PRIVATE'},'original_run_id':'PRIVATE'}],
        'api_outcomes':{'new_calls':1,'ordinary_paid_failure_retry_calls':0,'dependent_review_calls':1,
            'dependent_first_calls':1,'upstream_unresolved_not_called':0},
        'budget':{'budget_usd':'0.02','charged_or_reserved_usd':'0.002688000',
            'remaining_reservation_usd':'0.017312000','reservations':1,'ledger_id':'PRIVATE'}}


def test_recovery_projection_keeps_three_views_and_omits_raw_provenance(reports):
    raw=recovery_fixture()
    raw['conditions'][0]['recovered']['metrics']['n_failures']=0
    output=dashboard.project(*reports,raw)['recovery']
    row=output['conditions'][0]
    assert row['original_snapshot']['n_failures']==1 and row['recovered']['n_failures']==0
    assert row['same_request_as_original_snapshot'] is False
    assert row['dependent_review_first_calls']==1
    assert row['paired_group_bootstrap']['metrics']['accuracy']['estimate']==0
    assert output['budget']['remaining_reservation_usd']=='0.017312000'
    assert 'remaining_usd' not in output['budget']
    assert 'Balanced-accuracy differences have no interval' in output['interval_note']
    assert 'PRIVATE' not in json.dumps(output)


def test_unresolved_recovery_is_not_scored(reports):
    raw=recovery_fixture(); row=raw['conditions'][0]
    row.update(status='incomplete',paired_first_to_recovered=None,paired_group_bootstrap=None)
    row['first_attempt']=row['recovered']={'status':'incomplete','metrics':None}
    output=dashboard.project(*reports,raw)['recovery']['conditions'][0]
    assert output['first_attempt'] is output['recovered'] is None
    assert output['corrected'] is output['harmed'] is output['paired_group_bootstrap'] is None
    assert output['original_snapshot']['n_failures']==1
    raw['execution_status']='running'
    with pytest.raises(ValueError,match='completed audited'):
        dashboard.project(*reports,raw)


def control_recovery_fixture():
    metrics={'accuracy':.75,'balanced_accuracy':.7,'macro_f1':.6,'n_rows':36,'n_failures':1}
    before={'metrics':metrics,'group_bootstrap':bootstrap(.75)}
    after={'metrics':{**metrics,'accuracy':.8,'n_failures':0},'group_bootstrap':bootstrap(.8)}
    paired={'paired_group_bootstrap':bootstrap(.1,True),'balanced_accuracy_delta':.05,
        'transitions_B_to_A':{'wrong_to_correct':2,'correct_to_wrong':1}}
    return {'status':'complete','study_role':'secondary_failure_recovery_sensitivity','no_op':False,
        'runs':[{'dataset':d,'arm':a,'status':'complete','n_rows':36,'resolved_failures':1,
            'first_attempt':copy.deepcopy(before),'recovered':copy.deepcopy(after),
            'paired_first_to_recovered':bootstrap(.05,True)} for d in dashboard.CONTROL_DATASETS for a in dashboard.CONTROL_ARMS],
        'comparisons':[{'dataset':d,'a':'actual','b':b,'contrast':'actual_minus_'+b,'status':'complete',
            'first_attempt':copy.deepcopy(paired),'recovered':copy.deepcopy(paired)}
            for d in dashboard.CONTROL_DATASETS for b in ('no_proposal','shuffled')],
        'recovery_counts':{scope:{'original_failures':12,'remaining_failures':0,'resolved_failures':12,'new_calls':12}
            for scope in ('primary','repeat')},
        'new_call_outcomes':{'n_calls':24},'budget':{'budget_usd':'1','remaining_reservation_usd':'.9','ledger_id':'PRIVATE'},
        'provenance':{'original_control_artifact_sha256':'PRIVATE'}}


def test_control_recovery_projection_remains_separate_and_aggregate_only(reports):
    raw=control_recovery_fixture()
    before=copy.deepcopy(raw)
    output=dashboard.project(*reports,None,raw)
    assert raw==before
    assert output['controls']==dashboard.project_controls(reports[1])
    recovered=output['control_recovery']
    assert len(recovered['runs'])==12 and len(recovered['comparisons'])==8
    assert recovered['runs'][0]['first_attempt']['metrics']['n_failures']==1
    assert recovered['runs'][0]['recovered']['metrics']['n_failures']==0
    assert recovered['budget']['remaining_reservation_usd']=='.9'
    assert 'PRIVATE' not in json.dumps(output)


def test_control_recovery_default_pending_and_partial_rejected():
    assert dashboard.project_control_recovery(None)['status']=='not_available'
    raw=control_recovery_fixture()
    raw['status']='running'
    with pytest.raises(ValueError,match='Only complete'):
        dashboard.project_control_recovery(raw)
    raw['status']='complete';raw['runs'].pop()
    with pytest.raises(ValueError,match='inventory differs'):
        dashboard.project_control_recovery(raw)
