"""Synthetic full-condition recovery aggregation; no live writes or API calls."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import summarize_failed_retries as summary
from jevbench.types import Prediction, PreparedDataset, Row


def prediction(row_id,label=None,error=None):
    return asdict(Prediction(row_id,label if not error else None,
        None if error else [.9,.1] if label==0 else [.1,.9],error=error,
        metadata={"budget":{"ledger_id":"original-ledger","reservation_id":"r-"+row_id}}))


@pytest.fixture
def study():
    rows=[Row(str(i),"same grouped input" if i<2 else "input "+str(i),i%2) for i in range(4)]
    original=[prediction("0",0),prediction("1",0),prediction("2",error="http_error: status=402; no automatic retry"),prediction("3",1)]
    ds=PreparedDataset("wine",["a","b"],[],[],rows)
    record={"run_id":"old-condition","dataset":"wine","method":"few_shot","config":{"provider":"jev","model":"typesafe/jev-1.13","shots_per_class":4}}
    item={"failure_id":"failure-c","run_id":record["run_id"],"row_id":"2","kind":"paid_failure","scope":"current_four_dataset_study",
        "original_prediction_sha256":summary.runner.preparation.prediction_sha256(original[2]),
        "artifact_sha256":{"results/fake/predictions.jsonl":"e"*64}}
    identity={"condition_id":record["run_id"],"row_id":"2","prompt_sha256":"a"*64,
        "choices_sha256":summary.analysis.digest(ds.labels),"source_sha256":"b"*64,
        "original_prediction_sha256":summary.analysis.digest(original[2])}
    new=prediction("2",0)
    new["metadata"]["retry"]={"failure_id":"failure-c"}
    attempt={"attempt_index":1,"identity":identity,"prediction":new,"stage":"retry"}
    audited={"plan":{"inventory":{"failures":[item]}},"record":{"status":"complete","counts":{"upstream_unresolved_not_called":0}},
        "attempts":[attempt],"budget":{"charged_or_reserved_usd":"0.002688000","reservations":1},
        "artifact_sha256":{"run.json":"c"*64}}
    reservations={"original-ledger":{"r-"+p["row_id"]:{"row_id":p["row_id"],"model":record["config"]["model"],"prompt_sha256":"a"*64} for p in original}}
    return audited,{record["run_id"]:(ds,record,original)},reservations


def run(study):
    return summary.assemble(*study,samples=100)


def test_full_condition_not_only_retried_rows_and_paired_ci(study):
    original=deepcopy(study)
    report=run(study); c=report["conditions"][0]
    assert c["n_rows"]==4 and c["n_new_calls"]==1
    assert c["original_snapshot"]["metrics"]["accuracy"]==.5
    assert c["first_attempt"]["metrics"]["accuracy"]==.5
    assert c["recovered"]["metrics"]["accuracy"]==.75
    assert c["original_snapshot"]["metrics"]["n_failures"]==1
    assert c["recovered"]["metrics"]["n_failures"]==0
    assert c["paired_first_to_recovered"]["corrected"]==1 and c["paired_first_to_recovered"]["harmed"]==0
    assert c["paired_group_bootstrap"]["metrics"]["accuracy"]["estimate"]==.25
    assert c["paired_group_bootstrap"]["n_groups"]==3
    assert "rows" not in c and "row_id" not in json.dumps(c)
    assert report["api_outcomes"]["ordinary_paid_failure_retry_calls"]==1
    assert report["api_outcomes"]["dependent_first_calls"]==0
    assert study==original


def test_valid_but_wrong_recovery_is_not_selected_using_truth(study):
    study[0]["attempts"][0]["prediction"]=prediction("2",1)
    study[0]["attempts"][0]["prediction"]["metadata"]["retry"]={"failure_id":"failure-c"}
    c=run(study)["conditions"][0]
    assert c["recovered"]["metrics"]["accuracy"]==.5
    assert c["recovered"]["metrics"]["n_failures"]==0
    assert c["paired_first_to_recovered"]["corrected"]==0


def test_dependent_first_call_has_separate_snapshot_and_first_call(study):
    audited,loaded,_=study
    ds,record,values=loaded["old-condition"]
    values[2]=prediction("2",error="source_proposal_failed: Jev review not called")
    values[2]["metadata"]={"proposal":{"source_model":"gpt-6-astra"}}
    item=audited["plan"]["inventory"]["failures"][0]
    item["kind"]="upstream_skip"
    item["original_prediction_sha256"]=summary.runner.preparation.prediction_sha256(values[2])
    attempt=audited["attempts"][0]
    attempt["identity"]["condition_id"]+="::dependent-source-recovery-v1"
    attempt["identity"]["original_prediction_sha256"]=summary.analysis.digest(values[2])
    attempt["stage"]="dependent_first_call"
    unchanged=deepcopy(attempt)
    c=run(study)["conditions"][0]
    assert c["condition_variant"]=="dependent_source_recovery_overlay"
    assert c["same_request_as_original_snapshot"] is False
    assert c["original_snapshot"]["metrics"]["accuracy"]==.5
    assert c["first_attempt"]["metrics"]["accuracy"]==c["recovered"]["metrics"]["accuracy"]==.75
    assert c["paired_first_to_recovered"]["corrected"]==0
    assert c["dependent_review_first_calls"]==1
    assert c["identity_provenance"]["condition_id_aliases"]=={"old-condition::dependent-source-recovery-v1":"old-condition::dependent-source-recovery-overlay-v1"}
    assert attempt==unchanged


def test_unresolved_upstream_has_no_invented_prompt_or_partial_scores(study,monkeypatch):
    audited,loaded,_=study
    _,_,values=loaded["old-condition"]
    values[2]=prediction("2",error="source_proposal_failed: Jev review not called")
    values[2]["metadata"]={"proposal":{"source_model":"gpt-6-astra"}}
    item=audited["plan"]["inventory"]["failures"][0]
    item.update(kind="upstream_skip",original_prediction_sha256=summary.runner.preparation.prediction_sha256(values[2]))
    audited["attempts"]=[]
    audited["record"]["counts"]["upstream_unresolved_not_called"]=1
    monkeypatch.setattr(summary.analysis,"analyze_condition",lambda *a,**k:pytest.fail("Invented a sent prompt for unresolved source"))
    report=run(study); c=report["conditions"][0]
    assert report["execution_status"]=="complete" and c["status"]=="incomplete"
    assert c["first_attempt"]["metrics"] is c["recovered"]["metrics"] is None
    assert c["first_attempt"]["n_not_attempted"]==1
    assert c["paired_group_bootstrap"] is None and c["paired_first_to_recovered"] is None
    assert c["original_snapshot"]["metrics"]["n_failures"]==1
    assert "condition_id_aliases" not in c["identity_provenance"]


@pytest.mark.parametrize("tamper",["identity","missing_original","missing_inventory","success_retry","original_digest"])
def test_full_condition_and_attempt_tampering_rejected(study,tamper):
    audited,loaded,_=study
    ds,record,original=loaded["old-condition"]
    if tamper=="identity": audited["attempts"][0]["identity"]["choices_sha256"]="f"*64
    if tamper=="missing_original": original.pop()
    if tamper=="missing_inventory": audited["plan"]["inventory"]["failures"]=[]
    if tamper=="success_retry": audited["attempts"][0]["identity"]["row_id"]="0"
    if tamper=="original_digest": audited["plan"]["inventory"]["failures"][0]["original_prediction_sha256"]="f"*64
    with pytest.raises(ValueError): run(study)


def test_completed_policy_cannot_omit_paid_attempt(study):
    study[0]["attempts"]=[]
    with pytest.raises(ValueError,match="omitted"):
        run(study)


def test_collect_requires_final_audit_and_reaudits_after_analysis(study,monkeypatch):
    audited,loaded,reservations=study
    calls=[]
    monkeypatch.setattr(summary.runner,"audit_run",lambda: calls.append(1) or deepcopy(audited))
    monkeypatch.setattr(summary,"load_original",lambda item:loaded[item["run_id"]])
    monkeypatch.setattr(summary,"original_reservations",lambda plan:reservations)
    result=summary.collect(samples=100)
    assert result["status"]=="complete" and len(calls)==2
    def incomplete(): raise ValueError("Finite recovery policy has not completed")
    monkeypatch.setattr(summary.runner,"audit_run",incomplete)
    with pytest.raises(ValueError,match="not completed"): summary.collect(samples=100)


def test_collect_rejects_mid_report_mutation(study,monkeypatch):
    audited,loaded,reservations=study
    calls=[]
    def audit():
        result=deepcopy(audited)
        if calls: result["artifact_sha256"]["run.json"]="f"*64
        calls.append(1)
        return result
    monkeypatch.setattr(summary.runner,"audit_run",audit)
    monkeypatch.setattr(summary,"load_original",lambda item:loaded[item["run_id"]])
    monkeypatch.setattr(summary,"original_reservations",lambda plan:reservations)
    with pytest.raises(ValueError,match="changed during aggregation"): summary.collect(samples=100)


def test_optional_report_never_reads_scores_while_incomplete(tmp_path,monkeypatch):
    monkeypatch.setattr(summary.runner,"AREA",tmp_path)
    monkeypatch.setattr(summary,"collect",lambda **kwargs:pytest.fail("Read incomplete scores"))
    assert summary.optional_collect() is None
    (tmp_path/"run.json").write_text(json.dumps({"status":"running"}))
    assert summary.optional_collect() is None
    (tmp_path/"run.json").unlink()
    (tmp_path/"attempts.jsonl").write_text("")
    with pytest.raises(ValueError,match="Orphan"): summary.optional_collect()


def test_failure_only_report_does_not_claim_semantic_improvement(study):
    text=summary.findings(run(study))
    assert "full original condition" in text
    assert "not attribute semantic improvements" in text
    assert "Original outcomes and reservations remain immutable" in text
