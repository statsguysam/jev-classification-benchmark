"""Reconstruction and exclusive publication; all fixtures are offline."""
import copy
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import restore_review_controls_payload
    return restore_review_controls_payload


@pytest.fixture
def frozen(module,tmp_path,monkeypatch):
    folder=tmp_path/'prepared_full';folder.mkdir()
    requests=[{'request_id':'fixture-1','prompt':'invented UI test input','choices':['a','b']}]
    payload=module.prepare.request_bytes(requests)
    helpers={'fixture':'hash'}
    protocol={'schema_version':1,'study':'proposal-value-controls-v1','status':'prepared_unexecuted',
        'seed':42,'limit_per_dataset':None,'requests_content_sha256':module.prepare.sha(payload),
        'source_helper_sha256':helpers,'fixture_only':True}
    raw=json.dumps(protocol,sort_keys=True,indent=2).encode()+b'\n'
    manifest={'schema_version':1,'study':'proposal-value-controls-v1','status':'prepared_unexecuted',
        'files_sha256':{'protocol.json':module.prepare.sha(raw),'requests.jsonl':module.prepare.sha(payload)}}
    (folder/'protocol.json').write_bytes(raw)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(module.prepare,'source_hashes',lambda root:copy.deepcopy(helpers))
    monkeypatch.setattr(module.prepare,'load_jobs',lambda root:{'fixture':'jobs'})
    def generated(*args,**kwargs):
        p=copy.deepcopy(protocol);p.pop('source_helper_sha256')
        return {'protocol':p,'requests':copy.deepcopy(requests)}
    monkeypatch.setattr(module.prepare,'build_plan',generated)
    return folder,payload,protocol,manifest


def test_missing_payload_restored_atomically_and_fully_revalidated(module,frozen):
    folder,payload,_,_=frozen
    before={p.name:p.read_bytes() for p in folder.iterdir()}
    result=module.restore_payload(folder)
    assert result['status']=='restored_missing_payload' and result['written'] is True
    assert result['request_count']==1 and result['model_calls_performed']==0
    assert result['budget_allocated_usd']=='0'
    assert (folder/'requests.jsonl').read_bytes()==payload
    assert {name:(folder/name).read_bytes() for name in before}==before
    assert not list(folder.parent.glob('.review-controls-restore-*.tmp'))
    assert module.prepare.load_frozen_plan(folder)['requests'][0]['request_id']=='fixture-1'


def test_existing_valid_payload_is_verified_without_writes(module,frozen):
    folder,payload,_,_=frozen
    path=folder/'requests.jsonl';path.write_bytes(payload)
    before=path.stat()
    result=module.restore_payload(folder)
    after=path.stat()
    assert result['status']=='verified_existing_payload' and result['written'] is False
    assert (before.st_ino,before.st_mtime_ns,before.st_size)==(after.st_ino,after.st_mtime_ns,after.st_size)


def test_existing_partial_payload_never_overwritten(module,frozen):
    folder,_,_,_=frozen
    path=folder/'requests.jsonl';path.write_bytes(b'{"partial":')
    with pytest.raises(ValueError,match='partial or changed'):
        module.restore_payload(folder)
    assert path.read_bytes()==b'{"partial":'


@pytest.mark.parametrize('filename',['requests.jsonl.partial','run.json.tmp','unrecognized-file'])
def test_unknown_or_partial_files_block_restoration(module,frozen,filename):
    folder,_,_,_=frozen
    path=folder/filename;path.write_text('preserve me')
    with pytest.raises(ValueError,match='unknown/partial'):
        module.restore_payload(folder)
    assert path.read_text()=='preserve me' and not (folder/'requests.jsonl').exists()


def test_retained_protocol_tampering_blocks_write(module,frozen):
    folder,_,_,_=frozen
    path=folder/'protocol.json';path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='Retained protocol hash'):
        module.restore_payload(folder)
    assert not (folder/'requests.jsonl').exists()


def test_changed_reconstruction_protocol_blocks_write(module,frozen,monkeypatch):
    folder,_,_,_=frozen
    original=module.prepare.build_plan
    def changed(*a,**k):
        result=original(*a,**k);result['protocol']['seed']=43
        return result
    monkeypatch.setattr(module.prepare,'build_plan',changed)
    with pytest.raises(ValueError,match='Reconstructed protocol differs'):
        module.restore_payload(folder)
    assert not (folder/'requests.jsonl').exists()


def test_changed_reconstructed_payload_blocks_write(module,frozen,monkeypatch):
    folder,_,_,_=frozen
    original=module.prepare.build_plan
    def changed(*a,**k):
        result=original(*a,**k);result['requests'][0]['prompt']='changed'
        return result
    monkeypatch.setattr(module.prepare,'build_plan',changed)
    with pytest.raises(ValueError,match='request payload hash differs'):
        module.restore_payload(folder)
    assert not (folder/'requests.jsonl').exists()


def test_atomic_publish_race_preserves_competing_file(module,frozen,monkeypatch):
    folder,_,_,_=frozen
    original=module.os.link
    def competitor(source,target,**kwargs):
        Path(target).write_bytes(b'other process partial content')
        return original(source,target,**kwargs)
    monkeypatch.setattr(module.os,'link',competitor)
    with pytest.raises(ValueError,match='appeared during publication'):
        module.restore_payload(folder)
    assert (folder/'requests.jsonl').read_bytes()==b'other process partial content'
    assert not list(folder.parent.glob('.review-controls-restore-*.tmp'))


def test_symlink_payload_is_rejected_without_touching_target(module,frozen,tmp_path):
    folder,_,_,_=frozen
    outside=tmp_path/'outside';outside.write_text('untouched')
    (folder/'requests.jsonl').symlink_to(outside)
    with pytest.raises(ValueError,match='without symlinks'):
        module.restore_payload(folder)
    assert outside.read_text()=='untouched'


def test_final_full_audit_is_required(module,frozen,monkeypatch):
    folder,payload,_,_=frozen
    def reject(*a,**k):raise ValueError('source audit no longer agrees')
    monkeypatch.setattr(module.prepare,'load_frozen_plan',reject)
    with pytest.raises(ValueError,match='source audit no longer agrees'):
        module.restore_payload(folder)
    assert (folder/'requests.jsonl').read_bytes()==payload
    # Preserve the verified payload for inspection; never silently erase evidence.
