#!/usr/bin/env python3
"""Restore only the ignored control requests from an immutable, audited plan.

No API calls, model loads, credentials, ledger changes or budget allocations.
Existing payloads are verified without writes. Partial or unknown files fail
closed; an exclusive atomic link publishes only a previously absent payload.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import prepare_review_controls as prepare

DEFAULT_PLAN=ROOT/'results/review_controls/prepared_full'


def require(condition,message):
    if not condition:
        raise ValueError(message)


def retained_artifacts(plan_dir):
    """Reject aliases and unknown/partial files before reconstruction or writes."""
    requested=Path(plan_dir)
    require(not requested.is_symlink(),'Plan directory must not be a symlink')
    folder=requested.resolve()
    require(folder.is_dir(),'Frozen plan directory is missing')
    expected={'protocol.json','manifest.json'}
    actual={path.name for path in folder.iterdir()}
    require(actual in (expected,expected|{'requests.jsonl'}),
            'Frozen plan has missing metadata or unknown/partial files; nothing was overwritten')
    for name in actual:
        require((folder/name).is_file() and not (folder/name).is_symlink(),
                'Frozen plan files must be regular files without symlinks')
    manifest=json.loads((folder/'manifest.json').read_text())
    require(set(manifest)=={'schema_version','study','status','files_sha256'}
        and manifest['schema_version']==prepare.SCHEMA_VERSION
        and manifest['study']=='proposal-value-controls-v1'
        and manifest['status']=='prepared_unexecuted'
        and set(manifest['files_sha256'])=={'protocol.json','requests.jsonl'},
        'Frozen plan manifest schema differs')
    require(prepare.file_sha(folder/'protocol.json')==manifest['files_sha256']['protocol.json'],
            'Retained protocol hash differs from its manifest')
    protocol=json.loads((folder/'protocol.json').read_text())
    require(protocol['requests_content_sha256']==manifest['files_sha256']['requests.jsonl'],
            'Protocol and manifest request hashes disagree')
    return folder,protocol,manifest


def reconstruct(protocol,manifest,*,root=ROOT):
    """Reaudit source predictions and compare every protocol field, not only IDs."""
    helpers=prepare.source_hashes(root)
    require(protocol['source_helper_sha256']==helpers,'Frozen helper identities changed')
    generated=prepare.build_plan(prepare.load_jobs(root),
        limit_per_dataset=protocol['limit_per_dataset'],seed=protocol['seed'])
    generated['protocol']['source_helper_sha256']=helpers
    require(generated['protocol']==protocol,'Reconstructed protocol differs from the retained frozen plan')
    payload=prepare.request_bytes(generated['requests'])
    require(prepare.sha(payload)==protocol['requests_content_sha256']==manifest['files_sha256']['requests.jsonl'],
            'Reconstructed request payload hash differs')
    return payload


def restore_payload(plan_dir=DEFAULT_PLAN,*,root=ROOT):
    folder,protocol,manifest=retained_artifacts(plan_dir)
    target=folder/'requests.jsonl'
    if target.exists():
        # A complete but corrupted or partial existing payload is never replaced.
        require(prepare.file_sha(target)==manifest['files_sha256']['requests.jsonl'],
                'Existing payload is partial or changed; nothing was overwritten')
        verified=prepare.load_frozen_plan(folder,revalidate_sources=True,root=root)
        return {'status':'verified_existing_payload','written':False,'plan_directory':str(folder),
                'request_count':len(verified['requests']),'requests_sha256':prepare.file_sha(target),
                'model_calls_performed':0,'budget_allocated_usd':'0'}
    payload=reconstruct(protocol,manifest,root=root)
    # Confirm retained metadata did not change while expensive source checks ran.
    current_folder,current_protocol,current_manifest=retained_artifacts(folder)
    require(current_folder==folder and current_protocol==protocol and current_manifest==manifest,
            'Plan metadata changed during reconstruction')
    require(not target.exists() and not target.is_symlink(),
            'Payload appeared during reconstruction; refusing to overwrite it')
    stage=folder.parent/f'.review-controls-restore-{uuid.uuid4().hex}.tmp'
    try:
        flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)
        descriptor=os.open(stage,flags,0o600)
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        require(prepare.file_sha(stage)==manifest['files_sha256']['requests.jsonl'],
                'Staged request payload hash differs')
        # Same-filesystem link is atomic and fails if *anything* owns target,
        # unlike replace/rename APIs that can overwrite an existing payload.
        try:
            os.link(stage,target,follow_symlinks=False)
        except FileExistsError:
            raise ValueError('Payload appeared during publication; nothing was overwritten') from None
        directory_fd=os.open(folder,os.O_RDONLY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
    finally:
        if stage.exists():stage.unlink()
    verified=prepare.load_frozen_plan(folder,revalidate_sources=True,root=root)
    require(prepare.file_sha(target)==manifest['files_sha256']['requests.jsonl'],
            'Restored payload changed during final audit')
    return {'status':'restored_missing_payload','written':True,'plan_directory':str(folder),
            'request_count':len(verified['requests']),'requests_sha256':prepare.file_sha(target),
            'model_calls_performed':0,'budget_allocated_usd':'0'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    args=parser.parse_args(argv)
    result=restore_payload(args.plan)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError) as exc:
        raise SystemExit(f'RESTORE STOPPED: {exc}') from None
