"""Verify checksummed, allowlisted Colab text results before immutable import."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_text_extension_local as runner
from export_text_extension_colab import RESULT_ROOT, RESULT_MANIFEST, RUN_FILES, pins
from jevbench.runner import save_json


def verify_members(archive):
    inventory={}
    total=0
    for item in archive.infolist():
        name=PurePosixPath(item.filename)
        runner.require(not name.is_absolute() and '..' not in name.parts and '\\' not in item.filename
                       and not stat.S_ISLNK(item.external_attr>>16), 'Unsafe archive path or symlink')
        if item.is_dir(): continue
        runner.require(item.filename not in inventory, 'Duplicate archive member')
        runner.require(item.file_size<=20_000_000,'Result member exceeds size limit')
        total+=item.file_size
        runner.require(total<=100_000_000,'Result archive exceeds size limit')
        if item.filename!=RESULT_MANIFEST:
            runner.require(name.is_relative_to(PurePosixPath(RESULT_ROOT)),'Unexpected result archive root')
            relative=name.relative_to(PurePosixPath(RESULT_ROOT))
            runner.require(len(relative.parts)==2 and ((relative.parts[0]=='plans' and relative.name.endswith('-preflight.json'))
                or (relative.parts[0]!='plans' and relative.name in RUN_FILES)), 'Unexpected result file')
        inventory[item.filename]=item
    runner.require(RESULT_MANIFEST in inventory,'Result checksum manifest missing')
    manifest_raw=archive.read(inventory[RESULT_MANIFEST])
    manifest=json.loads(manifest_raw)
    runner.require(manifest['schema_version']==1 and all(manifest.get(k)==v for k,v in pins().items()),'Result producer/source pins differ')
    runner.require(set(manifest['files_sha256'])==set(inventory)-{RESULT_MANIFEST},'Result checksum inventory differs')
    contents={}
    for name,expected in manifest['files_sha256'].items():
        content=archive.read(inventory[name])
        runner.require(hashlib.sha256(content).hexdigest()==expected,'Result member checksum differs')
        contents[name]=content
    return manifest,contents,hashlib.sha256(manifest_raw).hexdigest()


def import_archive(path, *, output=None, data_root=None, model_keys=runner.MODEL_KEYS, expected_sha256=None, execute=False):
    path=Path(path)
    archive_sha=runner.file_sha(path)
    runner.require(expected_sha256 is None or archive_sha==expected_sha256,'Downloaded archive checksum differs')
    runner.require(model_keys and len(model_keys)==len(set(model_keys)) and set(model_keys)<=set(runner.MODEL_KEYS),'Unexpected model selection')
    output=Path(output or ROOT/RESULT_ROOT)
    data_root=Path(data_root or ROOT/'data/pilot')
    with zipfile.ZipFile(path) as archive: manifest,contents,manifest_sha=verify_members(archive)
    runner.require(set(manifest['model_keys'])==set(model_keys) and len(manifest['model_keys'])==len(model_keys),'Result model selection differs')
    expected={(dataset,key,shots) for dataset in runner.DATASETS for key in model_keys for shots in (0,4)}
    report={'schema_version':1,'status':'verified','archive_sha256':archive_sha,
            'result_manifest_sha256':manifest_sha,
            'conditions':[],'files_sha256':{PurePosixPath(name).relative_to(RESULT_ROOT).as_posix():sha for name,sha in manifest['files_sha256'].items()}}
    with tempfile.TemporaryDirectory(prefix='jev-text-import-') as temporary:
        stage=Path(temporary)
        for name,content in contents.items():
            target=stage/PurePosixPath(name).relative_to(RESULT_ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        datasets={name:runner.load_dataset(data_root/name) for name in runner.DATASETS}
        seen=set()
        for folder in sorted(stage.iterdir()):
            if folder.name=='plans':continue
            runner.require(folder.is_dir() and {p.name for p in folder.iterdir()}==set(RUN_FILES),'Incomplete evidence run')
            record=json.loads((folder/'run.json').read_text())
            key=record['config']['renderer']['model_key'];shots=record['config']['shots_per_class']
            condition=(record['dataset'],key,shots)
            runner.require(condition in expected and condition not in seen,'Unexpected or repeated source condition')
            audited,predictions=runner.audit_source_run(folder/'run.json',datasets[condition[0]],key,shots)
            runner.require(audited==record,'Audited record differs')
            seen.add(condition)
            report['conditions'].append({'dataset':condition[0],'model_key':key,'shots_per_class':shots,
                'run_id':record['run_id'],'n_predictions':len(predictions)})
        runner.require(seen==expected,'Required complete conditions are missing')
        for name,expected_sha in report['files_sha256'].items():
            target=output/name
            runner.require(not target.is_symlink() and all(not p.is_symlink() for p in target.parents),'Destination symlinks forbidden')
            runner.require(not target.exists() or (target.is_file() and runner.file_sha(target)==expected_sha),'Existing evidence differs; refusing overwrite')
        if execute:
            for name in report['files_sha256']:
                target=output/name
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(stage/name,target)
            report['status']='imported'
            save_json(output.parent/'imports'/f'{archive_sha}.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--model-keys',nargs='+',choices=runner.MODEL_KEYS,default=list(runner.MODEL_KEYS))
    parser.add_argument('--expected-sha256')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    print(json.dumps(import_archive(args.archive,model_keys=args.model_keys,output=args.output,
                                   expected_sha256=args.expected_sha256,execute=args.execute),indent=2))
