"""Export allowlisted frozen text inputs/code or audited local result evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_text_extension_local as runner

PREFIX = 'jev-text-extension'
RESULT_ROOT = 'results/text_extension/local'
RESULT_MANIFEST = 'results/text_extension/result_manifest.json'
RUN_FILES = ('run.json', 'predictions.jsonl', 'test_manifest.json')


def write_zip(target, contents):
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(contents.items()):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return target


def pins():
    runner.load_presets()
    return {'core_sha256': runner.frozen.FROZEN_CORE_SHA, 'numeric_helper_sha256': runner.FROZEN_HELPER_SHA,
            'text_wrapper_sha256': runner.file_sha(runner.__file__), 'preset_file_sha256': runner.FROZEN_PRESETS_SHA,
            'dataset_pins': runner.DATASET_PINS}


def export(root=ROOT):
    root = Path(root)
    identity = pins()
    names = ['pyproject.toml', 'configs/expanded_numeric_models.json', 'configs/datasets.json',
        'scripts/run_text_extension_local.py', 'scripts/export_text_extension_colab.py',
        'scripts/import_text_extension_colab.py', 'scripts/run_expanded_numeric_local.py',
        'scripts/run_tabular_classical.py', 'scripts/run_tabular_local.py', 'scripts/tabular_data.py',
        'tests/test_text_extension_local.py']
    names.extend(p.relative_to(root).as_posix() for p in (root/'src/jevbench').glob('*.py'))
    for dataset in runner.DATASETS:
        runner.load_dataset(root/'data/pilot'/dataset)
        names.extend(f'data/pilot/{dataset}/{name}' for name in ('manifest.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl'))
    contents = {}
    for name in sorted(set(names)):
        path = root/name
        runner.require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'External source member forbidden')
        contents[name] = path.read_bytes()
    manifest = {'schema_version': 1, **identity,
        'files_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()},
        'runtime': {'transformers': '4.57.6', 'huggingface-hub': '0.36.2', 'device': 'cuda', 'dtype': 'float16'},
        'data_notice': 'Exact SST2/TREC benchmark pilot text and labels from previously prepared public sources; source attribution, revisions and license notes are preserved in each manifest and configs/datasets.json. For research reproduction. No credentials, previous results, private records or model weights are included.'}
    contents['bundle_manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
    target = write_zip(root/'artifacts/jev-text-extension-colab.zip', {f'{PREFIX}/{name}': raw for name,raw in contents.items()})
    print(json.dumps({'path': str(target), 'files': len(contents), 'bytes': target.stat().st_size,
        'sha256': runner.file_sha(target), 'text_wrapper_sha256': identity['text_wrapper_sha256']}))
    return target


def export_results(root=ROOT, *, model_keys=runner.MODEL_KEYS):
    root = Path(root)
    runner.require(model_keys and len(model_keys)==len(set(model_keys)) and set(model_keys)<=set(runner.MODEL_KEYS), 'Unexpected model selection')
    expected = {(dataset, key, shots) for dataset in runner.DATASETS for key in model_keys for shots in (0,4)}
    datasets = {name: runner.load_dataset(root/'data/pilot'/name) for name in runner.DATASETS}
    contents, seen = {}, set()
    for path in sorted((root/RESULT_ROOT).glob('*/run.json')):
        record=json.loads(path.read_text())
        key=record['config']['renderer']['model_key']
        if key not in model_keys: continue
        condition=(record['dataset'], key, record['config']['shots_per_class'])
        runner.require(condition in expected and condition not in seen, 'Unexpected or repeated result condition')
        runner.audit_source_run(path,datasets[condition[0]],key,condition[2])
        runner.require({p.name for p in path.parent.iterdir()}==set(RUN_FILES), 'Unexpected result run member')
        seen.add(condition)
        for name in RUN_FILES:
            member=path.with_name(name)
            runner.require(not member.is_symlink(), 'Result symlink forbidden')
            contents[member.relative_to(root).as_posix()]=member.read_bytes()
    runner.require(seen==expected, 'Required complete result conditions are missing')
    for path in sorted((root/RESULT_ROOT/'plans').glob('*-preflight.json')):
        runner.require(not path.is_symlink(), 'Plan symlink forbidden')
        contents[path.relative_to(root).as_posix()]=path.read_bytes()
    manifest={'schema_version':1, **pins(), 'model_keys':list(model_keys),
              'files_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in contents.items()}}
    contents[RESULT_MANIFEST]=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    target=write_zip(root/'artifacts/jev-text-extension-results.zip',contents)
    print(json.dumps({'path':str(target),'sha256':runner.file_sha(target),'conditions':len(seen),'files':len(contents)}))
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',action='store_true')
    p.add_argument('--model-keys',nargs='+',choices=runner.MODEL_KEYS,default=list(runner.MODEL_KEYS))
    args=p.parse_args()
    export_results(model_keys=args.model_keys) if args.results else export()
