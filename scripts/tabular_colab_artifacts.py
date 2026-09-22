"""Strict private Colab result/adapter transfer for the frozen tabular benchmark.

No model/API execution, raw rows, pretrained weights, or credentials are exported.
Both ends verify the exact source bundle, completed runs and three QLoRA adapters.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
from types import SimpleNamespace
import zipfile

SOURCE_BUNDLE_SHA = '0e9d2453273883664c771ad9fe53b7a7cc3a357ffc6ac73ef2803d122c185c4e'
SOURCE_PREFIX = 'jev-tabular-benchmark/'
DATASETS = ('titanic', 'breast_cancer', 'wine')
MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
RESULTS = 'results/tabular/colab'
ADAPTER_ROOT = 'models/tabular/Qwen3-4B-Instruct-2507'
ADAPTER_FILES = {'adapter_config.json', 'adapter_model.safetensors', 'jevbench_training.json'}
RUN_FILES = {'run.json', 'predictions.jsonl', 'test_manifest.json'}
MAX_FILE = 512_000_000
MAX_TOTAL = 1_600_000_000
LORA_KEY = re.compile(r'base_model\.model\.model\.layers\.[0-9]+\.(?:self_attn|mlp)\.(?:q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)\.lora_([AB])(?:\.default)?\.weight')
SENSITIVE = re.compile(r'(?i)^(?:api[_-]?key|access[_-]?token|authorization|password|secret|credentials)$')
TOKEN = re.compile(rb'(?:\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|\bhf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n').encode()


def safe_name(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name and not p.is_absolute() and '\\' not in name and ':' not in name
            and p.parts and '..' not in p.parts and p.as_posix() == name and not name.endswith('/'), 'Unsafe archive member path')
    return name


def read_zip(path):
    payloads = {}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) <= 1000 and sum(i.file_size for i in entries) <= MAX_TOTAL, 'Archive exceeds transfer limits')
        for info in entries:
            name = safe_name(info.filename)
            require(name not in payloads, 'Duplicate archive member')
            mode = stat.S_IFMT(info.external_attr >> 16)
            require(mode in (0, stat.S_IFREG) and not info.is_dir(), 'Nonregular archive member')
            require(not info.flag_bits & 1 and info.file_size <= MAX_FILE, 'Encrypted or oversized archive member')
            payloads[name] = archive.read(info)
    return payloads


def local_bytes(root, name):
    safe_name(name)
    root = Path(root).resolve()
    path = root/name
    require(path.resolve().is_relative_to(root), 'Path escapes project')
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent), 'Symlink in project artifact path')
    require(path.is_file() and path.stat().st_size <= MAX_FILE, f'Missing or oversized regular artifact: {name}')
    return path.read_bytes()


def reject_secrets(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require(not (SENSITIVE.fullmatch(str(key)) and child not in (None, '', {}, [])), 'Sensitive field in transfer')
            reject_secrets(child)
    elif isinstance(value, list):
        for child in value:
            reject_secrets(child)


def check_text(name, content):
    if name.endswith('.safetensors'):
        return
    require(not TOKEN.search(content), 'Possible credential in transfer text')
    if name.endswith('.json'):
        reject_secrets(json.loads(content))
    elif name.endswith('.jsonl'):
        for line in content.splitlines():
            if line:
                reject_secrets(json.loads(line))


def verify_source(root, bundle):
    require(sha(Path(bundle).read_bytes()) == SOURCE_BUNDLE_SHA, 'Source ZIP differs from the frozen uploaded bundle')
    contents = read_zip(bundle)
    manifest = json.loads(contents.pop(SOURCE_PREFIX+'bundle_manifest.json'))
    expected = manifest['files_sha256']
    require(set(contents) == {SOURCE_PREFIX+n for n in expected}, 'Source bundle membership differs')
    for name, checksum in expected.items():
        require(sha(contents[SOURCE_PREFIX+name]) == checksum, 'Source bundle content mismatch')
        require(sha(local_bytes(root, name)) == checksum, f'Current frozen source/data differs: {name}')
    return manifest


def allowed_payload(name):
    safe_name(name)
    for dataset in DATASETS:
        if name == f'{RESULTS}/provenance/{dataset}_manifest.json':
            return True
        if name.startswith(f'{ADAPTER_ROOT}/{dataset}-k4-s42-qlora/'):
            return name.rsplit('/', 1)[-1] in ADAPTER_FILES and len(PurePosixPath(name).parts) == 5
    if name in {f'{RESULTS}/environment.json', f'{RESULTS}/provenance/source_bundle_manifest.json'}:
        return True
    parts = PurePosixPath(name).parts
    if len(parts) == 5 and '/'.join(parts[:3]) == RESULTS:
        if parts[3] == 'plans':
            return bool(re.fullmatch(r'[0-9a-f]{16}\.(?:json|models\.json|execution\.jsonl)', parts[4]))
        return bool(re.fullmatch(r'(?:titanic|breast_cancer|wine)__Qwen3-4B-Instruct-2507__[0-9a-f]{12}', parts[3])) and parts[4] in RUN_FILES
    return False


def validate_weights(content):
    require(8 < len(content) <= MAX_FILE, 'Invalid adapter weight size')
    length = struct.unpack('<Q', content[:8])[0]
    require(2 <= length <= min(len(content)-8, 8_000_000), 'Invalid safetensors header')
    header = json.loads(content[8:8+length])
    require(isinstance(header, dict) and header.get('__metadata__', {}) in ({}, {'format': 'pt'}), 'Unsupported tensor metadata')
    intervals = []
    for key, tensor in header.items():
        if key == '__metadata__':
            continue
        match = LORA_KEY.fullmatch(key)
        require(match and isinstance(tensor, dict), 'Pretrained/full-module tensors are forbidden')
        shape, offsets = tensor.get('shape'), tensor.get('data_offsets')
        width = {'F32': 4, 'F16': 2, 'BF16': 2}.get(tensor.get('dtype'))
        require(width and isinstance(shape, list) and len(shape) == 2 and all(type(x) is int and x > 0 for x in shape)
                and shape[0 if match.group(1) == 'A' else 1] == 8 and isinstance(offsets, list) and len(offsets) == 2
                and all(type(x) is int and x >= 0 for x in offsets) and offsets[1]-offsets[0] == math.prod(shape)*width,
                'Invalid rank/dtype/shape/offset in LoRA tensor')
        intervals.append(tuple(offsets))
    require(intervals, 'No LoRA tensors')
    end = 0
    for start, stop in sorted(intervals):
        require(start == end, 'Gapped/overlapping tensor storage')
        end = stop
    require(end == len(content)-8-length, 'Unexpected tensor data length')
    return len(intervals)


def close_values(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close_values(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close_values(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    return a == b


def validate_artifacts(root, payloads, source_manifest):
    """Recompute data/model/training and label/probability metrics before transfer."""
    from tabular_data import load_native_prepared
    from run_tabular_local import training_expectations
    from jevbench.runner import digest, environment
    from jevbench.metrics import evaluate
    from jevbench.prompts import build_prompt
    from jevbench.types import Prediction
    require(all(allowed_payload(name) for name in payloads), 'Non-allowlisted transfer payload')
    for name, content in payloads.items():
        check_text(name, content)
    core_sha = environment()['source_sha256']
    require(json.loads(payloads[f'{RESULTS}/provenance/source_bundle_manifest.json']) == source_manifest, 'Source manifest differs')
    env = json.loads(payloads[f'{RESULTS}/environment.json'])
    require(env.get('cuda_available') is True and env.get('torchao_version') is None, 'Expected CUDA environment with torchao absent')
    require(env.get('core_source_sha256') == core_sha and env.get('source_bundle_sha256') == SOURCE_BUNDLE_SHA, 'Environment provenance differs')
    records = sorted((name, json.loads(content)) for name, content in payloads.items() if name.endswith('/run.json'))
    require(len(records) == 9, 'Require all nine completed Qwen4B tabular evaluations')
    seen, summaries = set(), []
    for dataset_name in DATASETS:
        dataset, _ = load_native_prepared(Path(root)/'data/tabular-full'/dataset_name)
        require(payloads[f'{RESULTS}/provenance/{dataset_name}_manifest.json'] == local_bytes(root, f'data/tabular-full/{dataset_name}/manifest.json'), 'Prepared manifest differs')
        adapter_prefix = f'{ADAPTER_ROOT}/{dataset_name}-k4-s42-qlora'
        require(all(f'{adapter_prefix}/{name}' in payloads for name in ADAPTER_FILES), 'Incomplete adapter transfer')
        metadata = json.loads(payloads[f'{adapter_prefix}/jevbench_training.json'])
        expected = training_expectations(dataset, {'model': MODEL, 'revision': REVISION}, SimpleNamespace(device='cuda', load_in_4bit=True))
        require(all(metadata.get(k) == v for k, v in expected.items()), f'Adapter training provenance differs: {dataset_name}')
        config = json.loads(payloads[f'{adapter_prefix}/adapter_config.json'])
        expected_config = {'base_model_name_or_path': MODEL, 'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0, 'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'bias': 'none'}
        require(all(config.get(k) == v for k, v in expected_config.items()) and not config.get('modules_to_save') and not config.get('rank_pattern')
                and set(config.get('target_modules') or []) == {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj'}
                and config.get('use_dora', False) is False and config.get('use_rslora', False) is False, 'Adapter saves unexpected parameters')
        validate_weights(payloads[f'{adapter_prefix}/adapter_model.safetensors'])
        for name, record in records:
            if record.get('dataset') != dataset_name:
                continue
            key = (dataset_name, record.get('method'))
            require(key not in seen and key[1] in {'zero_shot', 'few_shot', 'lora'}, 'Duplicate/unsupported model condition')
            seen.add(key)
            require(record.get('status') == 'complete' and record.get('seed') == 42 and record.get('labels') == dataset.labels, 'Incomplete or changed run protocol')
            require(record.get('manifest_sha256') == digest(dataset.manifest) and record.get('dataset_manifest') == dataset.manifest, 'Run dataset content differs')
            require(record.get('prompt_template_sha256') == digest(build_prompt.__code__.co_consts.__repr__())
                    and record.get('test_ids_sha256') == digest([r.id for r in dataset.test]), 'Prompt/test identity hashes differ')
            require(record.get('implementation_sha256') == core_sha and record['environment']['source_sha256'] == core_sha
                    and all(s['environment']['source_sha256'] == core_sha for s in record.get('execution_sessions', [])), 'Inference core differs')
            cfg = record['config']; shots = 4 if key[1] == 'few_shot' else 0
            require(cfg.get('provider') == 'hf' and cfg.get('model') == MODEL and cfg.get('revision') == REVISION
                    and cfg.get('device') == 'cuda' and cfg.get('max_context_tokens') == 8192 and cfg.get('shots_per_class') == shots,
                    'Model/scoring configuration differs')
            train_ids = expected['training_row_ids'] if key[1] in {'few_shot', 'lora'} else []
            require(record.get('training_example_ids') == train_ids, 'Training IDs/ordering differ')
            folder = name.rsplit('/', 1)[0]
            require(folder.rsplit('/', 1)[-1] == record['run_id'], 'Run directory ID differs')
            if key[1] == 'lora':
                require(cfg.get('adapter_path', '').endswith('/'+adapter_prefix) and record.get('adapter_training') == metadata, 'Adapted model provenance differs')
                require(record.get('adapter_files_sha256') == {n: sha(payloads[f'{adapter_prefix}/{n}']) for n in ('adapter_config.json', 'adapter_model.safetensors')}, 'Adapter weights/config hash differs')
            else:
                require(not cfg.get('adapter_path'), 'Unexpected adapted base run')
            test_manifest = json.loads(payloads[f'{folder}/test_manifest.json'])
            expected_test = {'dataset': dataset.name, 'labels': dataset.labels, 'manifest_sha256': digest(dataset.manifest),
                             'rows': [{'id': r.id, 'label': r.label, 'text_sha256': sha(r.text.encode())} for r in dataset.test]}
            require(test_manifest == expected_test, 'Ordered heldout row identity/labels/text differ')
            predictions = [Prediction(**json.loads(line)) for line in payloads[f'{folder}/predictions.jsonl'].splitlines()]
            require([p.row_id for p in predictions] == [r.id for r in dataset.test], 'Missing, duplicate or reordered predictions')
            require(all(p.metadata.get('resolved_revision') == REVISION and p.metadata.get('probability_kind') == 'label_sequence_likelihood_normalized' for p in predictions), 'Prediction revision/probability protocol differs')
            actual = evaluate(dataset.test, predictions, len(dataset.labels))
            require(all(close_values(value, record['metrics'].get(k)) for k, value in actual.items()), 'Reported metrics differ from predictions')
            summaries.append({'dataset': dataset_name, 'method': key[1], 'run_id': record['run_id'], 'test_rows': len(dataset.test), 'n_failures': actual['n_failures'], 'accuracy': actual['accuracy'], 'macro_f1': actual['macro_f1']})
    require(seen == {(d, m) for d in DATASETS for m in ('zero_shot', 'few_shot', 'lora')}, 'Missing dataset/model condition')
    plans = [json.loads(c) for n, c in payloads.items() if re.fullmatch(re.escape(RESULTS)+r'/plans/[0-9a-f]{16}\.json', n)]
    require(plans and all(p['helper_sha256'] == source_manifest['files_sha256']['scripts/run_tabular_local.py'] and p['source_sha256'] == core_sha and p['model']['revision'] == REVISION for p in plans), 'Execution helper/plan provenance differs')
    return {'passed': True, 'source_bundle_sha256': SOURCE_BUNDLE_SHA, 'core_source_sha256': core_sha,
            'runs_verified': len(summaries), 'adapters_verified': 3, 'raw_rows_exported': False, 'runs': summaries}


def export_artifacts(root, source_bundle, output):
    root, output = Path(root).resolve(), Path(output)
    source = verify_source(root, source_bundle)
    payloads = {}
    for path in (root/RESULTS).rglob('*'):
        if path.is_file():
            name = path.relative_to(root).as_posix()
            require(allowed_payload(name), f'Unexpected result file: {name}')
            payloads[name] = local_bytes(root, name)
    for dataset in DATASETS:
        for filename in ADAPTER_FILES:
            name = f'{ADAPTER_ROOT}/{dataset}-k4-s42-qlora/{filename}'
            payloads[name] = local_bytes(root, name)
        payloads[f'{RESULTS}/provenance/{dataset}_manifest.json'] = local_bytes(root, f'data/tabular-full/{dataset}/manifest.json')
    payloads[f'{RESULTS}/provenance/source_bundle_manifest.json'] = json_bytes(source)
    audit = validate_artifacts(root, payloads, source)
    manifest = {'schema_version': 1, 'kind': 'private_tabular_colab_results_and_qlora_adapters',
                'transfer_helper_sha256': sha(Path(__file__).read_bytes()), 'source_bundle_sha256': SOURCE_BUNDLE_SHA,
                'files_sha256': {n: sha(c) for n, c in sorted(payloads.items())}, 'audit': audit,
                'contents': 'Completed predictions/metadata, environment, plans, source/dataset manifests, and only adapter config/LoRA weights/training metadata. Tokenizers and pretrained base weights are reloaded from the pinned base model.'}
    payloads['TRANSFER_MANIFEST.json'] = json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), 'Transfer output already exists; choose a new filename')
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.zip.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(payloads.items()):
                info = zipfile.ZipInfo(name); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(output.resolve()), 'sha256': sha(output.read_bytes()), 'bytes': output.stat().st_size, **audit}


def import_artifacts(root, source_bundle, archive, expected_sha):
    root = Path(root).resolve()
    require(sha(Path(archive).read_bytes()) == expected_sha, 'Downloaded transfer checksum differs')
    source = verify_source(root, source_bundle)
    payloads = read_zip(archive)
    manifest = json.loads(payloads.pop('TRANSFER_MANIFEST.json'))
    require(manifest.get('schema_version') == 1 and manifest.get('source_bundle_sha256') == SOURCE_BUNDLE_SHA
            and manifest.get('transfer_helper_sha256') == sha(Path(__file__).read_bytes()), 'Transfer provenance differs')
    require(set(payloads) == set(manifest['files_sha256']), 'Transfer archive membership differs from manifest')
    require(all(sha(content) == manifest['files_sha256'][name] for name, content in payloads.items()), 'Transfer payload hash mismatch')
    audit = validate_artifacts(root, payloads, source)
    require(audit == manifest['audit'], 'Export and import audits differ')
    payloads[f'{RESULTS}/provenance/transfer_manifest.json'] = json_bytes(manifest)
    report = {**audit, 'archive_sha256': expected_sha, 'original_run_records_modified': False,
              'files_sha256': {name: sha(content) for name, content in sorted(payloads.items())}}
    payloads['results/tabular/COLAB_IMPORT.json'] = json_bytes(report)
    # Complete collision/path preflight before creating any destination file.
    for name, content in payloads.items():
        destination = root/safe_name(name)
        require(not any(p.is_symlink() for p in [destination, *destination.parents]), 'Symlink at import destination')
        require(destination.resolve().is_relative_to(root), 'Import destination escapes project')
        if destination.exists():
            require(destination.is_file() and destination.read_bytes() == content, f'Refusing to overwrite mismatched destination: {name}')
    for name, content in payloads.items():
        destination = root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open('xb') as stream:
                stream.write(content)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'import'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('/content/jev-tabular-results.zip'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.action == 'export':
        result = export_artifacts(args.root, args.source_bundle, args.output)
    else:
        require(args.archive and args.sha256 and re.fullmatch('[0-9a-f]{64}', args.sha256), 'Import requires archive and externally recorded SHA-256')
        result = import_artifacts(args.root, args.source_bundle, args.archive, args.sha256)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
