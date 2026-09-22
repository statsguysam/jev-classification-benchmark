"""Path, integrity, no-overwrite, and semantic checks for private Colab transfer."""
import json
from pathlib import Path
import struct
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import tabular_colab_artifacts as transfer
from run_tabular_local import training_expectations
from tabular_data import load_native_prepared
from jevbench.metrics import evaluate
from jevbench.runner import _identity, digest, environment
from jevbench.types import Prediction, PreparedDataset, Row
from jevbench.data import save_prepared, load_prepared
import tabular_data
from dataclasses import asdict
from types import SimpleNamespace


def tiny_weights():
    header = {f'base_model.model.model.layers.0.self_attn.q_proj.lora_{letter}.weight':
              {'dtype': 'F32', 'shape': shape, 'data_offsets': offsets}
              for letter, shape, offsets in [('A', [8, 1], [0, 32]), ('B', [1, 8], [32, 64])]}
    raw = json.dumps(header).encode()
    return struct.pack('<Q', len(raw))+raw+b'\0'*64


def source_copy(path, bundle):
    path.mkdir()
    for name, content in transfer.read_zip(bundle).items():
        destination = path/name.removeprefix(transfer.SOURCE_PREFIX)
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(content)
    return path


@pytest.fixture
def completed(tmp_path, monkeypatch):
    # Standalone synthetic fixtures: no ignored local data/artifact or network dependency.
    source = tmp_path/'source'; source.mkdir()
    files = {}
    helper_name = 'scripts/run_tabular_local.py'
    target = source/helper_name; target.parent.mkdir(); target.write_bytes((ROOT/helper_name).read_bytes())
    files[helper_name] = target.read_bytes()
    for name in transfer.DATASETS:
        classes = 3 if name == 'wine' else 2
        rows = lambda split, n: [Row(f'{name}:{split}:{c}:{i}', f'{name}: {split}-{c}-{i}', c)
                                 for c in range(classes) for i in range(n)]
        dataset = PreparedDataset(name, [f'label {c}' for c in range(classes)], rows('train', 4), rows('validation', 1), rows('test', 2))
        folder = source/'data/tabular-full'/name; save_prepared(dataset, folder)
        for path in folder.iterdir():
            files[path.relative_to(source).as_posix()] = path.read_bytes()
    bundle = tmp_path/'source.zip'
    manifest = {'schema_version': 1, 'files_sha256': {n: transfer.sha(c) for n, c in files.items()}}
    with zipfile.ZipFile(bundle, 'w') as archive:
        for name, content in {**files, 'bundle_manifest.json': transfer.json_bytes(manifest)}.items():
            archive.writestr(transfer.SOURCE_PREFIX+name, content)
    monkeypatch.setattr(transfer, 'SOURCE_BUNDLE_SHA', transfer.sha(bundle.read_bytes()))
    monkeypatch.setattr(tabular_data, 'load_native_prepared', lambda p: (load_prepared(p), {}))
    source_manifest = transfer.verify_source(source, bundle)
    core = environment()['source_sha256']
    result_root = source/transfer.RESULTS; result_root.mkdir(parents=True)
    (result_root/'environment.json').write_bytes(transfer.json_bytes({'cuda_available': True, 'torchao_version': None,
             'core_source_sha256': core, 'source_bundle_sha256': transfer.SOURCE_BUNDLE_SHA}))
    (result_root/'plans').mkdir()
    (result_root/'plans/0000000000000000.json').write_bytes(transfer.json_bytes({'helper_sha256': source_manifest['files_sha256']['scripts/run_tabular_local.py'],
          'source_sha256': core, 'model': {'revision': transfer.REVISION}}))
    for name in transfer.DATASETS:
        ds = load_prepared(source/'data/tabular-full'/name)
        adapter = source/transfer.ADAPTER_ROOT/f'{name}-k4-s42-qlora'; adapter.mkdir(parents=True)
        metadata = training_expectations(ds, {'model': transfer.MODEL, 'revision': transfer.REVISION}, SimpleNamespace(device='cuda', load_in_4bit=True))
        config = {'base_model_name_or_path': transfer.MODEL, 'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.,
                  'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'bias': 'none',
                  'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj'], 'use_dora': False, 'use_rslora': False}
        (adapter/'jevbench_training.json').write_bytes(transfer.json_bytes(metadata))
        (adapter/'adapter_config.json').write_bytes(transfer.json_bytes(config))
        (adapter/'adapter_model.safetensors').write_bytes(tiny_weights())
        for method in ('zero_shot', 'few_shot', 'lora'):
            cfg = {'provider': 'hf', 'model': transfer.MODEL, 'revision': transfer.REVISION, 'device': 'cuda', 'max_context_tokens': 8192,
                   'shots_per_class': 4 if method == 'few_shot' else 0}
            if method == 'lora':
                cfg['adapter_path'] = str(adapter)
            record = _identity(ds, method, cfg, 42, [])
            record['training_example_ids'] = metadata['training_row_ids'] if method != 'zero_shot' else []
            if method == 'lora':
                record.update(adapter_training=metadata, adapter_files_sha256={n: transfer.sha((adapter/n).read_bytes()) for n in ('adapter_config.json', 'adapter_model.safetensors')})
            run_id = f'{name}__Qwen3-4B-Instruct-2507__{digest(record)[:12]}'
            folder = result_root/run_id; folder.mkdir()
            predictions = [Prediction(r.id, r.label, [0.9 if i == r.label else 0.1/(len(ds.labels)-1) for i in range(len(ds.labels))],
                     latency_s=.01, metadata={'resolved_revision': transfer.REVISION, 'probability_kind': 'label_sequence_likelihood_normalized'}) for r in ds.test]
            record.update(run_id=run_id, status='complete', environment=environment(), dataset_manifest=ds.manifest,
                          metrics=evaluate(ds.test, predictions, len(ds.labels)))
            (folder/'run.json').write_bytes(transfer.json_bytes(record))
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(asdict(p))+'\n' for p in predictions))
            test = {'dataset': name, 'labels': ds.labels, 'manifest_sha256': digest(ds.manifest),
                    'rows': [{'id': r.id, 'label': r.label, 'text_sha256': transfer.sha(r.text.encode())} for r in ds.test]}
            (folder/'test_manifest.json').write_bytes(transfer.json_bytes(test))
    return source, bundle, tmp_path


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'x/../y', 'x\\y', 'x//y', './x', '.', 'x:stream'])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError, match='Unsafe'):
        transfer.safe_name(name)


def test_symlinks_duplicate_members_and_unknown_members_rejected(tmp_path):
    p = tmp_path/'bad.zip'
    with zipfile.ZipFile(p, 'w') as z:
        info = zipfile.ZipInfo('payload'); info.external_attr = 0o120777 << 16; z.writestr(info, 'outside')
    with pytest.raises(ValueError, match='Nonregular'):
        transfer.read_zip(p)
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('same', 'a')
        with pytest.warns(UserWarning):
            z.writestr('same', 'b')
    with pytest.raises(ValueError, match='Duplicate'):
        transfer.read_zip(p)
    assert not transfer.allowed_payload('data/tabular-full/titanic/test.jsonl')
    assert not transfer.allowed_payload(f'{transfer.ADAPTER_ROOT}/titanic-k4-s42-qlora/model.safetensors')


def test_base_weights_cannot_masquerade_as_adapter():
    raw = json.dumps({'model.embed_tokens.weight': {'dtype': 'F32', 'shape': [8, 1], 'data_offsets': [0, 32]}}).encode()
    with pytest.raises(ValueError, match='Pretrained'):
        transfer.validate_weights(struct.pack('<Q', len(raw))+raw+b'\0'*32)
    assert transfer.validate_weights(tiny_weights()) == 2


def test_full_semantic_roundtrip_is_idempotent_and_contains_no_raw_data(completed):
    source, bundle, tmp = completed
    report = transfer.export_artifacts(source, bundle, tmp/'results.zip')
    assert report['runs_verified'] == 9 and report['adapters_verified'] == 3
    payloads = transfer.read_zip(tmp/'results.zip')
    assert not any(n.startswith('data/') or n.endswith('/tokenizer.json') for n in payloads)
    destination = source_copy(tmp/'destination', bundle)
    audit = transfer.import_artifacts(destination, bundle, tmp/'results.zip', report['sha256'])
    assert audit['passed'] and len(audit['runs']) == 9
    assert transfer.import_artifacts(destination, bundle, tmp/'results.zip', report['sha256']) == audit
    for name in payloads:
        if name != 'TRANSFER_MANIFEST.json':
            assert (destination/name).read_bytes() == payloads[name]


def test_mismatched_destination_preflight_leaves_other_outputs_absent(completed):
    source, bundle, tmp = completed
    report = transfer.export_artifacts(source, bundle, tmp/'results.zip')
    destination = source_copy(tmp/'destination', bundle)
    conflict = destination/transfer.RESULTS/'environment.json'; conflict.parent.mkdir(parents=True); conflict.write_text('keep')
    with pytest.raises(ValueError, match='overwrite mismatched'):
        transfer.import_artifacts(destination, bundle, tmp/'results.zip', report['sha256'])
    assert conflict.read_text() == 'keep' and not (destination/transfer.ADAPTER_ROOT).exists()


def test_tampered_prediction_metrics_refuse_export(completed):
    source, bundle, tmp = completed
    path = next((source/transfer.RESULTS).glob('*/run.json'))
    row = json.loads(path.read_text()); row['metrics']['accuracy'] = .123; path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='metrics differ'):
        transfer.export_artifacts(source, bundle, tmp/'bad.zip')
    assert not (tmp/'bad.zip').exists()


def test_changed_frozen_source_stops_export(completed):
    source, bundle, tmp = completed
    (source/'scripts/run_tabular_local.py').write_text('changed')
    with pytest.raises(ValueError, match='frozen source/data differs'):
        transfer.export_artifacts(source, bundle, tmp/'bad.zip')


def test_external_checksum_required_before_import(completed):
    source, bundle, tmp = completed
    transfer.export_artifacts(source, bundle, tmp/'results.zip')
    destination = source_copy(tmp/'destination', bundle)
    with pytest.raises(ValueError, match='checksum differs'):
        transfer.import_artifacts(destination, bundle, tmp/'results.zip', '0'*64)
    assert not (destination/transfer.RESULTS).exists()
