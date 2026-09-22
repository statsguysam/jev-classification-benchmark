#!/usr/bin/env python3
"""Plan/run pinned SmolLM2 and Granite sources on the immutable SST2/TREC pilot.

No LoRA, hosted calls, new dataset preparation or prompt tuning. Execution reuses
FixedChatClassifier, preflight, rendering and likelihood scoring unchanged from
the hashed numerical helper. The wrapper has its own run-identity source hash.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_expanded_numeric_local as frozen
from jevbench.data import load_prepared, validate_prepared
from jevbench.metrics import evaluate
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import _identity, digest, environment, run_model, save_json
from jevbench.types import Prediction

MODEL_KEYS = frozen.MODEL_KEYS
DATASETS = ('sst2', 'trec')
FROZEN_HELPER_SHA = 'b0c07e82e03b79040f13bd7810baf92bcd3a30fe8aa3b48d0704baedea8801a3'
FROZEN_PRESETS_SHA = 'ed176a77f40fd8cd842672193333559a8221b3ac9dd52259d275c54da4cf6421'
WRAPPER_VERSION = 'frozen-text-source-v1'
DATASET_PINS = {
    'sst2': {'manifest_sha256': '8c12945f6b673f72452096793b4777a94f2455ea55c51f4dc8c9dba9bb8a8201',
             'manifest_file_sha256': '9586c538f8e3a5d51720dd9f0923848b0dec2019e47c13986e67acc599ca0d1f',
             'n_test': 200, 'n_classes': 2},
    'trec': {'manifest_sha256': 'd44f342fb000a58a037309a667697f2cf9c3720d52437298b5c50cbc239e333a',
             'manifest_file_sha256': '88322777fa680f97845d10570dfe7814fe0b6cc1c56741caa936e29c0d550a2b',
             'n_test': 200, 'n_classes': 6},
}
require, file_sha, text_sha = frozen.require, frozen.file_sha, frozen.text_sha


def load_presets():
    require(file_sha(frozen.__file__) == FROZEN_HELPER_SHA and file_sha(frozen.PRESETS) == FROZEN_PRESETS_SHA,
            'Frozen renderer/scorer helper or model presets differ')
    return frozen.load_presets()


def validate_dataset(dataset):
    require(dataset.name in DATASET_PINS, 'Only the frozen SST2 and TREC pilot datasets are allowed')
    validate_prepared(dataset)
    expected = DATASET_PINS[dataset.name]
    require(digest(dataset.manifest) == expected['manifest_sha256'] and len(dataset.test) == expected['n_test']
            and len(dataset.labels) == expected['n_classes'], 'Prepared text dataset differs from frozen pilot')
    require(digest({split: [asdict(row) for row in getattr(dataset, split)] for split in ('train', 'validation', 'test')})
            == dataset.manifest['prepared_content_sha256'], 'Frozen prepared row contents differ')


def load_dataset(path):
    path = Path(path)
    dataset = load_prepared(path)
    validate_dataset(dataset)
    require(file_sha(path / 'manifest.json') == DATASET_PINS[dataset.name]['manifest_file_sha256'],
            'Prepared manifest bytes differ')
    return dataset


def model_config(key, device, presets=None):
    data = presets or load_presets()
    return {**frozen.model_config(key, device, data), 'text_extension': {
        'version': WRAPPER_VERSION, 'wrapper_sha256': file_sha(__file__),
        'numeric_helper_sha256': FROZEN_HELPER_SHA, 'preset_file_sha256': FROZEN_PRESETS_SHA,
        'core_sha256': frozen.FROZEN_CORE_SHA, 'dataset_pins': DATASET_PINS}}


def audit_source_run(path, dataset, expected_key, shots):
    """Read-only audit of a completed source; never load a model or use network."""
    path = Path(path)
    validate_dataset(dataset)
    data = load_presets()
    require(expected_key in MODEL_KEYS and type(shots) is int and shots in (0, 4), 'Unexpected source condition')
    record = json.loads(path.read_text())
    config = model_config(expected_key, record['config']['device'], data)
    examples = select_examples(dataset.train, dataset.labels, shots, 42)
    identity = _identity(dataset, 'zero_shot' if shots == 0 else 'few_shot', {**config, 'shots_per_class': shots}, 42, examples)
    identity['bootstrap_samples'] = record['bootstrap_samples']
    require(type(record['bootstrap_samples']) is int and record['bootstrap_samples'] >= 100
            and all(record.get(k) == v for k, v in identity.items()), 'Text source identity differs')
    expected_id = f"{dataset.name}__{config['model'].split('/')[-1]}__{digest(identity)[:12]}"
    require(record['status'] == 'complete' and record['run_id'] == path.parent.name == expected_id,
            'Incomplete or unexpected text source run')
    require(record['dataset_manifest'] == dataset.manifest and record['environment']['source_sha256'] == frozen.FROZEN_CORE_SHA,
            'Prepared/core provenance differs')
    expected_manifest = {'dataset': dataset.name, 'labels': dataset.labels, 'manifest_sha256': digest(dataset.manifest),
        'rows': [{'id': row.id, 'label': row.label, 'text_sha256': text_sha(row.text)} for row in dataset.test]}
    require(json.loads(path.with_name('test_manifest.json').read_text()) == expected_manifest, 'Source test manifest differs')
    raw = path.with_name('predictions.jsonl').read_text()
    require(raw.endswith('\n'), 'Completed source contains an incomplete prediction write')
    predictions = [Prediction(**json.loads(line)) for line in raw.splitlines()]
    require([p.row_id for p in predictions] == [row.id for row in dataset.test], 'Source prediction coverage/order differs')
    preset = data['models'][expected_key]
    for row, prediction in zip(dataset.test, predictions):
        expected = {'requested_model': config['model'], 'requested_revision': config['revision'],
            'resolved_revision': config['revision'], 'device': config['device'], 'dtype': 'float16',
            'probability_kind': 'label_sequence_likelihood_normalized', 'scoring': 'sum_logp_numeric_id_plus_eos',
            'renderer_version': frozen.RENDERER_VERSION, 'chat_template_sha256': text_sha(preset['chat_template']),
            'tokenizer_fingerprint_sha256': digest(preset['tokenizer_files_sha256']),
            'rendered_chat_prompt_sha256': text_sha(frozen.render_chat(preset, build_prompt(row, dataset.labels, examples)))}
        require(all(prediction.metadata.get(k) == v for k, v in expected.items()), 'Source renderer/model provenance differs')
        require(re.fullmatch('[0-9a-f]{64}', prediction.metadata.get('rendered_input_ids_sha256', ''))
                and type(prediction.metadata.get('rendered_input_tokens')) is int
                and 0 < prediction.metadata['rendered_input_tokens'] < 8192, 'Invalid token identity/context length')
        if prediction.error is None:
            require(type(prediction.label) is int and 0 <= prediction.label < len(dataset.labels)
                and prediction.probabilities is not None and len(prediction.probabilities) == len(dataset.labels)
                and prediction.label == max(range(len(dataset.labels)), key=prediction.probabilities.__getitem__)
                and prediction.input_tokens == prediction.metadata['rendered_input_tokens'] and prediction.output_tokens == 0
                and prediction.metadata['context_forward_passes'] == len(dataset.labels), 'Source scoring contract differs')
        else:
            require(isinstance(prediction.error, str) and prediction.error and prediction.label is None
                    and prediction.probabilities is None, 'Malformed source failure')
    measured = evaluate(dataset.test, predictions, len(dataset.labels))
    require(all(record['metrics'].get(k) == v for k, v in measured.items()), 'Saved source metrics differ from predictions')
    return record, predictions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cuda', 'mps'), default='cuda')
    p.add_argument('--model-keys', nargs='+', choices=MODEL_KEYS, default=list(MODEL_KEYS))
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--shots', nargs='+', type=int, choices=(0, 4), default=[0, 4])
    p.add_argument('--data-root', type=Path, default=ROOT / 'data/pilot')
    p.add_argument('--output', type=Path, default=ROOT / 'results/text_extension/local')
    p.add_argument('--cache-dir', type=Path, default=ROOT / 'data/hf')
    p.add_argument('--allow-download', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--bootstrap-samples', type=int, default=1000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.bootstrap_samples >= 100, 'At least 100 bootstrap samples required')
    require(all(len(x) == len(set(x)) for x in (args.model_keys, args.datasets, args.shots)), 'Duplicate condition arguments')
    presets = load_presets()
    jobs = []
    for name in args.datasets:
        dataset = load_dataset(args.data_root / name)
        jobs.extend((dataset, shots) for shots in args.shots)
    plan = {'status': 'plan', 'new_hosted_calls': 0, 'adapter_training': False, 'device': args.device, 'dtype': 'float16',
        'network_downloads_authorized': args.allow_download,
        'models': {key: model_config(key, args.device, presets) for key in args.model_keys},
        'jobs': [{'dataset': dataset.name, 'shots_per_class': shots, 'n_test': len(dataset.test),
            'manifest_sha256': digest(dataset.manifest), 'training_example_ids':
            [row.id for row in select_examples(dataset.train, dataset.labels, shots, 42)]} for dataset, shots in jobs],
        'prediction_rows': sum(len(dataset.test) for dataset, _ in jobs)*len(args.model_keys)}
    print(json.dumps(plan, indent=2), flush=True)
    if args.allow_download:
        for key in args.model_keys: frozen.download_public(presets['models'][key], args.cache_dir)
    if not args.execute and not args.preflight_only: return 0
    import torch
    require((args.device == 'cuda' and torch.cuda.is_available()) or
            (args.device == 'mps' and torch.backends.mps.is_available()), 'Requested device unavailable; no fallback')
    with frozen.exclusive_device_lock(ROOT / 'artifacts/tabular-neural.lock'):
        for key in args.model_keys:
            config = plan['models'][key]
            provider = frozen.FixedChatClassifier(config, presets['models'][key], args.cache_dir)
            try:
                checks = frozen.preflight(provider, jobs)
                save_json(args.output / 'plans' / f'{key}-{args.device}-preflight.json',
                          {**plan, 'status': 'preflight_complete', 'checks': checks})
                print(json.dumps({'model_key': key, 'preflight': checks}), flush=True)
                if args.execute:
                    for dataset, shots in jobs:
                        with patch.object(frozen.providers, 'build_provider', return_value=provider):
                            record = run_model(dataset, config, args.output, shots=shots, seed=42,
                                allow_paid=False, max_requests=len(dataset.test), bootstrap_samples=args.bootstrap_samples)
                        audit_source_run(args.output / record['run_id'] / 'run.json', dataset, key, shots)
                        print(json.dumps({'run_id': record['run_id'], 'accuracy': record['metrics']['accuracy'],
                                          'n_failures': record['metrics']['n_failures']}), flush=True)
            finally:
                provider._model = provider._tokenizer = provider._torch = None
                del provider
                gc.collect()
                if args.device == 'cuda': torch.cuda.empty_cache()
                else: torch.mps.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
