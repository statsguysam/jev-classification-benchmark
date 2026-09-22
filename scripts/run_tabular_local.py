"""Plan or sequentially run frozen HF classification/LoRA on prepared tabular rows.

No data preparation, prompt changes, hosted calls, or implicit downloads occur here.
Default invocation prints a plan; --phase base/train/eval/all executes that phase.
Each model process exits before the next begins. A project lock serializes helpers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from jevbench.data import load_prepared
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import digest, environment, save_json

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('titanic', 'breast_cancer', 'wine')
PHASES = {'base': ('zero', 'few'), 'train': ('train',), 'eval': ('adapter',),
          'all': ('zero', 'few', 'train', 'adapter'), 'plan': ('zero', 'few', 'train', 'adapter')}
RECIPE = {'seed': 42, 'train_per_class': 4, 'epochs': 3, 'max_length': 2048,
          'learning_rate': 2e-4, 'batch_size': 1, 'gradient_accumulation_steps': 8,
          'r': 8, 'lora_alpha': 16, 'target_modules': 'all-linear',
          'gradient_checkpointing': True, 'max_steps': None}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=PHASES, default='plan')
    p.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=DATASETS)
    p.add_argument('--data-root', type=Path, default=ROOT/'data/tabular-full')
    p.add_argument('--config', type=Path, default=ROOT/'configs/models.json')
    p.add_argument('--model-key', default='qwen_small')
    p.add_argument('--output', type=Path, default=ROOT/'results/tabular/local')
    p.add_argument('--adapter-root', type=Path, default=ROOT/'models/tabular')
    p.add_argument('--device', choices=['mps', 'cuda', 'cpu'], default='mps')
    p.add_argument('--load-in-4bit', action='store_true', help='CUDA training only; evaluation uses normal base precision')
    p.add_argument('--allow-download', action='store_true', help='Permit public HF downloads during execution; default is offline cache only')
    p.add_argument('--bootstrap-samples', type=int, default=1000)
    return p


def training_expectations(dataset, config, args):
    examples = select_examples(dataset.train, dataset.labels, 4, 42)
    prompt_hash = hashlib.sha256(json.dumps([build_prompt(r, dataset.labels) for r in examples],
                                          ensure_ascii=False).encode()).hexdigest()
    return {**RECIPE, 'model_id': config['model'], 'requested_revision': config['revision'],
            'resolved_revision': config['revision'], 'dataset': dataset.name,
            'labels': dataset.labels, 'training_rows': len(examples),
            'training_row_ids': [r.id for r in examples], 'validation_rows': 0,
            'test_accessed': False, 'selection': 'fixed_final_checkpoint',
            'supervision': 'numeric_class_id_plus_eos_response_only',
            'prompt_sha256': prompt_hash, 'load_in_4bit': args.load_in_4bit,
            'device': args.device}


def validate_adapter(path, expected):
    """Reject partial or differently trained checkpoints instead of overwriting them."""
    path = Path(path)
    required = ('jevbench_training.json', 'adapter_config.json', 'adapter_model.safetensors')
    if not all((path/name).is_file() and (path/name).stat().st_size for name in required):
        raise ValueError(f'Incomplete adapter at {path}; use a new output directory after inspection')
    metadata = json.loads((path/'jevbench_training.json').read_text())
    differences = [key for key, value in expected.items() if metadata.get(key) != value]
    if differences:
        raise ValueError(f'Adapter provenance mismatch at {path}: {", ".join(differences)}')
    config = json.loads((path/'adapter_config.json').read_text())
    if any(config.get(k) != v for k, v in {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0,
                                         'base_model_name_or_path': expected['model_id']}.items()):
        raise ValueError(f'Adapter config differs from fixed recipe at {path}')
    return metadata


def make_plan(args):
    if args.load_in_4bit and args.device != 'cuda':
        raise ValueError('4-bit training requires CUDA; use ordinary LoRA on MPS/CPU')
    if args.bootstrap_samples < 1:
        raise ValueError('bootstrap-samples must be positive')
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError('Duplicate datasets are not permitted')
    original = json.loads(args.config.read_text())[args.model_key]
    if original.get('provider') not in {'hf', 'huggingface'} or original.get('adapter_path'):
        raise ValueError('Select an unadapted Hugging Face model; hosted providers are forbidden')
    if not re.fullmatch('[0-9a-f]{40}', original.get('revision', '')):
        raise ValueError('A complete immutable 40-character model revision is required')
    allowed = {'provider', 'model', 'revision', 'device', 'max_context_tokens', 'dtype'}
    if set(original) - allowed:
        raise ValueError('Unexpected model-config fields; do not place credentials in this helper')
    model = {**original, 'device': args.device}
    output = args.output.resolve()
    datasets, summaries, adapters = {}, [], {}
    for name in args.datasets:
        path = (args.data_root/name).resolve()
        dataset = load_prepared(path)
        if dataset.name != name:
            raise ValueError(f'Prepared dataset name differs from directory: {name}')
        datasets[name] = dataset
        expected = training_expectations(dataset, model, args)
        adapter = (args.adapter_root / model['model'].split('/')[-1] /
                   f'{name}-k4-s42-{"qlora" if args.load_in_4bit else "lora"}').resolve()
        adapters[name] = adapter
        summaries.append({'dataset': name, 'data_path': str(path), 'manifest_sha256': digest(dataset.manifest),
                          'split_files_sha256': dataset.manifest['files_sha256'],
                          'test_rows': len(dataset.test), 'labels': dataset.labels,
                          'matched_training_row_ids': expected['training_row_ids'],
                          'training_labels': expected['training_rows'], 'validation_labels_used': 0,
                          'adapter_path': str(adapter), 'training_expectations': expected})
    plan_identity = {'datasets': summaries, 'model': model, 'recipe': RECIPE,
                     'load_in_4bit': args.load_in_4bit, 'bootstrap_samples': args.bootstrap_samples,
                     'output': str(output), 'source_sha256': environment()['source_sha256'],
                     'helper_sha256': sha_file(__file__)}
    plan_id = digest(plan_identity)[:16]
    config_path = output/'plans'/f'{plan_id}.models.json'
    generated_configs = {'base': model, **{f'adapter_{name}': {**model, 'adapter_path': str(path)}
                                         for name, path in adapters.items()}}
    jobs = []
    for item in summaries:
        common = ['--data', item['data_path'], '--seed', '42']
        for phase in PHASES[args.phase]:
            cmd = [sys.executable, '-m', 'jevbench.cli']
            if phase == 'train':
                cmd += ['train-lora', *common, '--model', model['model'], '--revision', model['revision'],
                        '--output', item['adapter_path'], '--train-per-class', '4', '--epochs', '3',
                        '--max-length', '2048', '--device', args.device]
                if args.load_in_4bit:
                    cmd.append('--load-in-4bit')
            else:
                cmd += ['model', *common, '--config', str(config_path), '--model-key',
                        f'adapter_{item["dataset"]}' if phase == 'adapter' else 'base',
                        '--shots', '4' if phase == 'few' else '0', '--output', str(output),
                        '--max-requests', str(item['test_rows']), '--bootstrap-samples', str(args.bootstrap_samples)]
            jobs.append({'dataset': item['dataset'], 'phase': phase, 'command': cmd,
                         'shell_command': shlex.join(cmd)})
    return {'plan_id': plan_id, **plan_identity, 'phase': args.phase, 'jobs': jobs,
            'generated_config_path': str(config_path), 'generated_configs': generated_configs,
            'evaluation_decisions_upper_bound': sum(next(x['test_rows'] for x in summaries if x['dataset'] == j['dataset'])
                                                   for j in jobs if j['phase'] != 'train'),
            'execution_policy': {'hosted_calls': False, 'network_downloads': args.allow_download,
                                 'serial_subprocesses': True, 'adapter_reuse_requires_exact_provenance': True,
                                 'no_automatic_retries': True,
                                 'prompt': 'unchanged frozen build_prompt; named feature serialization is row text'}}


@contextmanager
def exclusive_device_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another tabular local helper holds the device lock; no process started') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def execute_plan(plan, args):
    if args.phase == 'plan':
        raise ValueError('Dry-run phase must not execute')
    # One lock for all helper invocations, even if their output folders differ.
    with exclusive_device_lock(ROOT/'artifacts/tabular-neural.lock'):
        import torch
        if args.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable in this process; no CPU fallback. Run in the approved local environment.')
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no device fallback')
        config_path = Path(plan['generated_config_path'])
        save_json(config_path, plan['generated_configs'])
        save_json(config_path.with_name(plan['plan_id']+'.json'), plan)
        events_path = config_path.with_name(plan['plan_id']+'.execution.jsonl')
        env = dict(os.environ)
        env.setdefault('HF_HOME', str(ROOT/'data/hf'))
        if not args.allow_download:
            env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        else:
            env.pop('HF_HUB_OFFLINE', None)
            env.pop('TRANSFORMERS_OFFLINE', None)
        for job in plan['jobs']:
            item = next(x for x in plan['datasets'] if x['dataset'] == job['dataset'])
            # Stop if prepared data/source changed after the reviewed plan.
            current = load_prepared(Path(item['data_path']))
            if digest(current.manifest) != item['manifest_sha256'] or environment()['source_sha256'] != plan['source_sha256']:
                raise ValueError('Prepared data or frozen core changed after planning')
            adapter = Path(item['adapter_path'])
            status = 'execute'
            if job['phase'] == 'train' and adapter.exists() and any(adapter.iterdir()):
                validate_adapter(adapter, item['training_expectations'])
                status = 'reused_verified_adapter'
            if job['phase'] == 'adapter':
                validate_adapter(adapter, item['training_expectations'])
            print(json.dumps({'dataset': job['dataset'], 'phase': job['phase'], 'status': status}), flush=True)
            event = {'time_utc': datetime.now(timezone.utc).isoformat(), 'dataset': job['dataset'],
                     'phase': job['phase'], 'status': status}
            if status == 'execute':
                result = subprocess.run(job['command'], cwd=ROOT, env=env, check=False)
                event.update({'returncode': result.returncode, 'status': 'complete' if result.returncode == 0 else 'failed'})
            with events_path.open('a') as stream:
                stream.write(json.dumps(event)+'\n')
            if event.get('returncode', 0) != 0:
                raise RuntimeError(f'Phase {job["phase"]} failed for {job["dataset"]}; stopped without retry')


def main(argv=None):
    args = parser().parse_args(argv)
    plan = make_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.phase != 'plan':
        execute_plan(plan, args)


if __name__ == '__main__':
    main()
