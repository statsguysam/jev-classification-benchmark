#!/usr/bin/env python3
"""Audit the tabular matrix and report group-bootstrap uncertainty without paid calls.

Run artifacts stay unchanged. Only complete, fully aligned predictions are
scored. Exact prepared manifests must match in every execution environment.
"""
from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from tabular_data import load_native_prepared
from jevbench.metrics import evaluate
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import digest, environment
from jevbench.types import Prediction

DATASETS = ('titanic', 'breast_cancer', 'wine')
CLASSICAL = ('majority', 'logistic_regression', 'rbf_svc', 'random_forest', 'hist_gradient_boosting')
OPEN_MODELS = ('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen3-4B-Instruct-2507')
HOSTED = ('typesafe/jev-1.13', 'gpt-5.6-luna', 'gpt-6-astra')
DISPLAY = {'majority': 'Majority', 'logistic_regression': 'Logistic regression',
    'rbf_svc': 'RBF SVM', 'random_forest': 'Random forest', 'hist_gradient_boosting': 'Hist. gradient boosting',
    OPEN_MODELS[0]: 'Qwen2.5 0.5B', OPEN_MODELS[1]: 'Qwen3 4B',
    HOSTED[0]: 'Jev 1.13', HOSTED[1]: 'GPT-5.6 Luna', HOSTED[2]: 'GPT-6 Astra'}
FROZEN_CORE_SHA = 'd5547ccde4224e182653315d81c0a631c789bbe294ad7bcf95ef0cddd25ce608'
PROBABILITY_FIELDS = ('probability_coverage', 'n_probability_rows', 'log_loss', 'brier_sum', 'ece_15_equal_width')
APPROVED_HOSTED_SOURCE_HASHES = {
    'run_tabular_hosted.py': '53936be5c051b80cbee74943ec6544f29bc28044d99f5fd20226560f7163b230',
    'run_openrouter_jev.py': '12a4e66649a2f9533ba8bb16fd7bd386a0c9dfc808f400b4da2a6df26e5ce89f',
    'run_budgeted_hosted.py': '3a6447218cba5025fef3aa58eb56d9045bd89173a35c4150544a87e7a55b6f92',
}
APPROVED_PRICE_HASHES = {
    'gpt-5.6-luna': 'b51594ac146d6546ad57ab770d50f45a97f3c96a585b0400302a5a3b487190a9',
    'gpt-6-astra': '27b33bb81f80a13854f1b0681e9665658b9a47dc5dcb0f1a54a82e9debf5e3e2',
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def expected_conditions():
    conditions = []
    for dataset in DATASETS:
        conditions.extend((dataset, model, 'classical_tabular', budget) for model in CLASSICAL for budget in (4, None))
        conditions.extend((dataset, model, method, 0 if method == 'zero_shot' else 4)
                          for model in OPEN_MODELS for method in ('zero_shot', 'few_shot', 'lora'))
        conditions.extend((dataset, model, method, 0 if method == 'zero_shot' else 4)
                          for model in HOSTED for method in ('zero_shot', 'few_shot'))
    return conditions


def condition(record):
    method, config = record['method'], record['config']
    budget = (config.get('train_per_class') if method == 'classical_tabular' else
              record.get('adapter_training', {}).get('train_per_class') if method == 'lora' else
              config.get('shots_per_class', 0))
    return record['dataset'], config['model'], method, budget


def _confusions(y, predictions, inverse, n_groups, n_classes):
    result = np.zeros((n_groups, n_classes, n_classes + 1), dtype=np.int64)
    predictions = np.where((predictions >= 0) & (predictions < n_classes), predictions, n_classes)
    np.add.at(result, (inverse, y, predictions), 1)
    return result


def _scores_from_confusion(matrix):
    # Last column is failure: it contributes false negatives but no valid-class TP/FP.
    n_classes = matrix.shape[-2]
    true = matrix.sum(axis=-1)
    predicted = matrix[..., :n_classes].sum(axis=-2)
    tp = np.diagonal(matrix[..., :n_classes], axis1=-2, axis2=-1)
    denominator = true + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0).mean(axis=-1)
    accuracy = tp.sum(axis=-1) / matrix.sum(axis=(-2, -1))
    return accuracy, f1


def group_bootstrap(y, a, groups, b=None, *, n_classes, samples=2000, seed=42):
    """Unstratified percentile bootstrap over whole groups; paired result is A-B.

    Each replicate samples G groups uniformly with replacement. All observations
    of a drawn group are included, including conflicting labels and failures.
    Thus replicate row counts can vary while the draw count always equals G.
    """
    y, a, groups = np.asarray(y), np.asarray(a), np.asarray(groups)
    b = None if b is None else np.asarray(b)
    if (samples < 100 or n_classes < 2 or y.ndim != 1 or len(y) == 0
            or y.shape != a.shape or y.shape != groups.shape or (b is not None and b.shape != y.shape)):
        raise ValueError('Invalid group-bootstrap dimensions or settings')
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(a.dtype, np.integer) or (b is not None and not np.issubdtype(b.dtype, np.integer)):
        raise ValueError('Bootstrap labels must be integers')
    if np.any((y < 0) | (y >= n_classes)) or any(not isinstance(group, str) or not group for group in groups.tolist()):
        raise ValueError('Invalid truth label or group identifier')
    unique, inverse = np.unique(groups, return_inverse=True)
    ngroups = len(unique)
    ca = _confusions(y, a, inverse, ngroups, n_classes)
    cb = None if b is None else _confusions(y, b, inverse, ngroups, n_classes)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, ngroups, size=(samples, ngroups))
    resampled_a = ca[draws].sum(axis=1)
    scores_a = _scores_from_confusion(resampled_a)
    scores_b = (0, 0) if cb is None else _scores_from_confusion(cb[draws].sum(axis=1))
    point_a = _scores_from_confusion(ca.sum(axis=0))
    point_b = (0, 0) if cb is None else _scores_from_confusion(cb.sum(axis=0))
    counts = resampled_a.sum(axis=(1, 2))
    return {'method': 'paired unstratified group percentile bootstrap' if b is not None else 'unstratified group percentile bootstrap',
            'group_definition': 'identical serialized test text SHA256', 'samples': samples, 'seed': seed,
            'n_rows': len(y), 'n_groups': ngroups, 'draws_per_replicate': ngroups,
            'resampled_row_count_range': [int(counts.min()), int(counts.max())],
            'conditional_on': 'fixed split, fixed prompt/serialization, fixed training selection and fitted models',
            'metrics': {name: {'estimate': float(pa - pb), 'ci95': np.quantile(sa - sb, [.025, .975]).tolist()}
                        for name, pa, pb, sa, sb in zip(('accuracy', 'macro_f1'), point_a, point_b, scores_a, scores_b)}}


def _expected_ids(dataset, budget, seed=42):
    return [row.id for row in (dataset.train if budget is None else select_examples(dataset.train, dataset.labels, budget, seed))]


def literal_source_constant(path, name):
    # Read immutable literal declarations, never import or execute request code.
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'Missing source declaration: {name}')


def canonical_tabular_ledger_header():
    path = ROOT / 'results/tabular/api-budget.jsonl'
    with path.with_name(path.name + '.lock').open() as anchor:
        fcntl.flock(anchor.fileno(), fcntl.LOCK_SH)
        try:
            with path.open() as stream:
                header = json.loads(stream.readline())
            lock = json.load(anchor)
        finally:
            fcntl.flock(anchor.fileno(), fcntl.LOCK_UN)
    if (header.get('type') != 'header' or header.get('schema') != 1 or header.get('budget_nano') != 14700000000
            or not header.get('ledger_id') or lock.get('ledger_id') != header['ledger_id']
            or lock.get('budget_nano') != header['budget_nano']):
        raise ValueError('Canonical tabular ledger identity/cap mismatch')
    return header


def expected_hosted_guard(record, model_key):
    for name, expected in APPROVED_HOSTED_SOURCE_HASHES.items():
        if file_sha(ROOT / 'scripts' / name) != expected:
            raise ValueError('Approved hosted wrapper source differs')
    wrapper = ROOT / 'scripts/run_tabular_hosted.py'
    prior = literal_source_constant(wrapper, 'PRIOR')
    for name, expected in prior.items():
        path = ROOT / 'results' / name
        if file_sha(path) != expected['sha256'] or file_sha(path.with_name(name + '.lock')) != expected['lock_sha256']:
            raise ValueError('Earlier budget ledger anchors differ')
    budget = record['config']['shots_per_class']
    expected = {'tabular_wrapper_sha256': APPROVED_HOSTED_SOURCE_HASHES['run_tabular_hosted.py'],
        'wrapper_sha256': APPROVED_HOSTED_SOURCE_HASHES['run_openrouter_jev.py'],
        'ledger_driver_sha256': APPROVED_HOSTED_SOURCE_HASHES['run_budgeted_hosted.py'],
        'ledger_id': canonical_tabular_ledger_header()['ledger_id'], 'prior_ledgers': prior,
        'prior_charged_or_reserved_usd': literal_source_constant(wrapper, 'PRIOR_TOTAL'),
        'stage_cap_usd': literal_source_constant(wrapper, 'STAGE_CAP'),
        'aggregate_authorized_usd': literal_source_constant(wrapper, 'TOTAL_CAP'),
        'dataset': record['dataset'], 'seed': 42, 'shots_per_class': budget, 'model_key': model_key}
    if model_key == 'jev_openrouter':
        expected.update({'route': 'OpenRouter', 'endpoint': 'https://openrouter.ai/api/v1/systemone',
            'price_verified_on': '2026-09-22', 'price_source': 'https://openrouter.ai/typesafe/jev-1.13',
            'response_model_allowlist': ['typesafe/jev-1.13', 'typesafe/jev-1.13-20260917'],
            'native_protocol': 'frozen_JevClassifier_Choice', 'reserve_checks_included_in_latency': True})
    else:
        expected.update({'pricing_sha256': APPROVED_PRICE_HASHES[record['config']['model']], 'verified_on': '2026-09-22'})
    return expected


def audit_model_config(record):
    model, method, config = record['config']['model'], record['method'], record['config']
    if model in OPEN_MODELS:
        presets = read_json(ROOT / 'configs/models.json')
        small = model == OPEN_MODELS[0]
        expected = dict(presets['qwen_small' if small else 'qwen_main'])
        expected['device'] = 'mps' if small else 'cuda'
        expected['shots_per_class'] = 4 if method == 'few_shot' else 0
        actual = dict(config)
        adapter = actual.pop('adapter_path', None)
        if (method == 'lora') != (isinstance(adapter, str) and bool(adapter)):
            raise ValueError('Open model adapter/config mode differs')
        if actual != expected:
            raise ValueError('Open model pinned preset/revision/context/device differs')
        if method == 'lora':
            training = record['adapter_training']
            if (training.get('load_in_4bit') is not (not small) or training.get('device') != expected['device']
                    or training.get('dtype') not in (('float32',) if small else ('float16', 'bfloat16'))):
                raise ValueError('Open model adaptation precision/device/quantization differs')
    elif model in HOSTED:
        key = {HOSTED[0]: 'jev_openrouter', HOSTED[1]: 'openai_economical', HOSTED[2]: 'openai_frontier'}[model]
        expected = dict(read_json(ROOT / 'configs/tabular_hosted.json')[key])
        expected['shots_per_class'] = 4 if method == 'few_shot' else 0
        actual = {key: value for key, value in config.items() if key != 'budget_guard'}
        if actual != expected:
            raise ValueError('Hosted fixed preset/output cap differs')
        if config.get('budget_guard') != expected_hosted_guard(record, key):
            raise ValueError('Hosted wrapper/ledger/cap/anchor provenance differs')


def audit_record(record, dataset, native):
    """All environments must consume the identical prepared bundle, without exceptions."""
    expected_manifest = digest(dataset.manifest)
    if record['dataset'] != dataset.name or record['labels'] != dataset.labels or record['seed'] != 42:
        raise ValueError('Dataset, labels or seed mismatch')
    if record['dataset_manifest'] != dataset.manifest or record['manifest_sha256'] != expected_manifest:
        raise ValueError('Prepared manifest differs from frozen native/serialized bundle')
    if record['test_ids_sha256'] != digest([row.id for row in dataset.test]):
        raise ValueError('Ordered test IDs differ')
    if record['implementation_sha256'] != FROZEN_CORE_SHA or record['environment']['source_sha256'] != FROZEN_CORE_SHA:
        raise ValueError('Frozen core implementation differs')
    if record['prompt_template_sha256'] != digest(build_prompt.__code__.co_consts.__repr__()):
        raise ValueError('Prompt template differs')
    _, _, method, budget = condition(record)
    if condition(record) not in expected_conditions():
        raise ValueError('Run is outside the declared 66-condition matrix')
    audit_model_config(record)
    expected_ids = _expected_ids(dataset, budget)
    if record['training_example_ids'] != expected_ids:
        raise ValueError('Ordered training/example IDs differ from declared budget')
    training = record.get('adapter_training') or record.get('training') or {}
    if training.get('validation_rows', 0) != 0:
        raise ValueError('Tabular fixed-recipe runs may not use validation labels')
    if method == 'classical_tabular':
        config = record['config']
        if (config.get('input_representation') != 'native_columns' or config.get('native_records_sha256') != digest(native)
                or config.get('feature_schema_sha256') != digest(dataset.manifest['tabular']['features'])):
            raise ValueError('Native feature/config identity differs')
        if (config.get('tabular_classical_sha256') != file_sha(ROOT / 'scripts/run_tabular_classical.py') or
                config.get('tabular_data_sha256') != dataset.manifest['tabular']['preparation_implementation_sha256'] or
                config.get('tabular_data_sha256') != file_sha(ROOT / 'scripts/tabular_data.py')):
            raise ValueError('Native extension implementation differs')
        by_id = {value['id']: value for value in native['train']}
        if (record.get('status') == 'complete' or training) and (training.get('training_row_ids') != expected_ids or training.get('training_native_records_sha256') != digest([by_id[row_id] for row_id in expected_ids])
                or training.get('selection_metric') != 'fixed_a_priori' or training.get('candidate_budget') != 1):
            raise ValueError('Native fitting provenance differs')
        if training:
            from run_tabular_classical import (feature_matrix, make_pipeline,
                                              model_parameters, preprocessing_provenance)
            model = config['model']
            expected_parameters = model_parameters(model, 42)
            if (training.get('selected_parameters') != expected_parameters or training.get('model') != model
                    or training.get('seed') != 42 or training.get('train_per_class') != budget
                    or training.get('training_rows') != len(expected_ids)
                    or training.get('input_representation') != 'native_columns'
                    or training.get('refit_with_validation') is not False):
                raise ValueError('Native fixed recipe differs')
            # Independently refit only the inexpensive feature transformations,
            # never the classifier, to verify actual recorded medians/scales/
            # category levels against precisely the declared training rows.
            features = dataset.manifest['tabular']['features']
            pipeline = make_pipeline(model, features, 42)
            pipeline.named_steps['features'].fit(feature_matrix([by_id[row_id] for row_id in expected_ids], features))
            if training.get('features') != preprocessing_provenance(pipeline, features):
                raise ValueError('Native training-only preprocessing provenance differs')
    if method == 'lora':
        selected = select_examples(dataset.train, dataset.labels, 4, 42)
        expected_prompt = hashlib.sha256(json.dumps([build_prompt(r, dataset.labels) for r in selected], ensure_ascii=False).encode()).hexdigest()
        if (training['training_row_ids'] != expected_ids or training['prompt_sha256'] != expected_prompt
                or training['model_id'] != record['config']['model'] or training['resolved_revision'] != record['config']['revision']
                or training.get('test_accessed') is not False or training.get('selection') != 'fixed_final_checkpoint'):
            raise ValueError('Adapter training provenance differs')
        recipe = {'train_per_class': 4, 'epochs': 3, 'learning_rate': 2e-4, 'batch_size': 1,
                  'gradient_accumulation_steps': 8, 'r': 8, 'lora_alpha': 16,
                  'max_length': 2048, 'max_steps': None, 'target_modules': 'all-linear',
                  'gradient_checkpointing': True, 'supervision': 'numeric_class_id_plus_eos_response_only'}
        if any(training.get(key) != value for key, value in recipe.items()):
            raise ValueError('Adapter fixed recipe differs')
    return {'prepared_manifest_sha256': expected_manifest,
            'prepared_content_sha256': dataset.manifest['prepared_content_sha256'],
            'split_files_sha256': dataset.manifest['files_sha256'],
            'native_files_sha256': dataset.manifest['tabular']['native_files_sha256'],
            'ordered_training_ids_verified': True, 'ordered_test_ids_verified': True,
            'validation_labels_used': 0, 'core_source_sha256': FROZEN_CORE_SHA}


def _metadata_values(predictions, key):
    return sorted({str(p.metadata[key]) for p in predictions if p.metadata.get(key) is not None})


def audit_run(path, dataset, native, root, *, samples=2000):
    path = Path(path)
    record = read_json(path)
    audit = audit_record(record, dataset, native)
    key = condition(record)
    _, model, method, budget = key
    training = record.get('adapter_training') or record.get('training') or {}
    method_label = {'classical_tabular': 'full training' if budget is None else '4/class',
                    'zero_shot': 'zero-shot', 'few_shot': 'few-shot 4/class',
                    'lora': 'QLoRA 4/class' if training.get('load_in_4bit') else 'LoRA 4/class'}[method]
    row = {'dataset': dataset.name, 'model': model, 'display_model': DISPLAY[model], 'method': method,
        'method_label': method_label, 'train_per_class': budget, 'train_labels': len(record['training_example_ids']),
        'dev_labels': 0, 'track': 'full training; extra labels' if budget is None else 'zero/matched labels',
        'status': record.get('status', 'unknown'), 'run_id': record['run_id'], 'seed': record['seed'],
        'source_path': path.relative_to(root).as_posix(), 'manifest_sha256': record['manifest_sha256'],
        'requested_model': model, 'requested_revision': record['config'].get('revision'),
        'config': record['config'], 'training_dtype': training.get('dtype'),
        'training_quantized_4bit': training.get('load_in_4bit'),
        'training_seconds': training.get('training_s', training.get('fit_and_selection_s')),
        'audit': audit, 'n_test': len(dataset.test), 'accuracy': None, 'macro_f1': None,
        'n_failures': None, 'group_bootstrap': None, **{name: None for name in PROBABILITY_FIELDS}}
    manifest_path, predictions_path = path.with_name('test_manifest.json'), path.with_name('predictions.jsonl')
    if record.get('status') != 'complete':
        return row, None
    if not manifest_path.is_file() or not predictions_path.is_file():
        row['status'] = 'pending: completed marker lacks final artifacts'
        return row, None
    expected_test = {'dataset': dataset.name, 'labels': dataset.labels, 'manifest_sha256': digest(dataset.manifest),
        'rows': [{'id': r.id, 'label': r.label, 'text_sha256': hashlib.sha256(r.text.encode()).hexdigest()} for r in dataset.test]}
    if read_json(manifest_path) != expected_test:
        raise ValueError(f'Test manifest differs from frozen rows: {path}')
    predictions = [Prediction(**json.loads(line)) for line in predictions_path.read_text().splitlines() if line.strip()]
    if [p.row_id for p in predictions] != [row.id for row in dataset.test]:
        raise ValueError(f'Prediction order or coverage differs: {path}')
    for prediction in predictions:
        if prediction.error is not None:
            continue
        if model in OPEN_MODELS and (prediction.metadata.get('requested_model') != model or
                prediction.metadata.get('resolved_revision') != record['config']['revision'] or
                prediction.metadata.get('device') != record['config']['device'] or
                prediction.metadata.get('dtype') != record['config'].get('dtype', 'float16')):
            raise ValueError('Successful local prediction model/revision differs')
        allowed = record['config'].get('budget_guard', {}).get('response_model_allowlist')
        if allowed and prediction.metadata.get('resolved_model') not in allowed:
            raise ValueError('Successful hosted prediction model is outside recorded allowlist')
    metrics = evaluate(dataset.test, predictions, len(dataset.labels))
    # Recompute all metrics, including failures and probability coverage, rather
    # than trusting a stale edited summary. Original bootstrap is kept untouched.
    for name, value in metrics.items():
        saved = record['metrics'].get(name)
        if isinstance(value, float):
            equal = isinstance(saved, (int, float)) and math.isclose(value, saved, rel_tol=1e-10, abs_tol=1e-12)
        else:
            equal = saved == value
        if not equal:
            raise ValueError(f'Saved metric differs from raw predictions: {name} in {path}')
    y = [r.label for r in dataset.test]
    chosen = [p.label if p.error is None and p.label is not None and 0 <= p.label < len(dataset.labels) else -1 for p in predictions]
    groups = [item['text_sha256'] for item in expected_test['rows']]
    bootstrap = group_bootstrap(y, chosen, groups, n_classes=len(dataset.labels), samples=samples)
    row.update({key: value for key, value in metrics.items() if key != 'bootstrap'})
    row.update(group_bootstrap=bootstrap, resolved_models=_metadata_values(predictions, 'resolved_model'),
        resolved_revisions=_metadata_values(predictions, 'resolved_revision'),
        inference_dtypes=_metadata_values(predictions, 'dtype'), devices=_metadata_values(predictions, 'device'),
        probability_kinds=_metadata_values(predictions, 'probability_kind'),
        scoring_rules=_metadata_values(predictions, 'scoring'),
        output_protocol='native columns' if method == 'classical_tabular' else
            'native Choice' if model == HOSTED[0] else 'restricted-label likelihood' if model in OPEN_MODELS else 'generated numeric label',
        artifact_sha256={'run.json': file_sha(path), 'test_manifest.json': file_sha(manifest_path), 'predictions.jsonl': file_sha(predictions_path)})
    return row, {'y': y, 'predictions': chosen, 'groups': groups, 'n_classes': len(dataset.labels),
                 'training_example_ids': record['training_example_ids'], 'manifest_sha256': record['manifest_sha256']}


def comparison_specs():
    result = []
    for dataset in DATASETS:
        jev = (dataset, HOSTED[0], 'few_shot', 4)
        for model, method in ((HOSTED[2], 'few_shot'), (OPEN_MODELS[1], 'few_shot'),
                              ('logistic_regression', 'classical_tabular'), ('random_forest', 'classical_tabular')):
            result.append(('main matched comparison', jev, (dataset, model, method, 4)))
        for model in OPEN_MODELS:
            result.append(('main matched comparison', (dataset, model, 'lora', 4), (dataset, model, 'few_shot', 4)))
        for model in (*OPEN_MODELS, *HOSTED):
            result.append(('descriptive: few-shot versus zero-shot; unequal labels', (dataset, model, 'few_shot', 4), (dataset, model, 'zero_shot', 0)))
        for model in CLASSICAL:
            result.append(('descriptive: full versus 4/class; unequal labels', (dataset, model, 'classical_tabular', None), (dataset, model, 'classical_tabular', 4)))
    return result


def paired_contrasts(rows_by_key, payloads, *, samples=2000):
    result = []
    for category, ka, kb in comparison_specs():
        a, b = rows_by_key[ka], rows_by_key[kb]
        item = {'dataset': ka[0], 'category': category, 'a': {'model': ka[1], 'method': ka[2], 'train_per_class': ka[3], 'run_id': a.get('run_id')},
                'b': {'model': kb[1], 'method': kb[2], 'train_per_class': kb[3], 'run_id': b.get('run_id')},
                'direction': 'A minus B', 'status': 'pending', 'group_bootstrap': None}
        if a['status'] != 'complete' or b['status'] != 'complete':
            item['reason'] = 'Both fully audited complete runs are required'
            result.append(item)
            continue
        pa, pb = payloads[ka], payloads[kb]
        if any(pa[field] != pb[field] for field in ('y', 'groups', 'manifest_sha256', 'n_classes')):
            raise ValueError('Pair does not use identical held-out data')
        equal_training = pa['training_example_ids'] == pb['training_example_ids']
        if category == 'main matched comparison' and not equal_training:
            raise ValueError('Main paired comparison has unequal training IDs')
        item.update(status='complete', equal_training_ids=equal_training,
                    train_labels_a=a['train_labels'], train_labels_b=b['train_labels'],
                    group_bootstrap=group_bootstrap(pa['y'], pa['predictions'], pa['groups'], pb['predictions'],
                        n_classes=pa['n_classes'], samples=samples))
        result.append(item)
    return result


def collect(root=ROOT, data_root=None, *, samples=2000):
    root = Path(root).resolve()
    data_root = Path(data_root) if data_root else root / 'data/tabular-full'
    prepared = {name: load_native_prepared(data_root / name) for name in DATASETS}
    keys = expected_conditions()
    rows, payloads = {}, {}
    for folder in ('classical', 'local', 'colab', 'hosted'):
        for path in sorted((root / 'results/tabular' / folder).glob('*/run.json')):
            record = read_json(path)
            key = condition(record)
            if key not in keys:
                raise ValueError(f'Unexpected condition in tabular results: {path}')
            if key in rows:
                raise ValueError(f'Ambiguous duplicate run condition: {key}')
            rows[key], payload = audit_run(path, *prepared[key[0]], root, samples=samples)
            if payload is not None:
                payloads[key] = payload
    for key in keys:
        if key not in rows:
            dataset, model, method, budget = key
            rows[key] = {'dataset': dataset, 'model': model, 'display_model': DISPLAY[model], 'method': method,
                         'train_per_class': budget, 'track': 'full training; extra labels' if budget is None else 'zero/matched labels',
                         'status': 'pending: no run artifact', 'accuracy': None, 'macro_f1': None,
                         'n_failures': None, 'group_bootstrap': None}
    datasets = {name: {'labels': data.labels, 'manifest_sha256': digest(data.manifest),
                      'prepared_content_sha256': data.manifest['prepared_content_sha256'],
                      'source_sha256': data.manifest['source']['observed_sha256'],
                      'split_files_sha256': data.manifest['files_sha256'],
                      'native_files_sha256': data.manifest['tabular']['native_files_sha256'],
                      'splits': data.manifest['splits'], 'test_text_clusters': len({r.text for r in data.test}),
                      'max_serialized_characters': data.manifest['tabular']['max_serialized_characters']}
                for name, (data, _) in prepared.items()}
    plans = []
    for folder in ('local', 'colab'):
        for path in sorted((root / 'results/tabular' / folder / 'plans').glob('*.json')):
            plan = read_json(path)
            if 'plan_id' in plan:
                plans.append({'source_path': path.relative_to(root).as_posix(), 'file_sha256': file_sha(path),
                    **{key: plan.get(key) for key in ('plan_id', 'helper_sha256', 'source_sha256', 'model', 'recipe', 'load_in_4bit')}})
    return {'schema_version': 1, 'expected_runs': len(keys), 'complete_runs': sum(row['status'] == 'complete' for row in rows.values()),
            'analysis_sha256': file_sha(__file__), 'bootstrap_samples': samples, 'seed': 42,
            'audit_policy': 'Exact frozen prepared bundle, ordered test/train IDs, unchanged core; raw-prediction metric recomputation',
            'datasets': datasets, 'local_execution_plans': plans, 'runs': [rows[key] for key in keys],
            'comparisons': paired_contrasts(rows, payloads, samples=samples)}


def fmt(value, places=4):
    return '—' if value is None else f'{value:.{places}f}'


def score(row, metric):
    if row.get(metric) is None:
        return '—'
    ci = row['group_bootstrap']['metrics'][metric]['ci95']
    return f"{row[metric]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def relative_link(path, output):
    return os.path.relpath(path, output.parent).replace(os.sep, '/')


def write_report(summary, output, root=ROOT):
    output, root = Path(output), Path(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = summary['runs']
    lines = ['# Serialized tabular classification benchmark', '',
        f"**{summary['complete_runs']} of {summary['expected_runs']} planned runs are complete and audited.** Pending runs have no score. The matrix consists of 30 native classical, 18 open-model and 18 hosted-model conditions.", '',
        'All methods use the same held-out rows. Four-per-class prompting, LoRA/QLoRA and matched classical models use the identical eight labels for Titanic/Breast Cancer or 12 labels for Wine. Zero-shot uses zero new task labels. The full-training classical track uses extra labels and is reported separately. No validation labels are used in this fixed-recipe extension.', '',
        'Intervals below are **unstratified group percentile bootstrap intervals**: sample identical serialized-test-text groups with replacement, retaining every member. This accounts for exact repeated feature vectors, including conflicting targets, but not every possible dependency. These exploratory, unadjusted intervals condition on one fixed split, seed, prompt, serialization and fitted model; they do not measure variation across training seeds or establish general superiority.', '',
        f"Each interval uses {summary['bootstrap_samples']} replicates and seed 42. Original per-row bootstrap fields in run artifacts are preserved but are not the intervals displayed here.", '',
        '## Data and preparation audit', '', '| Dataset | Train / dev / test rows | Test feature groups | Maximum serialized characters |', '|---|---:|---:|---:|']
    for name, info in summary['datasets'].items():
        sizes = ' / '.join(str(info['splits'][split]['rows']) for split in ('train', 'validation', 'test'))
        lines.append(f"| {name} | {sizes} | {info['test_text_clusters']} | {info['max_serialized_characters']} |")
    lines += ['', 'Prepared manifests, serialized/native file hashes, ordered training/test IDs and unchanged inference-core identities are checked against the local frozen bundle. Completed metrics are recomputed from raw predictions. Imported Colab runs must pass the same checks; no cross-environment hash exceptions are allowed.', '']
    for dataset in DATASETS:
        lines += [f'## {dataset}: zero-shot and matched-label methods', '',
            '| Model | Method | Train labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Failures | Probability coverage | Run |',
            '|---|---|---:|---|---|---:|---:|---|']
        for row in rows:
            if row['dataset'] != dataset or row['train_per_class'] is None:
                continue
            link = f"[artifact]({relative_link(root / row['source_path'], output)})" if row.get('source_path') else '—'
            method = row.get('method_label', row['method'])
            if row['status'] != 'complete':
                lines.append(f"| {row['display_model']} | {method} | — | pending | pending | — | — | {link} |")
            else:
                lines.append(f"| {row['display_model']} | {method} | {row['train_labels']} | {score(row, 'accuracy')} | {score(row, 'macro_f1')} | {row['n_failures']} | {row['probability_coverage']:.1%} | {link} |")
        lines.append('')
    lines += ['## Full-training classical references: extra labels', '',
        'Every listed estimator uses fixed parameters; this report does not choose a winner using test results. Each full-training model fits preprocessing only on the full training split, still without validation labels. These rows do not have an equal-label budget with the four-per-class arms.', '',
        '| Dataset | Model | Train labels | Accuracy [95% CI] | Macro-F1 [95% CI] | Failures | Probability coverage |',
        '|---|---|---:|---|---|---:|---:|']
    for row in rows:
        if row['train_per_class'] is not None:
            continue
        if row['status'] != 'complete':
            lines.append(f"| {row['dataset']} | {row['display_model']} | — | pending | pending | — | — |")
        else:
            lines.append(f"| {row['dataset']} | {row['display_model']} | {row['train_labels']} | {score(row, 'accuracy')} | {score(row, 'macro_f1')} | {row['n_failures']} | {row['probability_coverage']:.1%} |")
    for category in dict.fromkeys(item['category'] for item in summary['comparisons']):
        lines += ['', f'## Paired contrasts — {category}', '',
            'Differences are A minus B. Both predictions receive the same resampled groups. All intervals are exploratory and unadjusted for multiple comparisons.', '',
            '| Dataset | A | B | Δ accuracy [95% CI] | Δ macro-F1 [95% CI] | Groups |',
            '|---|---|---|---|---|---:|']
        for item in summary['comparisons']:
            if item['category'] != category:
                continue
            def label(arm):
                method = 'full training' if arm['train_per_class'] is None else arm['method'].replace('_', ' ')
                return f"{DISPLAY[arm['model']]} {method}"
            if item['status'] != 'complete':
                lines.append(f"| {item['dataset']} | {label(item['a'])} | {label(item['b'])} | pending | pending | — |")
                continue
            bootstrap = item['group_bootstrap']
            entries = []
            for metric in ('accuracy', 'macro_f1'):
                value = bootstrap['metrics'][metric]
                entries.append(f"{value['estimate']:+.4f} [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}]")
            lines.append(f"| {item['dataset']} | {label(item['a'])} | {label(item['b'])} | {entries[0]} | {entries[1]} | {bootstrap['n_groups']} |")
    lines += ['', '## Probability and execution differences', '',
        'Classification failures count as incorrect predictions. Probability metrics cover only valid predictions with complete class distributions; coverage is always shown. Native classical probabilities are uncalibrated estimates (RBF SVM supplies none); Qwen values normalize numeric-label-plus-EOS likelihoods; Jev supplies native Choice distributions; OpenAI generated-label runs supply no class distribution. These are different probability-generating procedures.', '',
        '| Dataset | Model / method | Probability kind | Coverage | Log loss | Brier sum | ECE (15 bins) |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        if row['status'] == 'complete':
            lines.append(f"| {row['dataset']} | {row['display_model']} / {row['method_label']} | {', '.join(row['probability_kinds']) or 'unreported'} | {row['probability_coverage']:.1%} | {fmt(row.get('log_loss'))} | {fmt(row.get('brier_sum'))} | {fmt(row.get('ece_15_equal_width'))} |")
    lines += ['', '## Model identity and numerical precision', '',
        'The table aggregates only completed audited runs. Short revisions below are identifiers, not mutable model aliases; full values and wrapper provenance remain in the JSON/CSV. Missing hosted precision or hardware details are marked undisclosed.', '',
        '| Requested model | Method | Returned model / revision | Inference dtype | Training dtype / quantization | Device |',
        '|---|---|---|---|---|---|']
    identities = {}
    for row in rows:
        if row['status'] != 'complete' or row['method'] == 'classical_tabular':
            continue
        key = (row['model'], row['method_label'])
        target = identities.setdefault(key, {'resolved': set(), 'inference': set(), 'training': set(), 'devices': set()})
        target['resolved'].update(row['resolved_models'])
        target['resolved'].update(value[:12] for value in row['resolved_revisions'])
        target['inference'].update(row['inference_dtypes'])
        target['devices'].update(row['devices'])
        if row['training_dtype']:
            target['training'].add(row['training_dtype'] + (' / 4-bit base' if row['training_quantized_4bit'] else ' / ordinary LoRA'))
    for (model, method), values in identities.items():
        text = lambda key, default: ', '.join(sorted(values[key])) or default
        lines.append(f"| `{model}` | {method} | {text('resolved', 'unreported')} | {text('inference', 'undisclosed')} | {text('training', 'not applicable')} | {text('devices', 'undisclosed')} |")
    lines += ['', 'Requested and returned model IDs/revisions, actual recorded inference precision/devices, training precision/quantization, wrapper hashes, latency and training time are retained in the JSON/CSV. Hosted hardware is not disclosed. Local sequential likelihood scoring, hosted end-to-end request latency and native amortized batch inference are different timing conventions.', '',
        '## Scope and limitations', '',
        'These are familiar public benchmarks; memorized rows or dataset-specific priors may influence pretrained models. Grouped splits prevent exact feature overlap in newly supplied data, but cannot remove pretraining contamination. Wine cultivar numbers are arbitrary identifiers, so zero-shot results are not evidence that cultivar names have transferable semantic meaning. The Breast Cancer task is a benchmark of a historical dataset, not a clinical validation or deployment recommendation.', '',
        'This extension evaluates one fixed named-feature serialization, with source scales preserved and no truncation. It does not compare alternative serialization formats, feature-order permutations or missing-value wordings. Family relationships in Titanic may cause dependence beyond identical feature vectors. Tiny four-per-class adaptation is a narrow fixed-recipe test; a poor result does not establish a general limit of fine-tuning.', '',
        'Hosted LoRA for Jev/OpenAI is unsupported by this study and is not assigned a score. LoRA and QLoRA apply only to the downloadable open models.', '',
        f"Protocol and source citations: [TABULAR_PROTOCOL.md]({relative_link(root / 'docs/TABULAR_PROTOCOL.md', output)}). Machine-readable evidence: [{output.with_suffix('.json').name}]({output.with_suffix('.json').name}) and [{output.with_suffix('.csv').name}]({output.with_suffix('.csv').name}).", '']
    output.write_text('\n'.join(lines))
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    fields = ['dataset', 'model', 'method', 'method_label', 'track', 'train_per_class', 'train_labels', 'dev_labels', 'status', 'n_test',
        'accuracy', 'accuracy_ci_low', 'accuracy_ci_high', 'macro_f1', 'macro_f1_ci_low', 'macro_f1_ci_high', 'n_groups',
        'n_failures', *PROBABILITY_FIELDS, 'latency_p50_s', 'latency_p95_s', 'training_seconds', 'requested_model',
        'resolved_models', 'requested_revision', 'resolved_revisions', 'inference_dtypes', 'devices', 'training_dtype',
        'training_quantized_4bit', 'probability_kinds', 'output_protocol', 'run_id', 'source_path', 'manifest_sha256', 'config']
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {field: row.get(field) for field in fields}
            if row.get('group_bootstrap'):
                flat['n_groups'] = row['group_bootstrap']['n_groups']
                for metric in ('accuracy', 'macro_f1'):
                    flat[metric + '_ci_low'], flat[metric + '_ci_high'] = row['group_bootstrap']['metrics'][metric]['ci95']
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in flat.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/TABULAR_COMPARISON.md')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args(argv)
    summary = collect(args.root, args.data_root, samples=args.bootstrap_samples)
    write_report(summary, args.output, args.root)
    print(json.dumps({'expected_runs': summary['expected_runs'], 'complete_runs': summary['complete_runs'],
                      'completed_contrasts': sum(item['status'] == 'complete' for item in summary['comparisons']),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
