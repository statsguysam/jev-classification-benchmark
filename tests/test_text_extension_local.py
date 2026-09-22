"""Offline synthetic-model verification; no hosted calls or model downloads."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_text_extension_local as runner
import export_text_extension_colab as exporter
import import_text_extension_colab as importer
from jevbench.data import save_prepared
from jevbench.types import PreparedDataset,Row


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    datasets={}
    pins={}
    for name,n in [('sst2',2),('trec',6)]:
        train=[Row(f'{name}-train-{i}',f'Training example {i} in {name}',i%n) for i in range(n*5)]
        validation=[Row(f'{name}-validation-{i}',f'Validation {i} in {name}',i) for i in range(n)]
        test=[Row(f'{name}-test-{i}',f'Held out {i} in {name}',i%n) for i in range(200)]
        dataset=PreparedDataset(name,[f'class{i}' for i in range(n)],train,validation,test,{})
        dataset.manifest['prepared_content_sha256']=runner.digest({split:[asdict(row) for row in getattr(dataset,split)] for split in ('train','validation','test')})
        folder=tmp_path/'data/pilot'/name
        save_prepared(dataset,folder)
        pins[name]={'manifest_sha256':runner.digest(dataset.manifest),'manifest_file_sha256':runner.file_sha(folder/'manifest.json'),
                    'n_test':200,'n_classes':n}
        datasets[name]=dataset
    monkeypatch.setattr(runner,'DATASET_PINS',pins)
    return tmp_path,datasets


class HiddenTruth:
    id='unlabeled'
    text='A test text whose truth must not be inspected by inference.'
    @property
    def label(self): raise AssertionError('Inference accessed a test label')


class ToyTokenizer:
    eos_token_id=2
    def __init__(self,preset):self.preset,self.chat_template=preset,preset['chat_template']
    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt,thinking):
        assert messages[0]=={'role':'system','content':'You are a helpful assistant.'}
        assert add_generation_prompt and thinking is False
        return [0,1] if tokenize else runner.frozen.render_chat(self.preset,messages[1]['content'])
    def encode(self,text,add_special_tokens=False):
        assert not add_special_tokens
        return [3+int(text)]


def toy_provider(key='smollm2'):
    torch=pytest.importorskip('torch')
    preset=runner.load_presets()['models'][key]
    provider=runner.frozen.FixedChatClassifier(runner.model_config(key,'cuda'),preset,ROOT/'data/hf')
    class ToyModel:
        config=SimpleNamespace(max_position_embeddings=8192)
        def __call__(self,input_ids,**kwargs):
            assert kwargs['use_cache'] is False
            logits=torch.zeros((1,input_ids.shape[1],9));logits[:,:,3]=2
            return SimpleNamespace(logits=logits)
    provider._model,provider._tokenizer=ToyModel(),ToyTokenizer(preset)
    provider._torch,provider._device=torch,'cpu'
    provider.metadata.update(resolved_revision=preset['revision'],device='cuda',dtype='float16')
    return provider


def test_default_plan_has_exact_1600_rows_and_training_only_examples(prepared,monkeypatch,capsys):
    root,datasets=prepared
    monkeypatch.setattr(runner.frozen,'download_public',lambda *_:pytest.fail('Unrequested download'))
    monkeypatch.setattr(runner.frozen.FixedChatClassifier,'_load',lambda *_:pytest.fail('Plan loaded a model'))
    runner.main(['--data-root',str(root/'data/pilot')])
    plan=json.loads(capsys.readouterr().out)
    assert plan['prediction_rows']==1600 and plan['new_hosted_calls']==0 and not plan['adapter_training']
    for job in plan['jobs']:
        dataset=datasets[job['dataset']]
        assert job['n_test']==200
        assert job['training_example_ids']==[r.id for r in runner.select_examples(dataset.train,dataset.labels,job['shots_per_class'],42)]
        assert not set(job['training_example_ids']) & {r.id for r in dataset.test+dataset.validation}
    assert all(model['text_extension']['numeric_helper_sha256']==runner.FROZEN_HELPER_SHA for model in plan['models'].values())


def test_changed_prepared_content_or_frozen_helper_is_rejected(prepared,monkeypatch):
    _,datasets=prepared
    dataset=datasets['sst2']
    dataset.test[0]=Row(dataset.test[0].id,'Changed held-out text',dataset.test[0].label)
    with pytest.raises(ValueError,match='row contents'):runner.validate_dataset(dataset)
    monkeypatch.setattr(runner,'FROZEN_HELPER_SHA','0'*64)
    with pytest.raises(ValueError,match='helper or model presets'):runner.load_presets()


@pytest.mark.parametrize('key',runner.MODEL_KEYS)
@pytest.mark.parametrize('n_classes',[2,6])
def test_unchanged_scorer_uses_id_plus_eos_without_reading_truth(key,n_classes):
    provider=toy_provider(key)
    labels=[str(i) for i in range(n_classes)]
    prediction=provider.predict(HiddenTruth(),labels,runner.build_prompt(HiddenTruth(),labels))
    assert prediction.error is None and prediction.label==0
    assert sum(prediction.probabilities)==pytest.approx(1)
    assert prediction.metadata['context_forward_passes']==n_classes
    assert prediction.metadata['candidate_tokens_scored']==2*n_classes
    assert prediction.metadata['scoring']=='sum_logp_numeric_id_plus_eos'
    assert prediction.metadata['rendered_input_tokens']==prediction.input_tokens==2


def test_preflight_preserves_context_limit_and_complete_prompt(prepared):
    _,datasets=prepared
    provider=toy_provider()
    checks=runner.frozen.preflight(provider,[(datasets['trec'],4)])
    assert checks[0]['n_test']==200 and checks[0]['max_prompt_plus_label_tokens']==4
    provider._model.config.max_position_embeddings=3
    with pytest.raises(ValueError,match='no truncation'):runner.frozen.preflight(provider,[(datasets['trec'],4)])


def run_toy(root,dataset,shots):
    provider=toy_provider()
    with patch.object(runner.frozen.providers,'build_provider',return_value=provider):
        record=runner.run_model(dataset,provider.config,root/exporter.RESULT_ROOT,shots=shots,bootstrap_samples=100)
    return root/exporter.RESULT_ROOT/record['run_id']/'run.json'


def test_source_audit_is_model_free_and_rejects_provenance_or_metric_tampering(prepared):
    root,datasets=prepared;dataset=datasets['trec']
    path=run_toy(root,dataset,4)
    with patch.object(runner.frozen.FixedChatClassifier,'_load',side_effect=AssertionError('Audit loaded model')):
        record,predictions=runner.audit_source_run(path,dataset,'smollm2',4)
    assert len(predictions)==200 and len(record['training_example_ids'])==24
    record['config']['text_extension']['wrapper_sha256']='0'*64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='identity'):runner.audit_source_run(path,dataset,'smollm2',4)
    record['config']['text_extension']['wrapper_sha256']=runner.file_sha(runner.__file__)
    record['metrics']['accuracy']=1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='metrics'):runner.audit_source_run(path,dataset,'smollm2',4)


def test_result_checksums_full_inventory_and_immutable_import(prepared,tmp_path):
    root,datasets=prepared
    for dataset in datasets.values():
        for shots in (0,4):run_toy(root,dataset,shots)
    archive=exporter.export_results(root,model_keys=['smollm2'])
    output=tmp_path/'imported/local'
    report=importer.import_archive(archive,output=output,data_root=root/'data/pilot',model_keys=['smollm2'],expected_sha256=runner.file_sha(archive))
    assert report['status']=='verified' and len(report['conditions'])==4 and not output.exists()
    with pytest.raises(ValueError,match='Downloaded archive'):importer.import_archive(archive,expected_sha256='0'*64)
    report=importer.import_archive(archive,output=output,data_root=root/'data/pilot',model_keys=['smollm2'],execute=True)
    assert report['status']=='imported'
    prediction=next(output.glob('*/predictions.jsonl'));prediction.write_text('changed')
    with pytest.raises(ValueError,match='refusing overwrite'):
        importer.import_archive(archive,output=output,data_root=root/'data/pilot',model_keys=['smollm2'],execute=True)
    stream=io.BytesIO()
    with zipfile.ZipFile(archive) as original,zipfile.ZipFile(stream,'w') as modified:
        for info in original.infolist():
            raw=original.read(info.filename)
            if info.filename.endswith('/predictions.jsonl'):raw+=b'changed'
            modified.writestr(info.filename,raw)
    with zipfile.ZipFile(stream) as modified,pytest.raises(ValueError,match='checksum differs'):
        importer.verify_members(modified)


@pytest.mark.parametrize('member',['../escape.json','results/text_extension/local/run/secret.txt'])
def test_result_archive_rejects_unsafe_or_extra_members(member):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive:archive.writestr(member,'{}')
    with zipfile.ZipFile(stream) as archive,pytest.raises(ValueError):importer.verify_members(archive)
