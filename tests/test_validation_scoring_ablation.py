from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_validation_scoring_ablation as ablation


def test_same_forward_passes_separate_id_eos_and_change_argmax():
    torch = pytest.importorskip("torch")
    calls = []
    class Toy:
        def __call__(self, input_ids, **kwargs):
            calls.append(input_ids.tolist())
            assert kwargs["use_cache"] is False
            logits = torch.zeros((1, 4, 5))
            logits[0, 1, 3], logits[0, 1, 4] = 2., 1.
            logits[0, 2, 2] = -3. if input_ids[0, -2].item() == 3 else 3.
            return SimpleNamespace(logits=logits)
    provider = SimpleNamespace(_torch=torch, _device="cpu", _model=Toy())
    result = ablation.score_context(provider, [0, 1], [[3, 2], [4, 2]])
    assert len(calls) == result["context_forward_passes"] == 2
    assert result["label_only"]["label"] == 0 and result["label_plus_eos"]["label"] == 1
    for i in range(2):
        assert result["label_logp"][i] + result["conditional_eos_logp"][i] == pytest.approx(result["joint_logp"][i], abs=1e-6)
    context = {"candidate_token_ids": [[3, 2], [4, 2]]}
    value = {"context": context, "protocol_sha256": "frozen", **result}
    ablation.audit_prediction(value, context, "frozen")
    value["label_logp"][0] = -100
    with pytest.raises(ValueError, match="token contribution"): ablation.audit_prediction(value, context, "frozen")


def test_checkpoint_rejects_partial_line_and_changed_context(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"context":')
    with pytest.raises(ValueError, match="Partial diagnostic"): ablation.read_checkpoint(path, {"contexts": []}, "pin")


def test_normalization_fails_nonfinite_and_preserves_ranking():
    assert sum(ablation.normalize([-1, -2, -3])) == pytest.approx(1)
    assert ablation.normalize([-1000, -1001]) == ablation.normalize([0, -1])
    with pytest.raises(ValueError): ablation.normalize([float("nan"), 0])


def test_joint_arm_exactly_matches_unchanged_core_scorer(monkeypatch):
    torch = pytest.importorskip("torch")
    class Tokenizer:
        eos_token_id = 2
        def encode(self, label, add_special_tokens=False):
            return [3 + int(label)]
    class Model:
        config = SimpleNamespace(max_position_embeddings=16)
        def __call__(self, input_ids, **kwargs):
            logits = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5) / 7
            logits[0, 2, 2] = input_ids[0, -2].float()
            return SimpleNamespace(logits=logits)
    provider = ablation.frozen.providers.HuggingFaceClassifier({"model": "synthetic", "max_context_tokens": 16})
    provider._torch, provider._device = torch, "cpu"
    provider._model, provider._tokenizer = Model(), Tokenizer()
    monkeypatch.setattr(ablation.frozen.providers, "prompt_token_ids", lambda *_: [0, 1])
    original = provider.predict(SimpleNamespace(id="blind-row"), ["a", "b"], "fixed")
    paired = ablation.score_context(provider, [0, 1], [[3, 2], [4, 2]])
    assert original.error is None
    assert paired["label_plus_eos"] == {"label": original.label, "probabilities": original.probabilities}


def test_default_plan_does_not_load_tokenizer_or_model(monkeypatch, capsys):
    monkeypatch.setattr(ablation, "load_inputs", lambda: ({}, {}, {}, {"a": [1] * 16, "b": [1] * 18}, {}))
    monkeypatch.setattr(ablation, "build_protocol", lambda *_: pytest.fail("Default loaded tokenizer"))
    ablation.main([])
    assert '"validation_rows": 34' in capsys.readouterr().out
