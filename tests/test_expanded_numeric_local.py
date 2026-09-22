"""No downloads or real model weights; verify rendering, leakage and scorer reuse."""
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_expanded_numeric_local as h
from jevbench.types import PreparedDataset, Row


class HiddenLabel:
    id = "test:0"
    text = "radius=12.3; texture=20.1"

    @property
    def label(self):
        raise AssertionError("Inference accessed the test label")


class TinyTokenizer:
    eos_token_id = 2

    def __init__(self, preset):
        self.preset = preset
        self.chat_template = preset["chat_template"]
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, thinking):
        self.calls.append((messages, tokenize, add_generation_prompt, thinking))
        assert messages[0] == {"role": "system", "content": h.SYSTEM_MESSAGE}
        assert thinking is False and add_generation_prompt is True
        return [0, 1] if tokenize else h.render_chat(self.preset, messages[1]["content"])

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [3 + int(text)]


def tiny_provider(key="smollm2"):
    torch = pytest.importorskip("torch")
    preset = h.load_presets()["models"][key]
    provider = h.FixedChatClassifier(h.model_config(key, "cuda"), preset, ROOT / "data/hf")

    class TinyModel:
        config = SimpleNamespace(max_position_embeddings=8192)

        def __call__(self, input_ids, **kwargs):
            assert kwargs["use_cache"] is False
            logits = torch.zeros((1, input_ids.shape[1], 5))
            logits[:, :, 4] = 2
            return SimpleNamespace(logits=logits)

    provider._model, provider._tokenizer = TinyModel(), TinyTokenizer(preset)
    provider._torch, provider._device = torch, "cpu"
    provider.metadata.update(resolved_revision=preset["revision"], device="cuda", dtype="float16")
    return provider


@pytest.mark.parametrize("key", h.MODEL_KEYS)
def test_fixed_system_and_disabled_reasoning_are_deterministic(key):
    preset = h.load_presets()["models"][key]
    rendered = h.render_chat(preset, "radius=12.3")
    assert h.SYSTEM_MESSAGE in rendered
    assert "Today's Date:" not in rendered and "<think>" not in rendered
    assert rendered == h.render_chat(preset, "radius=12.3")
    ids, metadata = h.render_tokens(TinyTokenizer(preset), preset, "radius=12.3")
    assert metadata["rendered_chat_prompt_sha256"] == h.text_sha(rendered)
    assert metadata["rendered_input_ids_sha256"] == h.digest(ids)
    altered = TinyTokenizer(preset)
    altered.chat_template += "changed"
    with pytest.raises(ValueError, match="pinned template"):
        h.render_tokens(altered, preset, "radius=12.3")


def test_frozen_likelihood_scorer_never_reads_test_label():
    provider = tiny_provider()
    result = provider.predict(HiddenLabel(), ["a", "b"], h.build_prompt(HiddenLabel(), ["a", "b"]))
    assert result.error is None and result.label == 1
    assert result.probabilities[1] == pytest.approx(math.exp(2) / (1 + math.exp(2)))
    assert result.metadata["candidate_tokens_scored"] == 4
    assert result.metadata["context_forward_passes"] == 2
    assert result.metadata["rendered_input_tokens"] == result.input_tokens == 2
    assert result.output_tokens == 0
    assert h.providers.prompt_token_ids.__name__ == "prompt_token_ids"


def test_preflight_checks_all_prompts_and_refuses_context_overflow():
    provider = tiny_provider()
    dataset = PreparedDataset("breast_cancer", ["a", "b"], [], [], [HiddenLabel()])
    assert h.preflight(provider, [(dataset, 0)])[0]["max_prompt_plus_label_tokens"] == 4
    provider._model.config.max_position_embeddings = 3
    with pytest.raises(ValueError, match="no truncation"):
        h.preflight(provider, [(dataset, 0)])


def test_plan_has_frozen_training_ids_and_cannot_download_or_load(monkeypatch, capsys):
    monkeypatch.setattr(h, "download_public", lambda *a: pytest.fail("Plan downloaded weights"))
    monkeypatch.setattr(h.FixedChatClassifier, "_load", lambda *a: pytest.fail("Plan loaded a model"))
    h.main(["--device", "cuda"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["prediction_rows"] == 600 and plan["new_hosted_calls"] == 0
    for job in plan["jobs"]:
        dataset, _ = h.load_native_prepared(ROOT / "data/tabular-full" / job["dataset"])
        expected = h.select_examples(dataset.train, dataset.labels, job["shots_per_class"], 42)
        assert job["training_example_ids"] == [r.id for r in expected]
        assert not set(job["training_example_ids"]) & {r.id for r in dataset.test + dataset.validation}


def test_source_audit_detects_changed_renderer_metrics_and_ids(tmp_path):
    dataset = PreparedDataset("breast_cancer", ["a", "b"], [], [],
                              [Row("test:0", "radius=10", 0), Row("test:1", "radius=20", 1)])
    provider = tiny_provider()
    with patch.object(h.providers, "build_provider", return_value=provider):
        record = h.run_model(dataset, provider.config, tmp_path, shots=0, bootstrap_samples=100)
    path = tmp_path / record["run_id"] / "run.json"
    original_predictions = path.with_name("predictions.jsonl").read_text()
    with patch.object(h.FixedChatClassifier, "_load", side_effect=AssertionError("Audit loaded model")):
        audited, predictions = h.audit_source_run(path, dataset, "smollm2", 0)
    assert audited == record and len(predictions) == 2
    changed = copy.deepcopy(record)
    changed["metrics"]["accuracy"] = 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="metrics"):
        h.audit_source_run(path, dataset, "smollm2", 0)
    path.write_text(json.dumps(record))
    lines = [json.loads(line) for line in original_predictions.splitlines()]
    lines[0]["metadata"]["rendered_chat_prompt_sha256"] = "0" * 64
    path.with_name("predictions.jsonl").write_text("\n".join(map(json.dumps, lines)))
    with pytest.raises(ValueError, match="provenance"):
        h.audit_source_run(path, dataset, "smollm2", 0)
    lines[0]["row_id"] = lines[1]["row_id"]
    path.with_name("predictions.jsonl").write_text("\n".join(map(json.dumps, lines)))
    with pytest.raises(ValueError, match="coverage/order"):
        h.audit_source_run(path, dataset, "smollm2", 0)


def test_public_download_explicitly_disables_hf_authentication(monkeypatch):
    import huggingface_hub
    preset = h.load_presets()["models"]["smollm2"]
    monkeypatch.setattr(h, "verify_tokenizer_assets", lambda *args: None)
    with patch.object(huggingface_hub, "snapshot_download") as download:
        h.download_public(preset, Path("unused"))
    assert download.call_args.kwargs["token"] is False
    assert download.call_args.kwargs["revision"] == preset["revision"]
