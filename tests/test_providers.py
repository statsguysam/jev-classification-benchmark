import io
import json
import math
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error

import pytest

from jevbench.providers import (
    ProviderError, HuggingFaceClassifier, _post_json, build_provider, parse_label,
    prompt_token_ids, response_token_ids, validate_probabilities,
)
from jevbench.types import Row


class UnlabeledRow:
    id = "test-row"
    text = "target input"

    @property
    def label(self):
        raise AssertionError("Providers must never access the test label")


@pytest.mark.parametrize("text,label", [("0", 0), (" 1\n", 1), ("10", 10)])
def test_parse_complete_id(text, label):
    assert parse_label(text, 11) == label


@pytest.mark.parametrize("text", ["01", "-1", "1.0", "1 because positive", '{"label":1}', "1\n0", "", "12"])
def test_invalid_outputs_are_not_salvaged(text):
    with pytest.raises(ProviderError):
        parse_label(text, 12)


@pytest.mark.parametrize("values", [
    {"0": .5}, {"0": .4, "1": .4}, {"0": float("nan"), "1": 0},
    {"0": 1.1, "1": -.1}, {"0": True, "1": 0}, {"0": .5, "1": .5, "2": 0},
])
def test_invalid_distributions_fail(values):
    with pytest.raises(ProviderError):
        validate_probabilities(values, 2)


def test_jev_uses_full_prompt_and_maps_numeric_probabilities(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    result = {"model": "jev-pinned", "answers": {"classification": {
        "choice": "1", "probabilities": {"1": .9, "0": .1}, "confidence": .7}},
        "usage": {"input_tokens": 50, "output_tokens": 8}}
    with patch("jevbench.providers._post_json", return_value=result) as post:
        prediction = build_provider({"provider": "jev", "model": "jev-latest"}).predict(
            UnlabeledRow(), ["bad", "good"], "full shared prompt with examples")
    assert prediction.label == 1
    assert prediction.probabilities == [.1, .9]
    assert prediction.metadata["resolved_model"] == "jev-pinned"
    assert prediction.input_tokens == 50
    assert post.call_args.args[0] == "https://api.typesafe.ai/v1/systemone"
    body = post.call_args.args[1]
    assert body["state"] == "full shared prompt with examples"
    assert body["questions"]["classification"]["criteria"] == {"0": "bad", "1": "good"}


def test_jev_rejects_choice_inconsistent_with_distribution(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    result = {"answers": {"classification": {"choice": "0", "probabilities": {"0": .1, "1": .9}}}}
    with patch("jevbench.providers._post_json", return_value=result):
        prediction = build_provider({"provider": "jev", "model": "jev-pinned"}).predict(UnlabeledRow(), ["a", "b"], "p")
    assert prediction.label is None
    assert "disagrees" in prediction.error


@pytest.mark.parametrize("provider,env,result,token_count", [
    ("openai", "OPENAI_API_KEY", {"model": "snapshot", "choices": [{"finish_reason": "stop", "message": {"content": "1"}}],
                                "usage": {"prompt_tokens": 20, "completion_tokens": 1}}, 1),
    ("anthropic", "ANTHROPIC_API_KEY", {"model": "snapshot", "stop_reason": "end_turn", "content": [{"type": "text", "text": "1"}],
                                      "usage": {"input_tokens": 20, "output_tokens": 1}}, 1),
    ("gemini", "GEMINI_API_KEY", {"modelVersion": "snapshot", "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "1"}]}}],
                                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 1, "thoughtsTokenCount": 5}}, 6),
])
def test_hosted_no_invented_probabilities(provider, env, result, token_count, monkeypatch):
    monkeypatch.setenv(env, "test-key")
    with patch("jevbench.providers._post_json", return_value=result) as post:
        prediction = build_provider({"provider": provider, "model": "configured-model", "max_output_tokens": 99}).predict(
            UnlabeledRow(), ["a", "b"], "exact shared prompt")
    assert prediction.label == 1
    assert prediction.probabilities is None
    assert prediction.output_tokens == token_count
    assert prediction.metadata["resolved_model"] == "snapshot"
    assert "exact shared prompt" in json.dumps(post.call_args.args[1])
    assert "target input" not in json.dumps(post.call_args.args[1])


def test_openai_records_truncation_as_failure():
    with patch("jevbench.providers._post_json", return_value={"model": "snapshot", "usage": {"prompt_tokens": 90, "completion_tokens": 12},
                                                               "choices": [{"finish_reason": "length", "message": {"content": "1"}}]}):
        result = build_provider({"provider": "openai_compatible", "model": "local", "api_key_env": None,
                                 "base_url": "http://localhost:8000/v1", "token_limit_parameter": "max_tokens"}).predict(
            UnlabeledRow(), ["a", "b"], "prompt")
    assert result.label is None
    assert "did not finish" in result.error
    assert result.input_tokens == 90
    assert result.output_tokens == 12
    assert result.metadata["resolved_model"] == "snapshot"


def test_invalid_paid_answer_keeps_usage_but_not_response_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = build_provider({"provider": "anthropic", "model": "configured"})
    response = {"model": "resolved", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "SENSITIVE RAW RESPONSE"}],
                "usage": {"input_tokens": 20, "output_tokens": 3}}
    with patch("jevbench.providers._post_json", return_value=response):
        prediction = provider.predict(UnlabeledRow(), ["a", "b"], "prompt")
    assert prediction.label is None
    assert prediction.input_tokens == 20
    assert prediction.output_tokens == 3
    assert prediction.metadata["resolved_model"] == "resolved"
    assert "SENSITIVE" not in repr(prediction)
    with patch("jevbench.providers._post_json", side_effect=ProviderError("network_error")):
        next_prediction = provider.predict(UnlabeledRow(), ["a", "b"], "prompt")
    assert next_prediction.input_tokens is None
    assert next_prediction.output_tokens is None
    assert "resolved_model" not in next_prediction.metadata


def test_missing_credentials_do_not_call_api(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with patch("jevbench.providers._post_json") as post:
        result = build_provider({"provider": "jev", "model": "jev"}).predict(UnlabeledRow(), ["a", "b"], "prompt")
    post.assert_not_called()
    assert result.label is None
    assert result.error == "missing_credential: set environment variable TYPESAFE_API_KEY"


def test_http_error_does_not_include_body_or_credentials():
    error = urllib.error.HTTPError("https://test.invalid/?key=SECRET", 401, "SECRET", {}, io.BytesIO(b"SECRET"))
    with patch("urllib.request.build_opener") as opener:
        opener.return_value.open.side_effect = error
        with pytest.raises(ProviderError, match="status=401") as exc:
            _post_json("https://test.invalid", {}, {"Authorization": "SECRET"}, 10)
    assert "SECRET" not in str(exc.value)


def test_no_credentials_sent_to_insecure_remote(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "SECRET")
    with patch("jevbench.providers._post_json") as post:
        result = build_provider({"provider": "openai", "model": "model", "base_url": "http://example.com/v1"}).predict(
            UnlabeledRow(), ["a", "b"], "prompt")
    assert "HTTPS" in result.error
    post.assert_not_called()


class ToyTokenizer:
    eos_token_id = 2
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return [3 + int(text)] if text in {"0", "1"} else [0, 1]


def test_label_scoring_includes_end_token():
    tokenizer = ToyTokenizer()
    assert response_token_ids(tokenizer, 1) == [4, 2]
    assert prompt_token_ids(tokenizer, "p") == [0, 1]


def test_local_likelihood_normalization_and_no_label_access():
    torch = pytest.importorskip("torch")

    class ToyModel:
        config = SimpleNamespace(max_position_embeddings=32)

        def __call__(self, input_ids, **kwargs):
            logits = torch.zeros((1, input_ids.shape[1], 5))
            logits[:, :, 4] = 2
            return SimpleNamespace(logits=logits)

    provider = HuggingFaceClassifier({"provider": "hf", "model": "toy"})
    provider._model = ToyModel()
    provider._tokenizer, provider._torch, provider._device = ToyTokenizer(), torch, "cpu"
    result = provider.predict(UnlabeledRow(), ["a", "b"], "prompt")
    assert result.label == 1
    assert result.probabilities[1] == pytest.approx(math.exp(2)/(1+math.exp(2)))
    assert result.input_tokens == 2
    assert result.metadata["candidate_tokens_scored"] == 4


def test_local_context_does_not_silently_truncate():
    provider = HuggingFaceClassifier({"provider": "hf", "model": "toy", "max_context_tokens": 3})
    provider._model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32))
    provider._tokenizer = ToyTokenizer()
    result = provider.predict(UnlabeledRow(), ["a", "b"], "prompt")
    assert result.label is None
    assert "context_length" in result.error


def test_local_construction_does_not_load_weights():
    provider = build_provider({"provider": "hf", "model": "not-downloaded"})
    assert provider._model is None
