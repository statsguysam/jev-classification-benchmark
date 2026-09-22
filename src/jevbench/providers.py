"""Provider adapters. Importing this module neither loads weights nor calls an API."""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .types import Prediction, Row


class ProviderError(RuntimeError):
    """A credential-safe, actionable provider error."""


def parse_label(text: str, n_classes: int) -> int:
    """Accept only a complete numeric class ID; never salvage an explanation."""
    if not isinstance(text, str) or not re.fullmatch(r"0|[1-9][0-9]*", text.strip()):
        raise ProviderError("invalid_output: expected only a numeric class id")
    value = int(text.strip())
    if not 0 <= value < n_classes:
        raise ProviderError("invalid_output: class id outside the allowed range")
    return value


def validate_probabilities(values: Any, n_classes: int) -> list[float]:
    if not isinstance(values, dict) or set(values) != {str(i) for i in range(n_classes)}:
        raise ProviderError("invalid_output: missing or extra class probabilities")
    probabilities = []
    for i in range(n_classes):
        value = values[str(i)]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProviderError("invalid_output: probability is not a number")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ProviderError("invalid_output: probability outside [0,1]")
        probabilities.append(float(value))
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ProviderError("invalid_output: probabilities do not sum to one")
    return probabilities


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("HTTP redirect refused; configure the final API base URL")


def _post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    # Never include response bodies, credentials, or URLs with query keys in errors.
    request = urllib.request.Request(
        url, data=json.dumps(body, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"http_error: status={exc.code}; no automatic retry") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ProviderError("network_error: connection failed or timed out; no automatic retry") from None
    except (ValueError, UnicodeError):
        raise ProviderError("invalid_output: API returned invalid JSON") from None
    if not isinstance(result, dict):
        raise ProviderError("invalid_output: API response is not an object")
    return result


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class HTTPClassifier:
    """One request per prediction, without retries or implicit paid calls."""

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.model_id = config.get("model")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("provider config requires a nonempty model")
        self.timeout = float(config.get("timeout", 120))
        self.max_output_tokens = int(config.get("max_output_tokens", 32))
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.max_output_tokens <= 0:
            raise ValueError("timeout and max_output_tokens must be positive")

    def _key(self, default_env: str) -> str | None:
        env = self.config.get("api_key_env", default_env)
        if env is None:  # Explicitly opt out for a locally served API.
            return None
        key = os.environ.get(env)
        if not key:
            raise ProviderError(f"missing_credential: set environment variable {env}")
        return key

    def _url(self, default: str, endpoint: str) -> str:
        base = str(self.config.get("base_url", default)).rstrip("/")
        parsed = urllib.parse.urlparse(base)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ProviderError("invalid_config: base_url cannot contain credentials, query, or fragment")
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ProviderError("invalid_config: remote API base_url must use HTTPS")
        return base + endpoint

    def predict(self, row: Row, labels: list[str], prompt: str) -> Prediction:
        started = time.perf_counter()
        self._last_input_tokens = self._last_output_tokens = None
        self._last_metadata = {}
        try:
            if len(labels) < 2:
                raise ProviderError("invalid_config: classification needs at least two labels")
            prediction = self._predict(row.id, labels, prompt)
        except ProviderError as exc:
            prediction = Prediction(row.id, None, error=str(exc), input_tokens=self._last_input_tokens,
                                    output_tokens=self._last_output_tokens, metadata=dict(self._last_metadata))
        except (KeyError, IndexError, TypeError, ValueError):
            prediction = Prediction(row.id, None, error="invalid_output: unexpected API response structure",
                                    input_tokens=self._last_input_tokens, output_tokens=self._last_output_tokens,
                                    metadata=dict(self._last_metadata))
        prediction.latency_s = time.perf_counter() - started
        prediction.metadata.setdefault("requested_model", self.model_id)
        return prediction

    def _record_usage(self, result, usage, input_key, output_key, model_key="model"):
        """Preserve billable usage even when the generated answer is invalid."""
        self._last_metadata = {"resolved_model": result.get(model_key)}
        if isinstance(usage, dict):
            self._last_input_tokens = _token_count(usage.get(input_key))
            self._last_output_tokens = _token_count(usage.get(output_key))


class JevClassifier(HTTPClassifier):
    def _predict(self, row_id: str, labels: list[str], prompt: str) -> Prediction:
        key = self._key("TYPESAFE_API_KEY")
        if len(labels) > 255:
            raise ProviderError("invalid_config: Jev Choice accepts at most 255 options")
        result = _post_json(
            self._url("https://api.typesafe.ai/v1", "/systemone"),
            {"model": self.model_id, "state": prompt, "questions": {"classification": {
                "type": "choice",
                "instructions": "Choose the numeric class id for the final text in the state, following its classification task and examples.",
                "criteria": {str(i): label for i, label in enumerate(labels)},
            }}},
            {"Authorization": f"Bearer {key}"} if key else {}, self.timeout,
        )
        self._record_usage(result, result.get("usage"), "input_tokens", "output_tokens")
        answer = result["answers"]["classification"]
        label = parse_label(answer["choice"], len(labels))
        probabilities = validate_probabilities(answer["probabilities"], len(labels))
        if probabilities[label] < max(probabilities) - 1e-6:
            raise ProviderError("invalid_output: Jev choice disagrees with maximum probability")
        return Prediction(row_id, label, probabilities,
                          input_tokens=self._last_input_tokens,
                          output_tokens=self._last_output_tokens,
                          metadata={"resolved_model": result.get("model"),
                                    "probability_kind": "jev_choice_distribution"})


class OpenAIClassifier(HTTPClassifier):
    """Chat Completions adapter, also usable with a local vLLM server."""

    def _predict(self, row_id: str, labels: list[str], prompt: str) -> Prediction:
        key = self._key("OPENAI_API_KEY")
        limit_key = self.config.get("token_limit_parameter", "max_completion_tokens")
        if limit_key not in {"max_completion_tokens", "max_tokens"}:
            raise ProviderError("invalid_config: unsupported token_limit_parameter")
        body = {"model": self.model_id, "messages": [{"role": "user", "content": prompt}],
                limit_key: self.max_output_tokens}
        # Reasoning models can reject temperature/seed; only pass explicitly configured fields.
        for key_name in ("temperature", "seed", "reasoning_effort"):
            if key_name in self.config:
                body[key_name] = self.config[key_name]
        result = _post_json(self._url("https://api.openai.com/v1", "/chat/completions"), body,
                            {"Authorization": f"Bearer {key}"} if key else {}, self.timeout)
        self._record_usage(result, result.get("usage"), "prompt_tokens", "completion_tokens")
        self._last_metadata["system_fingerprint"] = result.get("system_fingerprint")
        choice = result["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ProviderError("invalid_output: response did not finish normally")
        if choice["message"].get("refusal"):
            raise ProviderError("invalid_output: model refused classification")
        label = parse_label(choice["message"]["content"], len(labels))
        return Prediction(row_id, label, input_tokens=self._last_input_tokens,
                          output_tokens=self._last_output_tokens,
                          metadata={"resolved_model": result.get("model"),
                                    "system_fingerprint": result.get("system_fingerprint"),
                                    "probability_kind": "unavailable"})


class AnthropicClassifier(HTTPClassifier):
    def _predict(self, row_id: str, labels: list[str], prompt: str) -> Prediction:
        key = self._key("ANTHROPIC_API_KEY")
        body = {"model": self.model_id, "max_tokens": self.max_output_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if "temperature" in self.config:
            body["temperature"] = self.config["temperature"]
        result = _post_json(self._url("https://api.anthropic.com/v1", "/messages"), body,
                            {"x-api-key": key or "", "anthropic-version": "2023-06-01"}, self.timeout)
        self._record_usage(result, result.get("usage"), "input_tokens", "output_tokens")
        if isinstance(result.get("usage"), dict):
            self._last_metadata.update({
                "cache_creation_input_tokens": result["usage"].get("cache_creation_input_tokens"),
                "cache_read_input_tokens": result["usage"].get("cache_read_input_tokens"),
            })
        if result.get("stop_reason") != "end_turn":
            raise ProviderError("invalid_output: response did not finish normally")
        content = "".join(block["text"] for block in result["content"] if block.get("type") == "text")
        label = parse_label(content, len(labels))
        return Prediction(row_id, label, input_tokens=self._last_input_tokens,
                          output_tokens=self._last_output_tokens,
                          metadata={**self._last_metadata, "probability_kind": "unavailable"})


class GeminiClassifier(HTTPClassifier):
    def _predict(self, row_id: str, labels: list[str], prompt: str) -> Prediction:
        key = self._key("GEMINI_API_KEY")
        generation = {"maxOutputTokens": self.max_output_tokens}
        if "temperature" in self.config:
            generation["temperature"] = self.config["temperature"]
        if "thinking_config" in self.config:
            generation["thinkingConfig"] = self.config["thinking_config"]
        model_path = urllib.parse.quote(self.model_id.removeprefix("models/"), safe="")
        result = _post_json(
            self._url("https://generativelanguage.googleapis.com/v1beta", f"/models/{model_path}:generateContent"),
            {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": generation},
            {"x-goog-api-key": key or ""}, self.timeout,
        )
        usage = result.get("usageMetadata", {})
        self._record_usage(result, usage, "promptTokenCount", "candidatesTokenCount", "modelVersion")
        thoughts = _token_count(usage.get("thoughtsTokenCount")) if isinstance(usage, dict) else None
        if thoughts is not None:
            self._last_output_tokens = (self._last_output_tokens or 0) + thoughts
        self._last_metadata["thought_tokens"] = thoughts
        candidate = result["candidates"][0]
        if candidate.get("finishReason") != "STOP":
            raise ProviderError("invalid_output: response did not finish normally")
        content = "".join(part["text"] for part in candidate["content"]["parts"]
                          if "text" in part and not part.get("thought", False))
        label = parse_label(content, len(labels))
        return Prediction(row_id, label, input_tokens=self._last_input_tokens,
                          output_tokens=self._last_output_tokens,
                          metadata={"resolved_model": result.get("modelVersion"),
                                    "probability_kind": "unavailable", "thought_tokens": thoughts})


def prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    """Use the model's own chat template consistently for training and evaluation."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        )
    return tokenizer.encode(prompt, add_special_tokens=True)


def response_token_ids(tokenizer, label: int) -> list[int]:
    """Include EOS so complete class strings define nonoverlapping events."""
    ids = tokenizer.encode(str(label), add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ProviderError("invalid_config: label scoring requires an EOS token")
    return ids + [tokenizer.eos_token_id]


def select_device(torch, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HuggingFaceClassifier:
    """Score complete numeric-label responses, conditional on one shared prompt."""

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.model_id = config["model"]
        self._model = self._tokenizer = self._torch = None
        self._adapter_metadata = None
        self.metadata = {"requested_model": self.model_id, "requested_revision": config.get("revision"),
                         "probability_kind": "label_sequence_likelihood_normalized",
                         "scoring": "sum_logp_numeric_id_plus_eos"}

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ProviderError("missing_dependency: install the neural optional dependencies") from None
        adapter_path = self.config.get("adapter_path")
        revision = self.config.get("revision")
        if adapter_path:
            metadata_path = Path(adapter_path) / "jevbench_training.json"
            if not metadata_path.is_file():
                raise ProviderError("invalid_config: adapter is missing jevbench_training.json")
            self._adapter_metadata = json.loads(metadata_path.read_text())
            if self._adapter_metadata["model_id"] != self.model_id:
                raise ProviderError("invalid_config: adapter base model mismatch")
            trained_revision = self._adapter_metadata.get("resolved_revision")
            if revision and trained_revision and revision != trained_revision:
                raise ProviderError("invalid_config: evaluation revision differs from adapter base revision")
            revision = trained_revision or revision
            self.metadata["adapter_path"] = str(adapter_path)
        kwargs = {"trust_remote_code": False}
        if revision:
            kwargs["revision"] = revision
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        device = select_device(torch, self.config.get("device", "auto"))
        dtype_name = self.config.get("dtype", "float32" if device == "cpu" else "float16")
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise ProviderError("invalid_config: dtype must be float32, float16, or bfloat16")
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=getattr(torch, dtype_name), **kwargs)
        self.metadata.update({"resolved_revision": getattr(model.config, "_commit_hash", None),
                              "device": device, "dtype": dtype_name})
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError:
                raise ProviderError("missing_dependency: install peft to load a LoRA adapter") from None
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        self._model, self._torch = model.to(device).eval(), torch
        self._device = device

    def predict(self, row: Row, labels: list[str], prompt: str) -> Prediction:
        self._load()  # Loading failures are configuration failures, not per-row results.
        started = time.perf_counter()
        try:
            if len(labels) < 2:
                raise ProviderError("invalid_config: classification needs at least two labels")
            if self._adapter_metadata and labels != self._adapter_metadata["labels"]:
                raise ProviderError("invalid_config: adapter label order differs from evaluation")
            context = prompt_token_ids(self._tokenizer, prompt)
            candidates = [response_token_ids(self._tokenizer, i) for i in range(len(labels))]
            maximum = int(self.config.get("max_context_tokens", 4096))
            model_limit = getattr(self._model.config, "max_position_embeddings", maximum)
            maximum = min(maximum, model_limit)
            if not context or len(context) + max(map(len, candidates)) > maximum:
                raise ProviderError("context_length: full prompt and label exceed max_context_tokens; no truncation")
            scores = []
            torch = self._torch
            with torch.inference_mode():
                for candidate in candidates:
                    ids = torch.tensor([context + candidate], dtype=torch.long, device=self._device)
                    logits = self._model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
                    # Position t predicts token t+1; only label and EOS positions contribute.
                    log_probs = torch.log_softmax(logits[0, len(context)-1:-1].float(), dim=-1)
                    target = ids[0, len(context):]
                    score = log_probs.gather(1, target.unsqueeze(1)).sum().item()
                    scores.append(score)
            peak = max(scores)
            weights = [math.exp(value - peak) for value in scores]
            total = sum(weights)
            if not math.isfinite(total) or total <= 0:
                raise ProviderError("invalid_output: model produced nonfinite likelihoods")
            probabilities = [value / total for value in weights]
            label = max(range(len(labels)), key=probabilities.__getitem__)
            return Prediction(row.id, label, probabilities, latency_s=time.perf_counter()-started,
                              input_tokens=len(context), output_tokens=0,
                              metadata={**self.metadata, "candidate_tokens_scored": sum(map(len, candidates)),
                                        "context_forward_passes": len(candidates)})
        except ProviderError as exc:
            return Prediction(row.id, None, error=str(exc), latency_s=time.perf_counter()-started,
                              metadata=dict(self.metadata))


def build_provider(config: dict[str, Any]):
    factories = {"jev": JevClassifier, "typesafe": JevClassifier,
                 "openai": OpenAIClassifier, "openai_compatible": OpenAIClassifier,
                 "anthropic": AnthropicClassifier, "gemini": GeminiClassifier,
                 "huggingface": HuggingFaceClassifier, "hf": HuggingFaceClassifier}
    name = config.get("provider")
    if name not in factories:
        raise ValueError(f"Unknown provider {name!r}; expected one of {sorted(factories)}")
    if not isinstance(config.get("model"), str) or not config["model"]:
        raise ValueError("provider config requires a nonempty model")
    return factories[name](config)
