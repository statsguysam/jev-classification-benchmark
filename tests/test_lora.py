import pytest

from jevbench.lora import _training_rows, encode_training_example, pad_examples, train_lora
from jevbench.prompts import select_examples
from jevbench.types import Row


class NoTestAccessDataset:
    name = "toy"
    labels = ["negative", "positive"]
    train = [Row(f"train-{i}", f"text {i}", i % 2) for i in range(20)]
    validation = []

    @property
    def test(self):
        raise AssertionError("Training must not access test data")


class Tokenizer:
    eos_token_id = 9
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return [int(text) + 3] if text in {"0", "1"} else [1, 2, 3]


def test_only_response_tokens_are_supervised():
    encoded = encode_training_example(Tokenizer(), "shared prompt", 1, 10)
    assert encoded["input_ids"] == [1, 2, 3, 4, 9]
    assert encoded["labels"] == [-100, -100, -100, 4, 9]
    assert encoded["attention_mask"] == [1, 1, 1, 1, 1]


def test_full_prompt_overflow_is_error():
    with pytest.raises(ValueError, match="no truncation"):
        encode_training_example(Tokenizer(), "shared prompt", 1, 4)


def test_padding_is_never_a_training_target():
    first = encode_training_example(Tokenizer(), "shared prompt", 0, 10)
    second = {"input_ids": [1, 9], "attention_mask": [1, 1], "labels": [-100, 9]}
    batch = pad_examples([first, second], pad_token_id=9)
    assert batch["input_ids"][1] == [1, 9, 9, 9, 9]
    assert batch["attention_mask"][1] == [1, 1, 0, 0, 0]
    assert batch["labels"][1] == [-100, 9, -100, -100, -100]


def test_matched_lora_has_exact_same_examples_as_few_shot():
    dataset = NoTestAccessDataset()
    selected = _training_rows(dataset, 3, 42)
    assert selected == select_examples(dataset.train, dataset.labels, 3, 42)
    assert len(selected) == 6
    assert _training_rows(dataset, None, 42) == dataset.train


def test_matched_budget_rejects_extra_validation_labels_before_loading_model(tmp_path):
    with pytest.raises(ValueError, match="cannot use validation labels"):
        train_lora(NoTestAccessDataset(), "Qwen/test", tmp_path, train_per_class=1, evaluate_validation=True)


def test_refuses_to_overwrite_checkpoint(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="empty"):
        train_lora(NoTestAccessDataset(), "Qwen/test", tmp_path)


@pytest.mark.parametrize("model_id", ["jev-1.13.0", "gpt-example", "claude-example", "gemini-example"])
def test_proprietary_models_cannot_be_lora_finetuned(model_id, tmp_path):
    with pytest.raises(ValueError, match="downloadable"):
        train_lora(NoTestAccessDataset(), model_id, tmp_path)


def test_missing_class_in_training_is_rejected():
    dataset = NoTestAccessDataset()
    dataset.train = [Row("a", "only one class", 0)]
    with pytest.raises(ValueError, match="every class"):
        _training_rows(dataset, None, 42)


def test_real_tiny_lora_training_without_downloads_or_test_access(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    class SavingTokenizer(Tokenizer):
        pad_token_id = 9
        eos_token = "EOS"

        def save_pretrained(self, path):
            (path / "test_tokenizer.txt").write_text("saved")

    config = LlamaConfig(vocab_size=10, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=32)
    config._commit_hash = "a" * 40
    model = LlamaForCausalLM(config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SavingTokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    dataset = NoTestAccessDataset()
    metadata = train_lora(dataset, "toy/llama", tmp_path, train_per_class=1,
                          max_steps=1, batch_size=2, gradient_accumulation_steps=1,
                          gradient_checkpointing=False, device="cpu", r=2,
                          prompt_builder=lambda row, labels: "shared prompt")
    assert metadata["training_rows"] == 2
    assert metadata["test_accessed"] is False
    assert metadata["validation_rows"] == 0
    assert metadata["optimizer_steps"] == 1
    assert metadata["resolved_revision"] == "a" * 40
    assert metadata["trainable_parameters"] > 0
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert (tmp_path / "jevbench_training.json").is_file()
