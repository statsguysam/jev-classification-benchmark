"""Fixed-budget, response-only LoRA training for downloadable causal models."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

from .providers import ProviderError, prompt_token_ids, response_token_ids, select_device
from .types import PreparedDataset, Row


def encode_training_example(tokenizer, prompt: str, label: int, max_length: int) -> dict:
    context = prompt_token_ids(tokenizer, prompt)
    response = response_token_ids(tokenizer, label)
    if not context or len(context) + len(response) > max_length:
        raise ValueError("Full training prompt and response exceed max_length; no truncation is applied")
    ids = context + response
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(context) + response}


def pad_examples(examples: list[dict], pad_token_id: int) -> dict:
    """Right padding with ignored targets, including ignored prompt targets."""
    width = max(len(example["input_ids"]) for example in examples)
    return {
        "input_ids": [example["input_ids"] + [pad_token_id] * (width-len(example["input_ids"])) for example in examples],
        "attention_mask": [example["attention_mask"] + [0] * (width-len(example["input_ids"])) for example in examples],
        "labels": [example["labels"] + [-100] * (width-len(example["input_ids"])) for example in examples],
    }


def _training_rows(dataset: PreparedDataset, train_per_class: int | None, seed: int) -> list[Row]:
    # Deliberately never accesses dataset.test, even for metadata.
    if train_per_class is None:
        rows = list(dataset.train)
    else:
        if train_per_class <= 0:
            raise ValueError("train_per_class must be positive")
        from .prompts import select_examples
        rows = select_examples(dataset.train, dataset.labels, train_per_class, seed)
    if not rows:
        raise ValueError("LoRA requires at least one training row")
    if {row.label for row in rows} != set(range(len(dataset.labels))):
        raise ValueError("LoRA training must contain every class")
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("Duplicate training row IDs")
    return rows


def train_lora(
    dataset: PreparedDataset,
    model_id: str,
    output_dir: str | Path,
    seed: int = 42,
    train_per_class: int | None = None,
    *,
    revision: str | None = None,
    epochs: int = 1,
    learning_rate: float = 2e-4,
    max_length: int = 1024,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    r: int = 8,
    lora_alpha: int = 16,
    device: str = "auto",
    dtype: str | None = None,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = True,
    target_modules: list[str] | str = "all-linear",
    max_steps: int | None = None,
    evaluate_validation: bool = False,
    prompt_builder: Callable[[Row, list[str]], str] | None = None,
) -> dict:
    """Train only on train rows; optionally report final validation loss.

    The fixed final checkpoint is always used. No early stopping or checkpoint
    selection occurs. Matched-budget runs forbid validation use, preserving the
    same labeled-data budget as few-shot prompting. No held-out test rows or
    labels are read here. `max_steps` is an upfront optimizer-step limit, intended
    for technical smoke runs, not a post-hoc result-based stopping decision.
    """
    if train_per_class is not None and evaluate_validation:
        raise ValueError("Matched-label-budget LoRA cannot use validation labels")
    for name, value in {"epochs": epochs, "batch_size": batch_size,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "max_length": max_length, "r": r, "lora_alpha": lora_alpha}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not model_id or model_id.startswith(("jev", "gpt-", "claude-", "gemini-")):
        raise ValueError("LoRA requires downloadable Hugging Face causal-model weights")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("output_dir must be empty to avoid overwriting a checkpoint")
    rows = _training_rows(dataset, train_per_class, seed)
    validation = list(dataset.validation) if evaluate_validation else []
    if validation and ({row.id for row in rows} & {row.id for row in validation}):
        raise ValueError("Training and validation row IDs overlap")
    if prompt_builder is None:
        from .prompts import build_prompt
        prompt_builder = build_prompt
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    except ImportError:
        raise ProviderError("Install the neural optional dependencies before LoRA training") from None

    random.seed(seed)
    set_seed(seed)
    actual_device = select_device(torch, device)
    if dtype is None:
        if actual_device.startswith("cuda"):
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        else:
            dtype = "float32"
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be float32, float16, or bfloat16")
    if load_in_4bit and not actual_device.startswith("cuda"):
        raise ValueError("4-bit training requires a CUDA device; use ordinary LoRA on CPU/MPS")
    load_kwargs = {"trust_remote_code": False}
    if revision:
        load_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("Training requires a tokenizer with EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_prompts = [prompt_builder(row, dataset.labels) for row in rows]
    train_examples = [encode_training_example(tokenizer, prompt, row.label, max_length)
                      for row, prompt in zip(rows, train_prompts)]
    validation_examples = [encode_training_example(tokenizer, prompt_builder(row, dataset.labels), row.label, max_length)
                           for row in validation]

    weight_kwargs = dict(load_kwargs, torch_dtype=getattr(torch, dtype))
    if load_in_4bit:
        weight_kwargs.update({
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
            ), "device_map": {"": actual_device},
        })
    model = AutoModelForCausalLM.from_pretrained(model_id, **weight_kwargs)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    model_limit = getattr(model.config, "max_position_embeddings", max_length)
    if any(len(example["input_ids"]) > model_limit for example in train_examples + validation_examples):
        raise ValueError("Training prompt exceeds the model context window")
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    else:
        model = model.to(actual_device)
    model.config.use_cache = False
    lora_config = LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_modules,
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    use_cuda_amp = actual_device.startswith("cuda") and dtype in {"float16", "bfloat16"}
    use_grad_scaler = use_cuda_amp and dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    def autocast():
        return torch.autocast("cuda", dtype=getattr(torch, dtype)) if use_cuda_amp else nullcontext()

    def collate(examples):
        padded = pad_examples(examples, tokenizer.pad_token_id)
        return {key: torch.tensor(value, dtype=torch.long, device=actual_device) for key, value in padded.items()}

    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(train_examples, batch_size=batch_size, shuffle=True,
                                         generator=generator, collate_fn=collate)
    steps = 0
    skipped_steps = 0
    training_history = []
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_batches = 0.0, 0
        for batch_index, batch in enumerate(loader):
            # Use the actual final accumulation-window size instead of silently
            # underweighting a remainder at the end of an epoch.
            window_start = (batch_index // gradient_accumulation_steps) * gradient_accumulation_steps
            window_size = min(gradient_accumulation_steps, len(loader)-window_start)
            with autocast():
                loss = model(**batch).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss is nonfinite; checkpoint was not saved")
            epoch_loss += loss.detach().float().item()
            epoch_batches += 1
            scaler.scale(loss / window_size).backward()
            if (batch_index + 1) % gradient_accumulation_steps == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
        training_history.append({"epoch": epoch+1, "mean_batch_loss": epoch_loss/epoch_batches})
        if max_steps is not None and steps >= max_steps:
            break
    training_s = time.perf_counter()-start
    validation_loss = None
    if validation_examples:
        model.eval()
        loss_sum, target_count = 0.0, 0
        with torch.inference_mode():
            for index in range(0, len(validation_examples), batch_size):
                batch = collate(validation_examples[index:index+batch_size])
                count = (batch["labels"][:, 1:] != -100).sum().item()
                with autocast():
                    loss_sum += model(**batch).loss.item() * count
                target_count += count
        validation_loss = loss_sum / target_count
    metadata = {
        "model_id": model_id, "requested_revision": revision, "resolved_revision": resolved_revision,
        "dataset": dataset.name, "labels": list(dataset.labels), "seed": seed,
        "train_per_class": train_per_class, "training_rows": len(rows),
        "training_row_ids": [row.id for row in rows], "validation_rows": len(validation),
        "test_accessed": False, "selection": "fixed_final_checkpoint",
        "supervision": "numeric_class_id_plus_eos_response_only", "epochs": epochs,
        "optimizer_steps": steps-skipped_steps, "attempted_optimizer_steps": steps,
        "skipped_optimizer_steps": skipped_steps, "max_steps": max_steps, "learning_rate": learning_rate,
        "batch_size": batch_size, "gradient_accumulation_steps": gradient_accumulation_steps,
        "r": r, "lora_alpha": lora_alpha, "target_modules": target_modules,
        "max_length": max_length, "load_in_4bit": load_in_4bit, "dtype": dtype,
        "device": actual_device, "gradient_checkpointing": gradient_checkpointing,
        "cuda_autocast": use_cuda_amp, "gradient_scaling": use_grad_scaler,
        "training_s": training_s, "training_history": training_history,
        "validation_loss": validation_loss,
        "prompt_sha256": hashlib.sha256(json.dumps(train_prompts, ensure_ascii=False).encode()).hexdigest(),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft"]},
    }
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    (destination / "jevbench_training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
