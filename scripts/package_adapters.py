"""Package pinned Qwen classification LoRA adapters, never pretrained base weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import tempfile
from pathlib import Path
import zipfile

SUPPORTED = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen3-4B-Instruct-2507": "cdbee75f17c01a7cc42f958dc650907174af0554",
}
SUPPORTED_DATASETS = {"sst2", "trec", "titanic", "breast_cancer", "wine"}
LICENSE_SHA256 = "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
# Verified against complete HF file listings at these immutable revisions on
# 2026-09-22. Neither distributes a separate NOTICE/NOTICE.txt file.
UPSTREAM_NOTICES = {model: {} for model in SUPPORTED}
ALLOWED = {"adapter_config.json", "adapter_model.safetensors", "jevbench_training.json",
           "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
           "merges.txt", "added_tokens.json", "README.md", "chat_template.jinja"}
REQUIRED = {"adapter_config.json", "adapter_model.safetensors", "jevbench_training.json"}
LORA_KEY = re.compile(r"base_model\.model\.model\.layers\.[0-9]+\.(?:self_attn|mlp)\."
                      r"(?:q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)\.lora_([AB])(?:\.default)?\.weight")
SENSITIVE_KEY = re.compile(r"(?i)^(?:api[_-]?key|access[_-]?token|authorization|password|secret|credentials)$")
TOKEN_PATTERN = re.compile(rb"(?:\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|\bhf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reject_secrets(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if SENSITIVE_KEY.fullmatch(str(key)) and child not in (None, "", {}, []):
                raise ValueError("Sensitive field in adapter metadata/config; refusing packaging")
            reject_secrets(child)
    elif isinstance(value, list):
        for child in value:
            reject_secrets(child)


def validate_lora_weights(path, rank):
    """Inspect safetensors without importing torch or loading the tensor data."""
    size = path.stat().st_size
    if not 8 < size <= 512_000_000:
        raise ValueError("Adapter weights are missing or exceed the 512 MB packaging limit")
    with path.open("rb") as file:
        length = struct.unpack("<Q", file.read(8))[0]
        if not 2 <= length <= min(size-8, 8_000_000):
            raise ValueError("Invalid safetensors header length")
        header = json.loads(file.read(length))
    if not isinstance(header, dict) or header.get("__metadata__", {}) not in ({}, {"format": "pt"}):
        raise ValueError("Unsupported safetensors metadata")
    intervals, keys = [], []
    for key, tensor in header.items():
        if key == "__metadata__":
            continue
        match = LORA_KEY.fullmatch(key)
        if not match or not isinstance(tensor, dict):
            raise ValueError("Only Qwen LoRA A/B tensors may be packaged; base or saved full-module weights are forbidden")
        shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
        width = {"F32": 4, "F16": 2, "BF16": 2}.get(tensor.get("dtype"))
        if (not width or not isinstance(shape, list) or len(shape) != 2
                or any(type(x) is not int or x <= 0 for x in shape)
                or shape[0 if match.group(1) == "A" else 1] != rank
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int or x < 0 for x in offsets)
                or offsets[1]-offsets[0] != math.prod(shape)*width):
            raise ValueError("Invalid LoRA tensor shape, rank, dtype or offsets")
        intervals.append(tuple(offsets))
        keys.append(key)
    if not keys:
        raise ValueError("No LoRA tensors in adapter")
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError("Safetensors data contain gaps or overlapping tensors")
        end = stop
    if end != size-8-length:
        raise ValueError("Unexpected trailing/missing weight data")
    return {"tensor_count": len(keys), "lora_rank": rank, "weights_bytes": size}


def _upstream_files(model, upstream_root, legacy_license=None):
    revision = SUPPORTED[model]
    slug = model.split("/")[-1]
    folder = Path(upstream_root)/slug
    license_path = Path(legacy_license) if legacy_license and model == "Qwen/Qwen2.5-0.5B-Instruct" else folder/"LICENSE"
    if license_path.is_symlink() or not license_path.is_file():
        raise ValueError(f"Missing regular pinned upstream license: {license_path}")
    content = license_path.read_bytes()
    if digest(content) != LICENSE_SHA256:
        raise ValueError("Upstream license differs from the verified pinned Apache-2.0 LICENSE")
    files = {f"upstream/{slug}/LICENSE": content}
    for name, expected in UPSTREAM_NOTICES[model].items():
        source = folder/name
        if source.is_symlink() or not source.is_file() or digest(source.read_bytes()) != expected:
            raise ValueError("Missing or incorrect upstream NOTICE")
        files[f"upstream/{slug}/{name}"] = source.read_bytes()
    provenance = {"model_id": model, "revision": revision, "license": "Apache-2.0",
                  "license_source": f"https://huggingface.co/{model}/resolve/{revision}/LICENSE", "license_sha256": LICENSE_SHA256,
                  "snapshot_file_listing": f"https://huggingface.co/api/models/{model}/revision/{revision}",
                  "verified_on": "2026-09-22", "separate_upstream_notice_files": list(UPSTREAM_NOTICES[model]),
                  "notice_status": "No separate upstream NOTICE file is present at either allowlisted snapshot"}
    files[f"upstream/{slug}/PROVENANCE.json"] = (json.dumps(provenance, indent=2)+"\n").encode()
    return files, provenance


def package(adapters, output, *, upstream_root=Path("artifacts/upstream"), base_license=None):
    """Validate all inputs before atomically publishing one deterministic ZIP."""
    output = Path(output)
    payloads, entries, upstream = {}, {}, {}
    if not adapters:
        raise ValueError("Supply at least one adapter folder")
    for folder in map(Path, adapters):
        if folder.is_symlink() or not folder.is_dir() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", folder.name):
            raise ValueError("Adapter inputs must be regular directories with safe unique names")
        if folder.name in entries:
            raise ValueError("Duplicate adapter folder name")
        children = list(folder.iterdir())
        names = {file.name for file in children}
        if not REQUIRED <= names or names-ALLOWED:
            raise ValueError(f"Missing required or non-allowlisted adapter files in {folder.name}: {sorted((REQUIRED-names) | (names-ALLOWED))}")
        if any(file.is_symlink() or not file.is_file() for file in children):
            raise ValueError("Symlinks, subdirectories and nonregular adapter files are forbidden")
        metadata = json.loads((folder/"jevbench_training.json").read_text())
        config = json.loads((folder/"adapter_config.json").read_text())
        reject_secrets(metadata)
        reject_secrets(config)
        model, revision = metadata.get("model_id"), metadata.get("resolved_revision")
        if model not in SUPPORTED or revision != SUPPORTED[model]:
            raise ValueError("Adapter base model/revision is not allowlisted")
        if metadata.get("dataset") not in SUPPORTED_DATASETS or metadata.get("test_accessed") is not False:
            raise ValueError("Only allowlisted text/tabular training-only adapters may be packaged")
        rank = metadata.get("r")
        if type(rank) is not int or not 1 <= rank <= 64 or config.get("r") != rank:
            raise ValueError("Adapter rank is missing, mismatched or outside pilot bounds")
        if (config.get("base_model_name_or_path") != model or config.get("peft_type") != "LORA"
                or config.get("task_type") != "CAUSAL_LM" or config.get("modules_to_save")
                or config.get("bias", "none") != "none" or config.get("rank_pattern")):
            raise ValueError("PEFT config is inconsistent or saves full model modules")
        weights = validate_lora_weights(folder/"adapter_model.safetensors", rank)
        file_hashes = {}
        for file in sorted(children):
            if file.name != "adapter_model.safetensors" and file.stat().st_size > 32_000_000:
                raise ValueError("Unexpectedly large non-weight file")
            content = file.read_bytes()
            if file.name not in {"adapter_model.safetensors", "tokenizer.json", "vocab.json", "merges.txt"}:
                if TOKEN_PATTERN.search(content):
                    raise ValueError("Possible credential/private key in adapter text; refusing packaging")
                if file.suffix == ".json":
                    reject_secrets(json.loads(content))
            member = f"{folder.name}/{file.name}"
            payloads[member] = content
            file_hashes[file.name] = digest(content)
        notice = (f"Downstream benchmark adapter for {model}\nBase revision: {revision}\n"
                  f"Training dataset: {metadata['dataset']}\n"
                  "The LoRA weights are newly trained downstream modifications; this is not an official Qwen release.\n"
                  "The PEFT configuration, training provenance and saved tokenizer files (when present) are exported by this benchmark.\n"
                  "All upstream copyright/license notices are retained under upstream/.\n"
                  "Pretrained base weights and raw dataset records are not included.\n")
        payloads[f"{folder.name}/DOWNSTREAM_NOTICE.txt"] = notice.encode()
        file_hashes["DOWNSTREAM_NOTICE.txt"] = digest(notice.encode())
        if model not in upstream:
            license_files, provenance = _upstream_files(model, upstream_root, base_license)
            payloads.update(license_files)
            upstream[model] = provenance
        entries[folder.name] = {"model_id": model, "base_revision": revision,
            "dataset": metadata["dataset"], "train_per_class": metadata["train_per_class"],
            "seed": metadata["seed"], "training_rows": metadata["training_rows"],
            "prompt_sha256": metadata["prompt_sha256"], "training_s": metadata.get("training_s"),
            "training_provenance": f"{folder.name}/jevbench_training.json", "weights": weights,
            "files_sha256": file_hashes}
    payloads["NOTICE.txt"] = ("Downstream Qwen classification LoRA adapters for text and serialized tabular tasks\n"
        "These are modified benchmark artifacts, not official Qwen releases.\n"
        "Applicable pinned upstream Apache-2.0 licenses, copyright statements and provenance are under upstream/.\n"
        "Separate upstream NOTICE files are included when present at an allowlisted revision.\n"
        "No pretrained base weights, raw datasets or environment credentials are included.\n"
        "Extract adapter folders beneath models/. Load the exact base revision identified in MANIFEST.json.\n"
        "Training provenance is included; evaluation results and limitations are published separately by the benchmark.\n").encode()
    manifest = {"schema_version": 2, "kind": "PEFT LoRA adapters only; separately downloaded pinned base weights required",
                "packager_sha256": digest(Path(__file__).read_bytes()), "adapters": entries, "upstream": upstream,
                "files_sha256": {name: digest(content) for name, content in sorted(payloads.items())}}
    payloads["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True)+"\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip.tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(payloads.items()):
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapters", nargs="+", type=Path)
    parser.add_argument("--upstream-root", type=Path, default=Path("artifacts/upstream"))
    parser.add_argument("--base-license", type=Path, help="Compatibility option: pinned Qwen2.5-0.5B LICENSE only")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen-pilot-adapters.zip"))
    args = parser.parse_args()
    print(package(args.adapters, args.output, upstream_root=args.upstream_root, base_license=args.base_license))


if __name__ == "__main__":
    main()
