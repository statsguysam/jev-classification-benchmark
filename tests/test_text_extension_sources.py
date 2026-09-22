"""Text reuse checks reject identity drift without model loading or corpus downloads."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import text_extension_sources as sources
from audit_cross_environment import AuditError
from compare_combined_pilot import JEV_BASE_URL, JEV_ENDPOINT, JEV_RESOLVED_MODELS
from jevbench.data import _split_audit, load_prepared
from jevbench.metrics import evaluate
from jevbench.prompts import build_prompt, select_examples
from jevbench.runner import digest
from jevbench.types import Prediction, Row


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "ROOT", tmp_path)
    monkeypatch.setattr(sources, "CONFIG", tmp_path / "configs/text_extension_sources.json")
    monkeypatch.setattr(sources, "LOCAL", tmp_path / "results/text_extension/local")
    monkeypatch.setattr(sources, "SOURCES_DIR", tmp_path / "results/text_extension/sources")
    config = {"schema_version": 1, "seed": 42, "core_sha256": sources.FROZEN_CORE_SHA,
              "models": sources.MODELS, "direct_model": sources.DIRECT, "datasets": {}, "historical_sources": {}}
    for name, labels in (("sst2", ["negative", "positive"]), ("trec", ["abbreviation", "entity", "description", "human", "location", "number"])):
        content = {split: [Row(f"{name}:{split}:{label}:{i}", f"{name} {split} category {label} example {i}", label)
                           for label in range(len(labels)) for i in range(count)]
                   for split, count in (("train", 5), ("validation", 1), ("test", 2))}
        folder = tmp_path / "data/pilot" / name
        folder.mkdir(parents=True)
        hashes = {}
        for split, rows in content.items():
            path = folder / f"{split}.jsonl"
            path.write_text("".join(json.dumps(asdict(row)) + "\n" for row in rows))
            hashes[path.name] = sources.file_sha(path)
        manifest = {"schema_version": 1, "seed": 42, "name": name, "labels": labels,
                    "source": {"revision": "a" * 40}, "files_sha256": hashes,
                    "splits": {key: _split_audit(value) for key, value in content.items()},
                    "post_transform_overlap_removed": {"train": 0, "validation": 0},
                    "prepared_content_sha256": digest({key: [asdict(row) for row in rows] for key, rows in content.items()}),
                    "text_policy": {"kind": "character_prefix", "max_chars": 2000}}
        write_json(folder / "manifest.json", manifest)
        config["datasets"][name] = {"prepared_path": folder.relative_to(tmp_path).as_posix(),
            "hashes": {"manifest.json": sources.file_sha(folder / "manifest.json"), **hashes},
            "manifest_sha256": digest(manifest), "prepared_content_sha256": manifest["prepared_content_sha256"],
            "labels": labels, "sizes": {key: len(value) for key, value in content.items()}}
        dataset = load_prepared(folder)
        for model_key in ("qwen_small", "qwen_main", "luna", "astra", "jev"):
            for shots in (0, 4):
                spec = sources.DIRECT if model_key == "jev" else sources.MODELS[model_key]
                model_config = {**spec, "shots_per_class": shots}
                if model_key == "jev":
                    model_config.update({"base_url": JEV_BASE_URL, "budget_guard": {
                        "route": "OpenRouter", "endpoint": JEV_ENDPOINT, "native_protocol": "frozen_JevClassifier_Choice",
                        "response_model_allowlist": sorted(JEV_RESOLVED_MODELS),
                        "wrapper_sha256": sources.file_sha(Path(sources.__file__).with_name("run_openrouter_jev.py")),
                        "dataset": name, "seed": 42, "shots_per_class": shots}})
                run_dir = tmp_path / "results/history" / sources.condition_key(name, model_key, shots)
                source_manifest = copy.deepcopy(manifest)
                if model_key == "qwen_main":
                    source_manifest["post_transform_overlap_removed_ids"] = {"train": [], "validation": []}
                    source_manifest["splits_before_text_transform"] = copy.deepcopy(source_manifest["splits"])
                predictions = []
                for index, row in enumerate(dataset.test):
                    if model_key == "jev" and shots == 0 and index == 0:
                        predictions.append(Prediction(row.id, None, error="preserved provider failure"))
                        continue
                    values = [0.1 / (len(labels) - 1)] * len(labels)
                    values[row.label] = 0.9
                    metadata = {"requested_model": spec["model"], "resolved_model": spec["model"]}
                    if model_key == "jev":
                        metadata["openrouter"] = {"provider": "TypeSafe", "endpoint": JEV_ENDPOINT}
                    predictions.append(Prediction(row.id, row.label,
                        probabilities=None if spec["provider"] == "openai" else values, metadata=metadata))
                record = {"run_id": run_dir.name, "dataset": name, "labels": labels, "method": "few_shot" if shots else "zero_shot",
                    "config": model_config, "status": "complete", "seed": 42,
                    "manifest_sha256": digest(source_manifest), "dataset_manifest": source_manifest,
                    "test_ids_sha256": digest([row.id for row in dataset.test]),
                    "training_example_ids": [row.id for row in select_examples(dataset.train, labels, shots, 42)],
                    "implementation_sha256": sources.FROZEN_CORE_SHA,
                    "prompt_template_sha256": digest(build_prompt.__code__.co_consts.__repr__()),
                    "environment": {"source_sha256": sources.FROZEN_CORE_SHA, "python": "3.13" if model_key == "qwen_main" else "3.12"},
                    "metrics": evaluate(dataset.test, predictions, len(labels))}
                write_json(run_dir / "run.json", record)
                write_json(run_dir / "test_manifest.json", {"dataset": name, "labels": labels, "manifest_sha256": digest(source_manifest),
                    "rows": [{"id": row.id, "label": row.label, "text_sha256": __import__('hashlib').sha256(row.text.encode()).hexdigest()} for row in dataset.test]})
                (run_dir / "predictions.jsonl").write_text("".join(json.dumps(asdict(prediction)) + "\n" for prediction in predictions))
                config["historical_sources"][run_dir.name] = {"model": spec["model"], "provider": spec["provider"],
                    "path": run_dir.relative_to(tmp_path).as_posix(), "hashes": {f: sources.file_sha(run_dir / f) for f in sources.FILE_NAMES}}
    write_json(sources.CONFIG, config)

    def refresh(name, model_key, shots):
        source = config["historical_sources"][sources.condition_key(name, model_key, shots)]
        source["hashes"] = {f: sources.file_sha(tmp_path / source["path"] / f) for f in sources.FILE_NAMES}
        write_json(sources.CONFIG, config)
        return tmp_path / source["path"]

    return config, refresh


def source_dir(registry, dataset="sst2", model="qwen_small", shots=0):
    return sources.ROOT / registry[0]["historical_sources"][sources.condition_key(dataset, model, shots)]["path"]


def test_all_historical_conditions_and_failure_denominators_without_provider_calls(registry, monkeypatch):
    from jevbench import providers
    monkeypatch.setattr(providers, "build_provider", lambda *a, **k: pytest.fail("No source inference is permitted"))
    for name in sources.DATASETS:
        data = sources.load_dataset(name)
        for model in ("qwen_small", "qwen_main", "luna", "astra"):
            for shots in sources.SHOTS:
                job = sources.load_source(data, model, shots)
                assert len(job["predictions"]) == len(data.test)
                assert len(job["examples"]) == shots * len(data.labels)
                assert job["manifest_sha256"] is None
        direct = sources.load_direct(data, 0)
        assert direct["record"]["metrics"]["n_failures"] == 1
        assert direct["record"]["metrics"]["accuracy"] == (len(data.test) - 1) / len(data.test)
        assert direct["predictions"][0].label is None
    assert not sources.SOURCES_DIR.exists()


def test_colab_metadata_compatibility_keeps_unequal_hashes_and_original_bytes(registry):
    data = sources.load_dataset("sst2")
    directory = source_dir(registry, model="qwen_main", shots=4)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    job = sources.load_source(data, "qwen_main", 4)
    assert not job["content_audit"]["manifest_hashes_are_equal"]
    assert job["content_audit"]["raw_content_recomputed"]
    assert job["record"]["manifest_sha256"] != digest(data.manifest)
    assert set(job["content_audit"]["manifest_difference_paths"]) == {"splits_before_text_transform", "post_transform_overlap_removed_ids"}
    assert all((directory / name).read_bytes() == value for name, value in before.items())
    assert not job["content_audit"]["run_a"]["path"].startswith("/")


def test_freeze_idempotent_and_manifest_drift_never_overwritten(registry):
    data = sources.load_dataset("sst2")
    job = sources.load_source(data, "qwen_main", 4, freeze=True)
    path = sources.ROOT / job["manifest_path"]
    original = path.read_bytes()
    assert sources.load_source(data, "qwen_main", 4, freeze=True)["manifest_sha256"] == job["manifest_sha256"]
    assert path.read_bytes() == original
    changed = json.loads(original)
    changed["shots_per_class"] = 0
    write_json(path, changed)
    with pytest.raises(sources.SourceError, match="manifest changed"):
        sources.load_source(data, "qwen_main", 4, freeze=True)
    assert json.loads(path.read_text())["shots_per_class"] == 0


def test_frozen_manifest_portable_to_another_checkout(registry, tmp_path, monkeypatch):
    data = sources.load_dataset("sst2")
    job = sources.load_source(data, "qwen_main", 4, freeze=True)
    destination = tmp_path.parent / (tmp_path.name + "-copy")
    shutil.copytree(tmp_path, destination)
    monkeypatch.setattr(sources, "ROOT", destination)
    monkeypatch.setattr(sources, "CONFIG", destination / "configs/text_extension_sources.json")
    monkeypatch.setattr(sources, "LOCAL", destination / "results/text_extension/local")
    monkeypatch.setattr(sources, "SOURCES_DIR", destination / "results/text_extension/sources")
    assert sources.load_source(sources.load_dataset("sst2"), "qwen_main", 4)["manifest_sha256"] == job["manifest_sha256"]


def test_changed_frozen_file_rejected_before_reuse(registry):
    data = sources.load_dataset("sst2")
    path = source_dir(registry) / "predictions.jsonl"
    path.write_text(path.read_text() + " ")
    with pytest.raises(sources.SourceError, match="artifact changed"):
        sources.load_source(data, "qwen_small", 0)


def test_changed_prepared_text_rejected(registry):
    path = sources.ROOT / "data/pilot/sst2/test.jsonl"
    path.write_text(path.read_text().replace("category", "altered category", 1))
    with pytest.raises(sources.SourceError, match="prepared text artifact changed"):
        sources.load_dataset("sst2")


def test_reordered_predictions_rejected_even_after_hashes_updated(registry):
    data = sources.load_dataset("sst2")
    path = source_dir(registry) / "predictions.jsonl"
    path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    registry[1]("sst2", "qwen_small", 0)
    with pytest.raises(sources.SourceError, match="prediction order"):
        sources.load_source(data, "qwen_small", 0)


def test_training_order_must_match_actual_seed_selection(registry):
    data = sources.load_dataset("sst2")
    path = source_dir(registry, shots=4) / "run.json"
    record = json.loads(path.read_text())
    record["training_example_ids"].reverse()
    write_json(path, record)
    registry[1]("sst2", "qwen_small", 4)
    with pytest.raises(sources.SourceError, match="training/core identity"):
        sources.load_source(data, "qwen_small", 4)


@pytest.mark.parametrize("values", [[float("nan"), 0.1], [0.9, 0.9], [True, False], [0.1, 0.9]])
def test_bad_probability_or_argmax_rejected_with_refreshed_artifact_hashes(registry, values):
    data = sources.load_dataset("sst2")
    path = source_dir(registry) / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["probabilities"] = values
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    registry[1]("sst2", "qwen_small", 0)
    with pytest.raises(sources.SourceError, match="probability vector"):
        sources.load_source(data, "qwen_small", 0)


def test_forbidden_manifest_change_is_not_treated_as_metadata(registry):
    data = sources.load_dataset("sst2")
    directory = source_dir(registry, model="qwen_main")
    record = json.loads((directory / "run.json").read_text())
    record["dataset_manifest"]["text_policy"]["max_chars"] = 999
    record["manifest_sha256"] = digest(record["dataset_manifest"])
    test = json.loads((directory / "test_manifest.json").read_text())
    test["manifest_sha256"] = record["manifest_sha256"]
    write_json(directory / "run.json", record)
    write_json(directory / "test_manifest.json", test)
    registry[1]("sst2", "qwen_main", 0)
    with pytest.raises(sources.SourceError, match="full-content/matched-training audit"):
        sources.load_source(data, "qwen_main", 0)


def test_tampered_metric_and_model_revision_rejected(registry):
    data = sources.load_dataset("sst2")
    path = source_dir(registry) / "run.json"
    original = json.loads(path.read_text())
    altered = copy.deepcopy(original)
    altered["metrics"]["accuracy"] = 0.1
    write_json(path, altered)
    registry[1]("sst2", "qwen_small", 0)
    with pytest.raises(AuditError, match="Recomputed accuracy"):
        sources.load_source(data, "qwen_small", 0)
    altered = copy.deepcopy(original)
    altered["config"]["revision"] = "f" * 40
    write_json(path, altered)
    registry[1]("sst2", "qwen_small", 0)
    with pytest.raises(sources.SourceError, match="revision differs"):
        sources.load_source(data, "qwen_small", 0)


def test_missing_local_sources_pending_without_import_or_freeze(registry, monkeypatch):
    data = sources.load_dataset("sst2")
    monkeypatch.setattr(sources.importlib, "import_module", lambda *args: pytest.fail("Absent source must not load inference helper"))
    assert sources.load_source(data, "smollm2", 0, freeze=True) is None
    assert not sources.SOURCES_DIR.exists()


def test_multiple_local_candidates_rejected_without_score_selection(registry):
    data = sources.load_dataset("sst2")
    for name in ("one", "two"):
        write_json(sources.LOCAL / name / "run.json", {"dataset": "sst2", "status": "complete",
            "config": {"model": sources.MODELS["smollm2"]["model"], "shots_per_class": 0}})
    with pytest.raises(sources.SourceError, match="Ambiguous"):
        sources.load_source(data, "smollm2", 0)
