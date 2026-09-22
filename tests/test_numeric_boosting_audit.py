"""Run native boosting checks in a fresh interpreter, isolated from torch tests.

The native-library tests execute below in a child pytest process. This avoids
mixing macOS OpenMP runtimes used by PyTorch and the boosting wheels in one
interpreter while preserving normal assertion failures and complete coverage.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

if os.environ.get("JEVBENCH_ISOLATED_NUMERIC_TESTS") != "1":
    def test_native_boosting_in_isolated_interpreter():
        if any(importlib.util.find_spec(name) is None for name in ("xgboost", "lightgbm")):
            pytest.skip("Optional pinned XGBoost/LightGBM packages are not installed")
        env = dict(os.environ, JEVBENCH_ISOLATED_NUMERIC_TESTS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        env.pop("PYTEST_ADDOPTS", None)
        result = subprocess.run([sys.executable, "-m", "pytest", str(Path(__file__).resolve()), "-q"],
            cwd=Path(__file__).parents[1], env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=120)
        assert result.returncode == 0, result.stdout
else:
    import copy
    from datetime import datetime, timezone
    import json
    from pathlib import Path
    import shutil

    import pytest

    from jevbench.runner import digest, save_json
    from jevbench.types import PreparedDataset, Row


    @pytest.fixture(params=[("xgboost", 2, 4), ("lightgbm", 3, None)])
    def artifact(request, tmp_path, monkeypatch):
        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        root = Path(__file__).parents[1]
        monkeypatch.syspath_prepend(str(root / "scripts"))
        import run_numeric_boosting as runner
        import audit_numeric_boosting as audit
        model, n_classes, budget = request.param
        config = json.loads((root / "configs/numeric_boosting.json").read_text())
        fixture_root = tmp_path / "repository"
        (fixture_root / "configs").mkdir(parents=True)
        shutil.copyfile(root / "configs/numeric_boosting.json", fixture_root / "configs/numeric_boosting.json")
        for name in runner.source_hashes():
            destination = fixture_root / name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(root / name, destination)
        native, rows = {}, {}
        for split, count, offset in (("train", 6, 0), ("validation", 2, 100), ("test", 3, 200)):
            native[split], rows[split] = [], []
            for label in range(n_classes):
                for i in range(count):
                    row_id = f"{split}-{label}-{i}"
                    native[split].append({"id": row_id, "label": label,
                        "features": {"value": None if split == "train" and i == 0 else label * 10 + i + offset,
                                     "empty": None if split == "train" else offset}})
                    rows[split].append(Row(row_id, f"audit fixture {row_id}", label))
        dataset = PreparedDataset("breast_cancer" if n_classes == 2 else "wine", [str(i) for i in range(n_classes)],
            rows["train"], rows["validation"], rows["test"], {"tabular": {"features": [
                {"name": "value", "kind": "numeric"}, {"name": "empty", "kind": "numeric"}]}})
        protocol = {"config": config, "config_file_sha256": runner.sha(fixture_root / "configs/numeric_boosting.json"),
            "source_sha256": runner.source_hashes(), "study_status": runner.STUDY_STATUS, "hosted_model_calls": 0,
            "validation_labels_used_for_fitting_or_selection": 0, "prior_results_already_viewed": True,
            "prepared_manifest_sha256": {name: digest(dataset.manifest) for name in config["datasets"]},
            "saved_before_first_fit_at": datetime.now(timezone.utc).isoformat()}
        out = tmp_path / "boosting"
        save_json(out / "protocol.json", protocol)
        # Synthetic fixtures exercise real libraries; the audit itself may never fit.
        record = runner.run_numeric(dataset, native, model, config, out, train_per_class=budget, protocol=protocol)
        path = out / record["run_id"] / "run.json"
        entry = {"run_id": record["run_id"], "run_json_sha256": runner.sha(path),
            **{k: record["metrics"][k] for k in ("n_test", "accuracy", "macro_f1", "balanced_accuracy", "n_failures",
                "probability_coverage", "log_loss", "brier_sum", "bootstrap")}}
        summary = {"study_status": runner.STUDY_STATUS, "protocol_sha256": digest(protocol), "runs": [entry]}
        save_json(out / "summary.json", summary)
        return audit, runner, path, dataset, native, fixture_root


    def test_audit_passes_without_refitting_or_predicting(artifact, monkeypatch):
        audit, runner, path, dataset, native, root = artifact
        from sklearn.pipeline import Pipeline
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        def prohibited(*args, **kwargs):
            raise AssertionError("Audit must not fit or predict")

        for cls in (Pipeline, XGBClassifier, LGBMClassifier):
            monkeypatch.setattr(cls, "fit", prohibited)
            monkeypatch.setattr(cls, "predict", prohibited)
            monkeypatch.setattr(cls, "predict_proba", prohibited)
        assert audit.audit_boosting_run(path, dataset, native, root=root) is None


    @pytest.mark.parametrize("tamper", ["source", "native", "version", "median", "parameters", "artifact", "probabilities", "metrics", "summary"])
    def test_audit_rejects_changed_provenance_or_results(artifact, tamper):
        audit, runner, path, dataset, native, root = artifact
        native = copy.deepcopy(native)
        record = json.loads(path.read_text())
        summary_path = path.parent.parent / "summary.json"
        summary = json.loads(summary_path.read_text())
        if tamper == "source":
            with (root / "scripts/run_numeric_boosting.py").open("a") as stream:
                stream.write("\n# source mutation\n")
        elif tamper == "native":
            native["train"][1]["features"]["value"] += 0.5
        elif tamper == "version":
            record["environment"]["packages"][record["config"]["model"]] = "0.0.0"
        elif tamper == "median":
            record["training"]["features"]["numeric_training_medians"][0] += 1
        elif tamper == "parameters":
            record["training"]["all_estimator_parameters"]["max_depth"] = 9
        elif tamper == "artifact":
            with path.with_name("predictions.jsonl").open("a") as stream:
                stream.write("\n")
        elif tamper == "probabilities":
            values = [json.loads(line) for line in path.with_name("predictions.jsonl").read_text().splitlines()]
            n = len(dataset.labels)
            probabilities = [0.0] * n
            probabilities[(values[0]["label"] + 1) % n] = 1.0
            values[0]["probabilities"] = probabilities
            path.with_name("predictions.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
            record["artifacts_sha256"]["predictions.jsonl"] = runner.sha(path.with_name("predictions.jsonl"))
        elif tamper == "metrics":
            record["metrics"]["accuracy"] = -1
        elif tamper == "summary":
            summary["runs"][0]["run_json_sha256"] = "0" * 64
        save_json(path, record)
        if tamper == "probabilities":
            summary["runs"][0]["run_json_sha256"] = runner.sha(path)
        save_json(summary_path, summary)
        with pytest.raises(ValueError, match="Numeric boosting audit"):
            audit.audit_boosting_run(path, dataset, native, root=root)
