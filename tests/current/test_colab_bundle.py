"""The portable source bundle must be runnable without repository evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tomllib
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("colab_bundle_export", ROOT / "scripts/export_colab_bundle.py")
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


@pytest.fixture
def portable_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = set(bundle.FIXED_FILES + bundle.OPTIONAL_FILES + bundle.PORTABLE_TEST_FILES)
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*.py"))
    for name in names:
        original = ROOT / name
        if original.is_file():
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
    for name in (
        "tests/current/test_frozen_runtime.py", "tests/current/test_saved_metrics.py",
        "tests/test_frozen_suite.py", "tests/test_original_audit.py", ".env",
        "data/private.json", "results/private.json", "reproducibility/study.zip",
        "scripts/reproduce_frozen_study.py",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not part of a portable source bundle\n")
    return source


def test_bundle_has_only_portable_tests_and_valid_manifest(portable_source):
    original_settings = (portable_source / "pyproject.toml").read_bytes()
    destination = bundle.export(portable_source)
    with zipfile.ZipFile(destination) as archive:
        contents = {
            name.removeprefix(bundle.BUNDLE_ROOT + "/"): archive.read(name)
            for name in archive.namelist()
        }
    test_files = {name for name in contents if name.startswith("tests/")}
    assert test_files == set(bundle.PORTABLE_TEST_FILES)
    assert all(not name.startswith(("data/", "results/", "reproducibility/")) for name in contents)
    assert ".env" not in contents
    assert "scripts/reproduce_frozen_study.py" not in contents
    settings = tomllib.loads(contents["pyproject.toml"].decode())
    assert settings["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests/current"]
    assert settings["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]
    assert (portable_source / "pyproject.toml").read_bytes() == original_settings
    manifest = json.loads(contents.pop("bundle_manifest.json"))
    assert manifest["files_sha256"] == {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
    }
    first = destination.read_bytes()
    assert bundle.export(portable_source).read_bytes() == first


def test_bundle_strips_notebook_outputs_and_metadata(portable_source):
    path = portable_source / "notebooks/colab_benchmark.ipynb"
    notebook = json.loads(path.read_text())
    notebook["metadata"]["private"] = "local-only"
    code_cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    code_cell["execution_count"] = 12
    code_cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "local output"}]
    code_cell["attachments"] = {"local.txt": {"text/plain": "local attachment"}}
    path.write_text(json.dumps(notebook))
    with zipfile.ZipFile(bundle.export(portable_source)) as archive:
        cleaned = json.loads(archive.read(f"{bundle.BUNDLE_ROOT}/notebooks/colab_benchmark.ipynb"))
    assert "private" not in cleaned["metadata"]
    assert all(not cell.get("outputs") and cell.get("execution_count") is None for cell in cleaned["cells"])
    assert all(not cell.get("attachments") for cell in cleaned["cells"])
    assert all(cell.get("id") for cell in cleaned["cells"])


def test_bundle_refuses_source_symlinks(portable_source, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("outside = True\n")
    (portable_source / "src/jevbench/outside.py").symlink_to(outside)
    with pytest.raises(ValueError, match="outside repository"):
        bundle.export(portable_source)


@pytest.mark.parametrize("source", [b"[project]\nname = 'demo'\n", b"[tool.pytest.ini_options]\npythonpath = ['src']\n"])
def test_portable_settings_fail_if_pytest_layout_changes(source):
    with pytest.raises(ValueError, match="pytest"):
        bundle.portable_pyproject(source)
