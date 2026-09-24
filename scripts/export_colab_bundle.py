"""Export an allowlisted, output-free source bundle for a private Colab upload.

This copies source code, so review source cells for manually pasted secrets before
sharing. It never includes .env files, data, model weights, or result directories.
"""
import hashlib
import json
from pathlib import Path
import re
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = "jev-classification-benchmark"
FIXED_FILES = (
    "pyproject.toml", "README.md", ".gitignore",
    "configs/datasets.json", "configs/experiment.json", "configs/models.json", "configs/hosted_budget.json", "configs/colab_adapters.json",
    "docs/PROTOCOL.md", "docs/SOURCES.md", "docs/MODEL_ACCESS.md", "docs/BUDGET.md", "docs/ADAPTERS.md",
    "scripts/export_colab_bundle.py", "scripts/run_classical_matrix.py", "scripts/run_model_matrix.py",
    "scripts/run_budgeted_hosted.py", "scripts/run_openrouter_jev.py", "scripts/package_adapters.py",
    "configs/jev_openrouter.json",
    "notebooks/colab_benchmark.ipynb",
)
OPTIONAL_FILES = (
    "scripts/audit_cross_environment.py", "scripts/summarize_neural_pilot.py", "scripts/summarize_api_costs.py",
    "scripts/compare_combined_pilot.py", "scripts/summarize_combined_pilot.py", "scripts/plot_combined_pilot.py",
    "scripts/audit_colab_import.py", "scripts/summarize_jev_pilot.py", "scripts/summarize_jev_costs.py",
    "scripts/summarize_hosted.py", "scripts/summarize_colab.py",
    "scripts/plot_neural_pilot.py", "scripts/plot_results.py", "scripts/plot_calibration.py",
    "configs/tabular_datasets.json", "configs/tabular_hosted.json", "configs/tabular_adapters.json", "docs/TABULAR_PROTOCOL.md",
    "scripts/tabular_data.py", "scripts/run_tabular_classical.py", "scripts/run_tabular_local.py",
    "scripts/run_tabular_hosted.py", "scripts/export_tabular_colab.py", "scripts/tabular_colab_artifacts.py",
    "scripts/summarize_tabular.py", "scripts/summarize_tabular_costs.py", "scripts/plot_tabular.py",
    "notebooks/colab_tabular_benchmark.ipynb",
)
# These tests use generated fixtures and travel with the maintained package.
# Frozen-study checks and saved-evidence comparisons require the full repository.
PORTABLE_TEST_FILES = (
    "tests/current/test_classical.py",
    "tests/current/test_data.py",
    "tests/current/test_lora.py",
    "tests/current/test_metrics.py",
    "tests/current/test_prompts.py",
    "tests/current/test_providers.py",
    "tests/current/test_runner.py",
    "tests/current/test_runtime_regressions.py",
)


def portable_pyproject(content: bytes) -> bytes:
    """Set the exported test target without changing the repository config."""
    source = content.decode("utf-8")
    section = re.search(r"(?ms)^\[tool\.pytest\.ini_options\]\n(.*?)(?=^\[|\Z)", source)
    if section is None:
        raise ValueError("Missing pytest configuration in pyproject.toml")
    settings, replacements = re.subn(
        r"(?m)^testpaths[ \t]*=[ \t]*\[[^\n]*\][ \t]*$",
        'testpaths = ["tests/current"]', section.group(1),
    )
    if replacements != 1:
        raise ValueError("Expected one single-line pytest testpaths setting")
    updated = source[:section.start(1)] + settings + source[section.end(1):]
    parsed = tomllib.loads(updated)
    if parsed["tool"]["pytest"]["ini_options"]["testpaths"] != ["tests/current"]:
        raise ValueError("Portable pytest configuration is invalid")
    return updated.encode("utf-8")


def clean_notebook(content: bytes, name: str = "colab_benchmark.ipynb") -> bytes:
    notebook = json.loads(content)
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": name},
    }
    for index, cell in enumerate(notebook["cells"]):
        cell["metadata"] = {}
        cell.pop("attachments", None)
        cell.setdefault("id", f"jevbench-{index:02d}")
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return (json.dumps(notebook, ensure_ascii=False, indent=2) + "\n").encode()


def export(root: Path = ROOT) -> Path:
    destination = root / "artifacts" / f"{BUNDLE_ROOT}-colab.zip"
    destination.parent.mkdir(exist_ok=True)
    paths = [root / name for name in FIXED_FILES]
    paths.extend(root / name for name in OPTIONAL_FILES if (root / name).is_file())
    paths.extend((root / "src").rglob("*.py"))
    paths.extend(root / name for name in PORTABLE_TEST_FILES)
    for name in ("LICENSE", "requirements-macos.lock.txt"):
        if (root / name).is_file():
            paths.append(root / name)
    payloads = {}
    for path in sorted(set(paths)):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Refusing source outside repository: {path.name}")
        content = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".ipynb":
            content = clean_notebook(content, path.name)
        elif relative == "pyproject.toml":
            content = portable_pyproject(content)
        payloads[relative] = content
    payloads["bundle_manifest.json"] = (json.dumps({
        "schema_version": 1,
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())},
        "note": "Source-only bundle; no data, weights, environment secrets, or measured results.",
        "test_policy": "Maintained standalone tests only; frozen-study and saved-evidence tests require the full repository.",
    }, indent=2, sort_keys=True) + "\n").encode()
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(f"{BUNDLE_ROOT}/{name}")
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    print(export())
