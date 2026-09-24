#!/usr/bin/env python3
"""Run the saved study's original code in a verified, disposable checkout.

The active package is never imported here. Historical audits load the archived
package and calculate its real source hash. Only prepare-data may download
public datasets; no mode runs model inference or paid API requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reproducibility/study-324d633.zip"
MANIFEST = ROOT / "reproducibility/study-324d633.manifest.json"
COMMIT = "324d63329e6ee6f98bcf5c4e00ca0256b74e0e64"
MANIFEST_SHA256 = "e0eac5012e34ee356780ad0d6d55d03ee28cb2f5270ea2dd9a2a728640ce0641"
CORE_SHA256 = "d5547ccde4224e182653315d81c0a631c789bbe294ad7bcf95ef0cddd25ce608"
TABULAR_REPORT_SHA256 = "7e3df4dc6a04aa0a88068e9639e3d04d95f86bcae38d3086ad12d23b4f11a7e2"
PREPARED_DIRS = ("data/pilot/sst2", "data/pilot/trec", "data/tabular-full/titanic",
                 "data/tabular-full/breast_cancer", "data/tabular-full/wine")
DASHBOARD_OVERLAY = (
    "scripts/build_expanded_numeric_dashboard.py",
    "scripts/build_text_extension_dashboard.py",
    "scripts/dashboard_aggregate.py",
    "tests/test_expanded_numeric_dashboard.py",
    "tests/test_text_extension_dashboard.py",
    "tests/test_dashboard_aggregate.py",
)
FRONTEND_OVERLAY = (
    "dashboard/dist/app.js", "dashboard/dist/numeric.js", "dashboard/dist/text.js",
    "dashboard/dist/review.js", "dashboard/dist/decision-dashboard.js",
    "dashboard/dist/index.html", "dashboard/dist/text.html", "dashboard/dist/review.html",
    "dashboard/dist/historical.html", "dashboard/dist/styles.css", "dashboard/dist/numeric.css",
    "dashboard/dist/text.css", "dashboard/dist/review.css",
    "dashboard/tests/dashboard.smoke.cjs", "dashboard/tests/numeric-dashboard.smoke.cjs",
    "dashboard/tests/text-dashboard.smoke.cjs", "dashboard/tests/review-dashboard.smoke.cjs",
    "dashboard/tests/decision-page-scripts.cjs",
)
DASHBOARD_FILES = ("numeric-data.json", "text-data.json", "review-data.json", "data.json")
AUDIT_REPORTS = (
    "results/numeric_expansion/COMPARISON.json",
    "results/text_extension/COMPARISON.json",
    "results/review_value/ANALYSIS.json",
    "results/completion_20260923/RECOVERY_COMPARISON.json",
    "results/review_controls/COMPARISON.json",
    "results/completion_20260923/CONTROL_RECOVERY_COMPARISON.json",
    "results/completion_20260923/COSTS.json",
    "results/review_value/validation_scoring/AUDITED_SUMMARY.json",
)


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    require(bool(name) and "\\" not in name and not path.is_absolute()
            and all(part not in ("", ".", "..") for part in name.split("/"))
            and ":" not in name and "\x00" not in name,
            f"Unsafe snapshot path: {name!r}")
    return path


def verify_snapshot(archive: Path = SNAPSHOT, manifest_path: Path = MANIFEST) -> tuple[dict, dict[str, bytes]]:
    require(archive.is_file() and not archive.is_symlink(), "Snapshot must be a regular ZIP file")
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "Manifest must be a regular file")
    raw_manifest = manifest_path.read_bytes()
    require(sha(raw_manifest) == MANIFEST_SHA256, "Frozen manifest checksum differs")
    manifest = json.loads(raw_manifest)
    require(manifest["schema_version"] == 1 and manifest["commit"] == COMMIT
            and manifest["core_sha256"] == CORE_SHA256, "Unexpected frozen snapshot identity")
    require(sha(archive.read_bytes()) == manifest["archive_sha256"], "Frozen archive checksum differs")
    payloads = {}
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        require(len(names) == len(set(names)) and set(names) == set(manifest["files"]),
                "Frozen archive inventory differs")
        for member in bundle.infolist():
            safe_relative(member.filename)
            entry = manifest["files"][member.filename]
            mode = member.external_attr >> 16
            require(not member.is_dir() and stat.S_ISREG(mode)
                    and entry["mode"] in ("100644", "100755"), "Snapshot contains a non-regular file")
            require(mode == int(entry["mode"], 8) and member.file_size == entry["size"],
                    "Snapshot member metadata differs")
            content = bundle.read(member)
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            require(sha(content) == entry["sha256"] and blob == entry["git_blob"],
                    f"Frozen file checksum differs: {member.filename}")
            payloads[member.filename] = content
    core = {Path(name).name: content.decode("utf-8") for name, content in payloads.items()
            if name.startswith("src/jevbench/") and name.count("/") == 2 and name.endswith(".py")}
    computed = sha(json.dumps(core, sort_keys=True, ensure_ascii=False).encode())
    require(computed == CORE_SHA256, "Archived package does not reproduce its original core digest")
    return manifest, payloads


def copy_regular_tree(source: Path, destination: Path) -> None:
    """Copy bytes, never symlinks or hardlinks to mutable live evidence."""
    require(source.is_dir() and not source.is_symlink(), f"Not a regular source directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        require(not entry.is_symlink(), f"Refusing a symlink in the audit input: {entry}")
        target = destination / entry.name
        if entry.is_dir():
            copy_regular_tree(entry, target)
        else:
            require(entry.is_file(), f"Refusing a non-regular audit input: {entry}")
            shutil.copyfile(entry, target)


def stage_checkout(destination: Path, repository: Path = ROOT, *, with_data: bool = True,
                   current_dashboard: bool = False) -> dict:
    manifest, payloads = verify_snapshot()
    require(not destination.exists(), "The disposable checkout must not already exist")
    destination.mkdir(parents=True)
    for name, content in payloads.items():
        target = destination.joinpath(*safe_relative(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(int(manifest["files"][name]["mode"], 8) & 0o777)
    copy_regular_tree(repository / "results", destination / "results")
    if with_data:
        for relative in PREPARED_DIRS:
            source = repository / relative
            require(source.is_dir(), "Prepared datasets are missing. Run prepare-data first; see RUNTIME_MAINTENANCE.md")
            copy_regular_tree(source, destination / relative)
    overlay = {}
    if current_dashboard:
        for relative in (*DASHBOARD_OVERLAY, *FRONTEND_OVERLAY):
            source = repository / relative
            require(source.is_file() and not source.is_symlink(), f"Missing regular overlay file: {relative}")
            content = source.read_bytes()
            (destination / relative).write_bytes(content)
            overlay[relative] = sha(content)
    return {"frozen_commit": COMMIT, "frozen_core_sha256": CORE_SHA256,
            "launcher_sha256": sha(Path(__file__).read_bytes()),
            "archive_sha256": manifest["archive_sha256"], "current_dashboard_overlay_sha256": overlay}


def child_environment(checkout: Path, *, dataset_downloads: bool = False) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().endswith(("API_KEY", "ACCESS_TOKEN"))
                   and key.upper() not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_KEY"}}
    environment.update(PYTHONPATH=os.pathsep.join((str(checkout / "src"), str(checkout / "scripts"))),
                       PYTHONNOUSERSITE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                       TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MPLCONFIGDIR=str(checkout / "artifacts/matplotlib"),
                       JEVBENCH_FROZEN_PROCESS="1")
    environment["HF_HUB_OFFLINE"] = "0" if dataset_downloads else "1"
    return environment


def run_python(checkout: Path, arguments: list[str], *, dataset_downloads: bool = False) -> None:
    print("Frozen study:", " ".join(arguments), flush=True)
    subprocess.run([sys.executable, *arguments], cwd=checkout,
                   env=child_environment(checkout, dataset_downloads=dataset_downloads), check=True)


def confirm_runtime(checkout: Path) -> None:
    code = ("from pathlib import Path; import jevbench.runner as r; "
            "assert Path(r.__file__).resolve().is_relative_to(Path.cwd()); "
            f"assert r.environment()['source_sha256'] == {CORE_SHA256!r}; "
            "print('Verified archived core:', r.environment()['source_sha256'])")
    run_python(checkout, ["-c", code])


def restore_controls(checkout: Path) -> None:
    run_python(checkout, ["scripts/restore_review_controls_payload.py"])


def verify_reports(checkout: Path, expected: dict[str, bytes]) -> None:
    changed = [name for name, content in expected.items() if (checkout / name).read_bytes() != content]
    require(not changed, "Historical report reproduction differs: " + ", ".join(changed))
    print(f"Verified {len(expected)} historical report files against saved bytes", flush=True)


def audit(checkout: Path) -> None:
    expected = {name: (checkout / name).read_bytes() for name in AUDIT_REPORTS}
    commands = (
        ["scripts/summarize_expanded_numeric.py", "--bootstrap-samples", "2000"],
        ["scripts/summarize_text_extension.py", "--bootstrap-samples", "2000"],
        ["scripts/analyze_review_value.py"],
        ["scripts/summarize_failed_retries.py", "--bootstrap-samples", "2000"],
        ["scripts/summarize_review_controls.py"],
        ["scripts/summarize_control_failed_retries.py", "--bootstrap-samples", "2000"],
        ["scripts/summarize_completion_costs.py"],
        ["scripts/audit_validation_scoring_ablation.py"],
    )
    for command in commands:
        run_python(checkout, command)
    verify_reports(checkout, expected)


def prepare_data(checkout: Path, destination: Path) -> None:
    # Check destinations before downloading or rebuilding anything.
    for relative in PREPARED_DIRS:
        require(not (destination / Path(relative).relative_to("data")).exists(),
                "Prepared-data export would replace an existing dataset")
    run_python(checkout, ["scripts/tabular_data.py", "--datasets", "titanic", "breast_cancer", "wine",
                         "--output", "data/tabular-full", "--seed", "42"], dataset_downloads=True)
    run_python(checkout, ["-m", "jevbench.cli", "prepare", "sst2", "trec", "--cache-dir", "data",
                         "--output", "data/pilot", "--seed", "42", "--train-limit", "10000",
                         "--validation-limit", "1000", "--test-limit", "200", "--max-text-chars", "2000"],
               dataset_downloads=True)
    run_python(checkout, ["scripts/restore_text_extension_data.py"])
    verify_tabular_preparation(checkout)
    restore_controls(checkout)
    # Recheck after preparation in case a destination appeared during the work.
    for relative in PREPARED_DIRS:
        require(not (destination / Path(relative).relative_to("data")).exists(),
                "Prepared-data export would replace an existing dataset")
    for relative in PREPARED_DIRS:
        copy_regular_tree(checkout / relative, destination / Path(relative).relative_to("data"))
    print(f"Exported verified prepared data to {destination}", flush=True)


def verify_tabular_preparation(checkout: Path) -> None:
    """Compare every prepared numeric/native byte with its original run manifest."""
    content = (checkout / "results/TABULAR_COMPARISON.json").read_bytes()
    require(sha(content) == TABULAR_REPORT_SHA256, "Original tabular report checksum differs")
    report = json.loads(content)
    for name in ("titanic", "breast_cancer", "wine"):
        run = next(row for row in report["runs"] if row["dataset"] == name)
        record_path = checkout.joinpath(*safe_relative(run["source_path"]).parts)
        record_bytes = record_path.read_bytes()
        require(sha(record_bytes) == run["artifact_sha256"]["run.json"], "Original tabular run checksum differs")
        original = json.loads(record_bytes)["dataset_manifest"]
        folder = checkout / "data/tabular-full" / name
        require(json.loads((folder / "manifest.json").read_bytes()) == original,
                f"Prepared tabular manifest differs: {name}")
        for relative, expected in {**original["files_sha256"], **original["tabular"]["native_files_sha256"]}.items():
            path = folder.joinpath(*safe_relative(relative).parts)
            require(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == expected,
                    f"Prepared tabular file differs: {name}/{relative}")


def current_dashboard(checkout: Path, repository: Path, export: Path | None, receipt: dict) -> None:
    run_python(checkout, ["-m", "pytest", "-q", "-rs", *[p for p in DASHBOARD_OVERLAY if p.startswith("tests/")]])
    for command in (["scripts/build_expanded_numeric_dashboard.py"],
                    ["scripts/build_text_extension_dashboard.py"],
                    ["scripts/build_review_value_dashboard.py"],
                    ["scripts/build_dashboard_data.py", "--output", "dashboard/dist/data.json"]):
        run_python(checkout, command)
    for name in ("dashboard", "numeric-dashboard", "text-dashboard", "review-dashboard"):
        subprocess.run(["node", f"dashboard/tests/{name}.smoke.cjs", "dashboard"],
                       cwd=checkout, env=child_environment(checkout), check=True)
    paths = {name: checkout / "dashboard/dist" / name for name in DASHBOARD_FILES}
    if export is None:
        for name, path in paths.items():
            require(path.read_bytes() == (repository / "dashboard/dist" / name).read_bytes(),
                    f"Current dashboard snapshot differs: {name}; use --export-dashboard to inspect a rebuilt copy")
        print("Verified all four current dashboard snapshots", flush=True)
    else:
        require(not export.exists(), "Dashboard export directory must not already exist")
        export.mkdir(parents=True)
        for name, path in paths.items():
            shutil.copyfile(path, export / name)
        receipt["dashboard_files_sha256"] = {name: sha(path.read_bytes()) for name, path in paths.items()}
        (export / "RUNTIME_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(f"Exported derived dashboard JSON only to {export}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "stage", "test", "audit", "current-dashboard", "prepare-data"))
    parser.add_argument("--workspace", type=Path,
                        help="stage only: create a persistent frozen checkout at this new path")
    parser.add_argument("--export-dashboard", type=Path,
                        help="current-dashboard only: export derived JSON to a new directory for review")
    parser.add_argument("--export-prepared", type=Path,
                        help="prepare-data only: export verified prepared datasets to this data directory")
    args = parser.parse_args(argv)
    require(args.export_dashboard is None or args.mode == "current-dashboard", "Invalid dashboard export mode")
    require((args.export_prepared is not None) == (args.mode == "prepare-data"),
            "prepare-data requires --export-prepared; other modes must omit it")
    require((args.workspace is not None) == (args.mode == "stage"),
            "stage requires --workspace; other modes must omit it")
    if args.mode == "verify":
        manifest, _ = verify_snapshot()
        print(json.dumps({"commit": COMMIT, "files": len(manifest["files"]), "core_sha256": CORE_SHA256,
                          "archive_sha256": manifest["archive_sha256"]}, indent=2))
        return
    if args.mode == "stage":
        checkout = args.workspace.resolve()
        receipt = stage_checkout(checkout)
        confirm_runtime(checkout)
        restore_controls(checkout)
        (checkout / "FROZEN_RUNTIME.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(f"Created frozen checkout: {checkout}\n"
              "Run original offline commands there with PYTHONPATH=src:scripts. "
              "Do not install this archived package over the active package.", flush=True)
        return
    with tempfile.TemporaryDirectory(prefix="jev-frozen-study-") as temporary:
        checkout = Path(temporary) / "checkout"
        receipt = stage_checkout(checkout, with_data=args.mode != "prepare-data",
                                 current_dashboard=args.mode == "current-dashboard")
        confirm_runtime(checkout)
        if args.mode == "prepare-data":
            prepare_data(checkout, args.export_prepared.resolve())
            return
        restore_controls(checkout)
        if args.mode == "test":
            run_python(checkout, ["-m", "pytest", "-q", "-rs"])
        elif args.mode == "audit":
            audit(checkout)
        else:
            current_dashboard(checkout, ROOT,
                              None if args.export_dashboard is None else args.export_dashboard.resolve(), receipt)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print(f"Frozen study stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
