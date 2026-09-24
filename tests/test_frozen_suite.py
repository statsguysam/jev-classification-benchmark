"""Run every original study test in its own verified historical checkout."""
from pathlib import Path
import subprocess
import sys


def test_original_study_suite():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/reproduce_frozen_study.py"), "test"],
                   cwd=root, check=True)
