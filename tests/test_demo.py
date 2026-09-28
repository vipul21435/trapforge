"""The demo script must keep working against the current CLI."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_demo_script_runs_every_step() -> None:
    env = {**os.environ, "TRAPFORGE": f"{sys.executable} -m trapforge.cli"}
    result = subprocess.run(
        ["sh", "scripts/demo.sh"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("\n== ") == 15
    assert "exactly one point in the box (unique)" in result.stdout
    assert result.stdout.count("certificate re-checked: True") == 6
    assert "unique: a=5, b=7" in result.stdout
    assert "\nverified: unique, 1 solution\n" in result.stdout
    assert "baseline: 4 of 7 output lines differ; matches the visible sample" in result.stdout
    assert result.stdout.rstrip().endswith(
        "demo finished: every step above ran on the bundled examples/ inputs or a seeded generator"
    )
