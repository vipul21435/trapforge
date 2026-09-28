"""The export and verify commands, end to end through the CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from trapforge.cli import app

runner = CliRunner()
GOLDEN = Path(__file__).resolve().parent / "golden"


def run(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def test_export_writes_the_bundle_and_prints_its_layout(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    code, output = run("export", "affine-ledger", "--out", str(out))
    assert code == 0, output
    assert output == (
        "affine-ledger / easy / seed 0\n"
        "  data/: 4 files\n"
        "  solution/: 20 files\n"
        "  baseline/: 20 files\n"
        "  tests/: 1 file\n"
        "  proof/: 1 file\n"
        f"wrote 49 files under {out}\n"
    )
    assert (out / "Dockerfile").read_bytes() == (GOLDEN / "bundle-Dockerfile.txt").read_bytes()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["export", "nope", "--out", "x"], "unknown family 'nope'"),
        (["export", "affine-ledger", "--out", "x", "--difficulty", "extreme"], "difficulty"),
    ],
)
def test_export_rejects_bad_arguments(tmp_path: Path, args: list[str], message: str) -> None:
    args = [str(tmp_path / a) if a == "x" else a for a in args]
    code, output = run(*args)
    assert code == 2
    assert message in output


def test_export_refuses_a_non_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x")
    code, output = run("export", "affine-ledger", "--out", str(tmp_path))
    assert code == 2
    assert "not an empty directory" in output


def test_export_reports_an_unwritable_target(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    code, output = run("export", "affine-ledger", "--out", str(blocker / "bundle"))
    assert code == 2
    assert "cannot write under" in output


@pytest.mark.slow
def test_verify_passes_an_export_and_fails_a_tampered_one(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    assert run("export", "affine-ledger", "--seed", "2", "--out", str(out))[0] == 0
    code, output = run("verify", str(out), "--no-docker")
    assert code == 0, output
    assert output.splitlines() == [
        "ok   certificate proves a unique answer: verified: unique, 1 solution",
        "ok   local solution passes the grader: PASS test_output_matches_byte_for_byte",
        output.splitlines()[2],
        "verified",
    ]
    assert output.splitlines()[2].startswith("ok   local baseline fails the grader: FAIL ")
    # A baseline that is secretly the reference makes the task worthless.
    (out / "baseline" / "solve.py").write_bytes((out / "solution" / "solve.py").read_bytes())
    code, output = run("verify", str(out), "--no-docker")
    assert code == 1
    assert "FAIL local baseline fails the grader" in output
    assert output.endswith("verification failed\n")


def test_verify_rejects_a_directory_that_is_not_a_bundle(tmp_path: Path) -> None:
    code, output = run("verify", str(tmp_path))
    assert code == 2
    assert "is not a task bundle" in output
