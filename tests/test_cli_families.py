"""The families and generate commands."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from trapforge.cli import app
from trapforge.families import REGISTRY, AffineLedger, Difficulty

runner = CliRunner()


def run(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def test_families_lists_every_registered_family() -> None:
    code, output = run("families")
    assert code == 0
    assert output == "".join(f"{f.name}: {f.summary}\n" for f in REGISTRY)
    assert "affine-ledger: " in output


def test_generate_proves_solves_and_traps() -> None:
    code, output = run("generate", "affine-ledger", "--seed", "7")
    instance = REGISTRY.generate("affine-ledger", 7, Difficulty.EASY)
    assert code == 0, output
    assert output.startswith("affine-ledger / easy / seed 7\n  anchors: 4\n")
    assert "proof: unique: a=17, b=7 (certificate re-checked: True)\n" in output
    assert "reference: matches the expected output\n" in output
    assert "baseline: 4 of 7 output lines differ; matches the visible sample\n" in output
    assert f"sha256: {instance.digest()}\n" in output


def test_generate_writes_the_bundle(tmp_path: Path) -> None:
    out = tmp_path / "inst"
    code, output = run(
        "generate", "affine-ledger", "--seed", "2", "--difficulty", "MEDIUM", "--out", str(out)
    )
    assert code == 0, output
    bundle = REGISTRY.generate("affine-ledger", 2, Difficulty.MEDIUM).bundle()
    written = {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    assert written == bundle
    assert f"wrote 9 files under {out}\n" in output
    # The written system is the prover's JSON format, so `prove` accepts it directly.
    code, output = run("prove", str(out / "meta" / "system.json"), "--require-unique")
    assert code == 0, output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["nope"], "unknown family 'nope'"),
        (["affine-ledger", "--difficulty", "brutal"], "unknown difficulty 'brutal'"),
    ],
)
def test_generate_rejects_bad_input(args: list[str], message: str) -> None:
    code, output = run("generate", *args)
    assert code == 2
    assert message in output


def test_generate_rejects_a_negative_seed() -> None:
    code, _ = run("generate", "affine-ledger", "--seed", "-1")
    assert code == 2


def test_generate_reports_an_unwritable_output(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("in the way")
    code, output = run("generate", "affine-ledger", "--out", str(blocker))
    assert code == 2
    assert "cannot write under" in output


def test_generate_fails_when_the_baseline_is_not_trapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AffineLedger, "baseline", AffineLedger.solve)
    code, output = run("generate", "affine-ledger")
    assert code == 1
    assert "baseline: 0 of 7 output lines differ; matches the visible sample" in output


def test_generate_fails_when_the_reference_is_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AffineLedger, "solve", AffineLedger.baseline)
    code, output = run("generate", "affine-ledger")
    assert code == 1
    assert "reference: DIFFERS FROM the expected output" in output


def test_generate_counts_missing_lines_as_differences(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AffineLedger, "baseline", lambda self, files: b"account,balance\n")
    code, output = run("generate", "affine-ledger")
    assert code == 1
    assert "baseline: 6 of 7 output lines differ; misses the visible sample" in output
