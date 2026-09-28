"""The bundle exporter and the verifier: layout, golden files, standalone runs and Docker.

Golden files live in ``tests/golden/``. After an intended change to the templates, rerun
with ``TRAPFORGE_UPDATE_GOLDEN=1`` to rewrite them, and review the diff.
"""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _toy_family import ToyShift
from trapforge import verify as verify_module
from trapforge.bundle import BASE_IMAGE, BundleError, bundle_files, write_bundle
from trapforge.families import REGISTRY, Difficulty
from trapforge.prover import ConstraintSystem, check_certificate
from trapforge.verify import docker_available, verify_bundle

GOLDEN = Path(__file__).resolve().parent / "golden"
LEDGER = REGISTRY.get("affine-ledger")
INSTANCE = REGISTRY.generate("affine-ledger", 0, Difficulty.EASY)
FILES = bundle_files(LEDGER, INSTANCE)

#: Bundle files compared in full against tests/golden/ (named .txt so pytest skips them).
TEMPLATED = {
    "Dockerfile": "bundle-Dockerfile.txt",
    "instruction.md": "bundle-instruction.md.txt",
    "solution/solve.py": "bundle-solution-solve.py.txt",
    "tests/test_outputs.py": "bundle-test_outputs.py.txt",
    "task.json": "bundle-task.json.txt",
}


def manifest(files: dict[str, bytes]) -> str:
    """One line per file: its SHA-256, or ``vendored`` for copied TrapForge sources."""
    lines = []
    for name, content in files.items():
        tag = "vendored" if "/vendor/" in name else hashlib.sha256(content).hexdigest()
        lines.append(f"{tag}  {name}")
    return "\n".join(lines) + "\n"


def compare_golden(name: str, actual: bytes) -> None:
    path = GOLDEN / name
    if os.environ.get("TRAPFORGE_UPDATE_GOLDEN") == "1":
        path.write_bytes(actual)
    assert actual == path.read_bytes(), f"{name} changed; see the module docstring"


def export(tmp_path: Path, files: dict[str, bytes] = FILES) -> Path:
    out = tmp_path / "bundle"
    write_bundle(files, out)
    return out


# -- layout and golden files --------------------------------------------------------------


def test_the_bundle_layout_matches_the_golden_manifest() -> None:
    compare_golden("bundle-affine-ledger-easy-0.manifest", manifest(FILES).encode("ascii"))


@pytest.mark.parametrize(("path", "golden"), sorted(TEMPLATED.items()))
def test_generated_files_match_their_golden_copies(path: str, golden: str) -> None:
    compare_golden(golden, FILES[path])


def test_export_is_deterministic_and_ascii() -> None:
    again = bundle_files(LEDGER, REGISTRY.generate("affine-ledger", 0, Difficulty.EASY))
    assert again == FILES
    assert all(content.isascii() for content in FILES.values())
    assert list(FILES) == sorted(FILES)


def test_the_bundle_hides_the_expected_output_but_pins_its_digest() -> None:
    assert all(INSTANCE.expected not in content for content in FILES.values())
    task = json.loads(FILES["task.json"])
    assert task["expected_sha256"] == hashlib.sha256(INSTANCE.expected).hexdigest()
    assert task["instance_digest"] == INSTANCE.digest()
    assert task["base_image"] == BASE_IMAGE
    assert FILES["Dockerfile"].decode().count(f"FROM {BASE_IMAGE}\n") == 1
    assert "@sha256:" in BASE_IMAGE


def test_the_bundle_carries_a_certificate_that_rechecks_as_unique() -> None:
    result = check_certificate(FILES["proof/uniqueness.cert.json"].decode("ascii"))
    assert (result.valid, result.verdict) == (True, "unique")


def test_the_vendored_copy_leaves_out_the_cli_and_the_exporter() -> None:
    vendored = {name.split("vendor/", 1)[1] for name in FILES if "/vendor/" in name}
    assert "trapforge/modular.py" in vendored
    assert "trapforge/prover/certificate.py" in vendored
    assert not {"trapforge/cli.py", "trapforge/bundle.py", "trapforge/verify.py"} & vendored
    assert not any("__pycache__" in name for name in vendored)


def test_an_instance_whose_answer_is_not_unique_is_refused() -> None:
    loose = ConstraintSystem.build({name: (v, v + 1) for name, v in INSTANCE.hidden.items()}, [])
    ambiguous = dataclasses.replace(INSTANCE, system=loose)
    with pytest.raises(BundleError, match="does not pin the answer"):
        bundle_files(LEDGER, ambiguous)


def test_a_family_in_a_package_module_cannot_be_vendored() -> None:
    class Packaged(ToyShift):
        pass

    Packaged.__module__ = "somepackage.families"
    toy = ToyShift().generate(0, Difficulty.EASY)
    with pytest.raises(BundleError, match="cannot vendor family module"):
        bundle_files(Packaged(), toy)


def test_write_bundle_refuses_a_non_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "keep.txt").write_text("x")
    with pytest.raises(BundleError, match="not an empty directory"):
        write_bundle(FILES, tmp_path / "bundle")


# -- running the bundle -------------------------------------------------------------------


@pytest.mark.slow
def test_the_grader_runs_under_pytest_and_standalone(tmp_path: Path) -> None:
    bundle = export(tmp_path)
    solved = subprocess.run(
        [sys.executable, "-I", "-S", "solution/solve.py"],
        cwd=bundle,
        capture_output=True,
        check=False,
    )
    assert solved.returncode == 0, solved.stderr
    assert (bundle / "output" / INSTANCE.output).read_bytes() == INSTANCE.expected
    under_pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_outputs.py"],
        cwd=bundle,
        check=False,
        capture_output=True,
        text=True,
    )
    assert under_pytest.returncode == 0, under_pytest.stdout
    assert "2 passed" in under_pytest.stdout
    (bundle / "output" / INSTANCE.output).write_bytes(INSTANCE.expected + b"\n")
    standalone = subprocess.run(
        [sys.executable, "-I", "-S", "tests/test_outputs.py"],
        cwd=bundle,
        check=False,
        capture_output=True,
        text=True,
    )
    assert standalone.returncode == 1
    assert "FAIL test_output_matches_byte_for_byte" in standalone.stdout


@pytest.mark.slow
def test_verify_passes_a_fresh_export_locally(tmp_path: Path) -> None:
    report = verify_bundle(export(tmp_path), docker=False)
    assert [(check.name, check.passed) for check in report.checks] == [
        ("certificate proves a unique answer", True),
        ("local solution passes the grader", True),
        ("local baseline fails the grader", True),
    ]
    assert report.passed


@pytest.mark.slow
def test_verify_catches_a_baseline_that_passes(tmp_path: Path) -> None:
    # The toy family's baseline is its reference solver, so the task is not adversarial.
    toy = ToyShift()
    bundle = export(tmp_path, bundle_files(toy, toy.generate(3, Difficulty.EASY)))
    assert (bundle / "solution" / "vendor" / "_toy_family.py").is_file()
    report = verify_bundle(bundle, docker=False)
    failed = [check.name for check in report.checks if not check.passed]
    assert failed == ["local baseline fails the grader"]
    assert not report.passed


@pytest.mark.slow
def test_verify_reports_broken_solvers_and_certificates(tmp_path: Path) -> None:
    bundle = export(tmp_path)
    (bundle / "solution" / "solve.py").write_text("raise SystemExit(3)\n")
    (bundle / "proof" / "uniqueness.cert.json").unlink()
    report = verify_bundle(bundle, docker=False)
    assert [check.passed for check in report.checks] == [False, False, True]
    assert report.checks[1].detail.startswith("solver exited 3")


@pytest.mark.slow
def test_verify_grades_a_baseline_that_writes_the_answer_and_then_exits_nonzero(
    tmp_path: Path,
) -> None:
    # Regression: a nonzero baseline exit used to count as trapped without running the grader.
    bundle = export(tmp_path)
    reference = (bundle / "solution" / "solve.py").read_text()
    tail = "    sys.exit(main(sys.argv))"
    assert tail in reference
    cheat = reference.replace(tail, "    main(sys.argv)\n    sys.exit(1)")
    (bundle / "baseline" / "solve.py").write_text(cheat)
    report = verify_bundle(bundle, docker=False)
    baseline = report.checks[2]
    assert baseline.name == "local baseline fails the grader"
    assert not baseline.passed
    assert baseline.detail.startswith("solver exited 1")
    assert not report.passed


@pytest.mark.slow
@pytest.mark.parametrize("damage", ["missing", "import-error", "syntax-error"])
def test_verify_fails_a_baseline_that_cannot_start(tmp_path: Path, damage: str) -> None:
    bundle = export(tmp_path)
    solver = bundle / "baseline" / "solve.py"
    if damage == "missing":
        solver.unlink()
    elif damage == "import-error":
        solver.write_text("import trapforge_no_such_module\n")
    else:
        solver.write_text("def broken(:\n")
    report = verify_bundle(bundle, docker=False)
    assert [check.passed for check in report.checks] == [True, True, False]
    if damage == "missing":
        assert report.checks[2].detail == "baseline/solve.py is missing"


@pytest.mark.slow
def test_verify_still_traps_a_baseline_that_crashes_after_a_wrong_answer(tmp_path: Path) -> None:
    bundle = export(tmp_path)
    (bundle / "baseline" / "solve.py").write_text("raise SystemExit(4)\n")
    report = verify_bundle(bundle, docker=False)
    assert report.passed
    assert report.checks[2].detail.startswith("solver exited 4: exit 4; grader: ")


def test_a_solver_past_the_time_limit_is_a_graded_run_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify_module, "_TIMEOUT", 1)
    bundle = export(tmp_path)
    (bundle / "solution" / "solve.py").write_text("import time\ntime.sleep(30)\n")
    (bundle / "baseline" / "solve.py").write_text("import time\ntime.sleep(30)\n")
    report = verify_bundle(bundle, docker=False)
    assert [check.passed for check in report.checks] == [True, False, True]
    assert report.checks[1].detail == "solver exited -9: timed out after 1 s"


def test_run_calls_the_cleanup_hook_on_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify_module, "_TIMEOUT", 1)
    cleaned: list[bool] = []
    result = verify_module._run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        on_timeout=lambda: cleaned.append(True),
    )
    assert (result.returncode, cleaned) == (-9, [True])


def test_required_docker_without_a_daemon_is_a_failed_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify_module, "docker_available", lambda: False)
    monkeypatch.setattr(verify_module, "_local", lambda *args, **kwargs: _ok("local"))
    report = verify_bundle(export(tmp_path), docker=True)
    assert report.checks[-1] == verify_module.Check(
        "docker is available", False, "no docker daemon answered"
    )


def test_docker_available_is_false_without_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify_module.shutil, "which", lambda name: None)
    assert not docker_available()


def _ok(name: str) -> verify_module.Check:
    return verify_module.Check(name, True, "stubbed")


@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.skipif(not docker_available(), reason="no Docker daemon")
def test_verify_in_docker_with_the_network_disabled(tmp_path: Path) -> None:
    report = verify_bundle(export(tmp_path), docker=True)
    assert [(check.name, check.passed) for check in report.checks][3:] == [
        ("docker image builds", True),
        ("docker solution passes the grader", True),
        ("docker baseline fails the grader", True),
    ]
    assert report.passed
