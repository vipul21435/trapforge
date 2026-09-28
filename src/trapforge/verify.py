"""Prove that an exported bundle grades the way it should: the reference passes, the baseline fails.

:func:`verify_bundle` never trusts the bundle's own claims. It copies the bundle to a fresh
temporary directory per run and then:

1. re-checks ``proof/uniqueness.cert.json`` with the standalone checker and requires the
   verdict ``unique``;
2. runs ``solution/solve.py`` and then the grader locally, with ``python -I -S`` (isolated
   mode, no site-packages), so the run proves that the vendored code needs nothing
   installed; the grader must pass;
3. does the same with ``baseline/solve.py``; the grader must fail;
4. when Docker is available (or required), builds the bundle's ``Dockerfile`` and repeats
   both runs in containers started with ``--network none``, mounting only ``solution/`` or
   ``baseline/``, ``tests/`` and an empty ``output/``; the corpus comes from the image. The
   image is removed afterwards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from trapforge.prover import check_certificate

__all__ = [
    "Check",
    "VerifyReport",
    "docker_available",
    "verify_bundle",
]

_TIMEOUT = 600


@dataclass(frozen=True, slots=True)
class Check:
    """One verification step and whether it came out as required."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class VerifyReport:
    """Every step :func:`verify_bundle` ran, in order."""

    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def docker_available() -> bool:
    """True when a ``docker`` client is on PATH and its daemon answers."""
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _last_line(result: subprocess.CompletedProcess[str]) -> str:
    lines = (result.stdout + result.stderr).strip().splitlines()
    return lines[-1] if lines else f"exit {result.returncode}"


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=_TIMEOUT, check=False
    )


def _solver_failed(name: str, solver: subprocess.CompletedProcess[str], must_pass: bool) -> Check:
    # A solver that crashes writes no output, so the grader could only fail.
    return Check(name, not must_pass, f"solver exited {solver.returncode}: {_last_line(solver)}")


def _local(bundle: Path, work: Path, folder: str, must_pass: bool) -> Check:
    task = work / f"local-{folder}"
    shutil.copytree(bundle, task)
    shutil.rmtree(task / "output", ignore_errors=True)
    solver = _run([sys.executable, "-I", "-S", f"{folder}/solve.py", "."], cwd=task)
    name = f"local {folder} {'passes' if must_pass else 'fails'} the grader"
    if solver.returncode != 0:
        return Check(
            name, not must_pass, f"solver exited {solver.returncode}: {_last_line(solver)}"
        )
    grade = _run([sys.executable, "-I", "-S", "tests/test_outputs.py"], cwd=task)
    graded_pass = grade.returncode == 0
    return Check(name, graded_pass == must_pass, _last_line(grade))


def _docker(bundle: Path, work: Path, tag: str, folder: str, must_pass: bool) -> Check:
    task = work / f"docker-{folder}"
    shutil.copytree(bundle, task)
    output = task / "output"
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir()
    user = [] if not hasattr(os, "getuid") else ["--user", f"{os.getuid()}:{os.getgid()}"]
    mounts = [
        "-v",
        f"{task / folder}:/task/{folder}:ro",
        "-v",
        f"{task / 'tests'}:/task/tests:ro",
        "-v",
        f"{output}:/task/output",
    ]
    base = ["docker", "run", "--rm", "--network", "none", *user, *mounts, tag]
    name = f"docker {folder} {'passes' if must_pass else 'fails'} the grader"
    solver = _run([*base, "python", "-I", "-S", f"{folder}/solve.py", "/task"])
    if solver.returncode != 0:
        return Check(
            name, not must_pass, f"solver exited {solver.returncode}: {_last_line(solver)}"
        )
    grade = _run([*base, "python", "-I", "-S", "tests/test_outputs.py"])
    return Check(name, (grade.returncode == 0) == must_pass, _last_line(grade))


def verify_bundle(bundle: Path, *, docker: bool | None = None) -> VerifyReport:
    """Verify an exported bundle; ``docker=None`` uses Docker only when it is available.

    ``docker=True`` requires Docker (a missing daemon is a failed check) and ``False``
    skips it.
    """
    checks: list[Check] = []
    certificate = bundle / "proof" / "uniqueness.cert.json"
    try:
        result = check_certificate(certificate.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError) as error:
        checks.append(Check("certificate proves a unique answer", False, str(error)))
    else:
        unique = result.valid and result.verdict == "unique"
        checks.append(Check("certificate proves a unique answer", unique, result.reason))
    with tempfile.TemporaryDirectory(prefix="trapforge-verify-") as scratch:
        work = Path(scratch)
        checks.append(_local(bundle, work, "solution", must_pass=True))
        checks.append(_local(bundle, work, "baseline", must_pass=False))
        use_docker = docker_available() if docker is None else docker
        if use_docker and not docker_available():
            checks.append(Check("docker is available", False, "no docker daemon answered"))
        elif use_docker:
            checks.extend(_in_docker(bundle, work))
    return VerifyReport(tuple(checks))


def _in_docker(bundle: Path, work: Path) -> list[Check]:
    tag = f"trapforge-task:{work.name.lower().replace('_', '-')}"
    build = _run(
        ["docker", "build", "--quiet", "--label", "project=trapforge", "-t", tag, "."], cwd=bundle
    )
    if build.returncode != 0:
        return [Check("docker image builds", False, _last_line(build))]
    try:
        return [
            Check("docker image builds", True, f"built {tag} from the pinned base image"),
            _docker(bundle, work, tag, "solution", must_pass=True),
            _docker(bundle, work, tag, "baseline", must_pass=False),
        ]
    finally:
        _run(["docker", "image", "rm", "--force", tag])
