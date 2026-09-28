"""Prove that an exported bundle grades the way it should: the reference passes, the baseline fails.

:func:`verify_bundle` copies the bundle to a fresh temporary directory per run and then:

1. re-checks ``proof/uniqueness.cert.json`` with the standalone checker and requires the
   verdict ``unique`` (it does not yet tie that certificate to ``data/`` or ``task.json``);
2. runs ``solution/solve.py`` and then the bundle's own grader locally, with
   ``python -I -S`` (isolated mode, no site-packages), so the run proves that the vendored
   code needs nothing installed; the solver must exit 0 and the grader must pass;
3. does the same with ``baseline/solve.py``; the grader must fail. The grader always runs
   on whatever the baseline wrote, even when it exits nonzero, because a solver can write
   the right bytes and then crash. A baseline that is missing or cannot start (an import
   or syntax error) is a failed check, not a trapped baseline. A solver that runs past
   the time limit is stopped and graded the same way;
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
import uuid
from collections.abc import Callable
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


#: Exceptions whose traceback means the solver never got to run its own logic.
_CANNOT_START = ("ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError")


def _run(
    command: list[str],
    cwd: Path | None = None,
    on_timeout: Callable[[], object] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``command``; past :data:`_TIMEOUT` seconds report a killed process, never raise."""
    try:
        return subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=_TIMEOUT, check=False
        )
    except subprocess.TimeoutExpired:
        if on_timeout is not None:
            on_timeout()
        return subprocess.CompletedProcess(command, -9, "", f"timed out after {_TIMEOUT} s")


def _cannot_start(solver: subprocess.CompletedProcess[str]) -> bool:
    return solver.returncode != 0 and _last_line(solver).split(":", 1)[0] in _CANNOT_START


def _judge(
    name: str,
    must_pass: bool,
    solver: subprocess.CompletedProcess[str],
    grade: Callable[[], subprocess.CompletedProcess[str]],
) -> Check:
    """Grade whatever the solver wrote; the reference must also exit 0."""
    exited = f"solver exited {solver.returncode}: {_last_line(solver)}"
    if solver.returncode != 0 and (must_pass or _cannot_start(solver)):
        return Check(name, False, exited)
    graded = grade()
    detail = _last_line(graded)
    if solver.returncode != 0:
        detail = f"{exited}; grader: {detail}"
    return Check(name, (graded.returncode == 0) == must_pass, detail)


def _missing(name: str, bundle: Path, folder: str) -> Check | None:
    if (bundle / folder / "solve.py").is_file():
        return None
    return Check(name, False, f"{folder}/solve.py is missing")


def _local(bundle: Path, work: Path, folder: str, must_pass: bool) -> Check:
    name = f"local {folder} {'passes' if must_pass else 'fails'} the grader"
    missing = _missing(name, bundle, folder)
    if missing is not None:
        return missing
    task = work / f"local-{folder}"
    shutil.copytree(bundle, task)
    shutil.rmtree(task / "output", ignore_errors=True)
    solver = _run([sys.executable, "-I", "-S", f"{folder}/solve.py", "."], cwd=task)
    return _judge(
        name,
        must_pass,
        solver,
        lambda: _run([sys.executable, "-I", "-S", "tests/test_outputs.py"], cwd=task),
    )


def _docker(bundle: Path, work: Path, tag: str, folder: str, must_pass: bool) -> Check:
    name = f"docker {folder} {'passes' if must_pass else 'fails'} the grader"
    missing = _missing(name, bundle, folder)
    if missing is not None:
        return missing
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

    def in_container(*command: str) -> subprocess.CompletedProcess[str]:
        # A timeout kills only the docker client, so the named container is removed too.
        container = f"trapforge-verify-{uuid.uuid4().hex[:12]}"
        base = ["docker", "run", "--rm", "--name", container, "--network", "none"]
        return _run(
            [*base, *user, *mounts, tag, *command],
            on_timeout=lambda: _run(["docker", "rm", "--force", container]),
        )

    solver = in_container("python", "-I", "-S", f"{folder}/solve.py", "/task")
    return _judge(
        name, must_pass, solver, lambda: in_container("python", "-I", "-S", "tests/test_outputs.py")
    )


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
