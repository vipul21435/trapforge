"""Export a task instance as a self-contained, digest-pinned task bundle.

A bundle is a directory an evaluation harness can use without TrapForge installed:

* ``instruction.md`` and ``data/``: what the solver is given;
* ``Dockerfile``: the task image, ``FROM`` a ``python:3.12-slim`` image pinned by digest,
  with the instruction and the corpus baked in and nothing else;
* ``solution/solve.py``: the reference solver, standalone: it puts ``solution/vendor/`` (a
  copy of TrapForge's pure-Python modules, without the CLI) first on ``sys.path`` and needs
  only the Python standard library;
* ``baseline/solve.py``: the naive solver, packaged the same way, which must fail;
* ``tests/test_outputs.py``: the grader, which compares the solver's output with an embedded
  SHA-256 digest and byte count (the expected bytes themselves are not in the bundle); it
  runs under pytest or on its own with ``python tests/test_outputs.py``;
* ``proof/uniqueness.cert.json``: the prover's certificate that exactly one hidden world fits
  the corpus, which ``trapforge check`` re-verifies independently;
* ``task.json``: the family, seed, difficulty, output name, expected digest and instance
  digest.

Every generated file is ASCII with ``\\n`` newlines, and the whole bundle is a pure function
of the instance and the TrapForge sources, so exporting twice gives identical bytes.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path

import trapforge
from trapforge.canonical import json_bytes, text_bytes, tree_digest
from trapforge.families import TaskFamily, TaskInstance
from trapforge.prover import UniqueProof, prove

__all__ = [
    "BASE_IMAGE",
    "BundleError",
    "bundle_files",
    "write_bundle",
]

#: The task image's base, pinned by digest (the same pin as the TrapForge CLI image).
BASE_IMAGE = (
    "python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f"
)

#: Top-level modules left out of the vendored copy: the CLI needs typer, and no solver needs
#: the exporter or the verifier.
_NOT_VENDORED = frozenset({"cli.py", "bundle.py", "verify.py"})


class BundleError(ValueError):
    """Raised when an instance cannot be exported (not unique, or a family we cannot vendor)."""


def _vendored_sources(family: TaskFamily) -> dict[str, bytes]:
    """TrapForge's pure-Python sources, plus the family's module when it lives elsewhere."""
    root = Path(trapforge.__file__).resolve().parent
    files = {
        f"trapforge/{path.relative_to(root).as_posix()}": path.read_bytes()
        for path in sorted(root.rglob("*.py"))
        if path.relative_to(root).as_posix() not in _NOT_VENDORED
        and "__pycache__" not in path.parts
    }
    module_name = type(family).__module__
    if module_name.split(".")[0] != "trapforge":
        module = sys.modules.get(module_name)
        source = Path(getattr(module, "__file__", None) or "")
        if "." in module_name or source.name == "__init__.py" or source.suffix != ".py":
            raise BundleError(
                f"cannot vendor family module {module_name!r}: only TrapForge's own families "
                "and single-file top-level modules can be exported"
            )
        files[f"{module_name}.py"] = source.read_bytes()
    return files


def _solver_script(family: TaskFamily, instance: TaskInstance, method: str, role: str) -> str:
    cls = type(family)
    title = f"{role} for the {instance.family} task ({instance.difficulty}, seed {instance.seed})"
    return f'''"""{title}.

Standalone: it needs only Python 3.12. The pure-Python modules it uses are vendored in
vendor/ next to this file and put first on sys.path, so nothing has to be installed.

Usage: python solve.py [TASK_DIR]   (TASK_DIR defaults to the current directory)
Reads every file under TASK_DIR/data/ and writes TASK_DIR/output/{instance.output}.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))

FAMILY_MODULE = "{cls.__module__}"
FAMILY_CLASS = "{cls.__qualname__}"
METHOD = "{method}"
OUTPUT = "{instance.output}"


def main(argv: list[str]) -> int:
    task = Path(argv[1]) if len(argv) > 1 else Path.cwd()
    data = task / "data"
    files = {{
        path.relative_to(data).as_posix(): path.read_bytes()
        for path in sorted(data.rglob("*"))
        if path.is_file()
    }}
    family = getattr(importlib.import_module(FAMILY_MODULE), FAMILY_CLASS)()
    answer = getattr(family, METHOD)(files)
    target = task / "output" / OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(answer)
    print(f"wrote {{len(answer)}} bytes to output/{{OUTPUT}}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _grader(instance: TaskInstance) -> str:
    digest = hashlib.sha256(instance.expected).hexdigest()
    task = f"the {instance.family} task ({instance.difficulty}, seed {instance.seed})"
    return f'''"""Byte-exact grader for {task}.

The expected output is embedded only as its SHA-256 digest and its size, so the bundle does
not reveal it. Runs under pytest (pytest tests/test_outputs.py) or on its own
(python tests/test_outputs.py, exit code 0 when every check passes), with the standard
library only.
"""

import hashlib
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
OUTPUT = TASK_DIR / "output" / "{instance.output}"
EXPECTED_SHA256 = "{digest}"
EXPECTED_SIZE = {len(instance.expected)}


def test_output_file_exists() -> None:
    assert OUTPUT.is_file(), f"output/{{OUTPUT.name}} was not written"


def test_output_matches_byte_for_byte() -> None:
    data = OUTPUT.read_bytes()
    assert len(data) == EXPECTED_SIZE, f"expected {{EXPECTED_SIZE}} bytes, got {{len(data)}}"
    actual = hashlib.sha256(data).hexdigest()
    assert actual == EXPECTED_SHA256, f"sha256 {{actual}} != expected {{EXPECTED_SHA256}}"


def main() -> int:
    failed = 0
    for name, check in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            check()
        except (AssertionError, OSError) as error:
            failed += 1
            print(f"FAIL {{name}}: {{error}}")
        else:
            print(f"PASS {{name}}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _dockerfile(instance: TaskInstance) -> str:
    return f"""# Task image for {instance.family} ({instance.difficulty}, seed {instance.seed}).
# The base image is pinned by digest, so the environment cannot drift. Only the instruction
# and the corpus are baked in; solution/, baseline/ and tests/ are mounted at grading time.
FROM {BASE_IMAGE}
LABEL project=trapforge \\
      trapforge.family="{instance.family}" \\
      trapforge.difficulty="{instance.difficulty}" \\
      trapforge.seed="{instance.seed}"
RUN useradd --create-home --uid 10001 solver \\
    && mkdir -p /task/output \\
    && chown solver /task/output
WORKDIR /task
COPY instruction.md ./
COPY data/ data/
USER solver
"""


def _instruction(instance: TaskInstance) -> str:
    return (
        instance.instruction.rstrip("\n")
        + f"""

## Files

The input files are under `data/`. Write your answer to `output/{instance.output}`; it is
graded byte for byte.
"""
    )


def bundle_files(family: TaskFamily, instance: TaskInstance) -> dict[str, bytes]:
    """Every file of the exported bundle, keyed by relative POSIX path.

    Raises :class:`BundleError` unless the instance's constraint system has exactly one
    solution and it is the planted world: a bundle ships only tasks with a proved answer.
    """
    proof = prove(instance.system)
    if not isinstance(proof, UniqueProof) or dict(proof.solution) != dict(instance.hidden):
        raise BundleError(f"refusing to export: the corpus does not pin the answer ({proof})")
    files = {f"data/{name}": content for name, content in instance.files.items()}
    files["instruction.md"] = text_bytes(_instruction(instance))
    files["Dockerfile"] = text_bytes(_dockerfile(instance))
    files["tests/test_outputs.py"] = text_bytes(_grader(instance))
    files["proof/uniqueness.cert.json"] = text_bytes(proof.certificate_json())
    vendored = _vendored_sources(family)
    for folder, method, role in (
        ("solution", "solve", "Reference solver"),
        ("baseline", "baseline", "Naive baseline (expected to fail the grader)"),
    ):
        files[f"{folder}/solve.py"] = text_bytes(_solver_script(family, instance, method, role))
        files.update({f"{folder}/vendor/{name}": body for name, body in vendored.items()})
    files["task.json"] = json_bytes(
        {
            "family": instance.family,
            "seed": instance.seed,
            "difficulty": instance.difficulty.value,
            "output": instance.output,
            "expected_sha256": hashlib.sha256(instance.expected).hexdigest(),
            "expected_size": len(instance.expected),
            "instance_digest": instance.digest(),
            "base_image": BASE_IMAGE,
        }
    )
    tree_digest(files)  # validates every path
    return dict(sorted(files.items()))


def write_bundle(files: Mapping[str, bytes], out: Path) -> None:
    """Write ``files`` under ``out``, which must not exist yet or be an empty directory."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise BundleError(f"{out} already exists and is not an empty directory")
    for name, content in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
