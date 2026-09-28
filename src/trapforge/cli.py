"""Command-line entry point for TrapForge.

The commands are thin wrappers over the library: ``crt`` over :mod:`trapforge.modular`,
``solve`` over :mod:`trapforge.linalg`, ``system`` over :mod:`trapforge.prover.model`, and
``prove`` and ``check`` over the prover's solver and its standalone certificate checker, and
``families`` and ``generate`` over the task-family registry.
Every "no solution" answer and every proof is printed together with the result of
re-checking its certificate, so the CLI never asks to be trusted.

Exit codes: 0 when the input was valid (whether or not it has a solution), 1 when a check
fails (``check`` on a certificate that does not hold, ``prove --require-unique`` on a sample
that is not unique, ``generate`` on an instance that is not unique, not solved by its
reference solver or not missed by its baseline), 2 for input that cannot be read or parsed.
"""

from __future__ import annotations

import itertools
import json
import platform
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from trapforge import __version__
from trapforge.families import REGISTRY, Difficulty, FamilyError
from trapforge.linalg import AffineLattice, LinalgError, Matrix, solve_diophantine
from trapforge.modular import Congruence, ModularError, crt
from trapforge.prover import (
    DEFAULT_CAP,
    ConstraintSystem,
    ModelError,
    UniqueProof,
    check_certificate,
    prove,
)

app = typer.Typer(
    name="trapforge",
    help="Author verifiable, adversarial data tasks for AI agents.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"trapforge {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Author verifiable, adversarial data tasks for AI agents."""


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=2)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        _fail(f"cannot read {path}: {error.strerror or error}")
    except json.JSONDecodeError as error:
        _fail(f"{path} is not valid JSON: {error}")


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _int_list(value: Any, what: str) -> list[int]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        _fail(f"{what} must be a list of integers")
    return value


@app.command()
def info() -> None:
    """Print the package version and the Python it runs on."""
    typer.echo(f"trapforge {__version__} on Python {platform.python_version()}")


@app.command("crt")
def crt_command(
    congruences: Annotated[
        list[str],
        typer.Argument(help="Congruences x = r (mod m), written r:m, e.g. 5:12 5:18."),
    ],
) -> None:
    """Combine congruences with any moduli (coprime or not) into one residue class."""
    parsed: list[Congruence] = []
    for text in congruences:
        residue, sep, modulus = text.partition(":")
        try:
            if not sep:
                raise ValueError
            parsed.append(Congruence.of(int(residue), int(modulus)))
        except ModularError as error:
            _fail(f"{text!r}: {error}")
        except ValueError:
            _fail(f"{text!r} is not of the form residue:modulus")
    for index, congruence in enumerate(parsed):
        typer.echo(f"#{index}: {congruence}")
    result = crt(parsed)
    if isinstance(result, Congruence):
        typer.echo(f"combined: {result}")
        return
    typer.echo(f"no solution: {result}")
    typer.echo(f"certificate re-checked: {result.verify(parsed)}")


def _verdict(count: int, limit: int) -> str:
    if count == 0:
        return "no point in the box"
    if count == 1:
        return "exactly one point in the box (unique)"
    if count > limit:
        return f"more than {limit} points in the box (ambiguous)"
    return f"{count} points in the box (ambiguous)"


@app.command()
def solve(
    path: Annotated[Path, typer.Argument(help="JSON file with matrix, rhs, lower, upper.")],
    limit: Annotated[
        int, typer.Option(min=1, help="Print at most this many points of the box.")
    ] = 10,
) -> None:
    """Solve an integer system A x = b exactly and list its solutions inside a box.

    The file holds {"matrix": [[...]], "rhs": [...]} and optionally "lower" and "upper"
    bounds per unknown. Without a solution, the certificate is printed and re-checked.
    """
    data = _load_json(path)
    if not isinstance(data, dict):
        _fail(f"{path} must hold a JSON object")
    rows = data.get("matrix")
    if not isinstance(rows, list) or not rows:
        _fail("matrix must be a non-empty list of rows")
    matrix_rows = [_int_list(row, "every matrix row") for row in rows]
    rhs = _int_list(data.get("rhs"), "rhs")
    try:
        matrix = Matrix.of(matrix_rows)
        result = solve_diophantine(matrix, rhs)
    except LinalgError as error:
        _fail(str(error))
    typer.echo(f"system: {_plural(matrix.nrows, 'equation')}, {_plural(matrix.ncols, 'unknown')}")
    if not isinstance(result, AffineLattice):
        typer.echo(str(result))
        typer.echo(f"certificate re-checked: {result.verify(matrix, rhs)}")
        return
    typer.echo(f"integer solutions: {result}")
    if "lower" not in data and "upper" not in data:
        return
    lower = _int_list(data.get("lower"), "lower")
    upper = _int_list(data.get("upper"), "upper")
    if len(lower) != matrix.ncols or len(upper) != matrix.ncols:
        _fail(f"lower and upper need {matrix.ncols} entries each")
    points = list(itertools.islice(result.points_in_box(lower, upper), limit + 1))
    for point in points[:limit]:
        typer.echo(f"  {point}")
    typer.echo(f"box {tuple(lower)}..{tuple(upper)}: {_verdict(len(points), limit)}")


def _parse_assignment(pairs: list[str]) -> dict[str, int]:
    assignment: dict[str, int] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        try:
            if not sep or not name:
                raise ValueError
            assignment[name.strip()] = int(value)
        except ValueError:
            _fail(f"{pair!r} is not of the form name=integer")
    return assignment


def _load_system(path: Path) -> ConstraintSystem:
    try:
        return ConstraintSystem.from_json_dict(_load_json(path))
    except ModelError as error:
        _fail(str(error))


def _summary(loaded: ConstraintSystem) -> str:
    counts = [
        _plural(len(loaded.unknowns), "unknown"),
        _plural(len(loaded.constraints), "constraint"),
        _plural(loaded.case_count, "case"),
    ]
    return ", ".join(counts)


@app.command()
def system(
    path: Annotated[Path, typer.Argument(help="Constraint system JSON (the prover's format).")],
    check: Annotated[
        list[str] | None,
        typer.Option(help="Test a candidate assignment, one name=value per option."),
    ] = None,
) -> None:
    """Validate and print a constraint system; optionally test a candidate assignment."""
    loaded = _load_system(path)
    typer.echo(str(loaded))
    typer.echo(_summary(loaded))
    if not check:
        return
    assignment = _parse_assignment(check)
    try:
        broken = loaded.violations(assignment)
    except ModelError as error:
        _fail(str(error))
    shown = ", ".join(f"{name}={value}" for name, value in assignment.items())
    if broken:
        typer.echo(f"{shown} violates: {', '.join(broken)}")
    else:
        typer.echo(f"{shown} satisfies every constraint")


@app.command("prove")
def prove_command(
    path: Annotated[Path, typer.Argument(help="Constraint system JSON (the prover's format).")],
    cap: Annotated[
        int, typer.Option(min=2, help="Count at most this many worlds before giving up.")
    ] = DEFAULT_CAP,
    certificate: Annotated[
        Path | None, typer.Option(help="Write the JSON certificate to this file.")
    ] = None,
    require_unique: Annotated[
        bool, typer.Option("--require-unique", help="Exit with code 1 unless exactly one fits.")
    ] = False,
) -> None:
    """Decide how many hidden worlds fit a constraint system, with a re-checked certificate.

    Prints the verdict (unique, ambiguous with a counterexample pair and the size of the
    ambiguity space, or infeasible) and the result of re-checking its certificate with the
    standalone checker.
    """
    loaded = _load_system(path)
    typer.echo(f"system: {_summary(loaded)}")
    proof = prove(loaded, cap=cap)
    typer.echo(str(proof))
    check = proof.check()
    typer.echo(f"certificate re-checked: {check.valid} ({check.reason})")
    if certificate is not None:
        try:
            with certificate.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(proof.certificate_json())
        except OSError as error:
            _fail(f"cannot write {certificate}: {error.strerror or error}")
        typer.echo(f"certificate written to {certificate}")
    if require_unique and not isinstance(proof, UniqueProof):
        raise typer.Exit(code=1)


@app.command("check")
def check_command(
    path: Annotated[Path, typer.Argument(help="Certificate JSON written by prove.")],
) -> None:
    """Re-check a uniqueness certificate from scratch, without running the solver.

    Exits with code 1 when the certificate does not hold.
    """
    result = check_certificate(_load_json(path))
    if not result.valid:
        typer.echo(f"invalid: {result.reason}")
        raise typer.Exit(code=1)
    typer.echo(result.reason)


@app.command()
def families() -> None:
    """List the registered task families."""
    for family in REGISTRY:
        typer.echo(f"{family.name}: {family.summary}")


def _lines(data: bytes) -> list[bytes]:
    return data.splitlines()


@app.command()
def generate(
    family: Annotated[str, typer.Argument(help="A family name from `trapforge families`.")],
    seed: Annotated[int, typer.Option(min=0, help="Non-negative seed.")] = 0,
    difficulty: Annotated[str, typer.Option(help="easy, medium or hard.")] = "easy",
    out: Annotated[
        Path | None, typer.Option(help="Write the instance's files under this directory.")
    ] = None,
) -> None:
    """Generate one task instance, prove it unique and run its reference and baseline.

    Exits with code 1 unless the constraint system the corpus implies has exactly one
    solution (the planted world), the reference solver reproduces the expected output byte
    for byte, and the naive baseline gets it wrong while matching the visible sample.
    """
    try:
        level = Difficulty.parse(difficulty)
        instance = REGISTRY.generate(family, seed, level)
    except FamilyError as error:
        _fail(str(error))
    found = REGISTRY.get(family)
    typer.echo(f"{instance.family} / {instance.difficulty} / seed {instance.seed}")
    for name, value in instance.extras.items():
        typer.echo(f"  {name}: {value}")
    proof = prove(instance.system)
    unique = isinstance(proof, UniqueProof) and dict(proof.solution) == dict(instance.hidden)
    typer.echo(f"proof: {proof} (certificate re-checked: {proof.check().valid})")
    solved = found.solve(instance.files) == instance.expected
    typer.echo(f"reference: {'matches' if solved else 'DIFFERS FROM'} the expected output")
    naive = found.baseline(instance.files)
    expected, guessed = _lines(instance.expected), _lines(naive)
    wrong = sum(1 for a, b in zip(expected, guessed, strict=False) if a != b)
    wrong += abs(len(expected) - len(guessed))
    keeps_sample = set(_lines(instance.sample)) <= set(guessed)
    typer.echo(
        f"baseline: {wrong} of {len(expected)} output lines differ; "
        f"{'matches' if keeps_sample else 'misses'} the visible sample"
    )
    typer.echo(f"sha256: {instance.digest()}")
    if out is not None:
        bundle = instance.bundle()
        try:
            for name, content in bundle.items():
                target = out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        except OSError as error:
            _fail(f"cannot write under {out}: {error.strerror or error}")
        typer.echo(f"wrote {_plural(len(bundle), 'file')} under {out}")
    if not (unique and solved and wrong and keeps_sample):
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
