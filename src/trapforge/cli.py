"""Command-line entry point for TrapForge.

The commands are thin wrappers over the library: ``crt`` over :mod:`trapforge.modular`,
``solve`` over :mod:`trapforge.linalg` and ``system`` over :mod:`trapforge.prover.model`.
Every "no solution" answer is printed together with its certificate and the result of
re-checking that certificate, so the CLI never asks to be trusted.

Exit codes: 0 when the input was valid (whether or not it has a solution), 2 for input that
cannot be read or parsed.
"""

from __future__ import annotations

import itertools
import json
import platform
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from trapforge import __version__
from trapforge.linalg import AffineLattice, LinalgError, Matrix, solve_diophantine
from trapforge.modular import Congruence, ModularError, crt
from trapforge.prover import ConstraintSystem, ModelError

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


@app.command()
def system(
    path: Annotated[Path, typer.Argument(help="Constraint system JSON (the prover's format).")],
    check: Annotated[
        list[str] | None,
        typer.Option(help="Test a candidate assignment, one name=value per option."),
    ] = None,
) -> None:
    """Validate and print a constraint system; optionally test a candidate assignment."""
    try:
        loaded = ConstraintSystem.from_json_dict(_load_json(path))
    except ModelError as error:
        _fail(str(error))
    typer.echo(str(loaded))
    counts = [
        _plural(len(loaded.unknowns), "unknown"),
        _plural(len(loaded.constraints), "constraint"),
        _plural(loaded.case_count, "case"),
    ]
    typer.echo(", ".join(counts))
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


if __name__ == "__main__":  # pragma: no cover
    app()
