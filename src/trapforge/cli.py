"""Command-line entry point for TrapForge."""

from __future__ import annotations

import platform
from typing import Annotated

import typer

from trapforge import __version__

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


@app.command()
def info() -> None:
    """Print the package version and the Python it runs on."""
    typer.echo(f"trapforge {__version__} on Python {platform.python_version()}")


if __name__ == "__main__":  # pragma: no cover
    app()
