from typer.testing import CliRunner

from trapforge import __version__
from trapforge.cli import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"trapforge {__version__}"


def test_info_reports_python_version() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Python 3.12" in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
