import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trapforge import __version__
from trapforge.cli import app

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def write_json(tmp_path: Path, data: object) -> str:
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


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


# -- crt ----------------------------------------------------------------------------------


def test_crt_combines_non_coprime_moduli() -> None:
    code, out = run("crt", "5:12", "5:18")
    assert code == 0
    assert out.splitlines() == [
        "#0: x = 5 (mod 12)",
        "#1: x = 5 (mod 18)",
        "combined: x = 5 (mod 36)",
    ]


def test_crt_reports_a_rechecked_conflict() -> None:
    code, out = run("crt", "1:4", "2:6")
    assert code == 0
    assert "no solution: congruences #0 and #1 are incompatible" in out
    assert out.splitlines()[-1] == "certificate re-checked: True"


def test_crt_canonicalizes_negative_residues() -> None:
    code, out = run("crt", "--", "-1:6")
    assert code == 0
    assert "#0: x = 5 (mod 6)" in out


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("5", "is not of the form residue:modulus"),
        ("a:12", "is not of the form residue:modulus"),
        ("5:0", "'5:0'"),
    ],
)
def test_crt_rejects_bad_input(argument: str, message: str) -> None:
    code, out = run("crt", argument)
    assert code == 2
    assert message in out


# -- solve --------------------------------------------------------------------------------


def test_solve_finds_the_unique_point_of_the_bundled_example() -> None:
    code, out = run("solve", str(EXAMPLES / "checksums.json"))
    assert code == 0
    assert out.splitlines() == [
        "system: 2 equations, 3 unknowns",
        "integer solutions: x = (4, 2, 7) + t0*(290, -699, 67), t in Z^1",
        "  (4, 2, 7)",
        "box (0, 0, 0)..(9, 9, 9): exactly one point in the box (unique)",
    ]


def test_solve_caps_the_listing_of_an_ambiguous_box() -> None:
    code, out = run("solve", str(EXAMPLES / "checksums-ambiguous.json"), "--limit", "2")
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "system: 1 equation, 3 unknowns"
    assert lines[2:] == [
        "  (0, 4, 9)",
        "  (0, 5, 8)",
        "box (0, 0, 0)..(9, 9, 9): more than 2 points in the box (ambiguous)",
    ]


def test_solve_counts_small_ambiguous_and_empty_boxes(tmp_path: Path) -> None:
    system = {"matrix": [[1, 1]], "rhs": [3], "lower": [0, 0], "upper": [1, 5]}
    code, out = run("solve", write_json(tmp_path, system))
    assert code == 0
    assert out.splitlines()[-1] == "box (0, 0)..(1, 5): 2 points in the box (ambiguous)"
    system["upper"] = [1, 1]
    code, out = run("solve", write_json(tmp_path, system))
    assert code == 0
    assert out.splitlines()[-1] == "box (0, 0)..(1, 1): no point in the box"


def test_solve_prints_and_rechecks_an_infeasibility_certificate() -> None:
    code, out = run("solve", str(EXAMPLES / "parity.json"))
    assert code == 0
    assert "no integer solution: weighting the equations by (0, 1)" in out
    assert out.splitlines()[-1] == "certificate re-checked: True"


def test_solve_without_a_box_prints_only_the_lattice(tmp_path: Path) -> None:
    code, out = run("solve", write_json(tmp_path, {"matrix": [[2, 3]], "rhs": [1]}))
    assert code == 0
    assert out.splitlines()[-1].startswith("integer solutions: x = ")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([1, 2], "must hold a JSON object"),
        ({"rhs": [1]}, "matrix must be a non-empty list of rows"),
        ({"matrix": [[1, 2.5]], "rhs": [1]}, "every matrix row must be a list of integers"),
        ({"matrix": [[1, 2]], "rhs": "1"}, "rhs must be a list of integers"),
        ({"matrix": [[1, 2]], "rhs": [1, 2]}, "needs 1 right-hand sides"),
        ({"matrix": [[1, 2], [3]], "rhs": [1, 2]}, "row 1 has 1 entries"),
        ({"matrix": [[1, 2]], "rhs": [1], "lower": [0]}, "upper must be a list of integers"),
        (
            {"matrix": [[1, 2]], "rhs": [1], "lower": [0], "upper": [5]},
            "lower and upper need 2 entries each",
        ),
    ],
)
def test_solve_rejects_malformed_input(tmp_path: Path, data: object, message: str) -> None:
    code, out = run("solve", write_json(tmp_path, data))
    assert code == 2
    assert message in out


def test_solve_reports_unreadable_files(tmp_path: Path) -> None:
    code, out = run("solve", str(tmp_path / "missing.json"))
    assert code == 2
    assert "cannot read" in out
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    code, out = run("solve", str(broken))
    assert code == 2
    assert "is not valid JSON" in out


# -- system -------------------------------------------------------------------------------


def test_system_prints_the_bundled_ledger_system() -> None:
    code, out = run("system", str(EXAMPLES / "ledger-anchors.json"))
    assert code == 0
    assert out.splitlines() == [
        "unknowns: 1 <= a <= 11, 0 <= b <= 11",
        "anchor x=1: a + b = 0 (mod 12)",
        "anchor x=3: 3*a + b = 10 (mod 12)",
        "2 unknowns, 2 constraints, 1 case",
    ]


def test_system_shows_that_a_composite_modulus_admits_two_worlds() -> None:
    path = str(EXAMPLES / "ledger-anchors.json")
    for a, b in [(5, 7), (11, 1)]:
        code, out = run("system", path, "--check", f"a={a}", "--check", f"b={b}")
        assert code == 0
        assert out.splitlines()[-1] == f"a={a}, b={b} satisfies every constraint"
    code, out = run("system", path, "--check", "a=5", "--check", "b=8")
    assert code == 0
    assert out.splitlines()[-1] == "a=5, b=8 violates: anchor x=1, anchor x=3"


def test_system_checks_choices_of_a_case_split() -> None:
    path = str(EXAMPLES / "wrapping-clock.json")
    code, out = run("system", path, "--check", "P=18", "--check", "T=41", "--check", "w=2")
    assert code == 0
    assert "2 unknowns, 2 constraints, 2 cases" in out
    assert out.splitlines()[-1] == "P=18, T=41, w=2 satisfies every constraint"


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        (["a5"], "is not of the form name=integer"),
        (["a=x"], "is not of the form name=integer"),
        (["=5"], "is not of the form name=integer"),
        (["a=5"], "an assignment must give exactly"),
    ],
)
def test_system_rejects_bad_assignments(checks: list[str], message: str) -> None:
    args = ["system", str(EXAMPLES / "ledger-anchors.json")]
    for check in checks:
        args += ["--check", check]
    code, out = run(*args)
    assert code == 2
    assert message in out


def test_system_rejects_a_malformed_system(tmp_path: Path) -> None:
    code, out = run("system", write_json(tmp_path, {"unknowns": []}))
    assert code == 2
    assert "malformed constraint system" in out


# -- prove and check ------------------------------------------------------------------------


def test_prove_finds_the_counterexample_pair_of_the_ledger() -> None:
    code, out = run("prove", str(EXAMPLES / "ledger-anchors.json"))
    assert code == 0
    assert out.splitlines() == [
        "system: 2 unknowns, 2 constraints, 1 case",
        "ambiguous: exactly 2 worlds fit, for example a=5, b=7 and a=11, b=1 (they differ in a, b)",
        "certificate re-checked: True (verified: ambiguous, 2 solutions)",
    ]


def test_prove_counts_the_worlds_of_a_case_split() -> None:
    code, out = run("prove", str(EXAMPLES / "wrapping-clock.json"))
    assert code == 0
    assert "system: 2 unknowns, 2 constraints, 2 cases" in out
    assert "ambiguous: exactly 6 worlds fit" in out


def test_prove_writes_the_bundled_certificate_byte_for_byte(tmp_path: Path) -> None:
    target = tmp_path / "ledger.cert.json"
    code, out = run(
        "prove", str(EXAMPLES / "ledger-anchors-unique.json"), "--certificate", str(target)
    )
    assert code == 0
    assert out.splitlines() == [
        "system: 2 unknowns, 3 constraints, 1 case",
        "unique: a=5, b=7",
        "certificate re-checked: True (verified: unique, 1 solution)",
        f"certificate written to {target}",
    ]
    assert target.read_bytes() == (EXAMPLES / "ledger-anchors-unique.cert.json").read_bytes()


def test_prove_require_unique_sets_the_exit_code() -> None:
    assert run("prove", str(EXAMPLES / "ledger-anchors.json"), "--require-unique")[0] == 1
    unique = str(EXAMPLES / "ledger-anchors-unique.json")
    assert run("prove", unique, "--require-unique")[0] == 0


def test_prove_caps_the_count_and_reports_infeasible_samples(tmp_path: Path) -> None:
    wide = {"choices": [], "unknowns": [{"name": "x", "lower": 0, "upper": 99}], "constraints": []}
    code, out = run("prove", write_json(tmp_path, wide), "--cap", "3")
    assert code == 0
    assert "ambiguous: more than 3 worlds fit, for example x=0 and x=1" in out
    assert "certificate re-checked: True (verified: ambiguous, more than 3)" in out
    wide["constraints"] = [
        {"kind": "equation", "label": "", "terms": [["x", 2]], "rhs": 3},
    ]
    code, out = run("prove", write_json(tmp_path, wide))
    assert code == 0
    assert "infeasible: no world fits every constraint and bound" in out
    assert run("prove", write_json(tmp_path, wide), "--cap", "1")[0] == 2


def test_prove_rejects_bad_systems_and_unwritable_targets(tmp_path: Path) -> None:
    code, out = run("prove", write_json(tmp_path, {"unknowns": []}))
    assert code == 2
    assert "malformed constraint system" in out
    target = tmp_path / "missing" / "cert.json"
    code, out = run("prove", str(EXAMPLES / "ledger-anchors.json"), "--certificate", str(target))
    assert code == 2
    assert f"cannot write {target}" in out


def test_check_verifies_the_bundled_certificate() -> None:
    code, out = run("check", str(EXAMPLES / "ledger-anchors-unique.cert.json"))
    assert code == 0
    assert out == "verified: unique, 1 solution\n"


def test_check_rejects_a_tampered_certificate(tmp_path: Path) -> None:
    data = json.loads((EXAMPLES / "ledger-anchors-unique.cert.json").read_text())
    data["witnesses"] = [{"a": 11, "b": 1}]
    code, out = run("check", write_json(tmp_path, data))
    assert code == 1
    assert out == "invalid: the witnesses are not the first solutions the evidence gives\n"
    data["cases"][0]["basis"][0][0] = 24
    code, out = run("check", write_json(tmp_path, data))
    assert code == 1
    assert out.startswith("invalid: a basis row is not a solution")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert run("check", str(broken))[0] == 2


@pytest.mark.parametrize("command", ["prove", "check", "solve"])
def test_undecodable_or_deeply_nested_files_exit_2_not_1(tmp_path: Path, command: str) -> None:
    # Exit 1 means "the check failed", so input that cannot be parsed must never produce it.
    binary = tmp_path / "bin.json"
    binary.write_bytes(b"\xff\xfe\x00bad")
    code, out = run(command, str(binary))
    assert code == 2
    assert "is not UTF-8 text" in out
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    code, out = run(command, str(deep))
    assert code == 2
    assert "nested too deeply" in out
