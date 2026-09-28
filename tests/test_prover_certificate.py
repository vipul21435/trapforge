"""The standalone certificate checker: it accepts true certificates and nothing else."""

import ast
import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from _prover_strategies import brute_force, small_systems
from trapforge.prover import (
    ChoiceTerm,
    Congruent,
    ConstraintSystem,
    Equation,
    check_certificate,
    prove,
)
from trapforge.prover import certificate as certificate_module

# Two cases: P = 4 is unsolvable, P = 3 leaves x = 4 + 6t, a one-dimensional lattice.
SPLIT = ConstraintSystem.build(
    {"x": (0, 20), "y": (0, 3)},
    [Congruent.of({"x": 1}, 1, ChoiceTerm("P"), "c0"), Congruent.of({"x": 1}, 0, 2, "c1")],
    {"P": (4, 3)},
)


def split_certificate() -> dict[str, Any]:
    return copy.deepcopy(dict(prove(SPLIT).certificate))


def lattice_case(cert: dict[str, Any]) -> dict[str, Any]:
    case: dict[str, Any] = cert["cases"][1]
    assert case["evidence"] == "lattice"
    return case


def test_the_split_certificate_is_valid_to_begin_with() -> None:
    check = check_certificate(split_certificate())
    assert check.valid
    assert bool(check)
    assert (check.verdict, check.count, check.exact) == ("ambiguous", 12, True)
    assert check.reason == "verified: ambiguous, 12 solutions"


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(cert: dict[str, Any]) -> None:
        target: Any = cert
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _swap_basis_rows(cert: dict[str, Any]) -> None:
    # Two-row basis for a two-dimensional lattice, kept consistent with its right inverse.
    case = lattice_case(cert)
    case["basis"].reverse()
    for row in case["right_inverse"]:
        row.reverse()


def _negate_basis_row(cert: dict[str, Any]) -> None:
    case = lattice_case(cert)
    case["basis"][0] = [-value for value in case["basis"][0]]
    for row in case["right_inverse"]:
        row[0] = -row[0]


def _double_basis_row(cert: dict[str, Any]) -> None:
    case = lattice_case(cert)
    case["basis"][0] = [2 * value for value in case["basis"][0]]


def _drop_basis_row(cert: dict[str, Any]) -> None:
    case = lattice_case(cert)
    case["basis"].pop()
    for row in case["right_inverse"]:
        row.pop()


def _drop_case(cert: dict[str, Any]) -> None:
    cert["cases"].pop()


def _swap_cases(cert: dict[str, Any]) -> None:
    cert["cases"].reverse()


MUTATIONS: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("format", _set(("format",), "other"), "not a trapforge-uniqueness-certificate"),
    ("version", _set(("version",), 2), "version 1"),
    ("cap", _set(("cap",), 1), "cap must be at least 2"),
    ("cap type", _set(("cap",), "9"), "cap must be an integer"),
    ("missing case", _drop_case, "must cover all 2 cases"),
    ("case order", _swap_cases, "case 0 should be {'P': 4}"),
    ("evidence kind", _set(("cases", 1, "evidence"), "trust me"), "unknown evidence kind"),
    ("point", _set(("cases", 1, "point", 0), 5), "point does not satisfy"),
    ("point type", _set(("cases", 1, "point", 0), True), "point must be an integer"),
    ("point length", _set(("cases", 1, "point"), [4]), "point must have 4 entries"),
    ("basis type", _set(("cases", 1, "basis"), "rows"), "basis must be a list"),
    ("basis row", _set(("cases", 1, "basis", 0, 0), 7), "not a solution of the homogeneous"),
    ("sublattice", _double_basis_row, "basis times right_inverse is not the identity"),
    ("too few rows", _drop_basis_row, "must be a 3x3 minor"),
    ("row order", _swap_basis_rows, "not in echelon form"),
    ("sign", _negate_basis_row, "not in echelon form"),
    ("inverse rows", _set(("cases", 1, "right_inverse"), [[1, 0]]), "must have 4 rows"),
    ("minor repeat", _set(("cases", 1, "minor", "rows"), [0, 0]), "repeats a row or a column"),
    ("minor range", _set(("cases", 1, "minor", "columns"), [0, 9]), "points outside"),
    ("minor singular", _set(("cases", 1, "minor", "columns"), [1, 3]), "minor is singular"),
    ("multiplier", _set(("cases", 0, "multiplier"), [1, 0]), "do not cancel"),
    ("modulus one", _set(("cases", 0, "modulus"), 1), "must be 0 or at least 2"),
    ("no proof", _set(("cases", 0, "multiplier"), [0, 0]), "nothing is proved"),
    ("verdict", _set(("verdict",), "unique"), "unique with 12 solutions, but"),
    ("count", _set(("count",), 7), "claims ambiguous with 7 solutions"),
    ("exact", _set(("exact",), False), "claims ambiguous with more than 12"),
    ("witness", _set(("witnesses", 1, "y"), 3), "witnesses are not the first"),
    ("choice term", _set(("system", "constraints", 0, "modulus", "choice"), "Q"), "unknown choice"),
    ("scalar", _set(("system", "unknowns", 0, "upper"), 2.5), "not an integer or a choice term"),
    ("modulus", _set(("system", "constraints", 1, "modulus"), 0), "modulus below 1"),
    ("kind", _set(("system", "constraints", 1, "kind"), "guess"), "unknown constraint kind"),
    ("undeclared", _set(("system", "constraints", 1, "terms", 0, 0), "z"), "undeclared unknown"),
    ("same name", _set(("system", "unknowns", 1, "name"), "x"), "used twice"),
    ("name clash", _set(("system", "unknowns", 1, "name"), "P"), "used twice"),
    ("name type", _set(("system", "unknowns", 1, "name"), 3), "names must be strings"),
    ("options", _set(("system", "choices", 0, "options"), [4, 4]), "lists an option twice"),
    ("options type", _set(("system", "choices", 0, "options"), 4), "options must be a list"),
    ("exact type", _set(("exact",), 1), "exact must be true or false"),
    ("count type", _set(("count",), True), "count must be an integer"),
    ("missing key", _set(("cases", 1, "minor"), {}), "malformed certificate"),
]


@pytest.mark.parametrize(
    ("mutate", "reason"), [(m, r) for _, m, r in MUTATIONS], ids=[n for n, _, _ in MUTATIONS]
)
def test_every_tampered_certificate_is_rejected(
    mutate: Callable[[dict[str, Any]], None], reason: str
) -> None:
    cert = split_certificate()
    if mutate in (_swap_basis_rows, _negate_basis_row, _drop_basis_row):
        cert = two_dimensional_certificate()
    mutate(cert)
    check = check_certificate(cert)
    assert not check.valid
    assert not check
    assert reason in check.reason
    assert (check.verdict, check.count, check.exact) == (None, None, None)


def two_dimensional_certificate() -> dict[str, Any]:
    # x - 3y = 0 (mod 6) in a box: a lattice with two basis rows, second case of a split.
    system = ConstraintSystem.build(
        {"x": (0, 11), "y": (0, 5)},
        [Congruent.of({"x": 1}, 1, ChoiceTerm("P")), Congruent.of({"x": 1, "y": -3}, 0, 6)],
        {"P": (4, 1)},
    )
    cert = copy.deepcopy(dict(prove(system).certificate))
    assert len(lattice_case(cert)["basis"]) == 2
    return cert


def test_text_that_is_not_json_is_malformed() -> None:
    check = check_certificate("{not json")
    assert not check.valid
    assert check.reason.startswith("malformed certificate")


def test_a_rational_infeasibility_certificate_is_accepted() -> None:
    system = ConstraintSystem.build(
        {"x": (0, 9)}, [Equation.of({"x": 1}, 1), Equation.of({"x": 2}, 3)]
    )
    proof = prove(system)
    (case,) = proof.certificate["cases"]
    assert case["modulus"] == 0
    assert proof.check().reason == "verified: infeasible, 0 solutions"


def test_the_checker_imports_only_the_standard_library() -> None:
    tree = ast.parse(Path(certificate_module.__file__).read_text())
    imported = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    stdlib = {"__future__", "itertools", "json", "collections.abc", "dataclasses", "typing"}
    assert imported == stdlib


def _integer_paths(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    if type(value) is int:
        return [path]
    if isinstance(value, dict):
        return [p for key in value for p in _integer_paths(value[key], (*path, key))]
    if isinstance(value, list):
        return [p for i, item in enumerate(value) for p in _integer_paths(item, (*path, i))]
    return []


@given(small_systems(), st.data())
def test_whatever_the_checker_accepts_is_true(
    system: ConstraintSystem, data: st.DataObject
) -> None:
    """Soundness: perturb one integer of the evidence or the claims; if the checker still
    accepts, its verdict and count must be the truth (only the system itself is trusted)."""
    truth = brute_force(system)
    cert = copy.deepcopy(dict(prove(system, cap=10_000).certificate))
    targets = [
        *_integer_paths(cert["cases"], ("cases",)),
        ("count",),
        *_integer_paths(cert["witnesses"], ("witnesses",)),
    ]
    path = data.draw(st.sampled_from(targets))
    delta = data.draw(st.sampled_from([-2, -1, 1, 2, 6]))
    target: Any = cert
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] += delta
    check = check_certificate(json.loads(json.dumps(cert)))
    if check.valid:
        assert check.count == len(truth)
        assert cert["witnesses"] == truth[:2]
