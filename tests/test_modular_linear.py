"""Linear congruences and one-unknown systems, checked against brute force and sympy."""

import math
from dataclasses import replace

import pytest
from hypothesis import assume, event, given
from hypothesis import strategies as st
from sympy.ntheory.modular import solve_congruence
from sympy.ntheory.residue_ntheory import linear_congruence

from trapforge.modular import (
    ConflictingCongruences,
    Congruence,
    Inconsistency,
    InvalidModulusError,
    LinearCongruence,
    UnsolvableCongruence,
    solve_linear_congruence,
    solve_linear_system,
)

small = st.integers(min_value=-60, max_value=60)
small_moduli = st.integers(min_value=1, max_value=60)
big = st.integers(min_value=-(10**30), max_value=10**30)
big_moduli = st.integers(min_value=1, max_value=10**24)

tiny_constraints = st.one_of(
    st.builds(LinearCongruence, small, small, st.integers(min_value=1, max_value=12)),
    st.builds(Congruence.of, small, st.integers(min_value=1, max_value=12)),
)
tiny_systems = st.lists(tiny_constraints, max_size=4)


def brute_force(system: list[Congruence | LinearCongruence]) -> list[int]:
    period = math.lcm(*(c.modulus for c in system))
    return [x for x in range(period) if all(x in c for c in system)]


# -- a single congruence ------------------------------------------------------------------


@given(small, small, small_moduli)
def test_single_congruence_matches_brute_force(a: int, b: int, m: int) -> None:
    solutions = [x for x in range(m) if (a * x - b) % m == 0]
    result = solve_linear_congruence(a, b, m)
    g = math.gcd(a, m)
    if solutions:
        assert isinstance(result, Congruence)
        assert list(result.residues_mod(m)) == solutions
        assert len(solutions) == g
    else:
        assert result == UnsolvableCongruence(index=0, divisor=g)
        assert result.verify([LinearCongruence(a, b, m)])


@given(st.integers(0, 10**4), st.integers(0, 10**4), st.integers(min_value=2, max_value=2000))
def test_single_congruence_matches_sympy(a: int, b: int, m: int) -> None:
    expected = sorted(int(x) for x in linear_congruence(a % m, b % m, m))
    result = solve_linear_congruence(a, b, m)
    if expected:
        assert isinstance(result, Congruence)
        assert list(result.residues_mod(m)) == expected
    else:
        assert isinstance(result, UnsolvableCongruence)


def sympy_solutions(a: int, b: int, m: int) -> list[int]:
    """sympy's list of solutions modulo m; callers keep gcd(a, m) small (it lists all)."""
    return sorted(int(x) for x in linear_congruence(a % m, b % m, m))


@given(
    st.integers(min_value=0, max_value=10**30),
    st.integers(min_value=0, max_value=10**30),
    st.integers(min_value=2, max_value=10**30),
)
def test_single_congruence_matches_sympy_on_large_moduli(a: int, b: int, m: int) -> None:
    assume(a % m != 0 and math.gcd(a, m) <= 1000)
    expected = sympy_solutions(a, b, m)
    result = solve_linear_congruence(a, b, m)
    if expected:
        assert isinstance(result, Congruence)
        assert list(result.residues_mod(m)) == expected
    else:
        assert isinstance(result, UnsolvableCongruence)


@given(
    st.integers(min_value=-(10**30), max_value=10**30),
    st.integers(min_value=1, max_value=10**12),
    st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=10**30),
            st.integers(min_value=1, max_value=10**9),
            st.sampled_from([0, 0, 0, 1]),
        ),
        min_size=1,
        max_size=5,
    ),
)
def test_system_matches_sympy_on_large_moduli(
    x: int, shared: int, rows: list[tuple[int, int, int]]
) -> None:
    # sympy has no solver for simultaneous a_i*x = b_i (mod m_i), so the oracle is built
    # from two sympy functions: linear_congruence turns each constraint into its class of
    # solutions modulo m_i / gcd(a_i, m_i), and solve_congruence intersects the classes.
    system = [LinearCongruence(a, a * x + nudge, shared * cofactor) for a, cofactor, nudge in rows]
    assume(
        all(
            a % c.modulus and math.gcd(a, c.modulus) <= 1000
            for (a, _, _), c in zip(rows, system, strict=True)
        )
    )
    result = solve_linear_system(system)
    classes = []
    for index, c in enumerate(system):
        solutions = sympy_solutions(c.a, c.b, c.modulus)
        if not solutions:
            event("a constraint is unsolvable on its own")
            assert result == UnsolvableCongruence(index, math.gcd(c.a, c.modulus))
            return
        classes.append((solutions[0], c.modulus // math.gcd(c.a, c.modulus)))
    expected = solve_congruence(*classes)
    event("consistent" if expected else "two constraints conflict")
    if expected is None:
        assert isinstance(result, ConflictingCongruences)
        assert result.verify(system)
    else:
        assert isinstance(result, Congruence)
        assert (result.residue, result.modulus) == tuple(map(int, expected))


@given(big, big, big_moduli)
def test_single_congruence_solution_class_is_exact_for_big_numbers(a: int, b: int, m: int) -> None:
    result = solve_linear_congruence(a, b, m)
    g = math.gcd(a, m)
    if b % g:
        assert result == UnsolvableCongruence(0, g)
    else:
        assert isinstance(result, Congruence)
        assert result.modulus == m // g
        for x in (result.residue, result.residue + result.modulus, result.residue - 7 * m):
            assert (a * x - b) % m == 0


def test_zero_coefficient_accepts_everything_or_nothing() -> None:
    assert solve_linear_congruence(0, 12, 6) == Congruence(0, 1)
    assert solve_linear_congruence(0, 5, 6) == UnsolvableCongruence(0, 6)


def test_invalid_modulus_is_rejected() -> None:
    with pytest.raises(InvalidModulusError):
        solve_linear_congruence(1, 1, 0)
    with pytest.raises(InvalidModulusError):
        LinearCongruence(1, 1, -3)


@given(small, small, small_moduli, small)
def test_linear_congruence_object_agrees_with_the_function(a: int, b: int, m: int, x: int) -> None:
    lc = LinearCongruence(a, b, m)
    assert (x in lc) == ((a * x - b) % m == 0)
    assert lc.solve() == solve_linear_congruence(a, b, m)
    assert str(lc) == f"{a}*x = {b} (mod {m})"
    assert "1" not in lc


# -- systems ------------------------------------------------------------------------------


@given(tiny_systems)
def test_system_matches_brute_force(system: list[Congruence | LinearCongruence]) -> None:
    solutions = brute_force(system)
    result = solve_linear_system(system)
    if solutions:
        assert isinstance(result, Congruence)
        period = math.lcm(*(c.modulus for c in system))
        assert list(result.residues_mod(period)) == solutions
    else:
        assert isinstance(result, Inconsistency)
        assert result.verify(system)
        named = (
            [system[result.index]]
            if isinstance(result, UnsolvableCongruence)
            else [system[result.first], system[result.second]]
        )
        assert brute_force(named) == []


@given(
    big,
    st.lists(st.tuples(big, big_moduli), min_size=1, max_size=6),
)
def test_planted_solution_is_always_recovered(x: int, rows: list[tuple[int, int]]) -> None:
    system = [LinearCongruence(a, a * x + 5 * m, m) for a, m in rows]
    result = solve_linear_system(system)
    assert isinstance(result, Congruence)
    assert x in result
    step = math.lcm(*(m // math.gcd(a, m) for a, m in rows))
    assert result.modulus == step
    for candidate in (result.residue, result.residue + 3 * result.modulus):
        assert all(candidate in c for c in system)


def test_unsolvable_constraints_are_reported_before_pairwise_conflicts() -> None:
    system = [Congruence(0, 2), Congruence(1, 2), LinearCongruence(2, 1, 4)]
    assert solve_linear_system(system) == UnsolvableCongruence(index=2, divisor=2)


def test_conflicts_report_the_reduced_classes() -> None:
    system = [LinearCongruence(2, 2, 8), LinearCongruence(3, 0, 6)]
    result = solve_linear_system(system)
    assert result == ConflictingCongruences(0, 1, 2, Congruence(1, 4), Congruence(0, 2))
    assert result.verify(system)


def test_empty_system_is_every_integer() -> None:
    assert solve_linear_system([]) == Congruence(0, 1)


# -- certificates -------------------------------------------------------------------------


UNSOLVABLE = [Congruence(3, 5), LinearCongruence(4, 3, 6)]
UNSOLVABLE_PROOF = UnsolvableCongruence(index=1, divisor=2)


def test_unsolvable_certificate_is_valid_and_explained() -> None:
    assert solve_linear_system(UNSOLVABLE) == UNSOLVABLE_PROOF
    assert UNSOLVABLE_PROOF.verify(UNSOLVABLE)
    assert str(UNSOLVABLE_PROOF) == (
        "congruence #1 has no solution: "
        "2 divides its coefficient and modulus but not its right-hand side"
    )


@pytest.mark.parametrize(
    "tampered",
    [
        replace(UNSOLVABLE_PROOF, divisor=1),
        replace(UNSOLVABLE_PROOF, divisor=3),
        replace(UNSOLVABLE_PROOF, divisor=4),
        replace(UNSOLVABLE_PROOF, index=0),
        replace(UNSOLVABLE_PROOF, index=2),
        replace(UNSOLVABLE_PROOF, index=-1),
    ],
)
def test_unsolvable_verify_rejects_tampering(tampered: UnsolvableCongruence) -> None:
    assert not tampered.verify(UNSOLVABLE)


CONFLICT = [LinearCongruence(2, 2, 8), LinearCongruence(3, 0, 6)]
CONFLICT_PROOF = ConflictingCongruences(0, 1, 2, Congruence(1, 4), Congruence(0, 2))


@pytest.mark.parametrize(
    "tampered",
    [
        # Right residue but the wrong modulus: not the full solution set of #0.
        replace(CONFLICT_PROOF, first_class=Congruence(1, 8)),
        # Right modulus but a residue that does not solve #1.
        replace(CONFLICT_PROOF, second_class=Congruence(1, 2)),
        replace(CONFLICT_PROOF, witness=4),
    ],
)
def test_conflict_verify_rejects_tampering_on_linear_systems(
    tampered: ConflictingCongruences,
) -> None:
    assert CONFLICT_PROOF.verify(CONFLICT)
    assert not tampered.verify(CONFLICT)
