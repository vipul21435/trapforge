"""Residue classes and CRT with non-coprime moduli, checked against brute force and sympy."""

import math
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sympy.ntheory.modular import crt as sympy_crt
from sympy.ntheory.modular import solve_congruence

from trapforge.modular import (
    ConflictingCongruences,
    Congruence,
    InvalidModulusError,
    ModularError,
    crt,
    crt_pair,
)

residues = st.integers(min_value=-(10**6), max_value=10**6)
# Products of a few small primes: moduli that share factors far more often than random ones.
smooth_moduli = st.lists(st.sampled_from([2, 2, 3, 3, 5, 7]), max_size=5).map(math.prod)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]


def _same_length(items: list[int]) -> dict[str, int]:
    return {"min_size": len(items), "max_size": len(items)}


def congruences(moduli: st.SearchStrategy[int]) -> st.SearchStrategy[Congruence]:
    return st.builds(Congruence.of, residues, moduli)


tiny_systems = st.lists(congruences(st.integers(min_value=1, max_value=12)), max_size=4)
smooth_systems = st.lists(congruences(smooth_moduli), min_size=1, max_size=6)


def brute_force_solutions(system: list[Congruence]) -> list[int]:
    period = math.lcm(*(c.modulus for c in system))
    return [x for x in range(period) if all(x in c for c in system)]


# -- Congruence ---------------------------------------------------------------------------


@given(residues, st.integers(min_value=1, max_value=10**6))
def test_of_reduces_to_the_canonical_residue(r: int, m: int) -> None:
    c = Congruence.of(r, m)
    assert c == Congruence(r % m, m)
    assert r in c
    assert r + m in c
    assert (r + 1 in c) == (m == 1)


@pytest.mark.parametrize(("r", "m"), [(-1, 5), (5, 5), (7, 3)])
def test_constructor_rejects_non_canonical_residues(r: int, m: int) -> None:
    with pytest.raises(ModularError, match="not canonical"):
        Congruence(r, m)


@pytest.mark.parametrize("m", [0, -4])
def test_non_positive_modulus_is_rejected(m: int) -> None:
    with pytest.raises(InvalidModulusError):
        Congruence(0, m)
    with pytest.raises(InvalidModulusError):
        Congruence.of(3, m)


def test_membership_is_false_for_non_integers() -> None:
    assert "3" not in Congruence(3, 5)
    assert 3.0 not in Congruence(3, 5)


@given(congruences(st.integers(1, 30)), st.integers(min_value=1, max_value=8))
def test_residues_mod_lists_the_class_inside_a_multiple(c: Congruence, k: int) -> None:
    big = c.modulus * k
    assert list(c.residues_mod(big)) == [x for x in range(big) if x in c]
    assert len(c.residues_mod(big)) == k


def test_residues_mod_rejects_non_multiples() -> None:
    with pytest.raises(ModularError, match="not a multiple"):
        Congruence(1, 4).residues_mod(6)
    with pytest.raises(InvalidModulusError):
        Congruence(0, 1).residues_mod(0)


@given(congruences(st.integers(1, 30)), st.integers(-200, 200), st.integers(-200, 200))
def test_values_between_matches_brute_force(c: Congruence, low: int, high: int) -> None:
    assert list(c.values_between(low, high)) == [x for x in range(low, high + 1) if x in c]


# -- CRT ----------------------------------------------------------------------------------


def test_classic_coprime_example() -> None:
    assert crt([Congruence(2, 3), Congruence(3, 5), Congruence(2, 7)]) == Congruence(23, 105)


def test_empty_system_is_every_integer() -> None:
    assert crt([]) == Congruence(0, 1)


def test_non_coprime_pair_combines_to_the_lcm() -> None:
    assert crt_pair(Congruence(1, 4), Congruence(3, 6)) == Congruence(9, 12)


def test_non_coprime_conflict_names_the_pair_and_witness() -> None:
    system = [Congruence(0, 5), Congruence(1, 4), Congruence(2, 7), Congruence(2, 6)]
    result = crt(system)
    assert result == ConflictingCongruences(
        first=1, second=3, witness=2, first_class=system[1], second_class=system[3]
    )
    assert result.verify(system)
    assert str(result) == (
        "congruences #1 and #3 are incompatible: "
        "#1 forces x = 1 (mod 2) but #3 forces x = 0 (mod 2)"
    )


@given(tiny_systems)
def test_crt_matches_brute_force(system: list[Congruence]) -> None:
    solutions = brute_force_solutions(system)
    result = crt(system)
    if solutions:
        # Exactly one class modulo the lcm, so exactly one solution below it.
        assert isinstance(result, Congruence)
        assert solutions == [result.residue]
        assert result.modulus == math.lcm(*(c.modulus for c in system))
    else:
        assert isinstance(result, ConflictingCongruences)
        assert result.first < result.second
        assert result.verify(system)
        pair = [system[result.first], system[result.second]]
        assert brute_force_solutions(pair) == []


@given(smooth_systems)
def test_crt_matches_sympy_on_non_coprime_moduli(system: list[Congruence]) -> None:
    result = crt(system)
    expected = solve_congruence(*((c.residue, c.modulus) for c in system))
    if expected is None:
        assert isinstance(result, ConflictingCongruences)
        assert result.verify(system)
    else:
        assert isinstance(result, Congruence)
        assert (result.residue, result.modulus) == tuple(map(int, expected))


@given(
    st.lists(st.sampled_from(PRIMES), min_size=1, max_size=8, unique=True).flatmap(
        lambda moduli: st.tuples(st.just(moduli), st.lists(residues, **_same_length(moduli)))
    )
)
def test_crt_matches_sympy_crt_on_coprime_moduli(case: tuple[list[int], list[int]]) -> None:
    # sympy's crt() reports the product of the moduli, which equals the lcm only when the
    # moduli are pairwise coprime, so it is used as an oracle on that case alone.
    moduli, values = case
    result = crt(Congruence.of(v, m) for v, m in zip(values, moduli, strict=True))
    assert isinstance(result, Congruence)
    expected = sympy_crt(moduli, values)
    assert expected is not None
    assert (result.residue, result.modulus) == tuple(map(int, expected))


@given(
    st.integers(min_value=-(10**30), max_value=10**30),
    st.lists(st.integers(min_value=1, max_value=10**9), min_size=1, max_size=8),
)
def test_consistent_systems_recover_the_hidden_value(x: int, moduli: list[int]) -> None:
    result = crt(Congruence.of(x, m) for m in moduli)
    assert isinstance(result, Congruence)
    assert result == Congruence.of(x, math.lcm(*moduli))


@given(st.data(), smooth_systems)
def test_crt_is_independent_of_order(data: st.DataObject, system: list[Congruence]) -> None:
    shuffled = data.draw(st.permutations(system))
    left, right = crt(system), crt(shuffled)
    assert isinstance(left, Congruence) == isinstance(right, Congruence)
    if isinstance(left, Congruence):
        assert left == right


# -- Certificates -------------------------------------------------------------------------


SYSTEM = (Congruence(1, 4), Congruence(2, 6))
PROOF = ConflictingCongruences(0, 1, 2, SYSTEM[0], SYSTEM[1])


def test_the_reference_certificate_is_valid() -> None:
    assert crt(SYSTEM) == PROOF
    assert PROOF.verify(SYSTEM)


@pytest.mark.parametrize(
    "tampered",
    [
        replace(PROOF, witness=1),
        replace(PROOF, witness=4),
        replace(PROOF, second=0),
        replace(PROOF, second=2),
        replace(PROOF, first=-1),
        replace(PROOF, first_class=Congruence(0, 4)),
        replace(PROOF, second_class=Congruence(1, 6)),
    ],
)
def test_verify_rejects_tampered_certificates(tampered: ConflictingCongruences) -> None:
    assert not tampered.verify(SYSTEM)


def test_verify_rejects_a_consistent_pair() -> None:
    consistent = (Congruence(1, 4), Congruence(3, 6))
    fake = ConflictingCongruences(0, 1, 2, consistent[0], consistent[1])
    assert not fake.verify(consistent)
