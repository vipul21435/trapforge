"""Extended gcd and modular inverse, checked against brute force, Python and sympy."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sympy import mod_inverse as sympy_mod_inverse
from sympy.core.intfunc import igcdex

from trapforge.modular import (
    Bezout,
    InvalidModulusError,
    ModularError,
    NotInvertibleError,
    extended_gcd,
    mod_inverse,
)

small = st.integers(min_value=-60, max_value=60)
big = st.integers(min_value=-(10**40), max_value=10**40)
small_moduli = st.integers(min_value=1, max_value=60)


@given(big, big)
def test_extended_gcd_satisfies_bezout_identity(a: int, b: int) -> None:
    g, x, y = extended_gcd(a, b)
    assert g == math.gcd(a, b)
    assert a * x + b * y == g


@given(big, big)
def test_extended_gcd_coefficients_are_small(a: int, b: int) -> None:
    g, x, y = extended_gcd(a, b)
    if g == 0:
        assert (x, y) == (1, 0)
    else:
        assert abs(x) <= max(1, abs(b) // (2 * g))
        assert abs(y) <= max(1, abs(a) // (2 * g))


@given(big, big)
def test_extended_gcd_agrees_with_sympy_gcd(a: int, b: int) -> None:
    _, _, sympy_g = igcdex(a, b)
    assert extended_gcd(a, b).gcd == sympy_g


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (0, 0, Bezout(0, 1, 0)),
        (0, 5, Bezout(5, 0, 1)),
        (0, -5, Bezout(5, 0, -1)),
        (7, 0, Bezout(7, 1, 0)),
        (-7, 0, Bezout(7, -1, 0)),
        (240, 46, Bezout(2, -9, 47)),
        (1, 1, Bezout(1, 0, 1)),
    ],
)
def test_extended_gcd_edge_cases(a: int, b: int, expected: Bezout) -> None:
    assert extended_gcd(a, b) == expected


@given(small, small_moduli)
def test_mod_inverse_matches_brute_force(a: int, m: int) -> None:
    candidates = [x for x in range(m) if (a * x - 1) % m == 0]
    if candidates:
        assert mod_inverse(a, m) == candidates[0]
        assert len(candidates) == 1
    else:
        with pytest.raises(NotInvertibleError) as info:
            mod_inverse(a, m)
        assert info.value.gcd == math.gcd(a, m) > 1
        assert (info.value.a, info.value.modulus) == (a, m)


@given(big, st.integers(min_value=2, max_value=10**30))
def test_mod_inverse_matches_python_and_sympy(a: int, m: int) -> None:
    if math.gcd(a, m) != 1:
        with pytest.raises(NotInvertibleError):
            mod_inverse(a, m)
        with pytest.raises(ValueError, match="does not exist"):
            sympy_mod_inverse(a, m)
        return
    inverse = mod_inverse(a, m)
    assert 0 <= inverse < m
    assert inverse == pow(a, -1, m) == int(sympy_mod_inverse(a, m))


@given(big)
def test_everything_is_invertible_modulo_one(a: int) -> None:
    # Z/1Z is the zero ring; Python agrees (pow(a, -1, 1) == 0) while sympy refuses m == 1.
    assert mod_inverse(a, 1) == 0 == pow(a, -1, 1)


@pytest.mark.parametrize("m", [0, -1, -12])
def test_non_positive_modulus_is_rejected(m: int) -> None:
    with pytest.raises(InvalidModulusError, match="positive") as info:
        mod_inverse(3, m)
    assert info.value.modulus == m


def test_errors_share_a_catchable_base() -> None:
    assert issubclass(NotInvertibleError, ModularError)
    assert issubclass(InvalidModulusError, ModularError)
    assert issubclass(ModularError, ValueError)
    with pytest.raises(ModularError, match=r"gcd\(6, 9\) = 3"):
        mod_inverse(6, 9)
