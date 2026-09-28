"""Exact modular arithmetic over Python integers.

This module is the number-theory floor that the rest of TrapForge stands on. Task families
hide structure behind residues (an affine map mod m, counters that wrap at unknown periods),
and the uniqueness prover has to decide exactly which hidden parameters survive the observed
data. Everything here is therefore exact, pure Python ``int`` arithmetic with no floats and
no third-party dependencies, so that exported reference solvers can vendor it unchanged.

Conventions:

* A modulus is a positive integer. ``m == 1`` is allowed and describes the trivial ring in
  which every integer is congruent to every other; it is the identity element for CRT.
* Residues returned by this module are canonical, i.e. in ``range(0, m)``.
* Errors raised for invalid input are subclasses of :class:`ModularError` (a
  ``ValueError``), so callers can catch the whole family at once.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "Bezout",
    "InvalidModulusError",
    "ModularError",
    "NotInvertibleError",
    "extended_gcd",
    "mod_inverse",
]


class ModularError(ValueError):
    """Base class for invalid input to the modular arithmetic core."""


class InvalidModulusError(ModularError):
    """Raised when a modulus is not a positive integer."""

    def __init__(self, modulus: int) -> None:
        super().__init__(f"modulus must be a positive integer, got {modulus}")
        self.modulus = modulus


class NotInvertibleError(ModularError):
    """Raised when ``a`` has no multiplicative inverse modulo ``modulus``.

    An inverse exists exactly when ``gcd(a, modulus) == 1``; ``gcd`` holds the common factor
    that rules it out, which is itself a proof: ``a * x`` is always a multiple of ``gcd``
    modulo ``modulus``, so it can never be congruent to 1.
    """

    def __init__(self, a: int, modulus: int, gcd: int) -> None:
        super().__init__(f"{a} has no inverse modulo {modulus}: gcd({a}, {modulus}) = {gcd} != 1")
        self.a = a
        self.modulus = modulus
        self.gcd = gcd


class Bezout(NamedTuple):
    """Result of :func:`extended_gcd`: ``a * x + b * y == gcd`` with ``gcd >= 0``."""

    gcd: int
    x: int
    y: int


def _require_modulus(modulus: int) -> None:
    if modulus < 1:
        raise InvalidModulusError(modulus)


def extended_gcd(a: int, b: int) -> Bezout:
    """Return ``Bezout(g, x, y)`` with ``g = gcd(a, b) >= 0`` and ``a * x + b * y == g``.

    Works for any signs, including zeros (``extended_gcd(0, 0) == Bezout(0, 1, 0)``). The
    coefficients are the ones produced by the iterative Euclidean algorithm run on
    ``|a|`` and ``|b|``, which keeps them small: whenever ``g > 0``,
    ``|x| <= max(1, |b| // (2 * g))`` and ``|y| <= max(1, |a| // (2 * g))``.

    >>> extended_gcd(240, 46)
    Bezout(gcd=2, x=-9, y=47)
    >>> extended_gcd(-6, 4)
    Bezout(gcd=2, x=-1, y=-1)
    """
    sign_a = -1 if a < 0 else 1
    sign_b = -1 if b < 0 else 1
    old_r, r = abs(a), abs(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return Bezout(old_r, sign_a * old_s, sign_b * old_t)


def mod_inverse(a: int, modulus: int) -> int:
    """Return the unique ``x`` in ``range(modulus)`` with ``a * x % modulus == 1 % modulus``.

    Raises :class:`NotInvertibleError` (carrying the offending gcd) when
    ``gcd(a, modulus) != 1`` and :class:`InvalidModulusError` when ``modulus < 1``. Modulo 1
    every integer is invertible and the inverse is 0, matching ``pow(a, -1, 1)``.

    >>> mod_inverse(3, 7)
    5
    >>> mod_inverse(-3, 7)
    2
    """
    _require_modulus(modulus)
    g, x, _ = extended_gcd(a % modulus, modulus)
    if g != 1:
        raise NotInvertibleError(a, modulus, g)
    return x % modulus
