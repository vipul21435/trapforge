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

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

__all__ = [
    "Bezout",
    "ConflictingCongruences",
    "Congruence",
    "Inconsistency",
    "InvalidModulusError",
    "LinearCongruence",
    "ModularError",
    "NotInvertibleError",
    "UnsolvableCongruence",
    "crt",
    "crt_pair",
    "extended_gcd",
    "mod_inverse",
    "solve_linear_congruence",
    "solve_linear_system",
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


@dataclass(frozen=True, slots=True)
class Congruence:
    """The residue class ``x = residue (mod modulus)``, stored canonically.

    The constructor insists on a canonical residue so that equal classes compare equal;
    use :meth:`Congruence.of` to build one from any integer residue.

    >>> c = Congruence.of(-1, 6)
    >>> c, 11 in c, 12 in c
    (Congruence(residue=5, modulus=6), True, False)
    >>> str(c)
    'x = 5 (mod 6)'
    """

    residue: int
    modulus: int

    def __post_init__(self) -> None:
        _require_modulus(self.modulus)
        if not 0 <= self.residue < self.modulus:
            raise ModularError(
                f"residue {self.residue} is not canonical modulo {self.modulus}; "
                "use Congruence.of() to reduce it"
            )

    @classmethod
    def of(cls, residue: int, modulus: int) -> Congruence:
        """Build the class of ``residue`` modulo ``modulus`` from any integer residue."""
        _require_modulus(modulus)
        return cls(residue % modulus, modulus)

    def __contains__(self, x: object) -> bool:
        return isinstance(x, int) and (x - self.residue) % self.modulus == 0

    def __str__(self) -> str:
        return f"x = {self.residue} (mod {self.modulus})"

    def residues_mod(self, modulus: int) -> range:
        """Every residue in ``range(modulus)`` that belongs to this class, in order.

        ``modulus`` must be a positive multiple of ``self.modulus``; the result then has
        exactly ``modulus // self.modulus`` elements.

        >>> list(Congruence(2, 5).residues_mod(20))
        [2, 7, 12, 17]
        """
        _require_modulus(modulus)
        if modulus % self.modulus:
            raise ModularError(f"{modulus} is not a multiple of the class modulus {self.modulus}")
        return range(self.residue, modulus, self.modulus)

    def values_between(self, low: int, high: int) -> range:
        """Every member ``x`` of the class with ``low <= x <= high``, in increasing order.

        >>> list(Congruence(3, 7).values_between(-10, 20))
        [-4, 3, 10, 17]
        """
        first = low + (self.residue - low) % self.modulus
        return range(first, high + 1, self.modulus)


@dataclass(frozen=True, slots=True)
class LinearCongruence:
    """The constraint ``a * x = b (mod modulus)`` on a single integer unknown ``x``.

    Coefficients are kept exactly as given (not reduced), so proofs about a system can
    point back at the constraint the author wrote.

    >>> lc = LinearCongruence(6, 4, 10)
    >>> str(lc), 4 in lc, 5 in lc
    ('6*x = 4 (mod 10)', True, False)
    """

    a: int
    b: int
    modulus: int

    def __post_init__(self) -> None:
        _require_modulus(self.modulus)

    def __contains__(self, x: object) -> bool:
        return isinstance(x, int) and (self.a * x - self.b) % self.modulus == 0

    def __str__(self) -> str:
        return f"{self.a}*x = {self.b} (mod {self.modulus})"

    def solve(self) -> Congruence | UnsolvableCongruence:
        """Solve this congruence alone; see :func:`solve_linear_congruence`."""
        return solve_linear_congruence(self.a, self.b, self.modulus)


def _as_linear(constraint: Congruence | LinearCongruence) -> tuple[int, int, int]:
    if isinstance(constraint, Congruence):
        return 1, constraint.residue, constraint.modulus
    return constraint.a, constraint.b, constraint.modulus


def _confines(constraint: Congruence | LinearCongruence, cls: Congruence) -> bool:
    """True when the solution set of ``constraint`` is exactly the class ``cls``.

    A solvable ``a * x = b (mod m)`` has exactly one class of solutions modulo
    ``m / gcd(a, m)``, so it suffices that ``cls`` has that modulus and its residue solves
    the congruence.
    """
    a, b, m = _as_linear(constraint)
    return cls.modulus * math.gcd(a, m) == m and (a * cls.residue - b) % m == 0


@dataclass(frozen=True, slots=True)
class ConflictingCongruences:
    """A checkable proof that two congruences of a system have no common solution.

    Congruence number ``first`` of the system confines every solution to ``first_class``
    and number ``second`` confines it to ``second_class``. Both class moduli are multiples
    of ``witness``, yet the classes disagree modulo ``witness``, so no integer lies in
    both. Because a system of congruences is solvable exactly when every pair of them is
    (the generalized Chinese remainder theorem), an unsolvable system always has such a
    pair. :meth:`verify` re-checks the proof with one gcd, one product and a few
    remainders per congruence, without solving anything.
    """

    first: int
    second: int
    witness: int
    first_class: Congruence
    second_class: Congruence

    def verify(self, system: Sequence[Congruence | LinearCongruence]) -> bool:
        """Return True when this proof holds for ``system``, however it was produced."""
        indices_ok = self.first != self.second and all(
            0 <= index < len(system) for index in (self.first, self.second)
        )
        if not indices_ok:
            return False
        d = self.witness
        return (
            _confines(system[self.first], self.first_class)
            and _confines(system[self.second], self.second_class)
            and d > 1
            and self.first_class.modulus % d == 0
            and self.second_class.modulus % d == 0
            and (self.first_class.residue - self.second_class.residue) % d != 0
        )

    def __str__(self) -> str:
        d = self.witness
        return (
            f"congruences #{self.first} and #{self.second} are incompatible: "
            f"#{self.first} forces x = {self.first_class.residue % d} (mod {d}) "
            f"but #{self.second} forces x = {self.second_class.residue % d} (mod {d})"
        )


@dataclass(frozen=True, slots=True)
class UnsolvableCongruence:
    """A checkable proof that congruence number ``index`` of a system has no solution.

    ``divisor`` divides both ``a`` and the modulus of ``a * x = b (mod m)`` but not ``b``.
    Reducing the congruence modulo ``divisor`` leaves ``0 = b (mod divisor)``, which is
    false for every ``x``. The solvers report ``divisor = gcd(a, m)``, the largest choice.
    """

    index: int
    divisor: int

    def verify(self, system: Sequence[Congruence | LinearCongruence]) -> bool:
        """Return True when this proof holds for ``system``, however it was produced."""
        if not 0 <= self.index < len(system):
            return False
        a, b, m = _as_linear(system[self.index])
        d = self.divisor
        return d > 1 and a % d == 0 and m % d == 0 and b % d != 0

    def __str__(self) -> str:
        return (
            f"congruence #{self.index} has no solution: "
            f"{self.divisor} divides its coefficient and modulus but not its right-hand side"
        )


Inconsistency = UnsolvableCongruence | ConflictingCongruences
"""Either kind of proof that a congruence system has no solution."""


def _merge(left: Congruence, right: Congruence) -> Congruence | None:
    """Intersect two classes; None when they are disjoint."""
    g, p, _ = extended_gcd(left.modulus, right.modulus)
    diff = right.residue - left.residue
    if diff % g:
        return None
    # x = left.residue + left.modulus * t, and left.modulus * p = g (mod right.modulus),
    # so t = p * diff / g solves left.modulus * t = diff (mod right.modulus).
    step = right.modulus // g
    t = (diff // g) * p % step
    # 0 <= residue < left.modulus * step == lcm, so the result is already canonical.
    return Congruence(left.residue + left.modulus * t, left.modulus * step)


def _first_conflict(system: Sequence[Congruence], index: int) -> ConflictingCongruences:
    """Find the earlier congruence that ``system[index]`` contradicts.

    Called only when ``system[:index]`` is jointly solvable but ``system[:index + 1]`` is
    not; pairwise solvability is equivalent to joint solvability, so the culprit pair must
    involve ``index``.
    """
    later = system[index]
    for position in range(index):
        earlier = system[position]
        g = math.gcd(earlier.modulus, later.modulus)
        if (earlier.residue - later.residue) % g:
            return ConflictingCongruences(position, index, g, earlier, later)
    raise AssertionError("unsolvable system without a conflicting pair")  # pragma: no cover


def crt(congruences: Iterable[Congruence]) -> Congruence | ConflictingCongruences:
    """Solve ``x = r_i (mod m_i)`` for all i; moduli need not be pairwise coprime.

    Returns the single class of solutions, whose modulus is the lcm of all moduli (the
    empty system gives ``Congruence(0, 1)``: every integer), or a
    :class:`ConflictingCongruences` proof naming the first incompatible pair.

    >>> crt([Congruence(2, 3), Congruence(3, 5), Congruence(2, 7)])
    Congruence(residue=23, modulus=105)
    >>> crt([Congruence(1, 4), Congruence(3, 6)])
    Congruence(residue=9, modulus=12)
    >>> print(crt([Congruence(1, 4), Congruence(2, 6)]))
    congruences #0 and #1 are incompatible: #0 forces x = 1 (mod 2) but #1 forces x = 0 (mod 2)
    """
    system = tuple(congruences)
    combined = Congruence(0, 1)
    for index, congruence in enumerate(system):
        merged = _merge(combined, congruence)
        if merged is None:
            return _first_conflict(system, index)
        combined = merged
    return combined


def crt_pair(first: Congruence, second: Congruence) -> Congruence | ConflictingCongruences:
    """Solve the two-congruence system ``[first, second]``; see :func:`crt`."""
    return crt((first, second))


def solve_linear_congruence(a: int, b: int, modulus: int) -> Congruence | UnsolvableCongruence:
    """Return the full solution set of ``a * x = b (mod modulus)``.

    With ``g = gcd(a, modulus)``, the congruence is solvable exactly when ``g`` divides
    ``b``, and then its solutions form one class modulo ``modulus // g``: exactly ``g``
    distinct solutions modulo ``modulus``, listed by ``result.residues_mod(modulus)``.
    Otherwise an :class:`UnsolvableCongruence` proof (with index 0) is returned.

    >>> solution = solve_linear_congruence(6, 4, 10)
    >>> solution, list(solution.residues_mod(10))
    (Congruence(residue=4, modulus=5), [4, 9])
    >>> solve_linear_congruence(4, 3, 6)
    UnsolvableCongruence(index=0, divisor=2)
    """
    _require_modulus(modulus)
    g, x, _ = extended_gcd(a, modulus)
    if b % g:
        return UnsolvableCongruence(0, g)
    # a * x = g (mod modulus), so a * x * (b / g) = b (mod modulus).
    return Congruence.of(x * (b // g), modulus // g)


def solve_linear_system(
    system: Iterable[Congruence | LinearCongruence],
) -> Congruence | Inconsistency:
    """Solve simultaneous congruences ``a_i * x = b_i (mod m_i)`` in one unknown.

    Moduli may be arbitrary and mixed; plain :class:`Congruence` entries count as
    ``1 * x = r (mod m)``. Each constraint is first reduced to its own class of solutions,
    then the classes are combined by :func:`crt`. The result is the single class holding
    every solution, or an :data:`Inconsistency` proof whose indices point into ``system``:
    an :class:`UnsolvableCongruence` if some constraint is unsolvable on its own (the first
    such is reported), otherwise a :class:`ConflictingCongruences` pair.

    >>> solve_linear_system([LinearCongruence(4, 2, 6), LinearCongruence(3, 1, 10),
    ...                      Congruence(1, 4)])
    Congruence(residue=17, modulus=60)
    >>> print(solve_linear_system([LinearCongruence(2, 2, 8), LinearCongruence(3, 0, 6)]))
    congruences #0 and #1 are incompatible: #0 forces x = 1 (mod 2) but #1 forces x = 0 (mod 2)
    """
    constraints = tuple(system)
    classes: list[Congruence] = []
    for index, constraint in enumerate(constraints):
        solved = solve_linear_congruence(*_as_linear(constraint))
        if isinstance(solved, UnsolvableCongruence):
            return UnsolvableCongruence(index, solved.divisor)
        classes.append(solved)
    return crt(classes)
