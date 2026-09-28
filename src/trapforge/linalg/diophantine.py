"""Systems of linear Diophantine equations ``A x = b`` over the integers.

:func:`solve_diophantine` returns either every integer solution, as an
:class:`~trapforge.linalg.lattice.AffineLattice`, or an :class:`UnsolvableSystem`
certificate. The certificate is the integer version of the Fredholm alternative: a weight
vector ``w`` and a modulus ``d`` such that ``w A`` is divisible by ``d`` entry by entry while
``w . b`` is not. For every integer ``x``, ``w . (A x) = (w A) . x`` is then a multiple of
``d``, so ``A x = b`` is impossible. ``d = 0`` covers systems that have no rational solution
either (``w A = 0`` but ``w . b != 0``), using the convention that 0 divides only 0. Anyone can
check a certificate with one vector-matrix product and a dot product, without trusting the
solver.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import SupportsIndex

from trapforge.linalg.lattice import AffineLattice
from trapforge.linalg.matrix import Matrix, ShapeError, Vector, dot
from trapforge.linalg.smith import smith_normal_form

__all__ = [
    "UnsolvableSystem",
    "solve_diophantine",
]


def _divides(d: int, value: int) -> bool:
    return value == 0 if d == 0 else value % d == 0


def _symmetric_residue(value: int, d: int) -> int:
    """The representative of ``value`` modulo ``d`` in ``(-d/2, d/2]``."""
    residue = value % d
    return residue - d if 2 * residue > d else residue


@dataclass(frozen=True, slots=True)
class UnsolvableSystem:
    """A checkable proof that ``A x = b`` has no integer solution.

    Every entry of ``multiplier @ A`` is divisible by ``modulus`` but ``rhs``, which equals
    ``multiplier . b``, is not. ``modulus == 0`` means ``multiplier @ A`` is exactly zero,
    so the system has no rational solution either; otherwise ``modulus >= 2``.
    """

    multiplier: Vector
    modulus: int
    rhs: int

    def verify(self, A: Matrix, b: Sequence[int]) -> bool:
        """Return True when this proof holds for ``A x = b``, however it was produced."""
        if len(self.multiplier) != A.nrows or len(b) != A.nrows:
            return False
        if self.modulus < 0 or self.modulus == 1:
            return False
        rhs = dot(self.multiplier, b)
        return (
            rhs == self.rhs
            and all(_divides(self.modulus, value) for value in A.left_apply(self.multiplier))
            and not _divides(self.modulus, rhs)
        )

    @property
    def is_rational(self) -> bool:
        """True when the system has no solution even over the rationals."""
        return self.modulus == 0

    def __str__(self) -> str:
        weights = ", ".join(str(w) for w in self.multiplier)
        if self.is_rational:
            return (
                f"no solution, not even rational: weighting the equations by ({weights}) "
                f"cancels every unknown but leaves 0 = {self.rhs}"
            )
        return (
            f"no integer solution: weighting the equations by ({weights}) makes every "
            f"coefficient a multiple of {self.modulus}, but the right-hand side becomes "
            f"{self.rhs}, which is not"
        )


def solve_diophantine(A: Matrix, b: Sequence[SupportsIndex]) -> AffineLattice | UnsolvableSystem:
    """All integer solutions of ``A x = b``, or a certificate that there are none.

    With ``U A V = S`` in Smith normal form, substituting ``x = V y`` gives the decoupled
    equations ``d_i * y_i = (U b)_i``: the first ``rank`` of them need ``d_i`` to divide the
    right-hand side, the rest need it to be zero, and the remaining ``y_i`` are free. Row
    ``i`` of ``U`` is exactly the certificate when equation ``i`` fails; a rational failure
    is reported before a divisibility one, and a divisibility certificate is shrunk modulo
    ``d_i``. The solution set is returned in canonical form, so it does not depend on which
    unimodular transforms the elimination happened to pick.

    >>> print(solve_diophantine(Matrix.of([[2, 3, 5], [1, 1, 1]]), [11, 4]))
    x = (1, 3, 0) + t0*(2, -3, 1), t in Z^1
    >>> A = Matrix.of([[2, 4], [3, 9]])
    >>> proof = solve_diophantine(A, [1, 0])
    >>> proof.multiplier, proof.modulus, proof.rhs, proof.verify(A, [1, 0])
    ((3, 2), 6, 3, True)
    """
    m, n = A.shape
    rhs = tuple(operator.index(value) for value in b)
    if len(rhs) != m:
        raise ShapeError(f"a {m}x{n} system needs {m} right-hand sides, got {len(rhs)}")
    smith = smith_normal_form(A)
    transformed = smith.U.apply(rhs)
    rank = smith.rank
    for i in range(rank, m):
        if transformed[i]:
            return UnsolvableSystem(smith.U.rows[i], 0, transformed[i])
    diagonal = smith.diagonal
    for i in range(rank):
        if transformed[i] % diagonal[i]:
            weights = tuple(_symmetric_residue(w, diagonal[i]) for w in smith.U.rows[i])
            return UnsolvableSystem(weights, diagonal[i], dot(weights, rhs))
    y = [transformed[i] // diagonal[i] for i in range(rank)] + [0] * (n - rank)
    kernel = Matrix(smith.V.columns[rank:], n)
    return AffineLattice.of(smith.V.apply(y), kernel)
