"""Smith normal form over the integers, with both unimodular transforms.

Every integer matrix ``A`` can be written as ``U @ A @ V == S`` with ``U`` and ``V``
unimodular and ``S`` diagonal, its diagonal entries ``d_1 | d_2 | ... | d_r`` positive and
each dividing the next, followed by zeros. The ``d_i`` (the invariant factors) are unique.

This is the workhorse for linear Diophantine systems: substituting ``x = V y`` turns
``A x = b`` into the decoupled system ``d_i * y_i = (U b)_i``, which can be read off one
coordinate at a time, including the exact reason when there is no integer solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from trapforge.linalg._elementary import (
    Rows,
    add_column_multiple,
    add_row_multiple,
    combine_columns,
    combine_rows,
    gcd_step,
    negate_row,
    swap_columns,
    swap_rows,
)
from trapforge.linalg.hermite import hermite_normal_form
from trapforge.linalg.matrix import Matrix

__all__ = [
    "SmithForm",
    "is_smith_normal_form",
    "smith_normal_form",
]


@dataclass(frozen=True, slots=True)
class SmithForm:
    """The Smith normal form ``S`` of ``A`` and unimodular ``U``, ``V`` with ``U @ A @ V == S``."""

    S: Matrix
    U: Matrix
    V: Matrix

    @property
    def diagonal(self) -> tuple[int, ...]:
        """The ``min(m, n)`` diagonal entries of ``S``, zeros included."""
        return tuple(self.S[i, i] for i in range(min(self.S.shape)))

    @property
    def invariant_factors(self) -> tuple[int, ...]:
        """The nonzero diagonal entries ``d_1 | d_2 | ... | d_r``."""
        return tuple(d for d in self.diagonal if d)

    @property
    def rank(self) -> int:
        """Rank of ``A`` (the number of invariant factors)."""
        return len(self.invariant_factors)


def is_smith_normal_form(S: Matrix) -> bool:
    """True when ``S`` is diagonal with a non-negative divisibility chain on its diagonal.

    Zeros may only trail: ``0`` divides nothing but ``0``, so a chain ``d_i | d_(i+1)``
    that reaches a zero stays at zero.

    >>> is_smith_normal_form(Matrix.of([[2, 0, 0], [0, 6, 0]]))
    True
    >>> is_smith_normal_form(Matrix.of([[2, 0], [0, 3]]))  # 2 does not divide 3
    False
    """
    m, n = S.shape
    if any(S[i, j] for i in range(m) for j in range(n) if i != j):
        return False
    diagonal = [S[i, i] for i in range(min(m, n))]
    if any(d < 0 for d in diagonal):
        return False
    return all(
        (later % earlier == 0) if earlier else later == 0 for earlier, later in pairwise(diagonal)
    )


def _smallest_entry(s: Rows, t: int, ncols: int) -> tuple[int, int] | None:
    """Position of a nonzero entry of least absolute value in ``s[t:, t:]``, if any."""
    best: tuple[int, int] | None = None
    best_size = 0
    for i in range(t, len(s)):
        for j in range(t, ncols):
            size = abs(s[i][j])
            if size and (best is None or size < best_size):
                best, best_size = (i, j), size
    return best


def _clear_column(s: Rows, u: Rows, t: int) -> None:
    """Zero ``s[i][t]`` for ``i > t`` with row operations (mirrored into ``u``)."""
    for i in range(t + 1, len(s)):
        below = s[i][t]
        if below == 0:
            continue
        pivot = s[t][t]
        if below % pivot == 0:
            add_row_multiple(s, i, t, -(below // pivot))
            add_row_multiple(u, i, t, -(below // pivot))
        else:
            step = gcd_step(pivot, below)
            combine_rows(s, t, i, step)
            combine_rows(u, t, i, step)


def _clear_row(s: Rows, v: Rows, t: int, ncols: int) -> None:
    """Zero ``s[t][j]`` for ``j > t`` with column operations (mirrored into ``v``)."""
    for j in range(t + 1, ncols):
        right = s[t][j]
        if right == 0:
            continue
        pivot = s[t][t]
        if right % pivot == 0:
            add_column_multiple(s, j, t, -(right // pivot))
            add_column_multiple(v, j, t, -(right // pivot))
        else:
            step = gcd_step(pivot, right)
            combine_columns(s, t, j, step)
            combine_columns(v, t, j, step)


def _settle_pivot(s: Rows, u: Rows, v: Rows, t: int, ncols: int) -> bool:
    """One round of work on pivot ``(t, t)``; True once it is finished.

    Finished means row ``t`` and column ``t`` are zero apart from the pivot and the pivot
    divides every entry of ``s[t+1:, t+1:]``. A round that is not finished either replaced
    the pivot by a proper divisor (a gcd step) or set up such a step for the next round, so
    ``|pivot|`` strictly decreases and the loop terminates.
    """
    _clear_column(s, u, t)
    _clear_row(s, v, t, ncols)
    if any(s[i][t] for i in range(t + 1, len(s))):
        return False  # a column gcd step refilled column t
    pivot = s[t][t]
    for i in range(t + 1, len(s)):
        if any(s[i][j] % pivot for j in range(t + 1, ncols)):
            # Copy the offending row into row t; clearing it forces a smaller pivot.
            add_row_multiple(s, t, i, 1)
            add_row_multiple(u, t, i, 1)
            return False
    return True


def smith_normal_form(A: Matrix) -> SmithForm:
    """Compute ``S``, ``U``, ``V`` with ``U @ A @ V == S`` in Smith normal form.

    Works for any shape and rank. The rows are first put in Hermite normal form, whose
    entries are reduced modulo the pivots; starting the diagonalization from there keeps the
    transforms far smaller than diagonalizing ``A`` directly, where repeated gcd steps make
    the entries of ``V`` grow quickly. Then, for each diagonal position, the smallest
    remaining entry is moved into place, its row and column are cleared with determinant-1
    operations, and the pivot is forced to divide the rest of the matrix, which yields the
    divisibility chain.

    >>> A = Matrix.of([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])
    >>> form = smith_normal_form(A)
    >>> form.invariant_factors
    (2, 6, 12)
    >>> form.U @ A @ form.V == form.S, form.U.is_unimodular(), form.V.is_unimodular()
    (True, True, True)
    """
    m, n = A.shape
    hermite = hermite_normal_form(A)
    s = hermite.H.to_lists()
    u = hermite.U.to_lists()
    v = Matrix.identity(n).to_lists()
    for t in range(min(m, n)):
        position = _smallest_entry(s, t, n)
        if position is None:
            break
        i, j = position
        swap_rows(s, t, i)
        swap_rows(u, t, i)
        swap_columns(s, t, j)
        swap_columns(v, t, j)
        while not _settle_pivot(s, u, v, t, n):
            pass
        if s[t][t] < 0:
            negate_row(s, t)
            negate_row(u, t)
    return SmithForm(Matrix.of(s, n), Matrix.of(u, m), Matrix.of(v, n))
