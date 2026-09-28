"""Hermite normal form over the integers, with the unimodular transform that produces it.

The Hermite normal form is to integer row operations what reduced row echelon form is to
rational ones: a canonical echelon basis of the lattice spanned by the rows of ``A``. Two
matrices with the same number of rows have the same HNF exactly when their rows generate the
same lattice, which makes it the tool for canonical kernel bases and for deciding lattice
membership by reduction.

Convention (row style): ``U @ A == H`` with ``U`` unimodular, and ``H`` in row echelon form
where every leading entry is positive and every entry above a leading entry lies in
``range(leading entry)``. Zero rows, if any, come last.
"""

from __future__ import annotations

from dataclasses import dataclass

from trapforge.linalg._elementary import (
    Rows,
    add_row_multiple,
    combine_rows,
    gcd_step,
    negate_row,
)
from trapforge.linalg.matrix import Matrix

__all__ = [
    "HermiteForm",
    "hermite_normal_form",
    "is_hermite_normal_form",
]


@dataclass(frozen=True, slots=True)
class HermiteForm:
    """The row-style Hermite normal form ``H`` of ``A`` and a unimodular ``U`` with ``U @ A == H``.

    ``pivots[i]`` is the column of the leading entry of row ``i`` of ``H``; there is one pivot
    per nonzero row, so ``len(pivots)`` is the rank of ``A``.
    """

    H: Matrix
    U: Matrix
    pivots: tuple[int, ...]

    @property
    def rank(self) -> int:
        """Rank of ``A`` (the number of nonzero rows of ``H``)."""
        return len(self.pivots)

    @property
    def basis(self) -> Matrix:
        """The nonzero rows of ``H``: the canonical basis of the row lattice of ``A``."""
        return Matrix(self.H.rows[: self.rank], self.H.ncols)

    @property
    def left_kernel(self) -> Matrix:
        """The rows of ``U`` that ``A`` sends to zero: a basis of ``{y in Z^m : y A = 0}``.

        They map to the zero rows of ``H``, and because ``U`` is unimodular they span every
        integer ``y`` with ``y A = 0``, not just a finite-index sublattice.
        """
        return Matrix(self.U.rows[self.rank :], self.U.ncols)


def _leading_column(row: tuple[int, ...] | list[int]) -> int | None:
    return next((j for j, value in enumerate(row) if value), None)


def is_hermite_normal_form(H: Matrix) -> bool:
    """True when ``H`` satisfies the row-style Hermite normal form conditions.

    >>> is_hermite_normal_form(Matrix.of([[2, 1, 5], [0, 3, 2], [0, 0, 0]]))
    True
    >>> is_hermite_normal_form(Matrix.of([[2, 4], [0, 3]]))  # 4 is not reduced modulo 3
    False
    """
    previous = -1
    seen_zero_row = False
    for i, row in enumerate(H.rows):
        lead = _leading_column(row)
        if lead is None:
            seen_zero_row = True
            continue
        pivot = row[lead]
        if seen_zero_row or lead <= previous or pivot < 0:
            return False
        if any(not 0 <= H.rows[k][lead] < pivot for k in range(i)):
            return False
        previous = lead
    return True


def _gather_column_gcd(h: Rows, u: Rows, top: int, col: int) -> None:
    """Row-reduce ``h[top:]`` so that column ``col`` is zero below row ``top``.

    Afterwards ``h[top][col]`` is plus or minus the gcd of the original column entries.
    Exact multiples are cleared by a subtraction, which keeps the transform entries small;
    otherwise a determinant-1 gcd step combines the two rows.
    """
    for i in range(top + 1, len(h)):
        below = h[i][col]
        if below == 0:
            continue
        pivot = h[top][col]
        if pivot and below % pivot == 0:
            add_row_multiple(h, i, top, -(below // pivot))
            add_row_multiple(u, i, top, -(below // pivot))
        else:
            step = gcd_step(pivot, below)
            combine_rows(h, top, i, step)
            combine_rows(u, top, i, step)


def hermite_normal_form(A: Matrix) -> HermiteForm:
    """Compute the row-style Hermite normal form of ``A`` and its unimodular transform.

    Works for any shape and rank. Columns are processed left to right; each one gets its gcd
    moved into the next pivot row by determinant-1 row operations, the pivot is made
    positive, and the entries above it are reduced modulo it.

    >>> form = hermite_normal_form(Matrix.of([[4, 6, 2], [6, 9, 5], [2, 3, 1]]))
    >>> print(form.H)
    [2 3 1]
    [0 0 2]
    [0 0 0]
    >>> form.pivots, form.U @ Matrix.of([[4, 6, 2], [6, 9, 5], [2, 3, 1]]) == form.H
    ((0, 2), True)
    """
    m, n = A.shape
    h = A.to_lists()
    u = Matrix.identity(m).to_lists()
    pivots: list[int] = []
    for col in range(n):
        top = len(pivots)
        if top == m:
            break
        _gather_column_gcd(h, u, top, col)
        pivot = h[top][col]
        if pivot == 0:
            continue
        if pivot < 0:
            negate_row(h, top)
            negate_row(u, top)
            pivot = -pivot
        for i in range(top):
            quotient = h[i][col] // pivot
            if quotient:
                add_row_multiple(h, i, top, -quotient)
                add_row_multiple(u, i, top, -quotient)
        pivots.append(col)
    return HermiteForm(Matrix.of(h, n), Matrix.of(u, m), tuple(pivots))
