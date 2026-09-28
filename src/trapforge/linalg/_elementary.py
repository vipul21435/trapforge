"""In-place elementary operations on row lists, shared by the normal-form algorithms.

Every operation here is invertible over the integers (a swap, adding an integer multiple of
one line to another, negating a line, or a 2x2 block of determinant 1), so applying the
same operation to an identity matrix records a unimodular transform.
"""

from __future__ import annotations

from trapforge.modular import extended_gcd

Rows = list[list[int]]


def gcd_step(a: int, b: int) -> tuple[int, int, int, int]:
    """Return ``(x, y, p, q)`` with ``x*a + y*b = gcd(a, b)``, ``p*a + q*b = 0``, ``x*q - y*p = 1``.

    Applied to two lines ``(first, second)`` as ``(x*first + y*second, p*first + q*second)``,
    it moves the gcd of the two leading entries into ``first`` and a zero into ``second``.
    Requires ``(a, b) != (0, 0)``.
    """
    g, x, y = extended_gcd(a, b)
    return x, y, -(b // g), a // g


def swap_rows(rows: Rows, i: int, j: int) -> None:
    rows[i], rows[j] = rows[j], rows[i]


def swap_columns(rows: Rows, i: int, j: int) -> None:
    for row in rows:
        row[i], row[j] = row[j], row[i]


def negate_row(rows: Rows, i: int) -> None:
    rows[i] = [-value for value in rows[i]]


def add_row_multiple(rows: Rows, target: int, source: int, factor: int) -> None:
    """``row[target] += factor * row[source]``."""
    rows[target] = [t + factor * s for t, s in zip(rows[target], rows[source], strict=True)]


def add_column_multiple(rows: Rows, target: int, source: int, factor: int) -> None:
    """``column[target] += factor * column[source]``."""
    for row in rows:
        row[target] += factor * row[source]


def combine_rows(rows: Rows, i: int, j: int, step: tuple[int, int, int, int]) -> None:
    """Replace rows ``i, j`` by ``x*ri + y*rj`` and ``p*ri + q*rj`` for ``step = (x, y, p, q)``."""
    x, y, p, q = step
    first, second = rows[i], rows[j]
    rows[i] = [x * a + y * b for a, b in zip(first, second, strict=True)]
    rows[j] = [p * a + q * b for a, b in zip(first, second, strict=True)]


def combine_columns(rows: Rows, i: int, j: int, step: tuple[int, int, int, int]) -> None:
    """Replace columns ``i, j`` by ``x*ci + y*cj`` and ``p*ci + q*cj`` for ``step``."""
    x, y, p, q = step
    for row in rows:
        a, b = row[i], row[j]
        row[i], row[j] = x * a + y * b, p * a + q * b
