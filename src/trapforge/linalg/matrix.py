"""Immutable integer matrices over Python ``int``.

The linear algebra layer never touches floats: every entry is an exact Python ``int``, so a
determinant of a 12x12 matrix with 30-digit entries is still exactly right. Matrices are
small (task families work with a handful of unknowns), so clarity wins over speed: a
:class:`Matrix` is a frozen dataclass holding a tuple of row tuples, and algorithms copy the
rows into lists, work in place and freeze the result.

The column count is stored explicitly because a matrix with no rows still has a width: the
integer kernel of an injective map ``Z^3 -> Z^5`` is spanned by a 0x3 matrix, not by an
"empty" object of unknown shape.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import SupportsIndex

__all__ = [
    "LinalgError",
    "Matrix",
    "ShapeError",
    "Vector",
    "dot",
]

Vector = tuple[int, ...]
"""An integer vector; the linear algebra layer never uses floats."""


class LinalgError(ValueError):
    """Base class for invalid input to the integer linear algebra layer."""


class ShapeError(LinalgError):
    """Raised when matrix or vector dimensions do not fit together."""


def dot(u: Sequence[int], v: Sequence[int]) -> int:
    """Exact dot product of two integer vectors of the same length.

    >>> dot((1, 2, 3), (4, -5, 6))
    12
    """
    if len(u) != len(v):
        raise ShapeError(f"cannot take the dot product of lengths {len(u)} and {len(v)}")
    return sum(a * b for a, b in zip(u, v, strict=True))


def _as_int(value: SupportsIndex) -> int:
    """Convert anything with ``__index__`` (int, bool, numpy or sympy integers) to ``int``.

    Floats and Fractions have no ``__index__`` and are rejected with a ``TypeError``, which
    is the point: a silently truncated 2.5 would break exactness.
    """
    return operator.index(value)


@dataclass(frozen=True, slots=True)
class Matrix:
    """An immutable ``nrows x ncols`` matrix of Python ints.

    The constructor expects the canonical representation (a tuple of int tuples, each of
    length ``ncols``); :meth:`Matrix.of` builds one from any nested iterable.

    >>> A = Matrix.of([[2, 1], [4, 3]])
    >>> A.shape, A[1, 0], A.determinant(), A.rank()
    ((2, 2), 4, 2, 2)
    >>> print(A @ Matrix.identity(2))
    [2 1]
    [4 3]
    >>> Matrix.of([], ncols=3).shape
    (0, 3)
    """

    rows: tuple[Vector, ...]
    ncols: int

    def __post_init__(self) -> None:
        if self.ncols < 0:
            raise ShapeError(f"a matrix cannot have {self.ncols} columns")
        for index, row in enumerate(self.rows):
            if len(row) != self.ncols:
                raise ShapeError(f"row {index} has {len(row)} entries, expected {self.ncols}")
            if not all(type(value) is int for value in row):
                raise TypeError(f"row {index} has a non-int entry; use Matrix.of() to convert")

    # -- construction ------------------------------------------------------------------------

    @classmethod
    def of(cls, rows: Iterable[Iterable[SupportsIndex]], ncols: int | None = None) -> Matrix:
        """Build a matrix from nested iterables of integers.

        ``ncols`` is required only to give a matrix with no rows a non-zero width; when rows
        are present it is checked against them.
        """
        frozen = tuple(tuple(_as_int(value) for value in row) for row in rows)
        width = len(frozen[0]) if frozen else 0
        return cls(frozen, width if ncols is None else ncols)

    @classmethod
    def identity(cls, n: int) -> Matrix:
        """The ``n x n`` identity matrix."""
        return cls(tuple(tuple(int(i == j) for j in range(n)) for i in range(n)), n)

    @classmethod
    def zeros(cls, nrows: int, ncols: int) -> Matrix:
        """The ``nrows x ncols`` zero matrix."""
        return cls(tuple((0,) * ncols for _ in range(nrows)), ncols)

    @classmethod
    def diagonal(cls, entries: Sequence[int], nrows: int, ncols: int) -> Matrix:
        """An ``nrows x ncols`` matrix with ``entries`` down its main diagonal.

        >>> print(Matrix.diagonal([2, 6], 2, 3))
        [2 0 0]
        [0 6 0]
        """
        if len(entries) > min(nrows, ncols):
            raise ShapeError(f"{len(entries)} diagonal entries do not fit in {nrows}x{ncols}")
        rows = [[0] * ncols for _ in range(nrows)]
        for i, value in enumerate(entries):
            rows[i][i] = _as_int(value)
        return cls.of(rows, ncols)

    # -- shape and access --------------------------------------------------------------------

    @property
    def nrows(self) -> int:
        """Number of rows."""
        return len(self.rows)

    @property
    def shape(self) -> tuple[int, int]:
        """``(nrows, ncols)``."""
        return self.nrows, self.ncols

    @property
    def is_square(self) -> bool:
        """True for an ``n x n`` matrix (including the 0x0 one)."""
        return self.nrows == self.ncols

    @property
    def columns(self) -> tuple[Vector, ...]:
        """The columns as vectors (``ncols`` of them, even when there are no rows)."""
        return tuple(tuple(row[j] for row in self.rows) for j in range(self.ncols))

    def __getitem__(self, index: tuple[int, int]) -> int:
        i, j = index
        return self.rows[i][j]

    def to_lists(self) -> list[list[int]]:
        """A mutable deep copy, handy for in-place algorithms and for serialization."""
        return [list(row) for row in self.rows]

    def __str__(self) -> str:
        if not self.rows:
            return f"<empty {self.nrows}x{self.ncols} matrix>"
        width = max((len(str(value)) for row in self.rows for value in row), default=0)
        return "\n".join(
            "[" + " ".join(str(value).rjust(width) for value in row) + "]" for row in self.rows
        )

    # -- arithmetic --------------------------------------------------------------------------

    def transpose(self) -> Matrix:
        """The ``ncols x nrows`` transpose."""
        return Matrix(self.columns, self.nrows)

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.ncols != other.nrows:
            raise ShapeError(f"cannot multiply {self.nrows}x{self.ncols} by {other.shape}")
        columns = other.columns
        return Matrix(
            tuple(tuple(dot(row, column) for column in columns) for row in self.rows),
            other.ncols,
        )

    def apply(self, vector: Sequence[int]) -> Vector:
        """The matrix-vector product ``A x`` for a column vector ``x``."""
        if len(vector) != self.ncols:
            raise ShapeError(f"cannot apply a {self.nrows}x{self.ncols} matrix to {len(vector)}")
        return tuple(dot(row, vector) for row in self.rows)

    def left_apply(self, vector: Sequence[int]) -> Vector:
        """The vector-matrix product ``y A`` for a row vector ``y``."""
        if len(vector) != self.nrows:
            raise ShapeError(f"cannot left-apply {len(vector)} to {self.nrows}x{self.ncols}")
        return tuple(dot(vector, column) for column in self.columns)

    # -- invariants --------------------------------------------------------------------------

    def determinant(self) -> int:
        """Exact determinant by fraction-free (Bareiss) elimination.

        Every intermediate value is a minor of the input, so entries grow only as fast as
        the determinant itself and every division is exact. The 0x0 determinant is 1.

        >>> Matrix.of([[0, 2, 1], [3, 1, 0], [1, 1, 1]]).determinant()
        -4
        """
        if not self.is_square:
            raise ShapeError(f"a {self.nrows}x{self.ncols} matrix has no determinant")
        n = self.nrows
        a = self.to_lists()
        sign = 1
        previous = 1
        for k in range(n - 1):
            if a[k][k] == 0:
                swap = next((i for i in range(k + 1, n) if a[i][k] != 0), None)
                if swap is None:
                    return 0
                a[k], a[swap] = a[swap], a[k]
                sign = -sign
            pivot = a[k][k]
            for i in range(k + 1, n):
                factor = a[i][k]
                row_i, row_k = a[i], a[k]
                for j in range(k + 1, n):
                    row_i[j] = (row_i[j] * pivot - factor * row_k[j]) // previous
            previous = pivot
        return sign * a[n - 1][n - 1] if n else 1

    def rank(self) -> int:
        """Rank over the rationals, by fraction-free row reduction to echelon form.

        >>> Matrix.of([[1, 2, 3], [2, 4, 6], [1, 0, 1]]).rank()
        2
        """
        a = self.to_lists()
        rank = 0
        previous = 1
        for col in range(self.ncols):
            if rank == self.nrows:
                break
            source = next((i for i in range(rank, self.nrows) if a[i][col] != 0), None)
            if source is None:
                continue
            a[rank], a[source] = a[source], a[rank]
            pivot_row = a[rank]
            pivot = pivot_row[col]
            for i in range(rank + 1, self.nrows):
                row_i = a[i]
                factor = row_i[col]
                for j in range(col + 1, self.ncols):
                    row_i[j] = (row_i[j] * pivot - factor * pivot_row[j]) // previous
                row_i[col] = 0
            previous = pivot
            rank += 1
        return rank

    def is_unimodular(self) -> bool:
        """True when the matrix is square with determinant +1 or -1 (invertible over Z)."""
        return self.is_square and abs(self.determinant()) == 1
