"""Integer kernels and affine lattices, with exact enumeration of lattice points in a box.

The integer solutions of ``A x = b`` are never "a vector space": they form an affine lattice
``p + L``, a single point ``p`` shifted by every integer combination of a kernel basis.
:class:`AffineLattice` stores that set canonically (the basis in Hermite normal form and the
point reduced modulo it), so two descriptions of the same set compare equal, and it can list
exactly the points that fall inside a box of bounds. That box enumeration is what the
uniqueness prover uses to decide whether bounded hidden parameters are pinned down.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import SupportsIndex

from trapforge.linalg.hermite import hermite_normal_form, is_hermite_normal_form
from trapforge.linalg.matrix import LinalgError, Matrix, ShapeError, Vector

__all__ = [
    "AffineLattice",
    "kernel_basis",
]


def kernel_basis(A: Matrix) -> Matrix:
    """A basis of the integer kernel ``{x in Z^n : A x = 0}``, as rows in Hermite normal form.

    The kernel of ``A`` is the left kernel of its transpose, which the Hermite transform of
    ``A^T`` exposes directly; putting those rows in Hermite normal form makes the basis
    canonical. The result has ``n - rank(A)`` rows, and every integer kernel vector is an
    integer combination of them (the basis is saturated, not just a finite-index sublattice).

    >>> print(kernel_basis(Matrix.of([[2, 4, 6], [1, 2, 3]])))
    [ 1  1 -1]
    [ 0  3 -2]
    """
    generators = hermite_normal_form(A.transpose()).left_kernel
    return hermite_normal_form(generators).basis


def _leading_columns(basis: Matrix) -> tuple[int, ...]:
    return tuple(next(j for j, value in enumerate(row) if value) for row in basis.rows)


def _reduce(vector: Sequence[int], basis: Matrix, pivots: Sequence[int]) -> list[int]:
    """Subtract basis rows so each pivot coordinate lands in ``range(pivot entry)``.

    With the basis in Hermite normal form, later rows are zero at earlier pivot columns, so
    one pass in order leaves every pivot coordinate reduced. The result differs from
    ``vector`` by a lattice vector, and it is zero exactly when ``vector`` is in the lattice.
    """
    reduced = list(vector)
    for row, column in zip(basis.rows, pivots, strict=True):
        quotient = reduced[column] // row[column]
        if quotient:
            reduced = [value - quotient * entry for value, entry in zip(reduced, row, strict=True)]
    return reduced


def _format_vector(vector: Sequence[int]) -> str:
    return "(" + ", ".join(str(value) for value in vector) + ")"


@dataclass(frozen=True, slots=True)
class AffineLattice:
    """The set ``{point + t_1 * basis[0] + ... + t_k * basis[k-1] : t in Z^k}``.

    The constructor expects the canonical representation (``basis`` in Hermite normal form
    without zero rows, ``point`` reduced modulo it); :meth:`AffineLattice.of` builds it from
    any point and generating set, so equal sets always compare equal.

    >>> lattice = AffineLattice.of((7, 0), Matrix.of([[3, -2], [6, -4]]))
    >>> print(lattice)
    x = (1, 4) + t0*(3, -2), t in Z^1
    >>> (4, 2) in lattice, (4, 3) in lattice
    (True, False)
    >>> list(lattice.points_in_box((-5, -5), (5, 5)))
    [(1, 4), (4, 2)]
    """

    point: Vector
    basis: Matrix

    def __post_init__(self) -> None:
        if self.basis.ncols != len(self.point):
            raise ShapeError(
                f"basis vectors have {self.basis.ncols} entries but the point has {len(self.point)}"
            )
        if not all(type(value) is int for value in self.point):
            raise TypeError("point has a non-int entry; use AffineLattice.of() to convert")
        if not is_hermite_normal_form(self.basis) or not all(any(r) for r in self.basis.rows):
            raise LinalgError("basis is not in Hermite normal form without zero rows")
        pivots = _leading_columns(self.basis)
        if any(
            not 0 <= self.point[column] < row[column]
            for row, column in zip(self.basis.rows, pivots, strict=True)
        ):
            raise LinalgError("point is not reduced modulo the basis; use AffineLattice.of()")

    @classmethod
    def of(cls, point: Sequence[SupportsIndex], generators: Matrix | None = None) -> AffineLattice:
        """The canonical form of ``point + (integer span of the rows of generators)``.

        ``generators`` may be dependent or contain zero rows; ``None`` means the single
        point.
        """
        exact = tuple(operator.index(value) for value in point)
        if generators is None:
            generators = Matrix.of([], ncols=len(exact))
        if generators.ncols != len(exact):
            raise ShapeError(
                f"generators have {generators.ncols} entries but the point has {len(exact)}"
            )
        basis = hermite_normal_form(generators).basis
        return cls(tuple(_reduce(exact, basis, _leading_columns(basis))), basis)

    @property
    def dimension(self) -> int:
        """Number of free integer parameters (0 for a single point)."""
        return self.basis.nrows

    @property
    def ambient_dimension(self) -> int:
        """Length of the vectors in the set."""
        return len(self.point)

    @property
    def is_point(self) -> bool:
        """True when the set has exactly one element."""
        return self.dimension == 0

    def __contains__(self, vector: object) -> bool:
        if not isinstance(vector, Sequence) or len(vector) != self.ambient_dimension:
            return False
        if not all(isinstance(value, int) for value in vector):
            return False
        offset = [value - base for value, base in zip(vector, self.point, strict=True)]
        return not any(_reduce(offset, self.basis, _leading_columns(self.basis)))

    def at(self, coefficients: Sequence[int]) -> Vector:
        """The element ``point + sum(coefficients[i] * basis[i])``."""
        if len(coefficients) != self.dimension:
            raise ShapeError(f"expected {self.dimension} coefficients, got {len(coefficients)}")
        combination = self.basis.left_apply(coefficients)
        return tuple(base + step for base, step in zip(self.point, combination, strict=True))

    def points_in_box(self, lower: Sequence[int], upper: Sequence[int]) -> Iterator[Vector]:
        """Every element ``x`` with ``lower[j] <= x[j] <= upper[j]`` for all ``j``.

        Points come out in increasing lexicographic order, and the search never leaves the
        box: in the Hermite basis, coordinate ``pivot[i]`` depends only on the first ``i + 1``
        parameters and grows with the last of them, so each parameter ranges over an exact
        interval and every coordinate is checked as soon as it is fully determined. The set
        is always finite, because every parameter moves some coordinate.
        """
        n = self.ambient_dimension
        if len(lower) != n or len(upper) != n:
            raise ShapeError(f"box bounds must have {n} entries each")
        pivots = _leading_columns(self.basis)
        return self._search(0, list(self.point), pivots, lower, upper)

    def _search(
        self,
        level: int,
        partial: list[int],
        pivots: tuple[int, ...],
        lower: Sequence[int],
        upper: Sequence[int],
    ) -> Iterator[Vector]:
        # Coordinates strictly between the previous pivot and this one are now final.
        start = pivots[level - 1] + 1 if level else 0
        stop = pivots[level] if level < len(pivots) else len(partial)
        if any(not lower[j] <= partial[j] <= upper[j] for j in range(start, stop)):
            return
        if level == len(pivots):
            yield tuple(partial)
            return
        row = self.basis.rows[level]
        column = pivots[level]
        step = row[column]
        first = -((partial[column] - lower[column]) // step)  # ceil((lower - x) / step)
        last = (upper[column] - partial[column]) // step
        for t in range(first, last + 1):
            moved = [value + t * entry for value, entry in zip(partial, row, strict=True)]
            yield from self._search(level + 1, moved, pivots, lower, upper)

    def __str__(self) -> str:
        terms = [_format_vector(self.point)]
        terms += [f"t{i}*{_format_vector(row)}" for i, row in enumerate(self.basis.rows)]
        suffix = f", t in Z^{self.dimension}" if self.dimension else ""
        return "x = " + " + ".join(terms) + suffix
