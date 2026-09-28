"""Hypothesis strategies for integer matrices, shared by the linear algebra tests.

Uniformly random integer matrices are almost always of full rank, which is exactly where
normal-form algorithms are least likely to break. These strategies therefore mix in sparse
matrices, duplicated rows and explicit low-rank products so that rank deficiency, zero
columns and repeated pivots show up in most runs.
"""

from __future__ import annotations

import sympy
from hypothesis import strategies as st

from trapforge.linalg import Matrix

small_entries = st.integers(min_value=-12, max_value=12)
sparse_entries = st.sampled_from([0, 0, 0, 0, 1, -1, 2, -2, 3, 6, -4, 9])


def dense_matrices(
    rows: st.SearchStrategy[int],
    cols: st.SearchStrategy[int],
    entries: st.SearchStrategy[int] = small_entries,
) -> st.SearchStrategy[Matrix]:
    """Matrices whose shape is drawn from ``rows`` x ``cols`` and entries from ``entries``."""

    def fill(shape: tuple[int, int]) -> st.SearchStrategy[Matrix]:
        m, n = shape
        row = st.lists(entries, min_size=n, max_size=n)
        return st.lists(row, min_size=m, max_size=m).map(lambda rs: Matrix.of(rs, ncols=n))

    return st.tuples(rows, cols).flatmap(fill)


@st.composite
def low_rank_matrices(draw: st.DrawFn, max_dim: int = 5) -> Matrix:
    """A product ``B @ C`` whose inner dimension is smaller than both outer ones."""
    m = draw(st.integers(1, max_dim))
    n = draw(st.integers(1, max_dim))
    inner = draw(st.integers(0, max(0, min(m, n) - 1)))
    left = draw(dense_matrices(st.just(m), st.just(inner), st.integers(-4, 4)))
    right = draw(dense_matrices(st.just(inner), st.just(n), st.integers(-4, 4)))
    return left @ right


@st.composite
def repeated_row_matrices(draw: st.DrawFn, max_dim: int = 5) -> Matrix:
    """A matrix with one row repeated, scaled by an integer, at a random position."""
    base = draw(dense_matrices(st.integers(1, max_dim - 1), st.integers(1, max_dim)))
    source = draw(st.integers(0, base.nrows - 1))
    factor = draw(st.integers(-3, 3))
    position = draw(st.integers(0, base.nrows))
    rows = list(base.rows)
    rows.insert(position, tuple(factor * value for value in base.rows[source]))
    return Matrix.of(rows, ncols=base.ncols)


def matrices(max_dim: int = 5, min_dim: int = 1) -> st.SearchStrategy[Matrix]:
    """A mix of dense, sparse, low-rank and repeated-row matrices up to ``max_dim``."""
    dims = st.integers(min_dim, max_dim)
    return st.one_of(
        dense_matrices(dims, dims),
        dense_matrices(dims, dims, sparse_entries),
        low_rank_matrices(max_dim),
        repeated_row_matrices(max_dim),
    )


def square_matrices(max_dim: int = 5, min_dim: int = 0) -> st.SearchStrategy[Matrix]:
    """Square matrices, dense or sparse."""

    def square(entries: st.SearchStrategy[int]) -> st.SearchStrategy[Matrix]:
        return st.integers(min_dim, max_dim).flatmap(
            lambda n: dense_matrices(st.just(n), st.just(n), entries)
        )

    return st.one_of(square(small_entries), square(sparse_entries))


def to_sympy(matrix: Matrix) -> sympy.Matrix:
    """The same matrix as a sympy Matrix, keeping the shape even when it is empty."""
    flat = [value for row in matrix.rows for value in row]
    return sympy.Matrix(matrix.nrows, matrix.ncols, flat)


def from_sympy(matrix: sympy.Matrix) -> Matrix:
    """Convert a sympy integer Matrix back, keeping the shape even when it is empty."""
    return Matrix.of(matrix.tolist(), ncols=matrix.cols)
