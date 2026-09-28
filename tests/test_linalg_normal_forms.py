"""Hermite and Smith normal forms: transform identities, canonical form, and sympy agreement."""

import math
import random
from itertools import combinations, pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sympy import ZZ
from sympy.matrices.normalforms import hermite_normal_form as sympy_hnf
from sympy.matrices.normalforms import smith_normal_form as sympy_snf

from _strategies import from_sympy, matrices, square_matrices, to_sympy, unimodular_matrices
from trapforge.linalg import (
    Matrix,
    hermite_normal_form,
    is_hermite_normal_form,
    is_smith_normal_form,
    smith_normal_form,
)


def reverse_columns(A: Matrix) -> Matrix:
    return Matrix.of([row[::-1] for row in A.rows], ncols=A.ncols)


def sympy_row_hermite_basis(A: Matrix) -> Matrix:
    """sympy's column-style HNF, translated to this library's row-style convention.

    sympy returns a basis of the column lattice in which each column's last nonzero entry
    is its pivot and entries to the right of a pivot are reduced. Transposing, reversing the
    coordinate order and reversing the basis order turns that into leading-entry pivots with
    reduced entries above them, which is the row-style HNF of the same lattice. The HNF of a
    lattice is unique, so the two must agree exactly.
    """
    column_form = from_sympy(sympy_hnf(to_sympy(reverse_columns(A).transpose())))
    rows = reverse_columns(column_form.transpose()).rows
    return Matrix.of(rows[::-1], ncols=A.ncols)


def leading_columns(H: Matrix) -> tuple[int, ...]:
    return tuple(next(j for j, v in enumerate(row) if v) for row in H.rows if any(row))


def determinantal_divisor(A: Matrix, k: int) -> int:
    """gcd of all k x k minors of A."""
    minors = (
        Matrix.of([[A[i, j] for j in cols] for i in rows]).determinant()
        for rows in combinations(range(A.nrows), k)
        for cols in combinations(range(A.ncols), k)
    )
    return math.gcd(*minors)


# -- Hermite normal form ------------------------------------------------------------------


@given(matrices(max_dim=6, min_dim=0))
def test_hermite_transform_is_unimodular_and_produces_h(A: Matrix) -> None:
    form = hermite_normal_form(A)
    assert form.U.shape == (A.nrows, A.nrows)
    assert form.U @ A == form.H
    assert abs(form.U.determinant()) == 1
    assert is_hermite_normal_form(form.H)
    assert form.rank == A.rank()
    assert form.pivots == leading_columns(form.H)


@given(matrices(max_dim=5))
def test_hermite_basis_matches_sympy(A: Matrix) -> None:
    assert hermite_normal_form(A).basis == sympy_row_hermite_basis(A)


@given(matrices(max_dim=5), st.data())
def test_hermite_form_depends_only_on_the_row_lattice(A: Matrix, data: st.DataObject) -> None:
    W = data.draw(unimodular_matrices(A.nrows))
    assert hermite_normal_form(W @ A).H == hermite_normal_form(A).H


@given(matrices(max_dim=5))
def test_hermite_form_is_idempotent(A: Matrix) -> None:
    H = hermite_normal_form(A).H
    assert hermite_normal_form(H).H == H


@given(matrices(max_dim=5, min_dim=0))
def test_left_kernel_rows_annihilate_a(A: Matrix) -> None:
    form = hermite_normal_form(A)
    assert form.left_kernel.nrows == A.nrows - form.rank
    assert form.left_kernel @ A == Matrix.zeros(A.nrows - form.rank, A.ncols)


def test_hermite_on_empty_shapes() -> None:
    no_rows = hermite_normal_form(Matrix.of([], ncols=3))
    assert no_rows.H.shape == (0, 3)
    assert no_rows.U.shape == (0, 0)
    assert no_rows.rank == 0
    no_cols = hermite_normal_form(Matrix.of([[], []], ncols=0))
    assert Matrix.identity(2) == no_cols.U
    assert no_cols.left_kernel == Matrix.identity(2)


def test_hermite_keeps_huge_entries_exact() -> None:
    big = 10**30
    A = Matrix.of([[big, big + 1], [big + 2, big + 3]])
    form = hermite_normal_form(A)
    # det A = -2, so the lattice has index 2 and the pivots multiply to 2.
    # The first column has gcd(big, big + 2) = 2, which leaves 1 for the second pivot.
    assert form.H.rows == ((2, 0), (0, 1))
    assert form.U @ A == form.H


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([[2, 1, 5], [0, 3, 2], [0, 0, 0]], True),
        ([[0, 0], [0, 0]], True),
        ([[0, 0], [1, 0]], False),  # zero row above a nonzero row
        ([[-2, 1], [0, 3]], False),  # negative pivot
        ([[0, 1], [1, 0]], False),  # pivots not strictly moving right
        ([[2, 3], [0, 3]], False),  # 3 above a pivot of 3 is not reduced
        ([[2, -1], [0, 3]], False),  # negative entry above a pivot
    ],
)
def test_is_hermite_normal_form_detects_each_violation(
    rows: list[list[int]], expected: bool
) -> None:
    assert is_hermite_normal_form(Matrix.of(rows)) is expected


# -- Smith normal form --------------------------------------------------------------------


@given(matrices(max_dim=6, min_dim=0))
def test_smith_transforms_are_unimodular_and_produce_s(A: Matrix) -> None:
    form = smith_normal_form(A)
    assert form.U.shape == (A.nrows, A.nrows)
    assert form.V.shape == (A.ncols, A.ncols)
    assert form.U @ A @ form.V == form.S
    assert abs(form.U.determinant()) == 1
    assert abs(form.V.determinant()) == 1
    assert is_smith_normal_form(form.S)
    assert form.rank == A.rank()


@given(matrices(max_dim=6))
def test_smith_diagonal_is_a_divisibility_chain(A: Matrix) -> None:
    factors = smith_normal_form(A).invariant_factors
    assert all(d > 0 for d in factors)
    assert all(later % earlier == 0 for earlier, later in pairwise(factors))


@given(matrices(max_dim=5))
def test_smith_form_matches_sympy(A: Matrix) -> None:
    assert from_sympy(sympy_snf(to_sympy(A), domain=ZZ)) == smith_normal_form(A).S


@given(matrices(max_dim=4))
def test_invariant_factors_are_ratios_of_determinantal_divisors(A: Matrix) -> None:
    # d_1 * ... * d_k equals the gcd of the k x k minors of A (and 0 past the rank).
    diagonal = smith_normal_form(A).diagonal
    for k in range(1, len(diagonal) + 1):
        assert math.prod(diagonal[:k]) == determinantal_divisor(A, k)


@given(square_matrices(max_dim=5))
def test_smith_diagonal_multiplies_to_the_absolute_determinant(A: Matrix) -> None:
    assert math.prod(smith_normal_form(A).diagonal) == abs(A.determinant())


@given(matrices(max_dim=4), st.data())
def test_smith_form_is_invariant_under_unimodular_changes(A: Matrix, data: st.DataObject) -> None:
    left = data.draw(unimodular_matrices(A.nrows))
    right = data.draw(unimodular_matrices(A.ncols))
    assert smith_normal_form(left @ A @ right).S == smith_normal_form(A).S


@pytest.mark.parametrize("seed", range(5))
def test_smith_transforms_stay_small_on_dense_matrices(seed: int) -> None:
    # Regression guard for coefficient growth. Diagonalizing a dense 10x10 matrix directly
    # gave transform entries with about 17 times as many digits as its determinant; starting
    # from the Hermite form keeps them within a small multiple of it.
    rng = random.Random(seed)
    A = Matrix.of([[rng.randint(-1000, 1000) for _ in range(10)] for _ in range(10)])
    form = smith_normal_form(A)
    digits = max(len(str(abs(x))) for M in (form.U, form.V) for row in M.rows for x in row)
    assert digits <= 3 * len(str(abs(A.determinant())))


def test_smith_on_empty_and_zero_shapes() -> None:
    empty = smith_normal_form(Matrix.of([], ncols=2))
    assert empty.S.shape == (0, 2)
    assert Matrix.identity(2) == empty.V
    assert empty.invariant_factors == ()
    zero = smith_normal_form(Matrix.zeros(2, 3))
    assert Matrix.zeros(2, 3) == zero.S
    assert zero.rank == 0


def test_smith_forces_divisibility_when_the_diagonal_is_already_clear() -> None:
    # diag(2, 3) is diagonal but not Smith: 2 does not divide 3, so it becomes diag(1, 6).
    form = smith_normal_form(Matrix.of([[2, 0], [0, 3]]))
    assert form.invariant_factors == (1, 6)
    assert form.U @ Matrix.of([[2, 0], [0, 3]]) @ form.V == form.S


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([[1, 0, 0], [0, 4, 0]], True),
        ([[3, 0], [0, 0], [0, 0]], True),
        ([[1, 1], [0, 2]], False),  # not diagonal
        ([[-1, 0], [0, 2]], False),  # negative entry
        ([[0, 0], [0, 5]], False),  # a zero before a nonzero
        ([[4, 0], [0, 6]], False),  # 4 does not divide 6
    ],
)
def test_is_smith_normal_form_detects_each_violation(rows: list[list[int]], expected: bool) -> None:
    assert is_smith_normal_form(Matrix.of(rows)) is expected
