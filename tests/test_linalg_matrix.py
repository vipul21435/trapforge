"""The immutable integer matrix layer, checked against sympy and algebraic identities."""

from fractions import Fraction

import pytest
import sympy
from hypothesis import given
from hypothesis import strategies as st

from _strategies import dense_matrices, matrices, square_matrices, to_sympy
from trapforge.linalg import Matrix, ShapeError, dot

dims = st.integers(0, 4)


# -- construction and validation ----------------------------------------------------------


def test_of_converts_index_like_values_to_plain_ints() -> None:
    A = Matrix.of([[True, sympy.Integer(7)], [-3, 0]])
    assert A.rows == ((1, 7), (-3, 0))
    assert all(type(value) is int for row in A.rows for value in row)


@pytest.mark.parametrize("bad", [2.5, Fraction(1, 2), "3"])
def test_of_rejects_values_without_an_exact_integer_meaning(bad: object) -> None:
    with pytest.raises(TypeError):
        Matrix.of([[1, bad]])  # type: ignore[list-item]


def test_constructor_insists_on_plain_int_entries() -> None:
    with pytest.raises(TypeError, match="non-int"):
        Matrix(((1, True),), 2)


def test_ragged_rows_are_rejected() -> None:
    with pytest.raises(ShapeError, match="row 1 has 1 entries"):
        Matrix.of([[1, 2], [3]])


def test_declared_width_is_checked_against_the_rows() -> None:
    with pytest.raises(ShapeError):
        Matrix.of([[1, 2]], ncols=3)
    with pytest.raises(ShapeError):
        Matrix((), -1)


def test_empty_matrices_keep_their_shape() -> None:
    A = Matrix.of([], ncols=3)
    assert A.shape == (0, 3)
    assert A.columns == ((), (), ())
    assert A.transpose().shape == (3, 0)
    assert (A.transpose() @ A) == Matrix.zeros(3, 3)
    assert Matrix.of([]).shape == (0, 0)
    assert str(A) == "<empty 0x3 matrix>"


def test_diagonal_places_entries_and_checks_they_fit() -> None:
    assert Matrix.diagonal([4, 12], 3, 2).rows == ((4, 0), (0, 12), (0, 0))
    with pytest.raises(ShapeError):
        Matrix.diagonal([1, 2, 3], 2, 3)


def test_matrices_are_hashable_values() -> None:
    A = Matrix.of([[1, 2], [3, 4]])
    assert Matrix(((1, 2), (3, 4)), 2) == A
    assert len({A, Matrix.of([[1, 2], [3, 4]]), Matrix.identity(2)}) == 2
    assert Matrix.of([], ncols=2) != Matrix.of([], ncols=3)


def test_str_right_aligns_columns() -> None:
    assert str(Matrix.of([[1, -20], [300, 4]])) == "[  1 -20]\n[300   4]"


def test_indexing_and_to_lists_copy() -> None:
    A = Matrix.of([[1, 2], [3, 4]])
    assert A[1, 0] == 3
    copy = A.to_lists()
    copy[0][0] = 99
    assert A[0, 0] == 1


# -- products -----------------------------------------------------------------------------


def test_shape_mismatches_raise() -> None:
    A = Matrix.of([[1, 2, 3]])
    with pytest.raises(ShapeError):
        A @ A
    with pytest.raises(ShapeError):
        A.apply((1, 2))
    with pytest.raises(ShapeError):
        A.left_apply((1, 2))
    with pytest.raises(ShapeError):
        dot((1, 2), (1, 2, 3))


@given(dense_matrices(dims, dims), st.data())
def test_product_matches_sympy(A: Matrix, data: st.DataObject) -> None:
    B = data.draw(dense_matrices(st.just(A.ncols), dims))
    assert to_sympy(A @ B) == to_sympy(A) * to_sympy(B)


@given(dense_matrices(dims, dims), st.data())
def test_apply_and_left_apply_are_products_with_vectors(A: Matrix, data: st.DataObject) -> None:
    x = data.draw(st.lists(st.integers(-50, 50), min_size=A.ncols, max_size=A.ncols))
    y = data.draw(st.lists(st.integers(-50, 50), min_size=A.nrows, max_size=A.nrows))
    column = Matrix.of([[value] for value in x], ncols=1)
    row = Matrix.of([y], ncols=A.nrows)
    assert A.apply(x) == (A @ column).columns[0]
    assert A.left_apply(y) == (row @ A).rows[0]


@given(dense_matrices(dims, dims))
def test_identity_is_neutral_and_transpose_is_an_involution(A: Matrix) -> None:
    assert Matrix.identity(A.nrows) @ A == A
    assert A @ Matrix.identity(A.ncols) == A
    assert A.transpose().transpose() == A
    assert A.transpose().shape == (A.ncols, A.nrows)


# -- determinant and rank -----------------------------------------------------------------


def test_determinant_needs_a_square_matrix() -> None:
    with pytest.raises(ShapeError):
        Matrix.of([[1, 2]]).determinant()
    assert Matrix.identity(0).determinant() == 1
    assert not Matrix.of([[1, 2]]).is_unimodular()


@given(square_matrices(max_dim=6))
def test_determinant_matches_sympy(A: Matrix) -> None:
    assert A.determinant() == to_sympy(A).det()


@given(square_matrices(max_dim=4), st.data())
def test_determinant_is_multiplicative(A: Matrix, data: st.DataObject) -> None:
    B = data.draw(dense_matrices(st.just(A.nrows), st.just(A.nrows)))
    assert (A @ B).determinant() == A.determinant() * B.determinant()


def test_determinant_is_exact_on_huge_entries() -> None:
    big = 10**40
    A = Matrix.of([[big + 1, big], [big, big - 1]])
    # (b + 1)(b - 1) - b^2 = -1 exactly; any float step would lose it.
    assert A.determinant() == -1
    assert A.is_unimodular()


def test_determinant_of_a_singular_matrix_needing_pivoting() -> None:
    assert Matrix.of([[0, 1, 2], [0, 3, 4], [0, 5, 6]]).determinant() == 0
    assert Matrix.of([[0, 1], [1, 0]]).determinant() == -1


@given(matrices(max_dim=6, min_dim=0))
def test_rank_matches_sympy(A: Matrix) -> None:
    assert A.rank() == to_sympy(A).rank()
    assert A.rank() == A.transpose().rank()


@given(square_matrices(max_dim=5))
def test_full_rank_iff_nonzero_determinant(A: Matrix) -> None:
    assert (A.rank() == A.nrows) == (A.determinant() != 0)
