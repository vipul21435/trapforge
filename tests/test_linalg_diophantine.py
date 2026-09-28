"""Integer kernels, affine lattices, box enumeration and linear Diophantine systems.

Oracles: brute force over small boxes (every integer point is tried), sympy's nullspace
dimension and rank for the rational picture, and planted solutions with 30-digit entries.
"""

import itertools
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from _strategies import dense_matrices, matrices, to_sympy, unimodular_matrices
from trapforge.linalg import (
    AffineLattice,
    LinalgError,
    Matrix,
    ShapeError,
    UnsolvableSystem,
    is_hermite_normal_form,
    kernel_basis,
    solve_diophantine,
)

small_dims = st.integers(1, 3)
BOX = 5


def vectors(length: int, bound: int = 20) -> st.SearchStrategy[tuple[int, ...]]:
    return st.lists(st.integers(-bound, bound), min_size=length, max_size=length).map(tuple)


def box_points(n: int, radius: int) -> list[tuple[int, ...]]:
    return list(itertools.product(range(-radius, radius + 1), repeat=n))


# -- kernel bases -------------------------------------------------------------------------


@given(matrices(max_dim=6, min_dim=0))
def test_kernel_basis_is_annihilated_canonical_and_has_the_right_size(A: Matrix) -> None:
    K = kernel_basis(A)
    assert K.shape == (A.ncols - A.rank(), A.ncols)
    assert A @ K.transpose() == Matrix.zeros(A.nrows, K.nrows)
    assert is_hermite_normal_form(K)
    assert K.rank() == K.nrows
    assert K.nrows == len(to_sympy(A).nullspace())


@given(dense_matrices(small_dims, small_dims, st.integers(-3, 3)))
def test_kernel_basis_is_saturated(A: Matrix) -> None:
    # Every integer kernel vector in a box is an integer combination of the basis, which
    # fails for a basis of a proper sublattice such as 2 * (a true basis).
    kernel = AffineLattice.of((0,) * A.ncols, kernel_basis(A))
    zero = (0,) * A.nrows
    for x in box_points(A.ncols, BOX):
        assert (A.apply(x) == zero) == (x in kernel)


@given(matrices(max_dim=5), st.data())
def test_kernel_basis_depends_only_on_the_row_lattice(A: Matrix, data: st.DataObject) -> None:
    W = data.draw(unimodular_matrices(A.nrows))
    assert kernel_basis(W @ A) == kernel_basis(A)


# -- affine lattices ----------------------------------------------------------------------


def test_lattice_constructor_insists_on_the_canonical_form() -> None:
    with pytest.raises(LinalgError, match="Hermite"):
        AffineLattice((0, 0), Matrix.of([[-3, 2]]))
    with pytest.raises(LinalgError, match="Hermite"):
        AffineLattice((0, 0), Matrix.of([[3, 2], [0, 0]]))
    with pytest.raises(LinalgError, match="reduced"):
        AffineLattice((3, 0), Matrix.of([[3, 2]]))
    with pytest.raises(ShapeError):
        AffineLattice((0,), Matrix.of([[3, 2]]))
    with pytest.raises(TypeError, match="non-int"):
        AffineLattice((True, 0), Matrix.of([[3, 2]]))
    with pytest.raises(ShapeError):
        AffineLattice.of((1, 2, 3), Matrix.of([[1, 0]]))


@given(dense_matrices(st.integers(0, 3), st.integers(1, 4), st.integers(-6, 6)), st.data())
def test_lattice_canonical_form_ignores_how_it_was_described(
    G: Matrix, data: st.DataObject
) -> None:
    point = data.draw(vectors(G.ncols))
    shift = data.draw(vectors(G.nrows, bound=5))
    W = data.draw(unimodular_matrices(G.nrows))
    moved = tuple(p + s for p, s in zip(point, G.left_apply(shift), strict=True))
    assert AffineLattice.of(moved, W @ G) == AffineLattice.of(point, G)


@given(dense_matrices(st.integers(0, 3), st.integers(1, 3), st.integers(-4, 4)), st.data())
def test_membership_and_at_agree(G: Matrix, data: st.DataObject) -> None:
    lattice = AffineLattice.of(data.draw(vectors(G.ncols)), G)
    coefficients = data.draw(vectors(lattice.dimension, bound=6))
    member = lattice.at(coefficients)
    assert member in lattice
    assert lattice.point in lattice
    if lattice.dimension < lattice.ambient_dimension:
        # A lattice of lower rank cannot contain every unit step from one of its points.
        n = lattice.ambient_dimension
        neighbours = [tuple(m + int(i == j) for j, m in enumerate(member)) for i in range(n)]
        assert any(neighbour not in lattice for neighbour in neighbours)


def test_membership_rejects_non_vectors() -> None:
    lattice = AffineLattice.of((1, 2))
    assert lattice.is_point
    assert (1, 2) in lattice
    assert [1, 2] in lattice
    assert (1, 2, 0) not in lattice
    assert (1.0, 2) not in lattice
    assert 7 not in lattice
    with pytest.raises(ShapeError):
        lattice.at((1,))


@given(
    dense_matrices(st.integers(0, 3), st.integers(1, 3), st.integers(-4, 4)),
    st.data(),
)
def test_points_in_box_match_brute_force_in_lexicographic_order(
    G: Matrix, data: st.DataObject
) -> None:
    lattice = AffineLattice.of(data.draw(vectors(G.ncols, bound=8)), G)
    n = lattice.ambient_dimension
    lower = data.draw(vectors(n, bound=BOX))
    upper = tuple(lo + data.draw(st.integers(-1, 2 * BOX)) for lo in lower)
    box = itertools.product(*(range(lo, hi + 1) for lo, hi in zip(lower, upper, strict=True)))
    expected = [x for x in box if x in lattice]
    assert list(lattice.points_in_box(lower, upper)) == expected


def test_points_in_box_stays_inside_a_thin_lattice_in_a_huge_box() -> None:
    # Only 11 points of this line lie in a box with about 2 * 10**12 integer points.
    lattice = AffineLattice.of((0, 0), Matrix.of([[100_000, 1]]))
    found = list(lattice.points_in_box((-500_000, -(10**6)), (500_000, 10**6)))
    assert found == [(100_000 * t, t) for t in range(-5, 6)]


def test_points_in_box_checks_its_bounds() -> None:
    lattice = AffineLattice.of((0, 0))
    with pytest.raises(ShapeError):
        list(lattice.points_in_box((0,), (1, 1)))
    assert list(lattice.points_in_box((1, 1), (0, 0))) == []


def test_lattice_str_lists_point_and_generators() -> None:
    assert str(AffineLattice.of((5, 5), Matrix.of([[2, 0], [0, 3]]))) == (
        "x = (1, 2) + t0*(2, 0) + t1*(0, 3), t in Z^2"
    )
    assert str(AffineLattice.of((4, -1))) == "x = (4, -1)"


# -- Diophantine systems ------------------------------------------------------------------


@given(dense_matrices(small_dims, small_dims, st.integers(-4, 4)), st.data())
def test_solver_matches_brute_force_in_a_box(A: Matrix, data: st.DataObject) -> None:
    b = data.draw(vectors(A.nrows, bound=8))
    brute = [x for x in box_points(A.ncols, BOX) if A.apply(x) == b]
    result = solve_diophantine(A, b)
    if isinstance(result, UnsolvableSystem):
        assert brute == []
        assert result.verify(A, b)
    else:
        assert list(result.points_in_box((-BOX,) * A.ncols, (BOX,) * A.ncols)) == brute
        assert A.apply(result.point) == b


@given(matrices(max_dim=5), st.data())
def test_planted_solutions_are_found_with_the_full_kernel(A: Matrix, data: st.DataObject) -> None:
    planted = data.draw(vectors(A.ncols, bound=10**30))
    b = A.apply(planted)
    result = solve_diophantine(A, b)
    assert isinstance(result, AffineLattice)
    assert planted in result
    assert A.apply(result.point) == b
    assert result.basis == kernel_basis(A)


@given(matrices(max_dim=5), st.data())
def test_every_answer_is_either_a_solution_set_or_a_valid_proof(
    A: Matrix, data: st.DataObject
) -> None:
    b = data.draw(vectors(A.nrows, bound=30))
    result = solve_diophantine(A, b)
    augmented = Matrix.of([[*row, rhs] for row, rhs in zip(A.rows, b, strict=True)])
    rationally_solvable = augmented.rank() == A.rank()
    if isinstance(result, UnsolvableSystem):
        assert result.verify(A, b)
        # Rational infeasibility is always reported as such, and only then.
        assert result.is_rational == (not rationally_solvable)
        if not result.is_rational:
            assert result.modulus >= 2
            assert all(2 * abs(w) <= result.modulus for w in result.multiplier)
    else:
        assert rationally_solvable
        assert A.apply(result.point) == b
        assert A @ result.basis.transpose() == Matrix.zeros(A.nrows, result.dimension)


@given(matrices(max_dim=4), st.data())
def test_tampered_certificates_are_rejected(A: Matrix, data: st.DataObject) -> None:
    b = data.draw(vectors(A.nrows, bound=30))
    proof = solve_diophantine(A, b)
    if not isinstance(proof, UnsolvableSystem):
        return
    assert not proof.verify(A, [*b, 0])
    assert not replace(proof, rhs=proof.rhs + 1).verify(A, b)
    assert not replace(proof, modulus=1).verify(A, b)
    assert not replace(proof, modulus=-2).verify(A, b)
    assert not replace(proof, multiplier=proof.multiplier[:-1]).verify(A, b)
    # A multiplier of zeros has rhs 0, which every modulus divides.
    zeros = (0,) * A.nrows
    assert not UnsolvableSystem(zeros, proof.modulus, 0).verify(A, b)


def test_rational_certificate_reads_as_a_contradiction() -> None:
    A = Matrix.of([[1, 2], [2, 4]])
    proof = solve_diophantine(A, [1, 3])
    assert isinstance(proof, UnsolvableSystem)
    assert proof.is_rational
    assert str(proof) == (
        "no solution, not even rational: weighting the equations by (-2, 1) cancels every "
        "unknown but leaves 0 = 1"
    )


def test_integer_certificate_names_the_modulus() -> None:
    A = Matrix.of([[2, 4], [6, 8]])
    proof = solve_diophantine(A, [1, 1])
    assert isinstance(proof, UnsolvableSystem)
    assert not proof.is_rational
    assert proof.verify(A, [1, 1])
    assert "multiple of 2" in str(proof)


def test_solver_checks_the_right_hand_side_length() -> None:
    with pytest.raises(ShapeError):
        solve_diophantine(Matrix.of([[1, 2]]), [1, 2])


def test_solver_handles_empty_shapes() -> None:
    free = solve_diophantine(Matrix.of([], ncols=2), [])
    assert free == AffineLattice.of((0, 0), Matrix.identity(2))
    assert solve_diophantine(Matrix.of([[], []], ncols=0), [0, 0]) == AffineLattice.of(())
    proof = solve_diophantine(Matrix.of([[], []], ncols=0), [0, 5])
    assert isinstance(proof, UnsolvableSystem)
    assert proof.verify(Matrix.of([[], []], ncols=0), [0, 5])


def test_box_bounds_pin_a_single_point_on_a_kernel_line() -> None:
    # A toy version of what the prover will do: three hidden digits in [0, 9] observed only
    # through two weighted checksums. The solution set is a line, but the box pins one point.
    A = Matrix.of([[1, 10, 100], [7, 3, 1]])
    hidden = (4, 2, 7)
    result = solve_diophantine(A, A.apply(hidden))
    assert isinstance(result, AffineLattice)
    assert result.dimension == 1
    assert list(result.points_in_box((0, 0, 0), (9, 9, 9))) == [hidden]
