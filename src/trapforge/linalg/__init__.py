"""Exact integer linear algebra over Python ``int``.

The uniqueness prover turns observations into integer linear systems, so it needs linear
algebra over the integers rather than over the reals: whether ``A x = b`` has an integer
solution, what the whole solution lattice looks like, and how many of its points fit in a
box. This package provides that in pure Python, with no floats and no dependencies, so the
exported reference solvers can vendor it unchanged.
"""

from trapforge.linalg.matrix import LinalgError, Matrix, ShapeError, Vector, dot

__all__ = [
    "LinalgError",
    "Matrix",
    "ShapeError",
    "Vector",
    "dot",
]
