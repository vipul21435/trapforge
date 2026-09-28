"""Exact integer linear algebra over Python ``int``.

The uniqueness prover turns observations into integer linear systems, so it needs linear
algebra over the integers rather than over the reals: whether ``A x = b`` has an integer
solution, what the whole solution lattice looks like, and how many of its points fit in a
box. This package provides that in pure Python, with no floats and no dependencies, so the
exported reference solvers can vendor it unchanged.
"""

from trapforge.linalg.hermite import HermiteForm, hermite_normal_form, is_hermite_normal_form
from trapforge.linalg.matrix import LinalgError, Matrix, ShapeError, Vector, dot
from trapforge.linalg.smith import SmithForm, is_smith_normal_form, smith_normal_form

__all__ = [
    "HermiteForm",
    "LinalgError",
    "Matrix",
    "ShapeError",
    "SmithForm",
    "Vector",
    "dot",
    "hermite_normal_form",
    "is_hermite_normal_form",
    "is_smith_normal_form",
    "smith_normal_form",
]
