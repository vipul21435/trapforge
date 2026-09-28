"""The exact solver: how many hidden worlds fit a constraint system, with a certificate.

:func:`prove` handles the case split one case at a time. A case (every choice fixed) is a
purely linear problem over bounded integers, which :func:`reduce_case` turns into one integer
system ``M z = r`` over ``z = (unknowns, one slack per congruence)``: the congruence
``a . x = b (mod m)`` becomes the equation ``a . x - m*s = b``. :func:`~trapforge.linalg.
solve_diophantine` then returns either a certificate that the case has no integer solution,
or every integer solution as an affine lattice. Every nonzero kernel vector moves some
unknown (a slack is fixed once the unknowns are, because ``m >= 1``), so the lattice
projects one-to-one onto the unknowns, and its points inside the box of bounds are listed
exactly, in lexicographic order.

The result is one of three verdicts, each carrying a JSON certificate that
:func:`~trapforge.prover.certificate.check_certificate` re-checks without this module:

* :class:`UniqueProof`: exactly one hidden world fits, and here it is;
* :class:`Ambiguity`: at least two fit; a concrete counterexample pair plus the exact size of
  the ambiguity space, or a lower bound when it exceeds ``cap``;
* :class:`Infeasible`: nothing fits (for a generated sample, a bug in the generator).

For a case with solutions the certificate carries the lattice point and basis, an integer
right inverse of the basis (so the basis spans every integer point of its rational span) and
a nonsingular square submatrix of ``M`` (so the kernel has no further dimensions). For a case
without, it carries the weights and modulus of :class:`~trapforge.linalg.UnsolvableSystem`.

>>> from trapforge.prover import ConstraintSystem, Congruent
>>> anchors = ConstraintSystem.build(
...     {"a": (1, 11), "b": (0, 11)},
...     [Congruent.of({"a": 1, "b": 1}, 0, 12), Congruent.of({"a": 3, "b": 1}, 10, 12)],
... )
>>> print(prove(anchors))
ambiguous: exactly 2 worlds fit, for example a=5, b=7 and a=11, b=1 (they differ in a, b)
>>> proof = prove(anchors.extend(Congruent.of({"a": 2, "b": 1}, 5, 12)))
>>> print(proof)
unique: a=5, b=7
>>> proof.check().reason
'verified: unique, 1 solution'
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trapforge.linalg import (
    AffineLattice,
    Matrix,
    UnsolvableSystem,
    hermite_normal_form,
    smith_normal_form,
    solve_diophantine,
)
from trapforge.prover.certificate import FORMAT, VERSION, CertificateCheck, check_certificate
from trapforge.prover.model import ConstraintSystem, Instance

__all__ = [
    "DEFAULT_CAP",
    "Ambiguity",
    "Infeasible",
    "Proof",
    "UniqueProof",
    "certificate_json",
    "prove",
    "reduce_case",
]

DEFAULT_CAP = 1000
"""How many solutions :func:`prove` counts before it reports "more than cap"."""


def _dump(value: Any, depth: int) -> str:
    pad = "  " * depth
    if isinstance(value, Mapping) and value:
        items = [
            f"{pad}  {json.dumps(key)}: {_dump(value[key], depth + 1)}" for key in sorted(value)
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list) and any(isinstance(item, list | Mapping) for item in value):
        items = [f"{pad}  {_dump(item, depth + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(value, separators=(", ", ": "))


def certificate_json(certificate: Mapping[str, Any]) -> str:
    """The canonical text of a certificate: ASCII, sorted keys, final newline.

    Objects and lists of lists get one entry per line; a list of plain values (a vector, a
    matrix row) stays on one line, so matrices read as matrices.

    >>> print(certificate_json({"b": [[1, 2], [3, 4]], "a": {"x": [5, "y"]}}), end="")
    {
      "a": {
        "x": [5, "y"]
      },
      "b": [
        [1, 2],
        [3, 4]
      ]
    }
    """
    return _dump(certificate, 0) + "\n"


def _format_world(world: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={value}" for name, value in world.items())


class _Certified:
    """Shared behaviour of the three verdicts: re-checking and printing the certificate."""

    __slots__ = ()
    certificate: Mapping[str, Any]

    def check(self) -> CertificateCheck:
        """Re-check the certificate with the standalone checker."""
        return check_certificate(self.certificate)

    def certificate_json(self) -> str:
        """The certificate as canonical JSON text."""
        return certificate_json(self.certificate)


@dataclass(frozen=True, slots=True)
class UniqueProof(_Certified):
    """Exactly one hidden world fits: ``solution`` maps every choice and unknown to its value."""

    system: ConstraintSystem
    solution: Mapping[str, int]
    certificate: Mapping[str, Any]

    verdict = "unique"

    def __str__(self) -> str:
        return f"unique: {_format_world(self.solution)}"


@dataclass(frozen=True, slots=True)
class Ambiguity(_Certified):
    """At least two hidden worlds fit.

    ``first`` and ``second`` are the two smallest in case order, then lexicographic order of
    the unknowns. ``count`` is the number of worlds when ``exact`` is True, and the cap (a
    strict lower bound) when it is False.
    """

    system: ConstraintSystem
    first: Mapping[str, int]
    second: Mapping[str, int]
    count: int
    exact: bool
    certificate: Mapping[str, Any]

    verdict = "ambiguous"

    @property
    def differing(self) -> tuple[str, ...]:
        """The choices and unknowns on which the counterexample pair disagrees."""
        return tuple(name for name in self.first if self.first[name] != self.second[name])

    @property
    def size(self) -> str:
        """The size of the ambiguity space in words: ``exactly 6`` or ``more than 1000``."""
        return f"exactly {self.count}" if self.exact else f"more than {self.count}"

    def __str__(self) -> str:
        return (
            f"ambiguous: {self.size} worlds fit, for example {_format_world(self.first)} and "
            f"{_format_world(self.second)} (they differ in {', '.join(self.differing)})"
        )


@dataclass(frozen=True, slots=True)
class Infeasible(_Certified):
    """No hidden world fits: every case is unsolvable over the integers or misses the box."""

    system: ConstraintSystem
    certificate: Mapping[str, Any]

    verdict = "infeasible"

    def __str__(self) -> str:
        return "infeasible: no world fits every constraint and bound"


Proof = UniqueProof | Ambiguity | Infeasible
"""What :func:`prove` returns."""


def reduce_case(instance: Instance) -> tuple[Matrix, tuple[int, ...]]:
    """One case as ``M z = r`` over ``z = (unknowns, one slack per congruence)``.

    Rows follow the constraints and slack columns follow the congruences, both in order.

    >>> from trapforge.prover import ConstraintSystem, Congruent, Equation
    >>> case = ConstraintSystem.build(
    ...     {"x": (0, 9), "y": (0, 9)},
    ...     [Equation.of({"x": 1, "y": 1}, 7), Congruent.of({"x": 2}, 1, 5)],
    ... ).instantiate({})
    >>> matrix, rhs = reduce_case(case)
    >>> print(matrix)
    [ 1  1  0]
    [ 2  0 -5]
    >>> rhs
    (7, 1)
    """
    n = len(instance.names)
    slacks = sum(1 for row in instance.rows if row.modulus is not None)
    rows: list[list[int]] = []
    slack = n
    for row in instance.rows:
        entries = [*row.coefficients, *([0] * slacks)]
        if row.modulus is not None:
            entries[slack] = -row.modulus
            slack += 1
        rows.append(entries)
    return Matrix.of(rows, ncols=n + slacks), tuple(row.rhs for row in instance.rows)


def _right_inverse(basis: Matrix) -> list[list[int]]:
    """An integer ``W`` with ``basis @ W == I``; it exists because the basis is saturated.

    With ``U @ basis @ V == [I | 0]`` in Smith form, ``W`` is the first ``k`` columns of
    ``V`` times ``U``.
    """
    k = basis.nrows
    if k == 0:
        return [[] for _ in range(basis.ncols)]
    smith = smith_normal_form(basis)
    if smith.invariant_factors != (1,) * k:  # pragma: no cover - kernels are saturated
        raise AssertionError("the kernel basis does not span a saturated lattice")
    head = Matrix(tuple(row[:k] for row in smith.V.rows), k)
    return (head @ smith.U).to_lists()


def _rank_witness(matrix: Matrix) -> dict[str, list[int]]:
    """Rows and columns of a nonsingular ``rank x rank`` submatrix of ``matrix``.

    The Hermite pivots of ``matrix`` are independent columns and those of its transpose are
    independent rows; a row basis and a column basis always meet in a nonsingular minor.
    """
    rows = hermite_normal_form(matrix.transpose()).pivots
    columns = hermite_normal_form(matrix).pivots
    return {"rows": list(rows), "columns": list(columns)}


def _project(lattice: AffineLattice, n: int) -> AffineLattice:
    """The lattice restricted to its first ``n`` coordinates (the unknowns).

    Every Hermite pivot of the kernel lies among the unknowns, so the truncated basis is
    still in Hermite form and the truncated point is still reduced.
    """
    basis = Matrix(tuple(row[:n] for row in lattice.basis.rows), n)
    return AffineLattice(lattice.point[:n], basis)


def prove(system: ConstraintSystem, *, cap: int = DEFAULT_CAP) -> Proof:
    """Decide exactly how many hidden worlds fit ``system`` and certify the answer.

    Counting stops after ``cap + 1`` worlds, so an ambiguous system is cheap to reject even
    when its ambiguity space is huge; ``cap`` must be at least 2.
    """
    if type(cap) is not int or cap < 2:
        raise ValueError(f"cap must be an int of at least 2, got {cap!r}")
    cases: list[dict[str, Any]] = []
    found: list[dict[str, int]] = []
    for choices in system.cases():
        instance = system.instantiate(choices)
        matrix, rhs = reduce_case(instance)
        result = solve_diophantine(matrix, rhs)
        if isinstance(result, UnsolvableSystem):
            cases.append(
                {
                    "choices": choices,
                    "evidence": "unsolvable",
                    "multiplier": list(result.multiplier),
                    "modulus": result.modulus,
                }
            )
            continue
        cases.append(
            {
                "choices": choices,
                "evidence": "lattice",
                "point": list(result.point),
                "basis": result.basis.to_lists(),
                "right_inverse": _right_inverse(result.basis),
                "minor": _rank_witness(matrix),
            }
        )
        inside = _project(result, len(instance.names)).points_in_box(instance.lower, instance.upper)
        for point in itertools.islice(inside, cap + 1 - len(found)):
            found.append({**choices, **dict(zip(instance.names, point, strict=True))})
    exact = len(found) <= cap
    count = min(len(found), cap)
    verdict = "infeasible" if not found else "unique" if len(found) == 1 else "ambiguous"
    certificate = {
        "format": FORMAT,
        "version": VERSION,
        "cap": cap,
        "system": system.to_json_dict(),
        "verdict": verdict,
        "count": count,
        "exact": exact,
        "witnesses": found[:2],
        "cases": cases,
    }
    if verdict == "infeasible":
        return Infeasible(system, certificate)
    if verdict == "unique":
        return UniqueProof(system, found[0], certificate)
    return Ambiguity(system, found[0], found[1], count, exact, certificate)
