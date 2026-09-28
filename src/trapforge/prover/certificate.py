"""Independent re-checking of the prover's JSON certificates.

A certificate says: "under this constraint system, the consistent hidden parameterizations
are exactly these" (one, several, or none). This module checks such a claim without
running the solver and without importing anything else from TrapForge: it uses the standard
library and integer arithmetic only, so it is small enough to audit or vendor on its own.

What is checked, case by case (a case fixes every discrete choice):

1. The certificate lists every case of the choice product, in order, so no case can be
   skipped.
2. Each case is rebuilt from the embedded system as an integer system ``M z = r`` over
   ``z = (unknowns, one slack per congruence)``; ``a . x = b (mod m)`` becomes
   ``a . x - m*s = b``.
3. The case carries one of two kinds of evidence:

   * ``unsolvable``: weights ``w`` and a modulus ``d`` with every entry of ``w M`` divisible
     by ``d`` and ``w . r`` not (``d = 0`` means ``w M = 0`` and ``w . r != 0``). Then no
     integer ``z`` can satisfy the case.
   * ``lattice``: a point ``p`` and basis rows ``K`` claimed to give *every* integer
     solution as ``p + (integer combinations of K)``. The checker verifies ``M p = r`` and
     ``M K^T = 0``; a right inverse ``W`` with ``K W = I`` (so the rows of ``K`` are
     independent and span every integer point of their rational span); and a nonsingular
     square submatrix of ``M`` of size ``len(z) - len(K)`` (so the rational kernel of ``M``
     has dimension at most ``len(K)``). Together these force the lattice to be the complete
     integer solution set. The unknown part of ``K`` must be in echelon form with positive
     leading entries, which lets the checker enumerate the lattice points inside the box of
     bounds exactly, in lexicographic order.

4. The points found across all cases (up to ``cap + 1`` of them) must match the claimed
   verdict, count and witnesses.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FORMAT",
    "VERSION",
    "CertificateCheck",
    "check_certificate",
]

FORMAT = "trapforge-uniqueness-certificate"
VERSION = 1


@dataclass(frozen=True, slots=True)
class CertificateCheck:
    """The outcome of :func:`check_certificate`; truthy exactly when the certificate holds."""

    valid: bool
    reason: str
    verdict: str | None = None
    count: int | None = None
    exact: bool | None = None

    def __bool__(self) -> bool:
        return self.valid


class _RejectError(Exception):
    """Raised inside the checker to reject a certificate, with the reason as message."""


def _int(value: Any, what: str) -> int:
    if type(value) is not int:
        raise _RejectError(f"{what} must be an integer, got {value!r}")
    return value


def _int_list(values: Any, what: str, length: int | None = None) -> list[int]:
    if not isinstance(values, list):
        raise _RejectError(f"{what} must be a list")
    if length is not None and len(values) != length:
        raise _RejectError(f"{what} must have {length} entries, got {len(values)}")
    return [_int(value, what) for value in values]


def _scalar(value: Any, choices: Mapping[str, int]) -> int:
    if type(value) is int:
        return value
    if isinstance(value, Mapping) and set(value) == {"choice", "scale", "offset"}:
        if value["choice"] not in choices:
            raise _RejectError(f"unknown choice {value['choice']!r}")
        scale = _int(value["scale"], "a choice scale")
        return scale * choices[value["choice"]] + _int(value["offset"], "a choice offset")
    raise _RejectError(f"not an integer or a choice term: {value!r}")


@dataclass(frozen=True, slots=True)
class _Case:
    """One case rebuilt as ``matrix z = rhs`` over ``z = (unknowns, one slack per congruence)``.

    ``width`` is the length of ``z``. It is stored rather than read off the matrix, because
    a system without constraints has no rows to read it from.
    """

    names: list[str]
    lower: list[int]
    upper: list[int]
    matrix: list[list[int]]
    rhs: list[int]
    width: int


def _build_case(system: Mapping[str, Any], choices: Mapping[str, int]) -> _Case:
    """Rebuild one case as ``M z = r`` over ``z = (unknowns, slacks)``."""
    names = [unknown["name"] for unknown in system["unknowns"]]
    if not all(isinstance(name, str) for name in names):
        raise _RejectError("unknown names must be strings")
    if len(set(names)) != len(names) or not set(names).isdisjoint(choices):
        raise _RejectError("a name is used twice among the unknowns and choices")
    position = {name: index for index, name in enumerate(names)}
    constraints = system["constraints"]
    slacks = sum(1 for constraint in constraints if constraint["kind"] == "congruence")
    width = len(names) + slacks
    matrix: list[list[int]] = []
    rhs: list[int] = []
    slack = len(names)
    for constraint in constraints:
        row = [0] * width
        for name, coefficient in constraint["terms"]:
            if name not in position:
                raise _RejectError(f"a constraint uses the undeclared unknown {name!r}")
            row[position[name]] += _scalar(coefficient, choices)
        if constraint["kind"] == "congruence":
            modulus = _scalar(constraint["modulus"], choices)
            if modulus < 1:
                raise _RejectError("a congruence has a modulus below 1")
            row[slack] = -modulus
            slack += 1
        elif constraint["kind"] != "equation":
            raise _RejectError(f"unknown constraint kind {constraint['kind']!r}")
        matrix.append(row)
        rhs.append(_scalar(constraint["rhs"], choices))
    lower = [_scalar(unknown["lower"], choices) for unknown in system["unknowns"]]
    upper = [_scalar(unknown["upper"], choices) for unknown in system["unknowns"]]
    return _Case(names, lower, upper, matrix, rhs, width)


def _dot(u: Sequence[int], v: Sequence[int]) -> int:
    return sum(a * b for a, b in zip(u, v, strict=True))


def _divides(d: int, value: int) -> bool:
    return value == 0 if d == 0 else value % d == 0


def _determinant(rows: list[list[int]]) -> int:
    """Exact determinant by fraction-free (Bareiss) elimination; 1 for the empty matrix."""
    a = [list(row) for row in rows]
    n = len(a)
    sign, previous = 1, 1
    for k in range(n - 1):
        if a[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if a[i][k]), None)
            if swap is None:
                return 0
            a[k], a[swap] = a[swap], a[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // previous
        previous = a[k][k]
    return sign * a[n - 1][n - 1] if n else 1


def _check_unsolvable(evidence: Mapping[str, Any], case: _Case) -> None:
    weights = _int_list(evidence["multiplier"], "multiplier", len(case.matrix))
    d = _int(evidence["modulus"], "modulus")
    if d < 0 or d == 1:
        raise _RejectError("an infeasibility modulus must be 0 or at least 2")
    columns = range(case.width)
    combined = [_dot(weights, [row[j] for row in case.matrix]) for j in columns]
    if not all(_divides(d, value) for value in combined):
        raise _RejectError("the weighted equations do not cancel modulo the claimed modulus")
    if _divides(d, _dot(weights, case.rhs)):
        raise _RejectError("the weighted right-hand side is compatible, so nothing is proved")


def _check_rank_witness(minor: Any, case: _Case, size: int) -> None:
    """Check that ``minor`` names a nonsingular ``size x size`` submatrix of the case."""
    rows = _int_list(minor["rows"], "minor rows")
    columns = _int_list(minor["columns"], "minor columns")
    if len(rows) != size or len(columns) != size:
        raise _RejectError(f"the rank witness must be a {size}x{size} minor")
    if len(set(rows)) != size or len(set(columns)) != size:
        raise _RejectError("the rank witness repeats a row or a column")
    if not all(0 <= r < len(case.matrix) for r in rows) or not all(
        0 <= c < case.width for c in columns
    ):
        raise _RejectError("the rank witness points outside the system")
    if _determinant([[case.matrix[r][c] for c in columns] for r in rows]) == 0:
        raise _RejectError("the rank witness minor is singular")


def _echelon_pivots(basis: list[list[int]], n: int) -> list[int]:
    """Leading columns of the unknown part of ``basis``, which must be in echelon form."""
    pivots: list[int] = []
    for vector in basis:
        lead = next((j for j in range(n) if vector[j]), None)
        if lead is None or (pivots and lead <= pivots[-1]) or vector[lead] < 0:
            raise _RejectError("the basis is not in echelon form over the unknowns")
        pivots.append(lead)
    return pivots


def _check_lattice(
    evidence: Mapping[str, Any], case: _Case
) -> tuple[list[int], list[list[int]], list[int]]:
    """Verify that ``point + span(basis)`` is exactly the integer solution set of the case.

    Returns the point, the basis and the pivot columns of the unknown part of the basis.
    """
    width = case.width
    point = _int_list(evidence["point"], "point", width)
    if [_dot(row, point) for row in case.matrix] != case.rhs:
        raise _RejectError("the point does not satisfy the constraints")
    if not isinstance(evidence["basis"], list):
        raise _RejectError("basis must be a list of rows")
    basis = [_int_list(row, "basis row", width) for row in evidence["basis"]]
    if any(_dot(row, vector) for row in case.matrix for vector in basis):
        raise _RejectError("a basis row is not a solution of the homogeneous system")
    k = len(basis)
    right = evidence["right_inverse"]
    if not isinstance(right, list) or len(right) != width:
        raise _RejectError(f"right_inverse must have {width} rows")
    inverse = [_int_list(row, "right_inverse row", k) for row in right]
    for i, vector in enumerate(basis):
        for j in range(k):
            if _dot(vector, [row[j] for row in inverse]) != int(i == j):
                raise _RejectError("basis times right_inverse is not the identity")
    _check_rank_witness(evidence["minor"], case, width - k)
    return point, basis, _echelon_pivots(basis, len(case.names))


def _box_points(
    point: list[int], basis: list[list[int]], pivots: list[int], case: _Case, budget: int
) -> list[tuple[int, ...]]:
    """Lattice points inside the box, in lexicographic order, at most ``budget`` of them."""
    n = len(case.names)
    found: list[tuple[int, ...]] = []

    def walk(level: int, x: list[int]) -> Iterator[None]:
        start = pivots[level - 1] + 1 if level else 0
        stop = pivots[level] if level < len(pivots) else n
        if any(not case.lower[j] <= x[j] <= case.upper[j] for j in range(start, stop)):
            return
        if level == len(pivots):
            found.append(tuple(x))
            yield None
            return
        row, column = basis[level][:n], pivots[level]
        step = row[column]
        first = -((x[column] - case.lower[column]) // step)
        last = (case.upper[column] - x[column]) // step
        for t in range(first, last + 1):
            yield from walk(level + 1, [a + t * b for a, b in zip(x, row, strict=True)])

    for _ in itertools.islice(walk(0, point[:n]), budget):
        pass
    return found


def _claim(verdict: Any, count: int, exact: bool) -> str:
    return f"{verdict} with {count if exact else f'more than {count}'} solutions"


def _verify(data: Mapping[str, Any]) -> CertificateCheck:
    if data["format"] != FORMAT or data["version"] != VERSION:
        raise _RejectError(f"not a {FORMAT} version {VERSION}")
    cap = _int(data["cap"], "cap")
    if cap < 2:
        raise _RejectError("cap must be at least 2")
    system = data["system"]
    choice_names = [choice["name"] for choice in system["choices"]]
    options = [_int_list(choice["options"], "options") for choice in system["choices"]]
    if any(len(set(values)) != len(values) for values in options):
        raise _RejectError("a choice lists an option twice")
    expected_cases = [
        dict(zip(choice_names, values, strict=True)) for values in itertools.product(*options)
    ]
    cases = data["cases"]
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        raise _RejectError(f"the certificate must cover all {len(expected_cases)} cases")
    found: list[dict[str, int]] = []
    for index, (choices, evidence) in enumerate(zip(expected_cases, cases, strict=True)):
        if evidence["choices"] != choices:
            raise _RejectError(f"case {index} should be {choices}")
        case = _build_case(system, choices)
        if evidence["evidence"] == "unsolvable":
            _check_unsolvable(evidence, case)
            continue
        if evidence["evidence"] != "lattice":
            raise _RejectError(f"unknown evidence kind {evidence['evidence']!r}")
        point, basis, pivots = _check_lattice(evidence, case)
        budget = cap + 1 - len(found)
        for x in _box_points(point, basis, pivots, case, budget):
            found.append({**choices, **dict(zip(case.names, x, strict=True))})
    exact = len(found) <= cap
    count = min(len(found), cap)
    verdict = "infeasible" if not found else "unique" if len(found) == 1 else "ambiguous"
    witnesses = found[: {"infeasible": 0, "unique": 1, "ambiguous": 2}[verdict]]
    claimed_count = _int(data["count"], "count")
    if type(data["exact"]) is not bool:
        raise _RejectError("exact must be true or false")
    claimed = (data["verdict"], claimed_count, data["exact"])
    if claimed != (verdict, count, exact):
        raise _RejectError(
            f"the certificate claims {_claim(*claimed)}, but its evidence gives "
            f"{_claim(verdict, count, exact)}"
        )
    if data["witnesses"] != witnesses:
        raise _RejectError("the witnesses are not the first solutions the evidence gives")
    size = f"{count} {'solution' if count == 1 else 'solutions'}" if exact else f"more than {cap}"
    return CertificateCheck(True, f"verified: {verdict}, {size}", verdict, count, exact)


def check_certificate(certificate: Mapping[str, Any] | str) -> CertificateCheck:
    """Re-check a prover certificate (a JSON string or the parsed object) from scratch.

    Never raises for a bad certificate: malformed or false certificates come back as an
    invalid :class:`CertificateCheck` with the reason.
    """
    try:
        data = json.loads(certificate) if isinstance(certificate, str) else certificate
        return _verify(data)
    except _RejectError as error:
        return CertificateCheck(False, str(error))
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        return CertificateCheck(False, f"malformed certificate: {error!r}")
