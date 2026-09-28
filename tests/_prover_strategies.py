"""Hypothesis strategies for small constraint systems, plus a brute-force oracle.

The systems are small enough to enumerate outright (at most three unknowns with boxes of at
most seven values, at most three cases), yet they mix every feature of the model: equations,
congruences, choice-dependent coefficients, moduli and bounds, and empty boxes.
"""

from __future__ import annotations

import itertools

from hypothesis import strategies as st

from trapforge.prover import ChoiceTerm, Congruent, Constraint, ConstraintSystem, Equation, Scalar

NAMES = ("x", "y", "z")


@st.composite
def small_systems(draw: st.DrawFn) -> ConstraintSystem:
    """A random system with 1-3 unknowns, 0-4 constraints and at most one choice ``P``."""
    has_choice = draw(st.booleans())
    options = (
        draw(st.lists(st.integers(1, 6), min_size=1, max_size=3, unique=True)) if has_choice else []
    )
    choices = {"P": options} if has_choice else {}

    def scalar(low: int, high: int) -> st.SearchStrategy[Scalar]:
        plain: st.SearchStrategy[Scalar] = st.integers(low, high)
        if not has_choice:
            return plain
        term = st.builds(ChoiceTerm, st.just("P"), st.sampled_from([1, -1, 2]), st.integers(-2, 2))
        return st.one_of(plain, plain, term)

    names = NAMES[: draw(st.integers(1, 3))]
    unknowns: dict[str, tuple[Scalar, Scalar]] = {}
    for name in names:
        lower = draw(st.integers(-4, 2))
        upper: Scalar = lower + draw(st.integers(-1, 6))
        if has_choice and draw(st.integers(0, 4)) == 0:
            upper = ChoiceTerm("P", 1, -1)
        unknowns[name] = (lower, upper)

    constraints: list[Constraint] = []
    for index in range(draw(st.integers(0, 4))):
        used = draw(st.lists(st.sampled_from(names), min_size=0, max_size=len(names), unique=True))
        terms = {name: draw(scalar(-4, 4)) for name in used}
        rhs = draw(scalar(-8, 8))
        if draw(st.booleans()):
            constraints.append(Equation.of(terms, rhs, f"e{index}"))
        else:
            modulus: Scalar = draw(st.integers(1, 8))
            if has_choice and draw(st.booleans()):
                modulus = ChoiceTerm("P")
            constraints.append(Congruent.of(terms, rhs, modulus, f"c{index}"))
    return ConstraintSystem.build(unknowns, constraints, choices)


def brute_force(system: ConstraintSystem) -> list[dict[str, int]]:
    """Every solution, in case order and then lexicographic order of the unknowns."""
    found: list[dict[str, int]] = []
    for choices in system.cases():
        instance = system.instantiate(choices)
        ranges = [range(lo, hi + 1) for lo, hi in zip(instance.lower, instance.upper, strict=True)]
        for values in itertools.product(*ranges):
            if instance.contains(values):
                found.append({**choices, **dict(zip(system.names, values, strict=True))})
    return found
