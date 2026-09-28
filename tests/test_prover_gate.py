"""The uniqueness gate: shortest prefix of extra observations that makes the proof unique."""

import random
from collections.abc import Iterator

import pytest
from hypothesis import given
from hypothesis import strategies as st

from trapforge.prover import (
    Ambiguity,
    Congruent,
    ConstraintSystem,
    Equation,
    GateError,
    UniqueProof,
    gate,
    prove,
)

HIDDEN = {"a": 5, "b": 7}


def anchor(x: int, y: int, modulus: int = 12) -> Congruent:
    return Congruent.of({"a": x, "b": 1}, y, modulus, f"anchor x={x}")


def ledger(modulus: int, xs: list[int], a: int, b: int) -> list[Congruent]:
    return [anchor(x, (a * x + b) % modulus, modulus) for x in xs]


SAMPLE = ConstraintSystem.build({"a": (1, 11), "b": (0, 11)}, ledger(12, [1, 3], 5, 7))


def test_a_unique_sample_passes_without_extra_constraints() -> None:
    unique = SAMPLE.extend(anchor(2, 5))
    result = gate(unique, [anchor(4, 3)], hidden=HIDDEN)
    assert result.passed
    assert result.added == ()
    assert result.sizes == (1,)
    assert result.system is unique
    assert str(result) == "passed after 0 more constraints (sizes 1): unique: a=5, b=7"


def test_the_gate_stops_at_the_first_prefix_that_is_unique() -> None:
    # x = 7 gives the same reading under both worlds, so it does not help; x = 2 does.
    consumed: list[int] = []

    def stream() -> Iterator[Congruent]:
        for x in (7, 2, 4, 5):
            consumed.append(x)
            yield anchor(x, (5 * x + 7) % 12)

    result = gate(SAMPLE, stream(), hidden=HIDDEN)
    assert result.passed
    assert isinstance(result.proof, UniqueProof)
    assert dict(result.proof.solution) == HIDDEN
    assert [c.label for c in result.added] == ["anchor x=7", "anchor x=2"]
    assert result.sizes == (2, 2, 1)
    assert consumed == [7, 2]
    assert result.system.constraints[-2:] == result.added
    assert result.proof.check().valid


def test_the_sample_is_rejected_when_the_candidates_run_out() -> None:
    result = gate(SAMPLE, [anchor(7, 6), anchor(9, 4)])
    assert not result.passed
    assert isinstance(result.proof, Ambiguity)
    assert result.sizes == (2, 2, 2)
    assert len(result.added) == 2
    assert str(result).startswith("rejected after 2 more constraints (sizes 2, 2, 2): ambiguous")


def test_max_added_bounds_the_extension() -> None:
    result = gate(SAMPLE, [anchor(7, 6), anchor(2, 5)], max_added=1)
    assert not result.passed
    assert [c.label for c in result.added] == ["anchor x=7"]
    assert str(result).startswith("rejected after 1 more constraint (sizes 2, 2)")
    assert gate(SAMPLE, [anchor(2, 5)], max_added=0).added == ()
    with pytest.raises(ValueError, match="max_added must be at least 0"):
        gate(SAMPLE, [], max_added=-1)


def test_sizes_above_the_cap_are_recorded_as_cap_plus_one() -> None:
    wide = ConstraintSystem.build({"x": (0, 999)})
    steps = [
        Congruent.of({"x": 1}, 7, 10),
        Congruent.of({"x": 1}, 7, 100),
        Equation.of({"x": 1}, 7),
    ]
    result = gate(wide, steps, cap=50)
    assert result.passed
    assert result.sizes == (51, 51, 10, 1)


def test_a_planted_world_that_breaks_the_sample_is_a_generator_bug() -> None:
    with pytest.raises(GateError, match="the planted world violates anchor x=1, anchor x=3"):
        gate(SAMPLE, hidden={"a": 5, "b": 8})
    with pytest.raises(GateError, match="violates bounds of a"):
        gate(SAMPLE, hidden={"a": 17, "b": 7})
    with pytest.raises(GateError, match="violates anchor x=2"):
        gate(SAMPLE, [anchor(7, 6), anchor(2, 6)], hidden=HIDDEN)


def test_an_infeasible_sample_is_a_generator_bug() -> None:
    with pytest.raises(GateError, match=r"infeasible \(last constraint: anchor x=2\)"):
        gate(SAMPLE, [anchor(2, 4)])
    empty_box = ConstraintSystem.build({"x": (3, 2)})
    with pytest.raises(GateError, match=r"infeasible \(last constraint: bounds\)"):
        gate(empty_box)


@given(st.integers(2, 40), st.data())
def test_every_planted_affine_map_passes_once_all_anchors_are_seen(
    modulus: int, data: st.DataObject
) -> None:
    a = data.draw(st.integers(1, modulus - 1), label="a") if modulus > 2 else 1
    b = data.draw(st.integers(0, modulus - 1), label="b")
    seed = data.draw(st.integers(0, 2**32 - 1), label="seed")
    xs = list(range(modulus))
    random.Random(seed).shuffle(xs)
    sample = ConstraintSystem.build(
        {"a": (1, modulus - 1), "b": (0, modulus - 1)}, ledger(modulus, xs[:2], a, b)
    )
    result = gate(sample, ledger(modulus, xs[2:], a, b), hidden={"a": a, "b": b})
    # Readings at every x reveal b (at x = 0) and a + b (at x = 1), so the gate must pass.
    assert result.passed
    assert isinstance(result.proof, UniqueProof)
    assert dict(result.proof.solution) == {"a": a, "b": b}
    assert list(result.sizes) == sorted(result.sizes, reverse=True)
    assert result.sizes[-1] == 1
    # The prefix is the shortest one: without its last constraint the sample is ambiguous.
    if result.added:
        shorter = ConstraintSystem(sample.unknowns, result.system.constraints[:-1], sample.choices)
        assert isinstance(prove(shorter), Ambiguity)
