"""The exact solver: verdicts, counterexample pairs, capped counts and certificates."""

import json
import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from _prover_strategies import brute_force, small_systems
from trapforge.modular import Congruence, crt
from trapforge.prover import (
    Ambiguity,
    ChoiceTerm,
    Congruent,
    ConstraintSystem,
    Equation,
    Infeasible,
    UniqueProof,
    certificate_json,
    check_certificate,
    prove,
    reduce_case,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name: str) -> ConstraintSystem:
    return ConstraintSystem.from_json_dict(json.loads((EXAMPLES / name).read_text()))


LEDGER = load("ledger-anchors.json")
CLOCK = load("wrapping-clock.json")


# -- reduction ------------------------------------------------------------------------------


def test_each_congruence_gets_its_own_slack_column_in_order() -> None:
    system = ConstraintSystem.build(
        {"x": (0, 5), "y": (0, 5)},
        [
            Congruent.of({"x": 1}, 1, ChoiceTerm("P"), "c0"),
            Equation.of({"x": 1, "y": -1}, 2, "e1"),
            Congruent.of({"y": 3}, 0, 4, "c2"),
        ],
        {"P": (5, 7)},
    )
    matrix, rhs = reduce_case(system.instantiate({"P": 7}))
    assert matrix.to_lists() == [[1, 0, -7, 0], [1, -1, 0, 0], [0, 3, 0, -4]]
    assert rhs == (1, 2, 0)
    empty, no_rhs = reduce_case(ConstraintSystem.build({"x": (0, 1)}).instantiate({}))
    assert empty.shape == (0, 1)
    assert no_rhs == ()


# -- the bundled examples -------------------------------------------------------------------


def test_a_composite_modulus_leaves_the_ledger_with_two_worlds() -> None:
    proof = prove(LEDGER)
    assert isinstance(proof, Ambiguity)
    assert (dict(proof.first), dict(proof.second)) == ({"a": 5, "b": 7}, {"a": 11, "b": 1})
    assert (proof.count, proof.exact, proof.size) == (2, True, "exactly 2")
    assert proof.differing == ("a", "b")
    assert proof.verdict == "ambiguous"
    assert str(proof) == (
        "ambiguous: exactly 2 worlds fit, for example a=5, b=7 and a=11, b=1 (they differ in a, b)"
    )
    assert proof.check().valid


def test_a_third_anchor_pins_the_ledger_down() -> None:
    proof = prove(LEDGER.extend(Congruent.of({"a": 2, "b": 1}, 5, 12, "anchor x=2")))
    assert isinstance(proof, UniqueProof)
    assert dict(proof.solution) == {"a": 5, "b": 7}
    assert str(proof) == "unique: a=5, b=7"
    check = proof.check()
    assert (check.valid, check.verdict, check.count, check.exact) == (True, "unique", 1, True)
    assert check.reason == "verified: unique, 1 solution"


def test_the_wrapping_clock_has_three_worlds_per_period() -> None:
    proof = prove(CLOCK)
    assert isinstance(proof, Ambiguity)
    assert (proof.count, proof.exact) == (6, True)
    assert dict(proof.first) == {"P": 12, "T": 5, "w": 0}
    assert dict(proof.second) == {"P": 12, "T": 41, "w": 3}
    assert proof.differing == ("T", "w")
    worlds = brute_force(CLOCK)
    assert [(w["P"], w["T"], w["w"]) for w in worlds] == [
        (12, 5, 0),
        (12, 41, 3),
        (12, 77, 6),
        (18, 5, 0),
        (18, 41, 2),
        (18, 77, 4),
    ]


def test_pinning_the_time_still_leaves_the_period_open() -> None:
    # A second device fixes T, but T = 41 fits both periods with different wrap counts.
    timed = CLOCK.extend(Congruent.of({"T": 1}, 41, 60, "second device"))
    proof = prove(timed)
    assert isinstance(proof, Ambiguity)
    assert (dict(proof.first), dict(proof.second)) == (
        {"P": 12, "T": 41, "w": 3},
        {"P": 18, "T": 41, "w": 2},
    )
    assert proof.differing == ("P", "w")
    # The parity of the wrap count settles the period.
    settled = prove(timed.extend(Congruent.of({"w": 1}, 0, 2, "wrap parity")))
    assert isinstance(settled, UniqueProof)
    assert dict(settled.solution) == {"P": 18, "T": 41, "w": 2}
    assert settled.check().reason == "verified: unique, 1 solution"


# -- verdicts in general --------------------------------------------------------------------


def test_the_count_is_capped_for_a_huge_ambiguity_space() -> None:
    wide = ConstraintSystem.build({"x": (0, 10**12), "y": (0, 10**12)})
    proof = prove(wide, cap=5)
    assert isinstance(proof, Ambiguity)
    assert (proof.count, proof.exact, proof.size) == (5, False, "more than 5")
    assert (dict(proof.first), dict(proof.second)) == ({"x": 0, "y": 0}, {"x": 0, "y": 1})
    check = proof.check()
    assert (check.valid, check.count, check.exact) == (True, 5, False)
    assert check.reason == "verified: ambiguous, more than 5"


def test_counting_stops_across_cases_once_the_cap_is_passed() -> None:
    system = ConstraintSystem.build({"x": (0, 9)}, choices={"P": (1, 2, 3)})
    proof = prove(system, cap=12)
    assert isinstance(proof, Ambiguity)
    assert (proof.count, proof.exact) == (12, False)
    assert proof.certificate["cases"][2]["evidence"] == "lattice"
    assert proof.check().valid


@pytest.mark.parametrize("cap", [1, 0, -3, True, 2.5])
def test_the_cap_must_be_an_int_of_at_least_two(cap: object) -> None:
    with pytest.raises(ValueError, match="cap must be an int of at least 2"):
        prove(LEDGER, cap=cap)  # type: ignore[arg-type]


def test_an_unsolvable_system_is_infeasible_with_a_parity_certificate() -> None:
    system = ConstraintSystem.build({"x": (0, 9), "y": (0, 9)}, [Equation.of({"x": 2, "y": 4}, 7)])
    proof = prove(system)
    assert isinstance(proof, Infeasible)
    assert str(proof) == "infeasible: no world fits every constraint and bound"
    (case,) = proof.certificate["cases"]
    assert (case["evidence"], case["modulus"]) == ("unsolvable", 2)
    assert proof.check().reason == "verified: infeasible, 0 solutions"


def test_solutions_outside_the_box_leave_the_system_infeasible() -> None:
    system = ConstraintSystem.build({"x": (0, 9)}, [Congruent.of({"x": 1}, 10, 20)])
    proof = prove(system)
    assert isinstance(proof, Infeasible)
    assert proof.certificate["cases"][0]["evidence"] == "lattice"
    assert proof.check().valid


def test_a_case_split_mixes_unsolvable_and_lattice_cases() -> None:
    # x = 1 (mod P) and x = 0 (mod 2): impossible for P = 4, and x = 4 for P = 3.
    system = ConstraintSystem.build(
        {"x": (0, 5)},
        [Congruent.of({"x": 1}, 1, ChoiceTerm("P")), Congruent.of({"x": 1}, 0, 2)],
        {"P": (4, 3)},
    )
    proof = prove(system)
    assert isinstance(proof, UniqueProof)
    assert dict(proof.solution) == {"P": 3, "x": 4}
    kinds = [case["evidence"] for case in proof.certificate["cases"]]
    assert kinds == ["unsolvable", "lattice"]
    assert proof.check().valid


def test_a_system_without_constraints_counts_its_box() -> None:
    proof = prove(ConstraintSystem.build({"x": (2, 4), "y": (7, 7)}))
    assert isinstance(proof, Ambiguity)
    assert (proof.count, proof.exact) == (3, True)
    assert proof.differing == ("x",)
    assert proof.check().valid


def test_a_kernel_of_dimension_two_is_enumerated_exactly() -> None:
    # One checksum over three digits: 1*d0 + 1*d1 + 1*d2 = 13 has 75 solutions in 0..9.
    system = ConstraintSystem.build(
        {"d0": (0, 9), "d1": (0, 9), "d2": (0, 9)}, [Equation.of({"d0": 1, "d1": 1, "d2": 1}, 13)]
    )
    proof = prove(system)
    assert isinstance(proof, Ambiguity)
    assert (proof.count, proof.exact) == (75, True)
    assert len(brute_force(system)) == 75
    assert len(proof.certificate["cases"][0]["basis"]) == 2


def test_huge_coprime_moduli_agree_with_crt() -> None:
    m1, m2 = 10**18 + 9, 10**18 + 7
    assert math.gcd(m1, m2) == 1
    hidden = 123_456_789_012_345_678_901_234_567
    system = ConstraintSystem.build(
        {"x": (0, m1 * m2 - 1)},
        [Congruent.of({"x": 1}, hidden % m1, m1), Congruent.of({"x": 1}, hidden % m2, m2)],
    )
    proof = prove(system)
    assert isinstance(proof, UniqueProof)
    combined = crt([Congruence(hidden % m1, m1), Congruence(hidden % m2, m2)])
    assert isinstance(combined, Congruence)
    assert proof.solution["x"] == combined.residue == hidden
    assert proof.check().valid


@given(small_systems())
def test_the_solver_agrees_with_brute_force(system: ConstraintSystem) -> None:
    truth = brute_force(system)
    proof = prove(system, cap=10_000)
    expected = "infeasible" if not truth else "unique" if len(truth) == 1 else "ambiguous"
    assert proof.verdict == expected
    assert proof.certificate["count"] == len(truth)
    assert proof.certificate["exact"] is True
    assert proof.certificate["witnesses"] == truth[:2]
    if isinstance(proof, UniqueProof):
        assert dict(proof.solution) == truth[0]
    if isinstance(proof, Ambiguity):
        assert [dict(proof.first), dict(proof.second)] == truth[:2]
        assert proof.count == len(truth)
    check = check_certificate(proof.certificate)
    assert (check.valid, check.verdict, check.count) == (True, expected, len(truth)), check.reason


@given(small_systems(), st.integers(2, 4))
def test_a_small_cap_gives_a_lower_bound_the_checker_accepts(
    system: ConstraintSystem, cap: int
) -> None:
    truth = brute_force(system)
    proof = prove(system, cap=cap)
    assert proof.certificate["count"] == min(len(truth), cap)
    assert proof.certificate["exact"] is (len(truth) <= cap)
    assert proof.check().valid


# -- certificate text -----------------------------------------------------------------------


def test_certificate_text_is_canonical_ascii_json() -> None:
    proof = prove(CLOCK)
    text = proof.certificate_json()
    assert text == certificate_json(json.loads(text)) == prove(CLOCK).certificate_json()
    assert text.isascii()
    assert text.endswith("}\n")
    assert json.loads(text) == proof.certificate
    assert '"point": [5, 0, 0]' in text
    assert check_certificate(text).reason == "verified: ambiguous, 6 solutions"


def test_the_certificate_embeds_the_system_it_talks_about() -> None:
    proof = prove(CLOCK)
    assert ConstraintSystem.from_json_dict(proof.certificate["system"]) == CLOCK
    assert proof.system is CLOCK


# -- wide boxes -----------------------------------------------------------------------------


def test_equations_over_wide_boxes_cost_the_solutions_not_the_width(
    time_limit: Callable[[float], AbstractContextManager[None]],
) -> None:
    # Regression: the walk used to try every value of the first unknown in its box, so
    # these took minutes (x + y = 10 at 10**9) or hours (three digits at 10**5).
    with time_limit(10):
        pair = ConstraintSystem.build(
            {"x": (0, 10**9), "y": (0, 10**9)}, [Equation.of({"x": 1, "y": 1}, 10, "total")]
        )
        proof = prove(pair)
        assert isinstance(proof, Ambiguity)
        assert (proof.count, proof.exact) == (11, True)
        assert check_certificate(proof.certificate).valid
        digits = ConstraintSystem.build(
            dict.fromkeys(("d0", "d1", "d2"), (0, 10**5)),
            [Equation.of({"d0": 1, "d1": 1, "d2": 1}, 13)],
        )
        proof = prove(digits)
        assert isinstance(proof, Ambiguity)
        assert (proof.count, proof.exact) == (105, True)
        assert check_certificate(proof.certificate).valid
        # A wrapping clock: T = P * w + 5 with a timestamp-sized T and a small wrap count.
        clock = ConstraintSystem.build(
            {"T": (0, 10**15), "w": (0, 9)},
            [Equation.of({"T": 1, "w": ChoiceTerm("P", -1)}, 5)],
            {"P": (7, 11)},
        )
        proof = prove(clock)
        assert isinstance(proof, Ambiguity)
        assert (proof.count, proof.exact) == (20, True)
        assert check_certificate(proof.certificate).valid


def test_a_product_of_two_choices_is_proved_in_case_order() -> None:
    system = ConstraintSystem.build(
        {"x": (ChoiceTerm("Q", 1, -1), 6)},
        [Congruent.of({"x": 1}, 1, ChoiceTerm("P")), Congruent.of({"x": 1}, 0, ChoiceTerm("Q"))],
        {"P": (2, 3), "Q": (1, 2, 4)},
    )
    proof = prove(system, cap=50)
    worlds = brute_force(system)
    assert [case["choices"] for case in proof.certificate["cases"]] == [
        {"P": p, "Q": q} for p in (2, 3) for q in (1, 2, 4)
    ]
    assert isinstance(proof, Ambiguity)
    assert proof.count == len(worlds)
    assert list(proof.certificate["witnesses"]) == worlds[:2]
    assert check_certificate(proof.certificate).valid
