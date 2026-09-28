"""The prover's constraint model: validation, case split, instantiation and JSON."""

import json
from typing import Any

import pytest
from hypothesis import given

from _prover_strategies import brute_force, small_systems
from trapforge.prover import (
    Choice,
    ChoiceTerm,
    Congruent,
    ConstraintSystem,
    Equation,
    InstanceRow,
    ModelError,
    Unknown,
    evaluate,
)

P = ChoiceTerm("P")

# A device counter that wraps at an unknown period P: the true time T is P*w + 5 for some
# wrap count w, and a beacon says T = 1 (mod 4).
CLOCK = ConstraintSystem.build(
    {"T": (0, 40), "w": (0, ChoiceTerm("P", 1, -1))},
    [
        Equation.of({"T": 1, "w": ChoiceTerm("P", -1)}, 5, "stamp"),
        Congruent.of({"T": 1}, 1, 4, "beacon"),
    ],
    {"P": (6, 8)},
)


# -- scalars and formatting ----------------------------------------------------------------


def test_choice_terms_evaluate_and_format() -> None:
    assert evaluate(7, {}) == 7
    assert evaluate(ChoiceTerm("P", -2, 1), {"P": 5}) == -9
    assert str(ChoiceTerm("P", 3)) == "3*P"
    assert str(ChoiceTerm("P", 1, 2)) == "P + 2"


@pytest.mark.parametrize(
    ("args", "message"),
    [(("",), "must name a choice"), (("P", 1.5), "int scale"), (("P", 1, True), "int scale")],
)
def test_choice_terms_reject_bad_fields(args: tuple[Any, ...], message: str) -> None:
    with pytest.raises(ModelError, match=message):
        ChoiceTerm(*args)


def test_constraints_format_like_the_math() -> None:
    assert str(Equation.of({}, 0)) == "0 = 0"
    assert str(Equation.of({"a": -1, "b": 2}, -3)) == "-a + 2*b = -3"
    assert str(Congruent.of({"w": ChoiceTerm("P", 1, -1)}, P, P)) == "(P - 1)*w = P (mod P)"
    assert str(Unknown("w", 0, ChoiceTerm("P", 1, -1))) == "0 <= w <= P - 1"
    assert str(Choice("P", (6, 8))) == "P in {6, 8}"


def test_system_prints_choices_unknowns_and_labelled_constraints() -> None:
    assert str(CLOCK) == (
        "choices: P in {6, 8}\n"
        "unknowns: 0 <= T <= 40, 0 <= w <= P - 1\n"
        "stamp: T - P*w = 5\n"
        "beacon: T = 1 (mod 4)"
    )
    unlabelled = ConstraintSystem.build({"a": (0, 1)}, [Equation.of({"a": 1}, 1)])
    assert str(unlabelled).endswith("#0: a = 1")


# -- structure and cases -------------------------------------------------------------------


def test_cases_are_the_product_of_the_options_in_order() -> None:
    system = ConstraintSystem.build({"a": (0, 1)}, choices={"P": [6, 8], "Q": [1, 2, 3]})
    assert system.case_count == 6
    assert list(system.cases())[:4] == [
        {"P": 6, "Q": 1},
        {"P": 6, "Q": 2},
        {"P": 6, "Q": 3},
        {"P": 8, "Q": 1},
    ]
    assert system.choice_names == ("P", "Q")
    assert system.names == ("a",)


def test_a_system_without_choices_has_one_empty_case() -> None:
    system = ConstraintSystem.build({"a": (0, 1)})
    assert system.case_count == 1
    assert list(system.cases()) == [{}]


def test_instantiate_fixes_every_choice() -> None:
    instance = CLOCK.instantiate({"P": 8})
    assert instance.choices == (("P", 8),)
    assert instance.names == ("T", "w")
    assert (instance.lower, instance.upper) == ((0, 0), (40, 7))
    assert instance.rows == (InstanceRow((1, -8), 5, None), InstanceRow((1, 0), 1, 4))
    assert instance.contains((21, 2))
    assert not instance.contains((13, 2))  # 13 != 8*2 + 5
    assert not instance.contains((69, 8))  # satisfies both rows, but lies outside the box


@pytest.mark.parametrize("choices", [{}, {"P": 7}, {"P": 6, "Q": 1}])
def test_instantiate_rejects_incomplete_or_invalid_choices(choices: dict[str, int]) -> None:
    with pytest.raises(ModelError):
        CLOCK.instantiate(choices)


def test_extend_appends_constraints_without_touching_the_original() -> None:
    extra = Congruent.of({"T": 1}, 0, 3, "second beacon")
    bigger = CLOCK.extend(extra)
    assert bigger.constraints == (*CLOCK.constraints, extra)
    assert len(CLOCK.constraints) == 2


# -- violations ----------------------------------------------------------------------------


def test_violations_name_what_breaks() -> None:
    assert CLOCK.violations({"P": 8, "T": 21, "w": 2}) == ()
    assert CLOCK.violations({"P": 6, "T": 11, "w": 1}) == ("beacon",)
    assert CLOCK.violations({"P": 8, "T": 69, "w": 8}) == ("bounds of T", "bounds of w")
    assert CLOCK.violations({"P": 7, "T": 21, "w": 2}) == ("options of P",)


@pytest.mark.parametrize(
    "assignment",
    [{"T": 21, "w": 2}, {"P": 8, "T": 21, "w": 2, "extra": 0}, {"P": 8, "T": 2.0, "w": 2}],
)
def test_violations_reject_malformed_assignments(assignment: dict[str, Any]) -> None:
    with pytest.raises(ModelError):
        CLOCK.violations(assignment)


@given(small_systems())
def test_violations_agree_with_the_brute_force_solution_set(system: ConstraintSystem) -> None:
    for solution in brute_force(system):
        assert system.violations(solution) == ()


# -- validation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: ConstraintSystem.build({"a": (0, 1)}, choices={"a": [1]}), "duplicate names: a"),
        (lambda: ConstraintSystem.build({"": (0, 1)}), "non-empty strings"),
        (lambda: ConstraintSystem.build({"a": (0, 1)}, choices={"P": []}), "non-empty tuple"),
        (lambda: ConstraintSystem.build({"a": (0, 1)}, choices={"P": [2, 2]}), "twice"),
        (lambda: ConstraintSystem.build({"a": (0, 1)}, choices={"P": [True]}), "tuple of ints"),
        (lambda: ConstraintSystem.build({"a": (0.5, 1)}), "lower bound of 'a'"),
        (lambda: ConstraintSystem.build({"a": (0, P)}), "undeclared choice 'P'"),
        (
            lambda: ConstraintSystem.build({"a": (0, 1)}, [Equation.of({"b": 1}, 0)]),
            "undeclared unknown 'b'",
        ),
        (
            lambda: ConstraintSystem.build({"a": (0, 1)}, [Equation((("a", 1), ("a", 2)), 0)]),
            "lists the unknown 'a' twice",
        ),
        (
            lambda: ConstraintSystem.build({"a": (0, 1)}, [Equation.of({"a": False}, 0, "e")]),
            "coefficient of 'a' in e",
        ),
        (
            lambda: ConstraintSystem.build({"a": (0, 1)}, [Congruent.of({"a": 1}, 0, 0)]),
            "modulus of #0 must be at least 1",
        ),
        (
            lambda: ConstraintSystem.build(
                {"a": (0, 1)},
                [Congruent.of({"a": 1}, 0, ChoiceTerm("P", 1, -2), "c")],
                {"P": [3, 1]},
            ),
            "modulus of c must be at least 1 in every case",
        ),
        (
            lambda: ConstraintSystem((Unknown("a", 0, 1),), ("a = 1",)),  # type: ignore[arg-type]
            "not an Equation or a Congruent",
        ),
        (
            lambda: ConstraintSystem.build({"a": (0, 1)}, [Equation.of({}, 0, 7)]),  # type: ignore[arg-type]
            "non-string label",
        ),
    ],
)
def test_malformed_systems_are_rejected(build: Any, message: str) -> None:
    with pytest.raises(ModelError, match=message):
        build()


# -- JSON ----------------------------------------------------------------------------------


def test_json_form_is_plain_and_readable() -> None:
    data = CLOCK.to_json_dict()
    assert data["choices"] == [{"name": "P", "options": [6, 8]}]
    assert data["unknowns"][1] == {
        "name": "w",
        "lower": 0,
        "upper": {"choice": "P", "scale": 1, "offset": -1},
    }
    assert data["constraints"][1] == {
        "kind": "congruence",
        "label": "beacon",
        "terms": [["T", 1]],
        "rhs": 1,
        "modulus": 4,
    }


@given(small_systems())
def test_json_round_trip_is_exact(system: ConstraintSystem) -> None:
    text = json.dumps(system.to_json_dict(), sort_keys=True)
    assert ConstraintSystem.from_json_dict(json.loads(text)) == system


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.pop("unknowns"), "malformed"),
        (lambda d: d["unknowns"][0].update(lower=1.5), "not an integer or a choice term"),
        (lambda d: d["unknowns"][0].update(lower={"choice": "P"}), "not an integer"),
        (lambda d: d["constraints"][0].update(kind="inequality"), "unknown constraint kind"),
        (lambda d: d["constraints"][0].update(terms={"T": 1}), "list of"),
        (lambda d: d["constraints"][0].update(terms=[["T"]]), "pair"),
        (lambda d: d["choices"][0].update(options=None), "malformed"),
    ],
)
def test_malformed_json_is_rejected(mutate: Any, message: str) -> None:
    data = CLOCK.to_json_dict()
    mutate(data)
    with pytest.raises(ModelError, match=message):
        ConstraintSystem.from_json_dict(data)
