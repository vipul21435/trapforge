"""The constraint model the uniqueness prover reasons about.

A task family states what an observed sample says about its hidden parameters as a
:class:`ConstraintSystem`:

* **bounded integer unknowns** (every unknown has a lower and an upper bound, so the set of
  candidate parameterizations is finite),
* **linear equations** over the integers, ``c_1*x_1 + ... + c_k*x_k = r``,
* **linear congruences**, ``c_1*x_1 + ... + c_k*x_k = r (mod m)``,
* optional **discrete choices**, such as a counter period drawn from a finite candidate set.

Coefficients, right-hand sides, moduli and bounds may depend on a choice through a
:class:`ChoiceTerm` (``scale * choice + offset``), which is what makes a period usable as a
modulus or as the coefficient of a wrap count. Fixing every choice turns the system into a
purely linear one, a *case*; the case split is the finite product of the choice options, and
the solver handles each case exactly.

Systems are immutable, validated on construction, and serialize to a plain JSON structure
(:meth:`ConstraintSystem.to_json_dict`) so that a certificate can carry the exact system it
talks about.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Choice",
    "ChoiceTerm",
    "Congruent",
    "Constraint",
    "ConstraintSystem",
    "Equation",
    "Instance",
    "InstanceRow",
    "ModelError",
    "Scalar",
    "Unknown",
    "evaluate",
]


class ModelError(ValueError):
    """Raised for a malformed constraint system or assignment."""


@dataclass(frozen=True, slots=True)
class ChoiceTerm:
    """The integer ``scale * <choice> + offset``, usable wherever a plain int is.

    >>> period = ChoiceTerm("P")
    >>> str(period), str(ChoiceTerm("P", -1)), str(ChoiceTerm("P", 1, -1))
    ('P', '-P', 'P - 1')
    >>> ChoiceTerm("P", 2, 3).evaluate({"P": 10})
    23
    """

    choice: str
    scale: int = 1
    offset: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.choice, str) or not self.choice:
            raise ModelError("a choice term must name a choice")
        if type(self.scale) is not int or type(self.offset) is not int:
            raise ModelError(f"choice term on {self.choice!r} needs int scale and offset")

    def evaluate(self, choices: Mapping[str, int]) -> int:
        """The value of this term once the choice is fixed."""
        return self.scale * choices[self.choice] + self.offset

    def __str__(self) -> str:
        if self.scale == 1:
            head = self.choice
        elif self.scale == -1:
            head = f"-{self.choice}"
        else:
            head = f"{self.scale}*{self.choice}"
        if self.offset > 0:
            return f"{head} + {self.offset}"
        if self.offset < 0:
            return f"{head} - {-self.offset}"
        return head


Scalar = int | ChoiceTerm
"""A plain integer or a :class:`ChoiceTerm`."""


def evaluate(value: Scalar, choices: Mapping[str, int]) -> int:
    """The integer value of ``value`` under a full assignment of the choices."""
    return value if isinstance(value, int) else value.evaluate(choices)


def _is_scalar(value: object) -> bool:
    return type(value) is int or isinstance(value, ChoiceTerm)


@dataclass(frozen=True, slots=True)
class Unknown:
    """A hidden integer parameter with ``lower <= value <= upper``."""

    name: str
    lower: Scalar
    upper: Scalar

    def __str__(self) -> str:
        return f"{self.lower} <= {self.name} <= {self.upper}"


@dataclass(frozen=True, slots=True)
class Choice:
    """A discrete hidden parameter that takes one of finitely many ``options``."""

    name: str
    options: tuple[int, ...]

    def __str__(self) -> str:
        return f"{self.name} in {{{', '.join(str(option) for option in self.options)}}}"


Terms = tuple[tuple[str, Scalar], ...]


def _freeze_terms(terms: Mapping[str, Scalar]) -> Terms:
    return tuple(terms.items())


def _format_coefficient(coefficient: Scalar, name: str) -> str:
    if isinstance(coefficient, int):
        if coefficient == 1:
            return name
        if coefficient == -1:
            return f"-{name}"
        return f"{coefficient}*{name}"
    if coefficient.offset:
        return f"({coefficient})*{name}"
    return f"{coefficient}*{name}"


def _format_form(terms: Terms) -> str:
    if not terms:
        return "0"
    text = " + ".join(_format_coefficient(c, name) for name, c in terms)
    return text.replace(" + -", " - ")


@dataclass(frozen=True, slots=True)
class Equation:
    """The constraint ``sum(coefficient * unknown) == rhs`` over the integers.

    >>> print(Equation.of({"a": 3, "b": -1}, 7))
    3*a - b = 7
    """

    terms: Terms
    rhs: Scalar
    label: str = ""

    @classmethod
    def of(cls, terms: Mapping[str, Scalar], rhs: Scalar, label: str = "") -> Equation:
        """Build an equation from a ``{unknown: coefficient}`` mapping."""
        return cls(_freeze_terms(terms), rhs, label)

    def __str__(self) -> str:
        return f"{_format_form(self.terms)} = {self.rhs}"


@dataclass(frozen=True, slots=True)
class Congruent:
    """The constraint ``sum(coefficient * unknown) == rhs (mod modulus)``.

    >>> print(Congruent.of({"T": 1, "w": ChoiceTerm("P", -1)}, 5, 12))
    T - P*w = 5 (mod 12)
    """

    terms: Terms
    rhs: Scalar
    modulus: Scalar
    label: str = ""

    @classmethod
    def of(
        cls, terms: Mapping[str, Scalar], rhs: Scalar, modulus: Scalar, label: str = ""
    ) -> Congruent:
        """Build a congruence from a ``{unknown: coefficient}`` mapping."""
        return cls(_freeze_terms(terms), rhs, modulus, label)

    def __str__(self) -> str:
        return f"{_format_form(self.terms)} = {self.rhs} (mod {self.modulus})"


Constraint = Equation | Congruent
"""Either kind of linear constraint."""


@dataclass(frozen=True, slots=True)
class InstanceRow:
    """One constraint of an :class:`Instance`: ``coefficients . x = rhs``, exactly or mod m.

    ``modulus`` is None for an equation.
    """

    coefficients: tuple[int, ...]
    rhs: int
    modulus: int | None

    def holds(self, values: tuple[int, ...]) -> bool:
        """True when the integer vector ``values`` satisfies this row."""
        lhs = sum(c * v for c, v in zip(self.coefficients, values, strict=True))
        if self.modulus is None:
            return lhs == self.rhs
        return (lhs - self.rhs) % self.modulus == 0


@dataclass(frozen=True, slots=True)
class Instance:
    """One case of a system: every choice fixed, every number a plain int.

    Coordinates follow the declaration order of the unknowns (``names``).
    """

    choices: tuple[tuple[str, int], ...]
    names: tuple[str, ...]
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    rows: tuple[InstanceRow, ...]

    def contains(self, values: tuple[int, ...]) -> bool:
        """True when ``values`` lies in the box and satisfies every row."""
        in_box = all(
            lo <= v <= hi for lo, v, hi in zip(self.lower, values, self.upper, strict=True)
        )
        return in_box and all(row.holds(values) for row in self.rows)


def _json_scalar(value: Scalar) -> Any:
    if isinstance(value, int):
        return value
    return {"choice": value.choice, "scale": value.scale, "offset": value.offset}


def _parse_scalar(value: Any) -> Scalar:
    if type(value) is int:
        return value
    if isinstance(value, Mapping) and set(value) == {"choice", "scale", "offset"}:
        return ChoiceTerm(value["choice"], value["scale"], value["offset"])
    raise ModelError(f"not an integer or a choice term: {value!r}")


def _parse_terms(raw: Any) -> Terms:
    if not isinstance(raw, list):
        raise ModelError(f"terms must be a list of [name, coefficient] pairs, got {raw!r}")
    terms: list[tuple[str, Scalar]] = []
    for pair in raw:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ModelError(f"a term must be a [name, coefficient] pair, got {pair!r}")
        terms.append((pair[0], _parse_scalar(pair[1])))
    return tuple(terms)


@dataclass(frozen=True, slots=True)
class ConstraintSystem:
    """Bounded integer unknowns, linear constraints and a finite case split.

    Build one with :meth:`ConstraintSystem.build`; the constructor validates names,
    references and moduli (every modulus must be at least 1 under every choice).

    >>> system = ConstraintSystem.build(
    ...     {"a": (1, 11), "b": (0, 11)},
    ...     [Congruent.of({"a": 1, "b": 1}, 0, 12, "anchor x=1"),
    ...      Congruent.of({"a": 3, "b": 1}, 10, 12, "anchor x=3")],
    ... )
    >>> print(system)
    unknowns: 1 <= a <= 11, 0 <= b <= 11
    anchor x=1: a + b = 0 (mod 12)
    anchor x=3: 3*a + b = 10 (mod 12)
    >>> system.violations({"a": 5, "b": 7}), system.violations({"a": 5, "b": 8})
    ((), ('anchor x=1', 'anchor x=3'))
    """

    unknowns: tuple[Unknown, ...]
    constraints: tuple[Constraint, ...] = ()
    choices: tuple[Choice, ...] = ()

    def __post_init__(self) -> None:
        self._check_names()
        for choice in self.choices:
            options = choice.options
            if not options or not all(type(option) is int for option in options):
                raise ModelError(f"choice {choice.name!r} needs a non-empty tuple of ints")
            if len(set(options)) != len(options):
                raise ModelError(f"choice {choice.name!r} lists an option twice")
        for unknown in self.unknowns:
            self._check_scalar(unknown.lower, f"lower bound of {unknown.name!r}")
            self._check_scalar(unknown.upper, f"upper bound of {unknown.name!r}")
        for index, constraint in enumerate(self.constraints):
            self._check_constraint(index, constraint)

    def _check_names(self) -> None:
        names = [choice.name for choice in self.choices] + [u.name for u in self.unknowns]
        for name in names:
            if not isinstance(name, str) or not name:
                raise ModelError(f"names must be non-empty strings, got {name!r}")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ModelError(f"duplicate names: {', '.join(duplicates)}")

    def _check_constraint(self, index: int, constraint: object) -> None:
        where = f"constraint #{index}"
        if not isinstance(constraint, Equation | Congruent):
            raise ModelError(f"{where} is not an Equation or a Congruent")
        if not isinstance(constraint.label, str):
            raise ModelError(f"{where} has a non-string label")
        where = self.describe(index)
        seen: set[str] = set()
        for name, coefficient in constraint.terms:
            if name not in self.names:
                raise ModelError(f"{where} uses the undeclared unknown {name!r}")
            if name in seen:
                raise ModelError(f"{where} lists the unknown {name!r} twice")
            seen.add(name)
            self._check_scalar(coefficient, f"coefficient of {name!r} in {where}")
        self._check_scalar(constraint.rhs, f"right-hand side of {where}")
        if isinstance(constraint, Congruent):
            self._check_modulus(constraint.modulus, where)

    def _check_scalar(self, value: object, where: str) -> None:
        if not _is_scalar(value):
            raise ModelError(f"{where} must be an int or a ChoiceTerm, got {value!r}")
        if isinstance(value, ChoiceTerm) and value.choice not in self.choice_names:
            raise ModelError(f"{where} refers to the undeclared choice {value.choice!r}")

    def _check_modulus(self, modulus: Scalar, where: str) -> None:
        self._check_scalar(modulus, f"modulus of {where}")
        if isinstance(modulus, int):
            values = [modulus]
        else:
            options = next(c.options for c in self.choices if c.name == modulus.choice)
            values = [modulus.evaluate({modulus.choice: option}) for option in options]
        if min(values) < 1:
            raise ModelError(f"the modulus of {where} must be at least 1 in every case")

    # -- construction --------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        unknowns: Mapping[str, tuple[Scalar, Scalar]],
        constraints: Iterable[Constraint] = (),
        choices: Mapping[str, Iterable[int]] | None = None,
    ) -> ConstraintSystem:
        """Build a system from ``{name: (lower, upper)}`` and ``{choice: options}`` maps."""
        return cls(
            tuple(Unknown(name, lower, upper) for name, (lower, upper) in unknowns.items()),
            tuple(constraints),
            tuple(Choice(name, tuple(options)) for name, options in (choices or {}).items()),
        )

    def extend(self, *constraints: Constraint) -> ConstraintSystem:
        """The same system with more constraints appended (a larger sample)."""
        return ConstraintSystem(self.unknowns, self.constraints + constraints, self.choices)

    # -- structure ------------------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """The unknown names, in declaration order."""
        return tuple(u.name for u in self.unknowns)

    @property
    def choice_names(self) -> tuple[str, ...]:
        """The choice names, in declaration order."""
        return tuple(c.name for c in self.choices)

    @property
    def case_count(self) -> int:
        """Number of cases: the product of the option counts (1 without choices)."""
        return math.prod(len(c.options) for c in self.choices)

    def cases(self) -> Iterator[dict[str, int]]:
        """Every assignment of the choices, in lexicographic order of option positions."""
        for values in itertools.product(*(c.options for c in self.choices)):
            yield dict(zip(self.choice_names, values, strict=True))

    def describe(self, index: int) -> str:
        """The label of constraint ``index``, or ``#index`` when it has none."""
        return self.constraints[index].label or f"#{index}"

    def instantiate(self, choices: Mapping[str, int]) -> Instance:
        """The case obtained by fixing every choice; all numbers become plain ints."""
        if set(choices) != set(self.choice_names):
            raise ModelError(f"expected values for exactly the choices {self.choice_names}")
        for choice in self.choices:
            if choices[choice.name] not in choice.options:
                raise ModelError(f"{choices[choice.name]} is not an option of {choice.name!r}")
        position = {name: index for index, name in enumerate(self.names)}
        rows = []
        for constraint in self.constraints:
            coefficients = [0] * len(position)
            for name, coefficient in constraint.terms:
                coefficients[position[name]] = evaluate(coefficient, choices)
            modulus = (
                evaluate(constraint.modulus, choices) if isinstance(constraint, Congruent) else None
            )
            rows.append(
                InstanceRow(tuple(coefficients), evaluate(constraint.rhs, choices), modulus)
            )
        return Instance(
            choices=tuple((name, choices[name]) for name in self.choice_names),
            names=self.names,
            lower=tuple(evaluate(u.lower, choices) for u in self.unknowns),
            upper=tuple(evaluate(u.upper, choices) for u in self.unknowns),
            rows=tuple(rows),
        )

    def violations(self, assignment: Mapping[str, int]) -> tuple[str, ...]:
        """What ``assignment`` (every choice and unknown) violates; empty when it fits.

        Constraints are named by their label (or ``#index``), bounds as ``bounds of x`` and
        choices as ``options of P``.
        """
        expected = set(self.choice_names) | set(self.names)
        if set(assignment) != expected:
            raise ModelError(f"an assignment must give exactly {sorted(expected)}")
        if not all(type(value) is int for value in assignment.values()):
            raise ModelError("assignment values must be ints")
        broken = [
            f"options of {c.name}" for c in self.choices if assignment[c.name] not in c.options
        ]
        if broken:
            return tuple(broken)
        choices = {name: assignment[name] for name in self.choice_names}
        instance = self.instantiate(choices)
        values = tuple(assignment[name] for name in self.names)
        for name, lo, value, hi in zip(
            self.names, instance.lower, values, instance.upper, strict=True
        ):
            if not lo <= value <= hi:
                broken.append(f"bounds of {name}")
        broken += [
            self.describe(index) for index, row in enumerate(instance.rows) if not row.holds(values)
        ]
        return tuple(broken)

    # -- serialization --------------------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        """A JSON-compatible description of the system (lists, dicts, str and int only)."""
        constraints: list[dict[str, Any]] = []
        for constraint in self.constraints:
            entry: dict[str, Any] = {
                "kind": "equation" if isinstance(constraint, Equation) else "congruence",
                "label": constraint.label,
                "terms": [[name, _json_scalar(c)] for name, c in constraint.terms],
                "rhs": _json_scalar(constraint.rhs),
            }
            if isinstance(constraint, Congruent):
                entry["modulus"] = _json_scalar(constraint.modulus)
            constraints.append(entry)
        return {
            "choices": [{"name": c.name, "options": list(c.options)} for c in self.choices],
            "unknowns": [
                {"name": u.name, "lower": _json_scalar(u.lower), "upper": _json_scalar(u.upper)}
                for u in self.unknowns
            ],
            "constraints": constraints,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> ConstraintSystem:
        """Rebuild a system from :meth:`to_json_dict` output, validating everything."""
        try:
            choices = tuple(Choice(c["name"], tuple(c["options"])) for c in data["choices"])
            unknowns = tuple(
                Unknown(u["name"], _parse_scalar(u["lower"]), _parse_scalar(u["upper"]))
                for u in data["unknowns"]
            )
            constraints: list[Constraint] = []
            for entry in data["constraints"]:
                terms = _parse_terms(entry["terms"])
                rhs = _parse_scalar(entry["rhs"])
                label = entry["label"]
                if entry["kind"] == "equation":
                    constraints.append(Equation(terms, rhs, label))
                elif entry["kind"] == "congruence":
                    constraints.append(
                        Congruent(terms, rhs, _parse_scalar(entry["modulus"]), label)
                    )
                else:
                    raise ModelError(f"unknown constraint kind {entry['kind']!r}")
        except (KeyError, TypeError) as error:
            raise ModelError(f"malformed constraint system: {error!r}") from error
        return cls(unknowns, tuple(constraints), choices)

    def __str__(self) -> str:
        lines = []
        if self.choices:
            lines.append("choices: " + ", ".join(str(c) for c in self.choices))
        lines.append("unknowns: " + ", ".join(str(u) for u in self.unknowns))
        lines += [f"{self.describe(i)}: {c}" for i, c in enumerate(self.constraints)]
        return "\n".join(lines)
