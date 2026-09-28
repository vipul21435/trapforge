"""The uniqueness gate: grow a generated sample until it pins down exactly one hidden world.

A task generator plants a hidden world, reveals a visible sample of it and states what the
sample implies as a :class:`~trapforge.prover.model.ConstraintSystem`. If that system admits
a second world, the task has two defensible answers and must not ship. :func:`gate` proves
the sample and, while it is still ambiguous, appends the next constraint from a stream of
further observations (the next records the generator is willing to reveal) and proves again.
It stops at the shortest prefix of the stream that makes the proof unique, or reports a
rejection when the stream runs out, so the generator can extend the sample or reseed.

A sample that contradicts itself, or a planted world that breaks an observation, is a bug
in the generator rather than a hard task, so both raise :class:`GateError`.

>>> from trapforge.prover import ConstraintSystem, Congruent
>>> def anchor(x: int, y: int) -> Congruent:
...     return Congruent.of({"a": x, "b": 1}, y, 12, f"anchor x={x}")
>>> sample = ConstraintSystem.build({"a": (1, 11), "b": (0, 11)}, [anchor(1, 0), anchor(3, 10)])
>>> result = gate(sample, [anchor(7, 6), anchor(2, 5), anchor(4, 3)], hidden={"a": 5, "b": 7})
>>> print(result)
passed after 2 more constraints (sizes 2, 2, 1): unique: a=5, b=7
>>> [constraint.label for constraint in result.added]
['anchor x=7', 'anchor x=2']
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from trapforge.prover.model import Constraint, ConstraintSystem
from trapforge.prover.solver import DEFAULT_CAP, Ambiguity, Infeasible, Proof, UniqueProof, prove

__all__ = [
    "GateError",
    "GateResult",
    "gate",
]


class GateError(ValueError):
    """The sample contradicts itself or the planted world: a generator bug, not a hard task."""


@dataclass(frozen=True, slots=True)
class GateResult:
    """Where the gate stopped.

    ``system`` is the starting sample plus ``added`` (a prefix of the candidate stream) and
    ``proof`` is its proof. ``sizes`` holds the ambiguity-space size after each step, starting
    with the sample alone: the number of worlds, or ``cap + 1`` standing for "more than cap".
    """

    proof: Proof
    system: ConstraintSystem
    added: tuple[Constraint, ...]
    sizes: tuple[int, ...]

    @property
    def passed(self) -> bool:
        """True when the final sample has exactly one solution."""
        return isinstance(self.proof, UniqueProof)

    def __str__(self) -> str:
        status = "passed" if self.passed else "rejected"
        count = len(self.added)
        steps = f"{count} more constraint{'' if count == 1 else 's'}"
        sizes = ", ".join(str(size) for size in self.sizes)
        return f"{status} after {steps} (sizes {sizes}): {self.proof}"


def _size(proof: Proof) -> int:
    if isinstance(proof, Ambiguity):
        return proof.count if proof.exact else proof.count + 1
    return 1


def _check_planted(system: ConstraintSystem, hidden: Mapping[str, int] | None) -> None:
    if hidden is None:
        return
    broken = system.violations(hidden)
    if broken:
        raise GateError(f"the planted world violates {', '.join(broken)}")


def gate(
    system: ConstraintSystem,
    candidates: Iterable[Constraint] = (),
    *,
    hidden: Mapping[str, int] | None = None,
    max_added: int | None = None,
    cap: int = DEFAULT_CAP,
) -> GateResult:
    """Append constraints from ``candidates`` until ``system`` has exactly one solution.

    ``candidates`` is consumed lazily, one constraint per step, and only as far as needed;
    ``max_added`` bounds how many may be appended. With ``hidden`` (the planted world, every
    choice and unknown), the sample and each candidate are checked against it before they
    are used, so a passing gate proves that the planted world is the only one.

    Raises :class:`GateError` when the planted world breaks a constraint or the sample
    becomes infeasible.
    """
    if max_added is not None and max_added < 0:
        raise ValueError(f"max_added must be at least 0, got {max_added}")
    _check_planted(system, hidden)
    added: list[Constraint] = []
    proof = prove(system, cap=cap)
    sizes = [_size(proof)]
    stream = iter(candidates)
    while isinstance(proof, Ambiguity) and (max_added is None or len(added) < max_added):
        constraint = next(stream, None)
        if constraint is None:
            break
        system = system.extend(constraint)
        _check_planted(system, hidden)
        added.append(constraint)
        proof = prove(system, cap=cap)
        sizes.append(_size(proof))
    if isinstance(proof, Infeasible):
        where = system.describe(len(system.constraints) - 1) if system.constraints else "bounds"
        raise GateError(f"the sample is infeasible (last constraint: {where}); {proof}")
    return GateResult(proof, system, tuple(added), tuple(sizes))
