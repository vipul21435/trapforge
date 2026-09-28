"""The task-family plugin API: difficulties, task instances and the TaskFamily protocol.

A task family is a generator of adversarial data tasks that share one hidden structure. It
plugs into TrapForge by providing:

* ``name`` and ``summary`` strings;
* ``generate(seed, difficulty)``, which plants a hidden world, writes the corpus a solver
  sees, computes the byte-exact expected output and states what the corpus implies about the
  hidden world as a :class:`~trapforge.prover.ConstraintSystem`;
* ``solve(files)``, the reference solver, which recomputes the expected output from the
  corpus alone;
* ``baseline(files)``, a naive solver that is right on the visible sample and wrong on the
  hidden deciding records, which is what makes the task adversarial.

All randomness comes from :func:`family_rng`, a ``random.Random`` seeded from a string, so
a (family, difficulty, seed) triple gives the same bytes in every process and on every
platform. Every file goes through the writers in :mod:`trapforge.canonical`.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from trapforge.canonical import CanonicalError, json_bytes, text_bytes, tree_digest
from trapforge.prover import ConstraintSystem, ModelError

__all__ = [
    "Difficulty",
    "FamilyError",
    "TaskFamily",
    "TaskInstance",
    "family_rng",
]


class FamilyError(ValueError):
    """Raised for an unknown or malformed family, or an instance that breaks the API."""


class Difficulty(StrEnum):
    """How hard a generated instance is; each family maps it to its own knobs.

    >>> Difficulty.parse("Hard"), [str(d) for d in Difficulty]
    (<Difficulty.HARD: 'hard'>, ['easy', 'medium', 'hard'])
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @classmethod
    def parse(cls, text: str) -> Difficulty:
        """The difficulty named ``text`` (case-insensitive)."""
        try:
            return cls(text.strip().lower())
        except ValueError:
            names = ", ".join(d.value for d in cls)
            raise FamilyError(f"unknown difficulty {text!r}; expected one of {names}") from None


def family_rng(family: str, seed: int, difficulty: Difficulty) -> random.Random:
    """The only source of randomness a family may use for one instance.

    Seeding from a string hashes it with SHA-512, so the stream does not depend on
    ``PYTHONHASHSEED`` or the process.

    >>> family_rng("demo", 7, Difficulty.EASY).randrange(10**6)
    976117
    """
    if type(seed) is not int or seed < 0:
        raise FamilyError(f"seed must be a non-negative int, got {seed!r}")
    return random.Random(f"trapforge:{family}:{difficulty.value}:{seed}")


def _frozen_files(files: Mapping[str, bytes], what: str) -> Mapping[str, bytes]:
    if not files:
        raise FamilyError(f"{what} must not be empty")
    for name, content in files.items():
        if not isinstance(content, bytes):
            raise FamilyError(f"{what}[{name!r}] must be bytes")
    try:
        tree_digest(files)  # validates every path
    except CanonicalError as error:
        raise FamilyError(f"{what}: {error}") from error
    return MappingProxyType(dict(sorted(files.items())))


@dataclass(frozen=True, slots=True)
class TaskInstance:
    """One generated task: the corpus, the hidden world, and the byte-exact answer.

    ``files`` is the corpus a solver reads, keyed by relative POSIX path. ``output`` is the
    name of the one file a solver must write and ``expected`` its exact bytes. ``sample`` is
    the part of the expected output the task reveals as a worked example (the visible
    sample); the remaining records are hidden. ``hidden`` is the planted world and ``system``
    what the corpus implies about it; the planted world must satisfy the system.
    """

    family: str
    seed: int
    difficulty: Difficulty
    instruction: str
    files: Mapping[str, bytes]
    output: str
    expected: bytes
    sample: bytes
    hidden: Mapping[str, int]
    system: ConstraintSystem
    extras: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Every way an instance can break the API surfaces as FamilyError, whichever layer
        # (canonical writers, constraint model) noticed it first.
        if not isinstance(self.difficulty, Difficulty):
            raise FamilyError(f"difficulty must be a Difficulty, got {self.difficulty!r}")
        family_rng(self.family, self.seed, self.difficulty)  # validates the seed
        object.__setattr__(self, "files", _frozen_files(self.files, "files"))
        _frozen_files({self.output: self.expected}, "output")  # validates name and bytes
        if not isinstance(self.sample, bytes):
            raise FamilyError("sample must be bytes")
        if not isinstance(self.system, ConstraintSystem):
            raise FamilyError("system must be a ConstraintSystem")
        if not isinstance(self.hidden, Mapping):
            raise FamilyError("hidden must map unknown names to ints")
        try:
            broken = self.system.violations(self.hidden)
        except ModelError as error:
            raise FamilyError(f"hidden: {error}") from error
        if broken:
            raise FamilyError(f"the planted world violates {', '.join(broken)}")
        if not isinstance(self.instruction, str):
            raise FamilyError("instruction must be a str")
        try:
            text_bytes(self.instruction)
        except CanonicalError as error:
            raise FamilyError(f"instruction: {error}") from error
        object.__setattr__(self, "hidden", MappingProxyType(dict(self.hidden)))
        if not isinstance(self.extras, Mapping) or not all(
            isinstance(name, str) and type(value) is int for name, value in self.extras.items()
        ):
            raise FamilyError("extras must map names to ints")
        object.__setattr__(self, "extras", MappingProxyType(dict(sorted(self.extras.items()))))
        try:
            self.bundle()  # the metadata must serialize canonically, too
        except CanonicalError as error:
            raise FamilyError(f"the instance cannot be written canonically: {error}") from error

    def bundle(self) -> dict[str, bytes]:
        """Every byte the instance consists of, as canonical files keyed by path.

        ``data/`` holds the corpus, ``expected/`` and ``sample/`` the full and visible
        answers, and ``meta/`` the instruction, the planted world, the constraint system and
        the family's extra statistics.
        """
        files = {f"data/{name}": content for name, content in self.files.items()}
        files[f"expected/{self.output}"] = self.expected
        files[f"sample/{self.output}"] = self.sample
        files["meta/instruction.md"] = text_bytes(self.instruction)
        files["meta/task.json"] = json_bytes(
            {
                "family": self.family,
                "seed": self.seed,
                "difficulty": self.difficulty.value,
                "output": self.output,
                "hidden": dict(self.hidden),
                "extras": dict(self.extras),
            }
        )
        files["meta/system.json"] = json_bytes(self.system.to_json_dict())
        return files

    def digest(self) -> str:
        """SHA-256 of :meth:`bundle`: equal digests mean byte-identical instances."""
        return tree_digest(self.bundle())


@runtime_checkable
class TaskFamily(Protocol):
    """What a task family must provide to plug into the registry."""

    @property
    def name(self) -> str:
        """A short lowercase identifier such as ``affine-ledger``."""
        ...

    @property
    def summary(self) -> str:
        """One line on the hidden structure and the trap."""
        ...

    def generate(self, seed: int, difficulty: Difficulty) -> TaskInstance:
        """Plant a hidden world and build one instance; the same inputs give the same bytes."""
        ...

    def solve(self, files: Mapping[str, bytes]) -> bytes:
        """The reference solver: the expected output, computed from the corpus alone."""
        ...

    def baseline(self, files: Mapping[str, bytes]) -> bytes:
        """The naive solver: right on the visible sample, wrong on hidden deciding records."""
        ...
