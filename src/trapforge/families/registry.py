"""The family registry: look task families up by name.

>>> registry = Registry()
>>> registry.names()
()
>>> registry.get("nope")
Traceback (most recent call last):
...
trapforge.families.base.FamilyError: unknown family 'nope'; registered: (none)
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from trapforge.families.base import Difficulty, FamilyError, TaskFamily, TaskInstance

__all__ = [
    "Registry",
]

_NAME = re.compile(r"[a-z][a-z0-9]*(-[a-z0-9]+)*")


class Registry:
    """A name-to-family mapping that validates plugins as they are registered."""

    def __init__(self) -> None:
        self._families: dict[str, TaskFamily] = {}

    def register(self, family: TaskFamily) -> TaskFamily:
        """Add ``family``; its name must be kebab-case and not taken yet."""
        if not isinstance(family, TaskFamily):
            raise FamilyError(f"{family!r} does not implement the TaskFamily protocol")
        name = family.name
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise FamilyError(f"family names must be lowercase kebab-case, got {name!r}")
        if name in self._families:
            raise FamilyError(f"a family named {name!r} is already registered")
        self._families[name] = family
        return family

    def get(self, name: str) -> TaskFamily:
        """The family registered as ``name``."""
        try:
            return self._families[name]
        except KeyError:
            known = ", ".join(self.names()) or "(none)"
            raise FamilyError(f"unknown family {name!r}; registered: {known}") from None

    def names(self) -> tuple[str, ...]:
        """Every registered name, sorted."""
        return tuple(sorted(self._families))

    def generate(self, name: str, seed: int, difficulty: Difficulty) -> TaskInstance:
        """Generate one instance and check that the family labelled it correctly."""
        instance = self.get(name).generate(seed, difficulty)
        if (instance.family, instance.seed, instance.difficulty) != (name, seed, difficulty):
            raise FamilyError(
                f"family {name!r} labelled its instance as "
                f"{instance.family}/{instance.difficulty}/{instance.seed}"
            )
        return instance

    def __contains__(self, name: object) -> bool:
        return name in self._families

    def __iter__(self) -> Iterator[TaskFamily]:
        return iter(self._families[name] for name in self.names())

    def __len__(self) -> int:
        return len(self._families)
