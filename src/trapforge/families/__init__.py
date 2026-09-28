"""Task families: generators of adversarial data tasks with a provably unique answer.

:class:`TaskFamily` is the plugin protocol, :class:`TaskInstance` what a family generates,
and :data:`REGISTRY` the process-wide registry the CLI reads.
"""

from trapforge.families.base import (
    Difficulty,
    FamilyError,
    TaskFamily,
    TaskInstance,
    family_rng,
)
from trapforge.families.ledger import AffineLedger
from trapforge.families.registry import Registry

REGISTRY = Registry()
"""Every built-in family, registered at import time."""

REGISTRY.register(AffineLedger())

__all__ = [
    "REGISTRY",
    "AffineLedger",
    "Difficulty",
    "FamilyError",
    "Registry",
    "TaskFamily",
    "TaskInstance",
    "family_rng",
]
