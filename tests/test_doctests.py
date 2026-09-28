"""Run the usage examples embedded in module docstrings, so they cannot drift."""

import doctest
from types import ModuleType

import pytest

from trapforge import canonical, modular
from trapforge.families import base, ledger, registry
from trapforge.linalg import diophantine, hermite, lattice, matrix, smith
from trapforge.prover import gating, model, solver

MODULES = [
    canonical,
    modular,
    matrix,
    hermite,
    smith,
    lattice,
    diophantine,
    model,
    solver,
    gating,
    base,
    registry,
    ledger,
]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_docstring_examples(module: ModuleType) -> None:
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0
    assert result.failed == 0
