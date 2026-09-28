"""Run the usage examples embedded in module docstrings, so they cannot drift."""

import doctest
from types import ModuleType

import pytest

from trapforge import modular
from trapforge.linalg import matrix


@pytest.mark.parametrize("module", [modular, matrix], ids=lambda m: m.__name__)
def test_docstring_examples(module: ModuleType) -> None:
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0
    assert result.failed == 0
