"""TrapForge: author verifiable, adversarial data tasks for AI agents.

A TrapForge task is a synthetic corpus with hidden integer structure, a
computational proof that exactly one hidden parameterization fits the data,
a reference solver that recomputes the answer from the corpus, and a
byte-exact pytest grader.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trapforge")
except PackageNotFoundError:  # pragma: no cover - only when run from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
