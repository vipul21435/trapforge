"""Shared pytest configuration.

Hypothesis profiles: ``dev`` (the default) explores fresh random examples on every run, and
``ci`` is derandomized so a CI failure always reproduces from the same seed. Select one
with the ``HYPOTHESIS_PROFILE`` environment variable.
"""

import os
import signal
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import pytest
from hypothesis import settings

settings.register_profile("dev", max_examples=200, deadline=None)
settings.register_profile("ci", max_examples=300, deadline=None, derandomize=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def time_limit() -> Callable[[float], AbstractContextManager[None]]:
    """A context manager that fails the test when its body runs longer than ``seconds``.

    It guards regression tests for algorithms that used to hang: without it, a slow path
    would stall the suite instead of failing it. Uses SIGALRM, so POSIX only.
    """

    @contextmanager
    def limit(seconds: float) -> Iterator[None]:
        def expire(signum: int, frame: object) -> None:
            raise TimeoutError(f"took longer than {seconds} s")

        previous = signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    return limit
