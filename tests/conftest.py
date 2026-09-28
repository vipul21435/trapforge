"""Shared pytest configuration.

Hypothesis profiles: ``dev`` (the default) explores fresh random examples on every run, and
``ci`` is derandomized so a CI failure always reproduces from the same seed. Select one
with the ``HYPOTHESIS_PROFILE`` environment variable.
"""

import os

from hypothesis import settings

settings.register_profile("dev", max_examples=200, deadline=None)
settings.register_profile("ci", max_examples=300, deadline=None, derandomize=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
