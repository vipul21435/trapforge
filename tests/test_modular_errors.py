"""The typed errors of the modular core survive pickling, copying and process pools."""

import copy
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import pytest

from trapforge.modular import (
    InvalidModulusError,
    ModularError,
    NotInvertibleError,
    mod_inverse,
)

ERRORS = [
    NotInvertibleError(6, 9, 3),
    InvalidModulusError(0),
    InvalidModulusError(-12),
]

ROUND_TRIPS: list[Callable[[ModularError], object]] = [
    lambda error: pickle.loads(pickle.dumps(error)),
    copy.copy,
    copy.deepcopy,
]


@pytest.mark.parametrize("error", ERRORS, ids=repr)
@pytest.mark.parametrize("round_trip", ROUND_TRIPS, ids=["pickle", "copy", "deepcopy"])
def test_errors_round_trip_with_type_payload_and_message(
    error: ModularError, round_trip: Callable[[ModularError], object]
) -> None:
    clone = round_trip(error)
    assert type(clone) is type(error)
    assert vars(clone) == vars(error)
    assert clone.args == error.args
    assert str(clone) == str(error)


def test_error_messages_are_formatted_once() -> None:
    assert str(InvalidModulusError(0)) == "modulus must be a positive integer, got 0"
    assert str(NotInvertibleError(6, 9, 3)) == "6 has no inverse modulo 9: gcd(6, 9) = 3 != 1"
    assert NotInvertibleError(6, 9, 3).args == (6, 9, 3)


@pytest.mark.slow
def test_process_pool_delivers_the_typed_error_and_stays_usable() -> None:
    # Before the fix the worker's exception could not be unpickled in the parent, which
    # killed the pool (BrokenProcessPool) and lost the typed error.
    with ProcessPoolExecutor(max_workers=1) as pool:
        with pytest.raises(NotInvertibleError) as info:
            pool.submit(mod_inverse, 6, 9).result(timeout=60)
        assert (info.value.a, info.value.modulus, info.value.gcd) == (6, 9, 3)
        with pytest.raises(InvalidModulusError):
            pool.submit(mod_inverse, 3, 0).result(timeout=60)
        assert pool.submit(mod_inverse, 3, 7).result(timeout=60) == 5
