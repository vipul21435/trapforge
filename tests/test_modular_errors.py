"""The typed errors of the modular core survive pickling, copying and process pools."""

import copy
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import pytest

from trapforge.modular import (
    Congruence,
    InvalidModulusError,
    ModularError,
    NonCanonicalResidueError,
    NotInvertibleError,
    mod_inverse,
)

ERRORS = [
    NotInvertibleError(6, 9, 3),
    InvalidModulusError(0),
    InvalidModulusError(-12),
    NonCanonicalResidueError(9, 4),
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


# -- operands past the int-to-str limit -----------------------------------------------------

HUGE = 10**5000  # far beyond sys.get_int_max_str_digits() == 4300


def test_huge_non_invertible_operands_still_raise_the_typed_error() -> None:
    # gcd(2 * 10**5000, 4 * 10**5000 + 2) == 2. Before the fix, formatting the message
    # inside the constructor raised a bare ValueError that was not a ModularError.
    with pytest.raises(NotInvertibleError) as info:
        mod_inverse(2 * HUGE, 4 * HUGE + 2)
    assert info.value.gcd == 2
    assert str(info.value) == (
        "<16611-bit integer> has no inverse modulo <16612-bit integer>: "
        "gcd(<16611-bit integer>, <16612-bit integer>) = 2 != 1"
    )
    clone = pickle.loads(pickle.dumps(info.value))
    assert (clone.a, clone.modulus, clone.gcd) == (2 * HUGE, 4 * HUGE + 2, 2)


def test_huge_invalid_modulus_still_raises_the_typed_error() -> None:
    with pytest.raises(InvalidModulusError) as info:
        mod_inverse(3, -HUGE)
    assert info.value.modulus == -HUGE
    assert str(info.value) == "modulus must be a positive integer, got <-16610-bit integer>"


def test_huge_non_canonical_residue_still_raises_a_modular_error() -> None:
    with pytest.raises(NonCanonicalResidueError) as info:
        Congruence(HUGE, 7)
    assert isinstance(info.value, ModularError)
    assert (info.value.residue, info.value.modulus) == (HUGE, 7)
    assert str(info.value).startswith("residue <16610-bit integer> is not canonical modulo 7")
    assert pickle.loads(pickle.dumps(info.value)).residue == HUGE


def test_huge_non_multiple_modulus_still_raises_a_modular_error() -> None:
    with pytest.raises(ModularError, match="is not a multiple of the class modulus 4"):
        Congruence(1, 4).residues_mod(HUGE + 1)


def test_small_operands_keep_their_decimal_messages() -> None:
    with pytest.raises(NonCanonicalResidueError, match="residue 5 is not canonical modulo 5"):
        Congruence(5, 5)
