"""The plugin API: difficulties, instances, the registry and cross-process determinism."""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _toy_family import ToyShift
from trapforge.canonical import tree_digest
from trapforge.families import (
    Difficulty,
    FamilyError,
    Registry,
    TaskFamily,
    TaskInstance,
    family_rng,
)

TESTS = Path(__file__).resolve().parent


def test_difficulty_parses_names_and_rejects_others() -> None:
    assert Difficulty.parse(" MEDIUM ") is Difficulty.MEDIUM
    assert [d.value for d in Difficulty] == ["easy", "medium", "hard"]
    with pytest.raises(FamilyError, match="expected one of easy, medium, hard"):
        Difficulty.parse("brutal")


def test_family_rng_depends_on_every_part_of_the_key() -> None:
    draw = family_rng("toy", 1, Difficulty.EASY).random()
    assert draw == family_rng("toy", 1, Difficulty.EASY).random()
    assert draw != family_rng("toy", 2, Difficulty.EASY).random()
    assert draw != family_rng("toy", 1, Difficulty.HARD).random()
    assert draw != family_rng("toy2", 1, Difficulty.EASY).random()


@pytest.mark.parametrize("seed", [-1, 1.0, True, "1"])
def test_family_rng_rejects_bad_seeds(seed: object) -> None:
    with pytest.raises(FamilyError, match="seed must be a non-negative int"):
        family_rng("toy", seed, Difficulty.EASY)  # type: ignore[arg-type]


def test_toy_family_implements_the_protocol() -> None:
    assert isinstance(ToyShift(), TaskFamily)
    assert not isinstance(object(), TaskFamily)


def test_instance_bundle_layout_and_digest() -> None:
    instance = ToyShift().generate(3, Difficulty.MEDIUM)
    bundle = instance.bundle()
    assert sorted(bundle) == [
        "data/pairs.csv",
        "expected/shift.csv",
        "meta/instruction.md",
        "meta/system.json",
        "meta/task.json",
        "sample/shift.csv",
    ]
    assert bundle["meta/instruction.md"] == b"Find the shift.\n"
    assert b'"difficulty": "medium"' in bundle["meta/task.json"]
    assert instance.digest() == tree_digest(bundle)
    assert instance.files["pairs.csv"].startswith(b"x,y\n")
    with pytest.raises(TypeError):
        instance.files["extra"] = b""  # type: ignore[index]


def _replace(instance: TaskInstance, **changes: object) -> TaskInstance:
    return dataclasses.replace(instance, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"difficulty": "easy"}, "must be a Difficulty"),
        ({"seed": -3}, "seed must be"),
        ({"files": {}}, "files must not be empty"),
        ({"files": {"a.csv": "text"}}, r"files\['a.csv'\] must be bytes"),
        ({"files": {"../a.csv": b""}}, "normalized relative POSIX path"),
        ({"output": "/abs.csv"}, "normalized relative POSIX path"),
        ({"expected": "text"}, "must be bytes"),
        ({"sample": None}, "sample must be bytes"),
        ({"system": {}}, "must be a ConstraintSystem"),
        ({"hidden": {"k": 11}}, "planted world violates bounds of k"),
        ({"instruction": "café"}, "ASCII"),
        ({"extras": {"size": "big"}}, "extras must map names to ints"),
    ],
)
def test_instance_validation(changes: dict[str, object], message: str) -> None:
    instance = ToyShift().generate(0, Difficulty.EASY)
    with pytest.raises(ValueError, match=message):
        _replace(instance, **changes)


def test_instance_rejects_a_planted_world_that_breaks_the_corpus() -> None:
    instance = ToyShift().generate(0, Difficulty.EASY)
    wrong = (instance.hidden["k"] + 1) % 10
    with pytest.raises(FamilyError, match="planted world violates pair x="):
        _replace(instance, hidden={"k": wrong})


def test_registry_registers_looks_up_and_generates() -> None:
    registry = Registry()
    toy = ToyShift()
    assert registry.register(toy) is toy
    assert "toy-shift" in registry
    assert len(registry) == 1
    assert list(registry) == [toy]
    assert registry.names() == ("toy-shift",)
    assert registry.get("toy-shift") is toy
    instance = registry.generate("toy-shift", 5, Difficulty.HARD)
    assert (instance.seed, instance.difficulty) == (5, Difficulty.HARD)
    with pytest.raises(FamilyError, match="already registered"):
        registry.register(ToyShift())
    with pytest.raises(FamilyError, match="unknown family 'toy'; registered: toy-shift"):
        registry.get("toy")


def test_registry_rejects_malformed_families() -> None:
    registry = Registry()
    with pytest.raises(FamilyError, match="does not implement the TaskFamily protocol"):
        registry.register(object())  # type: ignore[arg-type]
    for bad in ["Toy", "toy_shift", "-toy", "toy-", "", "1toy"]:
        family = ToyShift()
        family.name = bad  # type: ignore[misc]
        with pytest.raises(FamilyError, match="kebab-case"):
            registry.register(family)


def test_registry_catches_a_mislabelled_instance() -> None:
    class Liar(ToyShift):
        def generate(self, seed: int, difficulty: Difficulty) -> TaskInstance:
            return super().generate(seed + 1, difficulty)

    registry = Registry()
    registry.register(Liar())
    with pytest.raises(FamilyError, match="labelled its instance as toy-shift/easy/2"):
        registry.generate("toy-shift", 1, Difficulty.EASY)


DIGEST_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
from _toy_family import ToyShift
from trapforge.families import Difficulty
family = ToyShift()
for seed in range(4):
    for difficulty in Difficulty:
        print(seed, difficulty.value, family.generate(seed, difficulty).digest())
"""


@pytest.mark.slow
@pytest.mark.parametrize("hash_seed", ["0", "4242"])
def test_digests_match_across_processes(hash_seed: str) -> None:
    family = ToyShift()
    here = "".join(
        f"{seed} {difficulty.value} {family.generate(seed, difficulty).digest()}\n"
        for seed in range(4)
        for difficulty in Difficulty
    )
    result = subprocess.run(
        [sys.executable, "-c", DIGEST_SCRIPT, str(TESTS)],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout == here
