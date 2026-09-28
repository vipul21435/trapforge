"""The affine-scrambled ledger family: unique by proof, solved by the reference, and a trap."""

import math
import os
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from functools import cache

import pytest

from trapforge.canonical import csv_bytes, json_bytes, parse_csv
from trapforge.families import REGISTRY, AffineLedger, Difficulty, FamilyError, TaskFamily
from trapforge.families.base import TaskInstance
from trapforge.families.ledger import (
    KNOBS,
    LedgerError,
    _ledger,
    _pick,
    naive_map,
    recover_map,
    system_from_corpus,
)
from trapforge.prover import Ambiguity, ConstraintSystem, UniqueProof, prove

FAMILY = AffineLedger()
SEEDS = {Difficulty.EASY: 40, Difficulty.MEDIUM: 25, Difficulty.HARD: 8}
CASES = [(difficulty, seed) for difficulty, count in SEEDS.items() for seed in range(count)]
IDS = [f"{difficulty.value}-{seed}" for difficulty, seed in CASES]


@cache
def generated(difficulty: Difficulty, seed: int) -> TaskInstance:
    return FAMILY.generate(seed, difficulty)


def rows(data: bytes) -> list[tuple[int, ...]]:
    return [tuple(int(cell) for cell in row) for row in parse_csv(data)[1]]


def test_family_is_registered_and_implements_the_protocol() -> None:
    assert "affine-ledger" in REGISTRY
    assert isinstance(REGISTRY.get("affine-ledger"), AffineLedger)
    assert isinstance(FAMILY, TaskFamily)


@pytest.mark.parametrize(("difficulty", "seed"), CASES, ids=IDS)
def test_every_instance_passes_the_uniqueness_gate(difficulty: Difficulty, seed: int) -> None:
    instance = generated(difficulty, seed)
    # The system the instance carries is exactly what its corpus implies.
    assert instance.system == system_from_corpus(instance.files)
    proof = prove(instance.system)
    assert isinstance(proof, UniqueProof)
    assert proof.solution == instance.hidden
    assert proof.check().valid


@pytest.mark.parametrize(("difficulty", "seed"), CASES, ids=IDS)
def test_first_two_anchors_leave_several_worlds(difficulty: Difficulty, seed: int) -> None:
    instance = generated(difficulty, seed)
    m = instance.extras["modulus"]
    shared = instance.extras["shared_factor"]
    anchors = rows(instance.files["anchors.csv"])
    (x1, _), (x2, _) = anchors[:2]
    assert any(m % p == 0 for p in range(2, math.isqrt(m) + 1)), "m must be composite"
    assert math.gcd(instance.hidden["a"], m) == 1
    assert math.gcd(x2 - x1, m) == shared > 1
    first_two = ConstraintSystem(instance.system.unknowns, instance.system.constraints[:2])
    proof = prove(first_two)
    assert isinstance(proof, Ambiguity)
    assert proof.count == shared == instance.extras["worlds_from_first_two_anchors"]
    naive_a, naive_b = naive_map(instance.files)
    assert first_two.violations({"a": naive_a, "b": naive_b}) == ()
    assert (naive_a, naive_b) != (instance.hidden["a"], instance.hidden["b"])
    # Decoy anchors sit in the class of x1 mod the shared factor, so no candidate breaks them.
    decoys = anchors[2 : 2 + instance.extras["decoy_anchors"]]
    assert all((x - x1) % shared == 0 for x, _ in decoys)


@pytest.mark.parametrize(("difficulty", "seed"), CASES, ids=IDS)
def test_reference_solver_reproduces_the_expected_bytes(difficulty: Difficulty, seed: int) -> None:
    instance = generated(difficulty, seed)
    assert recover_map(instance.files) == (instance.hidden["a"], instance.hidden["b"])
    assert FAMILY.solve(instance.files) == instance.expected


@pytest.mark.parametrize(("difficulty", "seed"), CASES, ids=IDS)
def test_baseline_is_right_on_the_sample_and_wrong_on_deciding_records(
    difficulty: Difficulty, seed: int
) -> None:
    instance = generated(difficulty, seed)
    expected = rows(instance.expected)
    sample = rows(instance.sample)
    naive = rows(FAMILY.baseline(instance.files))
    assert set(sample) <= set(expected)
    assert set(sample) <= set(naive)
    assert len(sample) == instance.extras["sample_records"] == KNOBS[difficulty].agreeing_queries
    wrong = [row for row, guess in zip(expected, naive, strict=True) if row != guess]
    assert len(wrong) == instance.extras["deciding_records"] > 0
    assert not set(wrong) & set(sample)
    assert FAMILY.baseline(instance.files) != instance.expected


@pytest.mark.parametrize(("difficulty", "seed"), CASES[:5] + CASES[-3:], ids=IDS[:5] + IDS[-3:])
def test_instance_layout(difficulty: Difficulty, seed: int) -> None:
    instance = generated(difficulty, seed)
    assert sorted(instance.files) == ["anchors.csv", "ledger.csv", "params.json", "queries.csv"]
    assert instance.files["params.json"] == json_bytes({"modulus": instance.extras["modulus"]})
    assert instance.output == "balances.csv"
    assert len(rows(instance.files["anchors.csv"])) == instance.extras["anchors"]
    assert len(rows(instance.files["ledger.csv"])) == instance.extras["transactions"]
    assert instance.extras["transactions"] >= KNOBS[difficulty].transactions
    queries = [x for (x,) in rows(instance.files["queries.csv"])]
    assert queries == sorted(set(queries))
    assert [x for x, _ in rows(instance.expected)] == queries
    assert instance.sample.decode().rstrip() in instance.instruction
    assert f"mod {instance.extras['modulus']}" in instance.instruction


GOLDEN = {
    Difficulty.EASY: "c9bc784f50ad89f7574a1329294851eb87db3abf9e852979680c035869bfbfe6",
    Difficulty.MEDIUM: "bed93106b3f050feb94b2208b54d28342a5efc5c5d298a32fbcd2689b0e95f2f",
    Difficulty.HARD: "8c365aa78ffeb19c41d5fd877d5ca8a3b1c6a0dc018c6ab6f7b01a36388cbf05",
}


@pytest.mark.parametrize("difficulty", list(Difficulty), ids=str)
def test_seed_zero_matches_the_golden_digest(difficulty: Difficulty) -> None:
    """Pinned SHA-256 digests: CI on Linux must reproduce the bytes generated on macOS."""
    assert FAMILY.generate(0, difficulty).digest() == GOLDEN[difficulty]


def test_same_seed_same_bytes_and_different_seeds_differ() -> None:
    digests = {FAMILY.generate(seed, Difficulty.EASY).digest() for seed in range(20)}
    assert len(digests) == 20
    assert FAMILY.generate(3, Difficulty.EASY).bundle() == generated(Difficulty.EASY, 3).bundle()


DIGEST_SCRIPT = """
from trapforge.families import REGISTRY, Difficulty
for family in REGISTRY:
    for difficulty in Difficulty:
        for seed in range(3):
            digest = family.generate(seed, difficulty).digest()
            print(family.name, difficulty.value, seed, digest)
"""


@pytest.mark.slow
@pytest.mark.parametrize("hash_seed", ["0", "99"])
def test_registered_families_are_deterministic_across_processes(hash_seed: str) -> None:
    here = "".join(
        f"{family.name} {difficulty.value} {seed} {family.generate(seed, difficulty).digest()}\n"
        for family in REGISTRY
        for difficulty in Difficulty
        for seed in range(3)
    )
    result = subprocess.run(
        [sys.executable, "-c", DIGEST_SCRIPT],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert result.stdout == here


def corpus(
    anchors: Sequence[tuple[int, int]], modulus: object = 12, **extra: bytes
) -> Mapping[str, bytes]:
    files = {
        "params.json": json_bytes({"modulus": modulus}),
        "anchors.csv": csv_bytes(("account", "scrambled"), [list(pair) for pair in anchors]),
        "ledger.csv": b"txn,account,amount\n1,7,100\n2,7,-30\n",
        "queries.csv": b"account\n0\n1\n",
    }
    files.update(extra)
    return files


def test_solvers_on_a_hand_built_corpus() -> None:
    # a = 5, b = 7 mod 12: 0 -> 7 and 1 -> 0; the anchors at 1 and 3 fit a = 11 too.
    files = corpus([(1, 0), (3, 10), (2, 5)])
    assert recover_map(files) == (5, 7)
    assert FAMILY.solve(files) == b"account,balance\n0,70\n1,0\n"
    assert naive_map(files) == (5, 7)  # a mod 6 = 5 happens to be right here
    assert naive_map(corpus([(1, 2), (3, 0)])) == (5, 9)  # the truth could be a = 11, b = 3


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (corpus([(1, 0), (3, 10)]), r"leave 2 candidate maps \(a = 5 mod 6\)"),
        (corpus([(0, 0), (1, 2)]), "leave 0 candidate maps"),
        (corpus([(1, 0), (3, 10), (5, 1)]), "the anchors contradict each other"),
        (corpus([]), "anchors.csv lists no anchors"),
        (corpus([(12, 0)]), "outside range"),
        (corpus([(1, 0)], modulus=1), "at least 2"),
        (corpus([(1, 0)], modulus="12"), "at least 2"),
        ({"anchors.csv": b"account,scrambled\n"}, "params.json must hold"),
        (corpus([(1, 0)], **{"params.json": b"[]"}), "params.json must hold"),
        (corpus([(1, 0)], **{"anchors.csv": b"account,scrambled\n1,x\n"}), "not an integer"),
        (corpus([(1, 0)], **{"anchors.csv": b"account,scrambled\n--1,0\n"}), "not an integer"),
        (corpus([(1, 0)], **{"anchors.csv": b"account,scrambled\n1,-\n"}), "not an integer"),
        (corpus([(1, 0)], **{"anchors.csv": b"account,scrambled\n+1,0\n"}), "not an integer"),
        (
            corpus([(1, 0)], **{"anchors.csv": b"account,scrambled\n" + b"7" * 5000 + b",0\n"}),
            "an integer with 5000 characters is too long",
        ),
        (corpus([(1, 0)], **{"anchors.csv": b"x,y\n1,0\n"}), "expected the header"),
        (corpus([(1, 0)], **{"anchors.csv": b"account,scrambled\n1,0"}), "end with a newline"),
    ],
)
def test_reference_solver_explains_a_corpus_it_cannot_solve(
    files: Mapping[str, bytes], message: str
) -> None:
    with pytest.raises(LedgerError, match=message):
        recover_map(files)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"ledger.csv": b"txn,account,amount\n1,7,--100\n"}, "ledger.csv: '--100' is not"),
        ({"ledger.csv": b"txn,account,amount\n1," + b"9" * 4400 + b",1\n"}, "too long"),
        ({"queries.csv": b"account\n---7\n"}, "queries.csv: '---7' is not"),
        ({"queries.csv": b"account\n-5\n"}, r"query -5 is outside range\(12\)"),
        ({"queries.csv": b"account\n0\n100\n"}, r"query 100 is outside range\(12\)"),
    ],
)
def test_both_solvers_reject_a_malformed_ledger_or_query(
    extra: dict[str, bytes], message: str
) -> None:
    # Regression: these used to raise a plain ValueError or silently answer for an aliased
    # account instead of raising LedgerError.
    files = corpus([(1, 0), (3, 10), (2, 5)], **extra)
    for solver in (FAMILY.solve, FAMILY.baseline):
        with pytest.raises(LedgerError, match=message):
            solver(files)


def test_solvers_need_every_file() -> None:
    files = dict(corpus([(1, 0), (3, 10), (2, 5)]))
    del files["ledger.csv"]
    with pytest.raises(LedgerError, match=r"the corpus has no ledger\.csv"):
        FAMILY.solve(files)


def test_naive_solver_errors() -> None:
    with pytest.raises(LedgerError, match="needs two anchors"):
        naive_map(corpus([(1, 0)]))
    with pytest.raises(LedgerError, match="first two anchors contradict"):
        naive_map(corpus([(0, 0), (2, 1)]))


class _Rigged(random.Random):
    """Every draw returns the last option, so every amount is +1."""

    def random(self) -> float:
        return 0.0

    def randrange(self, start: int, stop: int | None = None, step: int = 1) -> int:
        return start if stop is None else stop - 1

    def choice(self, seq: Sequence[int]) -> int:  # type: ignore[override]
        return seq[-1]

    def shuffle(self, x: list[int]) -> None:  # type: ignore[override]
        pass


def test_ledger_adds_transactions_until_every_deciding_pair_differs() -> None:
    knobs = KNOBS[Difficulty.EASY]
    tiny = type(knobs)(
        knobs.moduli, 2, 6, 4, 0, accounts=2, transactions=2, agreeing_queries=0, deciding_queries=1
    )
    ledger = _ledger(_Rigged(), 12, tiny, [3], [(3, 5)])
    # Accounts 3 and 5 would both total 50000; one more posting to 3 breaks the tie.
    assert ledger == [[1, 3, 50000], [2, 5, 50000], [3, 3, 50000]]


def test_pick_prefers_accounts_outside_avoid_and_fails_when_short() -> None:
    rng = random.Random(0)
    assert sorted(_pick(rng, [1, 2, 3], {1}, 2)) == [2, 3]
    assert sorted(_pick(rng, [1, 2, 3], {1, 2}, 3)) == [1, 2, 3]
    with pytest.raises(FamilyError, match="only 3 accounts available for 4 queries"):
        _pick(rng, [1, 2, 3], set(), 4)
