"""The affine-scrambled ledger family.

A ledger export replaced every account number ``x`` (``0 <= x < m``) by the scrambled number
``(a*x + b) mod m``. The modulus ``m`` is published and composite; ``a`` (coprime to ``m``,
so the scramble is a bijection) and ``b`` are hidden. A few accounts were matched by hand
(the *anchors*), and the task is to report the balance of some accounts, by true number,
from transactions that only carry scrambled numbers.

The trap: the first two anchors differ by a multiple of a factor ``d`` of ``m``, so the
linear congruence ``(x2 - x1)*a = y2 - y1 (mod m)`` they give has ``d`` solutions, not one.
A solver that treats ``m`` like a prime keeps one of them. The generator plants the true
``a`` so that the smallest solution is wrong, adds decoy anchors that every candidate map
agrees on, and then reveals further anchors only until the uniqueness gate proves that one
map fits (:func:`~trapforge.prover.gate`). The visible sample of the answer shows accounts
on which the naive map happens to agree with the true one; the hidden deciding records are
the accounts on which it does not.

Corpus files (all canonical): ``params.json`` (the modulus), ``anchors.csv``
(``account,scrambled`` in the order they were confirmed), ``ledger.csv``
(``txn,account,amount`` with scrambled accounts and signed amounts in cents) and
``queries.csv`` (``account``). The answer is ``balances.csv`` (``account,balance``, one row
per query, sorted by account).

>>> family = AffineLedger()
>>> instance = family.generate(7, Difficulty.EASY)
>>> print(instance.files["anchors.csv"].decode(), end="")
account,scrambled
18,25
10,33
0,7
1,24
>>> recover_map(instance.files), naive_map(instance.files), dict(instance.hidden)
((17, 7), (8, 25), {'a': 17, 'b': 7})
>>> family.solve(instance.files) == instance.expected
True
>>> family.baseline(instance.files) == instance.expected
False
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trapforge.canonical import CanonicalError, csv_bytes, json_bytes, parse_csv
from trapforge.families.base import Difficulty, FamilyError, TaskInstance, family_rng
from trapforge.modular import (
    Congruence,
    LinearCongruence,
    solve_linear_congruence,
    solve_linear_system,
)
from trapforge.prover import Congruent, ConstraintSystem, gate

__all__ = [
    "KNOBS",
    "AffineLedger",
    "LedgerError",
    "LedgerKnobs",
    "anchor_constraint",
    "naive_map",
    "recover_map",
    "system_from_corpus",
]

HEADER = ("account", "balance")
OUTPUT = "balances.csv"
MAX_AMOUNT = 50_000
GATE_BUDGET = 64


class LedgerError(ValueError):
    """The corpus does not determine a map (or is malformed); raised by the solvers."""


@dataclass(frozen=True, slots=True)
class LedgerKnobs:
    """What a difficulty level means for this family.

    ``moduli`` are the composite moduli to draw from; the shared factor ``d`` of the first
    two anchors is a divisor of ``m`` with ``min_shared <= d <= max_shared`` and
    ``m // d >= min_quotient``.
    """

    moduli: tuple[int, ...]
    min_shared: int
    max_shared: int
    min_quotient: int
    decoys: int
    accounts: int
    transactions: int
    agreeing_queries: int
    deciding_queries: int


KNOBS: Mapping[Difficulty, LedgerKnobs] = {
    Difficulty.EASY: LedgerKnobs((24, 36, 40, 60), 2, 6, 4, 0, 10, 40, 2, 4),
    Difficulty.MEDIUM: LedgerKnobs((360, 420, 504, 720, 840), 4, 30, 12, 2, 30, 160, 3, 9),
    Difficulty.HARD: LedgerKnobs((27720, 30030, 55440, 65520), 12, 120, 100, 4, 80, 640, 4, 20),
}


def anchor_constraint(account: int, scrambled: int, modulus: int) -> Congruent:
    """What one anchor says about the hidden map: ``a*account + b = scrambled (mod m)``."""
    return Congruent.of({"a": account, "b": 1}, scrambled, modulus, f"anchor {account}")


def _map_system(modulus: int, anchors: Sequence[tuple[int, int]]) -> ConstraintSystem:
    return ConstraintSystem.build(
        {"a": (1, modulus - 1), "b": (0, modulus - 1)},
        [anchor_constraint(x, y, modulus) for x, y in anchors],
    )


# -- reading a corpus ----------------------------------------------------------------------


def _int(cell: str, where: str) -> int:
    if not cell.lstrip("-").isdigit():
        raise LedgerError(f"{where}: {cell!r} is not an integer")
    return int(cell)


def _table(files: Mapping[str, bytes], name: str, header: tuple[str, ...]) -> list[list[int]]:
    try:
        found, rows = parse_csv(files[name])
    except KeyError:
        raise LedgerError(f"the corpus has no {name}") from None
    except CanonicalError as error:
        raise LedgerError(f"{name}: {error}") from error
    if found != header:
        raise LedgerError(f"{name}: expected the header {','.join(header)}")
    return [[_int(cell, name) for cell in row] for row in rows]


def _modulus(files: Mapping[str, bytes]) -> int:
    try:
        modulus = json.loads(files["params.json"])["modulus"]
    except (KeyError, TypeError, ValueError) as error:
        raise LedgerError(f"params.json must hold an integer modulus ({error!r})") from error
    if type(modulus) is not int or modulus < 2:
        raise LedgerError(f"the modulus must be an integer of at least 2, got {modulus!r}")
    return modulus


def _anchors(files: Mapping[str, bytes], modulus: int) -> list[tuple[int, int]]:
    anchors = [(x, y) for x, y in _table(files, "anchors.csv", ("account", "scrambled"))]
    if not anchors:
        raise LedgerError("anchors.csv lists no anchors")
    for x, y in anchors:
        if not (0 <= x < modulus and 0 <= y < modulus):
            raise LedgerError(f"anchor {x} -> {y} is outside range({modulus})")
    return anchors


def system_from_corpus(files: Mapping[str, bytes]) -> ConstraintSystem:
    """The constraint system a corpus implies about the hidden map ``(a, b)``."""
    modulus = _modulus(files)
    return _map_system(modulus, _anchors(files, modulus))


def _balances(files: Mapping[str, bytes], modulus: int, a: int, b: int) -> bytes:
    totals: dict[int, int] = {}
    for _, account, amount in _table(files, "ledger.csv", ("txn", "account", "amount")):
        totals[account] = totals.get(account, 0) + amount
    queries = sorted(x for (x,) in _table(files, "queries.csv", ("account",)))
    return csv_bytes(HEADER, [[x, totals.get((a * x + b) % modulus, 0)] for x in queries])


# -- the two solvers -----------------------------------------------------------------------


def recover_map(files: Mapping[str, bytes]) -> tuple[int, int]:
    """Recover ``(a, b)`` from every anchor with linear congruences, or explain why not.

    Subtracting the first anchor ``(x1, y1)`` from anchor ``i`` eliminates ``b``:
    ``(xi - x1)*a = yi - y1 (mod m)``. The system of these congruences in the single unknown
    ``a`` is solved exactly (moduli need not be prime), and ``a`` must also be coprime to
    ``m``. The map is recovered only when exactly one ``a`` in ``range(m)`` survives.
    """
    modulus = _modulus(files)
    anchors = _anchors(files, modulus)
    x1, y1 = anchors[0]
    solved = solve_linear_system(
        [LinearCongruence(x - x1, y - y1, modulus) for x, y in anchors[1:]]
    )
    if not isinstance(solved, Congruence):
        raise LedgerError(f"the anchors contradict each other: {solved}")
    candidates = [a for a in solved.residues_mod(modulus) if math.gcd(a, modulus) == 1]
    if len(candidates) != 1:
        raise LedgerError(
            f"the anchors leave {len(candidates)} candidate maps (a = {solved.residue} "
            f"mod {solved.modulus}); a unique answer needs exactly one"
        )
    a = candidates[0]
    return a, (y1 - a * x1) % modulus


def naive_map(files: Mapping[str, bytes]) -> tuple[int, int]:
    """The map a solver gets by treating ``m`` as prime and using the first two anchors.

    It solves ``(x2 - x1)*a = y2 - y1 (mod m)`` as if the solution were unique and keeps its
    smallest representative, so it fits both anchors but not necessarily the rest.
    """
    modulus = _modulus(files)
    anchors = _anchors(files, modulus)
    if len(anchors) < 2:
        raise LedgerError("the naive solver needs two anchors")
    (x1, y1), (x2, y2) = anchors[:2]
    solved = solve_linear_congruence(x2 - x1, y2 - y1, modulus)
    if not isinstance(solved, Congruence):
        raise LedgerError("the first two anchors contradict each other")
    a = solved.residue
    return a, (y1 - a * x1) % modulus


# -- the family ----------------------------------------------------------------------------


def _unit_at_least(rng: random.Random, low: int, modulus: int) -> int:
    """A random ``u`` in ``range(low, modulus)`` with ``gcd(u, modulus) == 1``."""
    while True:
        candidate = rng.randrange(low, modulus)
        if math.gcd(candidate, modulus) == 1:
            return candidate


def _fresh(rng: random.Random, modulus: int, used: set[int], count: int) -> list[int]:
    """Up to ``count`` random accounts not in ``used`` (which is updated), in draw order."""
    fresh: list[int] = []
    while len(fresh) < count and len(used) < modulus:
        x = rng.randrange(modulus)
        if x not in used:
            used.add(x)
            fresh.append(x)
    return fresh


def _pick(rng: random.Random, pool: Sequence[int], avoid: set[int], count: int) -> list[int]:
    """``count`` members of ``pool``, taking members outside ``avoid`` first."""
    preferred = [x for x in pool if x not in avoid]
    fallback = [x for x in pool if x in avoid]
    rng.shuffle(preferred)
    rng.shuffle(fallback)
    chosen = (preferred + fallback)[:count]
    if len(chosen) < count:
        raise FamilyError(f"only {len(chosen)} accounts available for {count} queries")
    return chosen


def _amount(rng: random.Random) -> int:
    return rng.choice((-1, 1)) * rng.randrange(1, MAX_AMOUNT + 1)


def _ledger(
    rng: random.Random,
    modulus: int,
    knobs: LedgerKnobs,
    queried: Sequence[int],
    deciding: Sequence[tuple[int, int]],
) -> list[list[int]]:
    """Transactions ``[txn, scrambled account, amount]`` over a set of active accounts.

    ``queried`` holds the scrambled numbers of the queried accounts (each gets at least one
    transaction) and ``deciding`` the pairs (true, naive) of scrambled numbers of the
    deciding records. Transactions are added until every deciding pair has different
    totals, so the naive map is wrong on every deciding record, not just on its number.
    """
    active = set(queried)
    for _, naive in deciding:
        if rng.random() < 0.5:
            active.add(naive)
    while len(active) < knobs.accounts:
        active.add(rng.randrange(modulus))
    accounts = sorted(active)
    postings = list(queried)
    postings += [rng.choice(accounts) for _ in range(knobs.transactions - len(postings))]
    rng.shuffle(postings)
    amounts = [_amount(rng) for _ in postings]
    while True:
        totals: dict[int, int] = {}
        for account, amount in zip(postings, amounts, strict=True):
            totals[account] = totals.get(account, 0) + amount
        clashes = [true for true, naive in deciding if totals[true] == totals.get(naive, 0)]
        if not clashes:
            break
        for true in clashes:
            postings.append(true)
            amounts.append(_amount(rng))
    pairs = zip(postings, amounts, strict=True)
    return [[i, y, amount] for i, (y, amount) in enumerate(pairs, start=1)]


def _instruction(modulus: int, sample: bytes) -> str:
    return f"""# Reconcile a scrambled ledger

An export replaced every account number `x` (with `0 <= x < {modulus}`) by the scrambled
number `(a*x + b) mod {modulus}`. The integers `a` and `b` are unknown; `a` is coprime to
{modulus}, so no two accounts share a scrambled number. The modulus is also in
`data/params.json`.

- `data/anchors.csv` lists accounts whose scrambled number was confirmed by hand
  (`account,scrambled`).
- `data/ledger.csv` lists transactions (`txn,account,amount`) by *scrambled* account;
  amounts are signed integers in cents.
- `data/queries.csv` lists true account numbers (`account`).

Write `{OUTPUT}` with the header `account,balance` and one row per queried account: its
true account number and the sum of the amounts of its transactions (0 if it has none),
sorted by account, comma separated, `\\n` line endings, final newline. The file is compared
byte for byte.

Some rows of the expected output:

```
{sample.decode("ascii").rstrip()}
```
"""


class AffineLedger:
    """Balances behind an affine scramble of account numbers modulo a composite ``m``."""

    name = "affine-ledger"
    summary = "account numbers scrambled by x -> (a*x + b) mod m with composite m"

    def generate(self, seed: int, difficulty: Difficulty) -> TaskInstance:
        """Plant a map, reveal anchors until it is unique, and write the ledger."""
        rng = family_rng(self.name, seed, difficulty)
        knobs = KNOBS[difficulty]
        m = rng.choice(knobs.moduli)
        shared = rng.choice(
            [
                d
                for d in range(knobs.min_shared, knobs.max_shared + 1)
                if m % d == 0 and m // d >= knobs.min_quotient
            ]
        )
        quotient = m // shared
        # The naive solver keeps a mod quotient, so the true a must be at least quotient.
        a = _unit_at_least(rng, quotient, m)
        b = rng.randrange(m)

        def scramble(x: int) -> int:
            return (a * x + b) % m

        # Two anchors whose difference shares exactly the factor `shared` with m.
        first = rng.randrange(m)
        anchors = [first, (first + shared * _unit_at_least(rng, 1, quotient)) % m]
        used = set(anchors)
        decoys: list[int] = []  # same class as `first` mod `shared`: every candidate fits
        while len(decoys) < knobs.decoys:
            x = (first + shared * rng.randrange(quotient)) % m
            if x not in used:
                used.add(x)
                decoys.append(x)
        stream = decoys + _fresh(rng, m, used, GATE_BUDGET)
        result = gate(
            _map_system(m, [(x, scramble(x)) for x in anchors]),
            [anchor_constraint(x, scramble(x), m) for x in stream],
            hidden={"a": a, "b": b},
        )
        if not result.passed:  # pragma: no cover - random anchors cut the space quickly
            raise FamilyError(f"seed {seed}: the anchors never became unique")
        anchors += stream[: len(result.added)]

        # Where the naive map (a mod quotient, fitted to the first anchor) agrees with the
        # true one: (a - naive_a) * (x - first) = 0 (mod m).
        naive_a = a % quotient
        naive_b = (scramble(first) - naive_a * first) % m
        step = m // math.gcd(a - naive_a, m)
        agreeing = [(first + step * t) % m for t in range(m // step)]
        agree_set = set(agreeing)
        deciding = [x for x in range(m) if x not in agree_set]
        sample_queries = sorted(_pick(rng, agreeing, set(anchors), knobs.agreeing_queries))
        hidden_queries = sorted(_pick(rng, deciding, set(anchors), knobs.deciding_queries))
        queries = sorted(sample_queries + hidden_queries)

        ledger = _ledger(
            rng,
            m,
            knobs,
            [scramble(x) for x in queries],
            [(scramble(x), (naive_a * x + naive_b) % m) for x in hidden_queries],
        )
        files = {
            "params.json": json_bytes({"modulus": m}),
            "anchors.csv": csv_bytes(("account", "scrambled"), [[x, scramble(x)] for x in anchors]),
            "ledger.csv": csv_bytes(("txn", "account", "amount"), ledger),
            "queries.csv": csv_bytes(("account",), [[x] for x in queries]),
        }
        expected = _balances(files, m, a, b)
        sample_files = {
            **files,
            "queries.csv": csv_bytes(("account",), [[x] for x in sample_queries]),
        }
        sample = _balances(sample_files, m, a, b)
        return TaskInstance(
            family=self.name,
            seed=seed,
            difficulty=difficulty,
            instruction=_instruction(m, sample),
            files=files,
            output=OUTPUT,
            expected=expected,
            sample=sample,
            hidden={"a": a, "b": b},
            system=result.system,
            extras={
                "modulus": m,
                "shared_factor": shared,
                "worlds_from_first_two_anchors": result.sizes[0],
                "anchors": len(anchors),
                "decoy_anchors": len(decoys),
                "sample_records": len(sample_queries),
                "deciding_records": len(hidden_queries),
                "transactions": len(ledger),
            },
        )

    def solve(self, files: Mapping[str, bytes]) -> bytes:
        """The reference solver: recover the map from every anchor, then total the ledger."""
        a, b = recover_map(files)
        return _balances(files, _modulus(files), a, b)

    def baseline(self, files: Mapping[str, bytes]) -> bytes:
        """The naive solver: the map from the first two anchors, as if ``m`` were prime."""
        a, b = naive_map(files)
        return _balances(files, _modulus(files), a, b)
