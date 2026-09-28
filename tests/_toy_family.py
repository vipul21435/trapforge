"""A minimal task family used to test the plugin API without any real family.

The hidden world is a shift ``k`` in ``range(10)``; the corpus lists pairs ``(x, (x + k) %
10)`` and the answer is ``k``. The baseline reads the first pair only, which is enough, so
this family is deliberately not adversarial: it exists to exercise the API.
"""

from collections.abc import Mapping

from trapforge.canonical import csv_bytes, parse_csv
from trapforge.families import Difficulty, TaskInstance, family_rng
from trapforge.prover import Congruent, ConstraintSystem


class ToyShift:
    name = "toy-shift"
    summary = "a hidden shift mod 10 behind a few pairs"

    def generate(self, seed: int, difficulty: Difficulty) -> TaskInstance:
        rng = family_rng(self.name, seed, difficulty)
        k = rng.randrange(10)
        count = {Difficulty.EASY: 2, Difficulty.MEDIUM: 4, Difficulty.HARD: 8}[difficulty]
        xs = [rng.randrange(10) for _ in range(count)]
        pairs = [[x, (x + k) % 10] for x in xs]
        system = ConstraintSystem.build(
            {"k": (0, 9)},
            [Congruent.of({"k": 1}, y - x, 10, f"pair x={x}") for x, y in pairs],
        )
        expected = csv_bytes(["shift"], [[k]])
        return TaskInstance(
            family=self.name,
            seed=seed,
            difficulty=difficulty,
            instruction="Find the shift.",
            files={"pairs.csv": csv_bytes(["x", "y"], pairs)},
            output="shift.csv",
            expected=expected,
            sample=expected,
            hidden={"k": k},
            system=system,
        )

    def solve(self, files: Mapping[str, bytes]) -> bytes:
        _, rows = parse_csv(files["pairs.csv"])
        x, y = (int(cell) for cell in rows[0])
        return csv_bytes(["shift"], [[(y - x) % 10]])

    baseline = solve
