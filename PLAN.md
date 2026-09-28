# TrapForge build plan

TrapForge is a framework for authoring verifiable, adversarial data tasks for AI agents:
synthetic corpora with hidden integer structure, a computational proof that exactly one
interpretation fits the data, a reference solver that computes the answer genuinely, and
byte-exact pytest graders. This file is the working plan; decisions are logged at the end.

## Architecture (target)

```
src/trapforge/
  modular.py          extended gcd, modular inverse, CRT (non-coprime), linear congruences
  linalg/             exact integer matrices: HNF, SNF, kernel bases, Diophantine systems
  prover/             model.py, solver.py (verdicts + certificates), certificate.py
                      (standalone checker), gating.py (uniqueness gate)
  canonical.py        canonical byte-exact writers (CSV, JSON, text) and SHA-256 digests
  families/           plugin API + registry + the three original task families
  bundle/             exporter, digest-pinned Dockerfile template, byte-exact grader template
  verify.py           reference-passes / baseline-fails check, locally and in Docker
  report.py           difficulty report (baseline failure rate, ambiguity-space size)
  cli.py              Typer CLI
```

Conventions: integers are Python `int` everywhere (no floats, no numpy), matrices are
immutable `tuple[tuple[int, ...], ...]`, all randomness flows from an explicit
`random.Random(seed)` (never the global RNG), and every file a task emits is written through
one canonical writer (UTF-8, `\n` newlines, fixed key and row order) so outputs are
byte-identical across runs, processes and platforms.

## Slices

Each slice is a coherent feature delivered as 3-4 real commits with tests; the suite stays
green (ruff, mypy --strict, pytest with the 85% coverage gate) after every commit.

### Slice 1: Modular arithmetic core [x] done

Goal: Implement `trapforge.modular` in pure Python: extended gcd, modular inverse with a typed error when none exists, CRT for pairs and lists including non-coprime moduli (returning the combined residue and lcm, or a proven inconsistency), the full solution set of a single linear congruence a*x = b (mod m), and systems of linear congruences in one unknown with mixed moduli. Property-test every function with hypothesis against brute force on small moduli and against sympy.ntheory.modular (dev-only oracle).

### Slice 2: Exact integer linear algebra core [x] done

Goal: Implement `trapforge.linalg` over Python ints: an immutable integer matrix helper layer (multiply, transpose, identity, fraction-free determinant, rank), Hermite normal form with its unimodular transform, Smith normal form with both unimodular transforms, integer kernel bases, and a linear Diophantine system solver returning a particular solution plus a kernel lattice basis (or a proof of infeasibility), with bounded enumeration of lattice points in a box. Property-test with hypothesis: U*A = H, U*A*V = S, divisibility chain of the SNF diagonal, |det U| = 1, and agreement with sympy's hermite_normal_form / smith_normal_form.

### Slice 3: Uniqueness prover and gate [x] done

Goal: Build `trapforge.prover`: a constraint model over bounded integer unknowns (linear equalities over Z, linear congruences mod m, box bounds, and a finite case split for discrete unknowns such as periods), an exact solver built on slices 1-2 that reduces congruences to Diophantine form and enumerates the remaining lattice inside the box, and a result type that is either a UniqueProof (the single parameterization plus a JSON certificate that can be re-checked independently) or an Ambiguity carrying a concrete counterexample pair and the exact or capped size of the ambiguity space. Provide a gate helper that rejects or extends a generated sample until the proof holds, and test it on hand-built unique and deliberately ambiguous systems.

### Slice 4: Task-family plugin API and the affine-scrambled ledger family [x] done

Goal: Define the plugin API: a TaskFamily protocol with generate(seed, difficulty) returning a TaskInstance (corpus files, hidden parameters, expected output bytes, visible sample, constraint system), a registry, a Difficulty enum, the canonical byte-exact writers, and determinism tests (same seed gives identical SHA-256 across processes). Ship the first original family on it: a ledger whose record IDs pass through an unknown affine map x -> (a*x + b) mod m with composite m, where the solver must recover (a, b) from anchor records via linear congruences to reconcile balances; include the reference solver, a naive baseline that solves from two anchors as if m were prime (right on the visible sample, wrong on hidden deciding records), and a test that every generated instance passes the uniqueness gate.

### Slice 5: Multi-clock log merge family

Goal: Add the second original family: events are stamped by devices whose counters wrap at unknown periods (from a bounded candidate set, not pairwise coprime) with unknown offsets, and sync beacons heard by several devices give congruences on the true time, so recovering the true global order needs CRT with non-coprime moduli. Include the generator with difficulty knobs (devices, periods, event count, wrap density), the constraint system for the prover (case split over candidate periods), a reference solver that unwraps and merges the logs into the byte-exact expected order, a naive baseline that sorts by locally unwrapped stamps and only breaks on hidden wraparound cases, and tests for determinism, gate pass, reference correctness and baseline failure.

### Slice 6: Warehouse conservation family

Goal: Add the third original family: a hidden non-negative integer transfer matrix between warehouses must be recovered from aggregate observations (opening and closing stock per site, per-lane manifest counts and weighted totals), which form a linear Diophantine system whose kernel is cut to a single point by box bounds and the weighted aggregates. Include the generator, the constraint system and uniqueness gate (regenerating aggregates until unique), a reference solver built on the SNF/kernel machinery plus bounded lattice search, a naive baseline (greedy northwest-corner fill that matches every net flow but not the hidden matrix), and tests for determinism, exactness and baseline failure.

### Slice 7: Bundle exporter, verify command and Typer CLI

Goal: Export any TaskInstance as a self-contained task directory: instruction.md, data/, a Dockerfile pinned to a python:3.12-slim image digest, solution/ with a standalone reference solver that vendors the pure-Python math modules (no installs needed), baseline/, and tests/test_outputs.py that grades byte-exactly against an embedded SHA-256. Add `trapforge verify` that copies a bundle to a temp dir, proves the reference passes and the baseline fails the grader locally, and repeats the check inside Docker (network disabled) when a daemon is available. Wire the Typer CLI: families, generate, prove, export, verify, with CliRunner tests and golden-file tests for the bundle layout.

### Slice 8: Docker, compose and end-to-end make demo [~] partly done (CLI image, make demo, CI image job; compose and bundle verification open)

Goal: Add a slim multi-stage Dockerfile for TrapForge itself (uv-built, non-root, labelled project=trapforge) and a docker-compose.yml with a forge service that generates and exports one bundle per family into a shared volume and a verify service that verifies them. Make `make demo` run the full pipeline end to end (generate, prove, export, verify locally, and verify in Docker when available) with a readable summary, add a CI job that builds the image and runs the demo, and add a docker-clean target that prunes only this project's images.

### Slice 9: Difficulty report, benchmarks and docs polish

Goal: Add `trapforge report`, which runs N seeds per family and difficulty and reports the naive baseline failure rate, the ambiguity-space size seen from the visible sample alone versus the full corpus, and generation and proof latency, as JSON plus a Markdown table. Add a small benchmark script for HNF/SNF/prover scaling, then finish the docs: README numbers taken only from commands that were run (with the command next to each number), docs/ARCHITECTURE.md, docs/AUTHORING.md explaining how to write a new task family against the plugin API, and a final pass on examples and badges.

## Decisions

- 2026-09-29: Built fresh instead of forking. The closest candidate (lan496/hsnf, MIT) computes
  HNF/SNF on numpy int arrays and uses scipy for triangular solves, which can overflow or go
  through floats; the spec calls for exact pure-Python integers, so reusing it would mean
  rewriting it. The task-framework half of the project has no suitable base.
- 2026-09-29: sympy is a dev-only dependency used as a test oracle; the runtime depends only on
  typer, so exported reference solvers can run in a bare python:3.12-slim image.
- 2026-09-29: Coverage gate set at 85% branch coverage in pyproject and enforced in CI.
- 2026-09-29 (slice 1): Inconsistent congruence systems return proof objects
  (`UnsolvableCongruence`, `ConflictingCongruences`, together `Inconsistency`) instead of
  raising, because the prover must report why a sample is inconsistent; each proof has a
  `verify(system)` that re-checks it with a gcd, a product and remainders, never by solving.
  Invalid input (modulus < 1, a non-invertible element) still raises a `ModularError`.
- 2026-09-29 (slice 1): `mod_inverse(a, 1)` returns 0 (Z/1Z is the zero ring), matching
  `pow(a, -1, 1)`; sympy refuses m == 1, so the sympy oracle test draws m >= 2.
- 2026-09-29 (slice 1): sympy's `crt()` reports the product of the moduli even when they
  share factors, so it is an oracle only for pairwise-coprime moduli; `solve_congruence()`
  is the oracle for the non-coprime case.
- 2026-09-29 (slice 1): Hypothesis uses a random `dev` profile locally and a derandomized
  `ci` profile in CI (`HYPOTHESIS_PROFILE=ci`), so CI failures always reproduce.
- 2026-09-29 (slice 2): `Matrix` is a frozen dataclass holding the tuple-of-int-tuples rows
  plus an explicit `ncols`, instead of a bare `tuple[tuple[int, ...], ...]`: a matrix with no
  rows (the kernel basis of an injective map, a system with no equations) must keep its
  width. Entries must be plain `int`; `Matrix.of` converts anything with `__index__` and
  rejects floats and Fractions.
- 2026-09-29 (slice 2): The Hermite normal form is row style (`U @ A == H`, leading entries
  positive, entries above a pivot reduced into `range(pivot)`). sympy's
  `hermite_normal_form` is column style with pivots at the bottom of each column, so the
  oracle test compares against it after transposing and reversing both the coordinate order
  and the basis order; HNF uniqueness makes that an exact comparison.
- 2026-09-29 (slice 2): `smith_normal_form` diagonalizes the Hermite form of `A` rather than
  `A` itself. On one seeded dense 10x10 matrix with entries up to 1000 this cut the largest
  entry of `V` from 539 digits to 31; a regression test bounds the transform size on five
  seeded dense matrices.
- 2026-09-29 (slice 2): Infeasible systems return an `UnsolvableSystem(w, d, rhs)`
  certificate (integer Fredholm alternative): `d` divides every entry of `w A` but not
  `w . b`, with `d = 0` for systems that have no rational solution. Rational failures are
  reported before divisibility failures, and divisibility certificates are shrunk to
  symmetric residues modulo `d`.
- 2026-09-29 (slice 2): Solution sets are `AffineLattice` values in canonical form (kernel
  basis in Hermite form, point reduced modulo it), so equality of sets is `==`.
  `points_in_box` walks the Hermite basis pivot by pivot, is always finite, and yields points
  in lexicographic order as a lazy iterator, so a caller that only needs to know whether a
  second point exists can stop after two with `itertools.islice`.
- 2026-09-29 (slice 2): Shared Hypothesis matrix strategies live in `tests/_strategies.py`
  (dense, sparse, low-rank products, repeated rows, random unimodular matrices), because
  uniformly random integer matrices are almost always full rank.
- 2026-09-29 (deliverable pass): Before slice 3 was finished, the CLI gained `crt`, `solve` and
  `system` over the completed layers, with bundled `examples/` inputs, and a CLI Dockerfile
  (python:3.12-slim and the uv image both pinned by digest, non-root, labelled
  project=trapforge) plus `make demo` / `make docker-demo`, so the repo is usable end to end
  today. The slice 7 commands (families, generate, prove, export, verify) are still planned.
- 2026-09-29 (deliverable pass): An unfinished standalone certificate checker left by an
  interrupted session was parked in a local git stash instead of being committed, because no
  code produces certificates yet; the solver work in slice 3 should start from it.
- 2026-09-29 (slice 3): The parked checker became `prover/certificate.py` (commit 81a1986).
  Its unused `_Case.width` property summed over an empty tuple and read the width off the
  first matrix row; it is now a field computed while the case is rebuilt, and every width
  check uses it. The checker also now rejects clashing unknown/choice names, repeated options,
  a non-bool `exact` and witnesses that are not the first solutions. The stash entry still
  exists locally but is fully superseded and should not be applied.
- 2026-09-29 (slice 3): A case becomes one system `M z = r` over `z = (unknowns, slacks)` with
  one slack per congruence (`a . x - m*s = b`). A kernel vector with zero unknown part must
  have zero slacks too (`m >= 1`), so every Hermite pivot of the kernel lies among the
  unknowns: truncating the lattice to the unknowns keeps it canonical and one-to-one, and
  `AffineLattice.points_in_box` enumerates it without bounding the slacks.
- 2026-09-29 (slice 3): Lattice evidence is designed to be cheaper to check than to find: an
  integer right inverse `W` of the kernel basis (first `k` columns of `V` times `U` from the
  basis's Smith form) proves saturation, and a nonsingular minor at the Hermite pivots of
  `M^T` (rows) and `M` (columns) proves the rank. The checker multiplies, takes one Bareiss
  determinant and walks the box; it never computes a normal form and imports only the
  standard library (a test parses its imports).
- 2026-09-29 (slice 3): `prove` has three verdicts, not two: `Infeasible` exists because a
  generated sample with no solution is a generator bug worth a certificate of its own.
  Counting stops at `cap + 1` (default 1000) across all cases; `count == cap` with
  `exact == False` means "more than cap". The two witnesses of an ambiguity are the first two
  worlds in case order, then lexicographic order of the unknowns, so they are deterministic.
- 2026-09-29 (slice 3): Certificates are written by one canonical writer (sorted keys, one
  entry per line, lists of plain values kept on one line so matrices read as matrices, final
  newline); `examples/ledger-anchors-unique.cert.json` is a golden file a test regenerates
  byte for byte.
- 2026-09-29 (slice 3): `gate` has prefix semantics: it appends candidates in stream order and
  stops at the shortest prefix that is unique, keeping candidates that did not cut anything,
  because a family reveals records in order and must map the result back to "the first k
  records". A planted world that breaks the sample, or a sample that turns infeasible, raises
  `GateError`. The module is `gating.py` because a `prover.gate` submodule would be shadowed
  by the `gate` function the package exports.
- 2026-09-29 (slice 3): The CLI gained `prove` and `check` over a constraint-system file now,
  with exit code 1 for a failed check (`check` on a bad certificate, `prove
  --require-unique` on a non-unique sample); the slice 7 `prove` for generated instances
  should build on this command rather than add another.
- 2026-09-29 (slice 4): The canonical writers live in `trapforge/canonical.py`, not under
  `families/`, because the slice 7 bundle exporter writes through them too. CSV cells that
  would need quoting (comma, quote, newline, surrounding whitespace, non-ASCII) are rejected
  rather than quoted, so the byte format has exactly one spelling and the strict reader is a
  plain split. JSON rejects floats anywhere in the value.
- 2026-09-29 (slice 4): `TaskInstance` validates itself on construction: the planted world
  must satisfy the instance's constraint system, every path must be a normalized relative
  POSIX path, and every file must be bytes. `bundle()` lays it out as `data/`, `expected/`,
  `sample/` and `meta/` (instruction, task.json with the planted world and extras,
  system.json in the prover's format), and `digest()` is SHA-256 over that bundle with
  length-prefixed names and contents, so no two different bundles share an encoding.
- 2026-09-29 (slice 4): Families draw every random number from `family_rng`, a
  `random.Random` seeded with the string `trapforge:<family>:<difficulty>:<seed>` (Python
  hashes string seeds with SHA-512, independent of `PYTHONHASHSEED`). Tests compare digests
  across subprocesses with two hash seeds and pin the seed-0 digest of every difficulty, so a
  Linux CI run must reproduce the bytes generated on macOS; the Docker demo printed the same
  digest as the laptop.
- 2026-09-29 (slice 4): Affine ledger design. The first two anchors differ by `d * k` with
  `d | m` and `gcd(k, m/d) = 1`, so exactly `d` maps fit them; the planted `a` is a unit
  drawn from `[m/d, m)`, so the smallest solution of the two-anchor congruence (what the
  naive baseline keeps) is never the true one. Decoy anchors are drawn from the class of
  `x1` mod `d`, where all `d` maps agree, and come first in the gate's candidate stream; the
  gate's prefix semantics keep them. Queries are split into accounts where the naive and true
  maps agree (the visible sample) and accounts where they disagree (hidden deciding records),
  and transactions are appended until every deciding record has a different balance under
  the two maps, so the baseline is wrong on each of them, not only on its account number.
- 2026-09-29 (slice 4): "a is coprime to m" is not a linear constraint, so the prover's box
  is `1 <= a <= m - 1` and uniqueness is proved over a superset of the admissible maps. The
  reference solver solves the difference congruences with `solve_linear_system` and keeps
  only units; with a unique proof over the superset, exactly one survives.
- 2026-09-29 (slice 4): The CLI gained `families` and `generate` now (generate proves the
  instance, runs the reference and the baseline, prints the digest, and exits 1 when any of
  the three checks fails; `--out` writes the bundle). Slice 7 adds `export` and `verify` on
  top of these instead of new generate commands.
