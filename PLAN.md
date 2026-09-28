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
  prover/             constraint model, exact solver, uniqueness certificate / counterexample
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

### Slice 2: Exact integer linear algebra core

Goal: Implement `trapforge.linalg` over Python ints: an immutable integer matrix helper layer (multiply, transpose, identity, fraction-free determinant, rank), Hermite normal form with its unimodular transform, Smith normal form with both unimodular transforms, integer kernel bases, and a linear Diophantine system solver returning a particular solution plus a kernel lattice basis (or a proof of infeasibility), with bounded enumeration of lattice points in a box. Property-test with hypothesis: U*A = H, U*A*V = S, divisibility chain of the SNF diagonal, |det U| = 1, and agreement with sympy's hermite_normal_form / smith_normal_form.

### Slice 3: Uniqueness prover and gate

Goal: Build `trapforge.prover`: a constraint model over bounded integer unknowns (linear equalities over Z, linear congruences mod m, box bounds, and a finite case split for discrete unknowns such as periods), an exact solver built on slices 1-2 that reduces congruences to Diophantine form and enumerates the remaining lattice inside the box, and a result type that is either a UniqueProof (the single parameterization plus a JSON certificate that can be re-checked independently) or an Ambiguity carrying a concrete counterexample pair and the exact or capped size of the ambiguity space. Provide a gate helper that rejects or extends a generated sample until the proof holds, and test it on hand-built unique and deliberately ambiguous systems.

### Slice 4: Task-family plugin API and the affine-scrambled ledger family

Goal: Define the plugin API: a TaskFamily protocol with generate(seed, difficulty) returning a TaskInstance (corpus files, hidden parameters, expected output bytes, visible sample, constraint system), a registry, a Difficulty enum, the canonical byte-exact writers, and determinism tests (same seed gives identical SHA-256 across processes). Ship the first original family on it: a ledger whose record IDs pass through an unknown affine map x -> (a*x + b) mod m with composite m, where the solver must recover (a, b) from anchor records via linear congruences to reconcile balances; include the reference solver, a naive baseline that solves from two anchors as if m were prime (right on the visible sample, wrong on hidden deciding records), and a test that every generated instance passes the uniqueness gate.

### Slice 5: Multi-clock log merge family

Goal: Add the second original family: events are stamped by devices whose counters wrap at unknown periods (from a bounded candidate set, not pairwise coprime) with unknown offsets, and sync beacons heard by several devices give congruences on the true time, so recovering the true global order needs CRT with non-coprime moduli. Include the generator with difficulty knobs (devices, periods, event count, wrap density), the constraint system for the prover (case split over candidate periods), a reference solver that unwraps and merges the logs into the byte-exact expected order, a naive baseline that sorts by locally unwrapped stamps and only breaks on hidden wraparound cases, and tests for determinism, gate pass, reference correctness and baseline failure.

### Slice 6: Warehouse conservation family

Goal: Add the third original family: a hidden non-negative integer transfer matrix between warehouses must be recovered from aggregate observations (opening and closing stock per site, per-lane manifest counts and weighted totals), which form a linear Diophantine system whose kernel is cut to a single point by box bounds and the weighted aggregates. Include the generator, the constraint system and uniqueness gate (regenerating aggregates until unique), a reference solver built on the SNF/kernel machinery plus bounded lattice search, a naive baseline (greedy northwest-corner fill that matches every net flow but not the hidden matrix), and tests for determinism, exactness and baseline failure.

### Slice 7: Bundle exporter, verify command and Typer CLI

Goal: Export any TaskInstance as a self-contained task directory: instruction.md, data/, a Dockerfile pinned to a python:3.12-slim image digest, solution/ with a standalone reference solver that vendors the pure-Python math modules (no installs needed), baseline/, and tests/test_outputs.py that grades byte-exactly against an embedded SHA-256. Add `trapforge verify` that copies a bundle to a temp dir, proves the reference passes and the baseline fails the grader locally, and repeats the check inside Docker (network disabled) when a daemon is available. Wire the Typer CLI: families, generate, prove, export, verify, with CliRunner tests and golden-file tests for the bundle layout.

### Slice 8: Docker, compose and end-to-end make demo

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
