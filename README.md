# TrapForge

[![CI](https://github.com/vipul21435/trapforge/actions/workflows/ci.yml/badge.svg)](https://github.com/vipul21435/trapforge/actions/workflows/ci.yml)

**Author verifiable, adversarial data tasks for AI agents - and prove they have exactly one right answer.**

Most "hard" data tasks for agents are hard for the wrong reason: the instructions are
ambiguous, several answers are defensible, and the grader quietly rewards whichever one the
author happened to pick. TrapForge is being built to take the opposite approach: every task
ships with a machine-checked argument that its data admits exactly one interpretation. The
target pipeline is a synthetic corpus with hidden integer structure, an exact constraint
system built from what the corpus reveals, a uniqueness proof (or a concrete counterexample
pair of hidden worlds), a reference solver, a naive baseline that fails on hidden deciding
cases, and a byte-exact pytest grader.

**What exists today** is the exact math that pipeline stands on, the prover's constraint
model, and a CLI over them. Everything is pure Python 3.12 integers (no floats, no numpy),
implemented from scratch and property-tested against `sympy`, which is a dev-only oracle.
The task families, the solver/gate and the bundle exporter are on the [Roadmap](#roadmap).

## Why this exists

This project generalizes the kind of work I do building benchmark tasks for AI coding agents
at an AI-data company: reproducible environments, reference solutions that really compute the
answer, byte-exact graders, and synthetic datasets that are provably unambiguous. All task
designs, examples and data here are original.

## Features (implemented and tested)

| Area | Module | What it gives you |
| --- | --- | --- |
| Modular arithmetic | `trapforge.modular` | extended gcd, modular inverse with typed errors, CRT for **non-coprime** moduli, full solution sets of `a*x = b (mod m)` and of mixed-moduli systems, and re-checkable inconsistency proofs |
| Integer linear algebra | `trapforge.linalg` | immutable `Matrix` with Bareiss determinant and rank, Hermite and Smith normal forms with unimodular transforms, saturated integer kernels, `solve_diophantine` returning an `AffineLattice` or an `UnsolvableSystem` certificate, exact lexicographic enumeration of lattice points in a box |
| Prover constraint model | `trapforge.prover` | bounded integer unknowns, linear equations, linear congruences and a finite case split over discrete choices (for example an unknown counter period), with validation, case instantiation, violation reports and a JSON round trip |
| CLI | `trapforge` | `crt`, `solve` and `system` commands over the three layers above, plus `info` |
| Container | `Dockerfile` | multi-stage uv build on `python:3.12-slim` pinned by digest, non-root user, `LABEL project=trapforge` |
| Demo | `make demo` | `scripts/demo.sh` runs the CLI over the bundled `examples/` inputs, locally or in the image |

Every "no solution" answer comes with a certificate that can be re-checked without trusting
the solver: a conflicting pair of congruences and a witness modulus, or integer weights `w`
and a modulus `d` such that `d` divides every entry of `w A` but not `w . b`.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and make. No GPU, no API keys, no network at run time.

```bash
git clone https://github.com/vipul21435/trapforge.git
cd trapforge
uv sync        # locked runtime + dev dependencies
make check     # ruff, mypy --strict, pytest with the 85% branch-coverage gate
make demo      # the CLI end to end on examples/
```

These five commands were run in a fresh clone: `make check` reported 245 passed at 100%
branch coverage, and the whole sequence (clone, sync, check, demo) took 50 s wall time on an
Apple-silicon laptop. With Docker, `make docker-demo` builds the image and runs the same demo
inside it.

## CLI usage

The command list from `trapforge --help` (Typer draws it in a box; the frame is dropped here):

```text
Usage: trapforge [OPTIONS] COMMAND [ARGS]...
Commands:
  info    Print the package version and the Python it runs on.
  crt     Combine congruences with any moduli (coprime or not) into one residue class.
  solve   Solve an integer system A x = b exactly and list its solutions inside a box.
  system  Validate and print a constraint system; optionally test a candidate assignment.
```

Exit code 0 means the input was valid (with or without a solution); 2 means it could not be
read or parsed. The outputs below are copied from `make demo`.

**`crt`** combines congruences whose moduli share factors, or proves they conflict:

```text
$ trapforge crt 5:12 5:18
#0: x = 5 (mod 12)
#1: x = 5 (mod 18)
combined: x = 5 (mod 36)

$ trapforge crt 1:4 2:6
#0: x = 1 (mod 4)
#1: x = 2 (mod 6)
no solution: congruences #0 and #1 are incompatible: #0 forces x = 1 (mod 2) but #1 forces x = 0 (mod 2)
certificate re-checked: True
```

**`solve`** reads `{"matrix", "rhs", "lower", "upper"}` from JSON, prints the whole integer
solution lattice and the points inside the box, and says whether the box pins down one point:

```text
$ trapforge solve examples/checksums.json
system: 2 equations, 3 unknowns
integer solutions: x = (4, 2, 7) + t0*(290, -699, 67), t in Z^1
  (4, 2, 7)
box (0, 0, 0)..(9, 9, 9): exactly one point in the box (unique)

$ trapforge solve examples/checksums-ambiguous.json --limit 3
system: 1 equation, 3 unknowns
integer solutions: x = (0, 0, 13) + t0*(1, 0, -1) + t1*(0, 1, -1), t in Z^2
  (0, 4, 9)
  (0, 5, 8)
  (0, 6, 7)
box (0, 0, 0)..(9, 9, 9): more than 3 points in the box (ambiguous)

$ trapforge solve examples/parity.json
system: 2 equations, 2 unknowns
no integer solution: weighting the equations by (0, 1) makes every coefficient a multiple of 2, but the right-hand side becomes 1, which is not
certificate re-checked: True
```

**`system`** loads a constraint system in the prover's JSON format and tests candidate
hidden worlds against it. The bundled ledger example is the trap the planned affine-ledger
family is built around: with a composite modulus, two different affine maps fit every anchor.

```text
$ trapforge system examples/ledger-anchors.json --check a=5 --check b=7
unknowns: 1 <= a <= 11, 0 <= b <= 11
anchor x=1: a + b = 0 (mod 12)
anchor x=3: 3*a + b = 10 (mod 12)
2 unknowns, 2 constraints, 1 case
a=5, b=7 satisfies every constraint

$ trapforge system examples/ledger-anchors.json --check a=11 --check b=1
...
a=11, b=1 satisfies every constraint

$ trapforge system examples/wrapping-clock.json --check P=18 --check T=41 --check w=2
choices: P in {12, 18}
unknowns: 0 <= T <= 100, 0 <= w <= 8
beacon: T - P*w = 5
sync: T = 5 (mod 36)
2 unknowns, 2 constraints, 2 cases
P=18, T=41, w=2 satisfies every constraint
```

## Library usage

```python
from trapforge.modular import Congruence, LinearCongruence, crt, solve_linear_system

# Two counters that wrap at 12 and 18 both read 5: when can that happen?
print(crt([Congruence(5, 12), Congruence(5, 18)]))

# An affine scramble 6*x + 4 (mod 10) produced 0. Which x mod 10 are possible?
scramble = LinearCongruence(6, -4, 10)
print(list(scramble.solve().residues_mod(10)))

# One more fact (x is odd) pins x down; a contradicting fact yields a checkable proof.
print(solve_linear_system([scramble, Congruence(1, 2)]))
system = [scramble, Congruence(2, 10)]
proof = solve_linear_system(system)
print(proof)
print(proof.verify(system))
```

```text
x = 5 (mod 36)
[1, 6]
x = 1 (mod 10)
congruences #0 and #1 are incompatible: #0 forces x = 1 (mod 5) but #1 forces x = 2 (mod 5)
True
```

```python
from trapforge.linalg import Matrix, smith_normal_form, solve_diophantine

# Three hidden digits, observed only through two weighted checksums.
A = Matrix.of([[1, 10, 100], [7, 3, 1]])
solutions = solve_diophantine(A, [724, 41])
print(solutions)
print(list(solutions.points_in_box((0, 0, 0), (9, 9, 9))))

# Smith normal form with both transforms: U @ C @ V == S.
C = Matrix.of([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])
form = smith_normal_form(C)
print(form.invariant_factors, form.U @ C @ form.V == form.S)
```

```text
x = (4, 2, 7) + t0*(290, -699, 67), t in Z^1
[(4, 2, 7)]
(2, 6, 12) True
```

Both outputs come from running the snippets with `uv run python`; the module docstrings carry
more examples, and `tests/test_doctests.py` runs all of them so they cannot drift.

## Architecture

```mermaid
flowchart LR
    subgraph today["implemented"]
        MOD["modular<br/>gcd, inverse, CRT,<br/>linear congruences"]
        LIN["linalg<br/>Matrix, HNF, SNF,<br/>kernel, Diophantine,<br/>box enumeration"]
        MODEL["prover.model<br/>unknowns, equations,<br/>congruences, case split"]
        CLI["cli (Typer)<br/>crt, solve, system"]
        EX[("examples/*.json")]
    end
    subgraph planned["roadmap"]
        SOLVER["prover solver + gate<br/>UniqueProof / Ambiguity<br/>JSON certificates"]
        FAM["task families<br/>ledger, clocks, warehouse"]
        BUNDLE["bundle exporter<br/>+ verify in Docker"]
    end
    MOD --> CLI
    LIN --> CLI
    MODEL --> CLI
    EX --> CLI
    MOD -.-> SOLVER
    LIN -.-> SOLVER
    MODEL -.-> SOLVER
    SOLVER -.-> FAM
    FAM -.-> BUNDLE
```

```
src/trapforge/
  modular.py            number theory over Python int
  linalg/               matrix.py, hermite.py, smith.py, lattice.py, diophantine.py
  prover/model.py       the constraint model the uniqueness prover will reason about
  cli.py                Typer CLI
examples/               bundled JSON inputs for the CLI and the demo
scripts/demo.sh         the end-to-end demo, runnable locally or in the image
```

## Measured numbers

Every number here comes from a command in this repo, run on the current tree.

| Number | Value | Command |
| --- | --- | --- |
| Tests | 245 passed (31 CLI, 1 demo script, 7 doctest modules, 76 linalg, 89 modular, 41 prover model) | `make cov` and `uv run pytest -q --co` |
| Branch coverage of `src/` | 100% (1057 statements, 332 branches); CI fails under 85% | `make cov` |
| Demo wall time | 0.86 s for all 9 steps | `time make demo` |
| Image size | 47.6 MB content size (223 MB unpacked on disk) | `make docker-build && docker images trapforge` |
| Source and test size | 2236 lines in `src/`, 2102 lines in `tests/` | `git ls-files src \| xargs wc -l`, same for `tests` |

The property tests use Hypothesis against brute force on small inputs and against `sympy`
on large ones (moduli up to 10^30, 30-digit planted solutions, dense 10x10 matrices), and
tampered certificates must always be rejected. CI runs a derandomized profile
(`HYPOTHESIS_PROFILE=ci`) so any failure reproduces.

## Design decisions

- **Exact integers only.** Every value is a Python `int`; `Matrix.of` rejects floats and
  Fractions. A silently truncated 2.5 would break a uniqueness proof.
- **Proofs instead of "no".** Inconsistent systems return certificate objects with a
  `verify` method that re-checks them with a gcd or one vector-matrix product, never by
  re-solving. The CLI prints the re-check next to every certificate.
- **Canonical solution sets.** An `AffineLattice` keeps its kernel basis in Hermite normal
  form and its point reduced modulo it, so equal sets compare equal and box enumeration is
  exact, lazy and lexicographic.
- **SNF on top of HNF.** `smith_normal_form` diagonalizes the Hermite form of `A` rather than
  `A`; on one seeded dense 10x10 matrix this cut the largest entry of `V` from 539 digits to
  31, and a regression test bounds transform size.
- **sympy is a test oracle, not a dependency.** The runtime depends only on Typer, so a
  future exported reference solver can vendor the math modules into a bare
  `python:3.12-slim` image.
- **Reproducible builds.** `uv.lock` is enforced with `uv sync --locked` in CI and in the
  Dockerfile, and both base images are pinned by digest.

The full decision log lives in [PLAN.md](PLAN.md).

## Roadmap

Built in the slices listed in [PLAN.md](PLAN.md). Slices 1 and 2 are done; slice 3 is
partly done (the constraint model).

- **Slice 3, rest: solver and uniqueness gate.** An exact solver that reduces each case to a
  Diophantine system, enumerates the lattice in the box, and returns either a unique
  parameterization with a JSON certificate that a small standalone checker can re-verify, or
  an ambiguity with a concrete counterexample pair; plus a gate that extends a sample until
  the proof holds.
- **Slice 4: plugin API and the affine-scrambled ledger family** (the trap shown by
  `examples/ledger-anchors.json`), with a reference solver and a naive baseline.
- **Slice 5: multi-clock log merge family** (wrapping counters with unknown periods, CRT with
  non-coprime moduli).
- **Slice 6: warehouse conservation family** (a hidden transfer matrix behind aggregates).
- **Slice 7: bundle exporter and `trapforge verify`** (instruction, data, digest-pinned
  Dockerfile, standalone reference solution, byte-exact grader; reference passes and baseline
  fails, locally and in Docker).
- **Slice 8: compose pipeline** that forges and verifies one bundle per family.
- **Slice 9: difficulty report and benchmarks** (baseline failure rates, ambiguity-space
  size, generation and proof latency) and authoring docs.

## License

MIT - see [LICENSE](LICENSE).
