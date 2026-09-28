# TrapForge

[![CI](https://github.com/vipul21435/trapforge/actions/workflows/ci.yml/badge.svg)](https://github.com/vipul21435/trapforge/actions/workflows/ci.yml)

**Author verifiable, adversarial data tasks for AI agents - and prove they have exactly one right answer.**

Most "hard" data tasks for agents are hard for the wrong reason: the instructions are
ambiguous, several answers are defensible, and the grader quietly rewards whichever one the
author happened to pick. TrapForge takes the opposite approach. Every task it produces ships
with a machine-checked argument that the data admits exactly one interpretation:

1. **A synthetic corpus with hidden integer structure** (an unknown affine map, unknown
   wrapping clocks, an unknown transfer matrix), generated deterministically from a seed.
2. **A uniqueness proof.** The observed data is turned into an exact constraint system and
   solved over the integers. Either exactly one hidden parameterization is consistent - the
   task passes the gate - or TrapForge reports the ambiguity with a concrete counterexample:
   two different hidden worlds that both reproduce every observation.
3. **A reference solver that computes the answer genuinely** from the corpus (no lookup of
   the hidden parameters), and a **naive baseline** that looks right on the visible sample
   but fails on hidden deciding cases - so the task measures reasoning, not luck.
4. **A byte-exact pytest grader** and an exportable task bundle (`instruction.md`, `data/`,
   a digest-pinned `Dockerfile`, the reference solution and the grader), plus a `verify`
   command that proves the reference passes and the baseline fails, locally and in Docker.

Everything is pure Python 3.12: the exact integer linear algebra (Hermite and Smith normal
forms, integer kernels, linear Diophantine systems) and the modular arithmetic (CRT with
non-coprime moduli, systems of linear congruences) are implemented from scratch and
property-tested against `sympy`, which is used only as a dev-time oracle.

## Why this exists

This project generalizes the kind of work I do building benchmark tasks for AI coding agents
at an AI-data company: reproducible environments, reference solutions that really compute the
answer, byte-exact graders, and synthetic datasets that are provably unambiguous. All task
families here are original designs; none are derived from client tasks or data.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and GNU make. No GPU, no API keys.

```bash
git clone https://github.com/vipul21435/trapforge.git
cd trapforge
make install   # uv sync + pre-commit hooks
make check     # ruff, mypy --strict, pytest with coverage gate
make demo      # end-to-end demo of the CLI
```

## Planned task families

| Family | Hidden structure | Why the naive approach fails |
| --- | --- | --- |
| Affine-scrambled ledger | Record IDs pass through an unknown affine map mod m | Solving from two anchors assumes m is prime and picks the wrong branch |
| Multi-clock log merge | Devices stamp events with wrapping counters of unknown period and offset | Sorting by raw stamps ignores wraparound; the true order needs CRT |
| Warehouse conservation | An integer transfer matrix behind aggregate counts | A proportional or greedy fill matches the totals but not the matrix |

## Status

Under active development. The build is organized into the slices in [PLAN.md](PLAN.md);
this README only reports numbers that come from commands in the repo.

## License

MIT - see [LICENSE](LICENSE).
