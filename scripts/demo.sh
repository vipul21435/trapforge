#!/usr/bin/env sh
# End-to-end tour of the TrapForge CLI on the bundled examples/ inputs.
# TRAPFORGE is the command to run; `make demo` uses the local uv environment and
# `make docker-demo` runs the same steps inside the image.
set -eu

TRAPFORGE=${TRAPFORGE:-"uv run trapforge"}

step() {
    printf '\n== %s\n$ trapforge %s\n' "$1" "$2"
}

run() {
    title=$1
    shift
    step "$title" "$*"
    # shellcheck disable=SC2086
    $TRAPFORGE "$@"
}

run "installed version" --version
run "two counters that wrap at 12 and 18 both read 5 (non-coprime CRT)" crt 5:12 5:18
run "contradictory readings come back with a re-checked certificate" crt 1:4 2:6
run "three hidden digits behind two weighted checksums: unique in the box" \
    solve examples/checksums.json
run "one checksum alone leaves the digits ambiguous" \
    solve examples/checksums-ambiguous.json --limit 3
run "even coefficients can never produce an odd total: proof of infeasibility" \
    solve examples/parity.json
run "affine ledger anchors mod 12: one candidate world" \
    system examples/ledger-anchors.json --check a=5 --check b=7
run "the composite modulus admits a second world that fits every anchor" \
    system examples/ledger-anchors.json --check a=11 --check b=1
run "a clock with an unknown period is a finite case split" \
    system examples/wrapping-clock.json --check P=18 --check T=41 --check w=2
run "the prover finds the counterexample pair behind the ledger anchors" \
    prove examples/ledger-anchors.json
run "a third anchor pins the affine map down, with a certificate" \
    prove examples/ledger-anchors-unique.json
run "the case split over the period leaves six worlds" \
    prove examples/wrapping-clock.json
run "re-check the bundled certificate without running the solver" \
    check examples/ledger-anchors-unique.cert.json

printf '\ndemo finished: every step above ran on the bundled examples/ inputs\n'
