#!/usr/bin/env bash
# oracle/capture.sh — compile driver.c against the local PRIMA build and
# (re)capture the golden set into tests/goldens/.
# Re-running regenerates the goldens: this is the "deliberate campaign" lever
# of the freeze-then-forget strategy (see README.md).
set -euo pipefail
cd "$(dirname "$0")"

GOLDENS=${GOLDENS:-../tests/goldens}
BIN=${BIN:-driver}
BUILD=${BUILD:-build}

CC=${CC:-cc}
# -ffp-contract=off: no FMA contraction — the driver objectives must stay
# bit-identical to the Rust test objectives in tests/parity_prima.rs.
CFLAGS="-O2 -ffp-contract=off -Wall -Wextra"

LIBPRIMAC=$(find "$BUILD" -name 'libprimac.a' -print -quit)
LIBPRIMAF=$(find "$BUILD" -name 'libprimaf.a' -print -quit)
if [[ -z "$LIBPRIMAC" || -z "$LIBPRIMAF" ]]; then
    echo "PRIMA static libs not found under oracle/$BUILD/ — build PRIMA first (see README.md)" >&2
    exit 1
fi

# shellcheck disable=SC2086 # CFLAGS is intentionally word-split
$CC $CFLAGS driver.c -Iprima/c/include "$LIBPRIMAC" "$LIBPRIMAF" -lgfortran -lm -o "$BIN"

HASH=$(git -C prima rev-parse HEAD)
# The compiler that built the linked Fortran library, not whatever gfortran is installed now.
COMPILER=$(strings -a "$LIBPRIMAF" | grep '^GCC: ' | sed -E '1!d; s/^GCC: \(GNU\) /GNU Fortran (GCC) /' || true)
if [[ -z "$COMPILER" ]]; then
    echo "no GCC version string in $LIBPRIMAF" >&2
    exit 1
fi

capture() { # capture <file-stem> <problem> [driver flags...]
    local stem=$1
    shift
    local out n npt file
    out=$(./"$BIN" "$@")
    n=$(awk '$1 == "n" { print $2; exit }' <<<"$out")
    npt=$(awk '$1 == "npt" { print $2; exit }' <<<"$out")
    file="$GOLDENS/${stem}_n${n}_npt${npt}.txt"
    {
        echo "# bobyqa golden v1"
        echo "# prima $HASH"
        echo "# compiler $COMPILER"
        printf '%s\n' "$out"
    } >"$file"
    echo "captured $file" >&2
}

mkdir -p "$GOLDENS"
capture sphere sphere
capture rosenbrock rosenbrock
capture booth booth
capture rosenbrock10 rosenbrock10
capture beale beale
capture powell_singular powell_singular
capture sphere_onbound sphere_onbound
capture sphere_tight sphere_tight
capture nansphere nansphere
capture sphere sphere --npt 4
capture sphere sphere --npt 6
capture powell_singular powell_singular --npt 15
capture booth_rescue booth --rhoend 1e-12
capture rosenbrock10_rescue rosenbrock10 --rhoend 1e-12
capture quad1 quad1
capture quad1_lower quad1_lower
capture quartic1 quartic1
capture reml1 reml1
