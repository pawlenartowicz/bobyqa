#!/usr/bin/env bash
# oracle/capture_states.sh - deliberate state-capture campaign.
# Instruments the gitignored PRIMA clone, builds an instrumented tree, runs the
# corpus problems, post-processes dumps into tests/states/, de-instruments.
# The corpus is FROZEN - rerunning this is a deliberate campaign.
set -euo pipefail
cd "$(dirname "$0")"

STATES=../tests/states
HASH=$(git -C prima rev-parse HEAD)

# 1. Instrument (idempotent: skip if the patch is already applied). The dry run refuses a clone
#    the patch does not fit before touching it. Once the clone is patched, the trap
#    de-instruments on every exit, so a failed build or run cannot leave it patched for a later
#    rebuild of the clean build/ to pick up.
deinstrument() {
    rm -f prima/fortran/common/dump.f90
    patch -R -p1 -d prima <instrument.patch
}
if patch -p1 -d prima --dry-run --reverse <instrument.patch >/dev/null 2>&1; then
    echo "instrument.patch already applied" >&2
else
    patch -p1 -d prima --dry-run <instrument.patch >&2
    patch -p1 -d prima <instrument.patch
fi
trap deinstrument EXIT
cp dump.f90 prima/fortran/common/dump.f90

# 2. Build the instrumented tree (separate from the clean build/).
cmake -S prima -B build-instr -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_Fortran_FLAGS="-ffp-contract=off"
cmake --build build-instr --parallel

# 3. Compile the driver against the instrumented libs.
CC=${CC:-cc}
CFLAGS="-O2 -ffp-contract=off -Wall -Wextra"
# shellcheck disable=SC2086
$CC $CFLAGS driver.c -Iprima/c/include \
    "$(find build-instr -name libprimac.a -print -quit)" \
    "$(find build-instr -name libprimaf.a -print -quit)" -lgfortran -lm -o driver-instr

# 4. Campaign runs. keep=10 for each problem's default config (all if <=10 calls,
#    else first 5 + uniform to 10); keep=3 for the extra
#    branch-coverage / stress configs so the corpus stays near the ~2 MB target.
run() { # run <keep> <tag> <problem> [driver flags...]
    local keep=$1 tag=$2 problem=$3
    shift 3
    : >bobyqa_states.dump
    local out
    out=$(./driver-instr "$problem" "$@")
    local npt rhobeg rhoend maxfun
    npt=$(awk '$1 == "npt" { print $2; exit }' <<<"$out")
    rhobeg=$(awk '$1 == "rho_begin" { print $2; exit }' <<<"$out")
    rhoend=$(awk '$1 == "rho_end" { print $2; exit }' <<<"$out")
    maxfun=$(awk '$1 == "max_fun" { print $2; exit }' <<<"$out")
    python3 states.py bobyqa_states.dump --problem "$problem" --npt "$npt" \
        --rhobeg "$rhobeg" --rhoend "$rhoend" --maxfun "$maxfun" \
        --prima "$HASH" --outdir "$STATES" --keep "$keep" --tag "$tag"
}

mkdir -p "$STATES"
run 10 ''     sphere
run 3  ''     sphere --npt 4              # npt = n+2: ndiag < n branches in prelim
run 3  ''     sphere --npt 6              # npt = (n+1)(n+2)/2: ij paths in prelim
run 10 ''     rosenbrock
run 10 ''     booth
run 3  're12' booth --rhoend 1e-12        # rescue-trigger hunt
run 10 ''     rosenbrock10
run 3  ''     rosenbrock10 --npt 30       # ij paths at n = 10
run 3  're12' rosenbrock10 --rhoend 1e-12 # rescue-trigger hunt

# 5. Rescue verdict.
if [[ -d "$STATES/rescue" ]]; then
    echo "RESCUE TRIGGERED: states captured in $STATES/rescue" >&2
else
    echo "NO rescue states captured" >&2
fi

: >bobyqa_states.dump
du -sh "$STATES" >&2
