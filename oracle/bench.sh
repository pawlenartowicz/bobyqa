#!/usr/bin/env bash
# oracle/bench.sh — PRIMA vs this crate vs basin, wall-clock over the golden registry.
#
# Every contender runs the identical solve (same problem, same npt/rho/maxfun, same
# objective arithmetic) and uses the identical measurement rule: one warm-up pass
# discarded, then repeat for TIME seconds and report the minimum. Because the
# port is bit-exact with PRIMA, those two walk the same trajectory and spend the same
# number of objective evaluations -- so the difference is solver overhead alone.
#
# basin's BOBYQA is not bit-exact with PRIMA: it walks its own trajectory, so its
# evaluation count and final f differ and its time is printed with its own nf/f
# beside it. A basin time is only comparable to the others once that nf is read too.
#
# Pin to one performance core (CORE) and lock the CPU clock before trusting
# the numbers.
set -euo pipefail
cd "$(dirname "$0")"

TIME=${TIME:-2.0}
CORE=${CORE:-2}
PRIMA=${PRIMA:-./driver}            # PRIMA's default build (heap arrays)
PRIMA_STACK=${PRIMA_STACK:-./driver-stack}  # same, built -DPRIMA_HEAP_ARRAYS=OFF
RUST=${RUST:-./rusttime/target/release/rusttime}
BASIN=${BASIN:-./rusttime/target/release/basintime}

# problem + flags; the two `--rhoend 1e-12` entries are the rescue-path stressors.
# sphere_onbound is deliberately absent: PRIMA emits a preproc warning to stderr
# on every call for it, and that formatting lands inside the timed region.
CASES=(
    "sphere"
    "rosenbrock"
    "booth"
    "beale"
    "sphere_tight"
    "powell_singular"
    "rosenbrock10 --npt 21"
    "booth --rhoend 1e-12"
    "rosenbrock10 --npt 21 --rhoend 1e-12"
)

printf '%-34s %6s %11s %11s %11s %11s %7s %7s %11s %7s %7s %12s\n' \
       case nf prima_heap prima_stack rust_cold rust_warm vs_heap vs_stack \
       basin basin_nf vs_basin basin_f

for case in "${CASES[@]}"; do
    # shellcheck disable=SC2086 # $case carries intentional flag words
    p_out=$(taskset -c "$CORE" $PRIMA $case --time "$TIME" 2>/dev/null)
    # shellcheck disable=SC2086
    s_out=$(taskset -c "$CORE" $PRIMA_STACK $case --time "$TIME" 2>/dev/null)
    # shellcheck disable=SC2086
    r_out=$(taskset -c "$CORE" $RUST $case --time "$TIME" 2>/dev/null)
    # shellcheck disable=SC2086
    b_out=$(taskset -c "$CORE" $BASIN $case --time "$TIME" 2>/dev/null)

    field() { awk -v k="$2" '{for (i=1;i<NF;++i) if ($i==k) print $(i+1)}' <<<"$1"; }

    p_ns=$(field "$p_out" min_ns)
    s_ns=$(field "$s_out" min_ns)
    s_nf=$(field "$s_out" nf)
    p_f=$(field "$p_out" f)
    p_nf=$(field "$p_out" nf)
    r_cold=$(field "$r_out" cold_min_ns)
    r_warm=$(field "$r_out" warm_min_ns)
    r_f=$(field "$r_out" f)
    r_nf=$(field "$r_out" nf)
    b_ns=$(field "$b_out" cold_min_ns)
    b_nf=$(field "$b_out" nf)
    b_f=$(field "$b_out" f)

    # Same trajectory or the timing compares different work -- fail loudly.
    if [[ "$p_nf" != "$r_nf" || "$p_nf" != "$s_nf" ]]; then
        echo "MISMATCH $case: prima nf=$p_nf f=$p_f / stack nf=$s_nf / rust nf=$r_nf f=$r_f" >&2
        exit 1
    fi

    awk -v c="$case" -v nf="$p_nf" -v p="$p_ns" -v sp="$s_ns" -v rc="$r_cold" -v rw="$r_warm" \
        -v b="$b_ns" -v bnf="$b_nf" -v bf="$b_f" \
        'BEGIN { printf "%-34s %6s %11.0f %11.0f %11.0f %11.0f %6.2fx %6.2fx %11.0f %7s %6.2fx %12.3g\n", \
                        c, nf, p, sp, rc, rw, p/rw, sp/rw, b, bnf, b/rw, bf }'
done
