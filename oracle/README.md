# `oracle/` — the PRIMA differential-testing oracle

PRIMA (libprima, BSD-3-Clause) built from source; its C interface
(`prima_minimize`) is the differential-testing oracle for the Rust port.
The PRIMA clone (`prima/`), the build trees (`build*/`), the compiled drivers
(`driver`, `driver-*`), `bobyqa_states.dump` and `rusttime/target/` are
gitignored build products; everything else here is tracked.

## Pin

The durable record is `PRIMA_VERSION` (commit hash, branch, commit date) —
pinned to `main`, see `PRIMA_VERSION` for the exact commit.
Re-create the clone at the pinned commit with (from the repo root):

```sh
git init oracle/prima
git -C oracle/prima fetch --depth 1 https://github.com/libprima/prima \
    "$(cut -d' ' -f1 oracle/PRIMA_VERSION)"
git -C oracle/prima checkout FETCH_HEAD
```

(A commit pin is as reproducible as a tag; verify the clone's HEAD against
`PRIMA_VERSION` after cloning.)

## Build

From `oracle/` (as run on 2026-06-04, CMake 3.31 + gfortran 15.2.1):

```sh
cmake -S prima -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_Fortran_FLAGS="-ffp-contract=off"
cmake --build build --parallel
```

`-ffp-contract=off` is mandatory: PRIMA's CMake sets no FP-contraction flags
and gfortran defaults to `-ffp-contract=fast`, which would let the oracle use
FMA where the Rust port never will (FMA parity).

Static libs land in `oracle/build/` (`build/c/libprimac.a`,
`build/fortran/libprimaf.a`).

## Capture

```sh
oracle/capture.sh
```

Compiles `driver.c` (with `-ffp-contract=off` — the objectives must stay
bit-identical to the Rust test objectives) and regenerates the golden set in
`tests/goldens/`. New configs are new `capture` invocations inside
`capture.sh`; new objectives are new `driver` invocations in `driver.c`.

The `capture` shell function inside `capture.sh` has the signature:

```
capture <stem> <problem> [driver flags...]
```

`stem` is the output filename prefix (e.g. `booth_rescue`); `problem` is the
driver registry key; the remaining arguments are passed to the driver
(`--npt N`, `--rhobeg X`, `--rhoend X`, `--maxfun N`). The stem parameter
is required because `--rhoend` variants of the same problem would otherwise
produce filenames that collide with the default-config golden.

The registry holds thirteen problems (`REGISTRY` in `driver.c`):

| Problem | Note |
|---|---|
| `sphere` | n = 2 sphere, interior optimum at origin — baseline |
| `rosenbrock` | n = 2 Rosenbrock — curved valley coverage |
| `booth` | n = 2, upper bound `y <= 2.5` forces optimum to the bound at (1.4, 2.5) — bound-active coverage |
| `rosenbrock10` | n = 10 Rosenbrock — high-dim coverage; converges near x = (−1, 1, …, 1), f ≈ 3.99 |
| `beale` | n = 2 Beale function — classic DFO benchmark, interior optimum at (3, 0.5) |
| `powell_singular` | n = 4 Powell singular — Hessian is singular at the optimum; tests numerically difficult curvature |
| `sphere_onbound` | sphere with x0 near a bound — exercises `preproc` x0-shift and on-bound optimum path |
| `sphere_tight` | sphere with tight bounds `[−0.6, 0.6]^2` and `rho_begin = 0.5` — tight-box coverage |
| `nansphere` | sphere that returns NaN for `x[0] < 0` — graceful-NaN battery case |
| `quad1` | n = 1 quadratic, interior optimum at 0.7 |
| `quad1_lower` | n = 1, optimum -2 outside the box [0, 5], so the solution sits on the lower bound |
| `quartic1` | n = 1 non-quadratic, a double well |
| `reml1` | n = 1 profiled-REML-shaped objective, uses `log` |

The checked-in goldens come from the re-pinned, FP-pinned
(`-ffp-contract=off`) build — see `PRIMA_VERSION`.

`capture.sh` honours `BUILD`/`BIN`/`GOLDENS` env overrides (used by the
non-perturbation check below to capture from the instrumented tree
into a scratch directory).

The exact compile line that worked (no `-lquadmath` or gfortran-as-linker
contingency was needed; from `oracle/`):

```sh
cc -O2 -ffp-contract=off -Wall -Wextra driver.c -Iprima/c/include \
   "$(find build -name libprimac.a -print -quit)" \
   "$(find build -name libprimaf.a -print -quit)" -lgfortran -lm -o driver
```

Known-harmless link warning on Fedora 43 / binutils:
`ld: warning: cobyla_c.f90.o: requires executable stack` — a PRIMA build
product property, not a driver problem.

### Golden inventory (18 files, frozen)

| File | Battery requirement covered |
|---|---|
| `sphere_n2_npt5.txt` | baseline sphere, default config |
| `rosenbrock_n2_npt5.txt` | curved-valley convergence |
| `booth_n2_npt5.txt` | bound-active optimum |
| `rosenbrock10_n10_npt21.txt` | n = 10, high-dimensional coverage |
| `beale_n2_npt5.txt` | DFO classic benchmark, interior optimum |
| `powell_singular_n4_npt9.txt` | n = 4, singular Hessian at optimum; remaining DFO function |
| `sphere_onbound_n2_npt5.txt` | on-bound optimum + `preproc` x0-shift |
| `sphere_tight_n2_npt5.txt` | tight bounds `[−0.6, 0.6]^2` |
| `nansphere_n2_npt5.txt` | NaN objective for `x[0] < 0` — graceful-NaN battery |
| `sphere_n2_npt4.txt` | `npt` at minimum `n + 2` |
| `sphere_n2_npt6.txt` | `npt` at maximum `(n+1)(n+2)/2` |
| `powell_singular_n4_npt15.txt` | `npt` at maximum `(n+1)(n+2)/2` for n = 4 |
| `booth_rescue_n2_npt5.txt` | `--rhoend 1e-12` rescue stressor — triggers rescue on n = 2 |
| `rosenbrock10_rescue_n10_npt21.txt` | `--rhoend 1e-12` rescue stressor — triggers rescue on n = 10 |
| `quad1_n1_npt3.txt` | n = 1 quadratic, interior optimum |
| `quad1_lower_n1_npt3.txt` | n = 1, optimum on the lower bound |
| `quartic1_n1_npt3.txt` | n = 1 non-quadratic double well |
| `reml1_n1_npt3.txt` | n = 1 profiled-REML-shaped objective, uses `log` |

### Freeze

The 18 goldens and the state corpora (`tests/states/`) are now **THE
regression oracle**. PRIMA re-runs are deliberate campaigns only
("freeze then forget"). The PRIMA clone and build may be kept or
rebuilt without consequence, but recapturing goldens outside a sanctioned
campaign is forbidden.

**n = 1 campaign (2026-10-08, pin `1d76fb88`, sanctioned):** captured `quad1`,
`quad1_lower`, `quartic1`, `reml1`; the 14 existing goldens re-captured byte-identical; the crate
replays all 18 bit-exact natively, with `libm` and on `wasm32-wasip1`. `capture.sh` now labels
goldens with the compiler that built `libprimaf.a`. `reml1` depends on the platform's `log`, which
is not correctly rounded on every platform. It is the first golden whose objective uses such a
function; the other objectives use arithmetic only. It matched glibc and wasi-libc on
2026-10-08, and CI runs on Ubuntu only. On another platform (macOS, for example) or with a future
libm, `reml1` alone can fail with nothing wrong in the port.

`test_support.rs`'s objective registry intentionally stays at the **four
frozen problems** (sphere, rosenbrock, booth, rosenbrock10). The state corpus
uses only those four — the "third copy" does not grow.

## Benchmark

```sh
oracle/bench.sh
```

times PRIMA, this crate and `basin` on the same solves, pinned to one core
(`CORE`, default 2); each timed loop runs for `TIME` seconds (default 2.0). It
needs four binaries, built from `oracle/`:

- `driver` — the compile line under [Capture](#capture).
- `driver-stack` — the same driver against a PRIMA configured with
  `-DPRIMA_HEAP_ARRAYS=OFF`:

  ```sh
  cmake -S prima -B build-stack -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_Fortran_FLAGS="-ffp-contract=off" -DPRIMA_HEAP_ARRAYS=OFF
  cmake --build build-stack --parallel
  cc -O2 -ffp-contract=off -Wall -Wextra driver.c -Iprima/c/include \
     "$(find build-stack -name libprimac.a -print -quit)" \
     "$(find build-stack -name libprimaf.a -print -quit)" -lgfortran -lm -o driver-stack
  ```

- `rusttime/target/release/rusttime` and `rusttime/target/release/basintime` —
  `cargo build --release --manifest-path rusttime/Cargo.toml`.

## Instrumentation (state capture)

Two checked-in pieces:

- `dump.f90` — an I/O-only Fortran module (`dump_mod`); during campaigns it is
  copied into the clone as `fortran/common/dump.f90`. It never computes with,
  branches on, or modifies a dumped value, which is what makes the patch
  incapable of perturbing the oracle's FP results.
- `instrument.patch` — adds the `dump_mod` calls at the twelve dump sites
  (entry/exit of `initxf`, `initq`, `inith`, `updateh`, `updatexf`, `updateq`,
  `tryqalt`, `trsbox`, `trrad`, `geostep`, `setdrop_tr`, `rescue`) plus the
  `common/dump.f90` line in `fortran/CMakeLists.txt`. Early returns inside
  instrumented routines carry a duplicated exit dump (7 such sites on this
  pin: `updateh`×2, `updatexf`, `updateq`, `geostep`×2, `rescue`).

N.B. `info` is `intent(out), optional` in `initq`/`inith`/`updateh` and the
`bobyqb.f90` call sites on this pin pass it to **none** of them, so their
guarded `info` dumps emit nothing — those routines' state files have no
`info` field (and their Rust diff tests must not read one). `initxf` and
`rescue` do receive `info`.

**Non-perturbation check:** ran on 2026-06-04 against pin
`1d76fb88` — goldens captured from the instrumented build
(`BUILD=build-instr BIN=driver-instr GOLDENS=<scratch> ./capture.sh`) are
**bit-identical** to the checked-in set (`diff -r` empty).

### Campaign

```sh
oracle/capture_states.sh
```

applies the patch (idempotent), builds `build-instr/`, compiles
`driver-instr`, runs the campaign problems, splits each run's
`bobyqa_states.dump` via `states.py` into `tests/states/<routine>/`,
and de-instruments the clone. The corpus is **FROZEN** — rerunning
is a deliberate campaign. Subsampling: `keep=10` per default config (all
calls if ≤ 10, else the first 5 + a uniform sample), `keep=3` for the
stress/branch-coverage configs. Captured 2026-06-04 against pin `1d76fb88`:
all twelve routines populated, 4.2 MB total (over the ~2 MB target — the
n = 10 matrix-heavy states dominate).

**Rescue verdict:** **TRIGGERED** — the `--rhoend 1e-12`
campaigns (booth, rosenbrock10) produced real `rescue` states
(`tests/states/rescue/`, 7 files). The rescue port has a genuine
end-to-end-reached corpus; no fallback (synthetic-entry) campaign is needed.

## State file format (`bobyqa state v1`)

Plain text, line-oriented, whitespace-tokenised, written by `states.py` from
a `dump_mod` stream:

```
# bobyqa state v1
# prima <commit-hash>
routine <name>          one of the 12 instrumented routines
problem <name>          keys into the driver/test registries
npt / rho_begin / rho_end / max_fun   the driver config of the capturing run
seq <k>                 k-th call of this routine in the run
entry                   then `scalar|vector|matrix|imatrix <name> ...` lines
exit                    then the routine's outputs, same encoding
```

Value encodings: `scalar f 1.5`, `vector d <floats>`,
`matrix bmat <nr> <nc> <nr*nc floats, column-major>`,
`imatrix ij <nr> <nc> <ints>`. Floats are written at 17 significant digits
(exact f64 round-trip); logicals as 0/1. **All indices in the file are
PRIMA's 1-based values** — the Rust diff tests translate.

Rust diff tests compare every f64 value bit for bit — see the `assert_bits`
helpers in `src/test_support.rs`.

## Golden file format (`bobyqa golden v1`)

Plain text, line-oriented, whitespace-tokenised; all floats `%.17g`
(round-trips IEEE-754 `f64` exactly). The file is self-describing — the
filename `<problem>_n<N>_npt<NPT>.txt` is only a label, never parsed.

```
# bobyqa golden v1
# prima <commit-hash>
# compiler <the compiler that built libprimaf.a>
problem sphere
n 2
npt 5
rho_begin 0.5
rho_end 9.9999999999999995e-07
max_fun 500
x0 <n floats>
lower <n floats>
upper <n floats>
eval <n floats> <f>          one line per evaluation, in call order
…
final <n floats> <f> <n_eval> <prima_rc> <prima_rc_name>
```

`final` records PRIMA's raw return code plus a single-token name emitted by
`driver.c`'s own `rc_name()` (PRIMA's `prima_get_rc_string()` returns
sentences, unusable in a tokenised format).
