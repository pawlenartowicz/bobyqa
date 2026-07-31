# bobyqa

[![crates.io](https://img.shields.io/crates/v/bobyqa.svg)](https://crates.io/crates/bobyqa)
[![docs.rs](https://img.shields.io/docsrs/bobyqa)](https://docs.rs/bobyqa)
[![CI](https://github.com/pawlenartowicz/bobyqa/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/bobyqa/actions/workflows/ci.yml)
[![license: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Minimize a function from values alone — no derivatives — subject to box bounds.** A pure-Rust,
dependency-free port of M. J. D. Powell's **BOBYQA** (Bound Optimization BY Quadratic
Approximation), transcribed from [PRIMA](https://github.com/libprima/prima)  — see [Design](#design).
As of 2026-06 there is no other pure-Rust BOBYQA on crates.io; the alternatives are all C-library bindings.

## When to use it

Your objective `f(x)` is:

- **black-box** — you can evaluate it, but its gradient is unavailable or unreliable
  (simulation output, legacy code, fitted models, tuning knobs);
- **expensive** — each evaluation costs, so you want a model-based method that converges in few of
  them, not a pattern search;
- **small** — a handful to a few dozen variables;
- **box-bounded** — `lower ≤ x ≤ upper` (the "B" in BOBYQA); every trial point stays inside the
  box, so `f` is never evaluated out of bounds.

If you *do* have reliable gradients, a gradient-based method will beat this.

## API preview

Build the solver once per problem size, then call it repeatedly with **no heap allocation per
call** — `Bobyqa::new` owns every allocation.

```rust
use bobyqa::{Bobyqa, Config};

// Minimize the 2-D Rosenbrock function inside the box [-2, 2]².
let mut solver = Bobyqa::new(2, Config::new(2))?;

let mut x = [0.0, 0.0]; // starting point — overwritten with the best point found
let outcome = solver.minimize(
    |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
    &mut x,
    &[-2.0, -2.0], // lower bounds
    &[ 2.0,  2.0], // upper bounds
);

println!("f = {:.2e} at {:?} after {} evaluations", outcome.f, x, outcome.n_eval);
```

## Restarts

By default `Bobyqa` stops the moment `rho` reaches `rho_end` — the faithful PRIMA
behaviour. On some objectives that single convergence stalls short of the true optimum:
noisy or quantized landscapes, staircase functions, anything where the local quadratic
model runs out of useful curvature before `x` does. And on hard fits the last `rho`
levels can burn most of the budget squeezing out digits that no longer matter. For
those, set `Config::restart`. Instead of stopping, the solver starts a new cycle: `rho`
and `delta` go back to `rho_begin` and the interpolation set is rebuilt from scratch
around the best point found so far. Because the model is discarded rather than
re-widened, the new cycle samples a full `rho_begin` out and can walk out of the basin
the previous one converged in. The returned point is the best over every cycle, so a
restart never returns something worse than stopping would have.

```rust
use bobyqa::{Bobyqa, Config, RestartConfig};

// Same problem as above, but with restarts enabled — RestartConfig::new() is the
// recommended schedule: one restart, fired when a cycle has spent an eighth of the
// budget that remained when it started.
let mut config = Config::new(2);
config.restart = Some(RestartConfig::new());
let mut solver = Bobyqa::new(2, config)?;

let mut x = [0.0, 0.0];
let outcome = solver.minimize(
    |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
    &mut x,
    &[-2.0, -2.0],
    &[ 2.0,  2.0],
);

println!(
    "f = {:.2e} at {:?} after {} evaluations, {} restarts",
    outcome.f, x, outcome.n_eval, solver.last_restart_count(),
);
```

The knobs live on `RestartConfig`:

- `cycle_budget_frac` — the eval cap, and the recommended way to drive the schedule:
  restart once the current cycle has spent this fraction of the budget that *remained
  when the cycle started* (default `0.125`; `0.0` disables it). It is consulted at every
  trust-region iteration, so it fires even on a cycle that is crawling without reducing
  `rho`.
- `max_restarts` — cap on restarts before returning the last cycle's result (default
  `1`). A long schedule divides a fixed `max_fun` into cycles too short to descend.
- `improve_rel_tol` — stop restarting once a full cycle's improvement falls below this,
  relative to `max(1, |f|)` (default `1e-6`).
- `stall_reductions` — restart before `rho` reaches `rho_end`, after this many
  consecutive `rho` reductions that each improve `f` by less than `improve_rel_tol`
  (default `0`: off). It only fires where the solve reduces `rho` at all, which is why
  `cycle_budget_frac` is the recommended trigger instead.

Whichever trigger fires, the final cycle — once no restart remains — always runs down to
`rho_end`, so the returned point is never coarser than a plain solve's.

**Setting the cap changes what the `rho_end` trigger does.** With `cycle_budget_frac`
non-zero, reaching `rho_end` no longer restarts by itself — it restarts only when the cap
agrees the cycle was expensive. So a solve that converges well inside its cap is left
alone and returns exactly what `restart: None` returns, evaluation count included. With
the cap off, `rho_end` restarts on the settle test as before.

**`Config::max_fun` is the TOTAL evaluation budget across all restarts**, not a
per-cycle allowance — size it accordingly when enabling restarts.

`Config` and `RestartConfig` are `#[non_exhaustive]`: build them with `Config::new(n)` /
`RestartConfig::new()` and assign fields, as above — struct literals and
`..Config::new(n)` update syntax won't compile downstream.

## Design

| Design | Detail |
|---|---|
| Faithful port | behaviour-for-behaviour port of PRIMA's modern-Fortran BOBYQA — the same trust-region method, Lagrange-model maintenance, geometry-restoring rescue, and box handling that earn BOBYQA its robustness |
| Bit-exact parity | reproduces PRIMA bit-for-bit across the golden `(x, f)` trajectory battery — every evaluation in order, the rescue path included — natively and on `wasm32-wasip1` |
| Pure Rust | no C, Fortran, or system libraries; builds anywhere `cargo` does, including `wasm32-unknown-unknown` |
| Zero dependencies | std/core/alloc only (dev-dependencies for tests only) |
| No `unsafe` | `#![forbid(unsafe_code)]` at the crate root |
| Deterministic | no RNG, no global state, no threads, no I/O — same inputs → same outputs on a given target |
| Zero-alloc warm path | construct `Bobyqa` once; `minimize` performs no heap allocation |
| Errors, not panics | invalid arguments return a `Status`; the solver does not panic |
| Bounds honoured | every objective evaluation lies within `[lower, upper]` |

## Citing

If this crate contributes to published research, please cite Powell's algorithm paper and PRIMA:

> M. J. D. Powell, *The BOBYQA algorithm for bound constrained optimization without
> derivatives*, DAMTP 2009/NA06, University of Cambridge, 2009.

> Z. Zhang, *PRIMA: Reference Implementation for Powell's methods with Modernization and
> Amelioration*, https://www.libprima.net.

## Credits

Ported from **PRIMA** (libprima, BSD-3-Clause) by Zaikun Zhang et al. — `v0.7.2+`, commit
[`1d76fb88`](https://github.com/libprima/prima/commit/1d76fb88aeffb427cd17ed1e9d0d3b34f414913f),
2026-05-27.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

The ported BOBYQA algorithm derives from [PRIMA](https://github.com/libprima/prima)
(BSD-3-Clause); that copyright notice is retained for the ported portions in
[THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES).


---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
