# bobyqa

[![crates.io](https://img.shields.io/crates/v/bobyqa.svg)](https://crates.io/crates/bobyqa)
[![docs.rs](https://img.shields.io/docsrs/bobyqa)](https://docs.rs/bobyqa)
[![CI](https://github.com/pawlenartowicz/bobyqa/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/bobyqa/actions/workflows/ci.yml)
[![license: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Minimize a function from values alone — no derivatives — subject to box bounds.** A pure-Rust,
dependency-free-by-default, `no_std`-compatible port of M. J. D. Powell's **BOBYQA** (Bound Optimization BY Quadratic
Approximation), transcribed from [PRIMA](https://github.com/libprima/prima)  — see [Design](#design).
[How it compares](#how-it-compares) sets it against the alternatives.

The solver itself is a faithful port. On top of it sits one optional, off-by-default layer —
[restarts](#restarts) — which is not part of BOBYQA as Powell published it; see
[Credits](#credits) for what came from where.

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

## How it compares

| | `bobyqa` (this crate) | [basin](https://crates.io/crates/basin) | [nlopt](https://crates.io/crates/nlopt) (`LN_BOBYQA`) |
|---|---|---|---|
| Implementation | pure Rust, this one algorithm | pure Rust, multi-solver framework | C library, statically linked |
| Algorithm source | PRIMA (2026), bit-exact against its golden trajectories | ported from PRIMA v0.7.2; final-point parity to tolerance, trajectories diverge | C translation of Powell's 2009 Fortran |
| Required dependencies | none (`libm` opt-in) | `rand` family, `num-traits`, `web-time` | C toolchain; bundles libnlopt |
| WebAssembly | yes | yes | no |
| `no_std` | yes (`alloc` + `libm`) | no | no |
| License | MIT OR Apache-2.0 | MIT OR Apache-2.0 | LGPL as a combined work; the BOBYQA files themselves MIT |

Speed: on a clock-locked machine over the standard test functions, this crate is
**4.4–9.9× faster than PRIMA**with identical trajectories and evaluation
counts, a reused `Bobyqa` against `prima_minimize`, which allocates and packs its
workspace on every call. nlopt's C core is faster on cheap objectives, and the gap narrows
as the objective grows more expensive.

## Used by

[GLMM](https://github.com/pawlenartowicz/glmm) — mixed-model fitting for Python and R over
one Rust kernel — uses this crate as its optimizer. [MCPower](https://github.com/pawlenartowicz/MCPower)
builds on GLMM: a power-analysis application that fits mixed models thousands of times per
analysis, entirely in the browser — a production deployment of this crate compiled to
WebAssembly.

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

The idea of restarting a converged model-based solve this way is Cartis, Roberts and
co-authors', from the Py-BOBYQA papers ([TOMS 2019](https://doi.org/10.1145/3338517),
[Optimization 2022](https://doi.org/10.1080/02331934.2021.1883015)). What triggers a
restart here, and the guarantees around it, are this crate's own — see
[Credits](#credits).

`Config` and `RestartConfig` are `#[non_exhaustive]`: build them with `Config::new(n)` /
`RestartConfig::new()` and assign fields, as above — struct literals and
`..Config::new(n)` update syntax won't compile downstream.

## Design

| Design | Detail |
|---|---|
| Faithful port | behaviour-for-behaviour port of PRIMA's modern-Fortran BOBYQA — the same trust-region method, Lagrange-model maintenance, geometry-restoring rescue, and box handling that earn BOBYQA its robustness. [Restarts](#restarts) are the one addition, off by default, and leave the port untouched when unused |
| Bit-exact parity | reproduces PRIMA bit-for-bit across the golden `(x, f)` trajectory battery — every evaluation in order, the rescue path included — natively and on `wasm32-wasip1` |
| Pure Rust | no C, Fortran, or system libraries; builds anywhere `cargo` does, including `wasm32-unknown-unknown` |
| Zero dependencies | zero by default; the optional `libm` feature (the `no_std` math backend) is the only dependency, and only when you ask for it |
| `no_std` + `alloc` | `default-features = false, features = ["libm"]` builds without `std` — see [`no_std` usage](#no_std-usage) |
| No `unsafe` | `#![forbid(unsafe_code)]` at the crate root |
| Deterministic | no RNG, no global state, no threads, no I/O — same inputs → same outputs on a given target |
| Zero-alloc warm path | construct `Bobyqa` once; `minimize` performs no heap allocation |
| Errors, not panics | invalid arguments return a `Status`; the solver does not panic |
| Bounds honoured | every objective evaluation lies within `[lower, upper]` |

## `no_std` usage

The crate is `#![no_std]` + `alloc`: it needs an allocator (`Bobyqa::new` allocates once per
problem size; `minimize` allocates nothing) but not an operating system. Turn off the default
`std` feature and enable `libm`, which supplies the float math `core` lacks:

```toml
[dependencies]
bobyqa = { version = "0.3", default-features = false, features = ["libm"] }
```

Two things change without `std`: the `std::error::Error` impl on `Status` is absent (`Status`
keeps `Display`), and math routes through [`libm`](https://crates.io/crates/libm) — the crate's
only (optional) dependency. CI proves the libm backend bit-exact by running the entire PRIMA
golden-trajectory battery against it, and proves the `no_std` claim by building for the
bare-metal `thumbv7em-none-eabihf` target.

## Citing

If this crate contributes to published research, please cite Powell's algorithm paper and PRIMA:

> M. J. D. Powell, *The BOBYQA algorithm for bound constrained optimization without
> derivatives*, DAMTP 2009/NA06, University of Cambridge, 2009.

> Z. Zhang, *PRIMA: Reference Implementation for Powell's methods with Modernization and
> Amelioration*, https://www.libprima.net.

If you use [restarts](#restarts), please also cite the work the mechanism comes from:

> C. Cartis, J. Fiala, B. Marteau and L. Roberts, *Improving the Flexibility and Robustness
> of Model-Based Derivative-Free Optimization Solvers*, ACM Transactions on Mathematical
> Software 45(3), 32:1–32:41, 2019. <https://doi.org/10.1145/3338517>

> C. Cartis, L. Roberts and O. Sheridan-Methven, *Escaping local minima with derivative-free
> methods: a numerical investigation*, Optimization 71(8), 2343–2373, 2022.
> <https://doi.org/10.1080/02331934.2021.1883015>

## Credits

This crate is three things, and it is worth being clear about which is which.

**The solver is a port.** Everything that does the optimising — the trust-region method, the
`BMAT`/`ZMAT` Lagrange-model maintenance, the geometry-restoring rescue, the box handling —
is transcribed behaviour-for-behaviour from **PRIMA** (libprima, BSD-3-Clause) by Zaikun
Zhang et al. — `v0.7.2+`, commit
[`1d76fb88`](https://github.com/libprima/prima/commit/1d76fb88aeffb427cd17ed1e9d0d3b34f414913f),
2026-05-27 — which in turn implements Powell's BOBYQA. None of it is original here, and that
is the point: faithfulness is what earns BOBYQA its robustness, and the bit-exact parity
battery exists to prove none was lost in translation.

**The restart idea is Cartis, Fiala, Marteau, Roberts and Sheridan-Methven's**, from the
Py-BOBYQA papers cited under [Citing](#citing): that a converged model-based solve can be
usefully restarted by resetting the trust-region radius and rebuilding the interpolation set
around the best point found, and that doing so is how a local solver escapes a local minimum.
It is implemented here from the algorithmic description in those papers, on top of the
crate's own PRIMA-derived geometry routines.

**The restart policy is mine.** What decides *when* a restart happens, and what is guaranteed
around it, was designed and measured here: the `cycle_budget_frac` eval cap as the primary trigger
(consulted every trust-region iteration, so it fires on a cycle that is crawling without
reducing `rho` — where a reduction-sited trigger has no site to fire from); the coupling that
makes the cap suppress the `rho_end` trigger, so an ordinary solve that finishes inside its
cap is left bit-for-bit alone; the `stall_reductions` stalled-tail trigger; the monotone
best-point record that makes a restart unable to return a worse answer than stopping; and the
zero-alloc integration, in which a rebuild reuses buffers `Bobyqa::new` had already sized.
Three things from the papers were considered and deliberately left out: the noise-aware
auto-detection trigger (this crate is scoped to smooth deterministic objectives), the adaptive
`rho` reset scaling, and randomised interpolation-point replacement — which determinism, a
hard invariant here, rules out outright.

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
