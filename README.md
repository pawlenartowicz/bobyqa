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
