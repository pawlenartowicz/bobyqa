# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] — Unreleased

**With a default `Config`, every one-dimensional problem (and any problem run at the maximum
`npt = (n + 1)(n + 2) / 2`) can now depart from PRIMA's trajectory; set the new
`Config::prima_parity = true` to get PRIMA's trajectories bit for bit, as before.** Every other
`npt` is unchanged, bit for bit, in both modes.

### Added

- **`Config::prima_parity`** (default `false`). `true` reproduces PRIMA's BOBYQA bit for bit on
  every `n` (the PRIMA golden battery runs in this mode, natively, with `libm` and on
  `wasm32-wasip1`). `false` applies the crate's deliberate deviations from PRIMA, of which
  there is one, below. Future deviations, if any, join this switch.

### Changed

- **No spurious RESCUE on a fully determined model.** At `npt = (n + 1)(n + 2) / 2` (the only
  legal `npt` when `n = 1`) the quadratic model is fully determined, the updating formula's
  `beta` is zero in exact arithmetic, and each denominator equals its `vlag` squared. PRIMA's
  trust-region test for calling RESCUE (`bobyqb.f90` L397, `any(den > maxval(vlag**2))`) is
  then decided by rounding alone: it fired on 16 of 26 improving steps of a one-dimensional
  REML-shaped objective and on 21 of 50 of two-dimensional Rosenbrock at `npt = 6`, each time
  spending objective evaluations on a healthy model. With `prima_parity: false` the test uses
  Powell's original factor `0.5` (PRIMA's commented alternatives at L401-402), which still calls
  RESCUE on non-finite values and on a denominator damaged well below `vlag**2`. On a 2,000-case
  one-dimensional fuzz of quadratics with noise and random boxes the default took fewer
  evaluations than parity mode in 1,251 cases, the same in 673 and more in 76, and never ended
  above parity mode's final value beyond the solve's accuracy (`tests/one_dimensional.rs`).

## [0.3.1] — 2026-10-02

Additive only: `Bobyqa::minimize` and `bobyqa(...)` are unchanged, bit for bit.

### Added

- **`Bobyqa::minimize_with_radius`** and the `TrustRadius { rho, delta }` it hands the
  objective (`FnMut(&[f64], TrustRadius) -> f64`): the trust-region radii in force at each
  evaluation, for objectives whose own accuracy can follow the solver's resolution (an inner
  solve run looser while `rho` is large, as in Ehrhardt and Roberts 2021's inexact DFO). The
  engine only writes the radii into a stack `Cell` before each evaluation; nothing reads them
  back, so the solve is `minimize`'s (`tests/radius.rs`), and the call stays allocation-free
  (`tests/alloc.rs`). `TrustRadius` is `#[non_exhaustive]`.

## [0.3.0] — 2026-08-19

A default-features build with a default `Config` is unchanged by this release: std math, the
`std::error::Error` impl on `Status`, zero dependencies, and bit-exact `(x, f)` trajectories
against the PRIMA goldens.

### Added

- **Opt-in f-tolerance stopping** (ftol spec): `Config::ftol_rel` / `Config::ftol_abs`
  (both default `None`: off, bit-exact PRIMA). Stop when the best f improves by less than
  `ftol_rel * max(|f_best|, 1) + ftol_abs` over one full rho stage — checked only at the
  rho-reduction site, never during the first stage, never once `rho` reaches `rho_end`.
  A triggered stop returns the new `Status::FtolReached`, a converged-class outcome that
  does not spend a `Config::restart` cycle (ftol wins when both would fire at the same
  reduction).
- **Fallible workspace allocation** (safe-checks spec S2): every construction-time buffer
  now allocates via `try_reserve_exact`; an allocation the platform cannot satisfy returns
  the new `Status::AllocationFailed` from `Bobyqa::new` instead of aborting the process
  (previously `vec![]` aborted — fatal on `no_std`/embedded). The warm path was and stays
  allocation-free (`tests/alloc.rs`). `Bobyqa::new`'s docs now state the workspace size
  formula so callers on big problems can budget.
- **Workspace-size overflow gate** (safe-checks spec S1): `Bobyqa::new` validates every
  derived dimension sum/product with checked arithmetic before sizing anything, rejecting
  as `InvalidArgs` any `(n, npt)` whose buffer sizes would overflow `usize`. Previously
  debug builds panicked (contradicting the "Panics: never" docs) and release builds
  wrapped — on 32-bit targets (wasm32, thumbv7em) a wrapped size could stay in-bounds and
  return garbage silently. `Config::new`'s derived `npt`/`max_fun` now saturate instead of
  wrapping on absurd `n`. Adversarial-pair unit tests plus a
  `#[cfg(target_pointer_width = "32")]` rejection test that CI's wasm32 job executes.
- **`MAX_RESTARTS_CAP`** (= 10 000): `RestartConfig::max_restarts` above it is rejected,
  closing both a `+ 1` overflow and a caller-sized giant allocation of the per-cycle
  boundary store (safe-checks spec S2).
- **Hardening tests** (safe-checks spec S3): reuse after an objective panic reproduces a
  clean solver's trajectory bit-for-bit (`tests/safety.rs` pins the documented
  "re-initializes whatever it reads" contract), and `f_target = +inf` — which would
  "succeed" on the first evaluation — is now rejected like NaN.

- **`no_std` + `alloc` support.** The crate is now `#![no_std]`: it needs an allocator
  (`Bobyqa::new` still allocates once per problem size; `minimize` still allocates zero) but
  not an operating system. `no_std` consumers build with
  `default-features = false, features = ["libm"]`.
- **`std` cargo feature** (default-on) — gates the `std::error::Error` impl on `Status`;
  `Status` keeps `Display` without it. Nothing else.
- **`libm` cargo feature** — backs the math seam (`sqrt`/`floor`/`round`) with the optional
  [`libm`](https://crates.io/crates/libm) dependency instead of std intrinsics. Orthogonal to
  `std`: CI runs the full bit-exact parity battery with both on, so the libm backend is held
  to the same goldens as the default build. With neither feature on, the crate fails to
  compile with one clear `compile_error!` message.
- **CI gates**: the full battery on `--features libm`, and a
  `--no-default-features --features libm` build for the bare-metal `thumbv7em-none-eabihf`
  target, beside the existing wasm gates.

### Changed

- The two `powi` calls in the `norm` overflow rescue became consts (`2^-1022`/`2^1022`, both
  exact powers of two; a unit test pins them bit-for-bit to the original expressions), and one
  `f64::round` call site moved onto the math seam. Bit-exact no-ops, verified against the
  golden battery in isolation.

## [0.2.0] — 2026-07-31

With `restart: None` — the default — `Bobyqa` is unchanged by this release: its trust-region
core, allocation shape, and `(x, f)` trajectories remain bit-exact, and the existing PRIMA
golden-trajectory battery still passes bit-exact.

### Added

- **Restarts**, on `Bobyqa` itself via the new `Config::restart: Option<RestartConfig>`
  field (default `None`: plain BOBYQA). Rather than stopping, a solve can start a new cycle:
  `rho`/`delta` go back to `rho_begin` and the interpolation set is rebuilt from scratch
  around the best point found so far. Discarding the model rather than re-widening it is
  what lets a new cycle leave the basin the previous one converged in. The returned point is
  the best over every cycle, so a restart never returns something worse than stopping would
  have. `Config::max_fun` is the TOTAL evaluation budget across every cycle, not just the
  opening solve.
- **Three triggers**, all on `RestartConfig`, all gated by `max_restarts` (default 1) and by
  a relative-improvement settle test (`improve_rel_tol`, default `1e-6`):
  - `cycle_budget_frac` (default `0.125`) — the eval cap and the recommended trigger:
    restart once the cycle has spent that fraction of the budget that remained when it
    started. Consulted at every trust-region iteration, so it fires on a cycle that is
    crawling without reducing `rho`. `0.0` disables it.
  - `rho_end` — reaching the target radius. **With the cap set this trigger defers to it:**
    reaching `rho_end` restarts only when the cap agrees the cycle was expensive, so a solve
    that converges well inside its cap returns exactly what `restart: None` returns,
    evaluation count included. With the cap off it restarts on the settle test alone.
  - `stall_reductions` (default `0`: off) — restart before `rho_end` after that many
    consecutive `rho` reductions each improving the best value by less than
    `improve_rel_tol` relative to `max(1, |f|)`, cutting a stalled `rho` tail.

  Whichever trigger fires, the final cycle — once no restart remains — always runs down to
  `rho_end`, so the returned point is never coarser than a plain solve's.
- `Bobyqa::last_restart_count()` — restarts performed by the last `minimize` call — and
  `Bobyqa::last_cycle_boundaries()` — the cumulative evaluation count at each restart
  boundary, for per-cycle costs. Both are allocation-free at `minimize` time (the boundary
  store is sized at construction).
- With restarts enabled, `Status::Converged` also covers a settled restart schedule: it is
  returned both when a single `rho_end` convergence is reached and when a restart schedule
  stops because a cycle failed the `improve_rel_tol` settle test, `max_restarts` was
  reached, or a later cycle could not continue. Either way, the returned point is always
  the best one evaluated across all cycles.

### Changed

- **Breaking:** `Config` and `RestartConfig` are now `#[non_exhaustive]`. Struct literals
  and struct-update syntax (`..Config::new(n)`) no longer compile outside the crate — build
  with `Config::new(n)` / `RestartConfig::new()` and assign fields to override. Taken in one
  bump so future knobs (both structs are expected to grow) land as non-breaking additions.

## [0.1.3] — 2026-07-02

No behaviour or API change — housekeeping release.

### Changed

- Formatted the 0.1.2 test additions with `rustfmt` (CI's `cargo fmt --check` was red).
- Publishing on a version tag now runs the full CI suite as a required job — a tag
  whose commit fails any gate never reaches crates.io.

## [0.1.2] — 2026-07-02

No behaviour change — `(x, f)` trajectories remain bit-exact against the PRIMA
oracle. Purely an additive public constant.

### Added

- `pub const FUNCMAX` — PRIMA's moderated-extreme-barrier ceiling (`1e30` for
  `f64`). Every objective value is moderated to this ceiling before use (`NaN`/`+inf`
  → `FUNCMAX`), so a returned `Outcome::f >= FUNCMAX` means the solver never found a
  finite objective value: it distinguishes a degenerate run (e.g. an all-infeasible
  initial interpolation set, which still terminates faithfully as
  `Status::Converged` on the flat moderated surface) from a genuine one.
- `Outcome::found_finite()` — convenience predicate (`self.f < FUNCMAX`) for that
  check, documenting the intent at the call site.

## [0.1.1] — 2026-06-13

Performance release — no API or behaviour change; `(x, f)` trajectories remain
bit-exact against the PRIMA oracle.

### Changed

- Trust-region iterations now compute the VLAG/DEN/BETA kernel once and reuse the
  result across the point-dropping and H-update steps (previously up to three
  identical ≈15·n² recomputations per iteration); reuse is by copy, bit-identical.
- Hot-loop restructuring in the linear-algebra kernels (bounds-check-free column
  slicing, loop interchange, invariant hoisting) — bit-identical results.

### Added

- Criterion warm-path micro-benchmark (`benches/solver.rs`, dev-only).

## [0.1.0] — 2026-06-09

Initial release.

- Faithful, dependency-free pure-Rust port of M. J. D. Powell's **BOBYQA**,
  transcribed from PRIMA's modern-Fortran reference and differentially tested
  against it — bit-exact `(x, f)` trajectory parity across the golden battery,
  natively and on `wasm32-wasip1`.
- Public API: `Bobyqa` (built once per problem size, reused across `minimize`
  calls with **no heap allocation per call**), the one-shot `bobyqa` convenience
  function, and `Config` / `Outcome` / `Status`.
- `#![forbid(unsafe_code)]`; no required runtime dependencies (std/core/alloc
  only); deterministic — no RNG, global state, threads, or I/O; invalid
  arguments are returned as a `Status`, never panicked.

[0.3.1]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.3.1
[0.3.0]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.3.0
[0.2.0]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.2.0
[0.1.3]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.3
[0.1.2]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.2
[0.1.1]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.1
[0.1.0]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.0
