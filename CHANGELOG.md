# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.2.0
[0.1.3]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.3
[0.1.2]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.2
[0.1.1]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.1
[0.1.0]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.0
