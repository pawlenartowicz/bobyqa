# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.2]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.2
[0.1.1]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.1
[0.1.0]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.0
