# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.1
[0.1.0]: https://github.com/pawlenartowicz/bobyqa/releases/tag/v0.1.0
