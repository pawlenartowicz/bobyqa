//! A pure-Rust, dependency-free port of M. J. D. Powell's BOBYQA (Bound
//! Optimization BY Quadratic Approximation) — a derivative-free,
//! box-constrained local optimizer, ported from PRIMA's modern Fortran.
//!
//! Trajectory-parity-tested **bit-exact** against PRIMA (natively and on
//! `wasm32-wasip1`). Setting [`Config::prima_parity`] to `false` opts into one deliberate
//! deviation from PRIMA, only on a fully determined model; see that field.
//!
//! One public solver surface: [`Bobyqa`], the faithful PRIMA port. By default it
//! stops as soon as `rho` reaches `rho_end`; setting [`Config::restart`] lets the
//! solve continue instead — `rho`/`delta` go back to `rho_begin` and the
//! interpolation set is rebuilt from scratch around the best point found — for
//! objectives where a single `rho_end` convergence stalls short (noisy or
//! quantized landscapes, long `rho` tails on hard fits). With `restart: None` the
//! solver is bit-exact-unchanged by the restart machinery's existence.
//!
//! Three invariants hold across every call:
//! - **Feasibility** — every point at which the objective is evaluated lies
//!   within `[lower, upper]`.
//! - **Determinism** — no global mutable state, no RNG, no I/O, no threads;
//!   identical inputs give identical outputs on a given target.
//! - **Zero-alloc warm path** — heap allocation happens only at construction time
//!   ([`Bobyqa::new`] is the sole allocation site, sized once for the problem's
//!   `(n, npt)` and restart schedule); a built solver then runs
//!   [`Bobyqa::minimize`] with no further allocation.
//!
//! # Cargo features and `no_std`
//!
//! The crate is `#![no_std]` + `alloc`. Two orthogonal features (both hold the bit-exact
//! parity guarantee):
//! - **`std`** (default) — gates the `std::error::Error` impl on [`Status`]; without it
//!   `Status` keeps `Display` only. Nothing else.
//! - **`libm`** — backs the float math with the [`libm`](https://crates.io/crates/libm)
//!   crate instead of std intrinsics; the only dependency, pulled in only by this feature.
//!
//! `no_std` consumers build with `default-features = false, features = ["libm"]`. At least
//! one of the two features must be enabled.

#![forbid(unsafe_code)]
#![no_std]

// `Bobyqa` owns its buffers (allocated once in `new`), so the crate needs `alloc` but not `std`.
// `std` is linked only for the default-on `std` feature (the `Error` impl)
// and for tests, which always build with std.
extern crate alloc;
#[cfg(any(feature = "std", test))]
extern crate std;

// Compiles the README examples as doctests (rustdoc only; never part of the built crate).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod bobyqb;
mod consts;
mod geometry;
mod initialize;
mod linalg;
mod mat;
mod math;
#[cfg(not(feature = "count-kernels"))]
mod powalg;
#[cfg(feature = "count-kernels")]
pub mod powalg; // kernel-count instrumentation: the counter must be reachable from tests/kernel_counts.rs under this feature only
mod rescue;
#[cfg(test)]
mod test_support;
mod trustregion;
mod update;
mod util;

use consts::{
    BOUNDMAX, ETA1_DFT, ETA2_DFT, GAMMA1_DFT, GAMMA2_DFT, MAXFUN_DIM_DFT, RHOBEG_DFT, RHOEND_DFT,
};
use core::cell::Cell;
use util::moderatex1;

/// Tuning knobs for [`Bobyqa`]. No `Default`: `npt`'s default (`2n + 1`) needs `n` —
/// start from [`Config::new`] and assign the fields to override (`#[non_exhaustive]`
/// rules out struct literals, update syntax included, outside this crate).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Config {
    /// Number of interpolation points, in `n + 2 ..= (n + 1)(n + 2) / 2` (default `2n + 1`,
    /// Powell's recommendation). The `npt` initial model-building evaluations are a per-call
    /// floor on the evaluation count: hot-loop callers re-solving fresh objectives may prefer
    /// the minimum `n + 2` to trade model richness for fewer evaluations.
    pub npt: usize,
    /// Initial trust-region radius — positive and finite (default 1.0).
    pub rho_begin: f64,
    /// Final trust-region radius — the target accuracy, in `(0, rho_begin]` (default 1e-6).
    pub rho_end: f64,
    /// Objective-evaluation budget — must exceed `npt` (default `500 * n`).
    pub max_fun: usize,
    /// Stop as soon as an evaluation reaches `f <= f_target` (default `-inf`: disabled).
    /// NaN and `+inf` are rejected by [`Bobyqa::new`] (a `+inf` target would "succeed"
    /// on the first evaluation).
    pub f_target: f64,
    /// Restart schedule. `None` (the default) is plain BOBYQA: the solve ends the moment
    /// `rho` reaches `rho_end`. `Some` starts a new cycle instead:
    /// `rho`/`delta` go back to `rho_begin` and the interpolation set is rebuilt from scratch
    /// around the best point found — see [`RestartConfig`] for what triggers that and what
    /// stops it. `max_fun` stays the TOTAL evaluation budget across all cycles.
    pub restart: Option<RestartConfig>,
    /// Opt-in f-tolerance stopping: stop when the best f improves by less than
    /// `ftol_rel * max(|f_best|, 1.0) + ftol_abs` over one full rho stage. With both `ftol`
    /// fields `None` (the default) the check is off — it is not even reached, so it cannot
    /// move the solve.
    ///
    /// Semantics vs [`f_target`](Self::f_target): `f_target` is "stop when f is good enough
    /// in absolute terms"; `ftol` is "stop when f stops improving". The check runs at
    /// **stage granularity** — only where the solver is about to shrink `rho` — so a
    /// single micro-kink iteration cannot trigger it, and it never fires during the first
    /// rho stage or once `rho` has reached `rho_end` (the ladder finishing is
    /// [`Status::Converged`], as today). A triggered stop returns
    /// [`Status::FtolReached`] — a converged-class outcome that does not trigger a
    /// [`Config::restart`] cycle; when both are configured and both would fire at the same
    /// rho reduction, `ftol` wins and the solve stops.
    ///
    /// If `Some`, the value must be finite and `>= 0.0` (`Some(0.0)` is legal: stops only
    /// on exact stagnation).
    pub ftol_rel: Option<f64>,
    /// Absolute part of the `ftol` test — see [`ftol_rel`](Self::ftol_rel). Same validation:
    /// if `Some`, finite and `>= 0.0`. Either field alone enables the check (the missing
    /// one contributes 0).
    pub ftol_abs: Option<f64>,
    /// PRIMA parity switch. `true` (the default) reproduces PRIMA's BOBYQA bit for bit on
    /// every `n`; `false` applies this crate's deliberate deviation from PRIMA.
    ///
    /// The deviation applies only when `npt` is the maximum `(n + 1)(n + 2) / 2`: the quadratic
    /// model is then fully determined by its points (and at `n = 1` that maximum, 3, is the
    /// only legal `npt`). Every `npt` below the maximum runs PRIMA's code unchanged in both
    /// modes.
    ///
    /// With a fully determined model the updating formula's `beta` is zero in exact
    /// arithmetic, so each denominator `den[k]` equals `vlag[k]^2` and PRIMA's test for calling
    /// RESCUE after a trust-region step, `any(den > maxval(vlag^2))`, passes or fails on
    /// rounding alone. RESCUE then runs on ordinary iterations and spends objective
    /// evaluations. With `prima_parity: false` that test uses Powell's original factor,
    /// `any(den > 0.5 * maxval(vlag^2))` (PRIMA keeps it as commented alternatives at
    /// `bobyqb.f90` L401-402), which still calls RESCUE on non-finite values and on a
    /// denominator damaged well below `vlag^2`.
    ///
    /// What to expect from `false`: on a one-dimensional quadratic objective it ends as low as
    /// parity mode, within the solve's accuracy, usually in fewer evaluations. On other
    /// objectives the two modes take different paths. `false` usually stops sooner, so at the
    /// same `rho_end` it can end above parity mode, and on a problem with several local minima
    /// the two modes can stop in different ones.
    pub prima_parity: bool,
}

impl Config {
    /// PRIMA's defaults for an `n`-dimensional problem.
    ///
    /// The two derived defaults saturate instead of overflowing on absurd `n`:
    /// a saturated `npt`/`max_fun` then fails [`Bobyqa::new`]'s
    /// range checks, so no panic and no wrapped size can escape.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            npt: n.saturating_mul(2).saturating_add(1),
            rho_begin: RHOBEG_DFT,
            rho_end: RHOEND_DFT,
            max_fun: MAXFUN_DIM_DFT.saturating_mul(n),
            // PRIMA's FTARGET_DFT is -REALMAX, which would terminate on f = -REALMAX;
            // -inf is strictly "off".
            f_target: f64::NEG_INFINITY,
            // Off by default: existing users' numerics must not move.
            restart: None,
            // Off by default: `None` must not change the solve.
            ftol_rel: None,
            ftol_abs: None,
            // On: PRIMA's tests run literally. `false` opts into the deviation documented on
            // the field.
            prima_parity: true,
        }
    }
}

/// Why the solver stopped (or why construction failed).
///
/// Implements [`Display`](core::fmt::Display) unconditionally; the `std::error::Error` impl
/// requires the (default-on) `std` cargo feature, so `no_std` builds keep `Display` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The trust-region radius reached `rho_end`, with any restart schedule settled there —
    /// with [`Config::restart`] `None` that is the whole story; with restarts enabled it
    /// additionally means no further restart cycle was due (the
    /// [`RestartConfig::improve_rel_tol`] settle test, or the
    /// [`max_restarts`](RestartConfig::max_restarts) cap) or a later cycle could not continue.
    Converged,
    /// An evaluation reached `f_target`.
    TargetReached,
    /// The best f improved by less than the configured [`Config::ftol_rel`]/
    /// [`Config::ftol_abs`] tolerance over one full rho stage — a converged-class outcome
    /// (the fit stopped improving on the caller's own f-scale before the rho ladder
    /// finished). Only reachable when at least one `ftol` field is set.
    FtolReached,
    /// The `max_fun` evaluation budget was exhausted.
    MaxFunReached,
    /// The rescue procedure could not restore the interpolation geometry.
    ModelDegenerate,
    /// Bad bounds, `npt`, or slice sizes.
    InvalidArgs,
    /// [`Bobyqa::new`] could not allocate the solver workspace:
    /// the allocator refused the request, or a buffer's byte size overflowed `isize`.
    /// Construction-time only — a built solver never allocates again.
    AllocationFailed,
}

impl core::fmt::Display for Status {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Status::Converged => {
                "the trust-region radius reached rho_end, with any restart schedule settled"
            }
            Status::TargetReached => "an evaluation reached f_target",
            Status::FtolReached => {
                "the best f improved by less than the ftol tolerance over one rho stage"
            }
            Status::MaxFunReached => "the max_fun evaluation budget was exhausted",
            Status::ModelDegenerate => "the interpolation model degenerated beyond rescue",
            Status::InvalidArgs => "invalid arguments: bad bounds, npt, sizes, or config",
            Status::AllocationFailed => "the solver workspace could not be allocated",
        })
    }
}

// `Error` requires the (default-on) `std` feature; under `no_std` builds the impl is absent
// and `Status` keeps its `core`-only `Display`.
#[cfg(feature = "std")]
impl std::error::Error for Status {}

/// PRIMA's moderated-extreme-barrier ceiling (`consts.F90` L169, `10^min(30, range/2)`
/// = `1e30` for `f64`). Every objective value is passed through PRIMA's `moderatef`
/// before the solver uses it: `NaN` and `+inf` become `FUNCMAX`, and any finite value
/// above it is clamped to it. This is what lets BOBYQA keep making progress past an
/// occasional non-finite evaluation.
///
/// Consequence for callers: a returned [`Outcome::f`] `>= FUNCMAX` means the solver
/// **never found a genuinely finite objective value** — every point it evaluated was
/// `NaN`/`+inf` or beyond the ceiling. When the whole initial interpolation set is
/// infeasible this flat, moderated surface still terminates as [`Status::Converged`]
/// (a faithful PRIMA `SMALL_TR_RADIUS` exit on a flat model), so `status` alone does
/// not distinguish it — test `outcome.f >= FUNCMAX` to detect a degenerate fit that
/// never left an infeasible region.
pub const FUNCMAX: f64 = consts::FUNCMAX;

/// The result of one [`Bobyqa::minimize`] or [`Bobyqa::minimize_with_radius`] call.
#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    /// Best objective value found. `NaN` when nothing was evaluated:
    /// `status` is [`Status::InvalidArgs`], or [`Status::AllocationFailed`] from [`bobyqa`].
    /// A value `>= `[`FUNCMAX`] means every evaluated point was non-finite (see [`FUNCMAX`]):
    /// the run is degenerate even if `status` is [`Status::Converged`].
    pub f: f64,
    /// Objective evaluations consumed.
    pub n_eval: usize,
    /// Why the solver stopped.
    pub status: Status,
}

impl Outcome {
    /// Whether the solver ever evaluated a genuinely finite objective value.
    ///
    /// `false` means every point it tried was `NaN`/`+inf` (each moderated to
    /// [`FUNCMAX`]) — a degenerate run, even when [`status`](Self::status) is
    /// [`Status::Converged`] (a flat moderated surface exits faithfully as PRIMA's
    /// `SMALL_TR_RADIUS`). Equivalent to `self.f < FUNCMAX`; prefer this at call
    /// sites for intent. `NaN` `f` (the no-evaluation case: [`Status::InvalidArgs`] or
    /// [`Status::AllocationFailed`]) is also reported as not-finite.
    #[must_use]
    pub fn found_finite(&self) -> bool {
        self.f < FUNCMAX
    }
}

/// The trust-region radii in force when the solver requests an evaluation, handed to the
/// objective of [`Bobyqa::minimize_with_radius`].
///
/// `rho` is the lower bound on the trust-region radius (PRIMA's `RHO`): it starts at
/// [`Config::rho_begin`], never increases within a cycle, and ends at or above
/// [`Config::rho_end`] (a restart sets it back to `rho_begin`). `delta` is the current
/// trust-region radius (PRIMA's `DELTA`), always `>= rho`. Callers use them to set the
/// accuracy of an inexact objective: a value error far below `rho^2` is wasted on the model.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TrustRadius {
    /// Lower bound on the trust-region radius (PRIMA `RHO`).
    pub rho: f64,
    /// Current trust-region radius (PRIMA `DELTA`).
    pub delta: f64,
}

/// A reusable BOBYQA solver: holds every buffer the algorithm needs, built
/// once per problem size and driven across many [`Bobyqa::minimize`] calls.
#[derive(Debug, Clone)]
pub struct Bobyqa {
    n: usize,
    config: Config,
    /// All solver scratch — sized for `(n, npt)` in [`Bobyqa::new`], the crate's only
    /// allocation site; `minimize` re-initializes whatever it reads (the zero-alloc warm
    /// path, enforced by `tests/alloc.rs`).
    ws: bobyqb::SolverWs,
    /// Per-cycle instrumentation backing store: sized
    /// `max_restarts + 1` in [`Bobyqa::new`] when restarts are enabled (empty otherwise), lent
    /// to the engine per call, so filling it — at most one entry per restart — never allocates
    /// inside `minimize`. Its length is the restart count of the last call.
    cycle_boundaries: util::Reserved<usize>,
}

impl Bobyqa {
    /// Allocates all scratch for an `n`-dimensional problem with `config`.
    ///
    /// # Errors
    ///
    /// [`Status::InvalidArgs`] when `(n, config)` is rejected: `n = 0`; an `(n, npt)`
    /// whose workspace dimensions would overflow `usize`; `npt`
    /// outside `n + 2 ..= (n + 1)(n + 2) / 2` (PRIMA's preprocessing would
    /// clamp; we reject); `rho_begin` not a positive finite number; `rho_end`
    /// not in `(0, rho_begin]`; `max_fun <= npt` (PRIMA preproc would raise; we
    /// reject); `f_target` NaN or `+inf` (a `+inf` target would "succeed" on the first
    /// evaluation); `ftol_rel`/`ftol_abs` set but not a finite value `>= 0.0`. With
    /// `restart` set, additionally: `max_restarts`
    /// zero or above [`MAX_RESTARTS_CAP`]; `improve_rel_tol` negative or NaN;
    /// `cycle_budget_frac` outside `[0.0, 1.0]` or NaN.
    ///
    /// [`Status::AllocationFailed`] when the workspace allocation itself fails:
    /// the allocator refused, or a buffer's byte size overflowed
    /// `isize`. The workspace is
    /// `(npt + 13)(npt + n) + 3n(n + 5)/2` doubles by Powell's working-space bound, plus
    /// this port's hoisted per-call scratch of the same order — roughly `O(npt^2 + n*npt)`
    /// f64s overall (about 6 GB at `n = 10_000` with the default `npt = 2n + 1`) — so
    /// callers on big problems can budget before constructing.
    ///
    /// # Panics
    ///
    /// Never — invalid `(n, config)` is reported through [`Status::InvalidArgs`] and an
    /// unsatisfiable allocation through [`Status::AllocationFailed`].
    #[expect(clippy::neg_cmp_op_on_partial_ord)] // `!(a >= b)` is load-bearing for NaN — never `a < b`
    pub fn new(n: usize, config: Config) -> Result<Self, Status> {
        if n == 0 {
            return Err(Status::InvalidArgs);
        }
        // The workspace-size overflow gate (`ws_dims_ok`), deliberately FIRST after `n == 0`: the
        // range checks below (`n + 2`, `(n + 1)(n + 2) / 2`, `npt + 1`) and every workspace-buffer size
        // (`Mat::try_zeros` products, `2 * n + 1` in `SolverWs::new` — bounded by `npt + n`
        // once `npt >= n + 2` holds) compute unchecked arithmetic that is safe only once
        // this gate has passed. `Mat` indexing stays unchecked by the same reasoning:
        // overflowing sizes are unreachable past this point.
        if !ws_dims_ok(n, config.npt) {
            return Err(Status::InvalidArgs);
        }
        if config.npt < n + 2 {
            return Err(Status::InvalidArgs);
        }
        if config.npt > (n + 1) * (n + 2) / 2 {
            return Err(Status::InvalidArgs);
        }
        if !(config.rho_begin.is_finite() && config.rho_begin > 0.0) {
            return Err(Status::InvalidArgs);
        }
        if !(config.rho_end > 0.0 && config.rho_end <= config.rho_begin) {
            return Err(Status::InvalidArgs);
        }
        if config.max_fun < config.npt + 1 {
            return Err(Status::InvalidArgs);
        }
        // NaN and +inf are both rejected: a target of +inf is never
        // meaningful — the very first (moderated, hence finite) evaluation would "reach" it.
        if config.f_target.is_nan() || config.f_target == f64::INFINITY {
            return Err(Status::InvalidArgs);
        }
        // Each ftol field, if set, must be finite and >= 0 (0.0 is legal — stops
        // only on exact stagnation). NaN fails the `>= 0.0` half of the test.
        for tol in [config.ftol_rel, config.ftol_abs].into_iter().flatten() {
            if !(tol.is_finite() && tol >= 0.0) {
                return Err(Status::InvalidArgs);
            }
        }
        if let Some(restart) = config.restart {
            if restart.max_restarts < 1 || restart.max_restarts > MAX_RESTARTS_CAP {
                return Err(Status::InvalidArgs);
            }
            if !(restart.improve_rel_tol >= 0.0) {
                return Err(Status::InvalidArgs);
            }
            // NaN fails the range test (the `!(..)` form is load-bearing, as above).
            if !(restart.cycle_budget_frac >= 0.0 && restart.cycle_budget_frac <= 1.0) {
                return Err(Status::InvalidArgs);
            }
        }
        Ok(Self {
            n,
            config,
            ws: bobyqb::SolverWs::new(n, config.npt).map_err(|_| Status::AllocationFailed)?,
            // The per-cycle instrumentation store: one slot per
            // possible restart plus one spare, so `minimize` fills it without reallocating.
            // `max_restarts <= MAX_RESTARTS_CAP` above keeps the `+ 1` overflow-free.
            cycle_boundaries: util::try_capacity(
                config.restart.map_or(0, |restart| restart.max_restarts + 1),
            )
            .map_err(|_| Status::AllocationFailed)?,
        })
    }

    /// Minimises `f` starting from `x` (overwritten with the best point
    /// found), subject to `lower <= x <= upper`, with no heap allocation in
    /// this call ([`Bobyqa::new`] owns all allocation).
    ///
    /// # Panics
    ///
    /// Never on numerical input — out-of-range arguments return an [`Outcome`]
    /// with [`Status::InvalidArgs`] rather than panicking.
    ///
    /// # Examples
    ///
    /// ```
    /// use bobyqa::{Bobyqa, Config};
    /// let mut solver = Bobyqa::new(2, Config::new(2)).unwrap();
    /// let mut x = [1.0, 2.0];
    /// let outcome = solver.minimize(
    ///     |p: &[f64]| p.iter().map(|v| v * v).sum::<f64>(),
    ///     &mut x,
    ///     &[-5.0, -5.0],
    ///     &[5.0, 5.0],
    /// );
    /// assert!(outcome.f < 1e-8);
    /// ```
    pub fn minimize<F: FnMut(&[f64]) -> f64>(
        &mut self,
        f: F,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
    ) -> Outcome {
        self.run(f, x, lower, upper, None)
    }

    /// [`Bobyqa::minimize`] with an objective that also receives the current
    /// [`TrustRadius`] at each evaluation, for objectives whose own accuracy can follow
    /// the solver's resolution (an inner solve run looser while `rho` is large).
    ///
    /// The algorithm is the same: an objective that ignores its second argument gets the
    /// same `x`, [`Outcome`] and evaluation sequence as [`Bobyqa::minimize`], bit for bit.
    /// No heap allocation in this call.
    ///
    /// # Panics
    ///
    /// Never on numerical input, as [`Bobyqa::minimize`].
    ///
    /// # Examples
    ///
    /// ```
    /// use bobyqa::{Bobyqa, Config};
    /// let mut solver = Bobyqa::new(2, Config::new(2)).unwrap();
    /// let mut x = [1.0, 2.0];
    /// let mut smallest_rho = f64::INFINITY;
    /// let outcome = solver.minimize_with_radius(
    ///     |p: &[f64], radius| {
    ///         smallest_rho = smallest_rho.min(radius.rho);
    ///         p.iter().map(|v| v * v).sum::<f64>()
    ///     },
    ///     &mut x,
    ///     &[-5.0, -5.0],
    ///     &[5.0, 5.0],
    /// );
    /// assert!(outcome.f < 1e-8);
    /// assert_eq!(smallest_rho, 1e-6); // reached rho_end
    /// ```
    pub fn minimize_with_radius<F: FnMut(&[f64], TrustRadius) -> f64>(
        &mut self,
        mut f: F,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
    ) -> Outcome {
        let radius = Cell::new(TrustRadius {
            rho: self.config.rho_begin,
            delta: self.config.rho_begin,
        });
        self.run(
            |p: &[f64]| f(p, radius.get()),
            x,
            lower,
            upper,
            Some(&radius),
        )
    }

    /// The body of both entry points. `radius`, when set, is written by the engine with the
    /// current `(rho, delta)` before every evaluation; it never feeds back into the solve.
    fn run<F: FnMut(&[f64]) -> f64>(
        &mut self,
        mut f: F,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        radius: Option<&Cell<TrustRadius>>,
    ) -> Outcome {
        self.cycle_boundaries.clear();
        if !prepare_call(self.n, &self.config, &mut self.ws, x, lower, upper) {
            // f is NaN because nothing was evaluated.
            return Outcome {
                f: f64::NAN,
                n_eval: 0,
                status: Status::InvalidArgs,
            };
        }

        let mut restart_state = self.config.restart.map(|rc| bobyqb::RestartState {
            config: rc,
            last_fopt: None,
            stall_count: 0,
            stall_fopt: None,
            nf_cycle_start: 0,
            cycle_boundaries: &mut self.cycle_boundaries,
        });
        // Shape guard: both-None must not even reach the check site — the
        // engine takes `None` and the default path is the literal existing code.
        let ftol = if self.config.ftol_rel.is_none() && self.config.ftol_abs.is_none() {
            None
        } else {
            Some((
                self.config.ftol_rel.unwrap_or(0.0),
                self.config.ftol_abs.unwrap_or(0.0),
            ))
        };
        let (fopt, nf, info) = bobyqb::bobyqb(
            &mut f,
            self.config.max_fun,
            self.config.npt,
            ETA1_DFT,
            ETA2_DFT,
            self.config.f_target,
            ftol,
            GAMMA1_DFT,
            GAMMA2_DFT,
            self.config.rho_begin,
            self.config.rho_end,
            x,
            &mut self.ws,
            restart_state.as_mut(),
            radius,
            // `Config::prima_parity`: the deviation applies on a fully determined model only.
            // `false` here is the literal PRIMA path.
            !self.config.prima_parity && self.config.npt == (self.n + 1) * (self.n + 2) / 2,
        );
        Outcome {
            f: fopt,
            n_eval: nf,
            status: status_from_info(info),
        }
    }

    /// Restarts performed on the last [`Bobyqa::minimize`] or [`Bobyqa::minimize_with_radius`]
    /// call — always 0 when [`Config::restart`] is `None` (and before the first call).
    #[must_use]
    pub fn last_restart_count(&self) -> usize {
        self.cycle_boundaries.len()
    }

    /// Cumulative evaluation count at each restart boundary of the last [`Bobyqa::minimize`]
    /// or [`Bobyqa::minimize_with_radius`] call — one entry per restart, so empty when none
    /// fired (and always empty when [`Config::restart`] is `None`). Diff each entry against
    /// the next (the last against [`Outcome::n_eval`]) for per-cycle evaluation costs.
    #[must_use]
    pub fn last_cycle_boundaries(&self) -> &[usize] {
        &self.cycle_boundaries
    }
}

/// One-shot convenience over [`Bobyqa`]: infers `n` from `x.len()`, builds a
/// throwaway solver, and runs a single minimisation. **Allocates per call** — hot loops
/// re-solving many problems of one size should build a [`Bobyqa`] once and reuse it.
///
/// [`Bobyqa::new`] errors ([`Status::InvalidArgs`] for bad `npt`/`rho`/`max_fun`/`f_target` or
/// `x.len() == 0`, the same `n = 0` rejection as `new`; [`Status::AllocationFailed`]) fold into
/// the shape `minimize` already uses for runtime rejection: `Outcome { f: NaN, n_eval: 0, status }`
/// with that status — never a nested `Result`.
///
/// # Panics
///
/// Never — construction and runtime rejections are reported through [`Status`], not panics.
///
/// # Examples
///
/// ```
/// use bobyqa::{bobyqa, Config};
/// let mut x = [1.0, 2.0];
/// let outcome = bobyqa(
///     |p: &[f64]| p.iter().map(|v| v * v).sum::<f64>(),
///     &mut x,
///     &[-5.0, -5.0],
///     &[5.0, 5.0],
///     Config::new(2),
/// );
/// assert!(outcome.f < 1e-8);
/// ```
pub fn bobyqa<F: FnMut(&[f64]) -> f64>(
    f: F,
    x: &mut [f64],
    lower: &[f64],
    upper: &[f64],
    config: Config,
) -> Outcome {
    match Bobyqa::new(x.len(), config) {
        Ok(mut solver) => solver.minimize(f, x, lower, upper),
        // `new`'s error status (InvalidArgs or AllocationFailed) passes through unchanged.
        Err(status) => Outcome {
            f: f64::NAN,
            n_eval: 0,
            status,
        },
    }
}

/// Upper bound on [`RestartConfig::max_restarts`]: far above any real
/// schedule (the recommended one is `1`), it exists so the `max_restarts + 1` instrumentation
/// store can neither overflow nor become a caller-sized giant allocation.
pub const MAX_RESTARTS_CAP: usize = 10_000;

// The workspace-size overflow gate: every add/product any workspace buffer will
// compute, checked end to end. Sums are checked BEFORE their products — `npt + n`, `n + 1`,
// `n + 2` can each wrap before a `checked_mul` ever runs (e.g. `n = usize::MAX - 1` makes
// `n + 2` wrap to 0). On 32-bit targets (wasm32, thumbv7em) this is what stands between a
// large `n` and a wrapped `Mat` size that stays in-bounds and returns garbage silently;
// PRIMA guards the same class in preproc.f90 L197.
fn ws_dims_ok(n: usize, npt: usize) -> bool {
    n.checked_add(1)
        .zip(n.checked_add(2))
        .and_then(|(a, b)| a.checked_mul(b)) // the npt cap formula (n + 1)(n + 2)
        .is_some()
        && npt
            .checked_add(n) // npt + n itself (vlag/bmat cols), then (npt + n) * n (bmat)
            .and_then(|s| s.checked_mul(n))
            .is_some()
        && npt.checked_mul(npt).is_some() // dominates zmat's npt * (npt - n - 1)
        && npt.checked_mul(n).is_some() // xpt
        && n.checked_mul(n).is_some() // hq
}

// The per-call checks: slice lengths; bounds NaN-free, ordered, and at least
// `2 * rho_begin` apart (PRIMA's `NO_SPACE_BETWEEN_BOUNDS`, caught up front; +/-inf bounds
// are legal); x NaN-free. Config repair is rejected, x-space handling stays faithful to
// PRIMA. Ordering and gap are judged on the ±BOUNDMAX-clamped bounds, mirroring PRIMA's
// clamp-then-check order: a bound beyond ±BOUNDMAX passes the raw checks yet clamps to a
// crossed or too-narrow box in `minimize`. Clamping only shrinks the box, so this is
// strictly stronger than the raw checks and identical for every bound within ±BOUNDMAX.
fn args_are_valid(n: usize, config: &Config, x: &[f64], lower: &[f64], upper: &[f64]) -> bool {
    x.len() == n
        && lower.len() == n
        && upper.len() == n
        && !lower.iter().chain(upper).any(|v| v.is_nan())
        && lower.iter().zip(upper).all(|(l, u)| {
            let (cl, cu) = (l.max(-BOUNDMAX), u.min(BOUNDMAX));
            cl <= cu && cu - cl >= 2.0 * config.rho_begin
        })
        && !x.iter().any(|v| v.is_nan())
}

// `Bobyqa::minimize`'s per-call preamble: validate args,
// clamp bounds into `ws`, preproc x0. Returns `false` (leaving `ws`/`x` untouched beyond
// whatever `args_are_valid` itself reads) when the runtime args are rejected.
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA
fn prepare_call(
    n: usize,
    config: &Config,
    ws: &mut bobyqb::SolverWs,
    x: &mut [f64],
    lower: &[f64],
    upper: &[f64],
) -> bool {
    if !args_are_valid(n, config, x, lower, upper) {
        return false;
    }
    // PRIMA bobyqa.f90 L287-301: clamp bounds at +/-BOUNDMAX ("no bound" sentinel,
    // consts.F90 L172). NaN bounds were rejected above; only the magnitude clamp remains.
    // The clamped copies live in the workspace so that `minimize` allocates nothing per call.
    for i in 0..n {
        ws.bobyqb.xl[i] = lower[i].max(-BOUNDMAX);
        ws.bobyqb.xu[i] = upper[i].min(BOUNDMAX);
    }

    // PRIMA bobyqa.f90 L316: x = max(xl, min(xu, moderatex(x))) — in place, elementwise
    // (each x[i] depends only on x[i], so this is FP-identical
    // to PRIMA's whole-array form).
    for i in 0..n {
        let xm = moderatex1(x[i]);
        x[i] = ws.bobyqb.xl[i].max(ws.bobyqb.xu[i].min(xm));
    }

    // PRIMA preproc.f90 L341-350 (HONOUR_X0 = FALSE — the path the oracle runs):
    // revise X0 so its distance to each inactive bound is 0 or >= rhobeg. Valid because
    // validation guarantees XU - XL >= 2*RHOBEG and X is in the box (the L338 precondition).
    // The follow-up rhobeg-revision block (preproc.f90 L367-383) is omitted: after this
    // revision it is "unnecessary in precise arithmetic" (PRIMA's own L368 N.B.), and its
    // rounding-error repairs fall under the no-repair stance — validation rejects, never fixes.
    util::revise_x0(x, &ws.bobyqb.xl, &ws.bobyqb.xu, config.rho_begin);
    true
}

/// The restart schedule, plugged in via [`Config::restart`]. A restart puts `rho`/`delta`
/// back to `rho_begin` and rebuilds the interpolation set from scratch around the best point
/// found so far, then carries on; the returned point is the best over every cycle, so a
/// restart can never return something worse than stopping would have.
///
/// `Config::max_fun` stays the TOTAL evaluation budget across all restarts.
/// Start from [`RestartConfig::new`] and assign fields to override
/// (`#[non_exhaustive]`, like [`Config`]).
///
/// Interaction with [`Config::ftol_rel`]/[`Config::ftol_abs`]: a triggered ftol stop
/// ([`Status::FtolReached`]) is a converged-class outcome and does not spend a restart
/// cycle — when both are configured and both would fire at the same rho reduction, ftol
/// wins and the solve stops.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RestartConfig {
    /// Cap on restarts before returning the last cycle's result — at least 1 and at most
    /// [`MAX_RESTARTS_CAP`] ([`Bobyqa::new`] rejects both extremes). The recommended schedule
    /// pairs this at `1` with [`cycle_budget_frac`](Self::cycle_budget_frac) `0.125`: a long
    /// schedule divides a fixed `max_fun` into cycles too short to descend, so each one
    /// rebuilds a model it then cannot exploit.
    pub max_restarts: usize,
    /// Stop restarting once a full cycle improves the best value by less than this,
    /// relative to `max(1, |f|)` — the settle test. The same threshold also defines a
    /// *stalled* `rho` reduction for [`stall_reductions`](Self::stall_reductions): "an
    /// improvement this small is not worth chasing" is one judgement, applied at both scales.
    pub improve_rel_tol: f64,
    /// Consecutive `rho` reductions improving the best value by less than
    /// [`improve_rel_tol`](Self::improve_rel_tol) (relative to `max(1, |f|)`) that trigger a
    /// restart before `rho` reaches `rho_end`, cutting a stalled `rho` tail instead of paying
    /// for the rest of it. `0` disables this trigger.
    ///
    /// It only fires where the solve takes `rho` reductions at all; a cycle that crawls
    /// without reducing `rho` has no site for it to fire from, which is what
    /// [`cycle_budget_frac`](Self::cycle_budget_frac) — the documented trigger — covers
    /// instead.
    ///
    /// Whichever the trigger, the final cycle — once no restart remains — always runs down to
    /// `rho_end`, so the returned point is never coarser than a plain solve's.
    pub stall_reductions: usize,
    /// The eval-cap trigger, and the documented way to drive the schedule: restart once the
    /// current cycle has spent at least this fraction of the evaluation budget that *remained
    /// when the cycle started*, without settling. `0.0` disables it. Consulted at every
    /// trust-region iteration, so unlike [`stall_reductions`](Self::stall_reductions) it fires
    /// on a cycle that is crawling without reducing `rho`. Measuring against the remainder
    /// rather than `max_fun` makes successive cut points geometric, so the schedule is
    /// self-limiting.
    ///
    /// Recommended: `0.125` together with [`max_restarts`](Self::max_restarts) `1` (measured
    /// on large LMM fits).
    ///
    /// **Setting this changes what the `rho_end` trigger does.** With the cap on, reaching
    /// `rho_end` no longer restarts by itself — it restarts only when the cap agrees the cycle
    /// was expensive. The user-visible consequence: a solve that converges well inside its cap
    /// is left alone and returns exactly what [`Config::restart`] `None` returns, evaluation
    /// count included. With the cap off (`0.0`), `rho_end` restarts on the settle test as
    /// before, so the settle and stall schedules are unchanged.
    pub cycle_budget_frac: f64,
}

impl RestartConfig {
    /// The recommended schedule: one restart, triggered when a cycle has spent an eighth of
    /// the budget that remained when it started, stopping when a cycle's improvement falls
    /// below `1e-6` relative to `max(1, |f|)`.
    ///
    /// [`stall_reductions`](Self::stall_reductions) is off here: the eval cap covers the
    /// stalled-tail case and fires on crawling cycles the stall counter cannot see. A solve
    /// that finishes well inside its cap is untouched by this schedule — same point, same
    /// evaluation count as [`Config::restart`] `None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_restarts: 1,
            improve_rel_tol: 1e-6,
            stall_reductions: 0,
            cycle_budget_frac: 0.125,
        }
    }
}
impl Default for RestartConfig {
    fn default() -> Self {
        Self::new()
    }
}

// The PRIMA-info -> `Status` mapping. `SMALL_TR_RADIUS` shares PRIMA's value 0 with `INFO_DFT`
// (a normal loop exit IS convergence). `MAXTR_REACHED` is budget-class (`maxtr = 2 * max_fun` with
// restarts off, scaled by `max_restarts + 1` when restarts are enabled to absorb a worst-case zero-eval
// iteration burst per restart cycle; near-unreachable on both paths). `NAN_INF_X`/`NAN_INF_F` are
// `checkexit` defensive guards, near-unreachable
// behind `moderatex`/`moderatef` — numerical-breakdown class. `NO_SPACE_BETWEEN_BOUNDS` is
// caught by validation before the loop. `TRSUBP_FAILED` is never emitted by the BOBYQA port
// (`trsbox` returns only CRVMIN, no info code) — the constant exists for completeness against
// PRIMA's infos.f90, so it is absent here.
fn status_from_info(info: i32) -> Status {
    use crate::consts::{
        DAMAGING_ROUNDING, FTARGET_ACHIEVED, FTOL_REACHED, MAXFUN_REACHED, MAXTR_REACHED,
        NAN_INF_F, NAN_INF_MODEL, NAN_INF_X, SMALL_TR_RADIUS,
    };
    match info {
        SMALL_TR_RADIUS => Status::Converged,
        FTARGET_ACHIEVED => Status::TargetReached,
        FTOL_REACHED => Status::FtolReached,
        MAXFUN_REACHED | MAXTR_REACHED => Status::MaxFunReached,
        NAN_INF_MODEL | DAMAGING_ROUNDING | NAN_INF_X | NAN_INF_F => Status::ModelDegenerate,
        // bobyqb's info set is closed; a new code here is a port bug to
        // raise, never a silent mapping.
        other => {
            debug_assert!(false, "unmapped PRIMA info {other}");
            Status::ModelDegenerate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> (usize, Config) {
        (2, Config::new(2))
    }

    fn sphere(p: &[f64]) -> f64 {
        p.iter().map(|v| v * v).sum()
    }

    #[test]
    fn config_new_returns_prima_defaults() {
        let c = Config::new(3);
        assert_eq!(c.npt, 7); // 2n + 1
        assert_eq!(c.rho_begin, 1.0);
        assert_eq!(c.rho_end, 1e-6);
        assert_eq!(c.max_fun, 1500); // 500 * n
        assert_eq!(c.f_target, f64::NEG_INFINITY);
        assert_eq!(c.ftol_rel, None); // ftol off by default
        assert_eq!(c.ftol_abs, None);
        assert!(c.prima_parity); // deviations off by default
    }

    #[test]
    fn new_rejects_bad_ftol_values_and_accepts_zero() {
        let (n, c) = valid();
        for bad in [f64::NAN, -1.0, f64::INFINITY] {
            assert!(
                Bobyqa::new(
                    n,
                    Config {
                        ftol_rel: Some(bad),
                        ..c
                    }
                )
                .is_err()
            );
            assert!(
                Bobyqa::new(
                    n,
                    Config {
                        ftol_abs: Some(bad),
                        ..c
                    }
                )
                .is_err()
            );
        }
        // Some(0.0) is legal (stops only on exact stagnation), alone or together.
        assert!(
            Bobyqa::new(
                n,
                Config {
                    ftol_rel: Some(0.0),
                    ftol_abs: Some(0.0),
                    ..c
                }
            )
            .is_ok()
        );
        assert!(
            Bobyqa::new(
                n,
                Config {
                    ftol_abs: Some(1e-8),
                    ..c
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn new_accepts_the_default_config_and_the_npt_extremes() {
        let (n, c) = valid();
        assert!(Bobyqa::new(n, c).is_ok());
        assert!(Bobyqa::new(n, Config { npt: 4, ..c }).is_ok()); // n + 2
        assert!(Bobyqa::new(n, Config { npt: 6, ..c }).is_ok()); // (n+1)(n+2)/2
    }

    #[test]
    fn new_rejects_bad_n_npt_rho_maxfun_ftarget() {
        let (n, c) = valid();
        assert!(Bobyqa::new(0, Config::new(1)).is_err());
        assert!(Bobyqa::new(n, Config { npt: 3, ..c }).is_err()); // < n + 2: PRIMA preproc would clamp; we reject
        assert!(Bobyqa::new(n, Config { npt: 7, ..c }).is_err()); // > (n+1)(n+2)/2
        assert!(Bobyqa::new(n, Config { rho_end: 0.0, ..c }).is_err());
        assert!(Bobyqa::new(n, Config { rho_end: 2.0, ..c }).is_err()); // > rho_begin
        assert!(
            Bobyqa::new(
                n,
                Config {
                    rho_begin: 0.0,
                    ..c
                }
            )
            .is_err()
        ); // rho_begin must be > 0
        assert!(
            Bobyqa::new(
                n,
                Config {
                    rho_begin: -1.0,
                    ..c
                }
            )
            .is_err()
        );
        assert!(
            Bobyqa::new(
                n,
                Config {
                    rho_begin: f64::INFINITY,
                    rho_end: 1.0,
                    ..c
                }
            )
            .is_err()
        );
        assert!(Bobyqa::new(n, Config { max_fun: 5, ..c }).is_err()); // < npt + 1: PRIMA preproc would raise; we reject
        assert!(
            Bobyqa::new(
                n,
                Config {
                    f_target: f64::NAN,
                    ..c
                }
            )
            .is_err()
        );
        // f_target = +inf would "succeed" on the first evaluation — rejected like NaN;
        // -inf stays the documented "off" default and must keep passing.
        assert!(
            Bobyqa::new(
                n,
                Config {
                    f_target: f64::INFINITY,
                    ..c
                }
            )
            .is_err()
        );
        assert!(
            Bobyqa::new(
                n,
                Config {
                    f_target: f64::NEG_INFINITY,
                    ..c
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn ws_dims_ok_rejects_overflowing_dimension_arithmetic() {
        // Realistic shapes pass, including the npt cap boundary.
        assert!(ws_dims_ok(2, 5));
        assert!(ws_dims_ok(100, 101 * 102 / 2));
        // n + 2 wraps to 0 BEFORE any product — the checked-add-then-mul chain must catch it.
        assert!(!ws_dims_ok(usize::MAX - 1, 5));
        assert!(!ws_dims_ok(usize::MAX, usize::MAX));
        // npt-driven overflows: npt * npt and npt + n.
        assert!(!ws_dims_ok(2, usize::MAX));
        assert!(!ws_dims_ok(2, usize::MAX / 2));
        // n * n overflow at the half-width boundary.
        assert!(!ws_dims_ok(1_usize << (usize::BITS / 2), 5));
    }

    #[test]
    fn new_rejects_dimension_overflow_instead_of_panicking() {
        // Config::new saturates its derived npt/max_fun; the overflow gate (or the range checks
        // the saturation then fails) must reject — in debug builds a panic here is the bug.
        for n in [usize::MAX, usize::MAX - 1, usize::MAX / 2] {
            assert!(matches!(
                Bobyqa::new(n, Config::new(n)),
                Err(Status::InvalidArgs)
            ));
        }
        // A hand-built config with an overflowing npt, past the saturation path.
        let mut c = Config::new(2);
        c.npt = usize::MAX;
        c.max_fun = usize::MAX;
        assert!(matches!(Bobyqa::new(2, c), Err(Status::InvalidArgs)));
    }

    // The overflow conditions are untestable with real allocations on 64-bit hosts; this runs on
    // CI's wasm32 job: default npt at n = 65_535 makes npt * npt
    // (and the npt cap formula) overflow 32-bit usize.
    #[test]
    #[cfg(target_pointer_width = "32")]
    fn new_rejects_sizes_that_overflow_32bit_usize() {
        let n = 65_535;
        assert!(matches!(
            Bobyqa::new(n, Config::new(n)),
            Err(Status::InvalidArgs)
        ));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn new_reports_allocation_failure_instead_of_aborting() {
        // n = 2^20 with npt = 2^31 passes every range check, and the first large buffer `new`
        // asks for (bmat, n x (npt + n) f64) is 16 PB — more than a 64-bit address space
        // hands out, so the request fails at once without touching memory. The contract:
        // `AllocationFailed`, never an abort and never another rejection.
        let n = 1 << 20;
        let mut c = Config::new(n);
        c.npt = 1 << 31;
        c.max_fun = usize::MAX;
        assert!(matches!(Bobyqa::new(n, c), Err(Status::AllocationFailed)));
    }

    fn invalid_outcome(o: Outcome) {
        assert_eq!(o.status, Status::InvalidArgs);
        assert!(o.f.is_nan());
        assert_eq!(o.n_eval, 0);
    }

    #[test]
    fn minimize_rejects_bad_runtime_arguments() {
        let (n, c) = valid();
        let mut s = Bobyqa::new(n, c).unwrap();
        let f = |x: &[f64]| x[0];
        invalid_outcome(s.minimize(f, &mut [0.0], &[-1.0, -1.0], &[1.0, 1.0])); // x len
        invalid_outcome(s.minimize(f, &mut [0.0, 0.0], &[-1.0], &[1.0, 1.0])); // lower len
        invalid_outcome(s.minimize(f, &mut [0.0, 0.0], &[-1.0, -1.0], &[1.0])); // upper len
        invalid_outcome(s.minimize(f, &mut [0.0, 0.0], &[f64::NAN, -1.0], &[1.0, 1.0])); // lower NaN
        invalid_outcome(s.minimize(f, &mut [0.0, 0.0], &[-1.0, -1.0], &[f64::NAN, 1.0])); // upper NaN: exercises the `upper` half of the .chain() the lower-only case can't
        invalid_outcome(s.minimize(f, &mut [0.0, 0.0], &[2.0, -1.0], &[1.0, 1.0])); // crossed
        invalid_outcome(s.minimize(f, &mut [0.0, 0.0], &[-0.5, -1.0], &[0.5, 1.0])); // upper-lower < 2*rho_begin
        invalid_outcome(s.minimize(f, &mut [f64::NAN, 0.0], &[-9.0, -9.0], &[9.0, 9.0]));
    }

    #[test]
    fn minimize_rejects_bounds_that_cross_or_collapse_after_the_boundmax_clamp() {
        // A bound beyond ±BOUNDMAX passes the raw ordering/gap checks (lower <= upper, gap
        // = +inf) yet clamps to a crossed or zero-width box — validation must judge the
        // clamped values, like PRIMA (clamp at bobyqa.f90 L287-301 precedes its checks).
        let (n, c) = valid();
        let mut s = Bobyqa::new(n, c).unwrap();
        let f = |x: &[f64]| x[0];
        // upper < -BOUNDMAX: clamps to xl = -BOUNDMAX > xu = -1e308 (crossed).
        let below = -1e308;
        invalid_outcome(s.minimize(
            f,
            &mut [0.0, 0.0],
            &[f64::NEG_INFINITY, -9.0],
            &[below, 9.0],
        ));
        // upper = -BOUNDMAX exactly: clamps to a zero-width box (gap < 2 * rho_begin).
        invalid_outcome(s.minimize(
            f,
            &mut [0.0, 0.0],
            &[f64::NEG_INFINITY, -9.0],
            &[-BOUNDMAX, 9.0],
        ));
    }

    #[test]
    #[expect(clippy::many_single_char_names)] // n, c, s, x, o are PRIMA/test shorthands
    fn minimize_converges_on_the_sphere_and_reports_the_trajectory_floor() {
        let (n, c) = valid();
        let mut s = Bobyqa::new(n, c).unwrap();
        let mut n_calls = 0usize;
        let mut x = [1.0, 2.0];
        let o = s.minimize(
            |p: &[f64]| {
                n_calls += 1;
                sphere(p)
            },
            &mut x,
            &[-5.0, -5.0],
            &[5.0, 5.0],
        );
        assert_eq!(o.status, Status::Converged);
        assert_eq!(o.n_eval, n_calls);
        assert!(o.f < 1e-8);
        assert!(x.iter().all(|v| v.abs() < 1e-3));
        assert!(o.n_eval >= c.npt); // the npt model-building floor
    }

    #[test]
    fn minimize_projects_a_near_bound_start_onto_the_bound_like_prima_preproc() {
        // preproc.f90 L341-343: x0 within rhobeg/2 of a bound starts ON the bound — the first
        // evaluation (at XBASE) must sit exactly there.
        let (n, c) = valid(); // rho_begin = 1.0
        let mut s = Bobyqa::new(n, c).unwrap();
        let mut first_x0 = f64::NAN;
        let mut seen = false;
        let mut x = [1.4, 0.3]; // 0.4 (< rho_begin/2) above lower[0] = 1.0
        s.minimize(
            |p: &[f64]| {
                if !seen {
                    first_x0 = p[0];
                    seen = true;
                }
                sphere(p)
            },
            &mut x,
            &[1.0, -5.0],
            &[6.0, 5.0],
        );
        assert_eq!(first_x0, 1.0);
    }

    #[test]
    fn minimize_stops_on_f_target_and_on_the_budget() {
        let (n, c) = valid();
        let mut s = Bobyqa::new(n, Config { f_target: 0.5, ..c }).unwrap();
        let o = s.minimize(sphere, &mut [1.0, 2.0], &[-5.0, -5.0], &[5.0, 5.0]);
        assert_eq!(o.status, Status::TargetReached);
        assert!(o.f <= 0.5);

        let mut s = Bobyqa::new(n, Config { max_fun: 6, ..c }).unwrap(); // npt + 1
        let o = s.minimize(sphere, &mut [1.0, 2.0], &[-5.0, -5.0], &[5.0, 5.0]);
        assert_eq!(o.status, Status::MaxFunReached);
        assert_eq!(o.n_eval, 6);
    }

    #[test]
    fn all_infeasible_objective_is_detectable_via_funcmax_despite_converged() {
        // An objective that is +inf everywhere: `moderatef` maps every evaluation to
        // FUNCMAX, so the interpolation set is a flat, finite surface and BOBYQA exits
        // faithfully as Converged (SMALL_TR_RADIUS on a flat model). The public FUNCMAX
        // const is how a caller distinguishes this degenerate run from a real one — the
        // contract documented on `FUNCMAX` / `Outcome::f`.
        let (n, c) = valid();
        let mut s = Bobyqa::new(n, c).unwrap();
        let o = s.minimize(
            |_: &[f64]| f64::INFINITY,
            &mut [1.0, 2.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
        );
        assert_eq!(o.status, Status::Converged); // faithful PRIMA: flat moderated surface
        assert!(
            o.f >= FUNCMAX,
            "degenerate exit must be detectable: f = {} < FUNCMAX",
            o.f
        );
        assert!(
            !o.found_finite(),
            "all-infeasible run must report found_finite() == false"
        );

        // A normal fit finds finite values → found_finite() == true.
        let mut s = Bobyqa::new(n, c).unwrap();
        let good = s.minimize(sphere, &mut [1.0, 2.0], &[-5.0, -5.0], &[5.0, 5.0]);
        assert!(good.found_finite());
    }

    #[test]
    #[expect(clippy::many_single_char_names)] // n, c, s, x, o are PRIMA/test shorthands
    fn minimize_accepts_infinite_bounds_via_the_boundmax_clamp() {
        // |bound| >= BOUNDMAX means "no bound"; minimize clamps to +/-BOUNDMAX
        // (bobyqa.f90 L287-301) and must run, not panic.
        let (n, c) = valid();
        let mut s = Bobyqa::new(n, c).unwrap();
        let mut x = [1.0, 2.0];
        let o = s.minimize(
            sphere,
            &mut x,
            &[f64::NEG_INFINITY, -9.0],
            &[f64::INFINITY, 9.0],
        );
        assert_eq!(o.status, Status::Converged);
        assert!(o.f < 1e-8);
    }

    #[test]
    fn status_from_info_maps_every_reachable_prima_code() {
        use crate::consts::*;
        assert_eq!(status_from_info(SMALL_TR_RADIUS), Status::Converged);
        assert_eq!(status_from_info(FTARGET_ACHIEVED), Status::TargetReached);
        assert_eq!(status_from_info(FTOL_REACHED), Status::FtolReached);
        assert_eq!(status_from_info(MAXFUN_REACHED), Status::MaxFunReached);
        assert_eq!(status_from_info(MAXTR_REACHED), Status::MaxFunReached);
        assert_eq!(status_from_info(NAN_INF_MODEL), Status::ModelDegenerate);
        assert_eq!(status_from_info(DAMAGING_ROUNDING), Status::ModelDegenerate);
        assert_eq!(status_from_info(NAN_INF_X), Status::ModelDegenerate);
        assert_eq!(status_from_info(NAN_INF_F), Status::ModelDegenerate);
    }

    #[test]
    #[cfg(feature = "std")]
    fn status_displays_and_is_an_error() {
        use alloc::string::ToString;
        for (status, text) in [
            (
                Status::Converged,
                "the trust-region radius reached rho_end, with any restart schedule settled",
            ),
            (Status::TargetReached, "an evaluation reached f_target"),
            (
                Status::FtolReached,
                "the best f improved by less than the ftol tolerance over one rho stage",
            ),
            (
                Status::MaxFunReached,
                "the max_fun evaluation budget was exhausted",
            ),
            (
                Status::ModelDegenerate,
                "the interpolation model degenerated beyond rescue",
            ),
            (
                Status::InvalidArgs,
                "invalid arguments: bad bounds, npt, sizes, or config",
            ),
            (
                Status::AllocationFailed,
                "the solver workspace could not be allocated",
            ),
        ] {
            let e: &dyn std::error::Error = &status;
            assert_eq!(e.to_string(), text);
        }
    }

    #[test]
    fn restart_config_new_returns_the_documented_defaults() {
        let r = RestartConfig::new();
        assert_eq!(r.max_restarts, 1);
        assert_eq!(r.improve_rel_tol, 1e-6);
        assert_eq!(r.stall_reductions, 0);
        assert_eq!(r.cycle_budget_frac, 0.125);
    }

    // Builds the `Config` the way a downstream crate must under `#[non_exhaustive]`:
    // `Config::new` + field assignment, restart knobs included.
    fn with_restart(n: usize, r: RestartConfig) -> Config {
        let mut c = Config::new(n);
        c.restart = Some(r);
        c
    }

    #[test]
    fn new_accepts_valid_and_rejects_bad_restart_knobs() {
        let r = RestartConfig::new();
        let ok = |rc: RestartConfig| Bobyqa::new(2, with_restart(2, rc)).is_ok();
        assert!(ok(r)); // npt = 5
        // max_restarts >= 1
        assert!(!ok(RestartConfig {
            max_restarts: 0,
            ..r
        }));
        // max_restarts <= MAX_RESTARTS_CAP (kills both the `+ 1` overflow and the
        // caller-sized giant boundary store); the cap itself is legal.
        assert!(!ok(RestartConfig {
            max_restarts: usize::MAX,
            ..r
        }));
        assert!(!ok(RestartConfig {
            max_restarts: MAX_RESTARTS_CAP + 1,
            ..r
        }));
        assert!(ok(RestartConfig {
            max_restarts: MAX_RESTARTS_CAP,
            ..r
        }));
        // improve_rel_tol >= 0
        assert!(!ok(RestartConfig {
            improve_rel_tol: -1.0,
            ..r
        }));
        // stall_reductions is unconstrained: any count is as valid as the default 0 (off)
        assert!(ok(RestartConfig {
            stall_reductions: 2,
            ..r
        }));
        // cycle_budget_frac in [0.0, 1.0], NaN rejected
        assert!(ok(RestartConfig {
            cycle_budget_frac: 1.0,
            ..r
        }));
        assert!(!ok(RestartConfig {
            cycle_budget_frac: -0.1,
            ..r
        }));
        assert!(!ok(RestartConfig {
            cycle_budget_frac: 1.5,
            ..r
        }));
        assert!(!ok(RestartConfig {
            cycle_budget_frac: f64::NAN,
            ..r
        }));
        // the plain Config knobs are validated the same with restarts on
        assert!(Bobyqa::new(0, with_restart(1, r)).is_err());
    }

    #[test]
    fn restart_accessors_are_empty_before_any_minimize() {
        let s = Bobyqa::new(2, with_restart(2, RestartConfig::new())).unwrap();
        assert_eq!(s.last_restart_count(), 0);
        assert_eq!(s.last_cycle_boundaries(), []);
    }

    #[test]
    fn one_shot_bobyqa_minimizes_and_folds_construction_errors_into_the_outcome() {
        // Happy path: same sphere as the reusable-solver test.
        let mut x = [1.0, 2.0];
        let o = bobyqa(sphere, &mut x, &[-5.0, -5.0], &[5.0, 5.0], Config::new(2));
        assert_eq!(o.status, Status::Converged);
        assert!(o.f < 1e-8);

        // Construction rejections arrive as the InvalidArgs Outcome, not a Result.
        let f = |p: &[f64]| p[0];
        invalid_outcome(bobyqa(f, &mut [], &[], &[], Config::new(1))); // x.len() == 0 -> n = 0
        let bad_npt = Config {
            npt: 99,
            ..Config::new(2)
        };
        invalid_outcome(bobyqa(
            f,
            &mut [0.0, 0.0],
            &[-9.0, -9.0],
            &[9.0, 9.0],
            bad_npt,
        ));
    }
}
