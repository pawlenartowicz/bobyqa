//! A pure-Rust, dependency-free port of M. J. D. Powell's BOBYQA (Bound
//! Optimization BY Quadratic Approximation) — a derivative-free,
//! box-constrained local optimizer, ported from PRIMA's modern Fortran.
//!
//! Trajectory-parity-tested **bit-exact** against PRIMA (natively and on
//! `wasm32-wasip1`).
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

#![forbid(unsafe_code)]

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
pub mod powalg; // F0a: counter must be reachable from tests/kernel_counts.rs under this feature only
mod rescue;
#[cfg(test)]
mod test_support;
mod trustregion;
mod update;
mod util;

use consts::{
    BOUNDMAX, ETA1_DFT, ETA2_DFT, GAMMA1_DFT, GAMMA2_DFT, MAXFUN_DIM_DFT, RHOBEG_DFT, RHOEND_DFT,
};
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
    /// NaN is rejected by [`Bobyqa::new`].
    pub f_target: f64,
    /// Restart schedule. `None` (the default) is plain BOBYQA — bit-identical to 0.1.x:
    /// the solve ends the moment `rho` reaches `rho_end`. `Some` starts a new cycle instead:
    /// `rho`/`delta` go back to `rho_begin` and the interpolation set is rebuilt from scratch
    /// around the best point found — see [`RestartConfig`] for what triggers that and what
    /// stops it. `max_fun` stays the TOTAL evaluation budget across all cycles.
    pub restart: Option<RestartConfig>,
}

impl Config {
    /// PRIMA's defaults for an `n`-dimensional problem.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            npt: 2 * n + 1,
            rho_begin: RHOBEG_DFT,
            rho_end: RHOEND_DFT,
            max_fun: MAXFUN_DIM_DFT * n,
            // PRIMA's FTARGET_DFT is -REALMAX, which would terminate on f = -REALMAX;
            // -inf is strictly "off" (design §4.2).
            f_target: f64::NEG_INFINITY,
            // Off by default (stall-restart spec D5): existing users' numerics must not move.
            restart: None,
        }
    }
}

/// Why the solver stopped (or why construction failed).
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
    /// The `max_fun` evaluation budget was exhausted.
    MaxFunReached,
    /// The rescue procedure could not restore the interpolation geometry.
    ModelDegenerate,
    /// Bad bounds, `npt`, or slice sizes.
    InvalidArgs,
}

impl core::fmt::Display for Status {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Status::Converged => {
                "the trust-region radius reached rho_end, with any restart schedule settled"
            }
            Status::TargetReached => "an evaluation reached f_target",
            Status::MaxFunReached => "the max_fun evaluation budget was exhausted",
            Status::ModelDegenerate => "the interpolation model degenerated beyond rescue",
            Status::InvalidArgs => "invalid arguments: bad bounds, npt, sizes, or config",
        })
    }
}

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

/// The result of one [`Bobyqa::minimize`] call.
#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    /// Best objective value found. `NaN` when `status` is [`Status::InvalidArgs`] —
    /// nothing was evaluated. A value `>= `[`FUNCMAX`] means every evaluated point was
    /// non-finite (see [`FUNCMAX`]): the run is degenerate even if `status` is
    /// [`Status::Converged`].
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
    /// sites for intent. `NaN` `f` (the [`Status::InvalidArgs`] no-evaluation case)
    /// is also reported as not-finite.
    #[must_use]
    pub fn found_finite(&self) -> bool {
        self.f < FUNCMAX
    }
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
    last_restarts: usize,
    /// Per-cycle instrumentation backing store (stall-restart spec §5): sized
    /// `max_restarts + 1` in [`Bobyqa::new`] when restarts are enabled (empty otherwise), lent
    /// to the engine per call, so filling it — at most one entry per restart — never allocates
    /// inside `minimize`.
    cycle_boundaries: Vec<usize>,
}

impl Bobyqa {
    /// Allocates all scratch for an `n`-dimensional problem with `config`.
    ///
    /// # Errors
    ///
    /// [`Status::InvalidArgs`] when `(n, config)` is rejected: `n = 0`; `npt`
    /// outside `n + 2 ..= (n + 1)(n + 2) / 2` (PRIMA's preprocessing would
    /// clamp; we reject); `rho_begin` not a positive finite number; `rho_end`
    /// not in `(0, rho_begin]`; `max_fun <= npt` (PRIMA preproc would raise; we
    /// reject); `f_target` NaN. With `restart` set, additionally: `max_restarts`
    /// zero; `improve_rel_tol` negative or NaN; `cycle_budget_frac` outside
    /// `[0.0, 1.0]` or NaN.
    ///
    /// # Panics
    ///
    /// Never — invalid `(n, config)` is reported through [`Status::InvalidArgs`].
    #[expect(clippy::neg_cmp_op_on_partial_ord)] // `!(a >= b)` is load-bearing for NaN — never `a < b`
    pub fn new(n: usize, config: Config) -> Result<Self, Status> {
        if n == 0 {
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
        if config.f_target.is_nan() {
            return Err(Status::InvalidArgs);
        }
        if let Some(restart) = config.restart {
            if restart.max_restarts < 1 {
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
            ws: bobyqb::SolverWs::new(n, config.npt),
            last_restarts: 0,
            // The per-cycle instrumentation store (stall-restart spec §5): one slot per
            // possible restart plus one spare, so `minimize` fills it without reallocating.
            cycle_boundaries: match config.restart {
                Some(restart) => Vec::with_capacity(restart.max_restarts + 1),
                None => Vec::new(),
            },
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
        self.last_restarts = 0;
        self.cycle_boundaries.clear();
        if !prepare_call(self.n, &self.config, &mut self.ws, x, lower, upper) {
            // f is NaN because nothing was evaluated.
            return Outcome {
                f: f64::NAN,
                n_eval: 0,
                status: Status::InvalidArgs,
            };
        }
        let rhobeg = self.config.rho_begin;

        // The boundary store is lent to the engine by `take` (a pointer swap, no allocation)
        // and recovered below, keeping `RestartState` free of borrows into `self`.
        let mut restart_state = self.config.restart.map(|rc| bobyqb::RestartState {
            config: rc,
            restarts_done: 0,
            last_fopt: None,
            stall_count: 0,
            stall_fopt: None,
            nf_cycle_start: 0,
            cycle_boundaries: core::mem::take(&mut self.cycle_boundaries),
        });
        let mut f = f;
        let (fopt, nf, info) = bobyqb::bobyqb(
            &mut f,
            self.config.max_fun,
            self.config.npt,
            ETA1_DFT,
            ETA2_DFT,
            self.config.f_target,
            GAMMA1_DFT,
            GAMMA2_DFT,
            rhobeg,
            self.config.rho_end,
            x,
            &mut self.ws,
            restart_state.as_mut(),
        );
        if let Some(rs) = restart_state {
            self.last_restarts = rs.restarts_done;
            self.cycle_boundaries = rs.cycle_boundaries;
        }
        Outcome {
            f: fopt,
            n_eval: nf,
            status: status_from_info(info),
        }
    }

    /// Restarts performed on the last [`Bobyqa::minimize`] call — always 0 when
    /// [`Config::restart`] is `None` (and before the first call).
    #[must_use]
    pub fn last_restart_count(&self) -> usize {
        self.last_restarts
    }

    /// Cumulative evaluation count at each restart boundary of the last [`Bobyqa::minimize`]
    /// call — one entry per restart, so empty when none fired (and always empty when
    /// [`Config::restart`] is `None`). Diff each entry against the next (the last against
    /// [`Outcome::n_eval`]) for per-cycle evaluation costs.
    #[must_use]
    pub fn last_cycle_boundaries(&self) -> &[usize] {
        &self.cycle_boundaries
    }
}

/// One-shot convenience over [`Bobyqa`]: infers `n` from `x.len()`, builds a
/// throwaway solver, and runs a single minimisation. **Allocates per call** — hot loops
/// re-solving many problems of one size should build a [`Bobyqa`] once and reuse it.
///
/// [`Bobyqa::new`] rejections (bad `npt`/`rho`/`max_fun`/`f_target`, or `x.len() == 0` — the
/// same `n = 0` rejection as `new`) fold into the shape `minimize` already uses for runtime
/// rejection: `Outcome { f: NaN, n_eval: 0, status: InvalidArgs }` — never a nested `Result`.
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
        // `new` only ever fails with InvalidArgs; keep its word rather than re-spelling it.
        Err(status) => Outcome {
            f: f64::NAN,
            n_eval: 0,
            status,
        },
    }
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
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA (rust.md §5)
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
    // The clamped copies live in the solver workspace (M2 §4: zero-alloc warm path).
    for i in 0..n {
        ws.bobyqb.xl[i] = lower[i].max(-BOUNDMAX);
        ws.bobyqb.xu[i] = upper[i].min(BOUNDMAX);
    }
    let rhobeg = config.rho_begin;

    // PRIMA bobyqa.f90 L316: x = max(xl, min(xu, moderatex(x))) — in place, elementwise
    // (each x[i] depends only on x[i], so the moderate-then-clamp order is FP-identical
    // to the former moderatex-copy-then-clamp).
    for i in 0..n {
        let xm = moderatex1(x[i]);
        x[i] = ws.bobyqb.xl[i].max(ws.bobyqb.xu[i].min(xm));
    }

    // PRIMA preproc.f90 L341-350 (HONOUR_X0 = FALSE — the path the oracle runs; SPEC §7.6):
    // revise X0 so its distance to each inactive bound is 0 or >= rhobeg. Valid because
    // validation guarantees XU - XL >= 2*RHOBEG and X is in the box (the L338 precondition).
    // The follow-up rhobeg-revision block (preproc.f90 L367-383) is omitted: after this
    // revision it is "unnecessary in precise arithmetic" (PRIMA's own L368 N.B.), and its
    // rounding-error repairs fall under the no-repair stance — validation rejects, never fixes.
    for i in 0..n {
        if x[i] <= ws.bobyqb.xl[i] + 0.5 * rhobeg {
            x[i] = ws.bobyqb.xl[i];
        } else if x[i] < ws.bobyqb.xl[i] + rhobeg {
            x[i] = ws.bobyqb.xl[i] + rhobeg;
        }
    }
    for i in 0..n {
        if x[i] >= ws.bobyqb.xu[i] - 0.5 * rhobeg {
            x[i] = ws.bobyqb.xu[i];
        } else if x[i] > ws.bobyqb.xu[i] - rhobeg {
            x[i] = ws.bobyqb.xu[i] - rhobeg;
        }
    }
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
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RestartConfig {
    /// Cap on restarts before returning the last cycle's result. The recommended schedule
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
// (a normal loop exit IS convergence). `MAXTR_REACHED` is budget-class (`maxtr = 2 * max_fun` on
// `Bobyqa`, scaled by `max_restarts + 1` when restarts are enabled to absorb a worst-case zero-eval
// iteration burst per restart cycle; near-unreachable on both paths). `NAN_INF_X`/`NAN_INF_F` are
// `checkexit` defensive guards, near-unreachable
// behind `moderatex`/`moderatef` — numerical-breakdown class. `NO_SPACE_BETWEEN_BOUNDS` is
// caught by validation before the loop. `TRSUBP_FAILED` is never emitted by the BOBYQA port
// (`trsbox` returns only CRVMIN, no info code) — the constant exists for completeness against
// PRIMA's infos.f90, so it is absent here.
fn status_from_info(info: i32) -> Status {
    use crate::consts::{
        DAMAGING_ROUNDING, FTARGET_ACHIEVED, MAXFUN_REACHED, MAXTR_REACHED, NAN_INF_F,
        NAN_INF_MODEL, NAN_INF_X, SMALL_TR_RADIUS,
    };
    match info {
        SMALL_TR_RADIUS => Status::Converged,
        FTARGET_ACHIEVED => Status::TargetReached,
        MAXFUN_REACHED | MAXTR_REACHED => Status::MaxFunReached,
        NAN_INF_MODEL | DAMAGING_ROUNDING | NAN_INF_X | NAN_INF_F => Status::ModelDegenerate,
        // bobyqb's info set is closed; a new code here is a port bug or a spec amendment to
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

    #[test]
    fn config_new_returns_prima_defaults() {
        let c = Config::new(3);
        assert_eq!(c.npt, 7); // 2n + 1
        assert_eq!(c.rho_begin, 1.0);
        assert_eq!(c.rho_end, 1e-6);
        assert_eq!(c.max_fun, 1500); // 500 * n
        assert_eq!(c.f_target, f64::NEG_INFINITY);
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
        ); // (only +inf was tested before)
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
                p.iter().map(|v| v * v).sum::<f64>()
            },
            &mut x,
            &[-5.0, -5.0],
            &[5.0, 5.0],
        );
        assert_eq!(o.status, Status::Converged);
        assert_eq!(o.n_eval, n_calls);
        assert!(o.f < 1e-8);
        assert!(x.iter().all(|v| v.abs() < 1e-3));
        assert!(o.n_eval >= c.npt); // the npt model-building floor (SPEC §1)
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
                p.iter().map(|v| v * v).sum::<f64>()
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
        let sphere = |p: &[f64]| p.iter().map(|v| v * v).sum::<f64>();
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
        let good = s.minimize(
            |p: &[f64]| p.iter().map(|v| v * v).sum::<f64>(),
            &mut [1.0, 2.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
        );
        assert!(good.found_finite());
    }

    #[test]
    #[expect(clippy::many_single_char_names)] // n, c, s, x, o are PRIMA/test shorthands
    fn minimize_accepts_infinite_bounds_via_the_boundmax_clamp() {
        // |bound| >= BOUNDMAX means "no bound" (design §4.2); minimize clamps to +/-BOUNDMAX
        // (bobyqa.f90 L287-301) and must run, not panic.
        let (n, c) = valid();
        let mut s = Bobyqa::new(n, c).unwrap();
        let mut x = [1.0, 2.0];
        let o = s.minimize(
            |p: &[f64]| p.iter().map(|v| v * v).sum::<f64>(),
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
        assert_eq!(status_from_info(MAXFUN_REACHED), Status::MaxFunReached);
        assert_eq!(status_from_info(MAXTR_REACHED), Status::MaxFunReached);
        assert_eq!(status_from_info(NAN_INF_MODEL), Status::ModelDegenerate);
        assert_eq!(status_from_info(DAMAGING_ROUNDING), Status::ModelDegenerate);
        assert_eq!(status_from_info(NAN_INF_X), Status::ModelDegenerate);
        assert_eq!(status_from_info(NAN_INF_F), Status::ModelDegenerate);
    }

    #[test]
    fn status_displays_and_is_an_error() {
        let e: &dyn std::error::Error = &Status::InvalidArgs;
        assert!(!e.to_string().is_empty());
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
        assert!(s.last_cycle_boundaries().is_empty());
    }

    #[test]
    fn one_shot_bobyqa_minimizes_and_folds_construction_errors_into_the_outcome() {
        // Happy path: same sphere as the reusable-solver test.
        let mut x = [1.0, 2.0];
        let o = bobyqa(
            |p: &[f64]| p.iter().map(|v| v * v).sum::<f64>(),
            &mut x,
            &[-5.0, -5.0],
            &[5.0, 5.0],
            Config::new(2),
        );
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
