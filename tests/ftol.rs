//! ftol spec §3: the opt-in f-tolerance stop.
//!
//! Test 1 of the spec — the untouched golden battery with `ftol` at its `None` default —
//! lives in `tests/parity_prima.rs` and never sets the fields; nothing here re-proves it.
//! This file covers the passive-placement proof (§3.2), the triggered stop (§3.3), and the
//! restart precedence pinned in the spec's §2.

use bobyqa::{Bobyqa, Config, RestartConfig, Status};

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>()
}

fn booth(x: &[f64]) -> f64 {
    let a = x[0] + 2.0 * x[1] - 7.0;
    let b = 2.0 * x[0] + x[1] - 5.0;
    a * a + b * b
}

fn rosenbrock(x: &[f64]) -> f64 {
    let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
    a * a + 100.0 * b * b
}

/// Runs one solve and returns `(f, n_eval, status, x)`.
fn solve(
    config: Config,
    obj: fn(&[f64]) -> f64,
    x0: [f64; 2],
    lower: [f64; 2],
    upper: [f64; 2],
) -> (f64, usize, Status, [f64; 2]) {
    let mut solver = Bobyqa::new(2, config).expect("valid config");
    let mut x = x0;
    let o = solver.minimize(obj, &mut x, &lower, &upper);
    (o.f, o.n_eval, o.status, x)
}

/// Spec §3.2: `Some(0.0)/Some(0.0)` must reproduce the default trajectory bit-for-bit
/// whenever no stage has exactly zero improvement — the proof that the check's PLACEMENT
/// is passive (it reads best-f at the reduction site and changes nothing else).
///
/// Only a non-quadratic golden qualifies: on quadratics (sphere, booth) BOBYQA's model is
/// exact after the first stage, so every later stage improves by literally `0.0` and the
/// `df <= 0.0` test legitimately fires — that half of the semantics ("stops only on exact
/// stagnation") is pinned by the second test below.
#[test]
fn zero_ftol_reproduces_the_default_trajectory_bitwise() {
    let (obj, x0, lo, up): (fn(&[f64]) -> f64, _, _, _) =
        (rosenbrock, [-1.2, 1.0], [-5.0, -5.0], [5.0, 5.0]);
    let base = Config::new(2);
    let mut zero = Config::new(2);
    zero.ftol_rel = Some(0.0);
    zero.ftol_abs = Some(0.0);
    let (f_d, n_d, s_d, x_d) = solve(base, obj, x0, lo, up);
    let (f_z, n_z, s_z, x_z) = solve(zero, obj, x0, lo, up);
    assert_eq!(s_d, Status::Converged);
    assert_eq!(s_z, s_d);
    assert_eq!(n_z, n_d, "zero-ftol changed the evaluation count");
    assert_eq!(f_z.to_bits(), f_d.to_bits(), "zero-ftol moved f");
    for (a, b) in x_z.iter().zip(&x_d) {
        assert_eq!(a.to_bits(), b.to_bits(), "zero-ftol moved x");
    }
}

/// The other half of the `Some(0.0)` semantics: exact stagnation stops the solve. On a
/// quadratic objective the interpolation model is exact after the first rho stage, so the
/// very next stage improves by exactly zero and `ftol = 0` fires.
#[test]
fn zero_ftol_stops_on_exact_stagnation_of_a_quadratic() {
    for (obj, x0, lo, up) in [
        (
            sphere as fn(&[f64]) -> f64,
            [1.0, 2.0],
            [-5.0, -5.0],
            [5.0, 5.0],
        ),
        (booth, [0.0, 0.0], [-10.0, -10.0], [10.0, 10.0]),
    ] {
        let base = Config::new(2);
        let mut zero = Config::new(2);
        zero.ftol_rel = Some(0.0);
        zero.ftol_abs = Some(0.0);
        let (f_d, n_d, s_d, _) = solve(base, obj, x0, lo, up);
        let (f_z, n_z, s_z, _) = solve(zero, obj, x0, lo, up);
        assert_eq!(s_d, Status::Converged);
        assert_eq!(s_z, Status::FtolReached, "exact stagnation must stop");
        assert!(n_z < n_d);
        // A stage CAN stagnate exactly and later stages still buy meaningless digits
        // (booth: 5.5e-11 at the stop vs 1.4e-17 at rho_end) — ftol = 0 accepts that
        // trade by definition. The claim is only: the stop point is already essentially
        // at the optimum (both quadratics have minimum 0 from f0 ~ 5-74).
        assert!(
            f_d >= 0.0 && f_z >= f_d,
            "best-f is monotone; earlier can't be better"
        );
        assert!(
            f_z < 1e-8,
            "stop point must be essentially optimal: f = {f_z:e}"
        );
    }
}

/// Spec §3.3: a small `ftol_rel` on a smooth problem stops early as `FtolReached`, spends
/// fewer evaluations, and lands within `C * ftol_rel * max(|f_default|, 1)` of the default
/// answer, C = 100 (documented safety factor over the per-stage tolerance).
#[test]
fn small_ftol_rel_stops_early_close_to_the_default_answer() {
    let ftol_rel = 1e-8;
    for (obj, x0, lo, up) in [
        (
            sphere as fn(&[f64]) -> f64,
            [1.0, 2.0],
            [-5.0, -5.0],
            [5.0, 5.0],
        ),
        (booth, [0.0, 0.0], [-10.0, -10.0], [10.0, 10.0]),
    ] {
        let base = Config::new(2);
        let mut tol = Config::new(2);
        tol.ftol_rel = Some(ftol_rel);
        let (f_d, n_d, s_d, _) = solve(base, obj, x0, lo, up);
        let (f_t, n_t, s_t, _) = solve(tol, obj, x0, lo, up);
        assert_eq!(s_d, Status::Converged);
        assert_eq!(s_t, Status::FtolReached, "ftol did not trigger");
        assert!(n_t < n_d, "ftol stop must save evaluations: {n_t} vs {n_d}");
        let c = 100.0;
        assert!(
            (f_t - f_d).abs() <= c * ftol_rel * f_d.abs().max(1.0),
            "f drifted: {f_t} vs {f_d}"
        );
    }
}

/// Precedence: `FtolReached` is converged-class — when ftol and a restart trigger would fire
/// at the same `rho` reduction, the solve stops instead of spending a restart.
///
/// `stall_reductions = 1` with the default `improve_rel_tol = 1e-6` asks for a restart at the
/// first reduction that improved f by less than `1e-6 * max(|f|, 1)`; `ftol_rel = 1e-6` stops
/// at that same reduction. The first two runs show each trigger alone fires there.
#[test]
fn ftol_stop_does_not_spend_a_restart() {
    let mut rc = RestartConfig::new();
    rc.cycle_budget_frac = 0.0;
    rc.stall_reductions = 1;
    let run = |restart: Option<RestartConfig>, ftol_rel: Option<f64>| {
        let mut config = Config::new(2);
        config.restart = restart;
        config.ftol_rel = ftol_rel;
        let mut solver = Bobyqa::new(2, config).expect("valid config");
        let mut x = [1.0, 2.0];
        let o = solver.minimize(sphere, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
        (o.status, o.n_eval, solver.last_cycle_boundaries().to_vec())
    };

    let (_, _, restart_cuts) = run(Some(rc), None);
    let (ftol_status, ftol_n_eval, _) = run(None, Some(1e-6));
    assert_eq!(ftol_status, Status::FtolReached);
    assert_eq!(
        restart_cuts.first(),
        Some(&ftol_n_eval),
        "the stall trigger and ftol must fire at the same reduction for this test to mean anything"
    );

    let (status, n_eval, cuts) = run(Some(rc), Some(1e-6));
    assert_eq!(status, Status::FtolReached);
    assert_eq!(n_eval, ftol_n_eval);
    assert_eq!(cuts, [], "ftol must win over the restart schedule");
}
