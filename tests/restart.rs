//! Restart behavioural tests: the restart hook actually fires and counts restarts, all three
//! triggers (the eval cap, the stalled `rho` tail, and `rho_end`) do what they claim, both
//! backstops (`max_restarts`, the `improve_rel_tol` settle test) gate them, and the schedule
//! never returns a worse point or overruns the total budget.
//!
//! `Config` and `RestartConfig` are `#[non_exhaustive]`, so everything here is built the way
//! a downstream crate must build it: constructor + field assignment, never a struct literal.

use bobyqa::{Bobyqa, Config, Outcome, RestartConfig, Status, bobyqa};
use std::cell::RefCell;

// Rosenbrock: the canonical DFO valley; a single solve reaches rho_end well short of a
// tight optimum, so at least one restart must fire.
fn rosenbrock(p: &[f64]) -> f64 {
    let (a, b) = (1.0 - p[0], p[1] - p[0] * p[0]);
    a * a + 100.0 * b * b
}

fn sphere(p: &[f64]) -> f64 {
    p.iter().map(|v| v * v).sum()
}

// Descending wells: local minima on the integer lattice, each one deeper as the coordinate
// decreases. A rebuild samples npt fresh points a full `rho_begin` out from the incumbent, so
// it can land in a strictly better well; plain BOBYQA never leaves the one it converged in.
fn wells(p: &[f64]) -> f64 {
    p.iter()
        .map(|v| {
            let s = (std::f64::consts::PI * v).sin();
            4.0 * s * s + 0.3 * v
        })
        .sum()
}

/// `config` with the restart schedule plugged in — the downstream construction idiom.
fn with_restart(config: Config, rc: RestartConfig) -> Config {
    let mut c = config;
    c.restart = Some(rc);
    c
}

/// The `rho_end`-only schedule: no cap, no stall trigger, so the settle test at `rho_end` is
/// the only thing that can fire a restart. The baseline most tests below vary one knob from.
fn rho_end_only(max_restarts: usize) -> RestartConfig {
    let mut rc = RestartConfig::new();
    rc.cycle_budget_frac = 0.0;
    rc.stall_reductions = 0;
    rc.max_restarts = max_restarts;
    rc
}

/// The stalled-tail schedule: the `stall_reductions` trigger on, the cap off.
fn stall_schedule(max_restarts: usize) -> RestartConfig {
    let mut rc = rho_end_only(max_restarts);
    rc.stall_reductions = 2;
    rc
}

#[test]
fn restart_fires_at_least_once_on_a_stalling_problem() {
    let mut base = Bobyqa::new(2, Config::new(2)).unwrap();
    let mut xb = [-1.2, 1.0];
    let ob = base.minimize(rosenbrock, &mut xb, &[-5.0, -5.0], &[5.0, 5.0]);

    let mut s = Bobyqa::new(2, with_restart(Config::new(2), stall_schedule(8))).unwrap();
    let mut x = [-1.2, 1.0];
    let o = s.minimize(rosenbrock, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
    assert_eq!(o.status, Status::Converged);
    // The returned point must be at least as good as restart-off on the same problem.
    assert!(o.f <= ob.f + 1e-12, "restart worse: {} vs {}", o.f, ob.f);
    assert!(
        s.last_restart_count() >= 1,
        "expected a restart, got {}",
        s.last_restart_count()
    );
    assert_eq!(
        s.last_cycle_boundaries().len(),
        s.last_restart_count(),
        "one cycle boundary per restart"
    );
}

#[test]
fn restart_stops_at_the_max_restarts_backstop() {
    // Rosenbrock keeps improving cycle over cycle (improve_rel_tol: 0.0 never trips the settle
    // test), so max_restarts is the only thing that can stop it — confirm it does, exactly.
    let mut rc = rho_end_only(1);
    rc.improve_rel_tol = 0.0;
    let mut s = Bobyqa::new(2, with_restart(Config::new(2), rc)).unwrap();
    let mut x = [-1.2, 1.0];
    let o = s.minimize(rosenbrock, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
    assert_eq!(o.status, Status::Converged);
    assert_eq!(
        s.last_restart_count(),
        1,
        "max_restarts=1 must cap the restart count exactly"
    );
}

#[test]
fn restart_stops_early_on_the_improve_rel_tol_plateau() {
    // Sphere is already essentially exact by the time rho reaches rho_end; the next cycle
    // cannot improve fopt by the default 1e-6 relative tolerance, so the settle test — not
    // max_restarts=8 — must be what stops it, well short of the budget.
    let mut s = Bobyqa::new(2, with_restart(Config::new(2), rho_end_only(8))).unwrap();
    let mut x = [1.0, 1.0];
    let o = s.minimize(sphere, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
    assert_eq!(o.status, Status::Converged);
    assert_eq!(
        s.last_restart_count(),
        1,
        "the improve_rel_tol settle test must stop restarts after the first one",
    );
}

#[test]
fn max_fun_cutoff_holds_at_every_budget_across_the_restart_window() {
    // A restart cycle spends `max_fun` from a state the opening solve does not reach, so sweep
    // every budget across the window where the first restarts fire (the plain solve needs 127
    // evals on this problem) and pin the contract at each one: never more than `max_fun`
    // evaluations, exactly `max_fun` when the budget is what stopped it, and a finite best
    // point either way. At least one budget in the window must restart, and at least one must
    // exhaust — otherwise this test is not testing what it claims.
    //
    // The rebuild itself can never be the site of the cutoff: the rebuild-room rule keeps the
    // trigger inert unless more than `npt` evaluations remain, and a rebuild costs exactly
    // `npt`, so `initxf` always completes. That is why some budgets here converge instead —
    // the last restart is refused and the cycle runs on to `rho_end`.
    let mut hit_a_restart = false;
    let mut hit_the_budget = false;
    for max_fun in 120..=170 {
        let mut cfg = Config::new(2);
        cfg.max_fun = max_fun;
        let mut rc = stall_schedule(8);
        rc.improve_rel_tol = 0.0;
        let mut s = Bobyqa::new(2, with_restart(cfg, rc)).unwrap();
        let mut x = [-1.2, 1.0];
        let o = s.minimize(rosenbrock, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
        assert!(
            matches!(o.status, Status::Converged | Status::MaxFunReached),
            "max_fun = {max_fun}: {:?}",
            o.status
        );
        assert!(o.n_eval <= max_fun, "budget overrun at max_fun = {max_fun}");
        if o.status == Status::MaxFunReached {
            assert_eq!(
                o.n_eval, max_fun,
                "MaxFunReached short of the budget at max_fun = {max_fun}"
            );
            hit_the_budget = true;
        }
        assert!(o.found_finite(), "max_fun = {max_fun}");
        assert!(o.f < 1e-8, "max_fun = {max_fun}: f = {:e}", o.f);
        hit_a_restart |= s.last_restart_count() >= 1;
    }
    assert!(
        hit_a_restart,
        "no budget in the sweep reached a restart — window is wrong"
    );
    assert!(
        hit_the_budget,
        "no budget in the sweep was actually exhausted — window is wrong"
    );
}

#[test]
fn a_reused_solver_repeats_a_restart_run_exactly() {
    // A second `minimize` on the same solver must give an identical (x, f, n_eval, status,
    // restart count, cycle boundaries): nothing of the first run's restart state may leak.
    let mut s = Bobyqa::new(2, with_restart(Config::new(2), stall_schedule(8))).unwrap();
    let mut run = || {
        let mut x = [-1.2, 1.0];
        let o = s.minimize(rosenbrock, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
        (
            o.f.to_bits(),
            o.n_eval,
            o.status,
            s.last_restart_count(),
            s.last_cycle_boundaries().to_vec(),
            x,
        )
    };
    let a = run();
    assert_eq!(a, run(), "a reused solver must repeat the restart run");
    assert!(
        a.3 >= 1,
        "no restart fired, so reuse of restart state went untested"
    );
}

// Degenerate case 1: the opening solve exits before `rho_end`, so the restart hook is
// never reached — `last_restart_count() == 0` and the run is restart-off's, exactly. `f_target`
// at 1e9 is satisfied by the very first evaluation, which is the cheapest way to make the hook
// unreachable (every termination other than the `rho_end` path bypasses it).
#[test]
fn opening_solve_converging_before_rho_end_gives_zero_restarts_and_matches_restart_off() {
    let mut cfg = Config::new(2);
    cfg.f_target = 1e9;
    let mut s = Bobyqa::new(2, with_restart(cfg, RestartConfig::new())).unwrap();
    let mut x = [1.0, 2.0];
    let o = s.minimize(sphere, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
    assert_eq!(o.status, Status::TargetReached);
    assert_eq!(s.last_restart_count(), 0);
    assert_eq!(s.last_cycle_boundaries(), []);

    // "Matches restart-off" in the strongest available sense: with the hook unreachable the two
    // configurations are the same engine on the same state, so every returned byte must agree.
    let mut base = Bobyqa::new(2, cfg).unwrap();
    let mut xb = [1.0, 2.0];
    let ob = base.minimize(sphere, &mut xb, &[-5.0, -5.0], &[5.0, 5.0]);
    assert_eq!(ob.status, o.status);
    assert_eq!(ob.n_eval, o.n_eval);
    assert_eq!(ob.f.to_bits(), o.f.to_bits());
    assert_eq!(xb, x);
}

// The one-shot `bobyqa` free function carries the restart schedule inside `Config`, so it must
// be exactly "build one, run once": byte-for-byte agreement with the reusable surface on the
// same inputs, and construction rejections folded into the `InvalidArgs` `Outcome`.
#[test]
fn one_shot_bobyqa_with_restarts_matches_the_reusable_solver() {
    let cfg = with_restart(Config::new(2), stall_schedule(8));
    let mut x = [-1.2, 1.0];
    let o = bobyqa(rosenbrock, &mut x, &[-5.0, -5.0], &[5.0, 5.0], cfg);
    assert!(matches!(
        o.status,
        Status::Converged | Status::MaxFunReached
    ));
    assert!(o.found_finite());

    let mut s = Bobyqa::new(2, cfg).unwrap();
    let mut xs = [-1.2, 1.0];
    let os = s.minimize(rosenbrock, &mut xs, &[-5.0, -5.0], &[5.0, 5.0]);
    assert_eq!(os.status, o.status);
    assert_eq!(os.n_eval, o.n_eval);
    assert_eq!(os.f.to_bits(), o.f.to_bits());
    assert_eq!(xs, x);

    // Construction rejections arrive as the InvalidArgs Outcome, not a Result — the `bobyqa`
    // free function's contract, kept with restarts in the config.
    let mut bad_rc = RestartConfig::new();
    bad_rc.max_restarts = 0;
    let bad = bobyqa(
        rosenbrock,
        &mut [0.0, 0.0],
        &[-9.0, -9.0],
        &[9.0, 9.0],
        with_restart(Config::new(2), bad_rc),
    );
    assert_eq!(bad.status, Status::InvalidArgs);
    assert!(bad.f.is_nan());
    assert_eq!(bad.n_eval, 0);
}

// A restart must never turn a converged answer into a failed status. At `npt = (n+1)(n+2)/2`
// the quadratic is fully determined and the converged interpolation set is tight to machine
// precision; in PRIMA parity mode this exact configuration already drives plain restart-off
// runs through two RESCUE calls before any restart exists, so it is where a spurious
// `DAMAGING_ROUNDING` would surface.
// Pin both halves — the status is the convergence that was earned, and `f` is bit-identical to
// restart-off, because the cycle that triggered the restart had already reached `rho_end`
// holding the incumbent that gets returned.
#[test]
fn a_restart_never_downgrades_a_converged_answer_to_model_degenerate() {
    const N: usize = 8;
    let mut cfg = Config::new(N);
    cfg.npt = (N + 1) * (N + 2) / 2; // 45 — fully determined, NOT the 2n+1 default
    cfg.prima_parity = true; // with `false` this solve makes no RESCUE call
    let x0: Vec<f64> = (0..N)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect();
    let (lower, upper) = (vec![-5.0; N], vec![5.0; N]);

    let mut base = Bobyqa::new(N, cfg).unwrap();
    let mut xb = x0.clone();
    let ob = base.minimize(sphere, &mut xb, &lower, &upper);
    assert_eq!(ob.status, Status::Converged);

    // The rho_end-only schedule: the bit-identical claim needs the opening cycle to run its
    // full tail, and the stall trigger would cut it.
    let mut s = Bobyqa::new(N, with_restart(cfg, rho_end_only(8))).unwrap();
    let mut x = x0.clone();
    let o = s.minimize(sphere, &mut x, &lower, &upper);
    assert_eq!(o.status, Status::Converged);
    assert!(o.found_finite());
    assert!(s.last_restart_count() >= 1);
    // The answer was never in doubt — only the status was. Bit-identical, not "close".
    assert_eq!(
        o.f.to_bits(),
        ob.f.to_bits(),
        "f = {:e} vs baseline {:e}",
        o.f,
        ob.f
    );
}

#[test]
fn returned_point_matches_returned_value_across_restarts() {
    // The returned x must actually produce the returned f (best-across-restarts consistency).
    let mut s = Bobyqa::new(2, with_restart(Config::new(2), stall_schedule(8))).unwrap();
    let mut x = [-1.2, 1.0];
    let o = s.minimize(rosenbrock, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
    let f_at_x = rosenbrock(&x);
    assert!(
        (f_at_x - o.f).abs() <= 1e-9 * o.f.abs().max(1.0),
        "returned x gives {f_at_x}, outcome.f = {}",
        o.f
    );
}

// Restart-off bit-identity — that a `restart: None` solver in PRIMA parity mode reproduces PRIMA
// exactly — is pinned by `tests/parity_prima.rs`, which replays the frozen PRIMA `(x, f)`
// trajectories against a restart-off `Bobyqa`. It is not duplicated here.

// ---- The stall trigger ----

/// One sphere solve at the given `stall_reductions`, returning the outcome, restart count, and
/// cycle boundaries. The cap is off, so the stall counter and `rho_end` are the only triggers.
fn sphere_run(stall_reductions: usize) -> (Outcome, usize, Vec<usize>) {
    let mut rc = rho_end_only(8);
    rc.stall_reductions = stall_reductions;
    let mut s = Bobyqa::new(2, with_restart(Config::new(2), rc)).unwrap();
    let mut x = [1.0, 1.0];
    let o = s.minimize(sphere, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
    (
        o,
        s.last_restart_count(),
        s.last_cycle_boundaries().to_vec(),
    )
}

#[test]
fn stall_trigger_cuts_a_stalled_tail_before_rho_end() {
    // Sphere's opening cycle stalls in its rho tail: near the optimum, |f| < 1, so the stall
    // threshold is 1e-6 absolute, and the last few rho reductions each improve f by far less —
    // exactly the tail the trigger exists to cut. `stall_reductions: 0` runs that same tail to
    // rho_end before its first restart, so the first cycle boundary is the direct observation:
    // the stall run's must come strictly earlier.
    let (o0, r0, b0) = sphere_run(0);
    let (o2, r2, b2) = sphere_run(2);
    assert_eq!(o0.status, Status::Converged);
    assert_eq!(o2.status, Status::Converged);
    assert!(r0 >= 1, "the rho_end trigger never fired");
    assert!(r2 >= 1, "the stall trigger never fired");
    assert!(
        b2[0] < b0[0],
        "stall run's first cycle ({} evals) was not cut short of the rho_end-only run's ({})",
        b2[0],
        b0[0]
    );

    // The rho_end-only schedule really is rho_end-only: its opening cycle must match a plain
    // restart-off solve's whole evaluation count (± the post-loop Newton-Raphson tail step,
    // which plain Bobyqa may spend at final termination but a restarting cycle defers).
    let mut base = Bobyqa::new(2, Config::new(2)).unwrap();
    let mut xb = [1.0, 1.0];
    let ob = base.minimize(sphere, &mut xb, &[-5.0, -5.0], &[5.0, 5.0]);
    assert!(
        ob.n_eval.abs_diff(b0[0]) <= 1,
        "stall_reductions: 0 cut the opening cycle: boundary {} vs plain solve's {} evals",
        b0[0],
        ob.n_eval
    );

    // Whichever trigger fired, the FINAL cycle ran to rho_end, so the stall run's answer is as
    // fine as the plain solve's — the 1e-5 relative gate is the accuracy claim.
    assert!(
        (o2.f - ob.f).abs() <= 1e-5 * ob.f.abs().max(1.0),
        "stall run drifted: f = {:e} vs plain {:e}",
        o2.f,
        ob.f
    );
}

#[test]
fn productive_reductions_never_trip_the_stall_trigger() {
    // With improve_rel_tol at 0, every monotone rho reduction counts as productive, so the
    // stall counter resets at each one and the early trigger must never fire — any
    // `stall_reductions` value must reproduce the rho_end-only schedule byte-for-byte
    // ("a solve that is still making progress never restarts early").
    let run = |stall_reductions: usize| {
        let mut rc = rho_end_only(2); // tol 0 never settles, so cap the schedule instead
        rc.improve_rel_tol = 0.0;
        rc.stall_reductions = stall_reductions;
        let log = RefCell::new(Vec::new());
        let mut s = Bobyqa::new(2, with_restart(Config::new(2), rc)).unwrap();
        let mut x = [-1.2, 1.0];
        let o = s.minimize(
            |p: &[f64]| {
                log.borrow_mut().push([p[0], p[1]]);
                rosenbrock(p)
            },
            &mut x,
            &[-5.0, -5.0],
            &[5.0, 5.0],
        );
        (
            log.into_inner(),
            o.f.to_bits(),
            o.n_eval,
            s.last_cycle_boundaries().to_vec(),
        )
    };
    let baseline = run(0);
    assert_eq!(
        baseline.3.len(),
        2,
        "the capped schedule should restart twice"
    );
    for stall_reductions in [1, 2, 5] {
        assert_eq!(
            run(stall_reductions),
            baseline,
            "stall_reductions = {stall_reductions} diverged from the rho_end-only schedule \
             despite every reduction being productive"
        );
    }
}

// ---- The eval cap, the rebuild, and the monotone incumbent ----

/// The cap-firing configuration: Rosenbrock crawls its rho schedule so slowly toward a
/// 1e-10 `rho_end` that a 120-eval budget runs out with `rho` far from done. Neither the
/// stall counter nor `rho_end` has a site to fire from here, so the cap is the only trigger
/// that can act — which is exactly the case it exists for.
fn crawling_config(cycle_budget_frac: f64) -> Config {
    let mut cfg = Config::new(2);
    cfg.max_fun = 120;
    cfg.rho_end = 1e-10;
    let mut rc = rho_end_only(8);
    rc.cycle_budget_frac = cycle_budget_frac;
    rc.improve_rel_tol = 0.0;
    cfg.restart = Some(rc);
    cfg
}

#[test]
fn the_cap_fires_where_no_other_trigger_can_and_zero_disables_it() {
    let run = |cycle_budget_frac: f64| {
        let mut s = Bobyqa::new(2, crawling_config(cycle_budget_frac)).unwrap();
        let mut x = [-1.2, 1.0];
        let o = s.minimize(rosenbrock, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
        (o, s.last_restart_count())
    };
    let (on, restarts_on) = run(0.25);
    let (off, restarts_off) = run(0.0);
    assert!(
        restarts_on >= 1,
        "the cap never fired on a cycle that outspent its allowance"
    );
    assert_eq!(
        restarts_off, 0,
        "cycle_budget_frac = 0.0 must disable the cap, got {restarts_off} restarts"
    );
    assert!(on.found_finite() && off.found_finite());
}

#[test]
fn the_recommended_schedule_runs_clean_and_restarts_once() {
    // `RestartConfig::new()` as shipped, on a problem whose cycle does outspend its allowance.
    let mut cfg = Config::new(2);
    cfg.max_fun = 120;
    cfg.rho_end = 1e-10;
    let mut s = Bobyqa::new(2, with_restart(cfg, RestartConfig::new())).unwrap();
    let mut x = [-1.2, 1.0];
    let o = s.minimize(rosenbrock, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
    assert!(matches!(
        o.status,
        Status::Converged | Status::MaxFunReached
    ));
    assert!(o.found_finite());
    assert_eq!(
        s.last_restart_count(),
        1,
        "the shipped cap must fire once on a cycle that outspends its allowance"
    );
    assert_eq!(s.last_cycle_boundaries().len(), 1);
}

#[test]
fn the_recommended_schedule_is_bit_identical_on_a_solve_that_settles_inside_its_cap() {
    // The coupling between `cycle_budget_frac` and the `rho_end` trigger: with the cap set,
    // reaching `rho_end` restarts only when the cap agrees the cycle was expensive. Sphere
    // converges in a few dozen evaluations of a 1000-evaluation budget, so a 0.125 cap
    // provably cannot fire — and the recommended schedule must therefore be a no-op.
    //
    // `n_eval` is the assertion that matters. What this pins is a wasted extra cycle that
    // lands on the same optimum, so an x/f-only comparison would pass while the bug is present.
    let mut base = Bobyqa::new(2, Config::new(2)).unwrap();
    let mut xb = [1.0, 1.0];
    let ob = base.minimize(sphere, &mut xb, &[-5.0, -5.0], &[5.0, 5.0]);

    let mut s = Bobyqa::new(2, with_restart(Config::new(2), RestartConfig::new())).unwrap();
    let mut x = [1.0, 1.0];
    let o = s.minimize(sphere, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);

    assert_eq!(
        s.last_restart_count(),
        0,
        "a solve that settles well inside its cap must not restart"
    );
    assert_eq!(
        o.n_eval, ob.n_eval,
        "the recommended schedule spent {} evaluations against restart-off's {}",
        o.n_eval, ob.n_eval
    );
    assert_eq!(o.f.to_bits(), ob.f.to_bits());
    assert_eq!(o.status, ob.status);
    assert_eq!(x, xb);
}

#[test]
fn a_restart_returns_no_worse_than_its_cut_point() {
    // The returned point is a monotone incumbent across cycles. Strongest observable
    // form: the returned f equals the minimum over EVERY evaluation of the whole solve —
    // in particular it is no worse than the best at any restart's cut point, even though
    // each rebuild discards the model and every rebuilt point may be worse.
    let log = RefCell::new(Vec::new());
    let mut s = Bobyqa::new(2, crawling_config(0.25)).unwrap();
    let mut x = [-1.2, 1.0];
    let o = s.minimize(
        |p: &[f64]| {
            let f = rosenbrock(p);
            log.borrow_mut().push(f);
            f
        },
        &mut x,
        &[-5.0, -5.0],
        &[5.0, 5.0],
    );
    let log = log.into_inner();
    assert!(s.last_restart_count() >= 1, "the cap trigger never fired");
    let global_min = log.iter().copied().fold(f64::INFINITY, f64::min);
    assert_eq!(
        o.f, global_min,
        "returned f must be the best evaluation of the whole solve"
    );
}

#[test]
fn a_rebuild_leaves_a_basin_a_plain_solve_cannot() {
    // What the rebuild buys, in-crate: a restart discards the model and samples npt fresh
    // points a full rho_begin out from the incumbent, so on `wells` it walks well to well —
    // every neighbouring well bottom is 0.3 deeper — while plain BOBYQA converges in the
    // basin it started in and stays there.
    let mut plain = Bobyqa::new(2, Config::new(2)).unwrap();
    let mut xp = [0.1, 0.1];
    let op = plain.minimize(wells, &mut xp, &[-5.0, -5.0], &[5.0, 5.0]);

    let mut rc = rho_end_only(8);
    rc.improve_rel_tol = 0.0; // never settle: run the whole max_restarts = 8 schedule
    let mut cfg = Config::new(2);
    cfg.max_fun = 1000;
    let mut s = Bobyqa::new(2, with_restart(cfg, rc)).unwrap();
    let mut xs = [0.1, 0.1];
    let os = s.minimize(wells, &mut xs, &[-5.0, -5.0], &[5.0, 5.0]);

    assert!(
        os.f < op.f - 1.0,
        "the rebuild did not leave the basin: restart f = {}, plain f = {}",
        os.f,
        op.f
    );
    // The measured landing today is the deepest corner well (f = -3.0 at [-5, -5]); allow any
    // strictly-better-by-several-wells outcome rather than pinning the trajectory.
    assert!(os.f <= -2.0, "restart f = {}", os.f);
}

#[test]
fn cap_and_stall_off_reproduce_the_rho_end_only_schedule() {
    // With cycle_budget_frac 0.0 and stall_reductions 0 the settle test at rho_end is the
    // only trigger left, so the opening cycle must be a plain restart-off solve, exactly:
    // same evaluation sequence, and the first cut lands where the plain solve ends. (`wells`
    // terminates without the post-loop Newton-Raphson tail step, so the counts match
    // exactly; a tail-stepping problem would be off by that one deferred evaluation.)
    let plain_log = RefCell::new(Vec::new());
    let mut plain = Bobyqa::new(2, Config::new(2)).unwrap();
    let mut xp = [0.1, 0.1];
    let op = plain.minimize(
        |p: &[f64]| {
            plain_log.borrow_mut().push([p[0], p[1]]);
            wells(p)
        },
        &mut xp,
        &[-5.0, -5.0],
        &[5.0, 5.0],
    );

    let restart_log = RefCell::new(Vec::new());
    let mut rc = rho_end_only(8);
    rc.improve_rel_tol = 0.0;
    let mut s = Bobyqa::new(2, with_restart(Config::new(2), rc)).unwrap();
    let mut xs = [0.1, 0.1];
    s.minimize(
        |p: &[f64]| {
            restart_log.borrow_mut().push([p[0], p[1]]);
            wells(p)
        },
        &mut xs,
        &[-5.0, -5.0],
        &[5.0, 5.0],
    );
    assert!(s.last_restart_count() >= 1);
    let b0 = s.last_cycle_boundaries()[0];
    assert_eq!(
        b0, op.n_eval,
        "the rho_end-only opening cycle must end where the plain solve ends"
    );
    assert_eq!(
        restart_log.into_inner()[..b0],
        plain_log.into_inner()[..],
        "opening-cycle evaluations diverged from the plain solve"
    );
}

#[test]
fn a_solve_within_its_cycle_budget_never_trips_the_cap() {
    // Sphere converges each cycle at a small fraction of its allowance, so a generous cap
    // must change nothing at all — same outcome, same restart schedule as cap-off. This is
    // the property that makes the cap safe to ship: its cost on solves that do not need it
    // is zero. (The stall trigger drives both runs here, so the cap is the only variable.)
    let run = |cycle_budget_frac: f64| {
        let mut rc = stall_schedule(8);
        rc.cycle_budget_frac = cycle_budget_frac;
        let mut s = Bobyqa::new(2, with_restart(Config::new(2), rc)).unwrap();
        let mut x = [1.0, 1.0];
        let o = s.minimize(sphere, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
        (
            o.f.to_bits(),
            o.n_eval,
            s.last_restart_count(),
            s.last_cycle_boundaries().to_vec(),
        )
    };
    assert_eq!(
        run(0.5),
        run(0.0),
        "a generous cap changed a solve that stayed within its cycle budgets"
    );
}
