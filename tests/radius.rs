//! `Bobyqa::minimize_with_radius`: the objective sees the trust-region radii, and the solve is
//! `Bobyqa::minimize`'s, bit for bit.

use bobyqa::{Bobyqa, Config, Outcome, RestartConfig, Status, TrustRadius};

fn rosenbrock(p: &[f64]) -> f64 {
    let (a, b) = (1.0 - p[0], p[1] - p[0] * p[0]);
    a * a + 100.0 * b * b
}

// The descending wells of `tests/restart.rs`: a restart-enabled solver restarts on them.
fn wells(p: &[f64]) -> f64 {
    p.iter()
        .map(|v| {
            let s = (std::f64::consts::PI * v).sin();
            4.0 * s * s + 0.3 * v
        })
        .sum()
}

type Objective = fn(&[f64]) -> f64;

/// Every evaluated point with the radii it was evaluated at.
type Trace = Vec<(Vec<f64>, TrustRadius)>;

/// `(objective, config, upper bound)`: an interior solve, a bound-clipped one, and a
/// restarting one (the `rho_end`-only schedule, two restarts).
fn cases() -> Vec<(Objective, Config, f64)> {
    let mut rc = RestartConfig::new();
    rc.cycle_budget_frac = 0.0;
    rc.stall_reductions = 0;
    rc.max_restarts = 2;
    let mut restarting = Config::new(2);
    restarting.restart = Some(rc);
    vec![
        (rosenbrock, Config::new(2), 5.0),
        (rosenbrock, Config::new(2), 0.9),
        (wells, restarting, 5.0),
    ]
}

/// Runs one case through `minimize_with_radius`, recording every point and the radii it was
/// evaluated with.
fn run_with_radius(f: Objective, config: Config, upper: f64) -> (Bobyqa, Trace, [f64; 2], Outcome) {
    let mut s = Bobyqa::new(2, config).unwrap();
    let mut seen = Vec::new();
    let mut x = [-1.2, 1.0];
    let o = s.minimize_with_radius(
        |p: &[f64], r| {
            seen.push((p.to_vec(), r));
            f(p)
        },
        &mut x,
        &[-5.0, -5.0],
        &[upper, upper],
    );
    (s, seen, x, o)
}

#[test]
fn minimize_with_radius_follows_minimize_bit_for_bit() {
    for (f, config, upper) in cases() {
        let (s, seen, x, o) = run_with_radius(f, config, upper);

        let mut base = Bobyqa::new(2, config).unwrap();
        let mut base_points = Vec::new();
        let mut xb = [-1.2, 1.0];
        let ob = base.minimize(
            |p: &[f64]| {
                base_points.push(p.to_vec());
                f(p)
            },
            &mut xb,
            &[-5.0, -5.0],
            &[upper, upper],
        );

        let points: Vec<Vec<f64>> = seen.into_iter().map(|(p, _)| p).collect();
        assert_eq!(points, base_points);
        assert_eq!(x, xb);
        assert_eq!(o.f.to_bits(), ob.f.to_bits());
        assert_eq!((o.n_eval, o.status), (ob.n_eval, ob.status));
        assert_eq!(s.last_cycle_boundaries(), base.last_cycle_boundaries());
    }
}

#[test]
fn minimize_with_radius_reports_the_rho_schedule_of_each_cycle() {
    for (f, config, upper) in cases() {
        let (s, seen, _, o) = run_with_radius(f, config, upper);
        assert_eq!(o.status, Status::Converged);
        if config.restart.is_some() {
            assert!(s.last_restart_count() >= 1, "the restart case must restart");
        }

        // Each cycle (the opening solve, then one per restart) starts at rho_begin with
        // delta = rho, never raises rho, keeps delta >= rho, and stays at or above rho_end.
        let mut starts = vec![0];
        starts.extend_from_slice(s.last_cycle_boundaries());
        starts.push(seen.len());
        for cycle in starts.windows(2) {
            let radii: Vec<TrustRadius> = seen[cycle[0]..cycle[1]].iter().map(|e| e.1).collect();
            assert_eq!(radii[0].rho, config.rho_begin);
            assert_eq!(radii[0].delta, config.rho_begin);
            assert!(radii.windows(2).all(|w| w[1].rho <= w[0].rho));
            assert!(
                radii
                    .iter()
                    .all(|r| r.delta >= r.rho && r.rho >= config.rho_end)
            );
        }
        // A converged solve ends on the last rung of the ladder.
        assert_eq!(seen.last().unwrap().1.rho, config.rho_end);
    }
}
