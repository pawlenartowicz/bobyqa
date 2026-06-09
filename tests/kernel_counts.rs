//! Layer-0 F0a (spec §9): report `calvlag_noadd` invocations per `minimize` run, so the kernel
//! recompute multiplicity (and the realized F0b/F0c saving) is measured deterministically — no
//! wall-clock. Feature-gated; the counter and this test exist only under `count-kernels`.
//!
//! Run: cargo test --features count-kernels --test `kernel_counts` -- --nocapture
#![cfg(feature = "count-kernels")]

use bobyqa::powalg::counters;
use bobyqa::{Bobyqa, Config, Status};

fn sphere(x: &[f64]) -> f64 {
    let mut f = 0.0;
    for &xi in x {
        f += xi * xi;
    }
    f
}

fn booth(x: &[f64]) -> f64 {
    let a = x[0] + 2.0 * x[1] - 7.0;
    let b = 2.0 * x[0] + x[1] - 5.0;
    a * a + b * b
}

/// One battery entry: a label, the run, and the `calvlag_noadd` count it produced.
fn measure(label: &str, run: impl FnOnce()) -> usize {
    counters::reset();
    run();
    let calls = counters::calvlag_noadd_calls();
    eprintln!("kernel_counts: {label:<24} calvlag_noadd = {calls}");
    calls
}

#[test]
fn report_calvlag_noadd_multiplicity() {
    // Small n, trust-region path.
    let n2 = measure("sphere n2 npt5", || {
        let mut solver = Bobyqa::new(2, Config::new(2)).expect("valid config");
        let mut x = [1.0, 2.0];
        let o = solver.minimize(sphere, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
        assert_eq!(o.status, Status::Converged);
    });

    // Larger n: multiplicity should hold (it is per-iteration structure, not n).
    let n10 = measure("sphere n10 npt21", || {
        let config = Config {
            npt: 21,
            rho_begin: 0.5,
            rho_end: 1e-6,
            max_fun: 5000,
            f_target: f64::NEG_INFINITY,
        };
        let mut solver = Bobyqa::new(10, config).expect("valid config");
        let mut x = [1.0; 10];
        let o = solver.minimize(sphere, &mut x, &[-5.0; 10], &[5.0; 10]);
        assert_eq!(o.status, Status::Converged);
    });

    // Rescue path (the booth_rescue golden's exact problem — see tests/alloc.rs).
    let rescue = measure("booth_rescue n2 npt5", || {
        let config = Config {
            npt: 5,
            rho_begin: 0.5,
            rho_end: 1e-12,
            max_fun: 500,
            f_target: f64::NEG_INFINITY,
        };
        let mut solver = Bobyqa::new(2, config).expect("valid config");
        let mut x = [0.0, 0.0];
        let o = solver.minimize(booth, &mut x, &[-10.0, -10.0], &[10.0, 2.5]);
        assert_eq!(o.status, Status::Converged);
    });

    // Sanity: the kernel is actually exercised on every path.
    assert!(n2 > 0 && n10 > 0 && rescue > 0, "counter never incremented");
}
