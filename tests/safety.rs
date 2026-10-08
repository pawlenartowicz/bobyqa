//! Reuse after an objective panic.
//!
//! `Bobyqa::minimize` documents that it re-initializes whatever it reads, so a solver whose
//! objective unwound mid-solve must afterwards reproduce a clean solver's trajectory EXACTLY
//! (bit-identical x and f, equal `n_eval`). If this test ever fails, the documented contract —
//! not the state reset — is what must change ("solver must be dropped after an unwind").

use std::panic::{AssertUnwindSafe, catch_unwind};

use bobyqa::{Bobyqa, Config, Status};

/// Booth — the same arithmetic as the `tests/parity_prima.rs` golden objective.
fn booth(x: &[f64]) -> f64 {
    let a = x[0] + 2.0 * x[1] - 7.0;
    let b = 2.0 * x[0] + x[1] - 5.0;
    a * a + b * b
}

#[test]
fn minimize_after_an_objective_panic_reproduces_the_clean_trajectory() {
    let config = Config::new(2);
    let (lower, upper) = ([-10.0, -10.0], [10.0, 10.0]);

    // The reference trajectory, from a fresh solver.
    let mut fresh = Bobyqa::new(2, config).expect("valid config");
    let mut x_ref = [0.0, 0.0];
    let o_ref = fresh.minimize(booth, &mut x_ref, &lower, &upper);
    assert_eq!(o_ref.status, Status::Converged);

    // The same problem on another solver, but the objective panics at evaluation 3 —
    // deep enough that the workspace holds a partially built interpolation set.
    let mut solver = Bobyqa::new(2, config).expect("valid config");
    let mut calls = 0_usize;
    let unwound = catch_unwind(AssertUnwindSafe(|| {
        let mut x = [0.0, 0.0];
        solver.minimize(
            |p: &[f64]| {
                calls += 1;
                assert!(calls < 3, "objective blew up (deliberate test panic)");
                booth(p)
            },
            &mut x,
            &lower,
            &upper,
        )
    }));
    assert!(unwound.is_err(), "the objective's panic must propagate");

    // Reuse after the unwind: the frozen reference trajectory, exactly.
    let mut x = [0.0, 0.0];
    let o = solver.minimize(booth, &mut x, &lower, &upper);
    assert_eq!(o.status, o_ref.status);
    assert_eq!(
        o.n_eval, o_ref.n_eval,
        "post-unwind reuse left the trajectory"
    );
    assert_eq!(o.f.to_bits(), o_ref.f.to_bits(), "f must be bit-identical");
    for (a, b) in x.iter().zip(&x_ref) {
        assert_eq!(a.to_bits(), b.to_bits(), "x must be bit-identical");
    }
}
