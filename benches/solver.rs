//! Warm-path `minimize` micro-bench at small dims, the cheap-objective high-throughput
//! regime (batch refits): n = 2–6, npt at both a lean ⌈1.5n⌉+1 setting (n ≥ 3) and
//! Powell's 2n+1 default, with an aggressive rho schedule (0.5 → 1e-6). Per-call cost
//! of a re-solve on a warm `Bobyqa` is the headline number.
//!
//! The objectives are deterministic and cheap on purpose — that pushes the measurement onto
//! the model-maintenance kernels (trsbox/updateh/matprod/geostep), which is what the shape
//! work targets. Parity-safe changes keep trajectories bit-identical, so before/after numbers
//! compare the same instruction trace.
#![allow(missing_docs)] // criterion_group!/criterion_main! expand to undocumented pub items
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use bobyqa::{Bobyqa, Config};

/// Chained Rosenbrock — valid for any n >= 2, interior optimum at (1, …, 1).
fn rosenbrock(x: &[f64]) -> f64 {
    let mut f = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        f += 100.0 * a * a + b * b;
    }
    f
}

/// The (n, npt) grid mixes the lean ⌈1.5n⌉+1 setting (n = 3, 6) with Powell's 2n+1
/// default (n = 2–5), covering both interpolation-set sizes the warm path sees in practice.
const CASES: &[(usize, usize)] = &[(2, 5), (3, 6), (6, 10), (3, 7), (4, 9), (5, 11)];

fn bench_minimize(c: &mut Criterion) {
    for &(n, npt) in CASES {
        let config = Config {
            npt,
            rho_begin: 0.5,
            rho_end: 1e-6,
            ..Config::new(n)
        };

        // Interior optimum: the full TR loop with mostly-free variables.
        let mut solver = Bobyqa::new(n, config).unwrap();
        let lower = vec![-5.0; n];
        let upper = vec![5.0; n];
        // Start-point buffer hoisted out of the timed region: the headline is the
        // zero-alloc warm path, so the bench itself must not allocate per call.
        let mut x = vec![0.0; n];
        c.bench_function(&format!("minimize/rosen/n{n}_npt{npt}"), |b| {
            b.iter(|| {
                x.fill(0.0);
                let out = solver.minimize(rosenbrock, black_box(&mut x), &lower, &upper);
                black_box(out.f)
            });
        });

        // Bound-clipped optimum (upper = 0.9 < 1): exercises the xbdi/active-bound paths
        // that an interior solve never touches.
        let mut solver = Bobyqa::new(n, config).unwrap();
        let upper_clip = vec![0.9; n];
        c.bench_function(&format!("minimize/rosen_clip/n{n}_npt{npt}"), |b| {
            b.iter(|| {
                x.fill(0.0);
                let out = solver.minimize(rosenbrock, black_box(&mut x), &lower, &upper_clip);
                black_box(out.f)
            });
        });
    }
}

criterion_group!(benches, bench_minimize);
criterion_main!(benches);
