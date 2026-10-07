//! One-dimensional problems under both settings of `Config::prima_parity`.
//!
//! At `n = 1` the only legal `npt` is 3 = (n + 1)(n + 2) / 2, so every one-dimensional solve
//! takes the fully determined model on which the default config deviates from PRIMA (Powell's
//! RESCUE factor, see `Config::prima_parity`). These tests pin what that deviation may and may
//! not do on the case table and the quadratic fuzz below: the default must end at least as low
//! as PRIMA's own path (within the solve's accuracy), never evaluate outside the box, keep the
//! non-finite handling, and never return an error status where parity mode does not.

use bobyqa::{Bobyqa, Config, Outcome, Status};

/// One solve; returns the outcome, the final x and every evaluated point.
fn solve(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    lo: f64,
    hi: f64,
    rho_begin: f64,
    rho_end: f64,
    prima_parity: bool,
) -> (Outcome, f64, Vec<f64>) {
    let mut config = Config::new(1);
    config.rho_begin = rho_begin;
    config.rho_end = rho_end;
    config.max_fun = 2000;
    config.prima_parity = prima_parity;
    let mut solver = Bobyqa::new(1, config).expect("valid config");
    let mut x = [x0];
    let mut points = Vec::new();
    let o = solver.minimize(
        |p: &[f64]| {
            points.push(p[0]);
            f(p[0])
        },
        &mut x,
        &[lo],
        &[hi],
    );
    (o, x[0], points)
}

/// A status a caller treats as an error rather than a finished solve.
fn is_error(s: Status) -> bool {
    matches!(
        s,
        Status::InvalidArgs | Status::AllocationFailed | Status::ModelDegenerate
    )
}

struct Case {
    name: &'static str,
    f: Box<dyn Fn(f64) -> f64>,
    x0: f64,
    lo: f64,
    hi: f64,
    rho_begin: f64,
    rho_end: f64,
    /// Allowed excess of the default's final f over parity mode's.
    tol: f64,
}

fn reml(t: f64) -> f64 {
    // Profiled-REML-shaped objective of one variance parameter (150 groups of 20).
    let d = 1.0 + 20.0 * t * t;
    150.0 * d.ln() + 2980.0 * (1.0 + 64.0 / d).ln()
}

#[expect(clippy::too_many_lines)] // one table row per case, kept together for reading
fn cases() -> Vec<Case> {
    let c = |name, f: Box<dyn Fn(f64) -> f64>, x0, lo, hi, rb, re, tol| Case {
        name,
        f,
        x0,
        lo,
        hi,
        rho_begin: rb,
        rho_end: re,
        tol,
    };
    vec![
        c(
            "reml_cold",
            Box::new(reml),
            1.0,
            0.0,
            1000.0,
            0.5,
            1e-6,
            1e-9,
        ),
        c(
            "reml_warm",
            Box::new(reml),
            1.9,
            0.0,
            1000.0,
            0.1,
            1e-6,
            1e-9,
        ),
        c(
            "quad_interior",
            Box::new(|x| 4.0 * (x - 0.7).powi(2) + 1.0),
            3.0,
            -10.0,
            10.0,
            0.5,
            1e-8,
            1e-12,
        ),
        c(
            "quad_at_lower",
            Box::new(|x| (x + 2.0).powi(2)),
            3.0,
            0.0,
            5.0,
            0.5,
            1e-8,
            1e-12,
        ),
        c(
            "quad_at_upper",
            Box::new(|x| (x - 9.0).powi(2)),
            0.0,
            -1.0,
            5.0,
            0.5,
            1e-8,
            1e-12,
        ),
        c(
            "tight_box",
            Box::new(|x| (x - 0.5).powi(2)),
            0.3,
            0.3,
            0.302,
            1e-3,
            1e-9,
            1e-12,
        ),
        c("flat", Box::new(|_| 1.0), 0.2, -1.0, 1.0, 0.5, 1e-8, 0.0),
        c(
            "plateau",
            Box::new(|x: f64| ((x - 1.0).abs() - 0.5).max(0.0).powi(2)),
            4.0,
            -5.0,
            5.0,
            0.5,
            1e-8,
            1e-12,
        ),
        c(
            "step",
            Box::new(|x: f64| (4.0 * x).floor() / 4.0 + 0.01 * x * x),
            2.3,
            -3.0,
            3.0,
            0.5,
            1e-8,
            0.1,
        ),
        c(
            "noisy_1e-12",
            Box::new(|x: f64| (x - 0.3).powi(2) + 1e-12 * (1e4 * x).sin()),
            2.0,
            -5.0,
            5.0,
            0.5,
            1e-8,
            4e-12,
        ),
        c(
            "noisy_1e-8",
            Box::new(|x: f64| (x - 0.3).powi(2) + 1e-8 * (1e4 * x).sin()),
            2.0,
            -5.0,
            5.0,
            0.5,
            1e-8,
            4e-8,
        ),
        c(
            "noisy_1e-4",
            Box::new(|x: f64| (x - 0.3).powi(2) + 1e-4 * (1e4 * x).sin()),
            2.0,
            -5.0,
            5.0,
            0.5,
            1e-8,
            4e-4,
        ),
        c(
            "nan_right",
            Box::new(|x: f64| if x > 2.0 { f64::NAN } else { (x - 1.0).powi(2) }),
            0.0,
            -5.0,
            5.0,
            0.5,
            1e-8,
            1e-12,
        ),
        c(
            "inf_left",
            Box::new(|x: f64| {
                if x < 0.5 {
                    f64::INFINITY
                } else {
                    (x - 1.0).powi(2)
                }
            }),
            3.0,
            -5.0,
            5.0,
            0.5,
            1e-8,
            1e-12,
        ),
        c(
            "tiny_rho_begin",
            Box::new(|x| (x - 0.7).powi(2)),
            0.0,
            -10.0,
            10.0,
            1e-6,
            1e-9,
            1e-12,
        ),
        c(
            "wide_rho_begin",
            Box::new(|x| (x - 0.7).powi(2)),
            0.0,
            -1e6,
            1e6,
            1e4,
            1e-8,
            1e-12,
        ),
    ]
}

#[test]
fn one_dimensional_suite_default_never_worse_than_prima() {
    for case in cases() {
        let (od, xd, pd) = solve(
            &*case.f,
            case.x0,
            case.lo,
            case.hi,
            case.rho_begin,
            case.rho_end,
            false,
        );
        let (op, xp, _) = solve(
            &*case.f,
            case.x0,
            case.lo,
            case.hi,
            case.rho_begin,
            case.rho_end,
            true,
        );
        std::eprintln!(
            "{:<15} default f={:<24e} n_eval={:<4} {:?} x={:e} | parity f={:<24e} n_eval={:<4} {:?} x={:e}",
            case.name,
            od.f,
            od.n_eval,
            od.status,
            xd,
            op.f,
            op.n_eval,
            op.status,
            xp
        );
        for p in &pd {
            assert!(
                *p >= case.lo && *p <= case.hi,
                "{}: evaluated {p} outside the box",
                case.name
            );
        }
        assert!(
            !is_error(od.status) || is_error(op.status),
            "{}: default {:?} where parity mode finished with {:?}",
            case.name,
            od.status,
            op.status
        );
        assert!(
            od.f <= op.f + case.tol,
            "{}: default f {} worse than parity f {} beyond {}",
            case.name,
            od.f,
            op.f,
            case.tol
        );
    }
}

/// xorshift64*: a deterministic stream for the fuzz test (no RNG dependency).
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next_f64()
    }
}

#[test]
fn fuzz_one_dimensional_quadratics_with_noise_and_boxes() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut fewer, mut more, mut same) = (0usize, 0usize, 0usize);
    for i in 0..2000 {
        let a = 10f64.powf(rng.range(-3.0, 3.0));
        let lo = rng.range(-50.0, 50.0);
        let hi = lo + 10f64.powf(rng.range(-2.0, 2.0));
        let c = rng.range(lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo)); // optimum may sit outside
        let b = rng.range(-100.0, 100.0);
        let noise = [0.0, 1e-12, 1e-8][i % 3] * a;
        let x0 = rng.range(lo, hi);
        let rho_begin = 0.5 * (hi - lo) * rng.range(0.01, 1.0);
        let rho_end = rho_begin * 10f64.powf(rng.range(-10.0, -3.0));
        let f = move |x: f64| a * (x - c) * (x - c) + b + noise * (1e3 * x).sin();
        let (od, _, pd) = solve(&f, x0, lo, hi, rho_begin, rho_end, false);
        let (op, _, _) = solve(&f, x0, lo, hi, rho_begin, rho_end, true);
        assert!(
            pd.iter().all(|p| *p >= lo && *p <= hi),
            "case {i}: infeasible evaluation"
        );
        assert!(
            !is_error(od.status) || is_error(op.status),
            "case {i}: default {:?}, parity {:?}",
            od.status,
            op.status
        );
        // Accuracy of a solve that stops at rho_end: the model's step is O(rho_end), so f can
        // sit up to about a * (10 rho_end)^2 above the optimum, plus the noise.
        let tol = a * (10.0 * rho_end).powi(2) + 1e-12 * (1.0 + b.abs()) + 4.0 * noise;
        assert!(
            od.f <= op.f + tol,
            "case {i}: default f {} vs parity {} (tol {tol}; a={a} c={c} box=[{lo},{hi}] rho={rho_begin}->{rho_end})",
            od.f,
            op.f
        );
        match od.n_eval.cmp(&op.n_eval) {
            core::cmp::Ordering::Less => fewer += 1,
            core::cmp::Ordering::Greater => more += 1,
            core::cmp::Ordering::Equal => same += 1,
        }
    }
    std::eprintln!("fuzz n_eval default vs parity: fewer {fewer}, same {same}, more {more}");
}
