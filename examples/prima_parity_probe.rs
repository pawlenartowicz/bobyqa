//! Measures `Config::prima_parity = false` against parity mode on objectives the test suite
//! does not compare: kinked and smooth non-quadratic one-dimensional shapes, and `n >= 2`
//! problems at the maximum `npt = (n + 1)(n + 2) / 2`, the only `npt` where the modes differ.
//!
//! Run: `cargo run --release --example prima_parity_probe`. Deterministic: every case comes
//! from a fixed seed, so the printed table is the same on every run on a given target.
//!
//! Columns: `worse` / `better` count the cases where `false` ended above / below parity mode
//! by more than the tolerance named in the heading; `evals` are totals over all cases.

use bobyqa::{Bobyqa, Config, Outcome};

/// splitmix64: a deterministic stream with no RNG dependency.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.unit()
    }
}

/// One solve at the maximum `npt` for `x0.len()`.
fn solve(
    f: &dyn Fn(&[f64]) -> f64,
    x0: &[f64],
    bounds: (&[f64], &[f64]),
    rho: (f64, f64),
    prima_parity: bool,
) -> Outcome {
    let n = x0.len();
    let mut config = Config::new(n);
    config.npt = (n + 1) * (n + 2) / 2;
    config.rho_begin = rho.0;
    config.rho_end = rho.1;
    config.max_fun = 3000 * n;
    config.prima_parity = prima_parity;
    let mut x = x0.to_vec();
    Bobyqa::new(n, config)
        .expect("valid config")
        .minimize(f, &mut x, bounds.0, bounds.1)
}

#[derive(Default)]
struct Tally {
    worse: usize,
    better: usize,
    evals_deviating: usize,
    evals_parity: usize,
}

impl Tally {
    fn add(&mut self, deviating: &Outcome, parity: &Outcome, tol: f64) {
        if deviating.f > parity.f + tol {
            self.worse += 1;
        } else if deviating.f < parity.f - tol {
            self.better += 1;
        }
        self.evals_deviating += deviating.n_eval;
        self.evals_parity += parity.n_eval;
    }

    fn print(&self, heading: &str) {
        println!(
            "{heading:<52} worse {:>4}  better {:>4}  evals {:>7} vs {:>7}",
            self.worse, self.better, self.evals_deviating, self.evals_parity
        );
    }
}

/// One-dimensional shapes on random boxes, optima, starts and resolutions.
fn one_dimensional(shape: &str) {
    let mut rng = Rng(7);
    let mut tally = Tally::default();
    for _ in 0..1000 {
        let lo = -rng.range(0.1, 10.0);
        let hi = rng.range(0.1, 10.0);
        let c = rng.range(lo, hi);
        let s = 10f64.powf(rng.range(-2.0, 2.0));
        let x0 = rng.range(lo, hi);
        let rho_begin = (0.5 * (hi - lo)).min((hi - lo) * 10f64.powf(rng.range(-3.0, -0.5)));
        let rho_end = rho_begin * 10f64.powf(rng.range(-8.0, -2.0));
        let g = |t: f64| match shape {
            "sqrt|x-c|" => s * t.abs().sqrt(),
            "|x-c|" => s * t.abs(),
            "(x-c)^4" => s * t.powi(4),
            "cosh(x-c)+0.1|x-c|^3" => s * (t.cosh() + 0.1 * t.powi(3).abs()),
            _ => s * (3.0 * t).cosh().ln(),
        };
        let f = |x: &[f64]| g(x[0] - c);
        let f_range = [g(lo - c), g(hi - c), g(0.0)];
        let tol = 1e-4
            * (f_range.iter().copied().fold(f64::MIN, f64::max)
                - f_range.iter().copied().fold(f64::MAX, f64::min));
        let run = |p| solve(&f, &[x0], (&[lo], &[hi]), (rho_begin, rho_end), p);
        tally.add(&run(false), &run(true), tol);
    }
    tally.print(&format!("n=1 {shape}, tol 1e-4 of the f range"));
}

fn quartic(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, &v)| (v - 0.3 * (i as f64 + 1.0)).powi(4) + 0.01 * (v - 0.1).powi(2))
        .sum()
}

fn powell_singular(x: &[f64]) -> f64 {
    (x[0] + 10.0 * x[1]).powi(2)
        + 5.0 * (x[2] - x[3]).powi(2)
        + (x[1] - 2.0 * x[2]).powi(4)
        + 10.0 * (x[0] - x[3]).powi(4)
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn chained_rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 4.0 * (w[0] - w[1] * w[1]).powi(2) + (1.0 - w[1]).powi(2))
        .sum()
}

type Problem = (&'static str, fn(&[f64]) -> f64, usize, f64);

const PROBLEMS: [Problem; 5] = [
    ("rosenbrock", rosenbrock, 2, 3.0),
    ("rosenbrock", rosenbrock, 4, 3.0),
    ("chained rosenbrock", chained_rosenbrock, 6, 3.0),
    ("powell singular", powell_singular, 4, 3.0),
    ("quartic", quartic, 5, 3.0),
];

/// 100 random starts in `[-0.8 half, 0.8 half]^n`, box `[-half, half]^n`, `rho_begin = 0.5`.
/// `rho_end_deviating` lets the equal-budget comparison give `false` a finer resolution.
fn many_starts(problem: &Problem, rho_end: f64, rho_end_deviating: f64, seed: u64) -> Tally {
    let (_, f, n, half) = *problem;
    let mut rng = Rng(seed);
    let (lo, hi) = (vec![-half; n], vec![half; n]);
    let mut tally = Tally::default();
    for _ in 0..100 {
        let x0: Vec<f64> = (0..n).map(|_| rng.range(-0.8 * half, 0.8 * half)).collect();
        let parity = solve(&f, &x0, (&lo, &hi), (0.5, rho_end), true);
        let deviating = solve(&f, &x0, (&lo, &hi), (0.5, rho_end_deviating), false);
        tally.add(&deviating, &parity, 1e-8_f64.max(1e-6 * parity.f.abs()));
    }
    tally
}

fn main() {
    println!("== one-dimensional shapes, 1000 cases each");
    for shape in [
        "sqrt|x-c|",
        "|x-c|",
        "(x-c)^4",
        "cosh(x-c)+0.1|x-c|^3",
        "ln cosh 3(x-c)",
    ] {
        one_dimensional(shape);
    }
    for rho_end in [1e-2, 1e-4, 1e-6, 1e-8] {
        println!("== n >= 2 at the maximum npt, rho 0.5 -> {rho_end:e}, 100 starts each");
        for (seed, p) in (3u64..).zip(PROBLEMS.iter()) {
            many_starts(p, rho_end, rho_end, seed).print(&format!("{} n={}", p.0, p.2));
        }
    }
    for rho_end in [1e-2, 1e-4] {
        println!(
            "== equal budget: false at rho_end {:e} vs parity at {rho_end:e}",
            rho_end / 10.0
        );
        for (seed, p) in (3u64..).zip(PROBLEMS.iter()) {
            many_starts(p, rho_end, rho_end / 10.0, seed).print(&format!("{} n={}", p.0, p.2));
        }
    }
}
