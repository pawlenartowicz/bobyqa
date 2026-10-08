//! Shared half of the wall-clock comparison harness: the problem registry, the
//! objectives, the flag parser and the measurement rule. Two binaries use it —
//! `rusttime` (the `bobyqa` crate) and `basintime` (`basin`'s BOBYQA) — so both
//! solve literally the same problems under the same timing protocol.

use std::time::Instant;

/// Objectives — must stay arithmetically identical to `driver.c`'s renderings:
/// same operations in the same order, so both sides walk the same trajectory
/// and the timing compares equal work.
pub fn sphere(x: &[f64]) -> f64 {
    let mut f = 0.0;
    for &xi in x {
        f += xi * xi;
    }
    f
}

pub fn rosenbrock(x: &[f64]) -> f64 {
    let mut f = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        f += 100.0 * (a * a) + b * b;
    }
    f
}

pub fn booth(x: &[f64]) -> f64 {
    let a = x[0] + 2.0 * x[1] - 7.0;
    let b = 2.0 * x[0] + x[1] - 5.0;
    a * a + b * b
}

pub fn beale(x: &[f64]) -> f64 {
    let y = x[1];
    let a = 1.5 - x[0] + x[0] * y;
    let b = 2.25 - x[0] + x[0] * (y * y);
    let c = 2.625 - x[0] + x[0] * ((y * y) * y);
    a * a + b * b + c * c
}

pub fn powell_singular(x: &[f64]) -> f64 {
    let a = x[0] + 10.0 * x[1];
    let b = x[2] - x[3];
    let c = x[1] - 2.0 * x[2];
    let d = x[0] - x[3];
    let c2 = c * c;
    let d2 = d * d;
    a * a + 5.0 * (b * b) + c2 * c2 + 10.0 * (d2 * d2)
}

pub fn nansphere(x: &[f64]) -> f64 {
    if x[0] < 0.0 {
        return f64::NAN;
    }
    sphere(x)
}

pub struct Problem {
    pub name: &'static str,
    pub fun: fn(&[f64]) -> f64,
    pub x0: &'static [f64],
    pub lower: &'static [f64],
    pub upper: &'static [f64],
    pub rho_begin: f64,
    pub rho_end: f64,
    pub max_fun: usize,
}

/// The nine `n >= 2` entries of `driver.c`'s `REGISTRY`, same values; its four one-dimensional
/// problems are left out.
pub const REGISTRY: &[Problem] = &[
    Problem { name: "sphere", fun: sphere, x0: &[1.0, 2.0], lower: &[-5.0, -5.0], upper: &[5.0, 5.0], rho_begin: 0.5, rho_end: 1e-6, max_fun: 500 },
    Problem { name: "rosenbrock", fun: rosenbrock, x0: &[-1.2, 1.0], lower: &[-5.0, -5.0], upper: &[10.0, 10.0], rho_begin: 0.5, rho_end: 1e-6, max_fun: 500 },
    Problem { name: "booth", fun: booth, x0: &[0.0, 0.0], lower: &[-10.0, -10.0], upper: &[10.0, 2.5], rho_begin: 0.5, rho_end: 1e-6, max_fun: 500 },
    Problem {
        name: "rosenbrock10",
        fun: rosenbrock,
        x0: &[-1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0],
        lower: &[-5.0; 10],
        upper: &[10.0; 10],
        rho_begin: 0.5,
        rho_end: 1e-6,
        max_fun: 2000,
    },
    Problem { name: "beale", fun: beale, x0: &[1.0, 1.0], lower: &[-4.5, -4.5], upper: &[4.5, 4.5], rho_begin: 0.5, rho_end: 1e-6, max_fun: 500 },
    Problem { name: "powell_singular", fun: powell_singular, x0: &[3.0, -1.0, 0.0, 1.0], lower: &[-4.0; 4], upper: &[5.0; 4], rho_begin: 0.5, rho_end: 1e-6, max_fun: 2000 },
    Problem { name: "sphere_onbound", fun: sphere, x0: &[1.2, 0.3], lower: &[1.0, -5.0], upper: &[6.0, 5.0], rho_begin: 0.5, rho_end: 1e-6, max_fun: 500 },
    Problem { name: "sphere_tight", fun: sphere, x0: &[0.55, -0.3], lower: &[-0.6, -0.6], upper: &[0.6, 0.6], rho_begin: 0.5, rho_end: 1e-6, max_fun: 500 },
    Problem { name: "nansphere", fun: nansphere, x0: &[0.5, 2.0], lower: &[-5.0, -5.0], upper: &[5.0, 5.0], rho_begin: 0.5, rho_end: 1e-6, max_fun: 500 },
];

/// Per-run settings after the command line has been applied over the registry
/// defaults. `npt` starts at `2n+1`, PRIMA's default.
pub struct Settings {
    pub npt: usize,
    pub rho_begin: f64,
    pub rho_end: f64,
    pub max_fun: usize,
    pub budget: f64,
}

/// `<problem> [--npt N] [--rhobeg X] [--rhoend X] [--maxfun N] --time SECONDS`,
/// the same flag set `driver.c` accepts. Exits with status 2 on a bad command line.
pub fn parse_args(argv: &[String]) -> (&'static Problem, Settings) {
    if argv.len() < 2 {
        eprintln!("usage: {} <problem> [--npt N] [--rhobeg X] [--rhoend X] [--maxfun N] --time SECONDS", argv[0]);
        std::process::exit(2);
    }

    let p = REGISTRY.iter().find(|p| p.name == argv[1]).unwrap_or_else(|| {
        eprintln!("unknown problem '{}'", argv[1]);
        std::process::exit(2);
    });

    let mut settings = Settings { npt: 2 * p.x0.len() + 1, rho_begin: p.rho_begin, rho_end: p.rho_end, max_fun: p.max_fun, budget: 0.0 };

    let mut i = 2;
    while i < argv.len() {
        let value = argv.get(i + 1).unwrap_or_else(|| {
            eprintln!("flag '{}' needs a value", argv[i]);
            std::process::exit(2);
        });
        match argv[i].as_str() {
            "--npt" => settings.npt = value.parse().expect("npt"),
            "--rhobeg" => settings.rho_begin = value.parse().expect("rhobeg"),
            "--rhoend" => settings.rho_end = value.parse().expect("rhoend"),
            "--maxfun" => settings.max_fun = value.parse().expect("maxfun"),
            "--time" => settings.budget = value.parse().expect("time"),
            other => {
                eprintln!("unknown flag '{other}'");
                std::process::exit(2);
            }
        }
        i += 2;
    }

    (p, settings)
}

/// Repeat `run` until `budget` seconds are spent; return (min nanoseconds, reps).
/// The caller does its own warm-up pass first.
pub fn time_min(budget: f64, mut run: impl FnMut()) -> (f64, u32) {
    let mut best = f64::INFINITY;
    let mut total = 0.0;
    let mut reps = 0u32;
    while total < budget && reps < 1_000_000 {
        let t0 = Instant::now();
        run();
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed < best {
            best = elapsed;
        }
        total += elapsed;
        reps += 1;
    }
    (best * 1e9, reps)
}
