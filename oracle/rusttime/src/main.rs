//! oracle/rusttime — the `bobyqa`-crate half of the wall-clock comparison.
//!
//! Mirrors `oracle/driver.c`'s timing mode exactly: same problem registry, same
//! objectives, same measurement rule (one warm-up solve discarded, then repeat
//! until a time budget is spent and report the minimum). Timing noise is
//! one-sided — interference only ever slows a run — so the minimum is the
//! cleanest estimate of the cost. The registry, objectives, flags and timer live
//! in `lib.rs`, shared with `basintime`.
//!
//! Two Rust numbers are reported per problem, because the crate's calling
//! convention differs from PRIMA's:
//!   `cold` — `Bobyqa::new` + `minimize`, the like-for-like analogue of one
//!            `prima_minimize` call (which allocates its own workspace).
//!   `warm` — `minimize` alone on a solver built once, the zero-alloc reuse
//!            path a caller gets when solving many problems of the same size.
//!
//! Usage: rusttime <problem> [--npt N] [--rhobeg X] [--rhoend X] [--maxfun N] --time SECONDS

use bobyqa::{Bobyqa, Config};
use rusttime::{parse_args, time_min};

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let (p, settings) = parse_args(&argv);

    let n = p.x0.len();
    let mut config = Config::new(n);
    config.npt = settings.npt;
    config.rho_begin = settings.rho_begin;
    config.rho_end = settings.rho_end;
    config.max_fun = settings.max_fun;

    let npt = config.npt;
    let budget = settings.budget;
    let mut x = vec![0.0; n];

    // cold: one Bobyqa::new + minimize per repetition — the analogue of one
    // prima_minimize call, which allocates its own workspace each time.
    let cold_run = |x: &mut Vec<f64>| {
        x.copy_from_slice(p.x0);
        let mut solver = Bobyqa::new(n, config.clone()).expect("valid config");
        solver.minimize(p.fun, x, p.lower, p.upper)
    };
    let outcome = cold_run(&mut x); // warm-up, discarded
    let (cold_ns, cold_reps) = time_min(budget, || {
        std::hint::black_box(cold_run(&mut x));
    });

    // warm: solver built once, minimize only — the crate's zero-alloc reuse path.
    let mut solver = Bobyqa::new(n, config.clone()).expect("valid config");
    x.copy_from_slice(p.x0);
    solver.minimize(p.fun, &mut x, p.lower, p.upper); // warm-up, discarded
    let (warm_ns, warm_reps) = time_min(budget, || {
        x.copy_from_slice(p.x0);
        std::hint::black_box(solver.minimize(p.fun, &mut x, p.lower, p.upper));
    });

    println!(
        "time rust {} n {n} npt {npt} cold_reps {cold_reps} cold_min_ns {cold_ns:.0} \
         warm_reps {warm_reps} warm_min_ns {warm_ns:.0} f {:.17} nf {} status {:?} x {:?}",
        p.name, outcome.f, outcome.n_eval, outcome.status, x
    );
}
