//! oracle/rusttime/basintime — the `basin`-crate half of the wall-clock comparison.
//!
//! Same registry, same objectives, same measurement rule as `rusttime`, so the
//! three-way table compares the same solves. One caveat the table must carry:
//! `basin`'s BOBYQA is not bit-exact with PRIMA, so it walks its own trajectory
//! and spends its own number of evaluations — `nf` and `f` are printed for every
//! case and belong next to the time whenever these numbers are quoted.
//!
//! Only a "cold" number exists here: `basin` drives a solve through an
//! `Executor` that consumes a fresh `BobyqaState` per run, so there is no
//! reuse path to time.
//!
//! Usage: basintime <problem> [--npt N] [--rhobeg X] [--rhoend X] [--maxfun N] --time SECONDS

use basin::{Bobyqa, BobyqaState, BoxConstraints, CostFunction, Executor, MaxCostEvals};
use rusttime::{parse_args, time_min};

/// The registry entry as `basin` wants it: the objective behind `CostFunction`,
/// the box behind `BoxConstraints`. Bounds are owned `Vec`s because
/// `BoxConstraints` hands back `&Self::Param`.
struct Case {
    fun: fn(&[f64]) -> f64,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for Case {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok((self.fun)(x))
    }
}

impl BoxConstraints for Case {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let (p, settings) = parse_args(&argv);

    let n = p.x0.len();
    let npt = settings.npt;
    let max_fun = settings.max_fun;

    let run = || {
        let problem = Case { fun: p.fun, lower: p.lower.to_vec(), upper: p.upper.to_vec() };
        let solver = Bobyqa::new().with_npt(npt).with_rho_beg(settings.rho_begin).with_rho_end(settings.rho_end);
        let state = BobyqaState::new(p.x0.to_vec());
        Executor::new(problem, solver, state).terminate_on(MaxCostEvals(max_fun as u64)).run().expect("infallible objective")
    };

    let outcome = run(); // warm-up, discarded
    let (cold_ns, cold_reps) = time_min(settings.budget, || {
        std::hint::black_box(run());
    });

    println!(
        "time basin {} n {n} npt {npt} cold_reps {cold_reps} cold_min_ns {cold_ns:.0} \
         f {:e} nf {} status {:?} x {:?}",
        p.name,
        outcome.best_cost(),
        outcome.cost_evals(),
        outcome.reason,
        outcome.best_param()
    );
}
