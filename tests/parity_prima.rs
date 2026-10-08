//! Golden-trajectory files from the PRIMA oracle: parser, integrity checks and parity tests.
//!
//! Goldens are captured by `oracle/capture.sh` into `tests/goldens/` in the
//! line-oriented "bobyqa golden v1" format (see `oracle/README.md`).

use std::fs;
use std::path::{Path, PathBuf};

use bobyqa::{Bobyqa, Config, Status};

/// First line every golden must carry.
const GOLDEN_MAGIC: &str = "# bobyqa golden v1";

/// Trajectory-comparison tolerance: 0.0 = bit-exact, over all 18 goldens including both
/// rescue-stressor captures (`booth_rescue`, `rosenbrock10_rescue`).
const PARITY_TOL: f64 = 0.0;

/// One `eval` line: the evaluated point and objective value, in call order.
#[derive(Debug)]
struct Eval {
    x: Vec<f64>,
    f: f64,
}

/// The `final` line: PRIMA's reported result.
#[derive(Debug)]
struct FinalLine {
    x: Vec<f64>,
    f: f64,
    n_eval: usize,
    rc: i32,
    rc_name: String,
}

/// A fully parsed golden file.
#[derive(Debug)]
struct Golden {
    problem: String,
    n: usize,
    npt: usize,
    rho_begin: f64,
    rho_end: f64,
    max_fun: usize,
    x0: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    evals: Vec<Eval>,
    final_line: FinalLine,
}

fn parse_golden(text: &str) -> Result<Golden, String> {
    if text.lines().next() != Some(GOLDEN_MAGIC) {
        return Err(format!("first line is not `{GOLDEN_MAGIC}`"));
    }

    let mut problem: Option<String> = None;
    let mut n: Option<usize> = None;
    let mut npt: Option<usize> = None;
    let mut rho_begin: Option<f64> = None;
    let mut rho_end: Option<f64> = None;
    let mut max_fun: Option<usize> = None;
    let mut x0: Option<Vec<f64>> = None;
    let mut lower: Option<Vec<f64>> = None;
    let mut upper: Option<Vec<f64>> = None;
    let mut evals: Vec<Eval> = Vec::new();
    let mut final_line: Option<FinalLine> = None;

    for (idx, line) in text.lines().enumerate() {
        let lineno = idx + 1;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let (key, rest) = tokens.split_first().expect("non-blank line has a token");
        match *key {
            "problem" => problem = Some(single_token(rest, lineno)?.to_string()),
            "n" => n = Some(parse_single(rest, lineno)?),
            "npt" => npt = Some(parse_single(rest, lineno)?),
            "rho_begin" => rho_begin = Some(parse_single(rest, lineno)?),
            "rho_end" => rho_end = Some(parse_single(rest, lineno)?),
            "max_fun" => max_fun = Some(parse_single(rest, lineno)?),
            "x0" => x0 = Some(parse_floats(rest, lineno)?),
            "lower" => lower = Some(parse_floats(rest, lineno)?),
            "upper" => upper = Some(parse_floats(rest, lineno)?),
            "eval" => {
                let n = n.ok_or_else(|| format!("line {lineno}: `eval` before `n`"))?;
                let vals = parse_floats(rest, lineno)?;
                if vals.len() != n + 1 {
                    return Err(format!(
                        "line {lineno}: `eval` wants {} floats, got {}",
                        n + 1,
                        vals.len()
                    ));
                }
                evals.push(Eval {
                    x: vals[..n].to_vec(),
                    f: vals[n],
                });
            }
            "final" => {
                let n = n.ok_or_else(|| format!("line {lineno}: `final` before `n`"))?;
                if rest.len() != n + 4 {
                    return Err(format!(
                        "line {lineno}: `final` wants {} tokens, got {}",
                        n + 4,
                        rest.len()
                    ));
                }
                final_line = Some(FinalLine {
                    x: parse_floats(&rest[..n], lineno)?,
                    f: parse_token(rest[n], lineno)?,
                    n_eval: parse_token(rest[n + 1], lineno)?,
                    rc: parse_token(rest[n + 2], lineno)?,
                    rc_name: rest[n + 3].to_string(),
                });
            }
            other => return Err(format!("line {lineno}: unknown key `{other}`")),
        }
    }

    let golden = Golden {
        problem: problem.ok_or_else(|| "missing `problem`".to_string())?,
        n: n.ok_or_else(|| "missing `n`".to_string())?,
        npt: npt.ok_or_else(|| "missing `npt`".to_string())?,
        rho_begin: rho_begin.ok_or_else(|| "missing `rho_begin`".to_string())?,
        rho_end: rho_end.ok_or_else(|| "missing `rho_end`".to_string())?,
        max_fun: max_fun.ok_or_else(|| "missing `max_fun`".to_string())?,
        x0: x0.ok_or_else(|| "missing `x0`".to_string())?,
        lower: lower.ok_or_else(|| "missing `lower`".to_string())?,
        upper: upper.ok_or_else(|| "missing `upper`".to_string())?,
        evals,
        final_line: final_line.ok_or_else(|| "missing `final`".to_string())?,
    };
    for (label, v) in [
        ("x0", &golden.x0),
        ("lower", &golden.lower),
        ("upper", &golden.upper),
    ] {
        if v.len() != golden.n {
            return Err(format!(
                "`{label}` has {} floats, expected n = {}",
                v.len(),
                golden.n
            ));
        }
    }
    Ok(golden)
}

fn single_token<'a>(rest: &[&'a str], lineno: usize) -> Result<&'a str, String> {
    match rest {
        &[one] => Ok(one),
        _ => Err(format!(
            "line {lineno}: expected exactly one token, got {}",
            rest.len()
        )),
    }
}

fn parse_single<T>(rest: &[&str], lineno: usize) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    parse_token(single_token(rest, lineno)?, lineno)
}

fn parse_token<T>(token: &str, lineno: usize) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    token
        .parse()
        .map_err(|e| format!("line {lineno}: bad value `{token}`: {e}"))
}

fn parse_floats(rest: &[&str], lineno: usize) -> Result<Vec<f64>, String> {
    rest.iter().map(|t| parse_token(t, lineno)).collect()
}

/// Parses every golden in `tests/goldens/`, panicking with the file name on
/// the first failure.
fn load_goldens() -> Vec<Golden> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens");
    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|entry| entry.expect("readable directory entry").path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "txt"))
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "tests/goldens/ has no .txt files — run oracle/capture.sh"
    );
    paths
        .iter()
        .map(|p| {
            let text =
                fs::read_to_string(p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()));
            parse_golden(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Rust problem registry — goldens store no objective, so the golden's
// `problem` name keys into these. MUST stay arithmetically bit-identical to
// `oracle/driver.c`'s objectives: same operations, same order (the driver
// compiles with -ffp-contract=off so neither side fuses FMA) —
// full-trajectory parity dies on a last-bit f difference.
// ---------------------------------------------------------------------------

fn objective(problem: &str) -> fn(&[f64]) -> f64 {
    match problem {
        "sphere" | "sphere_onbound" | "sphere_tight" => sphere,
        "rosenbrock" | "rosenbrock10" => rosenbrock,
        "booth" => booth,
        "beale" => beale,
        "powell_singular" => powell_singular,
        "nansphere" => nansphere,
        "quad1" => quad1,
        "quad1_lower" => quad1_shift,
        "quartic1" => quartic1,
        "reml1" => reml1,
        other => panic!("no Rust objective for golden problem `{other}`"),
    }
}

fn sphere(x: &[f64]) -> f64 {
    let mut f = 0.0;
    for &xi in x {
        f += xi * xi;
    }
    f
}

fn rosenbrock(x: &[f64]) -> f64 {
    let mut f = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        f += 100.0 * (a * a) + b * b;
    }
    f
}

fn booth(x: &[f64]) -> f64 {
    let a = x[0] + 2.0 * x[1] - 7.0;
    let b = 2.0 * x[0] + x[1] - 5.0;
    a * a + b * b
}

#[expect(clippy::many_single_char_names)] // a/b/c/y are the standard DFO benchmark labels
fn beale(x: &[f64]) -> f64 {
    let y = x[1];
    let a = 1.5 - x[0] + x[0] * y;
    let b = 2.25 - x[0] + x[0] * (y * y);
    let c = 2.625 - x[0] + x[0] * ((y * y) * y);
    a * a + b * b + c * c
}

#[expect(clippy::many_single_char_names)] // a/b/c/d are the standard DFO benchmark labels
fn powell_singular(x: &[f64]) -> f64 {
    let a = x[0] + 10.0 * x[1];
    let b = x[2] - x[3];
    let c = x[1] - 2.0 * x[2];
    let d = x[0] - x[3];
    let c2 = c * c;
    let d2 = d * d;
    a * a + 5.0 * (b * b) + c2 * c2 + 10.0 * (d2 * d2)
}

fn nansphere(x: &[f64]) -> f64 {
    if x[0] < 0.0 { f64::NAN } else { sphere(x) }
}

fn quad1(x: &[f64]) -> f64 {
    let d = x[0] - 0.7;
    4.0 * (d * d) + 1.0
}

fn quad1_shift(x: &[f64]) -> f64 {
    let d = x[0] + 2.0;
    d * d
}

fn quartic1(x: &[f64]) -> f64 {
    let x2 = x[0] * x[0];
    x2 * x2 - 3.0 * x2 + x[0]
}

// Uses `ln`, which is not correctly rounded on every platform: if this golden alone fails on a
// new platform, suspect that platform's `log` before the port.
fn reml1(x: &[f64]) -> f64 {
    let d = 1.0 + 20.0 * (x[0] * x[0]);
    150.0 * d.ln() + 2980.0 * (1.0 + 64.0 / d).ln()
}

// ---------------------------------------------------------------------------
// Integrity tests over the checked-in goldens
// ---------------------------------------------------------------------------

#[test]
fn n_eval_matches_the_number_of_eval_lines() {
    for g in load_goldens() {
        assert_eq!(
            g.final_line.n_eval,
            g.evals.len(),
            "{}: corrupt capture",
            g.problem
        );
    }
}

#[test]
fn final_f_was_actually_evaluated_bit_exactly() {
    // BOBYQA returns the best *evaluated* point — a capture violating this
    // is corrupt.
    for g in load_goldens() {
        assert!(
            g.evals
                .iter()
                .any(|e| e.f.to_bits() == g.final_line.f.to_bits()),
            "{}: final f {:e} (rc {} {}) is not among the evaluated f values",
            g.problem,
            g.final_line.f,
            g.final_line.rc,
            g.final_line.rc_name
        );
    }
}

// ---------------------------------------------------------------------------
// Parity test
// ---------------------------------------------------------------------------

/// NaN-aware closeness: NaN matches NaN (the `nansphere` golden logs raw NaN evaluations on
/// both sides — C and Rust print/record the objective's value before PRIMA's `moderatef`);
/// otherwise |a - b| <= tol. With tol = 0.0 this is equality that also accepts -0.0 == 0.0.
fn close(want: f64, got: f64, tol: f64) -> bool {
    (want.is_nan() && got.is_nan()) || (want - got).abs() <= tol
}

#[test]
fn repeated_minimize_on_one_solver_is_bit_identical() {
    // The reuse contract: a solver's Nth `minimize` is independent of
    // calls 1..N-1. Running every golden twice on ONE instance and comparing the full
    // evaluation trajectories bitwise catches cross-call stale-workspace leakage that the
    // fresh-solver golden test below cannot see.
    for g in load_goldens() {
        // The goldens are PRIMA captures, so they replay in PRIMA parity mode.
        let mut s = solver(&g, true);
        let (t1, x1, o1) = replay(&mut s, &g);
        let (t2, x2, o2) = replay(&mut s, &g);
        assert_eq!(t1.len(), t2.len(), "{}: trajectory length", g.problem);
        for (i, (a, b)) in t1.iter().zip(&t2).enumerate() {
            for (j, (xa, xb)) in a.0.iter().zip(&b.0).enumerate() {
                assert!(
                    xa.to_bits() == xb.to_bits(),
                    "{}: eval {i}, x[{j}] differs across reuse",
                    g.problem
                );
            }
            assert!(
                a.1.to_bits() == b.1.to_bits(),
                "{}: eval {i}, f differs across reuse",
                g.problem
            );
        }
        for (j, (a, b)) in x1.iter().zip(&x2).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "{}: final x[{j}] differs across reuse",
                g.problem
            );
        }
        assert!(
            o1.f.to_bits() == o2.f.to_bits(),
            "{}: final f differs across reuse",
            g.problem
        );
        assert_eq!(
            o1.n_eval, o2.n_eval,
            "{}: n_eval differs across reuse",
            g.problem
        );
    }
}

/// A replayed solve: (trajectory of evaluated points and values, final x, outcome).
type Replay = (Vec<(Vec<f64>, f64)>, Vec<f64>, bobyqa::Outcome);

/// A fresh solver for one golden with the given parity mode.
fn solver(g: &Golden, prima_parity: bool) -> Bobyqa {
    // f_target stays Config::new's -INFINITY default — matches the driver's hardcoded value.
    let mut config = Config::new(g.n);
    config.npt = g.npt;
    config.rho_begin = g.rho_begin;
    config.rho_end = g.rho_end;
    config.max_fun = g.max_fun;
    config.prima_parity = prima_parity;
    Bobyqa::new(g.n, config).expect("golden config is valid")
}

/// One `minimize` call on a golden's problem.
fn replay(solver: &mut Bobyqa, g: &Golden) -> Replay {
    let f = objective(&g.problem);
    let mut trajectory: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut x = g.x0.clone();
    let outcome = solver.minimize(
        |p: &[f64]| {
            let fp = f(p);
            trajectory.push((p.to_vec(), fp));
            fp
        },
        &mut x,
        &g.lower,
        &g.upper,
    );
    (trajectory, x, outcome)
}

#[test]
fn full_trajectory_matches_every_golden() {
    // The frozen goldens are the regression oracle: with PARITY_TOL = 0.0 this replays each PRIMA
    // capture and asserts the full evaluation trajectory bitwise, so any deviation is a
    // port-faithfulness or determinism regression, not a tolerance miss.
    //
    // Parity mode must reproduce every golden. `prima_parity: false` deviates from PRIMA only
    // on the fully determined model (npt = (n + 1)(n + 2) / 2, see `Config::prima_parity`), so
    // it must reproduce every golden below that npt too.
    for prima_parity in [true, false] {
        for g in load_goldens() {
            if !prima_parity && g.npt == (g.n + 1) * (g.n + 2) / 2 {
                continue;
            }
            let name = format!("{} (prima_parity = {prima_parity})", g.problem);
            let (trajectory, x, outcome) = replay(&mut solver(&g, prima_parity), &g);
            assert_eq!(trajectory.len(), g.evals.len(), "{name}: trajectory length");
            for (i, (want, got)) in g.evals.iter().zip(&trajectory).enumerate() {
                for j in 0..g.n {
                    assert!(
                        close(want.x[j], got.0[j], PARITY_TOL),
                        "{name}: eval {i}, x[{j}]"
                    );
                    // Every point at which the objective is evaluated lies in [lower, upper].
                    // Asserted directly and strictly (no slack) on the real evaluated point: the
                    // trajectory match guards it only transitively (PRIMA stays feasible), which a
                    // Rust-side clamping bug that preserved the trajectory values could slip past.
                    assert!(
                        got.0[j] >= g.lower[j] && got.0[j] <= g.upper[j],
                        "{name}: eval {i} x[{j}]={} outside [{}, {}]",
                        got.0[j],
                        g.lower[j],
                        g.upper[j]
                    );
                }
                assert!(close(want.f, got.1, PARITY_TOL), "{name}: eval {i}, f");
            }
            for (j, (want_x, got_x)) in g.final_line.x.iter().zip(&x).enumerate() {
                assert!(close(*want_x, *got_x, PARITY_TOL), "{name}: final x[{j}]");
            }
            assert!(
                close(g.final_line.f, outcome.f, PARITY_TOL),
                "{name}: final f"
            );
            assert_eq!(outcome.n_eval, g.final_line.n_eval, "{name}: n_eval");
            // All 18 goldens terminate on PRIMA_SMALL_TR_RADIUS (rc 0) -> Status::Converged. The
            // trajectory/x/f/n_eval pinned above do not cover the status mapping: a
            // status_from_info regression mapping SMALL_TR_RADIUS to the wrong variant would
            // pass them.
            assert_eq!(outcome.status, Status::Converged, "{name}: status");
        }
    }
}

#[test]
fn deviations_on_a_fully_determined_golden_converge_as_well_as_prima() {
    // The fully determined goldens (npt = (n + 1)(n + 2) / 2) may take a different path with
    // `prima_parity: false`. They must still converge, stay feasible, and end at least as low
    // as PRIMA within the solve's own accuracy.
    let mut checked = 0;
    for g in load_goldens() {
        if g.npt != (g.n + 1) * (g.n + 2) / 2 {
            continue;
        }
        checked += 1;
        let (t_dev, _, o_dev) = replay(&mut solver(&g, false), &g);
        assert_eq!(o_dev.status, Status::Converged, "{}: status", g.problem);
        assert!(
            o_dev.f <= g.final_line.f + 1e-12 * (1.0 + g.final_line.f.abs()),
            "{}: prima_parity = false f {} vs PRIMA {}",
            g.problem,
            o_dev.f,
            g.final_line.f
        );
        for (p, _) in &t_dev {
            for ((v, lo), hi) in p.iter().zip(&g.lower).zip(&g.upper) {
                assert!(v >= lo && v <= hi, "{}: infeasible", g.problem);
            }
        }
    }
    assert!(checked >= 1);
}

#[test]
fn deviations_at_the_full_npt_end_near_prima_on_rosenbrock_from_the_classic_start() {
    // At npt = 6 with a coarse rho_end the model built at the larger rho is not yet accurate
    // when rho reaches rho_end, so with the fewer RESCUE calls of `prima_parity = false` the
    // solve could end far above PRIMA's final value.
    let solve = |prima_parity: bool| {
        let mut config = Config::new(2);
        config.npt = 6;
        config.rho_begin = 0.5;
        config.rho_end = 1e-2;
        config.prima_parity = prima_parity;
        let mut x = [-1.2, 1.0];
        Bobyqa::new(2, config)
            .expect("valid config")
            .minimize(rosenbrock, &mut x, &[-3.0; 2], &[3.0; 2])
    };
    let (dev, prima) = (solve(false), solve(true));
    assert_eq!(dev.status, Status::Converged);
    assert!(
        dev.f <= 10.0 * prima.f,
        "prima_parity = false f {} vs PRIMA {}",
        dev.f,
        prima.f
    );
}
