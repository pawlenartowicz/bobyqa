//! Test-only support: the "bobyqa state v1" parser, diff assertions with bit-exactness tracking,
//! and the Rust problem registry for replaying captured states (design §3.6-3.8).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::mat::Mat;

/// Subroutine-diff tolerance (design §3.7): integer/index/flag outputs match exactly; f64 outputs
/// per-element within `|a-b| <= REL_TOL * max(|a|, |b|)`, bit-exactness tracked and reported.
///
/// Calibrated on the M1a `prelim` corpus (2026-06-04, PRIMA `1d76fb88`, gfortran 15.2.1,
/// `-ffp-contract=off`): 3674 f64 outputs (initxf 1050, initq 468, inith 2156), 100% bit-exact,
/// max rel dev 0. Kept at 1e-14 rather than 0: the FP pin makes `prelim` exactly reproducible,
/// but M1b's heavier routines will accumulate ulp-level reduction differences (design §3.7 —
/// any `prelim` deviation above this is a bug, not a tolerance problem).
pub(crate) const STATE_DIFF_REL_TOL: f64 = 1e-14;

const STATE_MAGIC: &str = "# bobyqa state v1";

#[derive(Debug)]
pub(crate) enum Value {
    Scalar(String), // typed on access: f64, i64, usize
    Vector(Vec<f64>),
    Matrix(Mat),
    IMatrix { nrows: usize, data: Vec<i64> }, // column-major
}

#[derive(Debug, Default)]
pub(crate) struct Section(HashMap<String, Value>);

#[derive(Debug)]
pub(crate) struct State {
    pub(crate) routine: String,
    pub(crate) problem: String,
    pub(crate) entry: Section,
    pub(crate) exit: Section,
}

impl Section {
    fn get(&self, name: &str) -> &Value {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("state field `{name}` missing"))
    }

    pub(crate) fn f64(&self, name: &str) -> f64 {
        match self.get(name) {
            Value::Scalar(s) => s
                .parse()
                .unwrap_or_else(|_| panic!("`{name}`: bad f64 `{s}`")),
            _ => panic!("`{name}` is not a scalar"),
        }
    }

    pub(crate) fn i64(&self, name: &str) -> i64 {
        match self.get(name) {
            Value::Scalar(s) => s
                .parse()
                .unwrap_or_else(|_| panic!("`{name}`: bad int `{s}`")),
            _ => panic!("`{name}` is not a scalar"),
        }
    }

    pub(crate) fn usize(&self, name: &str) -> usize {
        usize::try_from(self.i64(name)).unwrap_or_else(|_| panic!("`{name}` is negative"))
    }

    pub(crate) fn vec(&self, name: &str) -> Vec<f64> {
        match self.get(name) {
            Value::Vector(v) => v.clone(),
            _ => panic!("`{name}` is not a vector"),
        }
    }

    pub(crate) fn mat(&self, name: &str) -> Mat {
        match self.get(name) {
            Value::Matrix(m) => m.clone(),
            _ => panic!("`{name}` is not a matrix"),
        }
    }

    /// PRIMA's 1-based IJ(2, m) -> 0-based (i, j) pairs (design §3.2 sentinel/index translation).
    pub(crate) fn ij(&self, name: &str) -> Vec<(usize, usize)> {
        match self.get(name) {
            Value::IMatrix { nrows, data } => {
                assert_eq!(*nrows, 2, "`{name}` is not 2-row");
                data.chunks_exact(2)
                    .map(|c| {
                        (
                            usize::try_from(c[0] - 1).unwrap(),
                            usize::try_from(c[1] - 1).unwrap(),
                        )
                    })
                    .collect()
            }
            _ => panic!("`{name}` is not an imatrix"),
        }
    }
}

fn parse_section(lines: &mut std::iter::Peekable<std::str::Lines<'_>>) -> Section {
    let mut sec = Section::default();
    while let Some(line) = lines.peek() {
        let mut t = line.split_whitespace();
        match t.next() {
            Some("scalar") => {
                let name = t.next().expect("scalar name").to_string();
                sec.0.insert(
                    name,
                    Value::Scalar(t.next().expect("scalar value").to_string()),
                );
            }
            Some("vector") => {
                let name = t.next().expect("vector name").to_string();
                let v: Vec<f64> = t.map(|s| s.parse().expect("vector float")).collect();
                sec.0.insert(name, Value::Vector(v));
            }
            Some(kind @ ("matrix" | "imatrix")) => {
                let name = t.next().expect("matrix name").to_string();
                let nr: usize = t.next().expect("nrows").parse().expect("nrows");
                let nc: usize = t.next().expect("ncols").parse().expect("ncols");
                if kind == "matrix" {
                    let data: Vec<f64> = t.map(|s| s.parse().expect("matrix float")).collect();
                    assert_eq!(data.len(), nr * nc, "`{name}`: shape/data mismatch");
                    sec.0
                        .insert(name, Value::Matrix(Mat::from_col_major(nr, nc, data)));
                } else {
                    let data: Vec<i64> = t.map(|s| s.parse().expect("matrix int")).collect();
                    assert_eq!(data.len(), nr * nc, "`{name}`: shape/data mismatch");
                    sec.0.insert(name, Value::IMatrix { nrows: nr, data });
                }
            }
            _ => return sec, // `exit` line or end of body
        }
        lines.next();
    }
    sec
}

pub(crate) fn parse_state(text: &str) -> State {
    let mut lines = text.lines().peekable();
    assert_eq!(
        lines.next(),
        Some(STATE_MAGIC),
        "not a bobyqa state v1 file"
    );
    let mut routine = None;
    let mut problem = None;
    while let Some(line) = lines.peek() {
        let mut t = line.split_whitespace();
        match t.next() {
            // `#` comments and blank lines skipped; npt/rho_*/max_fun/seq are provenance only.
            Some("#" | "npt" | "rho_begin" | "rho_end" | "max_fun" | "seq") | None => {}
            Some("routine") => routine = Some(t.next().expect("routine").to_string()),
            Some("problem") => problem = Some(t.next().expect("problem").to_string()),
            Some("entry") => break,
            Some(k) => panic!("unknown header key `{k}`"),
        }
        lines.next();
    }
    assert_eq!(
        lines.next().map(str::trim),
        Some("entry"),
        "missing `entry`"
    );
    let entry = parse_section(&mut lines);
    assert_eq!(lines.next().map(str::trim), Some("exit"), "missing `exit`");
    let exit = parse_section(&mut lines);
    State {
        routine: routine.expect("missing `routine`"),
        problem: problem.expect("missing `problem`"),
        entry,
        exit,
    }
}

/// Loads every state file for `routine`, panicking (with a capture hint) when none exist.
pub(crate) fn load_states(routine: &str) -> Vec<State> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("states")
        .join(routine);
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e} — run oracle/capture_states.sh", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|p| {
            let text = fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let st = parse_state(&text);
            assert_eq!(st.routine, routine, "{}: routine mismatch", p.display());
            st
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Diff assertions (design §3.7): exact ints, relative-tol floats, bit-exactness tracked.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub(crate) struct DiffStats {
    pub(crate) checked: usize,
    pub(crate) bit_exact: usize,
    pub(crate) max_rel: f64,
}

impl DiffStats {
    pub(crate) fn f64(&mut self, what: &str, got: f64, want: f64) {
        self.checked += 1;
        if got.to_bits() == want.to_bits() {
            self.bit_exact += 1;
            return;
        }
        // Design §3.7 in inequality form — no division, so signed-zero pairs (-0.0 vs 0.0:
        // bit-different but |a-b| = 0) pass instead of tripping on rel = 0/0 = NaN.
        let scale = got.abs().max(want.abs());
        let diff = (got - want).abs();
        assert!(
            diff <= STATE_DIFF_REL_TOL * scale,
            "{what}: got {got:e}, want {want:e}, rel {:e} > {STATE_DIFF_REL_TOL:e}",
            diff / scale
        );
        if scale > 0.0 {
            self.max_rel = self.max_rel.max(diff / scale);
        }
    }

    pub(crate) fn slice(&mut self, what: &str, got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            self.f64(&format!("{what}[{k}]"), *g, *w);
        }
    }

    pub(crate) fn mat(&mut self, what: &str, got: &Mat, want: &Mat) {
        assert_eq!(
            (got.nrows(), got.ncols()),
            (want.nrows(), want.ncols()),
            "{what}: shape"
        );
        self.slice(what, got.data(), want.data());
    }

    /// Print the calibration summary (visible with `--nocapture`; Task 15 reads it).
    pub(crate) fn report(&self, label: &str) {
        eprintln!(
            "{label}: {} values, {} bit-exact ({:.1}%), max rel dev {:e}",
            self.checked,
            self.bit_exact,
            100.0 * self.bit_exact as f64 / self.checked.max(1) as f64,
            self.max_rel
        );
    }
}

// ---------------------------------------------------------------------------
// Problem registry. THIRD bit-identical copy of the objectives — the others live in
// oracle/driver.c (C) and tests/parity_prima.rs (integration test, which cannot share code with
// unit tests). Same operations, same order; change all three together.
// ---------------------------------------------------------------------------

pub(crate) fn objective(problem: &str) -> fn(&[f64]) -> f64 {
    match problem {
        "sphere" => sphere,
        "rosenbrock" | "rosenbrock10" => rosenbrock,
        "booth" => booth,
        other => panic!("no Rust objective for state problem `{other}`"),
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

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# bobyqa state v1
# prima 0123456789abcdef0123456789abcdef01234567
routine initq
problem sphere
npt 6
rho_begin 0.5
rho_end 1e-06
max_fun 500
seq 1
entry
imatrix ij 2 1 2 1
vector fval 5 4.25 6 7 8 9
matrix xpt 2 6 0 0 0.5 0 0 0.5 -0.5 0 0 -0.5 -0.5 0.5
exit
vector gopt 1 2
matrix hq 2 2 1 0 0 1
vector pq 0 0 0 0 0 0
scalar info 0
";

    #[test]
    fn the_embedded_sample_parses_field_for_field() {
        let st = parse_state(SAMPLE);
        assert_eq!(st.routine, "initq");
        assert_eq!(st.problem, "sphere");
        assert_eq!(st.entry.ij("ij"), vec![(1, 0)]); // PRIMA (2,1) -> 0-based
        assert_eq!(st.entry.vec("fval").len(), 6);
        assert_eq!(st.entry.mat("xpt").ncols(), 6);
        assert_eq!(st.exit.i64("info"), 0);
        assert_eq!(st.exit.mat("hq")[[1, 1]], 1.0);
    }

    #[test]
    fn diff_stats_track_bit_exactness_and_tolerance() {
        let mut s = DiffStats::default();
        s.f64("a", 1.0, 1.0);
        s.f64("b", 1.0, 1.0 + 1e-15); // within 1e-14
        assert_eq!((s.checked, s.bit_exact), (2, 1));
        // Pin the actual tracked deviation, not just `> 0`: max_rel = |1.0 - (1.0+1e-15)| / scale,
        // a deterministic IEEE-754 value. A bare `> 0` passes even if max_rel were computed wrongly,
        // silently breaking the calibration summary that every replay test relies on.
        assert_eq!(s.max_rel, 1.110_223_024_625_155_4e-15);
    }

    #[test]
    #[should_panic(expected = "rel")]
    fn diff_stats_fail_beyond_the_tolerance() {
        DiffStats::default().f64("x", 1.0, 1.001);
    }
}
