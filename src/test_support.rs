//! Test-only support: the "bobyqa state v1" parser, bit-exact diff assertions,
//! and the Rust problem registry for replaying captured states.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::string::{String, ToString};
use std::vec::Vec;
use std::{format, vec};

use crate::mat::Mat;

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

    /// PRIMA's 1-based IJ(2, m) -> 0-based (i, j) pairs (sentinel/index translation).
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
// Diff assertions: every f64 output must equal the captured PRIMA value bit for bit.
// ---------------------------------------------------------------------------

pub(crate) fn assert_bits(what: &str, got: f64, want: f64) {
    assert!(
        got.to_bits() == want.to_bits(),
        "{what}: got {got:e}, want {want:e}"
    );
}

pub(crate) fn assert_slice_bits(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert_bits(&format!("{what}[{k}]"), *g, *w);
    }
}

pub(crate) fn assert_mat_bits(what: &str, got: &Mat, want: &Mat) {
    assert_eq!(
        (got.nrows(), got.ncols()),
        (want.nrows(), want.ncols()),
        "{what}: shape"
    );
    assert_slice_bits(what, got.data(), want.data());
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
    #[should_panic(expected = "want")]
    fn assert_bits_fails_on_a_one_ulp_difference() {
        assert_bits("x", 1.0, 1.0 + f64::EPSILON);
    }
}
