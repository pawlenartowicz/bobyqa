//! Minimal column-major matrix (design §3.2) — infrastructure, not a PRIMA port.
//!
//! A PRIMA array `A(nr, nc)` becomes a `Mat` with `a[[i, j]] = data[i + j*nr]`, 0-based; same
//! memory order as Fortran, so index translation is mechanical and `A(:, k)` column slices are
//! contiguous. No math methods: arithmetic stays in explicit loops that mirror PRIMA.
use core::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub(crate) struct Mat {
    data: Vec<f64>,
    nrows: usize,
}

impl Mat {
    pub(crate) fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![0.0; nrows * ncols],
            nrows,
        }
    }

    #[allow(dead_code)] // §3.4 audited constructor; used by test_support.rs parser and test modules (cfg(test) only)
    pub(crate) fn from_col_major(nrows: usize, ncols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), nrows * ncols, "shape/data mismatch");
        Self { data, nrows }
    }

    pub(crate) fn nrows(&self) -> usize {
        self.nrows
    }

    pub(crate) fn ncols(&self) -> usize {
        self.data.len().checked_div(self.nrows).unwrap_or(0)
    }

    pub(crate) fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.nrows..(j + 1) * self.nrows]
    }

    pub(crate) fn col_mut(&mut self, j: usize) -> &mut [f64] {
        &mut self.data[j * self.nrows..(j + 1) * self.nrows]
    }

    pub(crate) fn fill(&mut self, v: f64) {
        self.data.fill(v);
    }

    /// Disjoint mutable views of columns `a` and `b`, `a < b` — the two-column Givens-rotation
    /// update in `updateh` (slice access instead of per-element `[[i, j]]`).
    pub(crate) fn two_cols_mut(&mut self, a: usize, b: usize) -> (&mut [f64], &mut [f64]) {
        debug_assert!(a < b, "two_cols_mut needs a < b");
        let nr = self.nrows;
        let (lo, hi) = self.data.split_at_mut(b * nr);
        (&mut lo[a * nr..(a + 1) * nr], &mut hi[..nr])
    }

    /// Overwrites `self` with `src` (same shape) — workspace refill for the M2 reuse design.
    pub(crate) fn copy_from(&mut self, src: &Mat) {
        debug_assert_eq!(self.nrows, src.nrows, "copy_from: row mismatch");
        self.data.copy_from_slice(&src.data);
    }

    /// Flat column-major view — used by the diff tests to compare whole matrices.
    pub(crate) fn data(&self) -> &[f64] {
        &self.data
    }
}

impl Index<[usize; 2]> for Mat {
    type Output = f64;
    fn index(&self, [i, j]: [usize; 2]) -> &f64 {
        debug_assert!(i < self.nrows, "row {i} out of {}", self.nrows);
        &self.data[i + j * self.nrows]
    }
}

impl IndexMut<[usize; 2]> for Mat {
    fn index_mut(&mut self, [i, j]: [usize; 2]) -> &mut f64 {
        debug_assert!(i < self.nrows, "row {i} out of {}", self.nrows);
        &mut self.data[i + j * self.nrows]
    }
}

#[cfg(test)]
mod tests {
    use super::Mat;

    #[test]
    fn storage_is_column_major_like_fortran() {
        // A(nr=2, nc=3) laid out as [a11 a21 | a12 a22 | a13 a23].
        let m = Mat::from_col_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[1, 0]], 2.0);
        assert_eq!(m[[0, 1]], 3.0);
        assert_eq!(m[[1, 2]], 6.0);
    }

    #[test]
    fn columns_are_contiguous_slices() {
        let m = Mat::from_col_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.col(1), &[3.0, 4.0]);
    }

    #[test]
    fn zeros_has_the_requested_shape() {
        let mut m = Mat::zeros(3, 2);
        assert_eq!((m.nrows(), m.ncols()), (3, 2));
        m[[2, 1]] = 7.0;
        assert_eq!(m.col(1), &[0.0, 0.0, 7.0]);
        m.col_mut(0)[1] = 5.0;
        assert_eq!(m[[1, 0]], 5.0);
    }
}
