//! The `linalg.f90` subset BOBYQA uses (PRIMA common modules), transcribed loop-for-loop —
//! the "naive implementation to get full control" comments in `linalg.f90` are the point.
//!
//! Index convention: all indices 0-based.
//!
//! Call-site convention: Fortran *array sections* at call sites (`diag(xpt(:, 2:n+1))`,
//! `bmat(:, 1:npt)`) have no Rust equivalent — the call site writes the explicit indexed loop
//! with a citation; the helpers here take whole `Mat`s/slices.
use crate::consts::{REALMAX, SYMTOL};
use crate::mat::Mat;
use crate::math;

/// PRIMA linalg.f90 L465 `inprod`: z = x^T y, accumulated in element order.
pub(crate) fn inprod(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let mut z = 0.0;
    for i in 0..x.len() {
        z += x[i] * y[i];
    }
    z
}

/// PRIMA linalg.f90 L332 `matprod12`: row-vector x times matrix y; z(j) = inprod(x, y(:, j)).
/// Writes the full result into `z` (length `y.ncols()`).
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA (rust.md §5)
pub(crate) fn matprod12_into(x: &[f64], y: &Mat, z: &mut [f64]) {
    debug_assert_eq!(x.len(), y.nrows());
    debug_assert_eq!(z.len(), y.ncols());
    for j in 0..y.ncols() {
        z[j] = inprod(x, y.col(j));
    }
}

/// Allocating form of [`matprod12_into`] — thin wrapper, kept for tests; hot paths use `_into`.
#[cfg(test)]
pub(crate) fn matprod12(x: &[f64], y: &Mat) -> Vec<f64> {
    let mut z = vec![0.0; y.ncols()];
    matprod12_into(x, y, &mut z);
    z
}

/// PRIMA linalg.f90 L377 `matprod21`: matrix x times column-vector y; z accumulates x(:, j)*y(j)
/// column by column (NOT row-by-row dot products — the loop order is part of the contract).
/// Zeroes `z` (length `x.nrows()`) first, like the fresh allocation it replaces.
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA (rust.md §5)
pub(crate) fn matprod21_into(x: &Mat, y: &[f64], z: &mut [f64]) {
    debug_assert_eq!(x.ncols(), y.len());
    debug_assert_eq!(z.len(), x.nrows());
    z.fill(0.0);
    for j in 0..x.ncols() {
        let xj = x.col(j);
        for i in 0..z.len() {
            z[i] += xj[i] * y[j];
        }
    }
}

/// Allocating form of [`matprod21_into`] — thin wrapper, kept for tests; hot paths use `_into`.
#[cfg(test)]
pub(crate) fn matprod21(x: &Mat, y: &[f64]) -> Vec<f64> {
    let mut z = vec![0.0; x.nrows()];
    matprod21_into(x, y, &mut z);
    z
}

/// PRIMA linalg.f90 L420 `matprod22`: z(:, j) accumulates x(:, i)*y(i, j) — j outer, i inner.
/// Zeroes `z` (shape `x.nrows()` × `y.ncols()`) first, like the fresh allocation it replaces.
pub(crate) fn matprod22_into(x: &Mat, y: &Mat, z: &mut Mat) {
    debug_assert_eq!(x.ncols(), y.nrows());
    debug_assert_eq!((z.nrows(), z.ncols()), (x.nrows(), y.ncols()));
    z.fill(0.0);
    for j in 0..y.ncols() {
        for i in 0..x.ncols() {
            let xi = x.col(i);
            let yij = y[[i, j]];
            let zj = z.col_mut(j);
            for r in 0..zj.len() {
                zj[r] += xi[r] * yij;
            }
        }
    }
}

/// Allocating form of [`matprod22_into`] — thin wrapper, kept for tests; hot paths use `_into`.
#[cfg(test)]
pub(crate) fn matprod22(x: &Mat, y: &Mat) -> Mat {
    let mut z = Mat::zeros(x.nrows(), y.ncols());
    matprod22_into(x, y, &mut z);
    z
}

/// PRIMA linalg.f90 L502 `outprod`: z = x*y^T, filled column by column (full overwrite).
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA (rust.md §5)
pub(crate) fn outprod_into(x: &[f64], y: &[f64], z: &mut Mat) {
    debug_assert_eq!((z.nrows(), z.ncols()), (x.len(), y.len()));
    for i in 0..y.len() {
        let zi = z.col_mut(i);
        for r in 0..x.len() {
            zi[r] = x[r] * y[i];
        }
    }
}

/// Allocating form of [`outprod_into`] — thin wrapper, kept for tests; hot paths use `_into`.
#[cfg(test)]
pub(crate) fn outprod(x: &[f64], y: &[f64]) -> Mat {
    let mut z = Mat::zeros(x.len(), y.len());
    outprod_into(x, y, &mut z);
    z
}

/// PRIMA linalg.f90 L1862 `p_norm`, p = 2 branch only (BOBYQA only calls `norm(x)`): the
/// empty/non-finite/zero guards + naive sqrt(sum(x**2)) + the radix-scaling overflow rescue.
pub(crate) fn norm(x: &[f64]) -> f64 {
    // PRIMA p_norm: maxabs = maxval([abs(x), ZERO]).
    let mut maxabs = 0.0_f64;
    for &v in x {
        maxabs = maxabs.max(math::abs(v));
    }
    if x.is_empty() {
        0.0
    } else if !x.iter().all(|v| v.is_finite()) {
        // PRIMA p_norm: if X contains NaN, Y is NaN; otherwise Y is Inf when X contains +/-Inf.
        let mut y = 0.0;
        for &v in x {
            y += math::abs(v);
        }
        y
    } else if maxabs <= 0.0 {
        0.0
    } else {
        // Naive implementation for full control (PRIMA's own comment), then the rescue.
        let mut sumsq = 0.0;
        for &v in x {
            sumsq += v * v;
        }
        let mut y = math::sqrt(sumsq);
        if (y.is_infinite() && y.is_sign_positive()) || y <= 0.0 {
            // PRIMA p_norm overflow/underflow rescue. Fortran's minexponent/maxexponent equal
            // Rust's MIN_EXP/MAX_EXP. The min/max mirror the Fortran formula (no-ops for f64,
            // meaningful for other precisions PRIMA supports).
            #[expect(clippy::unnecessary_min_or_max)]
            let scalmin = f64::from(f64::RADIX).powi((f64::MIN_EXP - 1).max(1 - f64::MAX_EXP));
            #[expect(clippy::unnecessary_min_or_max)]
            let scalmax = f64::from(f64::RADIX).powi((f64::MAX_EXP - 1).min(1 - f64::MIN_EXP));
            let scaling = scalmax.min(scalmin.max(maxabs));
            let mut scaled_sumsq = 0.0;
            for &v in x {
                let s = v / scaling;
                scaled_sumsq += s * s;
            }
            y = scaling * math::sqrt(scaled_sumsq);
        }
        y
    }
}

/// PRIMA linalg.f90 L1129 `diag`, k = 0 only: the main diagonal.
#[allow(dead_code)] // §3.4 audited helper; call sites in initialize.rs use indexed loops per the call-site convention
pub(crate) fn diag(a: &Mat) -> Vec<f64> {
    let dlen = a.nrows().min(a.ncols());
    let mut d = vec![0.0; dlen];
    for i in 0..dlen {
        d[i] = a[[i, i]];
    }
    d
}

/// PRIMA linalg.f90 L1798 `issymmetric`, tol = `consts::SYMTOL` (debug checks).
#[allow(dead_code)] // §3.4 audited debug helper; used in linalg tests and future debug_assert sites
pub(crate) fn issymmetric(a: &Mat) -> bool {
    if a.nrows() != a.ncols() {
        return false;
    }
    // PRIMA: the check runs only when SYMTOL_DFT < 0.9*REALMAX — always true for our SYMTOL.
    let mut maxabs = 0.0_f64;
    for j in 0..a.ncols() {
        for &v in a.col(j) {
            maxabs = maxabs.max(math::abs(v));
        }
    }
    let tol = SYMTOL * maxabs.max(1.0);
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            // PRIMA: .not. any(|A - A^T| > tol) — NaN compares false, exactly as in Fortran.
            if math::abs(a[[i, j]] - a[[j, i]]) > tol {
                return false;
            }
            // PRIMA: all(is_nan(A) .eqv. is_nan(A^T)).
            if a[[i, j]].is_nan() != a[[j, i]].is_nan() {
                return false;
            }
        }
    }
    true
}

/// PRIMA linalg.f90 L2224 `trueloc` — returns **0-based** positions of `true`, ascending.
#[allow(dead_code)] // §3.4 audited helper; call sites in geometry/rescue use inline indexed loops per the call-site convention
pub(crate) fn trueloc(x: &[bool]) -> Vec<usize> {
    let mut loc = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        if v {
            loc.push(i);
        }
    }
    loc
}

/// PRIMA linalg.f90 L1568 `planerot`: a 2x2 Givens matrix G with G*x = [r; 0], all 0/NaN/inf
/// edge cases transcribed; returns `[[c, s], [-s, c]]` row-major.
///
/// Fortran `sign(a, b)` (|a| with b's sign, IEEE sign-bit semantics in gfortran) transcribes to
/// `a.copysign(b)` for non-negative `a`.
#[expect(clippy::many_single_char_names)] // c/s/t/u/r are PRIMA's identifiers (rust.md §5)
pub(crate) fn planerot(x: [f64; 2]) -> [[f64; 2]; 2] {
    let eps = f64::EPSILON;
    let (c, s);
    if x[0].is_nan() || x[1].is_nan() {
        // PRIMA: MATLAB would return NaN(2,2); PRIMA keeps G orthogonal.
        c = 1.0;
        s = 0.0;
    } else if x[0].is_infinite() && x[1].is_infinite() {
        c = (1.0 / math::sqrt(2.0)).copysign(x[0]);
        s = (1.0 / math::sqrt(2.0)).copysign(x[1]);
    } else if math::abs(x[0]) <= 0.0 && math::abs(x[1]) <= 0.0 {
        // X(1) == 0 == X(2).
        c = 1.0;
        s = 0.0;
    } else if math::abs(x[1]) <= eps * math::abs(x[0]) {
        c = 1.0_f64.copysign(x[0]);
        s = 0.0;
    } else if math::abs(x[0]) <= eps * math::abs(x[1]) {
        c = 0.0;
        s = 1.0_f64.copysign(x[1]);
    } else {
        // PRIMA linalg.f90 L1635-1639: the normal case — a stable & continuous Givens rotation
        // (Bindel, Demmel, Kahan & Marques, 2002). lo/hi are the L1639 guard thresholds bracketing
        // the magnitude range where the direct r = norm(x) form cannot over/underflow.
        let lo = math::sqrt(f64::MIN_POSITIVE);
        let hi = math::sqrt(REALMAX / 2.1);
        if x.iter().all(|&v| math::abs(v) > lo && math::abs(v) < hi) {
            // PRIMA: the direct calculation works better when no over/underflow is possible.
            let r = norm(&x);
            c = x[0] / r;
            s = x[1] / r;
        } else if math::abs(x[0]) > math::abs(x[1]) {
            let t = x[1] / x[0];
            // PRIMA: u = maxval([ONE, abs(t), sqrt(ONE + t**2)]) — precaution against rounding.
            let u = 1.0_f64
                .max(math::abs(t))
                .max(math::sqrt(1.0 + t * t))
                .copysign(x[0]);
            c = 1.0 / u;
            s = t / u;
        } else {
            let t = x[0] / x[1];
            let u = 1.0_f64
                .max(math::abs(t))
                .max(math::sqrt(1.0 + t * t))
                .copysign(x[1]);
            c = t / u;
            s = 1.0 / u;
        }
    }
    // PRIMA: G = reshape([c, -s, s, c], [2, 2]) — i.e. rows [c, s] and [-s, c].
    [[c, s], [-s, c]]
}

/// PRIMA linalg.f90 L1681 `symmetrize`: copy `A(LOWER_TRI)` to `A(UPPER_TRI)`.
pub(crate) fn symmetrize(a: &mut Mat) {
    debug_assert_eq!(a.nrows(), a.ncols());
    for j in 0..a.nrows() {
        // PRIMA: A(1:j-1, j) = A(j, 1:j-1).
        for i in 0..j {
            a[[i, j]] = a[[j, i]];
        }
    }
}

/// PRIMA linalg.f90 L142 `r1_sym`: A += alpha*x*x^T, lower triangle then symmetrize.
pub(crate) fn r1update(a: &mut Mat, alpha: f64, x: &[f64]) {
    let n = x.len();
    debug_assert_eq!((a.nrows(), a.ncols()), (n, n));
    // PRIMA: do j = 1, n: A(j:n, j) = A(j:n, j) + alpha * x(j:n) * x(j).
    for j in 0..n {
        for i in j..n {
            a[[i, j]] += alpha * x[i] * x[j];
        }
    }
    symmetrize(a);
}

/// PRIMA linalg.f90 L235 `r2_sym`: A += alpha*(x*y^T + y*x^T), lower triangle then symmetrize.
pub(crate) fn r2update(a: &mut Mat, alpha: f64, x: &[f64], y: &[f64]) {
    let n = x.len();
    debug_assert_eq!(y.len(), n);
    debug_assert_eq!((a.nrows(), a.ncols()), (n, n));
    // PRIMA linalg.f90 L269: A(j:n, j) = A(j:n, j) + alpha * x(j:n) * y(j) + alpha * y(j:n) * x(j),
    // evaluated LEFT-ASSOCIATIVELY: (A + t1) + t2, never A + (t1 + t2). The grouping is a 1-ulp
    // parity matter — `+=` would sum the cross-terms first.
    for j in 0..n {
        for i in j..n {
            a[[i, j]] = (a[[i, j]] + alpha * x[i] * y[j]) + alpha * y[i] * x[j];
        }
    }
    symmetrize(a);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;

    #[test]
    fn inprod_is_the_dot_product_in_order() {
        assert_eq!(inprod(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn matprod21_multiplies_matrix_by_column() {
        // [[1,3],[2,4]] * [5,6]^T = [23, 34]
        let a = Mat::from_col_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matprod21(&a, &[5.0, 6.0]), vec![23.0, 34.0]);
    }

    #[test]
    fn matprod12_multiplies_row_by_matrix() {
        let a = Mat::from_col_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matprod12(&[5.0, 6.0], &a), vec![17.0, 39.0]);
    }

    #[test]
    fn matprod22_multiplies_two_matrices() {
        let a = Mat::from_col_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Mat::from_col_major(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        // a*b = [[23,31],[34,46]] column-major [23,34,31,46]
        assert_eq!(matprod22(&a, &b).data(), &[23.0, 34.0, 31.0, 46.0]);
    }

    #[test]
    fn outprod_builds_x_y_transposed() {
        let m = outprod(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(m.data(), &[3.0, 6.0, 4.0, 8.0]);
    }

    #[test]
    fn norm_is_the_euclidean_norm_with_overflow_rescue() {
        assert_eq!(norm(&[3.0, 4.0]), 5.0);
        assert_eq!(norm(&[]), 0.0);
        assert_eq!(norm(&[0.0, 0.0]), 0.0);
        let big = norm(&[1.0e300, 1.0e300]); // naive sum of squares overflows
        assert!((big / 1.0e300 - core::f64::consts::SQRT_2).abs() < 1e-15);
        // Asymmetric overflow rescue: the all-equal case above makes every scaled component 1, so the
        // rescaled `s * s` is indistinguishable from `s / s`; distinct magnitudes pin that multiply.
        // Independent: sqrt((2e300)^2 + (1e300)^2) = sqrt(5) * 1e300.
        assert!((norm(&[2.0e300, 1.0e300]) / 1.0e300 - 5.0_f64.sqrt()).abs() < 1e-14);
        // Non-finite branch returns the L1 sum of |components| (+Inf here): pin the value and sign,
        // not just is_infinite(), so a `+=`->`-=` mutation (which yields -Inf) is caught.
        assert_eq!(norm(&[f64::INFINITY, 1.0]), f64::INFINITY);
        assert!(norm(&[f64::NAN, 1.0]).is_nan());
    }

    #[test]
    fn diag_extracts_the_main_diagonal() {
        let a = Mat::from_col_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(diag(&a), vec![1.0, 4.0]);
    }

    #[test]
    fn trueloc_returns_zero_based_positions() {
        assert_eq!(trueloc(&[false, true, false, true]), vec![1, 3]);
    }

    #[test]
    fn planerot_zeroes_the_second_component() {
        let g = planerot([3.0, 4.0]);
        let y2 = g[1][0] * 3.0 + g[1][1] * 4.0;
        let y1 = g[0][0] * 3.0 + g[0][1] * 4.0;
        assert!(y2.abs() < 1e-15);
        assert!((y1 - 5.0).abs() < 1e-15);
    }

    #[test]
    fn planerot_handles_the_nan_inf_zero_and_near_axis_branches() {
        // The norm-based normal case (tested above) never reaches PRIMA's five edge branches; pin the
        // exact returned `[[c, s], [-s, c]]` for each, including sign (copysign), which the impl tracks.
        let r = 0.707_106_781_186_547_5_f64; // the impl's `1.0 / sqrt(2.0)` — NOT FRAC_1_SQRT_2 (...476)
        // NaN in either component -> identity rotation (PRIMA keeps G orthogonal).
        assert_eq!(planerot([f64::NAN, 1.0]), [[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(planerot([1.0, f64::NAN]), [[1.0, 0.0], [0.0, 1.0]]);
        // Both infinite -> copysign(1/sqrt2) per component; sign follows each input.
        assert_eq!(planerot([f64::INFINITY, f64::INFINITY]), [[r, r], [-r, r]]);
        assert_eq!(
            planerot([f64::INFINITY, f64::NEG_INFINITY]),
            [[r, -r], [r, r]]
        );
        // Both zero -> identity.
        assert_eq!(planerot([0.0, 0.0]), [[1.0, 0.0], [0.0, 1.0]]);
        // |x1| <= eps*|x0| (y negligible) -> c = sign(x0), s = 0.
        assert_eq!(planerot([-2.0, 1e-300]), [[-1.0, 0.0], [0.0, -1.0]]);
        // |x0| <= eps*|x1| (x negligible) -> c = 0, s = sign(x1).
        assert_eq!(planerot([1e-300, -3.0]), [[0.0, -1.0], [1.0, 0.0]]);
        // Exactly ONE infinite component: the infinite one dominates -> near-axis branch (c=±1, s=0
        // or c=0, s=±1), NOT the both-infinite branch. Kills the `&&`->`||` at the both-inf guard.
        assert_eq!(planerot([f64::INFINITY, 1.0]), [[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(
            planerot([1.0, f64::NEG_INFINITY]),
            [[0.0, -1.0], [1.0, 0.0]]
        );
        // Overflow-safe Givens branches: components below lo = sqrt(MIN_POSITIVE) skip the direct
        // norm path and take the scaled t/u form. Pin the defining property (G rotates x to
        // [norm(x); 0] and is orthogonal) rather than fragile sub-ulp [c, s] literals — this still
        // kills the branch-selection and arithmetic mutations there (a wrong c/s breaks the rotation).
        // The ratio-0.5 pairs give u = sqrt(1 + t**2) > 1, so the t/u arithmetic is exercised
        // non-trivially (a ratio of 1 would make u = 1 and `/u` indistinguishable from `*u`); both
        // |x0| > |x1| and |x0| < |x1| orderings are covered.
        for x in [
            [1.0e-200, 1.0e-210],
            [1.0e-210, -1.0e-200],
            [1.0e-200, 5.0e-201],
            [5.0e-201, 1.0e-200],
        ] {
            let g = planerot(x);
            let y0 = g[0][0] * x[0] + g[0][1] * x[1];
            let y1 = g[1][0] * x[0] + g[1][1] * x[1];
            let nx = norm(&x);
            assert!((y0 - nx).abs() <= 1e-14 * nx, "G*x[0] != norm(x) for {x:?}");
            assert!(y1.abs() <= 1e-14 * nx, "G*x[1] != 0 for {x:?}");
            assert!(
                (g[0][0] * g[0][0] + g[0][1] * g[0][1] - 1.0).abs() < 1e-14,
                "G not orthogonal for {x:?}"
            );
        }
    }

    #[test]
    fn r1update_adds_alpha_x_x_transposed() {
        let mut a = Mat::zeros(2, 2);
        r1update(&mut a, 2.0, &[1.0, 3.0]);
        assert_eq!(a.data(), &[2.0, 6.0, 6.0, 18.0]);
    }

    #[test]
    fn r2update_adds_the_symmetric_cross_terms() {
        let mut a = Mat::zeros(2, 2);
        r2update(&mut a, 1.0, &[1.0, 0.0], &[0.0, 1.0]);
        assert_eq!(a.data(), &[0.0, 1.0, 1.0, 0.0]);
        // Non-degenerate: with x=[1,1], y=[1,2] both cross-terms x*y^T and y*x^T are nonzero in every
        // cell (a[i,j] = x[i]*y[j] + y[i]*x[j]): [[2,3],[3,4]]. The case above has x[i]*y[j] and
        // y[i]*x[j] never both nonzero, so dropping either term still passes; this case pins both.
        let mut b = Mat::zeros(2, 2);
        r2update(&mut b, 1.0, &[1.0, 1.0], &[1.0, 2.0]);
        assert_eq!(b.data(), &[2.0, 3.0, 3.0, 4.0]);
    }

    #[test]
    fn issymmetric_and_symmetrize_agree() {
        let mut a = Mat::from_col_major(2, 2, vec![1.0, 2.0, 2.0 + 1.0e-16, 4.0]);
        assert!(issymmetric(&a));
        a[[0, 1]] = 9.0;
        assert!(!issymmetric(&a));
        symmetrize(&mut a);
        assert!(issymmetric(&a));
    }
}
