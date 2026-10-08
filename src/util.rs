//! `xinbd.f90`, `univar.f90::interval_max`, `checkexit.f90` (unconstrained), and
//! `evaluate.f90::{evaluate, moderatex, moderatef}` — PRIMA common modules.
//!
//! Index convention: all indices 0-based.
use crate::consts::{
    FTARGET_ACHIEVED, FUNCMAX, INFO_DFT, MAXFUN_REACHED, NAN_INF_F, NAN_INF_X, REALMAX,
};
use crate::math;
#[cfg(test)]
use alloc::vec;

/// Fallible `vec![value; len]`: reserves via `try_reserve_exact` so an out-of-memory
/// (or byte-capacity-overflow) request surfaces as an `Err` instead of aborting the
/// process. Construction-time only — the warm path never allocates.
pub(crate) fn try_vec<T: Clone>(
    value: T,
    len: usize,
) -> Result<alloc::vec::Vec<T>, alloc::collections::TryReserveError> {
    let mut v = alloc::vec::Vec::new();
    v.try_reserve_exact(len)?;
    v.resize(len, value);
    Ok(v)
}

/// A `Vec` sized by capacity alone and filled later (`ij`, the restart boundary store), built by
/// [`try_capacity`]. Its `Clone` carries the capacity over: `Vec::clone` keeps only `len` of it,
/// which would leave a cloned solver to allocate on its first push inside `minimize`.
#[derive(Debug)]
pub(crate) struct Reserved<T>(alloc::vec::Vec<T>);

impl<T: Clone> Clone for Reserved<T> {
    fn clone(&self) -> Self {
        let mut v = alloc::vec::Vec::with_capacity(self.0.capacity());
        v.extend_from_slice(&self.0);
        Self(v)
    }
}

impl<T> core::ops::Deref for Reserved<T> {
    type Target = alloc::vec::Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> core::ops::DerefMut for Reserved<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Fallible `Vec::with_capacity(cap)` — same fallible-allocation contract as [`try_vec`], for the
/// capacity-only stores that are filled later.
pub(crate) fn try_capacity<T>(
    cap: usize,
) -> Result<Reserved<T>, alloc::collections::TryReserveError> {
    let mut v = alloc::vec::Vec::new();
    v.try_reserve_exact(cap)?;
    Ok(Reserved(v))
}

/// PRIMA evaluate.f90 L27 `moderatex`, one element: NaN -> 0, then clamp to
/// [-REALMAX, REALMAX]. `moderatex` is elementwise, so the scalar form composes exactly.
pub(crate) fn moderatex1(v: f64) -> f64 {
    let v = if v.is_nan() { 0.0 } else { v };
    (-REALMAX).max(REALMAX.min(v))
}

/// PRIMA evaluate.f90 L27 `moderatex`: NaN -> 0, then clamp to [-REALMAX, REALMAX].
/// Writes the moderated copy into `y` (full overwrite).
pub(crate) fn moderatex_into(x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = moderatex1(xi);
    }
}

/// PRIMA evaluate.f90 L47 `moderatef`: NaN -> FUNCMAX, then clamp to [-REALMAX, FUNCMAX].
pub(crate) fn moderatef(f: f64) -> f64 {
    let mut y = f;
    if y.is_nan() {
        y = FUNCMAX;
    }
    (-REALMAX).max(FUNCMAX.min(y))
}

/// PRIMA evaluate.f90 L93 `evaluatef`: the moderated-extreme-barrier evaluation. `xmod` is
/// n-length scratch for the moderated copy of `x` handed to the objective (the Fortran's
/// per-call MODERATEX result, hoisted to the solver workspace).
pub(crate) fn evaluate<F: FnMut(&[f64]) -> f64>(
    calfun: &mut F,
    x: &[f64],
    xmod: &mut [f64],
) -> f64 {
    if x.iter().any(|v| v.is_nan()) {
        // PRIMA evaluate.f90 L126: defensive branch — f = sum(x) propagates the NaN.
        x.iter().sum()
    } else {
        moderatex_into(x, xmod);
        moderatef(calfun(xmod))
    }
}

/// PRIMA preproc.f90 L341-350 (`HONOUR_X0 = FALSE`): revise X0 so its distance to each inactive
/// bound is 0 or >= `rhobeg`. Requires `xu - xl >= 2 * rhobeg` and `x` inside the box.
pub(crate) fn revise_x0(x: &mut [f64], xl: &[f64], xu: &[f64], rhobeg: f64) {
    for i in 0..x.len() {
        if x[i] <= xl[i] + 0.5 * rhobeg {
            x[i] = xl[i];
        } else if x[i] < xl[i] + rhobeg {
            x[i] = xl[i] + rhobeg;
        }
    }
    for i in 0..x.len() {
        if x[i] >= xu[i] - 0.5 * rhobeg {
            x[i] = xu[i];
        } else if x[i] > xu[i] - rhobeg {
            x[i] = xu[i] - rhobeg;
        }
    }
}

/// PRIMA checkexit.f90 L25 `checkexit_unc`. Later assignments win, exactly as in the Fortran.
pub(crate) fn checkexit(maxfun: usize, nf: usize, f: f64, ftarget: f64, x: &[f64]) -> i32 {
    let mut info = INFO_DFT;
    if x.iter().any(|v| v.is_nan() || v.is_infinite()) {
        info = NAN_INF_X;
    }
    if f.is_nan() || (f.is_infinite() && f.is_sign_positive()) {
        info = NAN_INF_F;
    }
    if f <= ftarget {
        info = FTARGET_ACHIEVED;
    }
    if nf >= maxfun {
        info = MAXFUN_REACHED;
    }
    info
}

/// PRIMA xinbd.f90 L12: X = XBASE + STEP projected so bound-hitting steps land exactly on
/// bounds. Writes into `x` (length n, fully overwritten).
pub(crate) fn xinbd_into(
    xbase: &[f64],
    step: &[f64],
    xl: &[f64],
    xu: &[f64],
    sl: &[f64],
    su: &[f64],
    x: &mut [f64],
) {
    let n = xbase.len();
    // PRIMA xinbd.f90 L61-64: s = max(sl, min(su, step)); x = max(xl, min(xu, xbase + s));
    // x(trueloc(s <= sl)) = xl(...); x(trueloc(s >= su)) = xu(...).
    for i in 0..n {
        let s = sl[i].max(su[i].min(step[i]));
        x[i] = xl[i].max(xu[i].min(xbase[i] + s));
        if s <= sl[i] {
            x[i] = xl[i];
        }
        if s >= su[i] {
            x[i] = xu[i];
        }
    }
}

/// PRIMA linalg.f90 `linspace_r` — `interval_max`'s internal grid builder.
/// Writes the grid into `x[..n]` (fully overwritten).
fn linspace_into(xstart: f64, xstop: f64, n: usize, x: &mut [f64]) {
    debug_assert_eq!(x.len(), n);
    if n == 0 {
        return;
    }
    let nm = n - 1;
    if n == 1 || (xstart <= xstop && xstop <= xstart) {
        // PRIMA: N == 1 or XSTART == XSTOP (expressed via two <= to dodge float-equality lints).
        x.fill(xstop);
    } else if math::abs(xstart) <= math::abs(xstop) && math::abs(xstop) <= math::abs(xstart) {
        // PRIMA: the symmetric case XSTOP == -XSTART: x = (xstop/nm) * [-nm, -nm+2, ..., nm],
        // with the exact midpoint zeroed when nm is even.
        let xunit = xstop / nm as f64;
        for (idx, v) in x.iter_mut().enumerate() {
            // The Fortran index runs i = -nm, -nm+2, ..., nm; as f64 it is exact.
            *v = xunit * (2.0 * idx as f64 - nm as f64);
        }
        if nm % 2 == 0 {
            x[nm / 2] = 0.0;
        }
    } else {
        let xunit = (xstop - xstart) / nm as f64;
        for (i, v) in x.iter_mut().enumerate() {
            *v = xstart + xunit * i as f64;
        }
    }
    // PRIMA: pin the endpoints exactly.
    x[0] = xstart;
    x[n - 1] = xstop;
}

/// PRIMA univar.f90 L211 `interval_max`: approximate maximizer of `fun(x, args)` on [lb, ub] —
/// a `grid_size`-point grid search refined by one quadratic-interpolation step, with PRIMA's
/// tie-breaking (first masked maximum) and NaN handling exactly as written. `xgrid`/`fgrid`
/// are `grid_size`-length scratch for the grid and its objective values.
pub(crate) fn interval_max<F: Fn(f64, &[f64]) -> f64>(
    fun: F,
    lb: f64,
    ub: f64,
    args: &[f64],
    grid_size: usize,
    xgrid: &mut [f64],
    fgrid: &mut [f64],
) -> f64 {
    if ub <= lb {
        return lb;
    }
    linspace_into(lb, ub, grid_size, xgrid);
    for (fg, &xg) in fgrid.iter_mut().zip(xgrid.iter()) {
        *fg = fun(xg, args);
    }

    if fgrid.iter().all(|f| f.is_nan()) {
        return lb;
    }

    // PRIMA: kopt = maxloc(fgrid, mask=(.not. is_nan(fgrid)), dim=1) — first masked maximum.
    let mut kopt = 0;
    let mut fopt = f64::NAN;
    let mut found = false;
    for (k, &f) in fgrid.iter().enumerate() {
        if !f.is_nan() && (!found || f > fopt) {
            kopt = k;
            fopt = f;
            found = true;
        }
    }

    if kopt == 0 {
        lb
    } else if kopt == grid_size - 1 {
        ub
    } else {
        let fprev = fgrid[kopt - 1];
        let fnext = fgrid[kopt + 1];
        let mut step = 0.0;
        if math::abs(fprev - fnext) > 0.0 {
            step = 0.5 * ((fnext - fprev) / (fopt + fopt - fprev - fnext));
        }
        if step.is_finite() && math::abs(step) > 0.0 {
            // PRIMA: x = lb + (ub-lb)*(real(kopt-1) + step)/real(grid_size-1) — the 1-based
            // kopt-1 is our 0-based kopt.
            lb + (ub - lb) * (kopt as f64 + step) / (grid_size - 1) as f64
        } else {
            xgrid[kopt]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::{
        FTARGET_ACHIEVED, FUNCMAX, INFO_DFT, MAXFUN_REACHED, NAN_INF_F, NAN_INF_X,
    };

    #[test]
    fn moderatef_caps_nan_and_huge_values() {
        assert_eq!(moderatef(f64::NAN), FUNCMAX);
        assert_eq!(moderatef(f64::INFINITY), FUNCMAX);
        assert_eq!(moderatef(1.0e40), FUNCMAX);
        assert_eq!(moderatef(-f64::INFINITY), -f64::MAX);
        assert_eq!(moderatef(1.5), 1.5);
    }

    #[test]
    fn moderatex_replaces_nan_and_clamps_to_realmax() {
        let mut y = [1.0; 3];
        moderatex_into(&[f64::NAN, f64::INFINITY, 2.0], &mut y);
        assert_eq!(y, [0.0, f64::MAX, 2.0]);
    }

    #[test]
    fn evaluate_moderates_input_and_output() {
        let mut xmod = [0.0; 2];
        let mut nan_objective = |_: &[f64]| f64::NAN;
        assert_eq!(
            evaluate(&mut nan_objective, &[1.0], &mut xmod[..1]),
            FUNCMAX
        );
        let mut sum = |x: &[f64]| x.iter().sum::<f64>();
        assert_eq!(evaluate(&mut sum, &[1.0, 2.0], &mut xmod), 3.0);
        assert!(evaluate(&mut sum, &[f64::NAN], &mut xmod[..1]).is_nan());
        // Infinite inputs reach the objective clamped to ±REALMAX.
        let mut seen = [0.0; 2];
        let mut capture = |x: &[f64]| {
            seen.copy_from_slice(x);
            0.0
        };
        evaluate(&mut capture, &[f64::INFINITY, f64::NEG_INFINITY], &mut xmod);
        assert_eq!(seen, [f64::MAX, -f64::MAX]);
    }

    #[test]
    fn checkexit_reports_the_prima_exit_codes_with_prima_precedence() {
        assert_eq!(checkexit(100, 5, 1.0, -f64::INFINITY, &[0.0]), INFO_DFT);
        assert_eq!(checkexit(100, 5, 1.0, 2.0, &[0.0]), FTARGET_ACHIEVED);
        assert_eq!(
            checkexit(100, 100, 1.0, -f64::INFINITY, &[0.0]),
            MAXFUN_REACHED
        );
        assert_eq!(
            checkexit(100, 5, 1.0, -f64::INFINITY, &[f64::INFINITY]),
            NAN_INF_X
        );
        assert_eq!(
            checkexit(100, 5, f64::NAN, -f64::INFINITY, &[0.0]),
            NAN_INF_F
        );
        // maxfun beats ftarget beats nan-f, as in checkexit.f90's assignment order:
        assert_eq!(checkexit(5, 5, 1.0, 2.0, &[0.0]), MAXFUN_REACHED);
    }

    #[test]
    fn xinbd_pins_x_to_the_exact_bound_when_the_step_hits_it() {
        let xbase = [0.5, 0.5];
        let (xl, xu) = ([0.0, 0.0], [1.0, 1.0]);
        let (sl, su) = ([-0.5, -0.5], [0.5, 0.5]);
        // step beyond su -> x lands exactly on xu (no rounding residue)
        let mut x = [f64::NAN; 2];
        xinbd_into(&xbase, &[0.7, 0.0], &xl, &xu, &sl, &su, &mut x);
        assert_eq!(x, [1.0, 0.5]);
        xinbd_into(&xbase, &[-0.6, 0.2], &xl, &xu, &sl, &su, &mut x);
        assert_eq!(x, [0.0, 0.7]);
        // xbase + su rounds below xu (0.7 + 0.1 is 0.7999999999999999 in f64), so only the
        // explicit `s >= su` snap puts x on the bound.
        let mut x1 = [f64::NAN];
        xinbd_into(&[0.7], &[0.5], &[0.0], &[0.8], &[-0.7], &[0.1], &mut x1);
        assert_eq!(x1, [0.8]);
    }

    #[test]
    fn interval_max_finds_the_grid_maximum() {
        // f(x) = x*(1-x) on [0, 1]: max at 0.5, between two grid points. The interpolation step
        // is exact on a quadratic, so the result is 0.5 to rounding, not merely the nearest
        // grid point (0.0102 away).
        let f = |x: f64, _: &[f64]| x * (1.0 - x);
        let (mut xgrid, mut fgrid) = (vec![0.0; 50], vec![0.0; 50]);
        let x = interval_max(f, 0.0, 1.0, &[], 50, &mut xgrid, &mut fgrid);
        assert!((x - 0.5).abs() <= 1e-12);
    }

    #[test]
    fn interval_max_returns_an_endpoint_on_its_degenerate_branches() {
        let (mut xgrid, mut fgrid) = (vec![0.0; 50], vec![0.0; 50]);
        let mut run = |f: fn(f64, &[f64]) -> f64, lb: f64, ub: f64| {
            interval_max(f, lb, ub, &[], 50, &mut xgrid, &mut fgrid)
        };
        assert_eq!(run(|x, _| x, 2.0, 1.0), 2.0); // ub <= lb -> lb
        assert_eq!(run(|_, _| f64::NAN, 0.25, 1.0), 0.25); // all-NaN grid -> lb
        assert_eq!(run(|x, _| -x, 0.25, 1.0), 0.25); // maximum at the first grid point -> lb
        assert_eq!(run(|x, _| x, 0.0, 1.0), 1.0); // maximum at the last grid point -> ub
    }
}
