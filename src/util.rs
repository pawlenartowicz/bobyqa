//! `xinbd.f90`, `univar.f90::interval_max`, `checkexit.f90` (unconstrained), and
//! `evaluate.f90::{evaluate, moderatex, moderatef}` — PRIMA common modules.
//!
//! Index convention: all indices 0-based.
use crate::consts::{
    FTARGET_ACHIEVED, FUNCMAX, INFO_DFT, MAXFUN_REACHED, NAN_INF_F, NAN_INF_X, REALMAX,
};
use crate::math;

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

/// Allocating form of [`moderatex_into`] — thin wrapper, kept for tests; hot paths use `_into`.
#[cfg(test)]
pub(crate) fn moderatex(x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    moderatex_into(x, &mut y);
    y
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

/// Allocating form of [`xinbd_into`] — thin wrapper, kept for tests; hot paths use `_into`.
#[cfg(test)]
pub(crate) fn xinbd(
    xbase: &[f64],
    step: &[f64],
    xl: &[f64],
    xu: &[f64],
    sl: &[f64],
    su: &[f64],
) -> Vec<f64> {
    let mut x = vec![0.0; xbase.len()];
    xinbd_into(xbase, step, xl, xu, sl, su, &mut x);
    x
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
        assert_eq!(
            moderatex(&[f64::NAN, f64::INFINITY, 2.0]),
            vec![0.0, f64::MAX, 2.0]
        );
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
        let x = xinbd(&xbase, &[0.7, 0.0], &xl, &xu, &sl, &su);
        assert_eq!(x, vec![1.0, 0.5]);
        let x = xinbd(&xbase, &[-0.6, 0.2], &xl, &xu, &sl, &su);
        assert_eq!(x, vec![0.0, 0.7]);
    }

    #[test]
    fn interval_max_finds_the_grid_maximum() {
        // f(x) = x*(1-x) on [0, 1]: max at 0.5; grid_size 50 must land on it or beside it.
        let f = |x: f64, _: &[f64]| x * (1.0 - x);
        let (mut xgrid, mut fgrid) = (vec![0.0; 50], vec![0.0; 50]);
        let x = interval_max(f, 0.0, 1.0, &[], 50, &mut xgrid, &mut fgrid);
        assert!((x - 0.5).abs() <= 0.5 / 50.0 + 1e-12);
    }
}
