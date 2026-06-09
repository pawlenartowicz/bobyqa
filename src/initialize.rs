//! `initialize.f90` (PRIMA bobyqa module): `initxf`, `initq`, `inith` — the `prelim` exemplar
//! port (design §5).
//!
//! Index convention: all indices 0-based (`kopt`, `k`, `ij` pairs); the diff tests translate
//! PRIMA's 1-based dump values. History (`xhist`/`fhist`) and `iprint`/`fmsg` are omitted
//! (SPEC §7.6); info codes are `consts.rs` values.
//!
//! Dimensions: `xpt` n×npt, `bmat` n×(npt+n), `zmat` npt×(npt−n−1), `hq` n×n,
//! `gopt`/`sl`/`su`/`xbase` n, `fval`/`pq` npt, `ij` len max(0, npt−2n−1).
use crate::consts::{INFO_DFT, REALMAX};
use crate::linalg::matprod21_into;
use crate::mat::Mat;
use crate::powalg::setij_into;
use crate::util::{checkexit, evaluate, xinbd_into};

/// Reused scratch for the initialize.f90 routines — PRIMA's per-call locals, hoisted to the
/// solver workspace. Lifetime and contents per call are identical to the Fortran
/// locals; only the allocation site moves: each field is re-initialized at the original
/// allocation site, per call. Field → Fortran-local map: `evaluated` initxf's EVALUATED,
/// `x` the per-evaluation XINBD result, `xmod` evaluate's MODERATEX copy, `xa`/`xb` the
/// XPT-diagonal sections of initq/inith, `shift` initq's L411 MATPROD(HQ, XPT(:, KOPT)).
#[derive(Debug, Clone)]
pub(crate) struct InitWs {
    evaluated: Vec<bool>, // npt
    x: Vec<f64>,          // n
    xmod: Vec<f64>,       // n
    xa: Vec<f64>,         // ndiag = min(n, npt - n - 1)
    xb: Vec<f64>,         // ndiag
    shift: Vec<f64>,      // n
}

impl InitWs {
    pub(crate) fn new(n: usize, npt: usize) -> Self {
        let ndiag = n.min(npt - n - 1);
        Self {
            evaluated: vec![false; npt],
            x: vec![0.0; n],
            xmod: vec![0.0; n],
            xa: vec![0.0; ndiag],
            xb: vec![0.0; ndiag],
            shift: vec![0.0; n],
        }
    }
}

/// PRIMA initialize.f90 L22 `initxf`: initialize the interpolation points and their function
/// values. Returns `(kopt 0-based, nf, info)`.
#[expect(clippy::too_many_arguments)] // out-params mirror the Fortran intent (rust.md §5)
pub(crate) fn initxf<F: FnMut(&[f64]) -> f64>(
    calfun: &mut F,
    maxfun: usize,
    ftarget: f64,
    rhobeg: f64,
    xl: &[f64],
    xu: &[f64],
    x0: &mut [f64],
    ij: &mut Vec<(usize, usize)>,
    fval: &mut [f64],
    sl: &mut [f64],
    su: &mut [f64],
    xbase: &mut [f64],
    xpt: &mut Mat,
    ws: &mut InitWs,
) -> (usize, usize, i32) {
    let n = xpt.nrows();
    let npt = xpt.ncols();

    let InitWs {
        evaluated, x, xmod, ..
    } = ws;

    // PRIMA initialize.f90 L127: info starts at the default.
    let mut info = INFO_DFT;

    // PRIMA L130-131: SL and SU are the bounds on feasible moves from X0.
    for i in 0..n {
        sl[i] = xl[i] - x0[i];
        su[i] = xu[i] - x0[i];
    }
    // PRIMA L136-149: the two WHERE blocks revising SL/SU/X0 (rounding guard; identity in
    // precise arithmetic) — two element loops in the same order, SL block first.
    for i in 0..n {
        if sl[i] < 0.0 {
            sl[i] = sl[i].min(-rhobeg);
        } else {
            x0[i] = xl[i];
            sl[i] = 0.0;
            su[i] = xu[i] - xl[i];
        }
    }
    for i in 0..n {
        if su[i] > 0.0 {
            su[i] = su[i].max(rhobeg);
        } else {
            x0[i] = xu[i];
            sl[i] = xl[i] - xu[i];
            su[i] = 0.0;
        }
    }

    // PRIMA L161: XBASE = X0.
    xbase.copy_from_slice(x0);

    // PRIMA L166: EVALUATED tracks which points have function values.
    evaluated.fill(false);

    // PRIMA L175: FVAL = REALMAX — unevaluated entries must compare bit-exact against the dump.
    fval.fill(REALMAX);

    // PRIMA L178-184: set XPT(:, 2 : N+1). 1-based column k+1 is 0-based k+1 with k from 0.
    xpt.fill(0.0);
    for k in 0..n {
        xpt[[k, k + 1]] = rhobeg;
        if su[k] <= 0.0 {
            // SU(K) == 0
            xpt[[k, k + 1]] = -rhobeg;
        }
    }
    // PRIMA L186-194: set XPT(:, N+2 : MIN(2*N+1, NPT)) — loop bound min(npt - n - 1, n).
    for k in 0..(npt - n - 1).min(n) {
        xpt[[k, k + n + 1]] = -rhobeg;
        if sl[k] >= 0.0 {
            // SL(K) == 0
            xpt[[k, k + n + 1]] = (2.0 * rhobeg).min(su[k]);
        }
        if su[k] <= 0.0 {
            // SU(K) == 0
            xpt[[k, k + n + 1]] = (-2.0 * rhobeg).max(sl[k]);
        }
    }

    // PRIMA L197-215: evaluate F at XPT(:, 1 : MIN(NPT, 2*N+1)). checkexit receives PRIMA's
    // 1-based k as nf (it compares against maxfun).
    for k in 0..npt.min(2 * n + 1) {
        xinbd_into(xbase, xpt.col(k), xl, xu, sl, su, x);
        let f = evaluate(calfun, x, xmod);
        evaluated[k] = true;
        fval[k] = f;
        let subinfo = checkexit(maxfun, k + 1, f, ftarget, x);
        if subinfo != INFO_DFT {
            info = subinfo;
            break;
        }
    }

    // PRIMA L228-234: for k = 2 .. min(npt - n, n + 1), switch XPT(:, K) <-> XPT(:, K+N) when
    // XPT(K-1, K) and XPT(K-1, K+N) have different signs and FVAL(K+N) < FVAL(K). 0-based: k in
    // 1 .. min(npt - n, n + 1) (exclusive), entries XPT[k-1, k] and XPT[k-1, k+n].
    for k in 1..(npt - n).min(n + 1) {
        if xpt[[k - 1, k]] * xpt[[k - 1, k + n]] < 0.0 && fval[k + n] < fval[k] {
            fval.swap(k, k + n);
            // Only entry k-1 is nonzero, but swap whole columns as PRIMA does.
            for r in 0..n {
                let tmp = xpt[[r, k]];
                xpt[[r, k]] = xpt[[r, k + n]];
                xpt[[r, k + n]] = tmp;
            }
        }
    }

    // PRIMA L241: set IJ (0-based pairs here).
    setij_into(n, npt, ij);

    // PRIMA L246: XPT(:, 2*N+2 : NPT) = XPT(:, IJ(1, :) + 1) + XPT(:, IJ(2, :) + 1). The +1 is
    // PRIMA's base-point offset (XPT(:, 1) is XBASE), NOT an index-base artifact: with 0-based
    // pairs the source columns are ij[k].0 + 1 and ij[k].1 + 1.
    for (k, &(i, j)) in ij.iter().enumerate() {
        for r in 0..n {
            xpt[[r, 2 * n + 1 + k]] = xpt[[r, i + 1]] + xpt[[r, j + 1]];
        }
    }

    // PRIMA L249-269: evaluate F at XPT(:, 2*N+2 : NPT), only if no abnormal exit so far.
    if info == INFO_DFT {
        for k in (2 * n + 1)..npt {
            xinbd_into(xbase, xpt.col(k), xl, xu, sl, su, x);
            let f = evaluate(calfun, x, xmod);
            evaluated[k] = true;
            fval[k] = f;
            let subinfo = checkexit(maxfun, k + 1, f, ftarget, x);
            if subinfo != INFO_DFT {
                info = subinfo;
                break;
            }
        }
    }

    // PRIMA L272-273: NF = COUNT(EVALUATED); KOPT = MINLOC(FVAL, MASK=EVALUATED) — the first
    // index of the minimum among evaluated points (strict < keeps "first").
    let nf = evaluated.iter().filter(|&&e| e).count();
    let mut kopt = 0;
    let mut found = false;
    for k in 0..npt {
        if evaluated[k] && (!found || fval[k] < fval[kopt]) {
            kopt = k;
            found = true;
        }
    }
    (kopt, nf, info)
}

/// PRIMA initialize.f90 L309 `initq`: initialize the quadratic model [GOPT, HQ, PQ]. Returns
/// `info` (NB: the `bobyqb.f90` call site on this pin omits the optional `info`, so the state
/// corpus has no `info` field to diff — the value is still computed, faithfully to the Fortran
/// body that runs under `present(info)`).
pub(crate) fn initq(
    ij: &[(usize, usize)],
    fval: &[f64],
    xpt: &Mat,
    gopt: &mut [f64],
    hq: &mut Mat,
    pq: &mut [f64],
    ws: &mut InitWs,
) -> i32 {
    let n = xpt.nrows();
    let npt = xpt.ncols();

    let InitWs { xa, xb, shift, .. } = ws;

    // PRIMA L373: FBASE is the function value at XBASE.
    let fbase = fval[0];

    // PRIMA L376: GOPT = (FVAL(2:N+1) - FBASE) / DIAG(XPT(:, 2:N+1)) — the array-section diag
    // becomes an indexed loop (Task 9 call-site convention): diag element i is XPT(i, i+1).
    for i in 0..n {
        gopt[i] = (fval[i + 1] - fbase) / xpt[[i, i + 1]];
    }

    // PRIMA L380-382: NDIAG = MIN(N, NPT-N-1); XA/XB are diags of XPT sections.
    let ndiag = n.min(npt - n - 1);
    xa.fill(0.0);
    xb.fill(0.0);
    for k in 0..ndiag {
        xa[k] = xpt[[k, k + 1]]; // diag(xpt(:, 2:ndiag+1))
        xb[k] = xpt[[k, n + 1 + k]]; // diag(xpt(:, n+2:n+ndiag+1))
    }

    // PRIMA L385: revise GOPT(1:NDIAG) to the three-point divided-difference gradient. With
    // da = (f(xa·e_k) - fbase)/xa (the forward difference already sitting in gopt[k]) and db the
    // same on xb, the exact gradient at 0 of the quadratic through {f(xa), fbase, f(xb)} is
    // (da·xb - db·xa)/(xb - xa).
    for k in 0..ndiag {
        gopt[k] = (gopt[k] * xb[k] - ((fval[n + 1 + k] - fbase) / xb[k]) * xa[k]) / (xb[k] - xa[k]);
    }

    // PRIMA L390-393: the diagonal of HQ — the curvature of that same axis-k quadratic,
    // HQ(k,k) = 2·(da - db)/(xa - xb), the second divided difference.
    hq.fill(0.0);
    for k in 0..ndiag {
        hq[[k, k]] = 2.0 * ((fval[k + 1] - fbase) / xa[k] - (fval[n + k + 1] - fbase) / xb[k])
            / (xa[k] - xb[k]);
    }

    // PRIMA L399-407: when NPT > 2*N+1, the off-diagonal entries of HQ. The (i, j) pair is
    // already 0-based here; the +1 in the FVAL indices is PRIMA's base-point offset.
    for (k, &(i, j)) in ij.iter().enumerate() {
        let xi = xpt[[i, 2 * n + 1 + k]];
        let xj = xpt[[j, 2 * n + 1 + k]];
        hq[[i, j]] = (fbase - fval[i + 1] - fval[j + 1] + fval[2 * n + 1 + k]) / (xi * xj);
        hq[[j, i]] = hq[[i, j]];
    }

    // PRIMA L409-412: KOPT = MINLOC(FVAL) (no mask; first minimum); shift GOPT if KOPT /= 1.
    let mut kopt = 0;
    for k in 0..npt {
        if fval[k] < fval[kopt] {
            kopt = k;
        }
    }
    if kopt != 0 {
        // PRIMA L411: GOPT = GOPT + MATPROD(HQ, XPT(:, KOPT)).
        matprod21_into(hq, xpt.col(kopt), shift);
        for i in 0..n {
            gopt[i] += shift[i];
        }
    }

    // PRIMA L414: PQ = ZERO.
    pq.fill(0.0);

    // PRIMA L416-422: info (the Fortran computes it under `present(info)`).
    if gopt.iter().any(|v| v.is_nan()) || hq.data().iter().any(|v| v.is_nan()) {
        crate::consts::NAN_INF_MODEL
    } else {
        INFO_DFT
    }
}

/// PRIMA initialize.f90 L438 `inith`: initialize [BMAT, ZMAT], representing the matrix H of
/// (2.7) of the BOBYQA paper. Returns `info` (same `present(info)` note as [`initq`]).
pub(crate) fn inith(
    ij: &[(usize, usize)],
    xpt: &Mat,
    bmat: &mut Mat,
    zmat: &mut Mat,
    ws: &mut InitWs,
) -> i32 {
    let n = xpt.nrows();
    let npt = xpt.ncols();

    let InitWs { xa, xb, .. } = ws;

    // PRIMA L496-497: read RHOBEG from XPT (XPT(:, 1) = 0); RHOBEG**2 -> rhobeg * rhobeg.
    let mut rhobeg = 0.0_f64;
    for &v in xpt.col(1) {
        rhobeg = rhobeg.max(crate::math::abs(v));
    }
    let rhosq = rhobeg * rhobeg;

    // PRIMA L500-502: NDIAG and the XA/XB diagonals (array-section diag -> indexed loops).
    let ndiag = n.min(npt - n - 1);
    xa.fill(0.0);
    xb.fill(0.0);
    for k in 0..ndiag {
        xa[k] = xpt[[k, k + 1]]; // diag(xpt(:, 2:ndiag+1))
        xb[k] = xpt[[k, n + 1 + k]]; // diag(xpt(:, n+2:n+ndiag+1))
    }

    bmat.fill(0.0);
    // PRIMA L506: BMAT(1:NDIAG, 1) = -(XA + XB) / (XA * XB). Row k holds the only three nonzero
    // ∂L_j/∂x_k of the Lagrange functions for the axis-k triple {f(xa·e_k), f(0), f(xb·e_k)};
    // this is the base-point (f(0)) weight, with denominator xa·xb.
    for k in 0..ndiag {
        bmat[[k, 0]] = -(xa[k] + xb[k]) / (xa[k] * xb[k]);
    }
    // PRIMA L507-510: the other two weights — the xa-point (col k+1) and the xb-point (col k+n+1).
    // As a first-derivative estimator the three weights sum to zero, hence
    // bmat(k, k+1) = -bmat(k, 0) - bmat(k, k+n+1).
    for k in 0..ndiag {
        bmat[[k, k + n + 1]] = -0.5 / xpt[[k, k + 1]];
        bmat[[k, k + 1]] = -bmat[[k, 0]] - bmat[[k, k + n + 1]];
    }
    // PRIMA L512-516: rows NDIAG+1 .. N (only when NDIAG < N, i.e. NPT < 2*N+1).
    for k in ndiag..n {
        bmat[[k, 0]] = -1.0 / xpt[[k, k + 1]];
        bmat[[k, k + 1]] = -bmat[[k, 0]];
        bmat[[k, npt + k]] = -0.5 * rhosq;
    }

    zmat.fill(0.0);
    // PRIMA L520: ZMAT(1, 1:NDIAG) = SQRT(TWO) / (XA * XB).
    for k in 0..ndiag {
        zmat[[0, k]] = crate::math::sqrt(2.0) / (xa[k] * xb[k]);
    }
    // PRIMA L521-524: the xa-point (row k+1) and xb-point (row k+n+1) entries complete ZMAT
    // column k for the axis-k triple; ZMAT's columns factor the implicit-Hessian block Ω of H.
    for k in 0..ndiag {
        zmat[[k + 1, k]] = -zmat[[0, k]] - crate::math::sqrt(0.5) / rhosq;
        zmat[[k + n + 1, k]] = crate::math::sqrt(0.5) / rhosq;
    }
    // PRIMA L526-530: columns NDIAG+1 .. NPT-N-1. The Fortran indexes IJ(:, K - N) with 1-based
    // K; 0-based that is ij[k - n] (here ndiag == n whenever this loop runs, so k - n >= 0).
    // Both rows IJ + 1 (1-based) get -ONE/RHOSQ: 0-based rows are ij.0 + 1 and ij.1 + 1.
    for k in ndiag..(npt - n - 1) {
        zmat[[0, k]] = 1.0 / rhosq;
        zmat[[k + n + 1, k]] = 1.0 / rhosq;
        let (i, j) = ij[k - n];
        zmat[[i + 1, k]] = -1.0 / rhosq;
        zmat[[j + 1, k]] = -1.0 / rhosq;
    }

    // PRIMA L532-538: info (computed under `present(info)` in the Fortran).
    if bmat.data().iter().any(|v| v.is_nan()) || zmat.data().iter().any(|v| v.is_nan()) {
        crate::consts::NAN_INF_MODEL
    } else {
        INFO_DFT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;
    use crate::test_support::{self, DiffStats};

    #[test]
    fn initxf_matches_prima_on_every_captured_state() {
        let corpus = test_support::load_states("initxf");
        assert!(!corpus.is_empty());
        let mut stats = DiffStats::default();
        for st in &corpus {
            let (e, x) = (&st.entry, &st.exit);
            let (xl, xu) = (e.vec("xl"), e.vec("xu"));
            let mut x0 = e.vec("x0");
            let n = xl.len();
            let npt = x.vec("fval").len();
            let f = test_support::objective(&st.problem);
            let mut ij = Vec::new();
            let mut fval = vec![0.0; npt];
            let (mut sl, mut su, mut xbase) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            let mut xpt = Mat::zeros(n, npt);
            let mut ws = InitWs::new(n, npt);
            let (kopt, nf, info) = initxf(
                &mut |p: &[f64]| f(p),
                e.usize("maxfun"),
                e.f64("ftarget"),
                e.f64("rhobeg"),
                &xl,
                &xu,
                &mut x0,
                &mut ij,
                &mut fval,
                &mut sl,
                &mut su,
                &mut xbase,
                &mut xpt,
                &mut ws,
            );
            // Integer/index outputs: exact, with the 1-based -> 0-based translation (design §3.2).
            assert_eq!(kopt + 1, x.usize("kopt"), "{}: kopt", st.problem);
            assert_eq!(nf, x.usize("nf"), "{}: nf", st.problem);
            assert_eq!(i64::from(info), x.i64("info"), "{}: info", st.problem);
            assert_eq!(ij, x.ij("ij"), "{}: ij", st.problem);
            // f64 outputs: within tolerance, bit-exactness tracked.
            stats.slice("x0", &x0, &x.vec("x0"));
            stats.slice("fval", &fval, &x.vec("fval"));
            stats.slice("sl", &sl, &x.vec("sl"));
            stats.slice("su", &su, &x.vec("su"));
            stats.slice("xbase", &xbase, &x.vec("xbase"));
            stats.mat("xpt", &xpt, &x.mat("xpt"));
        }
        stats.report("initxf");
    }

    #[test]
    fn initq_matches_prima_on_every_captured_state() {
        let corpus = test_support::load_states("initq");
        assert!(!corpus.is_empty());
        let mut stats = DiffStats::default();
        for st in &corpus {
            let (e, x) = (&st.entry, &st.exit);
            let ij = e.ij("ij");
            let fval = e.vec("fval");
            let xpt = e.mat("xpt");
            let n = xpt.nrows();
            let mut gopt = vec![0.0; n];
            let mut hq = Mat::zeros(n, n);
            let mut pq = vec![0.0; fval.len()];
            let mut ws = InitWs::new(xpt.nrows(), xpt.ncols());
            // No `info` diff: the bobyqb.f90 call site on this pin omits the optional INFO, so
            // the guarded dump emits nothing (oracle/README.md, Instrumentation).
            let _info = initq(&ij, &fval, &xpt, &mut gopt, &mut hq, &mut pq, &mut ws);
            stats.slice("gopt", &gopt, &x.vec("gopt"));
            stats.mat("hq", &hq, &x.mat("hq"));
            stats.slice("pq", &pq, &x.vec("pq"));
        }
        stats.report("initq");
    }

    #[test]
    fn inith_matches_prima_on_every_captured_state() {
        let corpus = test_support::load_states("inith");
        assert!(!corpus.is_empty());
        let mut stats = DiffStats::default();
        for st in &corpus {
            let (e, x) = (&st.entry, &st.exit);
            let ij = e.ij("ij");
            let xpt = e.mat("xpt");
            let (n, npt) = (xpt.nrows(), xpt.ncols());
            let mut bmat = Mat::zeros(n, npt + n);
            let mut zmat = Mat::zeros(npt, npt - n - 1);
            let mut ws = InitWs::new(n, npt);
            // No `info` diff: the bobyqb.f90 call site on this pin omits the optional INFO
            // (oracle/README.md, Instrumentation).
            let _info = inith(&ij, &xpt, &mut bmat, &mut zmat, &mut ws);
            stats.mat("bmat", &bmat, &x.mat("bmat"));
            stats.mat("zmat", &zmat, &x.mat("zmat"));
        }
        stats.report("inith");
    }
}
