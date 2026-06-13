//! `update.f90` (PRIMA bobyqa module): `updateh`, `updatexf`, `updateq`, `tryqalt` — the updates
//! when XPT(:, KNEW) becomes XNEW = XOPT + D (design §5).
//!
//! Index convention: all indices 0-based (`knew`, `kopt`, `k`); PRIMA's `KNEW = 0` sentinel is
//! `Option<usize>` (design §3.2); the diff tests translate the 1-based dump values. The optional
//! `INFO` out-arg is returned as `i32` (`consts.rs` values); the `bobyqb.f90` call sites on this
//! pin pass it nowhere, so the state corpus has no `info` field (oracle/README.md).
use crate::consts::{DAMAGING_ROUNDING, INFO_DFT};
use crate::linalg::{inprod, matprod12_into, matprod21_into, planerot, r1update};
use crate::mat::Mat;
use crate::math;
use crate::powalg::{CalWs, calbeta, calvlag_into, hess_mul_into};

/// Reused scratch for the update.f90 routines — PRIMA's per-call locals, hoisted to the solver
/// workspace (rust.md §4). Lifetime and contents per call are identical to the Fortran locals;
/// only the allocation site moves: each field is re-initialized at the original allocation
/// site, per call. Field → Fortran-local map: `hcol`/`vlag`/`v1`/`v2` are updateh's
/// HCOL/VLAG/V1/V2; `pqinc` updateq's PQINC; `pgopt`/`pqalt`/`galt`/`pgalt` tryqalt's
/// PGOPT/PQALT/GALT/PGALT; `zrow` the ZMAT(KNEW, :) row-extraction temp (updateh L119,
/// updateq L356), `zmat_zrow`/`inner` the MATPROD temps, `hm` the `HESS_MUL` results
/// (updateq L360/L363 — sequential, so one buffer — and tryqalt L464), `dxpt` `hess_mul`
/// scratch.
#[derive(Debug, Clone)]
pub(crate) struct UpdateWs {
    zrow: Vec<f64>,      // npt - n - 1
    hcol: Vec<f64>,      // npt + n
    vlag: Vec<f64>,      // npt + n
    v1: Vec<f64>,        // n
    v2: Vec<f64>,        // n
    pqinc: Vec<f64>,     // npt
    zmat_zrow: Vec<f64>, // npt
    hm: Vec<f64>,        // n
    pgopt: Vec<f64>,     // n
    inner: Vec<f64>,     // npt - n - 1
    pqalt: Vec<f64>,     // npt
    galt: Vec<f64>,      // n
    pgalt: Vec<f64>,     // n
    dxpt: Vec<f64>,      // npt
    cal: CalWs,
}

impl UpdateWs {
    pub(crate) fn new(n: usize, npt: usize) -> Self {
        Self {
            zrow: vec![0.0; npt - n - 1],
            hcol: vec![0.0; npt + n],
            vlag: vec![0.0; npt + n],
            v1: vec![0.0; n],
            v2: vec![0.0; n],
            pqinc: vec![0.0; npt],
            zmat_zrow: vec![0.0; npt],
            hm: vec![0.0; n],
            pgopt: vec![0.0; n],
            inner: vec![0.0; npt - n - 1],
            pqalt: vec![0.0; npt],
            galt: vec![0.0; n],
            pgalt: vec![0.0; n],
            dxpt: vec![0.0; npt],
            cal: CalWs::new(n, npt),
        }
    }
}

/// PRIMA update.f90 L22 `updateh`: update BMAT and ZMAT when XPT(:, KNEW) changes to XNEW.
/// Returns `info`: `INFO_DFT` on success, `DAMAGING_ROUNDING` if the denominator was bad.
#[expect(clippy::too_many_arguments)] // out-params mirror the Fortran intent (rust.md §5)
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA (rust.md §5)
pub(crate) fn updateh(
    knew: Option<usize>,
    kopt: usize,
    d: &[f64],
    xpt: &Mat,
    bmat: &mut Mat,
    zmat: &mut Mat,
    ws: &mut UpdateWs,
    precomputed: Option<(&[f64], f64)>, // (vlag with +1, beta) from bobyqb this iteration; spec §5
) -> i32 {
    let n = xpt.nrows();
    let npt = xpt.ncols();

    // PRIMA update.f90 L107-114: info defaults; KNEW == 0 (None) -> return immediately.
    let Some(knew) = knew else {
        return INFO_DFT;
    };

    let UpdateWs {
        zrow,
        hcol,
        vlag,
        v1,
        v2,
        cal,
        ..
    } = ws;

    // PRIMA update.f90 L119: HCOL(1:NPT) = MATPROD(ZMAT, ZMAT(KNEW, :)) — row-extraction temp,
    // the product written straight into HCOL's head.
    for j in 0..zmat.ncols() {
        zrow[j] = zmat[[knew, j]];
    }
    matprod21_into(zmat, zrow, &mut hcol[..npt]);
    // PRIMA update.f90 L120: HCOL(NPT+1:NPT+N) = BMAT(:, KNEW).
    hcol[npt..npt + n].copy_from_slice(bmat.col(knew));

    // PRIMA update.f90 L123-124: BETA and VLAG (kopt is already 0-based). Layer-0 spec §5
    // (Tier B): reuse bobyqb's same-iteration values (kref = kopt, same d/xpt/zmat/bmat, no
    // rescue since) by COPY — bit-identical, saves two kernels. Order in the recompute branch
    // is calbeta THEN calvlag (PRIMA's order); the reuse branch is order-free (pure copy).
    let beta = if let Some((vlag_src, beta_src)) = precomputed {
        vlag.copy_from_slice(vlag_src);
        beta_src
    } else {
        let beta = calbeta(kopt, bmat, d, xpt, zmat, cal);
        calvlag_into(kopt, bmat, d, xpt, zmat, cal, vlag);
        beta
    };

    // PRIMA update.f90 L128-130: ALPHA, TAU, DENOM.
    let alpha = hcol[knew];
    let tau = vlag[knew];
    let denom = alpha * beta + tau * tau;

    // PRIMA update.f90 L133: after this, VLAG = H*w - e_KNEW.
    vlag[knew] -= 1.0;

    // PRIMA update.f90 L138-143: damaging-rounding guard.
    // Uses !(is_finite(...) && denom > 0) — negated comparison is load-bearing for NaN (shared
    // convention: never simplify to the positive form).
    let hcol_sum_abs: f64 = hcol.iter().map(|v| math::abs(*v)).sum();
    let vlag_sum_abs: f64 = vlag.iter().map(|v| math::abs(*v)).sum();
    if !((hcol_sum_abs + vlag_sum_abs + math::abs(beta)).is_finite() && denom > 0.0) {
        return DAMAGING_ROUNDING;
    }

    // PRIMA update.f90 L146-147: V1 and V2.
    for i in 0..n {
        v1[i] = (alpha * vlag[npt + i] - tau * hcol[npt + i]) / denom;
        v2[i] = (-beta * hcol[npt + i] - tau * vlag[npt + i]) / denom;
    }

    // PRIMA update.f90 L148: BMAT = BMAT + OUTPROD(V1, VLAG) + OUTPROD(V2, HCOL).
    // Two separate loops to preserve Fortran's left-to-right FP order:
    // first add OUTPROD(V1, VLAG), then add OUTPROD(V2, HCOL).
    // Column-slice form: same i-ascending per-column adds, no per-element
    // index arithmetic or bounds checks; the lanes are independent.
    for j in 0..(npt + n) {
        let vj = vlag[j];
        let bj = &mut bmat.col_mut(j)[..n];
        let v1 = &v1[..n];
        for i in 0..n {
            bj[i] += v1[i] * vj;
        }
    }
    for j in 0..(npt + n) {
        let hj = hcol[j];
        let bj = &mut bmat.col_mut(j)[..n];
        let v2 = &v2[..n];
        for i in 0..n {
            bj[i] += v2[i] * hj;
        }
    }

    // PRIMA update.f90 L151: SYMMETRIZE(BMAT(:, NPT+1:NPT+N)) — in place on the trailing n x n
    // section: copy the lower triangle to the upper in linalg::symmetrize's exact assignment
    // order (M2 §4.3: the section-copy temporary is dropped; pure copies, FP-identical).
    // Mirrored by rescue.rs::updateh_rsc — change together.
    for j in 0..n {
        for i in 0..j {
            bmat[[i, npt + j]] = bmat[[j, npt + i]];
        }
    }

    // PRIMA update.f90 L155-161: Givens rotations to zero ZMAT(KNEW, 2:NPT-N-1).
    // Fortran j = 2..NPT-N-1 (1-based) -> j in 1..(npt - n - 1) (0-based, exclusive upper).
    for j in 1..(npt - n - 1) {
        // PRIMA update.f90 L156: threshold uses MAXVAL(ABS(ZMAT)) — re-evaluated each iteration
        // because ZMAT changes; NOT hoisted (Fortran re-evaluates; keep inside) — matches
        // rescue.rs::updateh_rsc.
        let max_abs_zmat = zmat
            .data()
            .iter()
            .map(|v| math::abs(*v))
            .fold(0.0_f64, f64::max);
        // PRIMA update.f90 L156: the literal 1.0E-20 has no _RP suffix — gfortran evaluates it in
        // SINGLE precision (= 9.999999682655225e-21), and the oracle binary compares against that.
        // rescue.rs::updateh_rsc carries the identical sentinel — change together.
        if math::abs(zmat[[knew, j]]) > f64::from(1.0e-20_f32) * max_abs_zmat {
            // PRIMA update.f90 L157-158: grot = planerot(zmat(knew, [1, j])).
            let grot = planerot([zmat[[knew, 0]], zmat[[knew, j]]]);
            // PRIMA update.f90 L158: ZMAT(:, [1, J]) = MATPROD(ZMAT(:, [1, J]), TRANSPOSE(GROT)).
            // Two-column slice view — same rotation, same per-row FP ops.
            let (z0, zj) = zmat.two_cols_mut(0, j);
            for i in 0..npt {
                let (a, b) = (z0[i], zj[i]);
                z0[i] = a * grot[0][0] + b * grot[0][1];
                zj[i] = a * grot[1][0] + b * grot[1][1];
            }
        }
        // PRIMA update.f90 L160: ZMAT(KNEW, J) = ZERO — always, inside the J loop.
        zmat[[knew, j]] = 0.0;
    }

    // PRIMA update.f90 L164-165: complete the ZMAT update.
    // Hoist ZMAT(KNEW, 1) before the loop — the LHS-read trap: the loop overwrites zmat[[knew, 0]]
    // at i == knew, so we must capture the unupdated value first.
    let sqrtdn = math::sqrt(denom);
    let zknew1 = zmat[[knew, 0]];
    // The two quotients are loop-invariant (hoisting computes the identical
    // values once); column-slice access for the rest.
    let tau_q = tau / sqrtdn;
    let zk_q = zknew1 / sqrtdn;
    let z0 = &mut zmat.col_mut(0)[..npt];
    let vl = &vlag[..npt];
    for i in 0..npt {
        z0[i] = tau_q * z0[i] - zk_q * vl[i];
    }

    INFO_DFT
}

/// PRIMA update.f90 L198 `updatexf`: update XPT, FVAL, KOPT when XPT(:, KNEW) -> XNEW.
/// `knew = None` is a silent no-op.
pub(crate) fn updatexf(
    knew: Option<usize>,
    ximproved: bool,
    f: f64,
    xnew: &[f64],
    kopt: &mut usize,
    fval: &mut [f64],
    xpt: &mut Mat,
) {
    // PRIMA update.f90 L249-252: KNEW == 0 (None) -> return.
    let Some(knew) = knew else {
        return;
    };

    // PRIMA update.f90 L254: XPT(:, KNEW) = XNEW.
    xpt.col_mut(knew).copy_from_slice(xnew);
    // PRIMA update.f90 L255: FVAL(KNEW) = F.
    fval[knew] = f;

    // PRIMA update.f90 L259-261: update KOPT only when XIMPROVED.
    if ximproved {
        *kopt = knew;
    }
}

/// PRIMA update.f90 L278 `updateq`: update GOPT, HQ, PQ when XPT(:, KNEW) changes.
#[expect(clippy::too_many_arguments)] // out-params mirror the Fortran intent (rust.md §5)
pub(crate) fn updateq(
    knew: Option<usize>,
    ximproved: bool,
    bmat: &Mat,
    d: &[f64],
    moderr: f64,
    xdrop: &[f64],
    xosav: &[f64],
    xpt: &Mat,
    zmat: &Mat,
    gopt: &mut [f64],
    hq: &mut Mat,
    pq: &mut [f64],
    ws: &mut UpdateWs,
) {
    // PRIMA update.f90 L342-345: KNEW == 0 (None) -> return.
    let Some(knew) = knew else {
        return;
    };

    let UpdateWs {
        zrow,
        zmat_zrow,
        pqinc,
        hm,
        dxpt,
        ..
    } = ws;

    // PRIMA update.f90 L352: absorb PQ(KNEW)*XDROP*XDROP^T into HQ, then zero PQ(KNEW).
    r1update(hq, pq[knew], xdrop);
    // PRIMA update.f90 L353: PQ(KNEW) = ZERO.
    pq[knew] = 0.0;

    // PRIMA update.f90 L356: PQINC = MODERR * MATPROD(ZMAT, ZMAT(KNEW, :)).
    // Row-extraction temp for the column-major ZMAT (the array-section→loop convention; see `linalg`).
    for j in 0..zmat.ncols() {
        zrow[j] = zmat[[knew, j]];
    }
    matprod21_into(zmat, zrow, zmat_zrow);
    let npt = pq.len();
    for k in 0..npt {
        pqinc[k] = moderr * zmat_zrow[k];
    }

    // PRIMA update.f90 L357: PQ += PQINC.
    for k in 0..npt {
        pq[k] += pqinc[k];
    }

    // PRIMA update.f90 L360: GOPT = GOPT + MODERR*BMAT(:, KNEW) + HESS_MUL(XOSAV, XPT, PQINC).
    // 3-arg call (no HQ) -> None. Two separate additions to preserve Fortran left-to-right FP order:
    // first GOPT += MODERR*BMAT(:, KNEW), then GOPT += HESS_MUL(...).
    hess_mul_into(xosav, xpt, pqinc, None, dxpt, hm);
    let n = gopt.len();
    for i in 0..n {
        gopt[i] += moderr * bmat[[i, knew]];
    }
    for i in 0..n {
        gopt[i] += hm[i];
    }

    // PRIMA update.f90 L363-365: further update GOPT if XIMPROVED (XOPT shifts from XOSAV to XNEW).
    // 4-arg call (with HQ) -> Some. `hm` is reused: its L360 value was consumed just above.
    if ximproved {
        hess_mul_into(d, xpt, pq, Some(hq), dxpt, hm);
        for i in 0..n {
            gopt[i] += hm[i];
        }
    }
}

/// PRIMA update.f90 L381 `tryqalt`: test whether to replace Q with the least Frobenius norm
/// interpolant; replace (gopt ← galt, pq ← pqalt, hq ← 0, itest ← 0) when `itest` reaches 3.
#[expect(clippy::too_many_arguments)] // out-params mirror the Fortran intent (rust.md §5)
#[expect(clippy::similar_names)] // xopt/xpt and pqalt/pgalt are PRIMA identifiers (rust.md §5)
pub(crate) fn tryqalt(
    bmat: &Mat,
    fval: &[f64],
    ratio: f64,
    sl: &[f64],
    su: &[f64],
    xopt: &[f64],
    xpt: &Mat,
    zmat: &Mat,
    itest: &mut i32,
    gopt: &mut [f64],
    hq: &mut Mat,
    pq: &mut [f64],
    ws: &mut UpdateWs,
) {
    let n = gopt.len();
    let npt = pq.len();

    let UpdateWs {
        pgopt,
        inner,
        pqalt,
        galt,
        pgalt,
        hm,
        dxpt,
        ..
    } = ws;

    // PRIMA update.f90 L458: PGOPT = GOPT.
    pgopt.copy_from_slice(gopt);
    // PRIMA update.f90 L459: PGOPT(TRUELOC(XOPT >= SU)) = MAX(ZERO, GOPT(...)).
    // Transcribed as ascending-index loop with if (shared convention: masked array ops).
    for i in 0..n {
        if xopt[i] >= su[i] {
            pgopt[i] = 0.0_f64.max(gopt[i]);
        }
    }
    // PRIMA update.f90 L460: PGOPT(TRUELOC(XOPT <= SL)) = MIN(ZERO, GOPT(...)).
    for i in 0..n {
        if xopt[i] <= sl[i] {
            pgopt[i] = 0.0_f64.min(gopt[i]);
        }
    }

    // PRIMA update.f90 L463: PQALT = MATPROD(ZMAT, MATPROD(FVAL, ZMAT)).
    // matprod12(fval, zmat) then matprod21(zmat, inner).
    matprod12_into(fval, zmat, inner);
    matprod21_into(zmat, inner, pqalt);

    // PRIMA update.f90 L464: GALT = MATPROD(BMAT(:, 1:NPT), FVAL) + HESS_MUL(XOPT, XPT, PQALT).
    // Section product: explicit column-outer loop over j in 0..npt (mirrors matprod21 order).
    galt.fill(0.0);
    for j in 0..npt {
        let fvalj = fval[j];
        for i in 0..n {
            galt[i] += bmat[[i, j]] * fvalj;
        }
    }
    hess_mul_into(xopt, xpt, pqalt, None, dxpt, hm);
    for i in 0..n {
        galt[i] += hm[i];
    }

    // PRIMA update.f90 L467-469: PGALT clamps, same shape as PGOPT.
    pgalt.copy_from_slice(galt);
    for i in 0..n {
        if xopt[i] >= su[i] {
            pgalt[i] = 0.0_f64.max(galt[i]);
        }
    }
    for i in 0..n {
        if xopt[i] <= sl[i] {
            pgalt[i] = 0.0_f64.min(galt[i]);
        }
    }

    // PRIMA update.f90 L476: TENTH = 0.1, TEN = 10.0 literals.
    // PRIMA update.f90 L476: if (ratio > TENTH .or. inprod(pgopt) < TEN * inprod(pgalt)) then
    //   itest = 0  else  itest += 1.
    let pgopt_sq = inprod(pgopt, pgopt);
    let pgalt_sq = inprod(pgalt, pgalt);
    if ratio > 0.1 || pgopt_sq < 10.0 * pgalt_sq {
        *itest = 0;
    } else {
        *itest += 1;
    }

    // PRIMA update.f90 L481-486: replace model when ITEST >= 3.
    if *itest >= 3 {
        gopt.copy_from_slice(galt);
        pq.copy_from_slice(pqalt);
        hq.fill(0.0);
        *itest = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;
    use crate::test_support::{self, DiffStats};

    #[test]
    #[expect(clippy::similar_names)] // states/stats are conventional diff-test locals
    fn updateh_matches_prima_on_every_captured_state() {
        let states = test_support::load_states("updateh");
        assert!(!states.is_empty());
        let mut stats = DiffStats::default();
        for st in &states {
            let (e, x) = (&st.entry, &st.exit);
            let knew = match e.usize("knew") {
                0 => None,
                k => Some(k - 1),
            };
            let kopt = e.usize("kopt") - 1;
            let d = e.vec("d");
            let xpt = e.mat("xpt");
            let mut bmat = e.mat("bmat");
            let mut zmat = e.mat("zmat");
            // No `info` diff: the bobyqb.f90 call site on this pin omits the optional INFO
            // (oracle/README.md, Instrumentation).
            let mut ws = UpdateWs::new(xpt.nrows(), xpt.ncols());
            let _info = updateh(knew, kopt, &d, &xpt, &mut bmat, &mut zmat, &mut ws, None);
            stats.mat("bmat", &bmat, &x.mat("bmat"));
            stats.mat("zmat", &zmat, &x.mat("zmat"));
        }
        stats.report("updateh");
    }

    #[test]
    #[expect(clippy::similar_names)] // states/stats are conventional diff-test locals
    fn updatexf_matches_prima_on_every_captured_state() {
        let states = test_support::load_states("updatexf");
        assert!(!states.is_empty());
        let mut stats = DiffStats::default();
        for st in &states {
            let (e, x) = (&st.entry, &st.exit);
            let knew = match e.usize("knew") {
                0 => None,
                k => Some(k - 1),
            };
            let ximproved = e.i64("ximproved") != 0;
            let f = e.f64("f");
            let xnew = e.vec("xnew");
            let mut kopt = e.usize("kopt") - 1;
            let mut fval = e.vec("fval");
            let mut xpt = e.mat("xpt");
            updatexf(knew, ximproved, f, &xnew, &mut kopt, &mut fval, &mut xpt);
            assert_eq!(kopt + 1, x.usize("kopt"), "{}: kopt", st.problem);
            stats.slice("fval", &fval, &x.vec("fval"));
            stats.mat("xpt", &xpt, &x.mat("xpt"));
        }
        stats.report("updatexf");
    }

    #[test]
    #[expect(clippy::similar_names)] // states/stats are conventional diff-test locals
    fn updateq_matches_prima_on_every_captured_state() {
        let states = test_support::load_states("updateq");
        assert!(!states.is_empty());
        let mut stats = DiffStats::default();
        for st in &states {
            let (e, x) = (&st.entry, &st.exit);
            let knew = match e.usize("knew") {
                0 => None,
                k => Some(k - 1),
            };
            let ximproved = e.i64("ximproved") != 0;
            let bmat = e.mat("bmat");
            let d = e.vec("d");
            let moderr = e.f64("moderr");
            let xdrop = e.vec("xdrop");
            let xosav = e.vec("xosav");
            let xpt = e.mat("xpt");
            let zmat = e.mat("zmat");
            let mut gopt = e.vec("gopt");
            let mut hq = e.mat("hq");
            let mut pq = e.vec("pq");
            let mut ws = UpdateWs::new(xpt.nrows(), xpt.ncols());
            updateq(
                knew, ximproved, &bmat, &d, moderr, &xdrop, &xosav, &xpt, &zmat, &mut gopt,
                &mut hq, &mut pq, &mut ws,
            );
            stats.slice("gopt", &gopt, &x.vec("gopt"));
            stats.mat("hq", &hq, &x.mat("hq"));
            stats.slice("pq", &pq, &x.vec("pq"));
        }
        stats.report("updateq");
    }

    #[test]
    #[expect(clippy::similar_names)] // states/stats and xopt/xpt are PRIMA identifiers (rust.md §5)
    fn tryqalt_matches_prima_on_every_captured_state() {
        let states = test_support::load_states("tryqalt");
        assert!(!states.is_empty());
        let mut stats = DiffStats::default();
        for st in &states {
            let (e, x) = (&st.entry, &st.exit);
            let bmat = e.mat("bmat");
            let fval = e.vec("fval");
            let ratio = e.f64("ratio");
            let sl = e.vec("sl");
            let su = e.vec("su");
            let xopt = e.vec("xopt");
            let xpt = e.mat("xpt");
            let zmat = e.mat("zmat");
            let mut itest = i32::try_from(e.i64("itest")).unwrap();
            let mut gopt = e.vec("gopt");
            let mut hq = e.mat("hq");
            let mut pq = e.vec("pq");
            let mut ws = UpdateWs::new(xpt.nrows(), xpt.ncols());
            tryqalt(
                &bmat, &fval, ratio, &sl, &su, &xopt, &xpt, &zmat, &mut itest, &mut gopt, &mut hq,
                &mut pq, &mut ws,
            );
            assert_eq!(
                itest,
                i32::try_from(x.i64("itest")).unwrap(),
                "{}: itest",
                st.problem
            );
            stats.slice("gopt", &gopt, &x.vec("gopt"));
            stats.mat("hq", &hq, &x.mat("hq"));
            stats.slice("pq", &pq, &x.vec("pq"));
        }
        stats.report("tryqalt");
    }

    // ---------------------------------------------------------------------------
    // Module-local unit tests (hand-computed).
    // ---------------------------------------------------------------------------

    #[test]
    fn updateh_with_no_knew_leaves_h_untouched() {
        // 1x4 toy state: any consistent shapes will do — the routine must return before touching them.
        let xpt = Mat::zeros(1, 4);
        let mut bmat = Mat::from_col_major(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut zmat = Mat::from_col_major(4, 2, (0..8).map(f64::from).collect());
        let (b0, z0) = (bmat.data().to_vec(), zmat.data().to_vec());
        let mut ws = UpdateWs::new(xpt.nrows(), xpt.ncols());
        let info = updateh(None, 0, &[0.0], &xpt, &mut bmat, &mut zmat, &mut ws, None);
        assert_eq!(info, crate::consts::INFO_DFT);
        assert_eq!(bmat.data(), &b0[..]);
        assert_eq!(zmat.data(), &z0[..]);
    }

    #[test]
    fn updatexf_without_improvement_keeps_kopt() {
        let mut xpt = Mat::zeros(1, 4);
        let mut fval = vec![3.0, 1.0, 4.0, 5.0];
        let mut kopt = 1;
        updatexf(Some(2), false, 2.0, &[0.5], &mut kopt, &mut fval, &mut xpt);
        assert_eq!(kopt, 1);
        assert_eq!(fval[2], 2.0);
        assert_eq!(xpt[[0, 2]], 0.5);
    }

    #[test]
    fn updatexf_with_no_knew_is_a_silent_no_op() {
        // knew = None must return before touching kopt/fval/xpt (PRIMA L249-252). The oracle corpus
        // never captures knew=0 for updatexf, so this is the sole guard for the sentinel path; pass
        // ximproved=true so a broken early-return that proceeds would visibly move kopt.
        let mut xpt = Mat::from_col_major(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let mut fval = vec![3.0, 1.0, 4.0, 5.0];
        let mut kopt = 1;
        let (x0, f0) = (xpt.data().to_vec(), fval.clone());
        updatexf(None, true, 9.0, &[7.0], &mut kopt, &mut fval, &mut xpt);
        assert_eq!(kopt, 1);
        assert_eq!(fval, f0);
        assert_eq!(xpt.data(), &x0[..]);
    }

    #[test]
    fn updateq_with_no_knew_is_a_silent_no_op() {
        // knew = None must return before touching gopt/hq/pq (PRIMA L342-345); also corpus-unreachable.
        let (n, npt) = (1, 4);
        let bmat = Mat::zeros(n, npt + n);
        let zmat = Mat::zeros(npt, npt - n - 1);
        let xpt = Mat::zeros(n, npt);
        let mut gopt = vec![5.0];
        let mut hq = Mat::from_col_major(1, 1, vec![7.0]);
        let mut pq = vec![1.0, 2.0, 3.0, 4.0];
        let (g0, h0, p0) = (gopt.clone(), hq.data().to_vec(), pq.clone());
        let mut ws = UpdateWs::new(n, npt);
        updateq(
            None,
            true,
            &bmat,
            &[0.0],
            0.5,
            &[0.0],
            &[0.0],
            &xpt,
            &zmat,
            &mut gopt,
            &mut hq,
            &mut pq,
            &mut ws,
        );
        assert_eq!(gopt, g0);
        assert_eq!(hq.data(), &h0[..]);
        assert_eq!(pq, p0);
    }

    #[test]
    #[expect(clippy::similar_names)] // xopt/xpt are PRIMA identifiers (rust.md §5)
    fn tryqalt_replaces_the_model_on_the_third_consecutive_failure() {
        // Bound-free toy where pgopt is huge vs pgalt (zmat = 0 => pqalt = 0, galt from bmat):
        // ratio <= 0.1 and the gradient test fails => itest increments; at 3 the model resets.
        let n = 1;
        let npt = 4;
        let bmat = Mat::zeros(n, npt + n);
        let zmat = Mat::zeros(npt, npt - n - 1);
        let xpt = Mat::zeros(n, npt);
        let fval = vec![0.0; npt];
        let (sl, su, xopt) = (vec![-1.0], vec![1.0], vec![0.0]);
        let mut itest = 2;
        let mut gopt = vec![5.0];
        let mut hq = Mat::from_col_major(1, 1, vec![7.0]);
        let mut pq = vec![1.0; npt];
        let mut ws = UpdateWs::new(n, npt);
        tryqalt(
            &bmat, &fval, 0.0, &sl, &su, &xopt, &xpt, &zmat, &mut itest, &mut gopt, &mut hq,
            &mut pq, &mut ws,
        );
        assert_eq!(itest, 0); // reached 3, reset
        assert_eq!(gopt, vec![0.0]); // galt = 0 here
        assert_eq!(hq[[0, 0]], 0.0);
        assert_eq!(pq, vec![0.0; npt]);
    }

    #[test]
    #[expect(clippy::similar_names)] // xopt/xpt, sl/su are PRIMA identifiers (rust.md §5)
    fn tryqalt_resets_itest_when_only_the_ratio_is_good() {
        // ratio > 0.1 is the SOLE reset trigger here: zero bmat/zmat/fval => galt = pgalt = 0, so the
        // second arm (pgopt_sq < 10*pgalt_sq) is false. A `>`->`==` mutation on `ratio > 0.1` (L380)
        // misses 0.5 and would increment itest (1 -> 2) instead of resetting it to 0. The existing
        // tryqalt test only drives the increment path (ratio = 0), never this reset arm.
        let (n, npt) = (1, 4);
        let bmat = Mat::zeros(n, npt + n);
        let zmat = Mat::zeros(npt, npt - n - 1);
        let xpt = Mat::zeros(n, npt);
        let fval = vec![0.0; npt];
        let (sl, su, xopt) = (vec![-1.0], vec![1.0], vec![0.0]);
        let mut itest = 1;
        let mut gopt = vec![5.0];
        let mut hq = Mat::from_col_major(1, 1, vec![7.0]);
        let mut pq = vec![1.0; npt];
        let mut ws = UpdateWs::new(n, npt);
        tryqalt(
            &bmat, &fval, 0.5, &sl, &su, &xopt, &xpt, &zmat, &mut itest, &mut gopt, &mut hq,
            &mut pq, &mut ws,
        );
        assert_eq!(itest, 0);
    }
}
