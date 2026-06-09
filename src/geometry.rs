//! `geometry.f90` (PRIMA bobyqa module): `setdrop_tr` (pick the point to drop after a TR step)
//! and `geostep` (geometry-improving step) — design §5.
//!
//! Index convention: all indices 0-based, with two PRIMA encodings kept verbatim because their
//! arithmetic bakes in 1-based values: `setdrop_tr`'s `KNEW = 0` sentinel maps to
//! `Option<usize>` at the seam, and `geostep`'s `isbd`/`ibd` entries are PRIMA's *signed 1-based*
//! variable indices (−j = j-th lower bound, +j = j-th upper bound, 0 = none) — translate with
//! `as usize - 1` only at the `xline` use sites.
use crate::linalg::{inprod, matprod12_into, matprod21_into, norm};
use crate::mat::Mat;
use crate::math;
use crate::powalg::{CalWs, calden_into, hess_mul_into};

/// Reused scratch for the geometry.f90 routines (`setdrop_tr` and `geostep`) — PRIMA's per-call
/// locals, hoisted to the solver workspace (the crate-`//!` zero-alloc warm path); each field is
/// re-initialized at its allocation site, per call (and per loop iteration where the original was
/// in-loop). Field → Fortran-local map: most fields carry their PRIMA names
/// (`distsq`/`weight`/`score`/`pqlag`/`glag`/`dderiv`/`stplen`/`isbd`/`vlag`/`betabd`/`predsq`/
/// `xline`/`xcauchy`/`s`/`xtemp`/`sxpt` …); `zrow_knew` is the ZMAT(KNEW, :) row-extraction
/// temp, `hm` the `HESS_MUL` result, `den`/`den_line`/`den_cauchy` the three CALDEN results,
/// `s_cauchy` the XCAUCHY − XOPT step, `dxpt` `hess_mul` scratch.
#[derive(Debug, Clone)]
pub(crate) struct GeostepWs {
    // setdrop_tr
    distsq: Vec<f64>, // npt (also geostep's DISTSQ)
    weight: Vec<f64>, // npt
    den: Vec<f64>,    // npt
    score: Vec<f64>,  // npt
    // geostep
    zrow_knew: Vec<f64>,      // npt - n - 1
    pqlag: Vec<f64>,          // npt
    xopt: Vec<f64>,           // n
    hm: Vec<f64>,             // n
    glag: Vec<f64>,           // n
    dderiv: Vec<f64>,         // npt
    stplen: Mat,              // 3 x npt
    isbd: Vec<[i64; 3]>,      // npt
    xdiff: Vec<f64>,          // n
    lfrac: Vec<f64>,          // n
    ufrac: Vec<f64>,          // n
    slbd_test: Vec<f64>,      // n
    subd_test: Vec<f64>,      // n
    vlag: Mat,                // 3 x npt
    betabd: Mat,              // 3 x npt
    predsq: Mat,              // 3 x npt
    xline: Vec<f64>,          // n
    den_line: Vec<f64>,       // npt
    xcauchy: Vec<f64>,        // n
    s: Vec<f64>,              // n
    mask_free: Vec<bool>,     // n
    xtemp: Vec<f64>,          // n
    mask_fixl: Vec<bool>,     // n
    mask_fixu: Vec<bool>,     // n
    new_mask_free: Vec<bool>, // n
    x: Vec<f64>,              // n
    sxpt: Vec<f64>,           // npt
    s_cauchy: Vec<f64>,       // n
    den_cauchy: Vec<f64>,     // npt
    dxpt: Vec<f64>,           // npt
    cal: CalWs,
}

impl GeostepWs {
    pub(crate) fn new(n: usize, npt: usize) -> Self {
        Self {
            distsq: vec![0.0; npt],
            weight: vec![0.0; npt],
            den: vec![0.0; npt],
            score: vec![0.0; npt],
            zrow_knew: vec![0.0; npt - n - 1],
            pqlag: vec![0.0; npt],
            xopt: vec![0.0; n],
            hm: vec![0.0; n],
            glag: vec![0.0; n],
            dderiv: vec![0.0; npt],
            stplen: Mat::zeros(3, npt),
            isbd: vec![[0; 3]; npt],
            xdiff: vec![0.0; n],
            lfrac: vec![0.0; n],
            ufrac: vec![0.0; n],
            slbd_test: vec![0.0; n],
            subd_test: vec![0.0; n],
            vlag: Mat::zeros(3, npt),
            betabd: Mat::zeros(3, npt),
            predsq: Mat::zeros(3, npt),
            xline: vec![0.0; n],
            den_line: vec![0.0; npt],
            xcauchy: vec![0.0; n],
            s: vec![0.0; n],
            mask_free: vec![false; n],
            xtemp: vec![0.0; n],
            mask_fixl: vec![false; n],
            mask_fixu: vec![false; n],
            new_mask_free: vec![false; n],
            x: vec![0.0; n],
            sxpt: vec![0.0; npt],
            s_cauchy: vec![0.0; n],
            den_cauchy: vec![0.0; npt],
            dxpt: vec![0.0; npt],
            cal: CalWs::new(n, npt),
        }
    }
}

/// PRIMA geometry.f90 L22 `setdrop_tr`: pick the interpolation point to drop after a TR step.
/// Returns `Some(knew)` (0-based) or `None` when PRIMA returns `KNEW = 0` (drop nothing).
///
/// N.B. If `ximproved = true`, the result is always `Some(k)` with `k` a valid index.
/// If `ximproved = false`, the result is never `Some(kopt)`.
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA (rust.md §5)
#[expect(clippy::too_many_arguments)] // out-params mirror the Fortran intent (rust.md §5)
pub(crate) fn setdrop_tr(
    kopt: usize,
    ximproved: bool,
    bmat: &Mat,
    d: &[f64],
    _delta: f64,
    rho: f64,
    xpt: &Mat,
    zmat: &Mat,
    ws: &mut GeostepWs,
) -> Option<usize> {
    let n = xpt.nrows();
    let npt = xpt.ncols();

    let GeostepWs {
        distsq,
        weight,
        den,
        score,
        cal,
        ..
    } = ws;

    // PRIMA geometry.f90 L106–112: DISTSQ — distance squares from each XPT col to the "optimal
    // point", which is XOPT + D when ximproved (the new trial point), or XOPT alone otherwise.
    if ximproved {
        // PRIMA: distsq = sum((xpt - spread(xpt(:, kopt) + d, dim=2, ncopies=npt))**2, dim=1)
        for k in 0..npt {
            let mut sq = 0.0;
            for i in 0..n {
                let diff = xpt[[i, k]] - (xpt[[i, kopt]] + d[i]);
                sq += diff * diff;
            }
            distsq[k] = sq;
        }
    } else {
        // PRIMA: distsq = sum((xpt - spread(xpt(:, kopt), dim=2, ncopies=npt))**2, dim=1)
        for k in 0..npt {
            let mut sq = 0.0;
            for i in 0..n {
                let diff = xpt[[i, k]] - xpt[[i, kopt]];
                sq += diff * diff;
            }
            distsq[k] = sq;
        }
    }

    // PRIMA geometry.f90 L114: weight = max(1, distsq/rho**2)**4
    // x**4 via repeated squaring: let m = …; let m2 = m * m; m2 * m2  (gfortran lowers **4 this way)
    let rho2 = rho * rho;
    for k in 0..npt {
        let m = (distsq[k] / rho2).max(1.0);
        let m2 = m * m;
        weight[k] = m2 * m2;
    }

    // PRIMA geometry.f90 L134–143: den = calden; score = weight * den
    calden_into(kopt, bmat, d, xpt, zmat, cal, den);
    for k in 0..npt {
        score[k] = weight[k] * den[k];
    }

    // PRIMA geometry.f90 L138–140: if not improved, exclude kopt from competition
    if !ximproved {
        score[kopt] = -1.0;
    }

    // PRIMA geometry.f90 L143: NaN scores → -1 (where is_nan(score) → score = -ONE)
    for k in 0..npt {
        if score[k].is_nan() {
            score[k] = -1.0;
        }
    }

    // PRIMA geometry.f90 L145–151: decide whether to pick, then first-max scan.
    // The take-condition is: any score > 1, OR (ximproved AND any score > 0).
    let mut knew: Option<usize> = None;
    if score.iter().any(|&v| v > 1.0) || (ximproved && score.iter().any(|&v| v > 0.0)) {
        // First-max scan with strict > (ties keep first), matching PRIMA maxloc behaviour.
        let mut best = score[0];
        let mut best_k = 0;
        for k in 1..npt {
            if score[k] > best {
                best = score[k];
                best_k = k;
            }
        }
        knew = Some(best_k);
    }

    // PRIMA geometry.f90 L158–160: if ximproved and still None (all-NaN den path), fall back to
    // the farthest point. The `knew < 0` arm is impossible in the Rust representation (no negative
    // Option<usize>) — not transcribed per §5 discipline.
    if ximproved && knew.is_none() {
        let mut best = distsq[0];
        let mut best_k = 0;
        for k in 1..npt {
            if distsq[k] > best {
                best = distsq[k];
                best_k = k;
            }
        }
        knew = Some(best_k);
    }

    knew
}

/// PRIMA geometry.f90 L178 `geostep`: compute a geometry-improving step D from XOPT.
/// Writes D into `d` (length n, fully overwritten on every path).
#[expect(clippy::too_many_arguments)] // out-params mirror the Fortran intent (rust.md §5)
#[expect(clippy::too_many_lines)] // faithful port of PRIMA's 300-line geostep — rust.md §5
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA (rust.md §5)
#[expect(clippy::similar_names)] // PRIMA identifiers: xopt/xpt, slbd/subd, ilbd/iubd/isbd/ibd, mask_fixl/mask_fixu — rust.md §5
#[expect(clippy::cast_possible_wrap)] // isbd signed-1-based encoding: usize+1 fits i64 for realistic n — module header
#[expect(clippy::cast_possible_truncation)] // sign(1.0, x) as i64: value is exactly ±1.0, no truncation — module header
#[expect(clippy::cast_sign_loss)] // ibd→usize: always positive at use site (guarded by ibd<0/ibd>0) — module header
pub(crate) fn geostep(
    knew: usize,
    kopt: usize,
    bmat: &Mat,
    delbar: f64,
    sl: &[f64],
    su: &[f64],
    xpt: &Mat,
    zmat: &Mat,
    d: &mut [f64],
    ws: &mut GeostepWs,
) {
    let n = xpt.nrows();
    let npt = xpt.ncols();

    let GeostepWs {
        distsq,
        zrow_knew,
        pqlag,
        xopt,
        hm,
        glag,
        dderiv,
        stplen,
        isbd,
        xdiff,
        lfrac,
        ufrac,
        slbd_test,
        subd_test,
        vlag,
        betabd,
        predsq,
        xline,
        den_line,
        xcauchy,
        s,
        mask_free,
        xtemp,
        mask_fixl,
        mask_fixu,
        new_mask_free,
        x,
        sxpt,
        s_cauchy,
        den_cauchy,
        dxpt,
        cal,
        ..
    } = ws;

    // PRIMA geometry.f90 L302–303: pqlag = matprod(zmat, zmat(knew, :)); alpha = pqlag(knew).
    // PRIMA geometry.f90 L302: the zmat(knew, :) row extraction — a gather temp, since rows of
    // the column-major Mat are not contiguous (shared convention: row extraction).
    for j in 0..zmat.ncols() {
        zrow_knew[j] = zmat[[knew, j]];
    }
    // PRIMA: pqlag = matprod(zmat, zmat(knew, :)) — mat×vec, matprod21 loop order
    matprod21_into(zmat, zrow_knew, pqlag);
    let alpha = pqlag[knew];

    // PRIMA geometry.f90 L306: xopt = xpt(:, kopt)
    xopt.copy_from_slice(xpt.col(kopt));

    // PRIMA geometry.f90 L309: glag = bmat(:, knew) + hess_mul(xopt, xpt, pqlag) — hq absent
    hess_mul_into(xopt, xpt, pqlag, None, dxpt, hm);
    for i in 0..n {
        glag[i] = bmat[[i, knew]] + hm[i];
    }

    // PRIMA geometry.f90 L314–318: early return 1 — if glag is not finite, return a scaled
    // displacement from xopt toward xpt(:, knew).
    let glag_abs_sum: f64 = glag.iter().map(|&v| math::abs(v)).sum();
    if !glag_abs_sum.is_finite() {
        for i in 0..n {
            d[i] = xpt[[i, knew]] - xopt[i];
        }
        let scale = (0.5_f64).min(delbar / norm(d));
        for i in 0..n {
            d[i] *= scale;
        }
        return;
    }

    // PRIMA geometry.f90 L337: dderiv = matprod(glag, xpt) - inprod(glag, xopt)
    // vec×mat (matprod12), then subtract scalar inprod element-wise.
    matprod12_into(glag, xpt, dderiv);
    let glag_dot_xopt = inprod(glag, xopt);
    for k in 0..npt {
        dderiv[k] -= glag_dot_xopt;
    }

    // PRIMA geometry.f90 L338: distsq = sum((xpt - spread(xopt, dim=2, ncopies=npt))**2, dim=1)
    distsq.fill(0.0);
    for k in 0..npt {
        let mut sq = 0.0;
        for i in 0..n {
            let diff = xpt[[i, k]] - xopt[i];
            sq += diff * diff;
        }
        distsq[k] = sq;
    }

    // PRIMA geometry.f90 L339–422: per-k line loop — fill stplen (3×npt) and isbd ([i64;3] per k).
    // isbd keeps PRIMA's signed 1-based encoding (module header).
    stplen.fill(0.0);
    for e in isbd.iter_mut() {
        *e = [0i64; 3];
    }

    for k in 0..npt {
        // PRIMA geometry.f90 L344: skip kopt and NaN dderiv
        if k == kopt || dderiv[k].is_nan() {
            dderiv[k] = 0.0;
            // stplen col already zero from zeros(); isbd[k] already [0;3]
            continue;
        }

        // PRIMA geometry.f90 L351: subd = delbar / sqrt(distsq[k])
        let subd_init = delbar / math::sqrt(distsq[k]);
        let mut subd = subd_init;
        let mut slbd = -subd_init;
        let mut ilbd: i64 = 0;
        let mut iubd: i64 = 0;
        let sumin = (1.0_f64).min(subd_init); // PRIMA: sumin = min(ONE, subd)

        // PRIMA geometry.f90 L365–369: xdiff, lfrac, ufrac
        xdiff.fill(0.0);
        lfrac.fill(0.0);
        ufrac.fill(0.0);
        for i in 0..n {
            xdiff[i] = xpt[[i, k]] - xopt[i];
            // PRIMA: lfrac = sign(subd, -xdiff) — initialize to sign(subd, -xdiff[i])
            lfrac[i] = subd_init.copysign(-xdiff[i]);
            // PRIMA where (sl - xopt > -abs(xdiff) * subd) lfrac = (sl - xopt) / xdiff
            if sl[i] - xopt[i] > -math::abs(xdiff[i]) * subd_init {
                lfrac[i] = (sl[i] - xopt[i]) / xdiff[i];
            }
            // PRIMA: ufrac = sign(subd, xdiff)
            ufrac[i] = subd_init.copysign(xdiff[i]);
            // PRIMA where (su - xopt < abs(xdiff) * subd) ufrac = (su - xopt) / xdiff
            if su[i] - xopt[i] < math::abs(xdiff[i]) * subd_init {
                ufrac[i] = (su[i] - xopt[i]) / xdiff[i];
            }
        }

        // PRIMA geometry.f90 L376–386: revise slbd
        // slbd_test: init to slbd, then overwrite where xdiff > 0 with lfrac, xdiff < 0 with ufrac
        slbd_test.fill(slbd);
        for i in 0..n {
            if xdiff[i] > 0.0 {
                slbd_test[i] = lfrac[i];
            } else if xdiff[i] < 0.0 {
                slbd_test[i] = ufrac[i];
            }
        }
        // PRIMA: if (any(slbd_test > slbd)) — first masked max (mask = !is_nan)
        let any_slbd_better = slbd_test.iter().any(|&v| v > slbd);
        if any_slbd_better {
            // First-max scan over non-NaN elements (ascending index, strict >)
            let mut best_val = f64::NEG_INFINITY;
            let mut best_i: Option<usize> = None;
            for i in 0..n {
                if !slbd_test[i].is_nan() && slbd_test[i] > best_val {
                    best_val = slbd_test[i];
                    best_i = Some(i);
                }
            }
            let ilbd_pos = best_i.unwrap(); // always Some since any() passed on non-NaN elements
            slbd = slbd_test[ilbd_pos];
            // PRIMA: ilbd = -ilbd * nint(sign(ONE, xdiff(ilbd))) — signed 1-based
            ilbd = -((ilbd_pos + 1) as i64) * (1.0_f64.copysign(xdiff[ilbd_pos]) as i64);
        }

        // PRIMA geometry.f90 L389–400: revise subd
        // subd_test: init to subd, then overwrite where xdiff > 0 with ufrac, xdiff < 0 with lfrac
        subd_test.fill(subd_init);
        for i in 0..n {
            if xdiff[i] > 0.0 {
                subd_test[i] = ufrac[i];
            } else if xdiff[i] < 0.0 {
                subd_test[i] = lfrac[i];
            }
        }
        // PRIMA: if (any(subd_test < subd)) — first masked min (mask = !is_nan)
        let any_subd_better = subd_test.iter().any(|&v| v < subd_init);
        if any_subd_better {
            // First-min scan over non-NaN elements (ascending index, strict <)
            let mut best_val = f64::INFINITY;
            let mut best_i: Option<usize> = None;
            for i in 0..n {
                if !subd_test[i].is_nan() && subd_test[i] < best_val {
                    best_val = subd_test[i];
                    best_i = Some(i);
                }
            }
            let iubd_pos = best_i.unwrap(); // always Some since any() passed
            subd = sumin.max(subd_test[iubd_pos]);
            // PRIMA: iubd = iubd * nint(sign(ONE, xdiff(iubd))) — signed 1-based
            iubd = ((iubd_pos + 1) as i64) * (1.0_f64.copysign(xdiff[iubd_pos]) as i64);
        }

        // PRIMA geometry.f90 L411–421: stpm — critical point of PHI_K(t)
        let mut stpm = 0.5_f64; // default for k != knew: PHI_K(0)=0=PHI_K(1), critical at 0.5
        if k == knew {
            stpm = slbd;
            // PRIMA: if (abs(ONE - dderiv(k)) > 0) stpm = -HALF * dderiv(k) / (ONE - dderiv(k))
            if math::abs(1.0 - dderiv[k]) > 0.0 {
                stpm = -0.5 * dderiv[k] / (1.0 - dderiv[k]);
            }
        }
        stpm = slbd.max(subd.min(stpm)); // PRIMA: stpm = max(slbd, min(subd, stpm))

        stplen[[0, k]] = slbd;
        stplen[[1, k]] = subd;
        stplen[[2, k]] = stpm;
        isbd[k] = [ilbd, iubd, 0i64];
    }

    // PRIMA geometry.f90 L428–443: compute vlag (3×npt), betabd (3×npt), predsq (3×npt)

    // PRIMA: vlag = stplen * (ONE - stplen) * spread(dderiv, dim=1, ncopies=3)
    vlag.fill(0.0);
    for k in 0..npt {
        for i in 0..3 {
            vlag[[i, k]] = stplen[[i, k]] * (1.0 - stplen[[i, k]]) * dderiv[k];
        }
    }
    // PRIMA geometry.f90 L430: overwrite column knew
    // vlag(:, knew) = stplen(:, knew) * (stplen(:, knew) * (ONE - dderiv(knew)) + dderiv(knew))
    for i in 0..3 {
        let t = stplen[[i, knew]];
        vlag[[i, knew]] = t * (t * (1.0 - dderiv[knew]) + dderiv[knew]);
    }
    // PRIMA geometry.f90 L433: where (is_nan(vlag)) vlag = ZERO
    for k in 0..npt {
        for i in 0..3 {
            if vlag[[i, k]].is_nan() {
                vlag[[i, k]] = 0.0;
            }
        }
    }

    // PRIMA geometry.f90 L436: betabd = HALF * (stplen * (ONE - stplen) * spread(distsq, …))**2
    betabd.fill(0.0);
    for k in 0..npt {
        for i in 0..3 {
            let t = stplen[[i, k]] * (1.0 - stplen[[i, k]]) * distsq[k];
            betabd[[i, k]] = 0.5 * t * t;
        }
    }

    // PRIMA geometry.f90 L440: predsq = vlag²·(vlag² + alpha·betabd), then NaN → 0 — the quantity
    // (3.11) of the BOBYQA paper. As vlag⁴ + alpha·vlag²·betabd it is vlag²·σ with σ = alpha·β + vlag²
    // the squared Lagrange-update denominator (SIGMA); the step maximizing predsq best conditions
    // the model.
    predsq.fill(0.0);
    for k in 0..npt {
        for i in 0..3 {
            let v = vlag[[i, k]];
            let v2 = v * v;
            predsq[[i, k]] = v2 * (v2 + alpha * betabd[[i, k]]);
            if predsq[[i, k]].is_nan() {
                predsq[[i, k]] = 0.0;
            }
        }
    }

    // PRIMA geometry.f90 L459–461: ksqs[i] = first max over k of row i (maxloc(predsq, dim=2))
    // Then isq = first max of the three picked values; ksq = ksqs[isq].
    let mut ksqs = [0usize; 3];
    for i in 0..3 {
        let mut best = predsq[[i, 0]];
        let mut best_k = 0;
        for k in 1..npt {
            if predsq[[i, k]] > best {
                best = predsq[[i, k]];
                best_k = k;
            }
        }
        ksqs[i] = best_k;
    }
    // isq = first max of [predsq(1, ksqs(1)), predsq(2, ksqs(2)), predsq(3, ksqs(3))]
    let picked = [
        predsq[[0, ksqs[0]]],
        predsq[[1, ksqs[1]]],
        predsq[[2, ksqs[2]]],
    ];
    let mut isq = 0;
    let mut best_predsq = picked[0];
    for i in 1..3 {
        if picked[i] > best_predsq {
            best_predsq = picked[i];
            isq = i;
        }
    }
    let ksq = ksqs[isq];

    // PRIMA geometry.f90 L468–477: construct xline satisfying bounds exactly.
    let stpsiz = stplen[[isq, ksq]];
    let ibd = isbd[ksq][isq];

    // PRIMA: xline = max(sl, min(su, xopt + stpsiz * (xpt(:, ksq) - xopt)))
    for i in 0..n {
        xline[i] = (xopt[i] + stpsiz * (xpt[[i, ksq]] - xopt[i]))
            .min(su[i])
            .max(sl[i]);
    }
    // PRIMA: if (ibd < 0) xline(-ibd) = sl(-ibd)  — 1-based index in PRIMA
    if ibd < 0 {
        let idx = (-ibd) as usize - 1; // translate PRIMA's 1-based signed index
        xline[idx] = sl[idx];
    }
    // PRIMA: if (ibd > 0) xline(ibd) = su(ibd)
    if ibd > 0 {
        let idx = ibd as usize - 1; // translate PRIMA's 1-based signed index
        xline[idx] = su[idx];
    }

    // PRIMA geometry.f90 L482–483: d = xline - xopt; den_line = calden(kopt, bmat, d, xpt, zmat)
    for i in 0..n {
        d[i] = xline[i] - xopt[i];
    }
    calden_into(kopt, bmat, d, xpt, zmat, cal, den_line);

    // PRIMA geometry.f90 L496–498: early return 2 — skip Cauchy step when delbar is large
    // PRIMA geometry.f90 L496: the literal 1.0E-2 has no _RP suffix — gfortran evaluates it in
    // SINGLE precision (= 0.009999999776482582), and the oracle binary compares against that. The
    // f32 round-trip must stay bit-identical to PRIMA — a wider literal would diverge the branch.
    if delbar > f64::from(1.0e-2_f32) {
        return;
    }

    // PRIMA geometry.f90 L506–579: Cauchy step loop (uphill = 0 downhill, uphill = 1 uphill)
    let bigstp = delbar + delbar;
    xcauchy.copy_from_slice(xopt);
    let mut vlagsq_cauchy = 0.0_f64;

    for uphill in 0usize..=1 {
        // PRIMA geometry.f90 L510–512: negate glag when uphill == 1
        if uphill == 1 {
            for i in 0..n {
                glag[i] = -glag[i];
            }
        }

        // PRIMA geometry.f90 L513–521: s, mask_free, ggfree
        s.fill(0.0);
        // PRIMA: mask_free = (min(xopt - sl, glag) > 0 .or. max(xopt - su, glag) < 0)
        mask_free.fill(false);
        for i in 0..n {
            mask_free[i] =
                (xopt[i] - sl[i]).min(glag[i]) > 0.0 || (xopt[i] - su[i]).max(glag[i]) < 0.0;
        }
        for i in 0..n {
            if mask_free[i] {
                s[i] = bigstp;
            }
        }
        // PRIMA: ggfree = sum(glag(trueloc(mask_free))**2)
        let mut ggfree = 0.0_f64;
        for i in 0..n {
            if mask_free[i] {
                ggfree += glag[i] * glag[i];
            }
        }
        // PRIMA: if (ggfree <= 0 .or. is_nan(ggfree)) cycle — skip this uphill iteration
        if ggfree <= 0.0 || ggfree.is_nan() {
            continue;
        }

        // PRIMA geometry.f90 L529–548: the fix-more loop
        let mut sfixsq = 0.0_f64;
        let mut grdstp = 0.0_f64;
        xtemp.fill(0.0);
        for _k in 0..n {
            // PRIMA: resis = delbar**2 - sfixsq
            let resis = delbar * delbar - sfixsq;
            if resis <= 0.0 {
                break;
            }
            let ssqsav = sfixsq;
            grdstp = math::sqrt(resis / ggfree);
            // PRIMA: xtemp = xopt - grdstp * glag
            for i in 0..n {
                xtemp[i] = xopt[i] - grdstp * glag[i];
            }
            // PRIMA masks (three passes in source order):
            // mask_fixl = (s >= bigstp .and. xtemp <= sl)
            // mask_fixu = (s >= bigstp .and. xtemp >= su)
            // mask_free = (s >= bigstp .and. .not.(mask_fixl .or. mask_fixu))
            mask_fixl.fill(false);
            mask_fixu.fill(false);
            for i in 0..n {
                mask_fixl[i] = s[i] >= bigstp && xtemp[i] <= sl[i];
                mask_fixu[i] = s[i] >= bigstp && xtemp[i] >= su[i];
            }
            new_mask_free.fill(false);
            for i in 0..n {
                new_mask_free[i] = s[i] >= bigstp && !(mask_fixl[i] || mask_fixu[i]);
            }
            // PRIMA: s(trueloc(mask_fixl)) = sl - xopt; s(trueloc(mask_fixu)) = su - xopt
            for i in 0..n {
                if mask_fixl[i] {
                    s[i] = sl[i] - xopt[i];
                }
                if mask_fixu[i] {
                    s[i] = su[i] - xopt[i];
                }
            }
            // PRIMA: sfixsq += sum(s(trueloc(mask_fixl .or. mask_fixu))**2)
            for i in 0..n {
                if mask_fixl[i] || mask_fixu[i] {
                    sfixsq += s[i] * s[i];
                }
            }
            // PRIMA: ggfree = sum(glag(trueloc(mask_free))**2)
            ggfree = 0.0;
            for i in 0..n {
                if new_mask_free[i] {
                    ggfree += glag[i] * glag[i];
                }
            }
            mask_free.copy_from_slice(new_mask_free);
            // PRIMA: if (.not. (sfixsq > ssqsav .and. ggfree > 0)) exit
            if !(sfixsq > ssqsav && ggfree > 0.0) {
                break;
            }
        }

        // PRIMA geometry.f90 L551–557: set remaining free components of s and x. The source order
        // of the assignment blocks below is load-bearing for FP — keep it as written.
        x.fill(0.0);
        for i in 0..n {
            if glag[i] > 0.0 {
                x[i] = sl[i];
            }
        }
        for i in 0..n {
            if glag[i] <= 0.0 {
                x[i] = su[i];
            }
        }
        for i in 0..n {
            if math::abs(s[i]) <= 0.0 {
                x[i] = xopt[i];
            }
        }
        // PRIMA: xtemp = max(sl, min(su, xopt - grdstp * glag))
        for i in 0..n {
            xtemp[i] = (xopt[i] - grdstp * glag[i]).min(su[i]).max(sl[i]);
        }
        // PRIMA: x(trueloc(s >= bigstp)) = xtemp(trueloc(s >= bigstp))  — S == BIGSTP
        for i in 0..n {
            if s[i] >= bigstp {
                x[i] = xtemp[i];
            }
        }
        // PRIMA: s(trueloc(s >= bigstp)) = -grdstp * glag(trueloc(s >= bigstp))
        for i in 0..n {
            if s[i] >= bigstp {
                s[i] = -grdstp * glag[i];
            }
        }
        let gs = inprod(glag, s);

        // PRIMA geometry.f90 L562–573: curvature, scaling branch, vlagsq
        // PRIMA: sxpt = matprod(s, xpt) — vec×mat (matprod12)
        matprod12_into(s, xpt, sxpt);
        // PRIMA: curv = inprod(sxpt, pqlag * sxpt). Group each term as sxpt·(pqlag·sxpt) — the
        // left-associative (sxpt·pqlag)·sxpt can round 1 ULP differently.
        let mut curv = 0.0_f64;
        for k in 0..npt {
            let t = pqlag[k] * sxpt[k];
            curv += sxpt[k] * t;
        }
        // PRIMA: if (uphill == 1) curv = -curv
        if uphill == 1 {
            curv = -curv;
        }

        let vlagsq: f64;
        // PRIMA geometry.f90 L567: take the shortened step xopt + (−gs/curv)·s instead of the full
        // step when its scaling −gs/curv lies in (0, 1) and yields a larger |Lagrange| along S (the
        // Lagrange function is gs·t + ½·curv·t²); curv < −(1+√2)·gs is exactly that crossover.
        if curv > -gs && curv < -(1.0 + math::sqrt(2.0)) * gs {
            let scaling = -gs / curv;
            for i in 0..n {
                x[i] = (xopt[i] + scaling * s[i]).min(su[i]).max(sl[i]);
            }
            let half_gs_scaling = 0.5 * gs * scaling;
            vlagsq = half_gs_scaling * half_gs_scaling;
        } else {
            let t = gs + 0.5 * curv;
            vlagsq = t * t;
        }

        // PRIMA geometry.f90 L575–578: keep the better xcauchy
        if vlagsq > vlagsq_cauchy {
            xcauchy.copy_from_slice(x);
            vlagsq_cauchy = vlagsq;
        }
    } // end uphill loop

    // PRIMA geometry.f90 L582–589: Cauchy denominator; take if better than line step
    for i in 0..n {
        s_cauchy[i] = xcauchy[i] - xopt[i];
    }
    calden_into(kopt, bmat, s_cauchy, xpt, zmat, cal, den_cauchy);
    // PRIMA: if (den_cauchy(knew) > max(den_line(knew), ZERO) .or. is_nan(den_line(knew))) d = s
    if den_cauchy[knew] > den_line[knew].max(0.0) || den_line[knew].is_nan() {
        d.copy_from_slice(s_cauchy);
    }

    // PRIMA geometry.f90 L593–596: zero/non-finite fallback — same as early return 1
    let d_abs_sum: f64 = d.iter().map(|&v| math::abs(v)).sum();
    if d_abs_sum <= 0.0 || !d_abs_sum.is_finite() {
        for i in 0..n {
            d[i] = xpt[[i, knew]] - xopt[i];
        }
        let scale = (0.5_f64).min(delbar / norm(d));
        for i in 0..n {
            d[i] *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;
    use crate::test_support::{self, DiffStats};

    #[test]
    fn setdrop_tr_matches_prima_on_every_captured_state() {
        let corpus = test_support::load_states("setdrop_tr");
        assert!(!corpus.is_empty());
        for st in &corpus {
            let (e, x) = (&st.entry, &st.exit);
            let xpt = e.mat("xpt");
            let mut ws = GeostepWs::new(xpt.nrows(), xpt.ncols());
            let knew = setdrop_tr(
                e.usize("kopt") - 1,
                e.i64("ximproved") != 0,
                &e.mat("bmat"),
                &e.vec("d"),
                e.f64("delta"),
                e.f64("rho"),
                &xpt,
                &e.mat("zmat"),
                &mut ws,
            );
            assert_eq!(
                knew.map_or(0, |k| k + 1),
                x.usize("knew"),
                "{}: knew",
                st.problem
            );
        }
    }

    #[test]
    fn geostep_matches_prima_on_every_captured_state() {
        let corpus = test_support::load_states("geostep");
        assert!(!corpus.is_empty());
        let mut stats = DiffStats::default();
        for st in &corpus {
            let (e, x) = (&st.entry, &st.exit);
            let xpt = e.mat("xpt");
            let mut d = vec![0.0; xpt.nrows()];
            let mut ws = GeostepWs::new(xpt.nrows(), xpt.ncols());
            geostep(
                e.usize("knew") - 1,
                e.usize("kopt") - 1,
                &e.mat("bmat"),
                e.f64("delbar"),
                &e.vec("sl"),
                &e.vec("su"),
                &xpt,
                &e.mat("zmat"),
                &mut d,
                &mut ws,
            );
            stats.slice("d", &d, &x.vec("d"));
        }
        stats.report("geostep");
    }

    #[test]
    fn setdrop_tr_without_improvement_never_drops_kopt() {
        // All-equal distances make kopt competitive; its score is forced to -1 (L139).
        // Identity-ish zmat keeps den finite; assert knew != Some(kopt) over several kopt values.
        // n=2, npt=5: minimal problem, xpt = 5 columns with kopt at various positions.
        let n = 2usize;
        let npt = 5usize;
        // xpt: identity-like, all columns near origin
        let xpt = Mat::from_col_major(
            n,
            npt,
            vec![
                0.0, 0.0, // col 0
                0.5, 0.0, // col 1
                0.0, 0.5, // col 2
                -0.5, 0.0, // col 3
                0.0, -0.5, // col 4
            ],
        );
        // bmat: n x (npt + n), identity-ish in first npt cols
        let bmat = Mat::zeros(n, npt + n);
        // zmat: npt x (npt - n - 1) = 5 x 2, identity-ish
        let zmat = Mat::from_col_major(
            npt,
            npt - n - 1,
            vec![
                -5.656_854_249_492_38,
                2.828_427_124_746_19,
                0.0,
                2.828_427_124_746_19,
                0.0,
                0.0,
                0.0,
                2.828_427_124_746_19,
                -5.656_854_249_492_38,
                2.828_427_124_746_19,
            ],
        );
        let d = vec![0.1, 0.1];
        let delta = 1.0;
        let rho = 0.5;

        for kopt in 0..npt {
            let mut ws = GeostepWs::new(n, npt);
            let knew = setdrop_tr(kopt, false, &bmat, &d, delta, rho, &xpt, &zmat, &mut ws);
            assert!(
                knew != Some(kopt),
                "setdrop_tr with ximproved=false returned kopt={kopt}"
            );
        }
    }

    #[test]
    fn setdrop_tr_excludes_kopt_even_when_it_would_otherwise_win() {
        // The test above never makes kopt the competitive max, so the `!ximproved` suppression
        // (geometry.f90 L138-140, score[kopt] = -1) survives mutation. Construct a case where it is
        // load-bearing: xpt at the origin and zmat = 0 give hdiag = 0 => den = vlag^2; bmat[0,0] = 3
        // with d = [1] gives vlag[kopt] = 3*1 + 1 = 4 => den[kopt] = 16, den[others] = 0; distsq = 0
        // => weight = 1 => score[kopt] = 16 is the unique max (> 1). Without the suppression the
        // first-max scan picks kopt; with it, kopt is excluded and nothing else qualifies => None.
        let (n, npt, kopt) = (1, 3, 0);
        let mut bmat = Mat::zeros(n, npt + n);
        bmat[[0, 0]] = 3.0;
        let zmat = Mat::zeros(npt, npt - n - 1);
        let xpt = Mat::zeros(n, npt);
        let mut ws = GeostepWs::new(n, npt);
        let knew = setdrop_tr(kopt, false, &bmat, &[1.0], 1.0, 1.0, &xpt, &zmat, &mut ws);
        assert_eq!(knew, None);
    }

    #[test]
    #[expect(clippy::similar_names)] // xopt/xpt are PRIMA identifiers (rust.md §5)
    fn geostep_returns_a_nonzero_step_inside_the_bounds() {
        // Use a concrete geometry state from the corpus (booth, npt=5, first entry).
        // Verify postconditions L603–612: ||d|| > 0, sl <= xopt + d <= su elementwise.
        let corpus = test_support::load_states("geostep");
        let st = &corpus[0];
        let e = &st.entry;
        let xopt_col = e.usize("kopt") - 1;
        let xpt = e.mat("xpt");
        let sl = e.vec("sl");
        let su = e.vec("su");
        let xopt: Vec<f64> = xpt.col(xopt_col).to_vec();
        let n = xpt.nrows();

        let mut d = vec![0.0; n];
        let mut ws = GeostepWs::new(n, xpt.ncols());
        geostep(
            e.usize("knew") - 1,
            xopt_col,
            &e.mat("bmat"),
            e.f64("delbar"),
            &sl,
            &su,
            &xpt,
            &e.mat("zmat"),
            &mut d,
            &mut ws,
        );

        // postcondition: step is nonzero
        let step_norm = norm(&d);
        assert!(
            step_norm > 0.0,
            "geostep returned zero step, ||d|| = {step_norm}"
        );

        // postcondition: sl <= xopt + d <= su elementwise.
        // Tolerance is 1e-12, not 1e-10: a real bounds violation (a missing clamp) is O(delbar) ~ 0.1,
        // while `xopt[i] + d[i]` here only re-adds what geostep already clamped internally, so its only
        // error is a few ulp of re-addition rounding (~2e-15 at these magnitudes). The exact eval-point
        // invariant is guarded strictly (no slack) in tests/parity_prima.rs over the real trajectory.
        for i in 0..n {
            let xi = xopt[i] + d[i];
            assert!(
                xi >= sl[i] - 1e-12,
                "d violates lower bound at i={i}: xopt[i]+d[i]={xi} < sl[i]={}",
                sl[i]
            );
            assert!(
                xi <= su[i] + 1e-12,
                "d violates upper bound at i={i}: xopt[i]+d[i]={xi} > su[i]={}",
                su[i]
            );
        }
    }
}
