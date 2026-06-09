//! `rescue.f90` (PRIMA): the geometry-restoring RESCUE procedure of Section 5 of the BOBYQA
//! paper, plus its private helper `updateh_rsc` — design §5.
//!
//! Index convention: point/variable indices 0-based (`kopt`, `kpt`, `korig`, `kprov`), EXCEPT the
//! `ip`/`iq` values decoded from `PTSID`: those stay PRIMA's 1-based variable indices with 0 as
//! the "no direction" sentinel, because PTSID's arithmetic encoding
//! (`ptsid = ip + iq/(n+1) + sfrac`) bakes them in — translate with `- 1` only when indexing
//! (`ptsaux`, `gopt`, `hq`, `xpt` rows). History (`xhist`/`fhist`) and `iprint`/`solver`/`fmsg`/
//! `savehist` are omitted (SPEC §7.6); info codes are `consts.rs` values.
use crate::consts::{DAMAGING_ROUNDING, INFO_DFT, MAXFUN_REACHED};
use crate::linalg::{inprod, matprod12_into, matprod21_into, planerot};
use crate::mat::Mat;
use crate::math;
use crate::powalg::{hess_mul_into, setij_into};
use crate::util::{checkexit, evaluate, xinbd_into};

/// Reused scratch for `updateh_rsc` — its per-call locals (VLAG copy, HCOL, V1, V2), hoisted
/// like the [`RescueWs`] fields. Split out so `rescue` can pass its own `vlag` field (read-only)
/// and this struct (mutably) in one call.
#[derive(Debug, Clone)]
struct UpdatehRscWs {
    vlag: Vec<f64>, // npt + n — the mutated local copy of the VLAG input
    hcol: Vec<f64>, // npt + n
    v1: Vec<f64>,   // n
    v2: Vec<f64>,   // n
}

/// Reused scratch for `rescue` — PRIMA's per-call locals, hoisted to the solver workspace
/// (rust.md §4). Lifetime and contents per call are identical to the Fortran locals; only the
/// allocation site moves: every field is re-initialized at the original allocation site, per
/// call and per loop iteration where the original was in-loop. Sized for the worst case even
/// though rescue rarely runs — zero-alloc means zero, not "zero on the happy path" (M2 §4.1).
/// Field → Fortran-local map: most fields carry their PRIMA names (`xopt`/`ptsaux1`/`ptsaux2`
/// = the two PTSAUX rows/`ptsid`/`score`/`vlag`/`wmv`/`den`/`xxpt`/`pqinc`); `v` is the L251
/// HQ-update vector, `ij` the SETIJ pairs, `wmv_z`/`z_wmv_z`/`bmat_wmv`/`z_zrow` the MATPROD
/// temps, `t` the L388 BSUM accumulator, `xnew` the refill point, `x`/`xmod` the evaluation
/// point and evaluate's MODERATEX copy, `zrow` the ZMAT(KPT, :) row-extraction temp,
/// `xpt_col` the XPT(:, KPT)/XPT(:, KOPT) column copies, `shift` the L578 `HESS_MUL` result,
/// `dxpt` `hess_mul` scratch, `rsc` `updateh_rsc`'s locals.
#[derive(Debug, Clone)]
pub(crate) struct RescueWs {
    xopt: Vec<f64>,          // n
    v: Vec<f64>,             // n
    ptsaux1: Vec<f64>,       // n
    ptsaux2: Vec<f64>,       // n
    ptsid: Vec<f64>,         // npt
    ij: Vec<(usize, usize)>, // capacity max(0, npt - 2n - 1)
    score: Vec<f64>,         // npt
    vlag: Vec<f64>,          // npt + n
    wmv: Vec<f64>,           // npt + n
    wmv_z: Vec<f64>,         // npt - n - 1
    z_wmv_z: Vec<f64>,       // npt
    bmat_wmv: Vec<f64>,      // n
    t: Vec<f64>,             // n
    den: Vec<f64>,           // npt
    xnew: Vec<f64>,          // n
    x: Vec<f64>,             // n
    xmod: Vec<f64>,          // n
    xxpt: Vec<f64>,          // npt
    pq_xxpt: Vec<f64>,       // npt
    zrow: Vec<f64>,          // npt - n - 1
    z_zrow: Vec<f64>,        // npt
    pqinc: Vec<f64>,         // npt
    xpt_col: Vec<f64>,       // n
    shift: Vec<f64>,         // n
    dxpt: Vec<f64>,          // npt
    rsc: UpdatehRscWs,
}

impl RescueWs {
    pub(crate) fn new(n: usize, npt: usize) -> Self {
        Self {
            xopt: vec![0.0; n],
            v: vec![0.0; n],
            ptsaux1: vec![0.0; n],
            ptsaux2: vec![0.0; n],
            ptsid: vec![0.0; npt],
            ij: Vec::with_capacity(npt.saturating_sub(2 * n + 1)),
            score: vec![0.0; npt],
            vlag: vec![0.0; npt + n],
            wmv: vec![0.0; npt + n],
            wmv_z: vec![0.0; npt - n - 1],
            z_wmv_z: vec![0.0; npt],
            bmat_wmv: vec![0.0; n],
            t: vec![0.0; n],
            den: vec![0.0; npt],
            xnew: vec![0.0; n],
            x: vec![0.0; n],
            xmod: vec![0.0; n],
            xxpt: vec![0.0; npt],
            pq_xxpt: vec![0.0; npt],
            zrow: vec![0.0; npt - n - 1],
            z_zrow: vec![0.0; npt],
            pqinc: vec![0.0; npt],
            xpt_col: vec![0.0; n],
            shift: vec![0.0; n],
            dxpt: vec![0.0; npt],
            rsc: UpdatehRscWs {
                vlag: vec![0.0; npt + n],
                hcol: vec![0.0; npt + n],
                v1: vec![0.0; n],
                v2: vec![0.0; n],
            },
        }
    }
}

/// PRIMA rescue.f90 L32 `rescue`: replace a few interpolation points by new ones to improve the
/// geometry of the set and the conditioning of the interpolation system. Returns `info`.
#[expect(clippy::too_many_arguments)] // out-params mirror the Fortran intent (rust.md §5)
#[expect(clippy::too_many_lines)] // one Fortran subroutine body, transcribed in place (rust.md §5)
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA (rust.md §5)
#[expect(clippy::cast_possible_truncation)] // floor(ptsid) as usize: value is a small exact integer — module header
#[expect(clippy::cast_sign_loss)] // floor(ptsid) as usize: ptsid >= 0 at every decode site — module header
pub(crate) fn rescue<F: FnMut(&[f64]) -> f64>(
    calfun: &mut F,
    maxfun: usize,
    delta: f64,
    ftarget: f64,
    xl: &[f64],
    xu: &[f64],
    kopt: &mut usize,
    nf: &mut usize,
    fval: &mut [f64],
    gopt: &mut [f64],
    hq: &mut Mat,
    pq: &mut [f64],
    sl: &mut [f64],
    su: &mut [f64],
    xbase: &mut [f64],
    xpt: &mut Mat,
    bmat: &mut Mat,
    zmat: &mut Mat,
    ws: &mut RescueWs,
) -> i32 {
    let n = xpt.nrows();
    let npt = xpt.ncols();

    let RescueWs {
        xopt,
        v,
        ptsaux1,
        ptsaux2,
        ptsid,
        ij,
        score,
        vlag,
        wmv,
        wmv_z,
        z_wmv_z,
        bmat_wmv,
        t,
        den,
        xnew,
        x,
        xmod,
        xxpt,
        pq_xxpt,
        zrow,
        z_zrow,
        pqinc,
        xpt_col,
        shift,
        dxpt,
        rsc,
    } = ws;

    // PRIMA rescue.f90 L230: info starts at the default.
    let mut info = INFO_DFT;

    // PRIMA L234-239: do nothing if NF already reaches its upper bound. Set BMAT and ZMAT before
    // returning, though they will not be used.
    if *nf >= maxfun {
        bmat.fill(0.0);
        zmat.fill(0.0);
        return MAXFUN_REACHED;
    }

    // PRIMA L242-247: shift the interpolation points so that XOPT becomes the origin.
    xopt.copy_from_slice(xpt.col(*kopt));
    for i in 0..n {
        sl[i] = (sl[i] - xopt[i]).min(0.0);
        su[i] = (su[i] - xopt[i]).max(0.0);
        // PRIMA L245: xbase = min(max(xl, xbase + xopt), xu).
        xbase[i] = xl[i].max(xbase[i] + xopt[i]).min(xu[i]);
    }
    // PRIMA L246: xpt = xpt - spread(xopt, ...).
    for k in 0..npt {
        let col = xpt.col_mut(k);
        for i in 0..n {
            col[i] -= xopt[i];
        }
    }
    // PRIMA L247: xpt(:, kopt) = 0.
    xpt.col_mut(*kopt).fill(0.0);

    // PRIMA L251-252: update HQ so that HQ and PQ define the second derivatives of the model after
    // XBASE has been shifted. v = matprod(xpt, pq) + HALF * sum(pq) * xopt.
    let mut pqsum = 0.0;
    for &p in pq.iter() {
        pqsum += p;
    }
    let halfsum = 0.5 * pqsum;
    matprod21_into(xpt, pq, v);
    for i in 0..n {
        v[i] += halfsum * xopt[i];
    }
    crate::linalg::r2update(hq, 1.0, xopt, v);

    // PRIMA L255-260: set the elements of PTSAUX (rows 1/2 -> ptsaux1/ptsaux2).
    ptsaux1.fill(0.0);
    ptsaux2.fill(0.0);
    for i in 0..n {
        ptsaux1[i] = delta.min(su[i]);
        ptsaux2[i] = (-delta).max(sl[i]);
    }
    // PRIMA L257-258: swap rows where ptsaux1 + ptsaux2 < 0.
    for i in 0..n {
        if ptsaux1[i] + ptsaux2[i] < 0.0 {
            std::mem::swap(&mut ptsaux1[i], &mut ptsaux2[i]);
        }
    }
    // PRIMA L259-260: where |ptsaux2| < HALF*|ptsaux1|, set ptsaux2 = HALF*ptsaux1.
    for i in 0..n {
        if math::abs(ptsaux2[i]) < 0.5 * math::abs(ptsaux1[i]) {
            ptsaux2[i] = 0.5 * ptsaux1[i];
        }
    }

    // PRIMA L264-284: set the identifiers of the artificial interpolation points along a coordinate
    // direction from XOPT, and the corresponding nonzero elements of BMAT and ZMAT.
    let sfrac = 0.5 / (n as f64 + 1.0);
    ptsid.fill(0.0);
    ptsid[0] = sfrac;
    bmat.fill(0.0);
    zmat.fill(0.0);
    for k in 0..n {
        // PRIMA L269: ptsid(k+1) = real(k) + sfrac (1-based k = k+1 here).
        ptsid[k + 1] = (k + 1) as f64 + sfrac;
        // PRIMA L270: 1-based `k <= npt-n-1`; kept as `k+1 <= ...` to mirror the Fortran condition.
        #[expect(clippy::int_plus_one)] // faithful to PRIMA's 1-based bound (rust.md §5)
        if k + 1 <= npt - n - 1 {
            // PRIMA L271: ptsid(k+n+1) = real(k)/real(n+1) + sfrac.
            ptsid[k + n + 1] = (k + 1) as f64 / (n as f64 + 1.0) + sfrac;
            // PRIMA L272-278.
            let temp = 1.0 / (ptsaux1[k] - ptsaux2[k]);
            bmat[[k, k + 1]] = -temp + 1.0 / ptsaux1[k];
            bmat[[k, k + n + 1]] = temp + 1.0 / ptsaux2[k];
            bmat[[k, 0]] = -bmat[[k, k + 1]] - bmat[[k, k + n + 1]];
            zmat[[0, k]] = math::sqrt(2.0) / math::abs(ptsaux1[k] * ptsaux2[k]);
            zmat[[k + 1, k]] = zmat[[0, k]] * ptsaux2[k] * temp;
            zmat[[k + n + 1, k]] = -zmat[[0, k]] * ptsaux1[k] * temp;
        } else {
            // PRIMA L280-282.
            bmat[[k, 0]] = -1.0 / ptsaux1[k];
            bmat[[k, k + 1]] = 1.0 / ptsaux1[k];
            bmat[[k, k + npt]] = -0.5 * ptsaux1[k] * ptsaux1[k];
        }
    }

    // PRIMA L287-295: set any remaining identifiers with their nonzero elements of ZMAT.
    // setij returns 0-based pairs; PTSID's arithmetic encoding needs PRIMA's 1-based ip/iq with 0
    // as the "no direction" sentinel, so add 1. (Mirrors inith's identical pattern, initialize.rs
    // L300-306: Fortran ZMAT rows ip+1/iq+1 1-based == ip/iq 0-based.)
    setij_into(n, npt, ij);
    for k in (2 * n + 1)..npt {
        let (i0, j0) = ij[k - (2 * n + 1)];
        let ip = i0 + 1; // 1-based
        let iq = j0 + 1; // 1-based
        ptsid[k] = ip as f64 + iq as f64 / (n as f64 + 1.0) + sfrac;
        let temp = 1.0 / (ptsaux1[ip - 1] * ptsaux1[iq - 1]);
        // PRIMA L293: zmat([1, k], k-n-1) = temp.
        zmat[[0, k - n - 1]] = temp;
        zmat[[k, k - n - 1]] = temp;
        // PRIMA L294: zmat([ip+1, iq+1], k-n-1) = -temp (1-based rows == ip/iq 0-based).
        zmat[[ip, k - n - 1]] = -temp;
        zmat[[iq, k - n - 1]] = -temp;
    }

    // PRIMA L300-305: exchange the 1st and the KOPT-th provisional points so that, after, the
    // KOPT-th provisional point is the zero vector (= the KOPT-th original point post-shift).
    if *kopt != 0 {
        // PRIMA L301: bmat(:, [1, kopt]) = bmat(:, [kopt, 1]).
        for i in 0..n {
            let tmp = bmat[[i, 0]];
            bmat[[i, 0]] = bmat[[i, *kopt]];
            bmat[[i, *kopt]] = tmp;
        }
        // PRIMA L302: zmat([1, kopt], :) = zmat([kopt, 1], :).
        for j in 0..zmat.ncols() {
            let tmp = zmat[[0, j]];
            zmat[[0, j]] = zmat[[*kopt, j]];
            zmat[[*kopt, j]] = tmp;
        }
    }
    // PRIMA L304-305: these two sit outside the KOPT /= 1 guard.
    ptsid[0] = ptsid[*kopt];
    ptsid[*kopt] = 0.0;

    // PRIMA L314-317: SCORE = sqrt(sum(xpt**2, dim=1)) (the paper variant); SCORE(KOPT) = 0;
    // SCOREINC = maxval(score).
    score.fill(0.0);
    for k in 0..npt {
        let mut s = 0.0;
        for &v in xpt.col(k) {
            s += v * v;
        }
        score[k] = math::sqrt(s);
    }
    score[*kopt] = 0.0;
    let mut scoreinc = 0.0_f64;
    for &s in score.iter() {
        scoreinc = scoreinc.max(s);
    }

    // PRIMA L320: NPROV is the number of provisional points not yet replaced.
    let mut nprov = npt - 1;

    // PRIMA L329-451: the main reinstatement loop, at most NPT^2 iterations. The Fortran's
    // integer-overflow guard on NPT**2 (L329: min with 10**range) is omitted — usize NPT^2
    // cannot overflow in the low-n target domain.
    let maxiter = npt * npt;
    vlag.fill(0.0);
    for _iter in 0..maxiter {
        // PRIMA L334: exit when all scores nonpositive, or only one provisional point left.
        if score.iter().all(|&s| s <= 0.0) || nprov <= 1 {
            break;
        }

        // PRIMA L340: KORIG = minloc(score, mask=(score > 0)) — first masked minimum.
        let mut korig = 0;
        let mut found = false;
        for k in 0..npt {
            if score[k] > 0.0 && (!found || score[k] < score[korig]) {
                korig = k;
                found = true;
            }
        }
        // The loop guard above confirmed a positive score exists, so the masked minloc cannot
        // come up empty — this is what licenses omitting the Fortran KNEW <= 0 guard in
        // updateh_rsc (an empty minloc would leave korig = 0, a wrong-but-valid index).
        debug_assert!(found, "masked minloc found no positive score");

        // PRIMA L356-377: form WMV = W - V for XPT(:, KORIG).
        wmv.fill(0.0);
        for k in 0..npt {
            if k == *kopt {
                // PRIMA L358: wmv(k) = 0.
                wmv[k] = 0.0;
            } else if ptsid[k] <= 0.0 {
                // PRIMA L360: wmv(k) = inprod(xpt(:, korig), xpt(:, k)).
                wmv[k] = inprod(xpt.col(korig), xpt.col(k));
            } else {
                // PRIMA L362-363: decode 1-based ip/iq (0 = none).
                let ip = math::floor(ptsid[k]) as usize;
                let iq = math::floor((n as f64 + 1.0) * ptsid[k] - ((n + 1) * ip) as f64) as usize;
                if ip > 0 && iq > 0 {
                    wmv[k] = xpt[[ip - 1, korig]] * ptsaux1[ip - 1]
                        + xpt[[iq - 1, korig]] * ptsaux1[iq - 1];
                } else if ip > 0 {
                    wmv[k] = xpt[[ip - 1, korig]] * ptsaux1[ip - 1];
                } else if iq > 0 {
                    wmv[k] = xpt[[iq - 1, korig]] * ptsaux2[iq - 1];
                } else {
                    wmv[k] = 0.0;
                }
            }
            // PRIMA L375: wmv(k) = HALF * wmv(k) * wmv(k).
            wmv[k] = 0.5 * wmv[k] * wmv[k];
        }
        // PRIMA L377: wmv(npt+1:npt+n) = xpt(:, korig).
        for i in 0..n {
            wmv[npt + i] = xpt[[i, korig]];
        }

        // PRIMA L380: vlag(1:npt) = matprod(zmat, matprod(wmv(1:npt), zmat))
        //                          + matprod(wmv(npt+1:npt+n), bmat(:, 1:npt)).
        matprod12_into(&wmv[..npt], zmat, wmv_z); // matprod(wmv(1:npt), zmat)
        matprod21_into(zmat, wmv_z, z_wmv_z); // matprod(zmat, ...)
        for k in 0..npt {
            // bmat(:, 1:npt) section: column k is bmat.col(k), top n rows.
            vlag[k] = z_wmv_z[k] + inprod(&wmv[npt..npt + n], bmat.col(k));
        }
        // PRIMA L381: vlag(npt+1:npt+n) = matprod(bmat, wmv(1:npt+n)).
        matprod21_into(bmat, wmv, bmat_wmv);
        vlag[npt..npt + n].copy_from_slice(bmat_wmv);

        // PRIMA L388: bsum = inprod(wmv(1:n), matprod(bmat(:, 1:npt), wmv(1:npt)) + matprod(bmat, wmv)).
        // N.B. wmv(1:n) is transcribed literally — it looks asymmetric but is exactly PRIMA.
        t.fill(0.0);
        // matprod(bmat(:, 1:npt), wmv(1:npt)): bmat section first npt columns.
        for j in 0..npt {
            let bj = bmat.col(j);
            for i in 0..n {
                t[i] += bj[i] * wmv[j];
            }
        }
        // + matprod(bmat, wmv): whole bmat, full-length wmv.
        for i in 0..n {
            t[i] += bmat_wmv[i];
        }
        let bsum = inprod(&wmv[..n], t);
        // PRIMA L389: beta = HALF*sum(xpt(:, korig)**2)**2 - sum(matprod(wmv(1:npt), zmat)**2) - bsum.
        let mut s = 0.0;
        for &x in xpt.col(korig) {
            s += x * x;
        }
        let mut wmvz_sq = 0.0;
        for &w in wmv_z.iter() {
            wmvz_sq += w * w;
        }
        let beta = 0.5 * (s * s) - wmvz_sq - bsum;

        // PRIMA L392: vlag(kopt) = vlag(kopt) + ONE.
        vlag[*kopt] += 1.0;

        // PRIMA L396-398: for all K with PTSID(K) > 0, DEN(K) = hdiag(k)*beta + vlag(k)**2.
        den.fill(0.0);
        for k in 0..npt {
            if ptsid[k] > 0.0 {
                let mut hdiag = 0.0;
                for j in 0..zmat.ncols() {
                    let z = zmat[[k, j]];
                    hdiag += z * z;
                }
                den[k] = hdiag * beta + vlag[k] * vlag[k];
            }
        }

        // PRIMA L418-423: the rejection test. vmax = maxval(vlag(1:npt)**2); reject when
        // .not. (is_finite(sum(abs(vlag))) .and. any(den > 5e-2 * vmax)).
        let vlag_abs_sum = vlag.iter().map(|v| math::abs(*v)).sum::<f64>();
        let mut vmax = 0.0_f64;
        for k in 0..npt {
            vmax = vmax.max(vlag[k] * vlag[k]);
        }
        if !(vlag_abs_sum.is_finite() && den.iter().any(|&d| d > 5.0e-2 * vmax)) {
            // PRIMA L421-422.
            score[korig] = -score[korig] - scoreinc;
            continue;
        }
        // PRIMA L424: KPROV = maxloc(den, mask=(.not. is_nan(den))) — first masked maximum.
        let mut kprov = 0;
        let mut kfound = false;
        for k in 0..npt {
            if !den[k].is_nan() && (!kfound || den[k] > den[kprov]) {
                kprov = k;
                kfound = true;
            }
        }

        // PRIMA L429-433: exchange the KPROV-th and KORIG-th provisional points.
        if kprov != korig {
            for i in 0..n {
                let tmp = bmat[[i, kprov]];
                bmat[[i, kprov]] = bmat[[i, korig]];
                bmat[[i, korig]] = tmp;
            }
            for j in 0..zmat.ncols() {
                let tmp = zmat[[kprov, j]];
                zmat[[kprov, j]] = zmat[[korig, j]];
                zmat[[korig, j]] = tmp;
            }
            vlag.swap(kprov, korig);
        }
        // PRIMA L434: ptsid(kprov) = ptsid(korig).
        ptsid[kprov] = ptsid[korig];
        // PRIMA L438: ptsid(korig) = 0.
        ptsid[korig] = 0.0;
        // PRIMA L440: score(korig) = 0.
        score[korig] = 0.0;
        // PRIMA L443: score = abs(score).
        for s in score.iter_mut() {
            *s = math::abs(*s);
        }

        // PRIMA L447: update BMAT and ZMAT so that the KORIG-th original point replaces the
        // KORIG-th provisional point (the Fortran call passes no info).
        let _ = updateh_rsc(korig, beta, vlag, bmat, zmat, rsc);

        // PRIMA L450: NPROV decreases by 1.
        nprov -= 1;
    }

    // PRIMA L459-574: refill phase.
    let kbase = *kopt;
    let fbase = fval[*kopt];
    if nprov > 0 {
        for kpt in 0..npt {
            // PRIMA L463-465: skip points with PTSID(KPT) <= 0.
            if ptsid[kpt] <= 0.0 {
                continue;
            }

            // PRIMA L469-470: absorb PQ(KPT)*XPT(:, KPT)*XPT(:, KPT)^T into HQ; PQ(KPT) = 0.
            // Copy the column first — xpt is borrowed mutably later in the iteration.
            xpt_col.copy_from_slice(xpt.col(kpt));
            crate::linalg::r1update(hq, pq[kpt], xpt_col);
            pq[kpt] = 0.0;

            // PRIMA L472-473: decode 1-based ip/iq.
            let ip = math::floor(ptsid[kpt]) as usize;
            let iq = math::floor((n as f64 + 1.0) * ptsid[kpt] - ((n + 1) * ip) as f64) as usize;

            // PRIMA L477-491: build XNEW with at most two nonzeros XP/XQ at IP/IQ.
            let mut xp = 0.0;
            let mut xq = 0.0;
            xnew.fill(0.0);
            if ip > 0 && iq > 0 {
                xp = ptsaux1[ip - 1];
                xnew[ip - 1] = xp;
                xq = ptsaux1[iq - 1];
                xnew[iq - 1] = xq;
            } else if ip > 0 {
                xp = ptsaux1[ip - 1];
                xnew[ip - 1] = xp;
            } else if iq > 0 {
                xq = ptsaux2[iq - 1];
                xnew[iq - 1] = xq;
            }

            // PRIMA L502-505: skip the new point if too close to XPT(:, KPT) or non-finite.
            let mut diff_abs_sum = 0.0;
            let mut xnew_abs_sum = 0.0;
            for i in 0..n {
                diff_abs_sum += math::abs(xnew[i] - xpt[[i, kpt]]);
                xnew_abs_sum += math::abs(xnew[i]);
            }
            // PRIMA rescue.f90 L502: the literal 1.0E-2 has no _RP suffix — gfortran evaluates it in
            // SINGLE precision (= 0.009999999776482582), and the oracle binary compares against that.
            if diff_abs_sum <= f64::from(1.0e-2_f32) * delta || !xnew_abs_sum.is_finite() {
                continue;
            }
            xpt.col_mut(kpt).copy_from_slice(xnew);

            // PRIMA L510-512: evaluate F at the new interpolation point.
            xinbd_into(xbase, xpt.col(kpt), xl, xu, sl, su, x);
            let f = evaluate(calfun, x, xmod);
            *nf += 1;
            // PRIMA L515-517: fmsg/savehist omitted (SPEC §7.6).

            // PRIMA L520-523: update FVAL and KOPT.
            fval[kpt] = f;
            if f < fval[*kopt] {
                *kopt = kpt;
            }

            // PRIMA L526-530: check whether to exit.
            let subinfo = checkexit(maxfun, *nf, f, ftarget, x);
            if subinfo != INFO_DFT {
                info = subinfo;
                break;
            }

            // PRIMA L534-547: set VQUAD to the current model at the new XPT(:, KPT).
            let mut vquad = fbase;
            xxpt.fill(0.0);
            if ip > 0 && iq > 0 {
                vquad += xp * (gopt[ip - 1] + 0.5 * xp * hq[[ip - 1, ip - 1]]);
                vquad += xq * (gopt[iq - 1] + 0.5 * xq * hq[[iq - 1, iq - 1]]);
                vquad += xp * xq * hq[[ip - 1, iq - 1]];
                // PRIMA L539: xxpt = xp*xpt(ip, :) + xq*xpt(iq, :) (row extraction).
                for k in 0..npt {
                    xxpt[k] = xp * xpt[[ip - 1, k]] + xq * xpt[[iq - 1, k]];
                }
            } else if ip > 0 {
                vquad += xp * (gopt[ip - 1] + 0.5 * xp * hq[[ip - 1, ip - 1]]);
                for k in 0..npt {
                    xxpt[k] = xp * xpt[[ip - 1, k]];
                }
            } else if iq > 0 {
                vquad += xq * (gopt[iq - 1] + 0.5 * xq * hq[[iq - 1, iq - 1]]);
                for k in 0..npt {
                    xxpt[k] = xq * xpt[[iq - 1, k]];
                }
            }
            // PRIMA L547: vquad = vquad + HALF * inprod(xxpt, pq * xxpt).
            pq_xxpt.fill(0.0);
            for k in 0..npt {
                pq_xxpt[k] = pq[k] * xxpt[k];
            }
            vquad += 0.5 * inprod(xxpt, pq_xxpt);

            // PRIMA L551-553: update the quadratic model.
            let moderr = f - vquad;
            for i in 0..n {
                gopt[i] += moderr * bmat[[i, kpt]];
            }
            // PRIMA L553: pqinc = moderr * matprod(zmat, zmat(kpt, :)).
            zrow.fill(0.0);
            for j in 0..zmat.ncols() {
                zrow[j] = zmat[[kpt, j]];
            }
            matprod21_into(zmat, zrow, z_zrow);
            pqinc.fill(0.0);
            for k in 0..npt {
                pqinc[k] = moderr * z_zrow[k];
            }
            // PRIMA L554: pq(trueloc(ptsid <= 0)) += pqinc(trueloc(ptsid <= 0)).
            for k in 0..npt {
                if ptsid[k] <= 0.0 {
                    pq[k] += pqinc[k];
                }
            }
            // PRIMA L555-571: the HQ loop over K with the ip/iq decode.
            for k in 0..npt {
                if ptsid[k] <= 0.0 {
                    continue;
                }
                let ipk = math::floor(ptsid[k]) as usize;
                let iqk =
                    math::floor((n as f64 + 1.0) * ptsid[k] - ((n + 1) * ipk) as f64) as usize;
                if ipk > 0 && iqk > 0 {
                    // PRIMA L566-568: explicit parens group the squared terms as
                    // pqinc * (ptsaux*ptsaux), mirroring PRIMA's `**2`; the IP*IQ cross term is
                    // genuinely left-associative.
                    hq[[ipk - 1, ipk - 1]] += pqinc[k] * (ptsaux1[ipk - 1] * ptsaux1[ipk - 1]);
                    hq[[iqk - 1, iqk - 1]] += pqinc[k] * (ptsaux1[iqk - 1] * ptsaux1[iqk - 1]);
                    hq[[ipk - 1, iqk - 1]] += pqinc[k] * ptsaux1[ipk - 1] * ptsaux1[iqk - 1];
                    hq[[iqk - 1, ipk - 1]] = hq[[ipk - 1, iqk - 1]];
                } else if ipk > 0 {
                    // PRIMA L571: pqinc * (ptsaux**2).
                    hq[[ipk - 1, ipk - 1]] += pqinc[k] * (ptsaux1[ipk - 1] * ptsaux1[ipk - 1]);
                } else if iqk > 0 {
                    // PRIMA L573: pqinc * (ptsaux**2).
                    hq[[iqk - 1, iqk - 1]] += pqinc[k] * (ptsaux2[iqk - 1] * ptsaux2[iqk - 1]);
                }
            }
            // PRIMA L572: ptsid(kpt) = 0.
            ptsid[kpt] = 0.0;
        }
    }

    // PRIMA L577-579: update GOPT if KOPT changed.
    if *kopt != kbase {
        xpt_col.copy_from_slice(xpt.col(*kopt));
        hess_mul_into(xpt_col, xpt, pq, Some(&*hq), dxpt, shift);
        for i in 0..n {
            gopt[i] += shift[i];
        }
    }

    info
}

/// PRIMA rescue.f90 L639 `updateh_rsc` — rescue-private variant of UPDATEH driven by a
/// precomputed (BETA, VLAG). KNEW is 0-based; the Fortran's KNEW <= 0 guard (L723) is omitted:
/// the only call site (rescue, L447) passes `korig`, the result of a masked minloc that only
/// runs after the loop guard has confirmed at least one `score > 0` — so `korig` is always a
/// valid point index, checked by the `debug_assert!(found)` at that minloc.
fn updateh_rsc(
    knew: usize,
    beta: f64,
    vlag_in: &[f64],
    bmat: &mut Mat,
    zmat: &mut Mat,
    ws: &mut UpdatehRscWs,
) -> i32 {
    let n = bmat.nrows();
    let npt = bmat.ncols() - n;

    let UpdatehRscWs { vlag, hcol, v1, v2 } = ws;

    // PRIMA L719: info defaults to INFO_DFT; the KNEW <= 0 guard (L723) is omitted — korig is
    // always a valid index (the loop guard ensures a score > 0 exists before the minloc search).

    // PRIMA L728-732: read VLAG, TAU, and DENOM (computed before the rotations).
    vlag.copy_from_slice(vlag_in);
    let tau = vlag[knew];
    // PRIMA L732: denom = sum(zmat(knew, :)**2) * beta + tau**2.
    let mut zknew_sq = 0.0;
    for j in 0..zmat.ncols() {
        let z = zmat[[knew, j]];
        zknew_sq += z * z;
    }
    let denom = zknew_sq * beta + tau * tau;

    // PRIMA L737-742: damaging-rounding guard — VLAG/BETA finiteness and DENOM > 0.
    let vlag_abs_sum = vlag.iter().map(|v| math::abs(*v)).sum::<f64>();
    if !((vlag_abs_sum + math::abs(beta)).is_finite() && denom > 0.0) {
        return DAMAGING_ROUNDING;
    }

    // PRIMA L745: vlag(knew) = vlag(knew) - ONE.
    vlag[knew] -= 1.0;

    // PRIMA L749-755: Givens rotations putting zeros in the KNEW-th row of ZMAT.
    let nz = zmat.ncols();
    // PRIMA rescue.f90 L749: DO J = 2, NPT-N-1 — 0-based j in 1..nz.
    for j in 1..nz {
        // PRIMA L750: threshold uses maxval(abs(zmat)) — re-evaluated each iteration because ZMAT
        // changes; NOT hoisted (matches update.rs's updateh; Fortran re-evaluates inside the loop).
        let zmaxabs = zmat
            .data()
            .iter()
            .map(|v| math::abs(*v))
            .fold(0.0_f64, f64::max);
        // PRIMA rescue.f90 L750: the literal 1.0E-20 has no _RP suffix — gfortran evaluates it in
        // SINGLE precision (= 9.999999682655225e-21), and the oracle binary compares against that.
        // update.rs::updateh carries the identical sentinel — change together.
        if math::abs(zmat[[knew, j]]) > f64::from(1.0e-20_f32) * zmaxabs {
            // PRIMA L751: grot = planerot(zmat(knew, [1, j])).
            let grot = planerot([zmat[[knew, 0]], zmat[[knew, j]]]);
            // PRIMA L752: zmat(:, [1, j]) = matprod(zmat(:, [1, j]), transpose(grot)).
            // grot = [[c, s], [-s, c]]; transpose(grot) = [[c, -s], [s, c]].
            // new col0 = c*col0 + (-s)*colj ... transpose means: [a, b]*G^T where
            // G^T row0 = [c, -s], row1 = [s, c]. result(:,0) = a*c + b*s; result(:,1) = a*(-s) + b*c.
            for i in 0..zmat.nrows() {
                let a = zmat[[i, 0]];
                let b = zmat[[i, j]];
                zmat[[i, 0]] = a * grot[0][0] + b * grot[0][1];
                zmat[[i, j]] = a * grot[1][0] + b * grot[1][1];
            }
        }
        // PRIMA L754: zmat(knew, j) = 0.
        zmat[[knew, j]] = 0.0;
    }

    // PRIMA L757-759: put the KNEW-th column of the unupdated H into HCOL.
    hcol.fill(0.0);
    let zknew0 = zmat[[knew, 0]];
    for k in 0..npt {
        hcol[k] = zknew0 * zmat[[k, 0]];
    }
    for i in 0..n {
        hcol[npt + i] = bmat[[i, knew]];
    }

    // PRIMA L762-765: complete the updating of ZMAT.
    let sqrtdn = math::sqrt(denom);
    let zknew1 = zmat[[knew, 0]] / sqrtdn;
    // PRIMA L764: zmat(:, 1) = (tau/sqrtdn)*zmat(:, 1) - zknew1*vlag(1:npt). Hoist the scalar.
    let tau_sqrtdn = tau / sqrtdn;
    for k in 0..npt {
        zmat[[k, 0]] = tau_sqrtdn * zmat[[k, 0]] - zknew1 * vlag[k];
    }
    // PRIMA L765: zmat(knew, 1) = zknew1 (the line Powell's code lacks).
    zmat[[knew, 0]] = zknew1;

    // PRIMA L768-771: update BMAT — the last N rows of (4.9).
    let alpha = hcol[knew];
    v1.fill(0.0);
    v2.fill(0.0);
    for i in 0..n {
        v1[i] = (alpha * vlag[npt + i] - tau * hcol[npt + i]) / denom;
        v2[i] = (-beta * hcol[npt + i] - tau * vlag[npt + i]) / denom;
    }
    // PRIMA L771: bmat = bmat + outprod(v1, vlag) + outprod(v2, hcol). The Fortran array sum
    // (bmat + outprod1) + outprod2 groups left-to-right per element, so keep that grouping rather
    // than fusing the two products.
    for j in 0..(npt + n) {
        for i in 0..n {
            bmat[[i, j]] = bmat[[i, j]] + v1[i] * vlag[j] + v2[i] * hcol[j];
        }
    }
    // PRIMA L774: symmetrize(bmat(:, npt+1:npt+n)) — in place on the trailing n x n section:
    // copy the lower triangle to the upper in linalg::symmetrize's exact assignment order
    // (the section-copy temporary is dropped; pure copies, FP-identical; mirrors
    // update.rs::updateh — change together).
    for j in 0..n {
        for i in 0..j {
            bmat[[i, npt + j]] = bmat[[j, npt + i]];
        }
    }

    INFO_DFT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;
    use crate::test_support::{self, DiffStats};

    #[test]
    #[expect(clippy::similar_names)] // states/stats are conventional diff-test locals
    fn rescue_matches_prima_on_every_captured_state() {
        let states = test_support::load_states("rescue");
        assert!(!states.is_empty());
        let mut stats = DiffStats::default();
        for st in &states {
            let (e, x) = (&st.entry, &st.exit);
            let f = test_support::objective(&st.problem);
            let mut kopt = e.usize("kopt") - 1;
            let mut nf = e.usize("nf");
            let (mut fval, mut gopt, mut pq) = (e.vec("fval"), e.vec("gopt"), e.vec("pq"));
            let (mut sl, mut su, mut xbase) = (e.vec("sl"), e.vec("su"), e.vec("xbase"));
            let (mut hq, mut xpt) = (e.mat("hq"), e.mat("xpt"));
            let (mut bmat, mut zmat) = (e.mat("bmat"), e.mat("zmat"));
            let mut ws = RescueWs::new(xpt.nrows(), xpt.ncols());
            let info = rescue(
                &mut |p: &[f64]| f(p),
                e.usize("maxfun"),
                e.f64("delta"),
                e.f64("ftarget"),
                &e.vec("xl"),
                &e.vec("xu"),
                &mut kopt,
                &mut nf,
                &mut fval,
                &mut gopt,
                &mut hq,
                &mut pq,
                &mut sl,
                &mut su,
                &mut xbase,
                &mut xpt,
                &mut bmat,
                &mut zmat,
                &mut ws,
            );
            assert_eq!(kopt + 1, x.usize("kopt"), "{}: kopt", st.problem);
            assert_eq!(nf, x.usize("nf"), "{}: nf", st.problem);
            assert_eq!(i64::from(info), x.i64("info"), "{}: info", st.problem);
            stats.slice("fval", &fval, &x.vec("fval"));
            stats.slice("gopt", &gopt, &x.vec("gopt"));
            stats.mat("hq", &hq, &x.mat("hq"));
            stats.slice("pq", &pq, &x.vec("pq"));
            stats.slice("sl", &sl, &x.vec("sl"));
            stats.slice("su", &su, &x.vec("su"));
            stats.slice("xbase", &xbase, &x.vec("xbase"));
            stats.mat("xpt", &xpt, &x.mat("xpt"));
            stats.mat("bmat", &bmat, &x.mat("bmat"));
            stats.mat("zmat", &zmat, &x.mat("zmat"));
        }
        stats.report("rescue");
    }

    #[test]
    fn rescue_at_the_evaluation_budget_returns_maxfun_reached_with_zeroed_h() {
        // nf >= maxfun: the L234 early return must zero BMAT/ZMAT and change nothing else.
        let (n, npt) = (2, 5);
        let mut kopt = 0;
        let mut nf = 10;
        let mut fval = vec![1.0; npt];
        let (mut gopt, mut sl, mut su, mut xbase) =
            (vec![0.0; n], vec![-1.0; n], vec![1.0; n], vec![0.0; n]);
        let (mut hq, mut xpt) = (Mat::zeros(n, n), Mat::zeros(n, npt));
        let mut pq = vec![0.0; npt];
        let mut bmat = Mat::from_col_major(n, npt + n, vec![1.0; n * (npt + n)]);
        let mut zmat = Mat::from_col_major(npt, npt - n - 1, vec![1.0; npt * (npt - n - 1)]);
        let mut ws = RescueWs::new(n, npt);
        let info = rescue(
            &mut |_: &[f64]| 0.0,
            10,
            0.5,
            f64::NEG_INFINITY,
            &[-1.0; 2],
            &[1.0; 2],
            &mut kopt,
            &mut nf,
            &mut fval,
            &mut gopt,
            &mut hq,
            &mut pq,
            &mut sl,
            &mut su,
            &mut xbase,
            &mut xpt,
            &mut bmat,
            &mut zmat,
            &mut ws,
        );
        assert_eq!(info, crate::consts::MAXFUN_REACHED);
        assert!(bmat.data().iter().all(|&v| v == 0.0));
        assert!(zmat.data().iter().all(|&v| v == 0.0));
        assert_eq!(nf, 10);
    }
}
