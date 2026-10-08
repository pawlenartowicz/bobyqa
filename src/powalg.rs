//! The `powalg.f90` subset BOBYQA uses (PRIMA common modules).
//!
//! Index convention: all indices 0-based (`kref`, `k`, the `setij` pairs).
//!
//! The `idz` parameter (NEWUOA machinery) is `optional` in Fortran and **never passed by
//! BOBYQA** (its call sites read `calbeta(kopt, bmat, d, xpt, zmat)` etc.) — this port
//! is the `idz = 1` specialization with the parameter omitted; each site where the Fortran
//! branches on `idz` carries a citation note.
use crate::linalg::{inprod, matprod12_into, matprod21_into};
use crate::mat::Mat;
use crate::util::try_vec;
use alloc::collections::TryReserveError;
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

/// Dev-only `calvlag_noadd` invocation counter (kernel-count instrumentation). Compiled in **only** under
/// the `count-kernels` feature; the default build has no global state
/// (determinism). `calvlag_noadd` is the single ≈15·n² H·w chokepoint that every kernel consumer
/// (`calvlag_into` / `calbeta` / `calden_into` / `calvlag_and_den_into`) routes through, so this
/// one counter measures the recompute multiplicity directly. Measurement instrumentation that
/// never ships — the same justification as tests/alloc.rs's `GlobalAlloc` shim.
#[cfg(feature = "count-kernels")]
pub mod counters {
    use core::sync::atomic::{AtomicUsize, Ordering};

    static CALVLAG_NOADD_CALLS: AtomicUsize = AtomicUsize::new(0);

    /// Zero the counter (call before each measured `minimize`).
    pub fn reset() {
        CALVLAG_NOADD_CALLS.store(0, Ordering::Relaxed);
    }

    /// `calvlag_noadd` invocations since the last [`reset`].
    pub fn calvlag_noadd_calls() -> usize {
        CALVLAG_NOADD_CALLS.load(Ordering::Relaxed)
    }

    /// Record one `calvlag_noadd` invocation.
    pub(crate) fn bump() {
        CALVLAG_NOADD_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

/// PRIMA powalg.f90 L1864 `setij`: the (P, Q) pairs of (2.4) of the BOBYQA paper for the
/// interpolation points beyond the first 2n+1. Writes **0-based** pairs into `ij` (cleared
/// first; pushes stay within the `new()`-reserved capacity `max(0, npt - 2n - 1)`); the
/// optional `sorting_direction` is omitted (BOBYQA calls `setij(n, npt)`).
pub(crate) fn setij_into(n: usize, npt: usize, ij: &mut Vec<(usize, usize)>) {
    // PRIMA powalg.f90 L1864: for k = n .. npt-n-2 (1-based values; empty when npt <= 2n+1):
    //   ell    = k / n                       (integer division)
    //   ij(1)  = k - n*ell + 1
    //   ij(2)  = modulo(ij(1) + ell - 1, n) + 1
    ij.clear();
    if npt > 2 * n + 1 {
        for k in n..=(npt - n - 2) {
            let ell = k / n;
            let i1 = k - n * ell + 1; // 1-based, in 1..=n
            let i2 = (i1 + ell - 1) % n + 1; // 1-based; arguments are positive, so % == modulo
            ij.push((i1 - 1, i2 - 1)); // 0-based for Rust callers
        }
    }
}

/// PRIMA powalg.f90 L790 `hess_mul`: HESSIAN*x with HESSIAN = HQ (0 if absent) +
/// `sum_k PQ(k)*XPT(:, k)*XPT(:, k)^T`. `hq` **is** omitted at three BOBYQA call sites
/// (update.f90 L360/L464, geometry.f90 L309) → `Option`, `None` ≡ Fortran "0 if absent".
/// `dxpt` is npt-length scratch; the result lands in `y` (length n).
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA
pub(crate) fn hess_mul_into(
    x: &[f64],
    xpt: &Mat,
    pq: &[f64],
    hq: Option<&Mat>,
    dxpt: &mut [f64],
    y: &mut [f64],
) {
    let n = xpt.nrows();
    let npt = xpt.ncols();
    // PRIMA: y = matprod(xpt, pq * matprod(x, xpt)). The PQ*MATPROD(X, XPT) product is
    // materialized in place over dxpt — IEEE multiplication commutes, so the `*=` spelling is
    // bit-identical to the Fortran's PQ-first operand order.
    matprod12_into(x, xpt, dxpt);
    for k in 0..npt {
        dxpt[k] *= pq[k];
    }
    matprod21_into(xpt, dxpt, y);
    // PRIMA: if (present(hq)) then do j = 1, n: y = y + hq(:, j) * x(j).
    if let Some(hq) = hq {
        // Length-equalized reslices: bounds-check-free, independent lanes.
        let y = &mut y[..n];
        for j in 0..n {
            let hqj = &hq.col(j)[..n];
            let xj = x[j];
            for i in 0..n {
                y[i] += hqj[i] * xj;
            }
        }
    }
}

/// PRIMA powalg.f90 L551 `quadinc_d0` (the only variant BOBYQA calls; always with `hq`):
/// QINC = Q(d) - Q(0) for Q defined via [gq, hq, pq]. `dxpt`/`pqdxpt` are npt-length scratch
/// and `hqd` n-length scratch (the Fortran's DXPT, PQ*DXPT, and GQ + HALF*MATPROD(HQ, D)).
#[expect(clippy::too_many_arguments)] // scratch params mirror the hoisted Fortran locals
pub(crate) fn quadinc(
    d: &[f64],
    xpt: &Mat,
    gq: &[f64],
    pq: &[f64],
    hq: &Mat,
    dxpt: &mut [f64],
    pqdxpt: &mut [f64],
    hqd: &mut [f64],
) -> f64 {
    let n = xpt.nrows();
    let npt = xpt.ncols();
    // PRIMA: dxpt = matprod(d, xpt);
    //        qinc = inprod(d, gq + HALF*matprod(hq, d)) + HALF*inprod(dxpt, pq*dxpt).
    matprod12_into(d, xpt, dxpt);
    matprod21_into(hq, d, hqd);
    // GQ + HALF*HQD is materialized in place over hqd — same elementwise sums, ascending i.
    for i in 0..n {
        hqd[i] = gq[i] + 0.5 * hqd[i];
    }
    for k in 0..npt {
        pqdxpt[k] = pq[k] * dxpt[k];
    }
    inprod(d, hqd) + 0.5 * inprod(dxpt, pqdxpt)
}

/// PRIMA powalg.f90 L896 `omega_mul`, idz = 1 specialization: y = OMEGA*x with
/// OMEGA = ZMAT*ZMAT^T (the Fortran negates xz(1:idz-1) — empty for idz = 1).
/// `xz` is (npt − n − 1)-length scratch; the result lands in `y` (length npt).
fn omega_mul_into(zmat: &Mat, x: &[f64], xz: &mut [f64], y: &mut [f64]) {
    matprod12_into(x, zmat, xz);
    matprod21_into(zmat, xz, y);
}

/// Reused scratch for [`calvlag_into`]/[`calbeta`]/[`calden_into`] — PRIMA's per-call locals,
/// hoisted to the owning solver workspace. Lifetime and contents per call are
/// identical to the Fortran locals; only the allocation site moves. Field → Fortran-local map:
/// `wcheck`/`wmv` are WCHECK/WMV (`calvlag_lfqint`, calbeta), `xrefxpt` the MATPROD(XREF, XPT)
/// temp, `xz` `omega_mul`'s XZ, `omega` its result, `vlag_beta` calbeta's local VLAG,
/// `vlag_den` calden's local VLAG, `hdiag` calden's HDIAG.
#[derive(Debug, Clone)]
pub(crate) struct CalWs {
    wcheck: Vec<f64>,    // npt
    xrefxpt: Vec<f64>,   // npt
    xz: Vec<f64>,        // npt - n - 1
    omega: Vec<f64>,     // npt
    wmv: Vec<f64>,       // npt + n
    vlag_beta: Vec<f64>, // npt + n
    vlag_den: Vec<f64>,  // npt + n
    hdiag: Vec<f64>,     // npt
}

impl CalWs {
    pub(crate) fn new(n: usize, npt: usize) -> Result<Self, TryReserveError> {
        Ok(Self {
            wcheck: try_vec(0.0, npt)?,
            xrefxpt: try_vec(0.0, npt)?,
            xz: try_vec(0.0, npt - n - 1)?,
            omega: try_vec(0.0, npt)?,
            wmv: try_vec(0.0, npt + n)?,
            vlag_beta: try_vec(0.0, npt + n)?,
            vlag_den: try_vec(0.0, npt + n)?,
            hdiag: try_vec(0.0, npt)?,
        })
    }
}

/// PRIMA powalg.f90 L1685-1692 (the `calbeta` tail): BETA from a step `d`, its reference column
/// `xref = xpt(:, kref)`, the `wcheck` weights, and VLAG computed **without** the `+1` at `kref`.
/// Shared by `calbeta` and `calvlag_and_den_into` (and through it `calden_into`), so BETA comes
/// from a single kernel result instead of a VLAG recompute. `vlag` is the no-`+1` H·w (length
/// npt + n).
fn beta_from_noadd_vlag(
    d: &[f64],
    xref: &[f64],
    wcheck: &[f64],
    vlag: &[f64],
    n: usize,
    npt: usize,
) -> f64 {
    // PRIMA powalg.f90 L1692: beta = dxref**2 + dsq*(xrefsq + dxref + dxref + HALF*dsq) - dvlag - wvlag.
    let dxref = inprod(d, xref);
    let dsq = inprod(d, d);
    let xrefsq = inprod(xref, xref);
    let dvlag = inprod(d, &vlag[npt..npt + n]);
    let wvlag = inprod(wcheck, &vlag[..npt]);
    dxref * dxref + dsq * (xrefsq + dxref + dxref + 0.5 * dsq) - dvlag - wvlag
}

/// The body shared by [`calvlag_into`], [`calbeta`] and [`calvlag_and_den_into`]: VLAG = H*w
/// *without* the +1 at `kref` (the difference between the two PRIMA routines). Narrow slice
/// params (not `&mut CalWs`) so each caller can lend disjoint fields of one workspace.
#[expect(clippy::too_many_arguments)] // scratch params mirror the hoisted Fortran locals
fn calvlag_noadd(
    kref: usize,
    bmat: &Mat,
    d: &[f64],
    xpt: &Mat,
    zmat: &Mat,
    wcheck: &mut [f64],
    xrefxpt: &mut [f64],
    xz: &mut [f64],
    omega: &mut [f64],
    wmv: &mut [f64],
    vlag: &mut [f64],
) {
    #[cfg(feature = "count-kernels")]
    counters::bump();
    let n = xpt.nrows();
    let npt = xpt.ncols();
    let xref = xpt.col(kref); // PRIMA: xref = xpt(:, kref)
    // PRIMA: wcheck = matprod(d, xpt); wcheck = wcheck * (HALF*wcheck + matprod(xref, xpt)).
    matprod12_into(d, xpt, wcheck);
    matprod12_into(xref, xpt, xrefxpt);
    for k in 0..npt {
        wcheck[k] *= 0.5 * wcheck[k] + xrefxpt[k];
    }
    // PRIMA: vlag(1:npt) = omega_mul(idz, zmat, wcheck) + matprod(d, bmat(:, 1:npt)).
    // The bmat(:, 1:npt) array section becomes a per-column inprod (the array-section→loop
    // call-site convention; see `linalg`).
    omega_mul_into(zmat, wcheck, xz, omega);
    for j in 0..npt {
        vlag[j] = omega[j] + inprod(d, bmat.col(j));
    }
    // PRIMA: vlag(npt+1:npt+n) = matprod(bmat, [wcheck, d]) — written straight into the tail.
    wmv[..npt].copy_from_slice(wcheck);
    wmv[npt..npt + n].copy_from_slice(d);
    matprod21_into(bmat, wmv, &mut vlag[npt..npt + n]);
}

/// PRIMA powalg.f90 L1507 `calvlag_lfqint`: VLAG = H*w for step `d` from XREF = XPT(:, kref),
/// written into `vlag` (length npt + n, fully overwritten). `kref` is 0-based.
pub(crate) fn calvlag_into(
    kref: usize,
    bmat: &Mat,
    d: &[f64],
    xpt: &Mat,
    zmat: &Mat,
    cw: &mut CalWs,
    vlag: &mut [f64],
) {
    let CalWs {
        wcheck,
        xrefxpt,
        xz,
        omega,
        wmv,
        ..
    } = cw;
    calvlag_noadd(
        kref, bmat, d, xpt, zmat, wcheck, xrefxpt, xz, omega, wmv, vlag,
    );
    // PRIMA: vlag(kref) = vlag(kref) + ONE.
    vlag[kref] += 1.0;
}

/// PRIMA powalg.f90 L1599 `calbeta`: BETA for step `d` from XREF = XPT(:, kref) — see (4.12) and
/// (4.26) of the NEWUOA paper. `kref` is 0-based. N.B. it recomputes VLAG *without* the +1 at
/// `kref`, exactly as the Fortran does (the `vlag(kref) + ONE` line there is commented out).
pub(crate) fn calbeta(
    kref: usize,
    bmat: &Mat,
    d: &[f64],
    xpt: &Mat,
    zmat: &Mat,
    cw: &mut CalWs,
) -> f64 {
    let n = xpt.nrows();
    let npt = xpt.ncols();
    let xref = xpt.col(kref); // PRIMA: xref = xpt(:, kref)
    let CalWs {
        wcheck,
        xrefxpt,
        xz,
        omega,
        wmv,
        vlag_beta,
        ..
    } = cw;
    calvlag_noadd(
        kref, bmat, d, xpt, zmat, wcheck, xrefxpt, xz, omega, wmv, vlag_beta,
    );
    beta_from_noadd_vlag(d, xref, wcheck, vlag_beta, n, npt)
}

/// PRIMA powalg.f90 L1721 `calden`: DEN(k) = SIGMA of (4.12) of the NEWUOA paper if XPT(:, k)
/// is replaced with XPT(:, kref) + d; written into `den` (length npt, fully overwritten).
/// `kref` is 0-based.
pub(crate) fn calden_into(
    kref: usize,
    bmat: &Mat,
    d: &[f64],
    xpt: &Mat,
    zmat: &Mat,
    cw: &mut CalWs,
    den: &mut [f64],
) {
    // `Vec::new()` does not allocate, so lending the buffer out of `cw` keeps the warm path
    // allocation-free.
    let mut vlag_den = core::mem::take(&mut cw.vlag_den);
    calvlag_and_den_into(kref, bmat, d, xpt, zmat, cw, &mut vlag_den, den);
    cw.vlag_den = vlag_den;
}

/// The fused VLAG/DEN kernel: VLAG **and** DEN from one ≈15·n² H·w kernel call, for the `bobyqb`
/// sites that call `calvlag_into` then `calden_into` on the same `(kref, d, xpt, zmat, bmat)`.
/// `vlag` (length npt + n) receives the `calvlag_lfqint` result (with the `+1` at `kref`); `den`
/// (length npt) the `calden` result. Also returns BETA (the `calbeta` value the kernel derives
/// on the way) so same-iteration reuse can thread it into `updateh` without a recompute.
/// Provenance: fuses `calvlag_lfqint` + `calden`; results are bit-identical to the two separate
/// calls (one kernel; the `+1` is applied after the BETA read).
#[expect(clippy::too_many_arguments)] // scratch params mirror the hoisted Fortran locals
pub(crate) fn calvlag_and_den_into(
    kref: usize,
    bmat: &Mat,
    d: &[f64],
    xpt: &Mat,
    zmat: &Mat,
    cw: &mut CalWs,
    vlag: &mut [f64],
    den: &mut [f64],
) -> f64 {
    let n = xpt.nrows();
    let npt = xpt.ncols();
    let xref = xpt.col(kref);
    let CalWs {
        wcheck,
        xrefxpt,
        xz,
        omega,
        wmv,
        hdiag,
        ..
    } = cw;
    // PRIMA: hdiag = -sum(zmat(:, 1:idz-1)**2, dim=2) + sum(zmat(:, idz:)**2, dim=2) —
    // the negative part is empty for idz = 1. Column-outer interchange:
    // contiguous ZMAT columns instead of stride-npt rows; every hdiag[k] still accumulates
    // its z² terms in ascending j, so the sums are bit-identical to the row-outer form.
    hdiag[..npt].fill(0.0);
    for j in 0..zmat.ncols() {
        let zj = &zmat.col(j)[..npt];
        for k in 0..npt {
            hdiag[k] += zj[k] * zj[k];
        }
    }
    // One kernel into the caller's `vlag` (no `+1` yet); BETA from it; then `+1`; then DEN.
    calvlag_noadd(
        kref, bmat, d, xpt, zmat, wcheck, xrefxpt, xz, omega, wmv, vlag,
    );
    let beta = beta_from_noadd_vlag(d, xref, wcheck, vlag, n, npt);
    vlag[kref] += 1.0; // calvlag_into's `+1` — now `vlag` equals the calvlag_into output exactly.
    // PRIMA: den = hdiag * beta + vlag(1:npt)**2.
    for k in 0..npt {
        den[k] = hdiag[k] * beta + vlag[k] * vlag[k];
    }
    beta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;

    #[test]
    fn quadinc_evaluates_the_model_increment() {
        // Q(d) - Q(0) with gq only (pq = 0, hq = 0): plain <d, gq>.
        let xpt = Mat::from_col_major(1, 3, vec![1.0, 2.0, 0.0]);
        let hq = Mat::zeros(1, 1);
        let (mut dxpt, mut pqdxpt, mut hqd) = (vec![0.0; 3], vec![0.0; 3], vec![0.0; 1]);
        assert_eq!(
            quadinc(
                &[2.0],
                &xpt,
                &[3.0],
                &[0.0, 0.0, 0.0],
                &hq,
                &mut dxpt,
                &mut pqdxpt,
                &mut hqd
            ),
            6.0
        );
        // Both second-order terms nonzero, computed independently:
        // qinc = <d, gq + 0.5*hq*d> + 0.5*<dxpt, pq*dxpt>, d=[2], gq=[3], hq=[[2]], pq=[1,2,0],
        // dxpt = matprod(d, xpt) = [2,4,0]  ->  2*(3 + 0.5*2*2) + 0.5*(1*4 + 2*16) = 2*5 + 18 = 28.
        // The gq-only case above zeros hq AND pq, so it cannot catch a broken 0.5 factor, hq matprod,
        // or pq-weighting; this case does.
        let hq2 = Mat::from_col_major(1, 1, vec![2.0]);
        assert_eq!(
            quadinc(
                &[2.0],
                &xpt,
                &[3.0],
                &[1.0, 2.0, 0.0],
                &hq2,
                &mut dxpt,
                &mut pqdxpt,
                &mut hqd
            ),
            28.0
        );
    }
}
