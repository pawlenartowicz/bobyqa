//! `trustregion.f90` (PRIMA bobyqa module): `trsbox` (truncated CG + 2-D boundary search under
//! box bounds) and `trrad` (radius update).
//!
//! Index convention: all indices 0-based; `xbdi` keeps PRIMA's −1/0/+1 bound-state encoding
//! (values, not indices). No index outputs — the diff tests translate nothing.
//!
//! Unenforced invariants: `trsbox` starts the step `d` at zero and accumulates it in place;
//! `crvmin` returns −REALMAX as the "unset" sentinel (no positive curvature was sampled).
use crate::consts::{EPS, REALMAX, REALMIN};
use crate::linalg::inprod;
use crate::mat::Mat;
use crate::math;
use crate::powalg::hess_mul_into;
use crate::util::interval_max;

/// `interval_max`'s largest grid: `GRID_SIZE` = 2*nint(17*`HANGT_BD` + 4.1) (trustregion.f90
/// L526) with `HANGT_BD` in (0, 1] (TANBD starts at ONE and is only ever min-reduced), so at
/// most 2*nint(21.1) = 42 points.
const GRID_SIZE_MAX: usize = 42;

/// Reused scratch for `trsbox` — PRIMA's per-call locals, hoisted to the solver workspace
/// (rust.md §4). Lifetime and contents per call are identical to the Fortran locals; only the
/// allocation site moves: every field is re-initialized at the original allocation site, per
/// call and per loop iteration where the original was in-loop. Field → Fortran-local map:
/// `gopt`/`pq`/`hq` the (possibly rescaled) GOPT/PQ/HQ copies, `xbdi`/`gnew`/`s`/`xnew`/
/// `xtest`/`sbound`/`hdred`/`ssq`/`tanbd`/`sqdscr` their PRIMA namesakes, `dold` the L356/L544
/// DOLD restore copies, `dred` the L441 reduced D, `hs` the `HESS_MUL` results, `dxpt`
/// `hess_mul` scratch, `xgrid`/`fgrid` `interval_max`'s grid (≤ [`GRID_SIZE_MAX`]).
#[derive(Debug, Clone)]
pub(crate) struct TrsboxWs {
    gopt: Vec<f64>,   // n
    pq: Vec<f64>,     // npt
    hq: Mat,          // n x n
    xbdi: Vec<i32>,   // n
    gnew: Vec<f64>,   // n
    s: Vec<f64>,      // n
    xnew: Vec<f64>,   // n
    xtest: Vec<f64>,  // n
    sbound: Vec<f64>, // n
    dold: Vec<f64>,   // n
    hdred: Vec<f64>,  // n
    dred: Vec<f64>,   // n
    ssq: Vec<f64>,    // n
    tanbd: Vec<f64>,  // n
    sqdscr: Vec<f64>, // n
    hs: Vec<f64>,     // n
    dxpt: Vec<f64>,   // npt
    xgrid: Vec<f64>,  // GRID_SIZE_MAX
    fgrid: Vec<f64>,  // GRID_SIZE_MAX
}

impl TrsboxWs {
    pub(crate) fn new(n: usize, npt: usize) -> Self {
        Self {
            gopt: vec![0.0; n],
            pq: vec![0.0; npt],
            hq: Mat::zeros(n, n),
            xbdi: vec![0; n],
            gnew: vec![0.0; n],
            s: vec![0.0; n],
            xnew: vec![0.0; n],
            xtest: vec![0.0; n],
            sbound: vec![0.0; n],
            dold: vec![0.0; n],
            hdred: vec![0.0; n],
            dred: vec![0.0; n],
            ssq: vec![0.0; n],
            tanbd: vec![0.0; n],
            sqdscr: vec![0.0; n],
            hs: vec![0.0; n],
            dxpt: vec![0.0; npt],
            xgrid: vec![0.0; GRID_SIZE_MAX],
            fgrid: vec![0.0; GRID_SIZE_MAX],
        }
    }
}

/// PRIMA trustregion.f90 L22 `trsbox`: approximately solves
/// `minimize Q(XOPT + D) s.t. ||D|| <= DELTA, SL <= XOPT + D <= SU` by truncated CG plus a
/// 2-D boundary search. Writes the step into `d` and returns `crvmin`.
// The lints below all stem from faithful-port discipline (rust.md §5): `trsbox` mirrors a wide,
// long Fortran routine with PRIMA's identifiers, its `!(a > b)` NaN-propagating negations (the
// shared transcription convention — never `a <= b`), `nint(sign(...))` -> `... as i32`, and the
// `nactsav` isize bookkeeping; the explicit indexed loops mirror PRIMA's array order.
#[expect(clippy::too_many_arguments)] // out-params mirror the Fortran intent
#[expect(clippy::too_many_lines)] // one Fortran body, transcribed block-for-block
#[expect(clippy::similar_names)] // xopt/xpt, sl/su, etc. are PRIMA's symbols
#[expect(clippy::neg_cmp_op_on_partial_ord)] // `!(a > b)` is load-bearing for NaN — never `a <= b`
#[expect(clippy::cast_possible_truncation)] // nint(sign(ONE, .)) is exactly ±1.0; grid_size > 0
#[expect(clippy::cast_sign_loss)] // 17*hangt_bd + 4.1 is positive at the cast site
#[expect(clippy::cast_possible_wrap)] // nact <= n is tiny; the isize cast cannot wrap
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA
pub(crate) fn trsbox(
    delta: f64,
    gopt_in: &[f64],
    hq_in: &Mat,
    pq_in: &[f64],
    sl: &[f64],
    su: &[f64],
    tol: f64,
    xopt: &[f64],
    xpt: &Mat,
    d: &mut [f64],
    ws: &mut TrsboxWs,
) -> f64 {
    let n = gopt_in.len();

    let TrsboxWs {
        gopt,
        pq,
        hq,
        xbdi,
        gnew,
        s,
        xnew,
        xtest,
        sbound,
        dold,
        hdred,
        dred,
        ssq,
        tanbd,
        sqdscr,
        hs,
        dxpt,
        xgrid,
        fgrid,
    } = ws;

    // PRIMA trustregion.f90 L168-180: scale the problem if GOPT contains large values (else FP
    // exceptions may occur). CRVMIN must be scaled back if nonzero; the step is scale invariant.
    let max_abs_gopt = gopt_in.iter().fold(0.0_f64, |m, &v| m.max(math::abs(v)));
    let (scaled, modscal): (bool, f64);
    if max_abs_gopt > 1.0e12 {
        // PRIMA L169: MAX is a precaution against underflow.
        let ms = (2.0 * REALMIN).max(1.0 / max_abs_gopt);
        for i in 0..n {
            gopt[i] = gopt_in[i] * ms;
        }
        for (pk, &pk_in) in pq.iter_mut().zip(pq_in) {
            *pk = pk_in * ms;
        }
        hq.copy_from(hq_in);
        for j in 0..n {
            for v in hq.col_mut(j) {
                *v *= ms;
            }
        }
        scaled = true;
        modscal = ms;
    } else {
        // PRIMA L175: MODSCAL is unused here but set to entertain Fortran compilers.
        gopt.copy_from_slice(gopt_in);
        pq.copy_from_slice(pq_in);
        hq.copy_from(hq_in);
        scaled = false;
        modscal = 1.0;
    }

    // PRIMA L184-186: IACT/DREDSQ/GGSAV initial values are unused but entertain the compiler. In
    // Rust `iact` is assigned before each read inside both loops, so it carries no initial value.
    let mut iact: Option<usize>;
    let mut dredsq = 0.0;
    let mut ggsav;

    // PRIMA L191-194: the sign of GOPT(I) gives the sign of the change to variable I that reduces
    // Q; XBDI(I) shows whether to fix variable I at a bound initially. NACT counts fixed variables.
    // The two masked assignments run in source order (SU block first, then SL).
    xbdi.fill(0);
    for i in 0..n {
        if xopt[i] >= su[i] && gopt[i] <= 0.0 {
            xbdi[i] = 1;
        }
    }
    for i in 0..n {
        if xopt[i] <= sl[i] && gopt[i] >= 0.0 {
            xbdi[i] = -1;
        }
    }
    let mut nact = xbdi.iter().filter(|&&v| v != 0).count();

    // PRIMA L197-198: initialize D and CRVMIN.
    d.fill(0.0);
    let mut crvmin = -REALMAX;

    // PRIMA L201-202: GNEW is the gradient at the current iterate; GREDSQ over the free variables.
    gnew.copy_from_slice(gopt);
    let mut gredsq = masked_sumsq(gnew, xbdi);
    // PRIMA L204: DELSQ is the upper bound on the sum of squares of the free variables.
    let mut delsq = delta * delta;
    // PRIMA L206: QRED is the reduction in Q so far.
    let mut qred = 0.0;
    // PRIMA L208: BETA is the coefficient for the previous search direction.
    let mut beta = 0.0;
    // PRIMA L211: ITERCG counts CG iterations for the current set of active bounds.
    let mut itercg = 0_usize;
    // PRIMA L214: TWOD_SEARCH defaults to FALSE.
    let mut twod_search = false;

    // PRIMA L222: MAXITER = min(10**min(4, range(0_IK)), (n - nact)**2); for f64 range >= 4 so
    // 10**min(4, .) = 10**4 = 10000.
    debug_assert!(nact <= n, "nact counts at-bound variables of n total");
    let mut maxiter = (10_000).min((n - nact) * (n - nact));

    // PRIMA L223-395: the truncated CG loop.
    s.fill(0.0);
    for _iter in 0..maxiter {
        // PRIMA L224-228: RESID = DELSQ - sum of squares of the free D; RESID <= 0 => boundary.
        let resid = delsq - masked_sumsq(d, xbdi);
        if resid <= 0.0 {
            twod_search = true;
            break;
        }

        // PRIMA L234-240: the next CG search direction (steepest descent on a restart), with the
        // fixed-variable components zeroed.
        if itercg == 0 {
            for i in 0..n {
                s[i] = -gnew[i];
            }
        } else {
            for i in 0..n {
                s[i] = beta * s[i] - gnew[i];
            }
        }
        for i in 0..n {
            if xbdi[i] != 0 {
                s[i] = 0.0;
            }
        }
        // PRIMA L241: STEPSQ = sum(s**2) over ALL variables (not masked).
        let stepsq = inprod(s, s);
        // PRIMA L242: DS = inprod over the free variables.
        let ds = masked_inprod(d, s, xbdi);

        // PRIMA L244: literal negation — TRUE when DS is NaN; never simplify to `a <= b`.
        if !(stepsq > EPS * delsq && gredsq * delsq > (tol * qred) * (tol * qred) && !ds.is_nan()) {
            break;
        }

        // PRIMA L252: SQRTD = square root of a discriminant; the MAXVAL avoids SQRTD < |DS| from
        // underflow.
        let sqrtd = math::sqrt(stepsq * resid + ds * ds)
            .max(math::sqrt(stepsq * resid))
            .max(math::abs(ds));

        // PRIMA L261-265: BSTEP. Powell's condition is DS >= 0 (the comment says it matters).
        let bstep = if ds >= 0.0 {
            resid / (sqrtd + ds)
        } else {
            (sqrtd - ds) / stepsq
        };
        // PRIMA L268: BSTEP <= 0 or non-finite should not happen but can; Powell did not guard.
        if bstep <= 0.0 || !bstep.is_finite() {
            break;
        }

        // PRIMA L272-277: HS, SHS, STPLEN.
        hess_mul_into(s, xpt, pq, Some(&*hq), dxpt, hs);
        let shs = masked_inprod(s, hs, xbdi);
        let mut stplen = bstep;
        if shs > 0.0 {
            stplen = bstep.min(gredsq / shs);
        }

        // PRIMA L306-325: reduce STPLEN to preserve the simple bounds; IACT is the new constrained
        // variable. The two WHERE blocks run in source order (SU then SL); each is a full pass.
        for i in 0..n {
            xnew[i] = xopt[i] + d[i];
        }
        for i in 0..n {
            xtest[i] = xnew[i] + stplen * s[i];
        }
        sbound.fill(stplen);
        for i in 0..n {
            if s[i] > 0.0 && xtest[i] > su[i] {
                sbound[i] = (su[i] - xnew[i]) / s[i];
            }
        }
        for i in 0..n {
            if s[i] < 0.0 && xtest[i] < sl[i] {
                sbound[i] = (sl[i] - xnew[i]) / s[i];
            }
        }
        // PRIMA L319: NaN entries of SBOUND -> STPLEN.
        for i in 0..n {
            if sbound[i].is_nan() {
                sbound[i] = stplen;
            }
        }
        // PRIMA L320-325: IACT and STPLEN from the first minimum, only if any SBOUND < STPLEN.
        iact = None;
        if sbound.iter().any(|&v| v < stplen) {
            let mut imin = 0;
            for i in 1..n {
                if sbound[i] < sbound[imin] {
                    imin = i;
                }
            }
            iact = Some(imin);
            stplen = sbound[imin];
        }

        // PRIMA L338-363: update CRVMIN, GNEW, D; set SDEC to the decrease in Q.
        let mut sdec = 0.0;
        if stplen > 0.0 {
            itercg += 1;
            let rayleighq = shs / stepsq;
            if iact.is_none() && rayleighq > 0.0 {
                if crvmin <= -REALMAX {
                    // PRIMA L343: CRVMIN <= -REALMAX means CRVMIN has not been set.
                    crvmin = rayleighq;
                } else {
                    crvmin = crvmin.min(rayleighq);
                }
            }
            ggsav = gredsq;
            for i in 0..n {
                gnew[i] += stplen * hs[i];
            }
            gredsq = masked_sumsq(gnew, xbdi);
            dold.copy_from_slice(d);
            for i in 0..n {
                d[i] += stplen * s[i];
            }
            // PRIMA L356-359: exit on Inf/NaN in D, restoring DOLD.
            let abs_sum: f64 = d.iter().map(|&v| math::abs(v)).sum();
            if !abs_sum.is_finite() {
                d.copy_from_slice(dold);
                break;
            }
            // PRIMA L361: the predicted reduction along this CG step,
            // sdec = α·(‖g‖² − ½·α·sᵀHs) with α = stplen and ‖g‖² = ggsav (the reduced-gradient
            // squared norm before the step), clamped to ≥ 0.
            sdec = (stplen * (ggsav - 0.5 * stplen * shs)).max(0.0);
            qred += sdec;
        } else {
            // Divergence: the Fortran leaves GGSAV stale here (its initial L186 value is unused,
            // per PRIMA's own L183 TODO). Assigning GREDSQ instead keeps the variable defined
            // without an Option; harmless — the path is effectively unreachable (GREDSQ == 0
            // exits earlier) and GGSAV is read only after STPLEN > 0 set it.
            ggsav = gredsq;
        }

        // PRIMA L366-394: the three-way branch.
        if let Some(ia) = iact {
            nact += 1;
            // PRIMA L368: mid-algorithm assert.
            debug_assert!(math::abs(s[ia]) > 0.0, "S(IACT) /= 0");
            // PRIMA L369: XBDI(IACT) = nint(sign(ONE, S(IACT))) = ±1.
            xbdi[ia] = 1.0_f64.copysign(s[ia]) as i32;
            // PRIMA L371-373: exit when NACT == N (must update XBDI before exiting).
            if nact >= n {
                break;
            }
            delsq -= d[ia] * d[ia];
            if delsq <= 0.0 {
                // PRIMA L376-379: DELSQ <= 0 means D reaches the trust-region boundary.
                twod_search = true;
                break;
            }
            beta = 0.0;
            itercg = 0;
            gredsq = masked_sumsq(gnew, xbdi);
        } else if stplen < bstep {
            // PRIMA L384-390: apply another CG iteration or exit. ITERCG > N - NACT is impossible.
            debug_assert!(
                nact < n,
                "nact==n breaks before reaching the CG continuation test"
            );
            if itercg >= n - nact || sdec <= tol * qred || sdec.is_nan() || qred.is_nan() {
                break;
            }
            beta = gredsq / ggsav;
        } else {
            twod_search = true;
            break;
        }
    }

    // PRIMA L400-405: set MAXITER for the 2-D search on the trust-region boundary.
    if twod_search {
        crvmin = 0.0;
        debug_assert!(
            nact < n,
            "twod_search is only set while a free variable remains"
        );
        maxiter = 10 * (n - nact);
    } else {
        maxiter = 0;
    }

    // PRIMA L424-557: improve D by a sequential 2-D search on the trust-region boundary for the
    // variables that have not reached a bound.
    let mut nactsav: isize = nact as isize - 1;
    hdred.fill(0.0);
    for iter1 in 1..=maxiter {
        for i in 0..n {
            xnew[i] = xopt[i] + d[i];
        }

        // PRIMA L429-431: update XBDI (which lower/upper bound is reached) in source order.
        for i in 0..n {
            if xbdi[i] == 0 && xnew[i] >= su[i] {
                xbdi[i] = 1;
            }
        }
        for i in 0..n {
            if xbdi[i] == 0 && xnew[i] <= sl[i] {
                xbdi[i] = -1;
            }
        }
        nact = xbdi.iter().filter(|&&v| v != 0).count();
        if nact >= n - 1 {
            break;
        }

        // PRIMA L437-445: update GREDSQ, DREDG, DREDSQ.
        gredsq = masked_sumsq(gnew, xbdi);
        let dredg = masked_inprod(d, gnew, xbdi);
        if iter1 == 1 || nact as isize > nactsav {
            // PRIMA L440-444: DREDSQ changes only when NACT increases.
            dredsq = masked_sumsq(d, xbdi);
            dred.copy_from_slice(d);
            for i in 0..n {
                if xbdi[i] != 0 {
                    dred[i] = 0.0;
                }
            }
            hess_mul_into(dred, xpt, pq, Some(&*hq), dxpt, hdred);
            nactsav = nact as isize;
        }

        // PRIMA L449-456: search direction S = linear combination of reduced D and reduced G,
        // orthogonal to reduced D.
        let mut temp = gredsq * dredsq - dredg * dredg;
        // PRIMA L450: literal negation — TEMP tiny or NaN.
        if !(temp > tol * tol * (gredsq * dredsq).max(qred * qred)) {
            break;
        }
        temp = math::sqrt(temp);
        for i in 0..n {
            s[i] = (dredg * d[i] - dredsq * gnew[i]) / temp;
        }
        for i in 0..n {
            if xbdi[i] != 0 {
                s[i] = 0.0;
            }
        }
        let sredg = -temp;

        // PRIMA L474-482: TANBD block, in exact source order. SSQ; TANBD = 1; the two WHERE pairs
        // (SL then SU), with SQDSCR reset to -REALMAX between them; NaN -> 0.
        ssq.fill(0.0);
        for i in 0..n {
            ssq[i] = d[i] * d[i] + s[i] * s[i];
        }
        tanbd.fill(1.0);
        sqdscr.fill(-REALMAX);
        for i in 0..n {
            if xbdi[i] == 0 && xopt[i] - sl[i] < math::sqrt(ssq[i]) {
                sqdscr[i] = math::sqrt(0.0_f64.max(ssq[i] - (xopt[i] - sl[i]) * (xopt[i] - sl[i])));
            }
        }
        for i in 0..n {
            if sqdscr[i] - s[i] > 0.0 {
                tanbd[i] = tanbd[i].min((xnew[i] - sl[i]) / (sqdscr[i] - s[i]));
            }
        }
        for v in sqdscr.iter_mut() {
            *v = -REALMAX;
        }
        for i in 0..n {
            if xbdi[i] == 0 && su[i] - xopt[i] < math::sqrt(ssq[i]) {
                sqdscr[i] = math::sqrt(0.0_f64.max(ssq[i] - (su[i] - xopt[i]) * (su[i] - xopt[i])));
            }
        }
        for i in 0..n {
            if sqdscr[i] + s[i] > 0.0 {
                tanbd[i] = tanbd[i].min((su[i] - xnew[i]) / (sqdscr[i] + s[i]));
            }
        }
        for v in tanbd.iter_mut() {
            if v.is_nan() {
                *v = 0.0;
            }
        }

        // PRIMA L499-508: IACT/HANGT_BD from the first minimum, only if any TANBD < 1.
        iact = None;
        let mut hangt_bd = 1.0;
        if tanbd.iter().any(|&v| v < 1.0) {
            let mut imin = 0;
            for i in 1..n {
                if tanbd[i] < tanbd[imin] {
                    imin = i;
                }
            }
            iact = Some(imin);
            hangt_bd = tanbd[imin];
        }
        if hangt_bd <= 0.0 {
            break;
        }

        // PRIMA L511-514: HS and curvatures for the alternative iteration.
        hess_mul_into(s, xpt, pq, Some(&*hq), dxpt, hs);
        let shs = masked_inprod(s, hs, xbdi);
        let dhs = masked_inprod(d, hs, xbdi);
        let dhd = masked_inprod(d, hdred, xbdi);

        // PRIMA L518-521: ARGS; exit on any NaN.
        let args = [shs, dhd, dhs, dredg, sredg];
        if args.iter().any(|v| v.is_nan()) {
            break;
        }
        // PRIMA L526: GRID_SIZE = 2 * nint(17*HANGT_BD + 4.1). HANGT_BD > 0 here, so f64::round
        // (half away from zero) matches Fortran nint.
        let grid_size = 2 * ((17.0 * hangt_bd + 4.1).round() as usize);
        debug_assert!(
            grid_size <= GRID_SIZE_MAX,
            "interval_max grid over capacity"
        );
        // PRIMA L528-529.
        let hangt = interval_max(
            interval_fun_trsbox,
            0.0,
            hangt_bd,
            &args,
            grid_size,
            &mut xgrid[..grid_size],
            &mut fgrid[..grid_size],
        );
        let sdec = interval_fun_trsbox(hangt, &args);
        // PRIMA L530: literal negation.
        if !(sdec > 0.0) {
            break;
        }

        // PRIMA L537-538: cth = cos θ, sth = sin θ for the rotation by θ (hangt = tan(θ/2)). The
        // .min() caps each fraction at its own numerator (1−hangt² for cth, 2·hangt for sth) — valid
        // since the denominator 1+hangt² ≥ 1, a rounding precaution so cth/sth can't exceed it.
        let cth = ((1.0 - hangt * hangt) / (1.0 + hangt * hangt)).min(1.0 - hangt * hangt);
        let sth = ((hangt + hangt) / (1.0 + hangt * hangt)).min(hangt + hangt);
        // PRIMA L539: GNEW = GNEW + (CTH - ONE)*HDRED + STH*HS — Fortran `+` is left-associative,
        // so the per-element order is ((gnew + (cth-1)*hdred) + sth*hs); keep that association.
        for i in 0..n {
            gnew[i] = gnew[i] + (cth - 1.0) * hdred[i] + sth * hs[i];
        }
        dold.copy_from_slice(d);
        // PRIMA L541: update the free entries of D.
        for i in 0..n {
            if xbdi[i] == 0 {
                d[i] = cth * d[i] + sth * s[i];
            }
        }
        // PRIMA L544-547: exit on Inf/NaN in D, restoring DOLD.
        let abs_sum: f64 = d.iter().map(|&v| math::abs(v)).sum();
        if !abs_sum.is_finite() {
            d.copy_from_slice(dold);
            break;
        }
        // PRIMA L549-550.
        for i in 0..n {
            hdred[i] = cth * hdred[i] + sth * hs[i];
        }
        qred += sdec;
        // PRIMA L551-556: the trailing branch.
        if let Some(ia) = iact {
            if hangt >= hangt_bd {
                // PRIMA L552: D(IACT) reaches lower/upper bound.
                xbdi[ia] = 1.0_f64.copysign(xopt[ia] + d[ia] - 0.5 * (sl[ia] + su[ia])) as i32;
            } else if !(sdec > tol * qred) {
                // PRIMA L554: literal negation — SDEC small or NaN.
                break;
            }
        } else if !(sdec > tol * qred) {
            break;
        }
    }

    // PRIMA L560-563: set D, giving careful attention to the bounds. The Fortran is
    // max(sl, min(su, xopt + d)); the XBDI overrides pin fixed variables exactly to their bounds.
    for i in 0..n {
        xnew[i] = (xopt[i] + d[i]).min(su[i]).max(sl[i]);
    }
    for i in 0..n {
        if xbdi[i] == -1 {
            xnew[i] = sl[i];
        }
    }
    for i in 0..n {
        if xbdi[i] == 1 {
            xnew[i] = su[i];
        }
    }
    for i in 0..n {
        d[i] = xnew[i] - xopt[i];
    }

    // PRIMA L566-568: set CRVMIN to ZERO if never set or NaN due to ill conditioning.
    if crvmin <= -REALMAX || crvmin.is_nan() {
        crvmin = 0.0;
    }

    // PRIMA L571-573: scale CRVMIN back before return (the step is scale invariant).
    if scaled && crvmin > 0.0 {
        crvmin /= modscal;
    }

    crvmin
}

/// PRIMA trustregion.f90 L593 `interval_fun_trsbox`: the objective of the HANGT search.
/// HANGT = tan(θ/2) parameterizes the 2-D boundary arc cos θ·d + sin θ·s (sin θ = 2·HANGT/(1+HANGT²));
/// the returned value is the model reduction achieved at angle θ.
/// args = [shs, dhd, dhs, dredg, sredg].
fn interval_fun_trsbox(hangt: f64, args: &[f64]) -> f64 {
    let mut f = 0.0;
    if math::abs(hangt) > 0.0 {
        let sth = (hangt + hangt) / (1.0 + hangt * hangt);
        f = args[0] + hangt * (hangt * args[1] - args[2] - args[2]);
        f = sth * (hangt * args[3] - args[4] - 0.5 * sth * f);
    }
    f
}

/// PRIMA trustregion.f90 L636 `trrad`: update the trust-region radius according to RATIO and DNORM.
/// Callers pass the thresholds ordered `eta1 < eta2` and factors `0 < gamma1 <= 1 < gamma2`.
///
/// # Panics
///
/// Never — pure arithmetic on the arguments.
pub(crate) fn trrad(
    delta_in: f64,
    dnorm: f64,
    eta1: f64,
    eta2: f64,
    gamma1: f64,
    gamma2: f64,
    ratio: f64,
) -> f64 {
    // PRIMA trustregion.f90 L678-689: three branches by RATIO —
    //   shrink (ratio ≤ eta1):  min(γ1·δ, dnorm)
    //   keep   (ratio ≤ eta2):  max(γ1·δ, dnorm)
    //   expand (ratio  > eta2): max(γ1·δ, γ2·dnorm)
    if ratio <= eta1 {
        (gamma1 * delta_in).min(dnorm)
    } else if ratio <= eta2 {
        (gamma1 * delta_in).max(dnorm)
    } else {
        (gamma1 * delta_in).max(gamma2 * dnorm)
    }
}

/// PRIMA `sum(x(trueloc(xbdi == 0))**2)`: ascending masked sum of squares over the free variables,
/// in the same FP order as Fortran's gather-then-sum.
fn masked_sumsq(x: &[f64], xbdi: &[i32]) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() {
        if xbdi[i] == 0 {
            acc += x[i] * x[i];
        }
    }
    acc
}

/// PRIMA `inprod(x(trueloc(xbdi == 0)), y(trueloc(xbdi == 0)))`: ascending masked inner product
/// over the free variables.
fn masked_inprod(x: &[f64], y: &[f64], xbdi: &[i32]) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() {
        if xbdi[i] == 0 {
            acc += x[i] * y[i];
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    // The diff tests bind PRIMA's symbols (xopt/xpt, sl/su, hq_in) and `states`/`stats` side by
    // side — faithful naming over clippy's similarity heuristic (rust.md §5).
    #![expect(clippy::similar_names)]

    use super::*;
    use crate::mat::Mat;
    use crate::test_support::{self, DiffStats};

    #[test]
    fn trrad_matches_prima_on_every_captured_state() {
        let states = test_support::load_states("trrad");
        assert!(!states.is_empty());
        let mut stats = DiffStats::default();
        for st in &states {
            let (e, x) = (&st.entry, &st.exit);
            let delta = trrad(
                e.f64("delta_in"),
                e.f64("dnorm"),
                e.f64("eta1"),
                e.f64("eta2"),
                e.f64("gamma1"),
                e.f64("gamma2"),
                e.f64("ratio"),
            );
            stats.f64("delta", delta, x.f64("delta"));
        }
        stats.report("trrad");
    }

    #[test]
    fn trsbox_matches_prima_on_every_captured_state() {
        let states = test_support::load_states("trsbox");
        assert!(!states.is_empty());
        let mut stats = DiffStats::default();
        for st in &states {
            let (e, x) = (&st.entry, &st.exit);
            let (gopt_in, pq_in) = (e.vec("gopt_in"), e.vec("pq_in"));
            let (hq_in, xpt) = (e.mat("hq_in"), e.mat("xpt"));
            let (sl, su, xopt) = (e.vec("sl"), e.vec("su"), e.vec("xopt"));
            let mut d = vec![0.0; gopt_in.len()];
            let mut ws = TrsboxWs::new(gopt_in.len(), pq_in.len());
            let crvmin = trsbox(
                e.f64("delta"),
                &gopt_in,
                &hq_in,
                &pq_in,
                &sl,
                &su,
                e.f64("tol"),
                &xopt,
                &xpt,
                &mut d,
                &mut ws,
            );
            stats.f64("crvmin", crvmin, x.f64("crvmin"));
            stats.slice("d", &d, &x.vec("d"));
        }
        stats.report("trsbox");
    }

    #[test]
    fn interval_fun_trsbox_is_zero_at_the_origin_and_matches_the_formula() {
        // hangt = 0 -> f = 0 (the guard `abs(hangt) > 0` is false).
        assert_eq!(interval_fun_trsbox(0.0, &[1.0, 2.0, 3.0, 4.0, 5.0]), 0.0);
        // hangt = 1, args = [1, 2, 3, 4, 5]: sth = 2/2 = 1.
        //   inner f = 1 + 1*(1*2 - 3 - 3) = 1 + (2 - 6) = -3.
        //   f = 1 * (1*4 - 5 - 0.5*1*(-3)) = 4 - 5 + 1.5 = 0.5.
        let f = interval_fun_trsbox(1.0, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(math::abs(f - 0.5) < 1e-15);
    }

    #[test]
    fn trrad_takes_the_shrink_keep_and_expand_branches() {
        // eta1 = 0.1, eta2 = 0.7, gamma1 = 0.5, gamma2 = 2.0 (the consts.rs defaults).
        assert_eq!(trrad(1.0, 0.4, 0.1, 0.7, 0.5, 2.0, 0.05), 0.4); // min(0.5, 0.4)
        assert_eq!(trrad(1.0, 0.8, 0.1, 0.7, 0.5, 2.0, 0.5), 0.8); // max(0.5, 0.8)
        assert_eq!(trrad(1.0, 0.8, 0.1, 0.7, 0.5, 2.0, 0.9), 1.6); // max(0.5, 1.6)
    }

    #[test]
    fn trsbox_takes_the_unconstrained_newton_step_on_a_separable_quadratic() {
        // Q(x) = 0.5*(x1^2 + x2^2) via hq = I, gopt = (1, 0.5) at xopt = 0, wide bounds,
        // delta large: minimizer d = -gopt, interior, crvmin = 1 (the Rayleigh quotient).
        let n = 2;
        let npt = 4;
        let xpt = Mat::zeros(n, npt);
        let pq = vec![0.0; npt];
        let mut hq = Mat::zeros(n, n);
        hq[[0, 0]] = 1.0;
        hq[[1, 1]] = 1.0;
        let (sl, su) = (vec![-10.0; n], vec![10.0; n]);
        let mut d = vec![0.0; n];
        let mut ws = TrsboxWs::new(n, npt);
        let crvmin = trsbox(
            5.0,
            &[1.0, 0.5],
            &hq,
            &pq,
            &sl,
            &su,
            1e-2,
            &[0.0; 2],
            &xpt,
            &mut d,
            &mut ws,
        );
        assert!(math::abs(d[0] + 1.0) < 1e-12 && math::abs(d[1] + 0.5) < 1e-12);
        assert!(math::abs(crvmin - 1.0) < 1e-12);
    }

    #[test]
    fn trsbox_fixes_a_variable_pinned_at_its_bound() {
        // xopt[0] sits on the upper bound with a gradient pushing further up: xbdi fixes it,
        // d[0] must stay 0 while the free variable moves.
        let n = 2;
        let npt = 4;
        let xpt = Mat::zeros(n, npt);
        let pq = vec![0.0; npt];
        let mut hq = Mat::zeros(n, n);
        hq[[0, 0]] = 1.0;
        hq[[1, 1]] = 1.0;
        let (sl, su) = (vec![-10.0; n], vec![0.0, 10.0]); // su[0] = 0 = xopt[0]
        let mut d = vec![0.0; n];
        let mut ws = TrsboxWs::new(n, npt);
        let _ = trsbox(
            5.0,
            &[-1.0, 1.0],
            &hq,
            &pq,
            &sl,
            &su,
            1e-2,
            &[0.0; 2],
            &xpt,
            &mut d,
            &mut ws,
        );
        assert_eq!(d[0], 0.0);
        // The free variable's unconstrained Newton step is exactly -gopt[1]/hq[1,1] = -1.0 (interior:
        // delta=5 and the [-10,10] box don't bind). Pin the magnitude — a bare `d[1] < 0.0` accepts
        // any step-scaling bug (e.g. -1e-200 or a wrong trust-radius factor) as long as the sign holds.
        assert!((d[1] + 1.0).abs() < 1e-12, "d[1] = {}", d[1]);
    }
}
