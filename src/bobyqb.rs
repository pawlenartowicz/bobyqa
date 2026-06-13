//! `bobyqb.f90` (PRIMA): the major calculations of BOBYQA — the trust-region main loop — plus
//! its private helper `errbd` and the loop-only common helpers `shiftbase` (shiftbase.f90),
//! `redrat` (ratio.f90), and `redrho` (redrho.f90).
//!
//! Index convention: all point/variable indices 0-based (`kopt`, `knew_geo`, `tr`); the
//! `KNEW_TR = 0` sentinel from `setdrop_tr` is `Option<usize>`. History
//! (`xhist`/`fhist`), `iprint`/`fmsg`/`savehist`/`rangehist`/`retmsg`/`rhomsg`, and the optional
//! `callback_fcn` are omitted; info codes are `consts.rs` values, mapped to the
//! public `Status` in `lib.rs`.

use crate::consts::{
    DAMAGING_ROUNDING, INFO_DFT, MAXTR_REACHED, NAN_INF_MODEL, REALMAX, SMALL_TR_RADIUS,
};
use crate::geometry::{GeostepWs, geostep, setdrop_tr};
use crate::initialize::{InitWs, inith, initq, initxf};
use crate::linalg::{inprod, matprod12_into, matprod21_into, matprod22_into, norm, outprod_into};
use crate::mat::Mat;
use crate::math;
use crate::powalg::{CalWs, calvlag_and_den_into, hess_mul_into, quadinc};
use crate::rescue::{RescueWs, rescue};
use crate::trustregion::{TrsboxWs, trrad, trsbox};
use crate::update::{UpdateWs, tryqalt, updateh, updateq, updatexf};
use crate::util::{checkexit, evaluate, xinbd_into};

/// PRIMA bobyqb.f90 L188: convergence tolerance of the trust-region subproblem solver.
const TRTOL: f64 = 1.0e-2;

/// Reused scratch for the private `shiftbase` helper — its per-call locals (the XOPT copy and
/// the MATPROD/OUTPROD temporaries), hoisted like the [`BobyqbWs`] fields. Field names follow
/// the existing locals, which map to shiftbase.f90's expressions at their citation sites.
#[derive(Debug, Clone)]
struct ShiftbaseWs {
    xopt: Vec<f64>, // n
    xptxav: Mat,    // n x npt
    sxpt: Vec<f64>, // npt
    ymat: Mat,      // n x npt
    bmat_npt: Mat,  // n x npt
    ymat_t: Mat,    // npt x n
    bymat: Mat,     // n x n
    yzmat: Mat,     // n x (npt - n - 1)
    yzmat_c: Mat,   // n x (npt - n - 1)
    yzmat_c_t: Mat, // (npt - n - 1) x n
    zmat_t: Mat,    // (npt - n - 1) x npt
    yzyzmat: Mat,   // n x n
    yz_zt: Mat,     // n x npt
    v: Vec<f64>,    // n
    vxopt: Mat,     // n x n
}

impl ShiftbaseWs {
    fn new(n: usize, npt: usize) -> Self {
        Self {
            xopt: vec![0.0; n],
            xptxav: Mat::zeros(n, npt),
            sxpt: vec![0.0; npt],
            ymat: Mat::zeros(n, npt),
            bmat_npt: Mat::zeros(n, npt),
            ymat_t: Mat::zeros(npt, n),
            bymat: Mat::zeros(n, n),
            yzmat: Mat::zeros(n, npt - n - 1),
            yzmat_c: Mat::zeros(n, npt - n - 1),
            yzmat_c_t: Mat::zeros(npt - n - 1, n),
            zmat_t: Mat::zeros(npt - n - 1, npt),
            yzyzmat: Mat::zeros(n, n),
            yz_zt: Mat::zeros(n, npt),
            v: vec![0.0; n],
            vxopt: Mat::zeros(n, n),
        }
    }
}

/// Reused scratch for the private `errbd` helper — its per-call locals (XNEW, GNEW, BFIRST,
/// BSECOND and our `HESS_MUL` temporaries), hoisted like the [`BobyqbWs`] fields.
#[derive(Debug, Clone)]
struct ErrbdWs {
    xnew: Vec<f64>,    // n
    hm: Vec<f64>,      // n
    gnew: Vec<f64>,    // n
    bfirst: Vec<f64>,  // n
    v: Vec<f64>,       // n
    bsecond: Vec<f64>, // n
    dxpt: Vec<f64>,    // npt
}

impl ErrbdWs {
    fn new(n: usize, npt: usize) -> Self {
        Self {
            xnew: vec![0.0; n],
            hm: vec![0.0; n],
            gnew: vec![0.0; n],
            bfirst: vec![0.0; n],
            v: vec![0.0; n],
            bsecond: vec![0.0; n],
            dxpt: vec![0.0; npt],
        }
    }
}

/// Reused scratch for `bobyqb` itself — PRIMA bobyqb.f90 L160-188's local arrays plus
/// `lib.rs::minimize`'s clamped-bound copies, hoisted to the solver workspace (the crate-`//!`
/// zero-alloc warm path); the model state is re-initialized at the top of `bobyqb` and every body
/// temporary is fully written before use.
/// Field → Fortran-local map: the model state carries PRIMA's names (`bmat`/`zmat`/`xpt`/`hq`/
/// `fval`/`pq`/`den`/`distsq`/`gopt`/`sl`/`su`/`xbase`/`d`/`xdrop`/`xosav`/`vlag`/`ij`);
/// `xl`/`xu` are minimize's ±BOUNDMAX-clamped bounds (bobyqa.f90 L287-301), `xnew`/
/// `xnew_clamped`/`fval_shift`/`xopt_copy` the main-loop body temporaries, `dxpt_q`/`pqdxpt`/
/// `hqd` quadinc's scratch, `xmod` evaluate's MODERATEX copy.
#[derive(Debug, Clone)]
pub(crate) struct BobyqbWs {
    pub(crate) xl: Vec<f64>, // n — written by minimize before each bobyqb call
    pub(crate) xu: Vec<f64>, // n — written by minimize before each bobyqb call
    bmat: Mat,               // n x (npt + n)
    zmat: Mat,               // npt x (npt - n - 1)
    xpt: Mat,                // n x npt
    hq: Mat,                 // n x n
    fval: Vec<f64>,          // npt
    pq: Vec<f64>,            // npt
    den: Vec<f64>,           // npt
    distsq: Vec<f64>,        // npt
    gopt: Vec<f64>,          // n
    sl: Vec<f64>,            // n
    su: Vec<f64>,            // n
    xbase: Vec<f64>,         // n
    d: Vec<f64>,             // n
    xdrop: Vec<f64>,         // n
    xosav: Vec<f64>,         // n
    vlag: Vec<f64>,          // npt + n
    ij: Vec<(usize, usize)>, // capacity max(0, npt - 2n - 1)
    dxpt_q: Vec<f64>,        // npt
    pqdxpt: Vec<f64>,        // npt
    hqd: Vec<f64>,           // n
    xmod: Vec<f64>,          // n
    xnew: Vec<f64>,          // n
    xnew_clamped: Vec<f64>,  // n
    fval_shift: Vec<f64>,    // npt
    xopt_copy: Vec<f64>,     // n
    cal: CalWs,
    shiftbase: ShiftbaseWs,
    errbd: ErrbdWs,
}

/// All solver scratch, allocated once in `Bobyqa::new` and reused across `minimize` calls —
/// `bobyqb` borrows the sub-workspaces disjointly via destructuring.
#[derive(Debug, Clone)]
pub(crate) struct SolverWs {
    pub(crate) bobyqb: BobyqbWs,
    trsbox: TrsboxWs,
    geostep: GeostepWs,
    rescue: RescueWs,
    update: UpdateWs,
    init: InitWs,
}

impl SolverWs {
    /// Allocates every sub-workspace for an `(n, npt)` problem.
    pub(crate) fn new(n: usize, npt: usize) -> Self {
        Self {
            bobyqb: BobyqbWs {
                xl: vec![0.0; n],
                xu: vec![0.0; n],
                bmat: Mat::zeros(n, npt + n),
                zmat: Mat::zeros(npt, npt - n - 1),
                xpt: Mat::zeros(n, npt),
                hq: Mat::zeros(n, n),
                fval: vec![0.0; npt],
                pq: vec![0.0; npt],
                den: vec![0.0; npt],
                distsq: vec![0.0; npt],
                gopt: vec![0.0; n],
                sl: vec![0.0; n],
                su: vec![0.0; n],
                xbase: vec![0.0; n],
                d: vec![0.0; n],
                xdrop: vec![0.0; n],
                xosav: vec![0.0; n],
                vlag: vec![0.0; npt + n],
                ij: Vec::with_capacity(npt.saturating_sub(2 * n + 1)),
                dxpt_q: vec![0.0; npt],
                pqdxpt: vec![0.0; npt],
                hqd: vec![0.0; n],
                xmod: vec![0.0; n],
                xnew: vec![0.0; n],
                xnew_clamped: vec![0.0; n],
                fval_shift: vec![0.0; npt],
                xopt_copy: vec![0.0; n],
                cal: CalWs::new(n, npt),
                shiftbase: ShiftbaseWs::new(n, npt),
                errbd: ErrbdWs::new(n, npt),
            },
            trsbox: TrsboxWs::new(n, npt),
            geostep: GeostepWs::new(n, npt),
            rescue: RescueWs::new(n, npt),
            update: UpdateWs::new(n, npt),
            init: InitWs::new(n, npt),
        }
    }
}

/// PRIMA ratio.f90 L20 `redrat`: the reduction ratio of a trust-region step, Inf/NaN-safe.
fn redrat(ared: f64, pred: f64, rshrink: f64) -> f64 {
    // PRIMA ratio.f90 L49-69: the Inf/NaN ladder. The ±inf/+inf indeterminate forms are defined by
    // sign rather than computed (the division would give NaN): +inf/+inf -> 1, -inf/+inf -> -REALMAX
    // (PRIMA L63/L65), here one outer branch with an inner sign split. is_posinf/is_neginf transcribe
    // as is_infinite() && is_sign_positive()/negative().
    if ared.is_nan() {
        -REALMAX
    } else if pred.is_nan() || pred <= 0.0 {
        if ared > 0.0 { 0.5 * rshrink } else { -REALMAX }
    } else if pred.is_infinite() && pred.is_sign_positive() && ared.is_infinite() {
        // PRIMA ratio.f90 L63-66: +inf/+inf -> 1, -inf/+inf -> -REALMAX (both NaN if computed
        // directly).
        if ared.is_sign_positive() {
            1.0
        } else {
            -REALMAX
        }
    } else {
        ared / pred
    }
}

/// PRIMA redrho.f90 L20 `redrho`: the reduced RHO (shared by UOBYQA/NEWUOA/BOBYQA/LINCOA).
fn redrho(rho_in: f64, rhoend: f64) -> f64 {
    // PRIMA redrho.f90 L50-58.
    let rho_ratio = rho_in / rhoend;
    if rho_ratio > 250.0 {
        0.1 * rho_in
    } else if rho_ratio <= 16.0 {
        rhoend
    } else {
        math::sqrt(rho_ratio) * rhoend
    }
}

/// PRIMA shiftbase.f90 L26 `shiftbase_lfqint`: shift XBASE to XBASE + XOPT, updating BMAT and HQ
/// (PQ and ZMAT are unchanged). The optional IDZ is absent in BOBYQA (== 1, L87-91).
/// XOPT is `xpt.col(kopt)`, read inside the body (Fortran L116).
#[expect(clippy::too_many_arguments)] // out-params mirror the Fortran intent (rust.md §5)
#[expect(clippy::too_many_lines)] // one Fortran body, transcribed block-for-block
fn shiftbase(
    kopt: usize,
    xbase: &mut [f64],
    xpt: &mut Mat,
    zmat: &Mat,
    bmat: &mut Mat,
    pq: &[f64],
    hq: &mut Mat,
    ws: &mut ShiftbaseWs,
) {
    let n = xpt.nrows();
    let npt = xpt.ncols();

    let ShiftbaseWs {
        xopt,
        xptxav,
        sxpt,
        ymat,
        bmat_npt,
        ymat_t,
        bymat,
        yzmat,
        yzmat_c,
        yzmat_c_t,
        zmat_t,
        yzyzmat,
        yz_zt,
        v,
        vxopt,
    } = ws;

    // PRIMA shiftbase.f90 L116-117: read XOPT and its squared norm.
    xopt.copy_from_slice(xpt.col(kopt));
    let xoptsq = inprod(xopt, xopt);

    // PRIMA shiftbase.f90 L121: XPTXAV = XPT - HALF * spread(XOPT, dim=2, ncopies=npt),
    // i.e., XPT - XAV where XAV = (XBASE + XOPT)/2 relative to XBASE.
    xptxav.fill(0.0);
    for k in 0..npt {
        for i in 0..n {
            xptxav[[i, k]] = xpt[[i, k]] - 0.5 * xopt[i];
        }
    }

    // PRIMA shiftbase.f90 L124: the numerically-preferred variant.
    // SXPT(k) = INPROD(XOPT, XPT(:,k)) - HALF*XOPTSQ
    matprod12_into(xopt, xpt, sxpt);
    for s in sxpt.iter_mut() {
        *s -= 0.5 * xoptsq;
    }

    // PRIMA shiftbase.f90 L127-130: YMAT(:,k) = SXPT(k)*XPTXAV(:,k) + QXOPTQ*XOPT.
    let qxoptq = 0.25 * xoptsq;
    ymat.fill(0.0);
    for k in 0..npt {
        for i in 0..n {
            ymat[[i, k]] = sxpt[k] * xptxav[[i, k]] + qxoptq * xopt[i];
        }
    }

    // PRIMA shiftbase.f90 L133: BYMAT = MATPROD(BMAT(:, 1:NPT), TRANSPOSE(YMAT)).
    // BMAT(:, 1:NPT) is n x npt; YMAT is n x npt so TRANSPOSE(YMAT) is npt x n.
    bmat_npt.fill(0.0);
    for k in 0..npt {
        for i in 0..n {
            bmat_npt[[i, k]] = bmat[[i, k]];
        }
    }
    ymat_t.fill(0.0);
    for k in 0..npt {
        for i in 0..n {
            ymat_t[[k, i]] = ymat[[i, k]];
        }
    }
    matprod22_into(bmat_npt, ymat_t, bymat); // n x n

    // PRIMA shiftbase.f90 L134: BMAT(:, NPT+1:NPT+N) += BYMAT + TRANSPOSE(BYMAT).
    for i in 0..n {
        for j in 0..n {
            bmat[[i, npt + j]] += bymat[[i, j]] + bymat[[j, i]];
        }
    }

    // PRIMA shiftbase.f90 L136-140: BMAT updates that depend on ZMAT.
    // L136: YZMAT = MATPROD(YMAT, ZMAT) — YMAT is n x npt, ZMAT is npt x (npt-n-1), result n x (npt-n-1).
    matprod22_into(ymat, zmat, yzmat);

    // L137-138: YZMAT_C = YZMAT; YZMAT_C(:, 1:IDZ_LOC-1) = -YZMAT_C(:, 1:IDZ_LOC-1).
    // IDZ_LOC = 1 in BOBYQA (L87-91), so the slice 1:0 is empty — negation is a no-op.
    // YZMAT_C equals YZMAT exactly.
    yzmat_c.copy_from(yzmat); // L138 negation slice (1:idz-1) is empty for BOBYQA (IDZ==1)

    // L139: BMAT(:, NPT+1:NPT+N) += MATPROD(YZMAT, TRANSPOSE(YZMAT_C)).
    yzmat_c_t.fill(0.0);
    for j in 0..yzmat_c.ncols() {
        for i in 0..n {
            yzmat_c_t[[j, i]] = yzmat_c[[i, j]];
        }
    }
    matprod22_into(yzmat, yzmat_c_t, yzyzmat); // n x n
    for i in 0..n {
        for j in 0..n {
            bmat[[i, npt + j]] += yzyzmat[[i, j]];
        }
    }

    // L140: BMAT(:, 1:NPT) += MATPROD(YZMAT_C, TRANSPOSE(ZMAT)).
    zmat_t.fill(0.0);
    for j in 0..zmat.ncols() {
        for k in 0..npt {
            zmat_t[[j, k]] = zmat[[k, j]];
        }
    }
    matprod22_into(yzmat_c, zmat_t, yz_zt); // n x npt
    for k in 0..npt {
        for i in 0..n {
            bmat[[i, k]] += yz_zt[[i, k]];
        }
    }

    // PRIMA shiftbase.f90 L144: the numerically-preferred variant.
    // V = MATPROD(XPT, PQ) - HALF * SUM(PQ) * XOPT
    matprod21_into(xpt, pq, v);
    let pq_sum: f64 = pq.iter().sum();
    for i in 0..n {
        v[i] -= (0.5 * pq_sum) * xopt[i];
    }

    // PRIMA shiftbase.f90 L145-146: HQ = (VXOPT + TRANSPOSE(VXOPT)) + HQ.
    // Keep Fortran's operand order: (vxopt + vxopt^T) is evaluated first, then added to hq.
    outprod_into(v, xopt, vxopt); // n x n
    #[expect(clippy::assign_op_pattern)] // L146: operand order is (sum) + hq, not hq += sum
    for i in 0..n {
        for j in 0..n {
            hq[[i, j]] = (vxopt[[i, j]] + vxopt[[j, i]]) + hq[[i, j]];
        }
    }

    // PRIMA shiftbase.f90 L150-152: complete the shift of XBASE.
    for i in 0..n {
        xbase[i] += xopt[i]; // L150
    }
    for k in 0..npt {
        for i in 0..n {
            xpt[[i, k]] -= xopt[i]; // L151: xpt -= spread(xopt, ...)
        }
    }
    xpt.col_mut(kopt).fill(0.0); // L152: XPT(:, KOPT) = ZERO
}

/// PRIMA bobyqb.f90 L700 `errbd`: the bound used to test whether recent model errors are small
/// (BOBYQA paper, around (6.8)-(6.11)). Called only on SHORTD/TRFAIL iterations (L352).
#[expect(clippy::too_many_arguments)] // the argument list mirrors the Fortran signature (rust.md §5)
#[expect(clippy::similar_names)] // PRIMA identifiers xpt/xopt are load-bearing (rust.md §5)
fn errbd(
    crvmin: f64,
    d: &[f64],
    gopt: &[f64],
    hq: &Mat,
    moderr_rec: &[f64],
    pq: &[f64],
    rho: f64,
    sl: &[f64],
    su: &[f64],
    xopt: &[f64],
    xpt: &Mat,
    ws: &mut ErrbdWs,
) -> f64 {
    let n = xpt.nrows();
    let npt = xpt.ncols();

    let ErrbdWs {
        xnew,
        hm,
        gnew,
        bfirst,
        v,
        bsecond,
        dxpt,
    } = ws;

    // PRIMA bobyqb.f90 L766: XNEW = XOPT + D.
    xnew.fill(0.0);
    for i in 0..n {
        xnew[i] = xopt[i] + d[i];
    }

    // PRIMA bobyqb.f90 L767: GNEW = GOPT + HESS_MUL(D, XPT, PQ, HQ).
    hess_mul_into(d, xpt, pq, Some(hq), dxpt, hm);
    gnew.fill(0.0);
    for i in 0..n {
        gnew[i] = gopt[i] + hm[i];
    }

    // PRIMA bobyqb.f90 L768: BFIRST = MAXVAL(ABS(MODERR_REC)) — scalar fill (ascending scan).
    let max_abs_moderr = moderr_rec.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    bfirst.fill(max_abs_moderr);

    // PRIMA bobyqb.f90 L769: BFIRST(TRUELOC(XNEW <= SL)) = GNEW(...) * RHO — lower bound mask.
    for i in 0..n {
        if xnew[i] <= sl[i] {
            bfirst[i] = gnew[i] * rho;
        }
    }
    // PRIMA bobyqb.f90 L770: BFIRST(TRUELOC(XNEW >= SU)) = -GNEW(...) * RHO — upper bound mask.
    for i in 0..n {
        if xnew[i] >= su[i] {
            bfirst[i] = -gnew[i] * rho;
        }
    }

    // PRIMA bobyqb.f90 L771: BSECOND = HALF * (DIAG(HQ) + MATPROD(XPT**2, PQ)) * RHO**2.
    // V[i] = sum_{k} xpt[[i,k]]^2 * pq[k].
    let rhosq = rho * rho;
    v.fill(0.0);
    for k in 0..npt {
        for i in 0..n {
            v[i] += (xpt[[i, k]] * xpt[[i, k]]) * pq[k];
        }
    }
    bsecond.fill(0.0);
    for i in 0..n {
        bsecond[i] = 0.5 * (hq[[i, i]] + v[i]) * rhosq;
    }

    // PRIMA bobyqb.f90 L772: EBOUND = MINVAL(MAX(BFIRST, BFIRST + BSECOND)) — ascending min.
    let mut ebound = f64::INFINITY;
    for i in 0..n {
        let val = bfirst[i].max(bfirst[i] + bsecond[i]);
        if val < ebound {
            ebound = val;
        }
    }

    // PRIMA bobyqb.f90 L773-775: IF (CRVMIN > 0) EBOUND = MIN(EBOUND, 0.125 * CRVMIN * RHO**2).
    if crvmin > 0.0 {
        ebound = ebound.min(0.125 * crvmin * rhosq);
    }

    ebound
}

/// PRIMA bobyqb.f90 L46 `bobyqb`: the major calculations of BOBYQA. Returns `(f, nf, info)`;
/// `x` is overwritten with the best point (in original coordinates).
///
/// # Panics
///
/// Never when `ws` was built by `SolverWs::new(n, npt)` for this `(n, npt)` — the workspace
/// dimensions are then consistent with `x` and `npt`.
// The lints below stem from faithful-port discipline (rust.md §5): wide out-param signature,
// one long Fortran body, PRIMA's identifiers and its `!(a > b)` NaN-propagating negations, and
// explicit indexed loops that mirror PRIMA's array order.
#[expect(clippy::too_many_arguments)] // out-params mirror the Fortran intent (rust.md §5)
#[expect(clippy::too_many_lines)] // one Fortran body, transcribed block-for-block
#[expect(clippy::neg_cmp_op_on_partial_ord)] // `!(a > b)` is load-bearing for NaN — never `a <= b`
#[expect(clippy::needless_range_loop)] // explicit indexed loops mirror PRIMA
pub(crate) fn bobyqb<F: FnMut(&[f64]) -> f64>(
    calfun: &mut F,
    maxfun: usize,
    npt: usize,
    eta1: f64,
    eta2: f64,
    ftarget: f64,
    gamma1: f64,
    gamma2: f64,
    rhobeg: f64,
    rhoend: f64,
    x: &mut [f64],
    ws: &mut SolverWs,
) -> (f64, usize, i32) {
    let n = x.len();

    // Split the solver workspace into its disjoint sub-workspaces (single &mut each).
    let SolverWs {
        bobyqb: bws,
        trsbox: tws,
        geostep: gws,
        rescue: rws,
        update: uws,
        init: iws,
    } = ws;
    let BobyqbWs {
        xl,
        xu,
        bmat,
        zmat,
        xpt,
        hq,
        fval,
        pq,
        den,
        distsq,
        gopt,
        sl,
        su,
        xbase,
        d,
        xdrop,
        xosav,
        vlag,
        ij,
        dxpt_q,
        pqdxpt,
        hqd,
        xmod,
        xnew,
        xnew_clamped,
        fval_shift,
        xopt_copy,
        cal,
        shiftbase: sbws,
        errbd: ebws,
    } = bws;
    let (xl, xu): (&[f64], &[f64]) = (xl, xu); // written by minimize

    // PRIMA bobyqb.f90 L160-188: local variables — allocated once in `Bobyqa::new`,
    // re-initialized here to the Fortran's entry state on every call (a solver's Nth `minimize`
    // is independent of calls 1..N-1).
    bmat.fill(0.0);
    zmat.fill(0.0);
    xpt.fill(0.0);
    hq.fill(0.0);
    fval.fill(0.0);
    pq.fill(0.0);
    den.fill(0.0);
    distsq.fill(0.0);
    gopt.fill(0.0);
    sl.fill(0.0);
    su.fill(0.0);
    xbase.fill(0.0);
    d.fill(0.0);
    xdrop.fill(0.0);
    xosav.fill(0.0);
    vlag.fill(0.0);
    ij.clear();
    // quadinc scratch (DXPT, PQ*DXPT, GQ + HALF*HQD) and evaluate's MODERATEX copy.
    dxpt_q.fill(0.0);
    pqdxpt.fill(0.0);
    hqd.fill(0.0);
    xmod.fill(0.0);
    // PRIMA L168: DNORM_REC has size 2 on this pin (Powell's implementation used size 3). The
    // L291-292 `= REALMAX` reset coincides with this declared value (no use sits between).
    let mut dnorm_rec = [REALMAX; 2];
    let mut moderr_rec = [REALMAX; 2];

    // PRIMA bobyqb.f90 L221-222: initialize XBASE, XPT, SL, SU, FVAL, KOPT, NF, IJ. `x` is the
    // &mut x0 (overwritten in place); INITXF returns (kopt, nf, info).
    let (mut kopt, mut nf, mut subinfo) = initxf(
        calfun, maxfun, ftarget, rhobeg, xl, xu, x, ij, fval, sl, su, xbase, xpt, iws,
    );

    // PRIMA bobyqb.f90 L234-235: initialize X and F according to KOPT (X = XBASE + XOPT).
    xinbd_into(xbase, xpt.col(kopt), xl, xu, sl, su, x);
    let mut f = fval[kopt];

    // PRIMA bobyqb.f90 L239-249: finish the model initialization if INITXF completed normally.
    if subinfo == INFO_DFT {
        let _ = inith(ij, xpt, bmat, zmat, iws);
        let _ = initq(ij, fval, xpt, gopt, hq, pq, iws);
        // PRIMA L246: literal NaN-bearing negation of the model-finiteness test.
        if !(gopt.iter().all(|v| v.is_finite())
            && hq.data().iter().all(|v| v.is_finite())
            && pq.iter().all(|v| v.is_finite()))
        {
            subinfo = NAN_INF_MODEL;
        }
    }

    // PRIMA bobyqb.f90 L252-276: return on abnormal initialization (rangehist/retmsg/postconditions
    // omitted).
    if subinfo != INFO_DFT {
        return (f, nf, subinfo);
    }

    // PRIMA bobyqb.f90 L284-295: set some more initial values. SHORTD is read in the epilogue
    // (L652), so it stays function-scoped; TRFAIL and KNEW_GEO are set before every use and so are
    // declared at their loop/block scope below (the Fortran's defensive L289/L294 init is moot
    // because MAXTR >= 2 guarantees the loop body runs and writes them first).
    let mut rho = rhobeg;
    let mut delta = rho;
    let mut ebound = 0.0;
    let mut rescued = false;
    let mut shortd = false;
    let mut ratio = -1.0;
    // DNORM_REC/MODERR_REC already hold [REALMAX; 2] from their declaration (PRIMA L291-292).
    let mut knew_tr: Option<usize> = None;
    let mut itest: i32 = 0;

    // PRIMA bobyqb.f90 L304: GAMMA3 must be less than GAMMA2; see the long comment there.
    let gamma3 = 1.0_f64.max((0.75 * gamma2).min(1.5));

    // PRIMA bobyqb.f90 L315-316: MAXTR is the maximal number of trust-region iterations. Deliberate
    // divergence from the Fortran's HUGE(MAXTR) - 1 (budget-class; near-unreachable):
    // each TR iteration consumes 1-2 evaluations unless SHORTD/TRFAIL with no geometry step.
    let maxtr = maxfun.saturating_mul(2);
    let mut info = MAXTR_REACHED;

    // PRIMA L327: DNORM is set every iteration before use (the Fortran leaves it undefined).
    let mut dnorm = 0.0;

    // PRIMA bobyqb.f90 L324: begin the iterative procedure.
    for _tr in 1..=maxtr {
        // PRIMA bobyqb.f90 L326-328: generate the next trust-region step D.
        let crvmin = trsbox(
            delta,
            gopt,
            hq,
            pq,
            sl,
            su,
            TRTOL,
            xpt.col(kopt),
            xpt,
            d,
            tws,
        );
        dnorm = delta.min(norm(d));
        shortd = dnorm <= 0.5 * rho; // `<=` works better than `<` in case of underflow.

        // PRIMA bobyqb.f90 L332-333: QRED = Q(XOPT) - Q(XOPT + D); TRFAIL when QRED tiny/neg/NaN.
        let qred = -quadinc(d, xpt, gopt, pq, hq, dxpt_q, pqdxpt, hqd);
        // PRIMA bobyqb.f90 L333: the literal 1.0E-6 has no _RP suffix — gfortran evaluates it in
        // SINGLE precision (= 9.99999997475242708e-07), and the oracle binary compares against that.
        let trfail = !(qred > f64::from(1.0e-6_f32) * (rho * rho)); // literal NaN-bearing negation (rust.md §5).

        if shortd || trfail {
            // PRIMA bobyqb.f90 L347-350: D is short — adjust DELTA.
            #[expect(clippy::assign_op_pattern)]
            // L347: operand order is TENTH * delta (rust.md §5)
            {
                delta = 0.1 * delta;
            }
            if delta <= gamma3 * rho {
                delta = rho;
            }
            // PRIMA bobyqb.f90 L352: EBOUND tests whether MODERR_REC entries are small.
            ebound = errbd(
                crvmin,
                d,
                gopt,
                hq,
                &moderr_rec,
                pq,
                rho,
                sl,
                su,
                xpt.col(kopt),
                xpt,
                ebws,
            );
        } else {
            // PRIMA bobyqb.f90 L355-357: evaluate F at X = XBASE + XOPT + D.
            for i in 0..n {
                xnew[i] = xpt[[i, kopt]] + d[i];
            }
            xinbd_into(xbase, xnew, xl, xu, sl, su, x);
            f = evaluate(calfun, x, xmod);
            nf += 1;
            rescued = false;
            // PRIMA L361-363: fmsg/savehist omitted.

            // PRIMA bobyqb.f90 L366-370: check whether to exit.
            subinfo = checkexit(maxfun, nf, f, ftarget, x);
            if subinfo != INFO_DFT {
                info = subinfo;
                break;
            }

            // PRIMA bobyqb.f90 L374-378: update DNORM_REC and MODERR_REC. With QRED =
            // -QUADINC(...) this is f - fval[kopt] - quadinc(...), the same model error as the
            // geometry path's L590 form — not a sign discrepancy.
            dnorm_rec = [dnorm_rec[1], dnorm];
            let mut moderr = f - fval[kopt] + qred;
            moderr_rec = [moderr_rec[1], moderr];

            // PRIMA bobyqb.f90 L381: reduction ratio (Inf/NaN-safe).
            ratio = redrat(fval[kopt] - f, qred, eta1);

            // PRIMA bobyqb.f90 L384-387: update DELTA. After this, DELTA < DNORM may hold.
            delta = trrad(delta, dnorm, eta1, eta2, gamma1, gamma2, ratio);
            if delta <= gamma3 * rho {
                delta = rho;
            }

            // PRIMA bobyqb.f90 L390: is the new X better than the current best?
            let mut ximproved = f < fval[kopt];

            // PRIMA bobyqb.f90 L395-397: VLAG and DEN, then call RESCUE if rounding has damaged the
            // denominator. One kernel, not two (calvlag_into + calden_into recompute the same H·w)
            // — bit-identical.
            let ref_beta = calvlag_and_den_into(kopt, bmat, d, xpt, zmat, cal, vlag, den);
            // Layer-0 spec §5 (Tier B): the vlag/den/ref_beta just computed are reusable by
            // setdrop_tr/updateh below ONLY while (kopt, d, xpt, zmat, bmat) stay unchanged. The
            // rescue branch (below) invalidates them; this flag is cleared there if rescue runs.
            let mut model_unchanged = true;
            let vlag_abs_sum: f64 = vlag.iter().map(|v| math::abs(*v)).sum();
            // PRIMA L397: VMAX = MAXVAL(VLAG(1:NPT)**2) — ascending max over the first NPT entries.
            // The 0.0 is dead: npt >= 1, so the k == 0 arm below always overwrites it. It only
            // satisfies definite-initialization — the scan, not the init, sets the value.
            let mut vmax = 0.0_f64;
            for k in 0..npt {
                let sq = vlag[k] * vlag[k];
                if k == 0 || sq > vmax {
                    vmax = sq;
                }
            }
            let to_rescue =
                ximproved && !(vlag_abs_sum.is_finite() && den.iter().any(|&v| v > vmax));
            if to_rescue {
                // PRIMA bobyqb.f90 L405-417: the RESCUE call (solver/iprint/xhist/fhist omitted).
                if rescued {
                    info = DAMAGING_ROUNDING; // the last RESCUE did not improve the situation.
                    break;
                }
                subinfo = rescue(
                    calfun, maxfun, delta, ftarget, xl, xu, &mut kopt, &mut nf, fval, gopt, hq, pq,
                    sl, su, xbase, xpt, bmat, zmat, rws,
                );
                if subinfo != INFO_DFT {
                    info = subinfo;
                    break;
                }
                rescued = true;
                model_unchanged = false; // rescue moved XBASE/kopt/xpt/bmat/zmat and recomputed D
                dnorm_rec = [REALMAX; 2];
                moderr_rec = [REALMAX; 2];

                // PRIMA bobyqb.f90 L422-424: RESCUE shifted XBASE; update D, MODERR, XIMPROVED. Do
                // NOT recompute QRED (D is no longer a real trust-region step). D = max(sl, min(su, d)) - XOPT.
                for i in 0..n {
                    d[i] = sl[i].max(su[i].min(d[i])) - xpt[[i, kopt]];
                }
                moderr = f - fval[kopt] - quadinc(d, xpt, gopt, pq, hq, dxpt_q, pqdxpt, hqd);
                ximproved = f < fval[kopt];
            }

            // PRIMA bobyqb.f90 L429: index of the point to replace with XOPT + D.
            knew_tr = setdrop_tr(
                kopt,
                ximproved,
                bmat,
                d,
                delta,
                rho,
                xpt,
                zmat,
                gws,
                model_unchanged.then_some(den.as_slice()),
            );

            // PRIMA bobyqb.f90 L435-448: update [BMAT, ZMAT], [GOPT, HQ, PQ], [FVAL, XPT, KOPT].
            if let Some(knew) = knew_tr {
                xdrop.copy_from_slice(xpt.col(knew));
                xosav.copy_from_slice(xpt.col(kopt));
                let _ = updateh(
                    Some(knew),
                    kopt,
                    d,
                    xpt,
                    bmat,
                    zmat,
                    uws,
                    model_unchanged.then_some((vlag.as_slice(), ref_beta)),
                );
                for i in 0..n {
                    xnew_clamped[i] = sl[i].max(su[i].min(xosav[i] + d[i]));
                }
                updatexf(Some(knew), ximproved, f, xnew_clamped, &mut kopt, fval, xpt);
                updateq(
                    Some(knew),
                    ximproved,
                    bmat,
                    d,
                    moderr,
                    xdrop,
                    xosav,
                    xpt,
                    zmat,
                    gopt,
                    hq,
                    pq,
                    uws,
                );
                // PRIMA bobyqb.f90 L443: FVAL - FVAL(KOPT) reads the UPDATED KOPT (after updatexf).
                for k in 0..npt {
                    fval_shift[k] = fval[k] - fval[kopt];
                }
                xopt_copy.copy_from_slice(xpt.col(kopt));
                tryqalt(
                    bmat, fval_shift, ratio, sl, su, xopt_copy, xpt, zmat, &mut itest, gopt, hq,
                    pq, uws,
                );
                // PRIMA bobyqb.f90 L444-447: literal NaN-bearing model-finiteness test.
                if !(gopt.iter().all(|v| v.is_finite())
                    && hq.data().iter().all(|v| v.is_finite())
                    && pq.iter().all(|v| v.is_finite()))
                {
                    info = NAN_INF_MODEL;
                    break;
                }
            }
        } // End of IF (SHORTD .OR. TRFAIL). The normal trust-region calculation ends.

        // PRIMA bobyqb.f90 L463: ACCURATE_MOD — are the recent models sufficiently accurate?
        let accurate_mod = moderr_rec.iter().all(|v| math::abs(*v) <= ebound)
            && dnorm_rec.iter().all(|&v| v <= rho);
        // PRIMA bobyqb.f90 L465-467: CLOSE_ITPSET — are the interpolation points close to XOPT?
        for k in 0..npt {
            let mut sq = 0.0;
            for i in 0..n {
                let diff = xpt[[i, k]] - xpt[[i, kopt]];
                sq += diff * diff;
            }
            distsq[k] = sq;
        }
        let close_itpset = distsq
            .iter()
            .all(|&v| v <= (delta * delta).max((10.0 * rho) * (10.0 * rho)));
        // PRIMA bobyqb.f90 L476: ADEQUATE_GEO — is the geometry of the interpolation set adequate?
        let adequate_geo = (shortd && accurate_mod) || close_itpset;
        // PRIMA bobyqb.f90 L478: SMALL_TRRAD — is the trust-region radius small?
        let small_trrad = delta.max(dnorm) <= rho;

        // PRIMA bobyqb.f90 L486-487: BAD_TRSTEP (for IMPROVE_GEO) uses RATIO <= ETA1.
        let bad_trstep = shortd || trfail || ratio <= eta1 || knew_tr.is_none();
        let improve_geo = bad_trstep && !adequate_geo;
        // PRIMA bobyqb.f90 L489-490: BAD_TRSTEP (for REDUCE_RHO) uses RATIO <= 0.
        let bad_trstep = shortd || trfail || ratio <= 0.0 || knew_tr.is_none();
        let reduce_rho = bad_trstep && adequate_geo && small_trrad;

        // PRIMA bobyqb.f90 L521: improve the geometry of the interpolation set.
        if improve_geo {
            // PRIMA bobyqb.f90 L523: KNEW_GEO = MAXLOC(DISTSQ) — first max (ascending scan).
            let mut knew_geo = 0usize;
            for k in 0..npt {
                if k == 0 || distsq[k] > distsq[knew_geo] {
                    knew_geo = k;
                }
            }

            // PRIMA bobyqb.f90 L527: DELBAR = max(min(TENTH*sqrt(maxval(distsq)), delta), rho).
            // MAXVAL(DISTSQ) equals DISTSQ(KNEW_GEO) from the MAXLOC scan above; the separate
            // scan mirrors the Fortran's two intrinsic calls — do not merge the loops.
            let mut max_distsq = 0.0_f64;
            for k in 0..npt {
                if k == 0 || distsq[k] > max_distsq {
                    max_distsq = distsq[k];
                }
            }
            let delbar = (0.1 * math::sqrt(max_distsq)).min(delta).max(rho);

            // PRIMA bobyqb.f90 L533: geometry-improving step.
            geostep(knew_geo, kopt, bmat, delbar, sl, su, xpt, zmat, d, gws);

            // PRIMA bobyqb.f90 L546-548: VLAG and DEN, then call RESCUE if rounding has damaged the
            // denominator. One fused kernel.
            calvlag_and_den_into(kopt, bmat, d, xpt, zmat, cal, vlag, den);
            let vlag_abs_sum: f64 = vlag.iter().map(|v| math::abs(*v)).sum();
            let to_rescue = !(vlag_abs_sum.is_finite()
                && den[knew_geo] > 0.5 * (vlag[knew_geo] * vlag[knew_geo]));
            if to_rescue {
                // PRIMA bobyqb.f90 L550-562: the RESCUE call (no post-rescue D/MODERR update here).
                if rescued {
                    info = DAMAGING_ROUNDING; // the last RESCUE did not improve the situation.
                    break;
                }
                subinfo = rescue(
                    calfun, maxfun, delta, ftarget, xl, xu, &mut kopt, &mut nf, fval, gopt, hq, pq,
                    sl, su, xbase, xpt, bmat, zmat, rws,
                );
                if subinfo != INFO_DFT {
                    info = subinfo;
                    break;
                }
                rescued = true;
                dnorm_rec = [REALMAX; 2];
                moderr_rec = [REALMAX; 2];
            } else {
                // PRIMA bobyqb.f90 L565-568: evaluate F at X = XBASE + XOPT + D.
                for i in 0..n {
                    xnew[i] = xpt[[i, kopt]] + d[i];
                }
                xinbd_into(xbase, xnew, xl, xu, sl, su, x);
                f = evaluate(calfun, x, xmod);
                nf += 1;
                rescued = false;
                // PRIMA L571-573: fmsg/savehist omitted.

                // PRIMA bobyqb.f90 L576-580: check whether to exit.
                subinfo = checkexit(maxfun, nf, f, ftarget, x);
                if subinfo != INFO_DFT {
                    info = subinfo;
                    break;
                }

                // PRIMA bobyqb.f90 L586-591: update DNORM_REC and MODERR_REC (DNORM uses DELBAR).
                dnorm = delbar.min(norm(d));
                dnorm_rec = [dnorm_rec[1], dnorm];
                let moderr = f - fval[kopt] - quadinc(d, xpt, gopt, pq, hq, dxpt_q, pqdxpt, hqd);
                moderr_rec = [moderr_rec[1], moderr];

                // PRIMA bobyqb.f90 L594: is the new X better than the current best?
                let ximproved = f < fval[kopt];

                // PRIMA bobyqb.f90 L598-606: update [BMAT, ZMAT], [FVAL, XPT, KOPT], [GOPT, HQ, PQ].
                xdrop.copy_from_slice(xpt.col(knew_geo));
                xosav.copy_from_slice(xpt.col(kopt));
                let _ = updateh(Some(knew_geo), kopt, d, xpt, bmat, zmat, uws, None);
                for i in 0..n {
                    xnew_clamped[i] = sl[i].max(su[i].min(xosav[i] + d[i]));
                }
                updatexf(
                    Some(knew_geo),
                    ximproved,
                    f,
                    xnew_clamped,
                    &mut kopt,
                    fval,
                    xpt,
                );
                updateq(
                    Some(knew_geo),
                    ximproved,
                    bmat,
                    d,
                    moderr,
                    xdrop,
                    xosav,
                    xpt,
                    zmat,
                    gopt,
                    hq,
                    pq,
                    uws,
                );
                // No TRYQALT here: PRIMA bobyqb.f90 L598-606 (the geometry branch) has no TRYQALT —
                // unlike the trust-region step, which calls it after updateq.
                // PRIMA bobyqb.f90 L603-606: literal NaN-bearing model-finiteness test.
                if !(gopt.iter().all(|v| v.is_finite())
                    && hq.data().iter().all(|v| v.is_finite())
                    && pq.iter().all(|v| v.is_finite()))
                {
                    info = NAN_INF_MODEL;
                    break;
                }
            }
        } // End of IF (IMPROVE_GEO).

        // PRIMA bobyqb.f90 L612-625: reduce RHO; update DELTA at the same time.
        if reduce_rho {
            if rho <= rhoend {
                info = SMALL_TR_RADIUS;
                break;
            }
            // PRIMA L617-618: DELTA is computed BEFORE updating RHO (order matters). REDRHO is
            // pure; the duplicate call mirrors the Fortran's two statements — do not hoist a
            // local in a way that could invert the DELTA/RHO update order.
            delta = (0.5 * rho).max(redrho(rho, rhoend));
            rho = redrho(rho, rhoend);
            // PRIMA L619: rhomsg omitted.
            dnorm_rec = [REALMAX; 2];
            moderr_rec = [REALMAX; 2];
        } // End of IF (REDUCE_RHO).

        // PRIMA bobyqb.f90 L632-638: shift XBASE if XOPT may be too far from XBASE.
        let mut xopt_sq = 0.0_f64;
        for i in 0..n {
            xopt_sq += xpt[[i, kopt]] * xpt[[i, kopt]];
        }
        if xopt_sq >= 1.0e3 * (delta * delta) {
            // PRIMA L634-635: read XPT(:, KOPT) before mutating SL/SU (the Fortran reads the
            // un-shifted column; the borrow checker forces the copy regardless).
            xopt_copy.copy_from_slice(xpt.col(kopt));
            for i in 0..n {
                sl[i] = (sl[i] - xopt_copy[i]).min(0.0);
                su[i] = (su[i] - xopt_copy[i]).max(0.0);
            }
            shiftbase(kopt, xbase, xpt, zmat, bmat, pq, hq, sbws);
            for i in 0..n {
                xbase[i] = xl[i].max(xu[i].min(xbase[i]));
            }
        }

        // PRIMA bobyqb.f90 L640-647: the callback is omitted.
    } // End of DO TR = 1, MAXTR.

    // PRIMA bobyqb.f90 L652-661: try the Newton-Raphson step if it has not been tried yet.
    if info == SMALL_TR_RADIUS && shortd && dnorm > 0.1 * rhoend && nf < maxfun {
        for i in 0..n {
            xnew[i] = xpt[[i, kopt]] + d[i];
        }
        xinbd_into(xbase, xnew, xl, xu, sl, su, x);
        f = evaluate(calfun, x, xmod);
        nf += 1;
        // PRIMA L656-660: fmsg/savehist omitted.
    }

    // PRIMA bobyqb.f90 L664-667: choose [X, F] to return — current [X, F] or [XBASE + XOPT, FOPT].
    if fval[kopt] < f || f.is_nan() {
        xinbd_into(xbase, xpt.col(kopt), xl, xu, sl, su, x);
        f = fval[kopt];
    }

    // PRIMA bobyqb.f90 L670-673: rangehist/retmsg omitted.
    (f, nf, info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;

    #[test]
    fn bobyqb_converges_on_the_sphere_to_small_tr_radius() {
        let mut sphere = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
        let mut x = vec![1.0, 2.0];
        let mut ws = SolverWs::new(2, 5);
        ws.bobyqb.xl.copy_from_slice(&[-5.0, -5.0]);
        ws.bobyqb.xu.copy_from_slice(&[5.0, 5.0]);
        let (f, nf, info) = bobyqb(
            &mut sphere,
            500,
            5,
            0.1,
            0.7,
            f64::NEG_INFINITY,
            0.5,
            2.0,
            0.5,
            1e-6,
            &mut x,
            &mut ws,
        );
        assert_eq!(info, SMALL_TR_RADIUS);
        // Same params/bounds/x0 as the frozen `sphere_n2_npt5` golden, which pins these bit-exact
        // (n_eval 22, f ~1.2e-32, x ~[-1.1e-16, 0]). Pin the eval count and tighten f/x far below the
        // golden's actual values — a spuriously-early convergence (wrong rho schedule) no longer slips
        // through the old 5..=500 / 1e-8 / 1e-3 slack.
        assert_eq!(nf, 22);
        assert!(f < 1e-20, "f = {f:e}");
        assert!(x.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn errbd_takes_the_interior_bound_and_crvmin_arms() {
        let (n, npt) = (2, 5);
        let xpt = Mat::zeros(n, npt);
        let (hq, pq) = (Mat::zeros(n, n), vec![0.0; npt]);
        let gopt = vec![1.0, -2.0];
        let d = vec![0.1, 0.1];
        let moderr_rec = [0.25, -0.5];
        let (sl, su) = (vec![-1.0; n], vec![1.0; n]);
        let mut ws = ErrbdWs::new(n, npt);
        // Interior xnew, crvmin = 0: bfirst = maxval|moderr_rec| = 0.5 everywhere, bsecond = 0.
        let e = errbd(
            0.0,
            &d,
            &gopt,
            &hq,
            &moderr_rec,
            &pq,
            0.5,
            &sl,
            &su,
            &[0.0; 2],
            &xpt,
            &mut ws,
        );
        assert_eq!(e, 0.5);
        // crvmin > 0 caps it at 0.125 * crvmin * rho^2 (L773-775).
        let e = errbd(
            2.0,
            &d,
            &gopt,
            &hq,
            &moderr_rec,
            &pq,
            0.5,
            &sl,
            &su,
            &[0.0; 2],
            &xpt,
            &mut ws,
        );
        assert_eq!(e, 0.0625);
        // xnew[0] on the upper bound: bfirst[0] = -gnew[0] * rho = -0.5 -> ebound = -0.5 (L770).
        let su_active = vec![0.1, 1.0];
        let e = errbd(
            0.0,
            &d,
            &gopt,
            &hq,
            &moderr_rec,
            &pq,
            0.5,
            &sl,
            &su_active,
            &[0.0; 2],
            &xpt,
            &mut ws,
        );
        assert_eq!(e, -0.5);
    }

    #[test]
    fn errbd_pins_the_bsecond_and_active_bound_paths() {
        // The zero-xpt/pq/hq cases in errbd_takes_the_interior_bound_and_crvmin_arms leave BSECOND
        // (L451-463) and the HM term of GNEW (L427-431) identically zero, so mutations there survive.
        // Three n=1 cases with nonzero xpt/pq/hq, all values exactly representable, pin those paths:
        // (a) interior, nonzero BSECOND: xpt=[[3]], pq=[2], hq=1, rho=2; v[0]=9*2=18,
        //     bsecond[0]=0.5*(1+18)*4=38, bfirst[0]=0 => ebound=38.
        let xpt1 = Mat::from_col_major(1, 1, vec![3.0]);
        let mut hq1 = Mat::zeros(1, 1);
        hq1[[0, 0]] = 1.0;
        let mut ws1 = ErrbdWs::new(1, 1);
        let e = errbd(
            0.0,
            &[0.0],
            &[0.0],
            &hq1,
            &[0.0, 0.0],
            &[2.0],
            2.0,
            &[-10.0],
            &[10.0],
            &[0.0],
            &xpt1,
            &mut ws1,
        );
        assert_eq!(e, 38.0);
        // (b) upper bound active, exercises GNEW=GOPT+HM (L431) and BFIRST=-GNEW*RHO (L447):
        //     d=0.5, hq=2 => hm=1, gnew=2; xnew=0.5>=su=0.25 => bfirst=-1; bsecond=0.5*2*0.25=0.25;
        //     ebound=max(-1,-0.75)=-0.75.
        let mut hq2 = Mat::zeros(1, 1);
        hq2[[0, 0]] = 2.0;
        let xpt2 = Mat::zeros(1, 1);
        let mut ws2 = ErrbdWs::new(1, 1);
        let e = errbd(
            0.0,
            &[0.5],
            &[1.0],
            &hq2,
            &[0.0, 0.0],
            &[0.0],
            0.5,
            &[-10.0],
            &[0.25],
            &[0.0],
            &xpt2,
            &mut ws2,
        );
        assert_eq!(e, -0.75);
        // (c) lower bound active, exercises BFIRST=GNEW*RHO (L441): d=-0.5, hq=2 => hm=-1, gnew=2;
        //     xnew=-0.5<=sl=-0.25 => bfirst=1; bsecond=0.25; ebound=max(1,1.25)=1.25.
        let mut hq3 = Mat::zeros(1, 1);
        hq3[[0, 0]] = 2.0;
        let xpt3 = Mat::zeros(1, 1);
        let mut ws3 = ErrbdWs::new(1, 1);
        let e = errbd(
            0.0,
            &[-0.5],
            &[3.0],
            &hq3,
            &[0.0, 0.0],
            &[0.0],
            0.5,
            &[-0.25],
            &[10.0],
            &[0.0],
            &xpt3,
            &mut ws3,
        );
        assert_eq!(e, 1.25);
    }

    #[test]
    fn redrat_handles_nan_inf_and_the_plain_quotient() {
        // ratio.f90 L49-69, branch by branch.
        assert_eq!(redrat(f64::NAN, 1.0, 0.1), -REALMAX); // NaN ared
        assert_eq!(redrat(1.0, f64::NAN, 0.1), 0.05); // bad pred, ared > 0: HALF * rshrink
        assert_eq!(redrat(1.0, -2.0, 0.1), 0.05); // pred <= 0, ared > 0
        assert_eq!(redrat(-1.0, f64::NAN, 0.1), -REALMAX); // bad pred, ared <= 0
        assert_eq!(redrat(f64::INFINITY, f64::INFINITY, 0.1), 1.0); // +inf/+inf (L63)
        assert_eq!(redrat(f64::NEG_INFINITY, f64::INFINITY, 0.1), -REALMAX); // -inf/+inf (L65)
        assert_eq!(redrat(1.0, 2.0, 0.1), 0.5); // the ordinary case
        // ared == 0.0 is NOT > 0.0 in IEEE, so the bad-pred arm (L208) returns -REALMAX, not
        // HALF*rshrink — a `>`->`>=` mutation would wrongly return 0.05.
        assert_eq!(redrat(0.0, -1.0, 0.1), -REALMAX);
        assert_eq!(redrat(0.0, f64::NAN, 0.1), -REALMAX);
        // Exactly ONE operand infinite -> ordinary division, NOT the both-infinite branch (L209). The
        // `(+inf||finite)` cases below pin that the special-case && chain stays an &&, not an ||.
        assert_eq!(redrat(1.0, f64::INFINITY, 0.1), 0.0); // 1/+inf
        assert_eq!(redrat(f64::INFINITY, 2.0, 0.1), f64::INFINITY); // +inf/2 (finite positive pred)
        assert_eq!(redrat(f64::NEG_INFINITY, 2.0, 0.1), f64::NEG_INFINITY); // -inf/2
    }

    #[test]
    fn redrho_takes_the_tenth_floor_and_sqrt_branches() {
        // redrho.f90 L50-58: ratio > 250 -> rho/10; <= 16 -> rhoend; else sqrt(ratio)*rhoend.
        assert_eq!(redrho(1.0, 1e-6), 0.1);
        assert_eq!(redrho(1e-5, 1e-6), 1e-6);
        // Expected value computed independently at full precision (Python: sqrt(2e-5/1e-6)*1e-6),
        // NOT via the implementation's own `sqrt(ratio)*rhoend` expression — a self-referential RHS
        // re-runs the code under test and guards nothing. (2e-5/1e-6 is not an exact 20.0 in f64, so a
        // literal `sqrt(20.0)` would itself be an ulp trap; the pinned decimal round-trips exactly.)
        assert_eq!(redrho(2e-5, 1e-6), 4.472_135_954_999_579e-6);
        // ratio == 250.0 EXACTLY (250.0/1.0): PRIMA's threshold is strict `>`, so 250 belongs to the
        // sqrt branch, not the tenth-floor branch. A `>`->`>=` mutation would return 0.1*250=25 instead.
        // Independent: sqrt(250.0) = 15.811388300841896.
        assert_eq!(redrho(250.0, 1.0), 15.811_388_300_841_896);
    }

    /// The full dense Hessian: `HQ + sum_k pq[k] * xpt[:,k] * xpt[:,k]^T`.
    fn full_hessian(hq: &Mat, pq: &[f64], xpt: &Mat) -> Mat {
        let n = xpt.nrows();
        let mut h = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                h[[i, j]] = hq[[i, j]];
                for k in 0..xpt.ncols() {
                    h[[i, j]] += pq[k] * xpt[[i, k]] * xpt[[j, k]];
                }
            }
        }
        h
    }

    #[test]
    #[expect(clippy::similar_names)] // xpt/xopt are PRIMA identifiers (rust.md §5)
    fn shiftbase_zeroes_the_kopt_column_moves_xbase_and_preserves_the_model_hessian() {
        let (n, npt, kopt) = (2, 6, 3);
        let xpt = Mat::from_col_major(
            n,
            npt,
            vec![
                0.0, 0.0, 0.5, 0.1, -0.2, 0.5, 0.3, -0.4, -0.5, 0.0, 0.2, 0.25,
            ],
        );
        let pq = vec![0.4, -0.3, 0.2, 0.1, -0.5, 0.6];
        let mut hq = Mat::zeros(n, n);
        hq[[0, 0]] = 1.0;
        hq[[1, 1]] = 2.0;
        hq[[0, 1]] = 0.5;
        hq[[1, 0]] = 0.5;
        let zmat = Mat::from_col_major(npt, npt - n - 1, (1..=18u8).map(f64::from).collect());
        let mut bmat = Mat::from_col_major(n, npt + n, (1..=16u8).map(f64::from).collect());
        // Make the bmat trailing n x n section symmetric (a shiftbase precondition, shiftbase.f90 L101).
        bmat[[1, npt]] = bmat[[0, npt + 1]];
        let mut xbase = vec![10.0, 20.0];
        let xopt = xpt.col(kopt).to_vec();
        let h_before = full_hessian(&hq, &pq, &xpt);

        let mut xpt2 = xpt.clone();
        let mut ws = ShiftbaseWs::new(n, npt);
        shiftbase(
            kopt, &mut xbase, &mut xpt2, &zmat, &mut bmat, &pq, &mut hq, &mut ws,
        );

        assert!(xpt2.col(kopt).iter().all(|&v| v == 0.0)); // L152
        assert_eq!(xbase, vec![10.0 + xopt[0], 20.0 + xopt[1]]); // L150
        for j in (0..npt).filter(|&j| j != kopt) {
            for (i, &x) in xopt.iter().enumerate() {
                assert_eq!(xpt2[[i, j]], xpt[[i, j]] - x); // L151: xpt -= spread(xopt), every column
            }
        }
        let h_after = full_hessian(&hq, &pq, &xpt2);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (h_after[[i, j]] - h_before[[i, j]]).abs() <= 1e-12,
                    "model Hessian not preserved at [{i},{j}]"
                );
            }
        }
        // L134/L146 keep the bmat trailing section and hq symmetric.
        for i in 0..n {
            for j in 0..n {
                assert_eq!(bmat[[i, npt + j]], bmat[[j, npt + i]]);
                assert_eq!(hq[[i, j]], hq[[j, i]]);
            }
        }
    }
}
