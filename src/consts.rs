//! `consts.F90` subset + `infos.f90` codes (PRIMA common modules).
//!
//! Values are the f64 instantiations of PRIMA's parameterized constants.
/// PRIMA consts.F90 L138: machine epsilon.
pub(crate) const EPS: f64 = f64::EPSILON;
/// PRIMA consts.F90 L145: smallest positive normalized f64.
pub(crate) const REALMIN: f64 = f64::MIN_POSITIVE;
/// PRIMA consts.F90 L152: largest finite f64.
pub(crate) const REALMAX: f64 = f64::MAX;
/// PRIMA consts.F90 L169: moderated-extreme-barrier ceiling, 10^min(30, range/2) = 1e30 for f64.
pub(crate) const FUNCMAX: f64 = 1.0e30;
/// PRIMA consts.F90 L172: any |bound| >= BOUNDMAX means "no bound".
pub(crate) const BOUNDMAX: f64 = 0.25 * REALMAX;
/// PRIMA consts.F90 L210 (released, f64): symmetry-test tolerance max(10*EPS, 1e-10).
#[allow(dead_code)] // used by linalg::issymmetric, a §3.4 audited debug helper
pub(crate) const SYMTOL: f64 = 1.0e-10;

// Defaults (consts.F90 L229-254). FTARGET diverges deliberately: PRIMA's -REALMAX would make
// f = -REALMAX terminate; -inf is strictly "off". Set in Config::new, not here.
pub(crate) const RHOBEG_DFT: f64 = 1.0;
pub(crate) const RHOEND_DFT: f64 = 1.0e-6;
pub(crate) const MAXFUN_DIM_DFT: usize = 500;
pub(crate) const ETA1_DFT: f64 = 0.1; // shrink threshold (internal — SPEC §7.6)
pub(crate) const ETA2_DFT: f64 = 0.7; // expand threshold
pub(crate) const GAMMA1_DFT: f64 = 0.5; // shrink factor
pub(crate) const GAMMA2_DFT: f64 = 2.0; // expand factor

// infos.f90 L34-46 exit codes, kept as i32 with PRIMA's exact values so the state-corpus diff
// tests compare them directly. Mapping to the public `Status` is M1c (design §6.1).
pub(crate) const INFO_DFT: i32 = 0;
pub(crate) const SMALL_TR_RADIUS: i32 = 0; // == INFO_DFT, as in PRIMA
pub(crate) const FTARGET_ACHIEVED: i32 = 1;
#[allow(dead_code)] // PRIMA infos.f90 completeness constant; BOBYQA never emits it (lib.rs §6.1)
pub(crate) const TRSUBP_FAILED: i32 = 2;
pub(crate) const MAXFUN_REACHED: i32 = 3;
pub(crate) const MAXTR_REACHED: i32 = 20;
pub(crate) const NAN_INF_X: i32 = -1;
pub(crate) const NAN_INF_F: i32 = -2;
pub(crate) const NAN_INF_MODEL: i32 = -3;
#[allow(dead_code)] // validation rejects this before the loop; constant kept for PRIMA infos.f90 parity
pub(crate) const NO_SPACE_BETWEEN_BOUNDS: i32 = 6;
pub(crate) const DAMAGING_ROUNDING: i32 = 7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // The min/max chain deliberately mirrors the Fortran formula shape, not idiomatic Rust.
    #[allow(clippy::manual_clamp, clippy::unnecessary_min_or_max)]
    fn funcmax_matches_the_fortran_formula_for_f64() {
        // consts.F90: TEN**max(4, min(30, floor(range/2))) with range(0.0_f64) = 307.
        assert_eq!(FUNCMAX, 10.0_f64.powi(30.min(307 / 2).max(4)));
    }

    #[test]
    fn boundmax_is_a_quarter_of_realmax() {
        assert_eq!(BOUNDMAX, f64::MAX / 4.0);
    }
}
