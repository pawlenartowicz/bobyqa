//! The `f64` math seam: every float intrinsic the port uses that `core` lacks goes
//! through here, so the math backend is a one-file concern. Two backends: `std` intrinsics
//! (default) and `libm` free functions (the `libm` feature — the `no_std` backend). When both
//! features are on, `libm` wins: that combination is CI's bit-exact parity gate,
//! running the full golden battery on libm math under a std test harness.
//!
//! `f64::abs` is `core`, so it needs no backend split. `powi` is not on the seam: its only sites
//! became consts (`linalg::SCALMIN`/`SCALMAX`), so no backend has to supply one.

// Neither feature on would surface as thirty missing-method errors; say it once instead.
#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("bobyqa needs float math: enable the `std` feature (default) or `libm` for no_std");

#[cfg(feature = "libm")]
#[inline]
pub(crate) fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[cfg(not(feature = "libm"))]
#[inline]
pub(crate) fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[inline]
pub(crate) fn abs(x: f64) -> f64 {
    x.abs()
}

#[cfg(feature = "libm")]
#[inline]
pub(crate) fn floor(x: f64) -> f64 {
    libm::floor(x)
}

#[cfg(not(feature = "libm"))]
#[inline]
pub(crate) fn floor(x: f64) -> f64 {
    x.floor()
}

#[cfg(feature = "libm")]
#[inline]
pub(crate) fn round(x: f64) -> f64 {
    libm::round(x)
}

#[cfg(not(feature = "libm"))]
#[inline]
pub(crate) fn round(x: f64) -> f64 {
    x.round()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_seam_forwards_to_std() {
        assert_eq!(super::sqrt(4.0), 2.0);
        assert_eq!(super::abs(-1.5), 1.5);
        assert_eq!(super::floor(2.7), 2.0);
        assert_eq!(super::floor(-0.5), -1.0);
        assert_eq!(super::round(2.5), 3.0);
        assert_eq!(super::round(2.4), 2.0);
    }

    #[test]
    #[cfg(feature = "libm")]
    fn the_backends_agree_bit_for_bit() {
        // Tests always build with std, so the libm seam can be pinned against the std intrinsics
        // directly. The libm parity battery is the real gate, this is the fast local
        // tripwire.
        let samples = [
            0.0,
            1.0,
            2.0,
            0.5,
            1.5,
            2.5,
            -2.5,
            1.0e-300,
            1.0e300,
            core::f64::consts::PI,
            f64::MIN_POSITIVE,
            f64::MAX,
        ];
        for &x in &samples {
            if x >= 0.0 {
                assert_eq!(super::sqrt(x).to_bits(), x.sqrt().to_bits(), "sqrt({x})");
            }
            assert_eq!(super::floor(x).to_bits(), x.floor().to_bits(), "floor({x})");
            assert_eq!(super::round(x).to_bits(), x.round().to_bits(), "round({x})");
            assert_eq!(super::abs(x).to_bits(), x.abs().to_bits(), "abs({x})");
        }
    }
}
