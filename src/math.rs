//! The `f64` math seam (rust.md §9): every `std`-only float intrinsic the port uses goes through
//! here, so the v0.1.1+ `no_std`/`libm` swap is a one-file change. Extend as ports need.
#[inline]
pub(crate) fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[inline]
pub(crate) fn abs(x: f64) -> f64 {
    x.abs()
}

#[inline]
pub(crate) fn floor(x: f64) -> f64 {
    x.floor()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_seam_forwards_to_std() {
        assert_eq!(super::sqrt(4.0), 2.0);
        assert_eq!(super::abs(-1.5), 1.5);
        assert_eq!(super::floor(2.7), 2.0);
        assert_eq!(super::floor(-0.5), -1.0);
    }
}
