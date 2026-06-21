//! Host-side `f32`/`f64` ⇄ `f64` bit-reinterpret helpers (IN-02).
//!
//! mlrs kernels are generic over the float type `F` (`f32` or `f64`), but the
//! host-side "combine" steps that stitch device results back together work in
//! `f64` for accuracy. Bridging a generic `F` to/from `f64` cannot use `as`
//! casts (the compiler does not know `F` is a float), so we reinterpret the
//! bytes through [`bytemuck`] under a `Pod` bound, dispatching on
//! [`size_of::<F>()`](size_of) — the only two inhabited cases being the 4-byte
//! `f32` and 8-byte `f64` instantiations.
//!
//! These two functions were previously copy-pasted into ~30 algorithm and prim
//! modules with identical bodies (differing only in the `unreachable!` message).
//! Hoisting them here gives a single source of truth so the conversion logic
//! cannot drift between estimators.

use bytemuck::Pod;

/// Reinterpret an `F` (`f32` / `f64`) as an `f64` for host-side combine.
///
/// `f32` values widen to `f64`; `f64` values pass through unchanged.
///
/// # Panics
/// Panics if `F` is neither `f32` nor `f64` (no other float type is ever
/// instantiated in mlrs kernels).
#[inline]
pub fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("mlrs floats are f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (`f32` / `f64`) from an `f64`.
///
/// When `F = f32` the value is narrowed through `f64 as f32`; when `F = f64`
/// it passes through unchanged.
///
/// # Panics
/// Panics if `F` is neither `f32` nor `f64`.
#[inline]
pub fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("mlrs floats are f32/f64 only"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_to_f64_roundtrips_f32() {
        let x: f32 = 1.5;
        assert_eq!(host_to_f64(x), 1.5_f64);
    }

    #[test]
    fn host_to_f64_passthrough_f64() {
        let x: f64 = 1.5;
        assert_eq!(host_to_f64(x), 1.5_f64);
    }

    #[test]
    fn f64_to_host_narrows_f32() {
        let x: f32 = f64_to_host(1.5_f64);
        assert_eq!(x, 1.5_f32);
    }

    #[test]
    fn f64_to_host_passthrough_f64() {
        let x: f64 = f64_to_host(1.5_f64);
        assert_eq!(x, 1.5_f64);
    }

    #[test]
    fn roundtrip_f32() {
        let x: f32 = 0.1;
        let back: f32 = f64_to_host(host_to_f64(x));
        assert_eq!(x, back);
    }
}
