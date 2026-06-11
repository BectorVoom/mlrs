//! Float comparison core for the scikit-learn oracle (D-09).
//!
//! [`is_close`] requires **both** absolute AND relative error to pass (the
//! stricter form, not numpy's abs-OR-rel), with a documented near-zero guard:
//! when `|expected|` is below [`NEAR_ZERO_FLOOR`] the relative term would
//! explode for genuinely-correct near-zero results, so the check falls back to
//! abs-only. See `docs/tolerance-policy.md`.

use crate::tolerance::Tolerance;

/// Below this magnitude of `expected`, the relative-error term is unstable
/// (a tiny absolute error produces a huge relative error), so [`is_close`]
/// falls back to an absolute-only check.
///
/// `1e-8` sits comfortably below the `1e-5` absolute tolerance: any value the
/// guard admits is already within the absolute bound, so the guard only ever
/// *prevents spurious relative-error failures* — it never loosens the abs
/// check. Rationale documented in `docs/tolerance-policy.md`.
pub const NEAR_ZERO_FLOOR: f64 = 1e-8;

/// Returns `true` when `got` matches `expected` within `tol`.
///
/// - If `|expected| < NEAR_ZERO_FLOOR`: pass when `abs_err <= tol.abs`
///   (near-zero guard — abs-only, D-09 ⚠).
/// - Otherwise: pass when `abs_err <= tol.abs` **AND** `rel_err <= tol.rel`
///   (BOTH must hold, D-09).
///
/// NaNs never compare close; matching infinities compare close.
pub fn is_close(got: f64, expected: f64, tol: &Tolerance) -> bool {
    if got == expected {
        // Catches the exact-equal case including matching infinities.
        return true;
    }
    if got.is_nan() || expected.is_nan() || got.is_infinite() || expected.is_infinite() {
        return false;
    }
    let abs_err = (got - expected).abs();
    if expected.abs() < NEAR_ZERO_FLOOR {
        // Near-zero guard: the relative term explodes; fall back to abs-only.
        return abs_err <= tol.abs;
    }
    let rel_err = abs_err / expected.abs();
    abs_err <= tol.abs && rel_err <= tol.rel
}

/// Panics with an informative message when `got` is not close to `expected`.
///
/// The panic reports got / expected / absolute error / relative error so a
/// failing oracle comparison is immediately diagnosable.
pub fn assert_close(got: f64, expected: f64, tol: &Tolerance) {
    if !is_close(got, expected, tol) {
        let abs_err = (got - expected).abs();
        let rel_err = if expected.abs() < NEAR_ZERO_FLOOR {
            f64::NAN // undefined under the near-zero guard
        } else {
            abs_err / expected.abs()
        };
        panic!(
            "assert_close failed: got={got:e}, expected={expected:e}, \
             abs_err={abs_err:e} (tol.abs={:e}), rel_err={rel_err:e} (tol.rel={:e})",
            tol.abs, tol.rel
        );
    }
}

/// Element-wise [`assert_close`] over two equal-length slices.
///
/// Panics if the lengths differ, or on the first element that is not close
/// (reporting the offending index alongside the value detail).
pub fn assert_slice_close(got: &[f64], expected: &[f64], tol: &Tolerance) {
    assert_eq!(
        got.len(),
        expected.len(),
        "assert_slice_close: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if !is_close(g, e, tol) {
            let abs_err = (g - e).abs();
            let rel_err = if e.abs() < NEAR_ZERO_FLOOR {
                f64::NAN
            } else {
                abs_err / e.abs()
            };
            panic!(
                "assert_slice_close failed at index {i}: got={g:e}, expected={e:e}, \
                 abs_err={abs_err:e} (tol.abs={:e}), rel_err={rel_err:e} (tol.rel={:e})",
                tol.abs, tol.rel
            );
        }
    }
}
