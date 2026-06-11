//! SVD / PCA sign-alignment helper (FOUND-08).
//!
//! Singular vectors and PCA components are only defined up to a sign: a
//! component `v` and its negation `-v` are equally valid, so a backend that
//! produces `-v` would spuriously fail [`crate::compare::assert_close`] against
//! a sklearn oracle that produced `v`. This module canonicalizes the sign so
//! sign-flipped-but-equal vectors compare equal, while genuinely different
//! vectors still differ.
//!
//! The convention mirrors scikit-learn's `svd_flip` (the `u_based_decision`
//! variant used by PCA): make the element with the **largest absolute value**
//! positive, flipping the sign of the whole vector if that element is negative.
//! Ties (equal magnitudes) are broken by the lowest index, which is
//! deterministic.

/// Returns the sign (`+1.0` or `-1.0`) that, multiplied through `v`, makes the
/// largest-magnitude element non-negative.
///
/// An all-zero (or empty) vector has no informative element, so the canonical
/// sign is `+1.0` (a no-op).
pub fn canonical_sign(v: &[f64]) -> f64 {
    let mut best_idx: Option<usize> = None;
    let mut best_mag = 0.0_f64;
    for (i, &x) in v.iter().enumerate() {
        let mag = x.abs();
        // Strict `>` keeps the lowest index on ties — deterministic.
        if best_idx.is_none() || mag > best_mag {
            best_mag = mag;
            best_idx = Some(i);
        }
    }
    match best_idx {
        Some(i) if v[i] < 0.0 => -1.0,
        _ => 1.0,
    }
}

/// Returns a sign-canonicalized copy of `v` (largest-magnitude element made
/// non-negative). See [`canonical_sign`].
pub fn align_sign(v: &[f64]) -> Vec<f64> {
    let s = canonical_sign(v);
    v.iter().map(|&x| x * s).collect()
}

/// Canonicalize the sign of `v` in place.
pub fn align_sign_in_place(v: &mut [f64]) {
    let s = canonical_sign(v);
    if s < 0.0 {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
}

/// Sign-align every row of a row-major `n_components x n_features` matrix
/// independently (each component canonicalized on its own largest element).
///
/// This mirrors how sklearn flips each PCA component / singular vector
/// separately before comparison.
pub fn align_rows(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter().map(|r| align_sign(r)).collect()
}
