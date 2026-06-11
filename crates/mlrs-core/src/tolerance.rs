//! Numerical tolerance policy for the scikit-learn oracle harness.
//!
//! Per **D-08**, the project starts with a *single global tolerance* rather
//! than a populated per-estimator-family table. The `Tolerance` struct plus
//! the `F32_TOL` / `F64_TOL` constants are the global default; the
//! [`Tolerance::for_family`] hook is the documented growth point where
//! per-family rows are added later (Phase 3/4/5) WITHOUT restructuring the
//! call sites that already use `for_family(...)`.
//!
//! See `docs/tolerance-policy.md` for the documented policy and the near-zero
//! floor rationale.

/// Absolute + relative error bounds used by [`crate::compare::is_close`].
///
/// Both fields are `f64` even for the f32 policy: comparisons are performed in
/// `f64` so the tolerance itself is never the limiting precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Maximum allowed absolute error `|got - expected|`.
    pub abs: f64,
    /// Maximum allowed relative error `|got - expected| / |expected|`.
    pub rel: f64,
}

/// Global default tolerance for `f32`-precision oracle comparisons (D-08).
pub const F32_TOL: Tolerance = Tolerance {
    abs: 1e-5,
    rel: 1e-5,
};

/// Global default tolerance for `f64`-precision oracle comparisons (D-08).
///
/// Identical to [`F32_TOL`] today; kept as a separate constant so an f64 path
/// can be tightened independently later without touching call sites.
pub const F64_TOL: Tolerance = Tolerance {
    abs: 1e-5,
    rel: 1e-5,
};

impl Tolerance {
    /// Construct a tolerance from explicit absolute and relative bounds.
    pub const fn new(abs: f64, rel: f64) -> Self {
        Self { abs, rel }
    }

    /// Per-estimator-family tolerance lookup — the **growth point** for D-08.
    ///
    /// Today every family resolves to the single global default ([`F32_TOL`]).
    /// When a family (e.g. `"pca"`, `"kmeans"`) demonstrates it needs looser
    /// bounds, add a `match family { ... }` arm here; existing callers that
    /// already pass a family name pick up the new row automatically. This is
    /// the documented extension path, not a populated table (D-08 / FOUND-08).
    pub fn for_family(_family: &str) -> Self {
        // Intentionally returns the global default for all families in Phase 1.
        // New rows are added here as families need them (Phase 3/4/5).
        F32_TOL
    }
}
