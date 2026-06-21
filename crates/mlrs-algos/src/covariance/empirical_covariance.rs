//! `EmpiricalCovariance` (COV-01) — the maximum-likelihood (biased, `ddof = 0`)
//! covariance estimator, matching `sklearn.covariance.EmpiricalCovariance`.
//!
//! ## Algorithm (sklearn `EmpiricalCovariance.fit`, RESEARCH-verified)
//! 1. `location_` = column means of `X` (or `0` when `assume_centered`, D-07).
//! 2. Center `X_c = X − location_` (skipped when `assume_centered`).
//! 3. `covariance_ = X_cᵀ · X_c / n` — the MLE Gram, `ddof = 0` (NOT `1`,
//!    RESEARCH Pitfall 1; `== np.cov(X, rowvar=False, bias=True)`), composed from
//!    the validated Phase-2 [`covariance`] primitive (the `ddof = 0` arm).
//! 4. `precision_` (only when `store_precision`, D-08) = `pinvh(covariance_)` via
//!    the Phase-3 symmetric [`eig`] (the eig-based pseudo-inverse, D-05):
//!    eigendecompose `covariance_ = V·diag(w)·Vᵀ`, floor near-zero eigenvalues to
//!    a zero inverse (`inv_w_i = 1/w_i` iff `|w_i| > cutoff`, else `0`, with
//!    `cutoff = (RCOND·max|w|).max(NEAR_ZERO_FLOOR)`), and reassemble
//!    `precision_ = V·diag(inv_w)·Vᵀ`. The eigenvalue floor makes this
//!    singular-safe for the rank-deficient `n ≤ p` case (D-05).
//!
//! ## eig-based pseudo-inverse (D-05)
//! `precision_` is the eig-based pseudo-inverse — deliberately NOT an SPD-only
//! factorization inverse: the MLE `covariance_` is singular whenever `n ≤ p`, and
//! an SPD-only factor would raise. The eigenvalue floor returns the
//! Moore–Penrose pseudo-inverse instead, exactly matching sklearn's
//! `linalg.pinvh`.
//!
//! ## Device residency (D-03)
//! `covariance_` / `location_` / `precision_` are stored as device-resident
//! [`DeviceArray`]s; the host accessors materialize them on demand. The pinvh
//! reassembly is a host finalize in `f64` (the covariance is at most `p × p` with
//! `p ≤ 64`, so the `O(p³)` host reassembly is cheap).
//!
//! Tests live in `crates/mlrs-algos/tests/empirical_covariance_test.rs`
//! (AGENTS.md §2), never an in-source `#[cfg(test)] mod tests`.

use std::sync::OnceLock;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::covariance::covariance;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::AlgoError;
use crate::traits::Fit;

/// Near-zero floor for the pinvh eigenvalue cutoff (mirrors the
/// `linear_regression.rs` precedent — below the 1e-5 tolerance so it never
/// loosens a real check). Keeps the cutoff strictly positive for a degenerate
/// (all-zero) spectrum so a tiny eigenvalue is always zeroed rather than
/// reciprocated.
const NEAR_ZERO_FLOOR: f64 = 1e-8;

/// Relative eigenvalue cutoff `rcond` for the pinvh — eigenvalues with
/// `|w_i| ≤ rcond·max|w|` are dropped (`inv_w = 0`). Pinned to `1e-6`, matching
/// the `linear_regression.rs` `RCOND` and sklearn's `pinvh` relative threshold
/// for the rank-deficient MLE covariance (D-05 / RESEARCH A1).
const RCOND: f64 = 1e-6;

/// Maximum-likelihood (biased, `ddof = 0`) covariance estimator (COV-01).
///
/// Construct with [`EmpiricalCovariance::new`] (`assume_centered`,
/// `store_precision`), then [`Fit::fit`]. Fitted attributes are device-resident
/// (D-03); the host accessors materialize them on demand.
pub struct EmpiricalCovariance<F> {
    /// When `true`, skip mean subtraction and set `location_ = 0` (D-07).
    assume_centered: bool,
    /// When `true`, compute and store `precision_ = pinvh(covariance_)` (D-08).
    store_precision: bool,
    /// `covariance_` (`n_features × n_features`), row-major, device-resident.
    covariance_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `location_` (length `n_features`), device-resident.
    location_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `precision_` (`n_features × n_features`), device-resident; `None` unless
    /// `store_precision`.
    precision_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Memoized host copies of the device-resident attrs (IN-05). The host
    /// accessors download from the device once, then serve the cached `Vec<F>`
    /// on repeated access — the Python `@property` getters read these in loops,
    /// so a per-call device→host copy is wasteful. Reset on every `fit`.
    cov_host: OnceLock<Vec<F>>,
    loc_host: OnceLock<Vec<F>>,
    prec_host: OnceLock<Vec<F>>,
}

impl<F> EmpiricalCovariance<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `EmpiricalCovariance`.
    ///
    /// - `assume_centered`: when `true`, the data is assumed already centered;
    ///   `location_` is set to `0` and no mean is subtracted (D-07).
    /// - `store_precision`: when `true`, the eig-based pinvh `precision_` is
    ///   computed and stored at `fit` (D-08).
    pub fn new(assume_centered: bool, store_precision: bool) -> Self {
        Self {
            assume_centered,
            store_precision,
            covariance_: None,
            location_: None,
            precision_: None,
            cov_host: OnceLock::new(),
            loc_host: OnceLock::new(),
            prec_host: OnceLock::new(),
        }
    }

    /// Host copy of `covariance_` (`n_features × n_features`, row-major).
    /// Memoized after the first call (IN-05).
    pub fn covariance_(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.attr(&self.covariance_, &self.cov_host, pool, "covariance_")
    }

    /// Host copy of `location_` (length `n_features`). Memoized (IN-05).
    pub fn location_(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.attr(&self.location_, &self.loc_host, pool, "location_")
    }

    /// Host copy of `precision_` (`n_features × n_features`). Errors with
    /// `NotFitted` when `store_precision` was `false`. Memoized (IN-05).
    pub fn precision_(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.attr(&self.precision_, &self.prec_host, pool, "precision_")
    }

    /// Materialize a device-resident attr to the host, caching the result so
    /// repeated accesses (e.g. the Python `@property` getters in a loop) skip the
    /// device→host copy after the first call (IN-05). The cache is reset on every
    /// `fit`, so it never serves stale state.
    fn attr(
        &self,
        slot: &Option<DeviceArray<ActiveRuntime, F>>,
        cache: &OnceLock<Vec<F>>,
        pool: &BufferPool<ActiveRuntime>,
        operation: &'static str,
    ) -> Result<Vec<F>, AlgoError> {
        let arr = slot.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "empirical_covariance",
            operation,
        })?;
        Ok(cache.get_or_init(|| arr.to_host(pool)).clone())
    }
}

impl<F> Fit<F> for EmpiricalCovariance<F>
where
    F: Float + CubeElement + Pod,
{
    fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        // `_y` is unused: the retained `Fit`-trait slot for Phase-10 MBSGD reuse
        // (this estimator is unsupervised; see traits.rs) — not unfinished wiring
        // (IN-02).
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-07-07 / ASVS V5: reject inconsistent geometry BEFORE any prim
        //     launch (untrusted shapes → typed error, not an OOB device read). ---
        if n_features == 0 || n_samples == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }

        // --- 1. location_ = column means (or 0 when assume_centered, D-07). ---
        let location_dev: DeviceArray<ActiveRuntime, F> = if self.assume_centered {
            DeviceArray::from_host(pool, &vec![F::from_int(0i64); n_features])
        } else {
            column_reduce::<F>(
                pool,
                x,
                n_samples,
                n_features,
                ScalarOp::Mean,
                ReducePath::Shared,
            )?
            .expect("shared path is never plane-gated to None")
        };

        // --- 2. covariance_ = MLE Gram, ddof=0 (Pitfall 1; the prim folds ddof
        //        into the scale). The Phase-2 covariance prim ALWAYS subtracts the
        //        column means before forming Xᵀ·X. For the default
        //        (!assume_centered) path that centering IS the desired MLE
        //        covariance (sklearn subtracts the same means). For the
        //        assume_centered path (D-07) sklearn divides Xᵀ·X by n WITHOUT
        //        centering, so the prim's mandatory centering would be wrong —
        //        that case is built directly via the uncentered host Gram. ---
        let covariance_dev: DeviceArray<ActiveRuntime, F> = if self.assume_centered {
            mle_gram_uncentered::<F>(pool, x, n_samples, n_features)
        } else {
            covariance::<F>(pool, x, (n_samples, n_features), /*ddof=*/ 0, None)?
        };

        // --- 3. precision_ = pinvh(covariance_) via eig (D-05, the eig-based
        //        pseudo-inverse, NOT an SPD-only factor), only when
        //        store_precision (D-08). ---
        let precision_dev: Option<DeviceArray<ActiveRuntime, F>> = if self.store_precision {
            Some(pinvh::<F>(pool, &covariance_dev, n_features)?)
        } else {
            None
        };

        // --- 4. Store device-resident fitted state (D-03). ---
        self.covariance_ = Some(covariance_dev);
        self.location_ = Some(location_dev);
        self.precision_ = precision_dev;
        // Invalidate any memoized host copies from a previous fit (IN-05) so a
        // re-fit on the same instance never serves stale cached attrs.
        self.cov_host = OnceLock::new();
        self.loc_host = OnceLock::new();
        self.prec_host = OnceLock::new();
        Ok(self)
    }
}

/// MLE Gram `Xᵀ·X / n` with NO mean removal (the `assume_centered = true` path,
/// D-07). The covariance prim always subtracts the column means, so the
/// assume-centered case (which sklearn computes WITHOUT centering) is built here
/// directly: read `X` to host, accumulate `Xᵀ·X` in `f64`, scale by `1/n`, and
/// upload the `p × p` Gram. `p ≤ 64` so the host `O(n·p²)` accumulation is cheap.
fn mle_gram_uncentered<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_features: usize,
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let x_host = x.to_host(pool);
    let x64: Vec<f64> = x_host.iter().map(|&v| host_to_f64(v)).collect();
    let inv_n = 1.0_f64 / (n_samples as f64);
    let mut gram = vec![0.0f64; n_features * n_features];
    for i in 0..n_features {
        for j in 0..n_features {
            let mut acc = 0.0f64;
            for r in 0..n_samples {
                acc += x64[r * n_features + i] * x64[r * n_features + j];
            }
            gram[i * n_features + j] = acc * inv_n;
        }
    }
    let gram_host: Vec<F> = gram.iter().map(|&v| f64_to_host::<F>(v)).collect();
    DeviceArray::from_host(pool, &gram_host)
}

/// Symmetric pseudo-inverse `pinvh(cov)` via the Phase-3 [`eig`] (D-05, the
/// eig-based pseudo-inverse, NOT an SPD-only factor). Eigendecompose
/// `cov = V·diag(w)·Vᵀ`, floor near-zero eigenvalues
/// to a zero inverse, and reassemble `precision_ = V·diag(inv_w)·Vᵀ` on the host
/// respecting `eig`'s column-major `V` layout (`v[c*n + r] = V[r, c]`). The floor
/// makes the singular `n ≤ p` case finite (no inf/NaN), matching sklearn's
/// `linalg.pinvh`.
fn pinvh<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    cov: &DeviceArray<ActiveRuntime, F>,
    n: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    // eig: cov is symmetric-by-construction (TRUSTED, D-06); returns w descending
    // and V column-major (v[c*n + r] = V[r, c]).
    let (w, v) = eig::<F>(pool, cov, n, None)?;
    let w_host = w.to_host(pool);
    let v_host = v.to_host(pool);
    let w64: Vec<f64> = w_host.iter().map(|&x| host_to_f64(x)).collect();
    let v64: Vec<f64> = v_host.iter().map(|&x| host_to_f64(x)).collect();
    w.release_into(pool);
    v.release_into(pool);

    // cutoff = (RCOND · max|w|).max(NEAR_ZERO_FLOOR), reusing the
    // linear_regression constants (RESEARCH A1). inv_w_i = 1/w_i iff
    // |w_i| > cutoff, else 0 (floored — handles rank-deficient n≤p without
    // inf/NaN).
    let w_abs_max = w64.iter().fold(0.0f64, |m, &wi| m.max(wi.abs()));
    let cutoff = (RCOND * w_abs_max).max(NEAR_ZERO_FLOOR);
    let inv_w: Vec<f64> = w64
        .iter()
        .map(|&wi| if wi.abs() > cutoff { 1.0 / wi } else { 0.0 })
        .collect();

    // precision_[r, s] = Σ_c V[r, c] · inv_w[c] · V[s, c]
    //                  = Σ_c v[c*n + r] · inv_w[c] · v[c*n + s].
    let mut precision = vec![0.0f64; n * n];
    for r in 0..n {
        for s in 0..n {
            let mut acc = 0.0f64;
            for c in 0..n {
                acc += v64[c * n + r] * inv_w[c] * v64[c * n + s];
            }
            precision[r * n + s] = acc;
        }
    }
    let precision_host: Vec<F> = precision.iter().map(|&v| f64_to_host::<F>(v)).collect();
    Ok(DeviceArray::from_host(pool, &precision_host))
}
