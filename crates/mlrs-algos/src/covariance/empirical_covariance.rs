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

use std::marker::PhantomData;
use std::sync::OnceLock;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::covariance::covariance;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Unfit};

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
/// Construct with the zero-arg [`EmpiricalCovariance::new`] (sklearn defaults:
/// `assume_centered = false`, `store_precision = true`) or
/// [`EmpiricalCovariance::builder`] (`.assume_centered(bool)`/
/// `.store_precision(bool)`), then the consuming [`Fit::fit`] (returns the
/// `Fitted`-tagged sibling). Fitted attributes are device-resident (D-03); the
/// host accessors materialize them on demand and exist ONLY on
/// `EmpiricalCovariance<F, Fitted>` (the compile-time typestate replaces the old
/// runtime `NotFitted` guard, D-03).
pub struct EmpiricalCovariance<F, S = Unfit> {
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
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> EmpiricalCovariance<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfitted `EmpiricalCovariance` with sklearn's defaults
    /// (`assume_centered = false`, `store_precision = true`) directly in the
    /// `Unfit` state. SINGLE source of truth for the defaults (D-08): the builder
    /// `Default` re-derives via [`EmpiricalCovariance::into_builder`].
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
            _state: PhantomData,
        }
    }

    /// Start building an `EmpiricalCovariance` from sklearn's defaults (D-08
    /// single source).
    pub fn builder() -> EmpiricalCovarianceBuilder {
        EmpiricalCovarianceBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`EmpiricalCovarianceBuilder::default`] to
    /// re-derive the defaults from [`EmpiricalCovariance::new`] (D-08).
    pub fn into_builder(self) -> EmpiricalCovarianceBuilder {
        EmpiricalCovarianceBuilder {
            assume_centered: self.assume_centered,
            store_precision: self.store_precision,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators. Used by the
    /// defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.assume_centered == other.assume_centered
            && self.store_precision == other.store_precision
    }
}

impl<F> Default for EmpiricalCovariance<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new(false, true)
    }
}

/// Builder for [`EmpiricalCovariance`] (D-01). `Default` re-derives the sklearn
/// defaults from [`EmpiricalCovariance::new`] (D-08 single source).
#[derive(Debug, Clone, Copy)]
pub struct EmpiricalCovarianceBuilder {
    assume_centered: bool,
    store_precision: bool,
}

impl Default for EmpiricalCovarianceBuilder {
    /// Re-derive the sklearn defaults from [`EmpiricalCovariance::new`] (D-08
    /// single source). `f64` is pinned only to read the F-independent flag
    /// defaults — the builder is non-generic.
    fn default() -> Self {
        EmpiricalCovariance::<f64, Unfit>::new(false, true).into_builder()
    }
}

impl EmpiricalCovarianceBuilder {
    /// Set whether the data is assumed already centered (D-07).
    pub fn assume_centered(mut self, v: bool) -> Self {
        self.assume_centered = v;
        self
    }

    /// Set whether the eig-based pinvh `precision_` is computed and stored at
    /// `fit` (D-08).
    pub fn store_precision(mut self, v: bool) -> Self {
        self.store_precision = v;
        self
    }

    /// Build the (unfit) estimator. EmpiricalCovariance has NO data-INDEPENDENT
    /// hyperparameter to validate at construction (both knobs are booleans), so
    /// `build()` is infallible-but-typed (kept for family uniformity so the
    /// `build_err_to_py` PyO3 mapper is shape-identical).
    pub fn build<F>(self) -> Result<EmpiricalCovariance<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        Ok(EmpiricalCovariance {
            assume_centered: self.assume_centered,
            store_precision: self.store_precision,
            covariance_: None,
            location_: None,
            precision_: None,
            cov_host: OnceLock::new(),
            loc_host: OnceLock::new(),
            prec_host: OnceLock::new(),
            _state: PhantomData,
        })
    }
}

impl<F> EmpiricalCovariance<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of `covariance_` (`n_features × n_features`, row-major).
    /// Memoized after the first call (IN-05). `Some` by construction on `Fitted`.
    pub fn covariance_(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.cov_host
            .get_or_init(|| {
                self.covariance_
                    .as_ref()
                    .expect("covariance_ is Some by construction on Fitted")
                    .to_host(pool)
            })
            .clone()
    }

    /// Host copy of `location_` (length `n_features`). Memoized (IN-05). `Some`
    /// by construction on `Fitted`.
    pub fn location_(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.loc_host
            .get_or_init(|| {
                self.location_
                    .as_ref()
                    .expect("location_ is Some by construction on Fitted")
                    .to_host(pool)
            })
            .clone()
    }

    /// Host copy of `precision_` (`n_features × n_features`). Errors with
    /// `NotFitted` when `store_precision` was `false` (the attribute was not
    /// stored — a runtime "not stored" condition, distinct from the unfitted
    /// state which the typestate now rules out). Memoized (IN-05).
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

impl<F> Fit<F> for EmpiricalCovariance<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = EmpiricalCovariance<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        // `_y` is unused: the retained `Fit`-trait slot for Phase-10 MBSGD reuse
        // (this estimator is unsupervised; see typestate.rs) — not unfinished
        // wiring (IN-02).
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<EmpiricalCovariance<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-07-07 / ASVS V5: reject inconsistent geometry BEFORE any prim
        //     launch (untrusted shapes → typed error, not an OOB device read). ---
        validate_geometry(x, shape)?;

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

        // --- 4. Reconstruct into the device-resident `Fitted` value (D-03). The
        //        memo caches start fresh (`OnceLock::new()`) — the `Unfit` value's
        //        caches were always empty (no accessor exists on `Unfit`). ---
        Ok(EmpiricalCovariance {
            assume_centered: self.assume_centered,
            store_precision: self.store_precision,
            covariance_: Some(covariance_dev),
            location_: Some(location_dev),
            precision_: precision_dev,
            cov_host: OnceLock::new(),
            loc_host: OnceLock::new(),
            prec_host: OnceLock::new(),
            _state: PhantomData,
        })
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
