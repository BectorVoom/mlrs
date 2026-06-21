//! `LedoitWolf` (COV-02) — the Ledoit–Wolf shrinkage covariance estimator,
//! matching `sklearn.covariance.LedoitWolf`.
//!
//! ## Algorithm (exact sklearn 1.7.1 `ledoit_wolf_shrinkage`, RESEARCH Pattern 3)
//! The shrinkage estimator blends the empirical (MLE, `ddof = 0`) `covariance_`
//! toward a scaled-identity target `μ·I` by the closed-form optimal shrinkage
//! intensity `shrinkage_ ∈ [0, 1]`:
//!
//! ```text
//! X        = X_batch − mean        (unless assume_centered, D-07)        # n × p
//! emp_cov  = Xᵀ·X / n              (empirical_covariance, ddof = 0)      # p × p
//! X2       = X**2
//! emp_cov_trace = sum(X2, axis=0) / n                                    # length p
//! mu       = sum(emp_cov_trace) / p                                      # scalar
//! beta_    = sum( X2ᵀ @ X2 )                                             # scalar
//! delta_   = sum( (Xᵀ @ X)**2 ) / n²   (Frobenius² of the Gram, /n²)     # scalar
//! beta     = (1/(p·n)) · (beta_/n − delta_)
//! delta    = (delta_ − 2·mu·sum(emp_cov_trace) + p·mu²) / p
//! beta     = min(beta, delta)
//! shrinkage_ = 0 if beta == 0 else beta/delta
//! covariance_      = (1 − shrinkage_)·emp_cov
//! covariance_[diag] += shrinkage_·mu
//! ```
//!
//! ## ddof = 0 (RESEARCH Pitfall 1)
//! The empirical covariance under LedoitWolf is the MLE (`ddof = 0`,
//! `Xᵀ·X / n`), NOT the sample covariance (`ddof = 1`). The `β/δ/μ` closed form
//! is consistent with that normalisation.
//!
//! ## Host finalize (D-03)
//! `emp_cov` and the unnormalized Gram `Xᵀ·X` are small (`p × p`, `p ≤ 64`); the
//! `β/δ` scalar reductions over `X²` and the Gram are a HOST finalize in `f64`
//! (mirrors the kmeans inertia host-sum idiom). Fitted `covariance_` /
//! `location_` / `shrinkage_` are stored device-resident; the host accessors
//! materialize them on demand.
//!
//! Tests live in `crates/mlrs-algos/tests/ledoit_wolf_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use std::sync::OnceLock;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::AlgoError;
use crate::traits::Fit;

/// Ledoit–Wolf shrinkage covariance estimator (COV-02).
///
/// Construct with [`LedoitWolf::new`] (`assume_centered`), then [`Fit::fit`].
/// Fitted attributes are device-resident (D-03); the host accessors materialize
/// them on demand.
pub struct LedoitWolf<F> {
    /// When `true`, skip mean subtraction and set `location_ = 0` (D-07).
    assume_centered: bool,
    /// `covariance_` (`n_features × n_features`), row-major, device-resident.
    covariance_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `location_` (length `n_features`), device-resident.
    location_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `shrinkage_` ∈ [0, 1], the optimal Ledoit–Wolf shrinkage intensity.
    shrinkage_: Option<f64>,
    /// Memoized host copies of the device-resident attrs (IN-05). The host
    /// accessors download from the device once, then serve the cached `Vec<F>`
    /// on repeated access — the Python `@property` getters read these in loops.
    /// Reset on every `fit`.
    cov_host: OnceLock<Vec<F>>,
    loc_host: OnceLock<Vec<F>>,
}

impl<F> LedoitWolf<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `LedoitWolf`.
    ///
    /// - `assume_centered`: when `true`, the data is assumed already centered;
    ///   `location_` is set to `0` and no mean is subtracted (D-07).
    pub fn new(assume_centered: bool) -> Self {
        Self {
            assume_centered,
            covariance_: None,
            location_: None,
            shrinkage_: None,
            cov_host: OnceLock::new(),
            loc_host: OnceLock::new(),
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

    /// The fitted `shrinkage_` ∈ [0, 1]. Errors with `NotFitted` before `fit`.
    pub fn shrinkage_(&self) -> Result<f64, AlgoError> {
        self.shrinkage_.ok_or(AlgoError::NotFitted {
            estimator: "ledoit_wolf",
            operation: "shrinkage_",
        })
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
            estimator: "ledoit_wolf",
            operation,
        })?;
        Ok(cache.get_or_init(|| arr.to_host(pool)).clone())
    }
}

impl<F> Fit<F> for LedoitWolf<F>
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

        let n = n_samples as f64;
        let p = n_features as f64;

        // --- 1. location_ = column means (or 0 when assume_centered, D-07). The
        //        mean term is computed on-device via the Phase-2 reduction; the
        //        β/δ host finalize then runs on the small centered X. ---
        let (location_dev, mean64): (DeviceArray<ActiveRuntime, F>, Vec<f64>) =
            if self.assume_centered {
                let zeros = vec![F::from_int(0i64); n_features];
                (DeviceArray::from_host(pool, &zeros), vec![0.0f64; n_features])
            } else {
                let mean_dev = column_reduce::<F>(
                    pool,
                    x,
                    n_samples,
                    n_features,
                    ScalarOp::Mean,
                    ReducePath::Shared,
                )?
                .expect("shared path is never plane-gated to None");
                let mean_host = mean_dev.to_host(pool);
                let mean64: Vec<f64> = mean_host.iter().map(|&v| host_to_f64(v)).collect();
                (mean_dev, mean64)
            };

        // --- 2. Center X on host (X = X_batch − mean; a no-op when
        //        assume_centered since mean = 0). p ≤ 64 and these are small v2
        //        sizes, so the β/δ/Gram reductions are a single host pass in f64
        //        (RESEARCH Pattern 3 host finalize). ---
        let x_host = x.to_host(pool);
        let mut xc = vec![0.0f64; n_samples * n_features];
        for r in 0..n_samples {
            for c in 0..n_features {
                xc[r * n_features + c] = host_to_f64(x_host[r * n_features + c]) - mean64[c];
            }
        }

        // emp_cov = Xᵀ·X / n (ddof=0 MLE on the CENTERED X), and the unnormalized
        // Gram G = Xᵀ·X (reused for delta_). Both p × p, accumulated in f64.
        let mut gram = vec![0.0f64; n_features * n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                let mut acc = 0.0f64;
                for r in 0..n_samples {
                    acc += xc[r * n_features + i] * xc[r * n_features + j];
                }
                gram[i * n_features + j] = acc;
            }
        }
        // ddof=0 MLE normalisation: divide the centered Gram by n (NOT n−1,
        // RESEARCH Pitfall 1). ddof is hard-pinned to 0 for the MLE the β/δ form
        // requires — there is no configurable ddof here (IN-04).
        let emp_cov: Vec<f64> = gram.iter().map(|&g| g / n).collect();

        // emp_cov_trace = sum(X2, axis=0) / n  (length p) — the per-feature mean
        // of the squared centered entries (= diag(emp_cov)).
        let mut emp_cov_trace = vec![0.0f64; n_features];
        for c in 0..n_features {
            let mut acc = 0.0f64;
            for r in 0..n_samples {
                let v = xc[r * n_features + c];
                acc += v * v;
            }
            emp_cov_trace[c] = acc / n;
        }
        let trace_sum: f64 = emp_cov_trace.iter().sum();
        let mu = trace_sum / p;

        // beta_ = sum( X2ᵀ @ X2 ) = Σ_{i,j} Σ_t X2[t,i]·X2[t,j], where
        // X2[t,c] = xc[t,c]². Equivalently Σ_{i,j} ( Σ_t X2[t,i]·X2[t,j] ).
        // Compute the p×p matrix S = X2ᵀ·X2 and sum all entries.
        // delta_ = sum( G**2 ) = Σ_{i,j} G[i,j]²  (Frobenius² of Xᵀ·X).
        let mut beta_ = 0.0f64;
        for i in 0..n_features {
            for j in 0..n_features {
                let mut s = 0.0f64;
                for r in 0..n_samples {
                    let x2i = {
                        let v = xc[r * n_features + i];
                        v * v
                    };
                    let x2j = {
                        let v = xc[r * n_features + j];
                        v * v
                    };
                    s += x2i * x2j;
                }
                beta_ += s;
            }
        }
        // delta_ = sum( (Xᵀ·X)² ) / n²  — sklearn divides the Frobenius² of the
        // UNNORMALIZED Gram by n² (the `delta_ /= n_samples**2` step) BEFORE using
        // it in both beta and delta. beta_ is NOT divided by n².
        let delta_: f64 = gram.iter().map(|&g| g * g).sum::<f64>() / (n * n);

        // beta  = (1/(p·n)) · (beta_/n − delta_)
        // delta = (delta_ − 2·mu·trace_sum + p·mu²) / p
        let beta = (1.0 / (p * n)) * (beta_ / n - delta_);
        let delta = (delta_ - 2.0 * mu * trace_sum + p * mu * mu) / p;
        let beta = beta.min(delta);
        let mut shrinkage = if beta == 0.0 { 0.0 } else { beta / delta };
        // ∈ [0,1] by construction; clip anyway per COV-02 wording.
        shrinkage = shrinkage.clamp(0.0, 1.0);

        // --- 3. covariance_ = (1 − shrinkage)·emp_cov; add shrinkage·mu to the
        //        diagonal (shrink toward the μ·I target). ---
        let mut cov_out = vec![0.0f64; n_features * n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                cov_out[i * n_features + j] = (1.0 - shrinkage) * emp_cov[i * n_features + j];
            }
            cov_out[i * n_features + i] += shrinkage * mu;
        }
        let cov_host: Vec<F> = cov_out.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let covariance_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &cov_host);

        // --- 4. Store device-resident fitted state (D-03). ---
        self.covariance_ = Some(covariance_dev);
        self.location_ = Some(location_dev);
        self.shrinkage_ = Some(shrinkage);
        // Invalidate any memoized host copies from a previous fit (IN-05) so a
        // re-fit on the same instance never serves stale cached attrs.
        self.cov_host = OnceLock::new();
        self.loc_host = OnceLock::new();
        Ok(self)
    }
}
