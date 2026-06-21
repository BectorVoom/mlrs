//! `IncrementalPCA` (DECOMP-03) — streaming principal component analysis that
//! merges one batch at a time via the PRIM-07 incremental-SVD merge (Plan 07-03),
//! matching `sklearn.decomposition.IncrementalPCA`.
//!
//! ## Algorithm (sklearn `IncrementalPCA`, RESEARCH Pattern 1)
//! Unlike [`Pca`](super::pca::Pca), which runs a single SVD of the whole centered
//! design, `IncrementalPCA` maintains a *running* thin decomposition and folds in
//! each new batch with a small stacked re-SVD. The heavy lifting lives in
//! [`mlrs_backend::prims::incremental_svd::merge`] — this estimator is the thin
//! sklearn-faithful driver around it (D-01):
//!
//! - [`PartialFit::partial_fit`] merges a single batch into the running state
//!   (calling `merge` once), accumulating `n_samples_seen_`.
//! - [`Fit::fit`] is sklearn-faithful (D-02): it RESETS all fitted state, computes
//!   `batch_size = self.batch_size.unwrap_or(5 · n_features)` (D-03), then loops
//!   `partial_fit` over `gen_batches(n_samples, batch_size)` (equal batches, the
//!   trailing remainder folded in — exactly sklearn's `gen_batches`).
//! - [`Transform::transform`] projects `(X − mean_) · components_ᵀ`; with
//!   `whiten=True` each component direction is additionally scaled by
//!   `1/sqrt(explained_variance_[i])` so the output has unit per-component
//!   variance (D-06). [`Transform::inverse_transform`] un-whitens then
//!   reconstructs `Z · components_ + mean_`.
//!
//! ## ddof distinction (Pitfall 1)
//! `explained_variance_ = S²/(n_total − 1)` (ddof=1, computed in the merge),
//! DISTINCT from the covariance estimators' ddof=0. The
//! `explained_variance_ratio_` denominator is `sum(col_var) · n_total` (Pitfall 6,
//! NOT the truncated S² sum) — both produced by the merge.
//!
//! ## Device residency (D-03)
//! The running [`IncrementalSvdState`] keeps `components_` device-resident (as
//! `f64`); the small running statistics (`singular_values_`, `mean_`, `var_`,
//! `explained_variance_*`, `n_samples_seen_`) live host-side in `f64`. The host
//! accessors materialize each attribute in the estimator's `F` precision on
//! demand; `transform`/`inverse_transform` run the heavy `gemm` on-device.
//!
//! Tests live in `crates/mlrs-algos/tests/incremental_pca_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::incremental_svd::{merge, IncrementalSvdState};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::AlgoError;
use crate::traits::{Fit, PartialFit, Transform};

/// Whiten-scale floor: `1/sqrt(explained_variance_)` is guarded against a
/// (near-)zero variance so a degenerate component never produces a non-finite
/// scale (mirrors sklearn, which divides by `sqrt(explained_variance_)` directly
/// but never hits zero on a fitted component). Below this floor the scale is left
/// at 1 (the component carries no whitened energy anyway).
const WHITEN_VAR_FLOOR: f64 = 1e-12;

/// Streaming principal component analysis (DECOMP-03), fitted by merging batches
/// through the PRIM-07 incremental-SVD merge.
///
/// Construct with [`IncrementalPCA::new`] (`n_components`, `whiten`,
/// `batch_size`), then either stream batches with [`PartialFit::partial_fit`] or
/// fit the whole matrix in one call with [`Fit::fit`] (which loops `partial_fit`
/// over `gen_batches`). Fitted attributes are exposed via the host accessors.
pub struct IncrementalPCA<F> {
    /// Number of components to keep (`1 ..= min(n_samples_seen, n_features)`).
    n_components: usize,
    /// Whether to whiten the transform output (scale by `1/sqrt(ev_)`, D-06).
    whiten: bool,
    /// Explicit `partial_fit` batch size for `fit()`; `None` → `5 · n_features`
    /// at fit time (D-03/D-09).
    batch_size: Option<usize>,
    /// The running incremental-SVD state (`None` until the first `partial_fit`).
    state: Option<IncrementalSvdState>,
    /// `n_features` seen at fit, for `transform`/`inverse_transform` geometry.
    n_features: usize,
    /// Phantom: the estimator is generic over the upload/compute precision `F`,
    /// even though the running statistics are kept in `f64`.
    _marker: std::marker::PhantomData<F>,
}

impl<F> IncrementalPCA<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `IncrementalPCA`.
    ///
    /// - `n_components` — number of principal components to retain
    ///   (`1 ..= min(n_samples, n_features)`; validated at fit).
    /// - `whiten` — if `true`, scale the transform output so each component has
    ///   unit variance (D-06).
    /// - `batch_size` — explicit `partial_fit` batch size used by [`Fit::fit`];
    ///   `None` defaults to `5 · n_features` at fit time (D-03/D-09).
    pub fn new(n_components: usize, whiten: bool, batch_size: Option<usize>) -> Self {
        Self {
            n_components,
            whiten,
            batch_size,
            state: None,
            n_features: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Total samples merged so far (`n_samples_seen_`), accumulated across
    /// `partial_fit` calls (D-03). `0` before the first batch.
    pub fn n_samples_seen(&self) -> usize {
        self.state.as_ref().map(|s| s.n_samples_seen_).unwrap_or(0)
    }

    /// The configured `n_components` hyperparameter (sklearn `__init__` arg).
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// The configured `whiten` hyperparameter (sklearn `__init__` arg).
    pub fn whiten(&self) -> bool {
        self.whiten
    }

    /// The configured `batch_size` hyperparameter (`None` → `5·n_features` at
    /// fit; sklearn `__init__` arg).
    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }

    /// Host copy of `components_` (`n_components × n_features`, row-major), in the
    /// estimator precision `F`.
    pub fn components(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        let s = self.fitted("components_")?;
        let comp64 = s.components_.to_host(pool);
        Ok(comp64.iter().map(|&v| f64_to_host::<F>(v)).collect())
    }

    /// Host copy of `explained_variance_` (length `n_components`), `S²/(n−1)`.
    pub fn explained_variance(&self, _pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        let s = self.fitted("explained_variance_")?;
        Ok(s.explained_variance_.iter().map(|&v| f64_to_host::<F>(v)).collect())
    }

    /// Host copy of `explained_variance_ratio_` (length `n_components`); the
    /// denominator is `sum(col_var)·n_total` (Pitfall 6).
    pub fn explained_variance_ratio(
        &self,
        _pool: &BufferPool<ActiveRuntime>,
    ) -> Result<Vec<F>, AlgoError> {
        let s = self.fitted("explained_variance_ratio_")?;
        Ok(s.explained_variance_ratio_.iter().map(|&v| f64_to_host::<F>(v)).collect())
    }

    /// Host copy of `singular_values_` (length `n_components`).
    pub fn singular_values(&self, _pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        let s = self.fitted("singular_values_")?;
        Ok(s.singular_values_.iter().map(|&v| f64_to_host::<F>(v)).collect())
    }

    /// Host copy of `mean_` (length `n_features`), the running per-feature mean.
    pub fn mean(&self, _pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        let s = self.fitted("mean_")?;
        Ok(s.mean_.iter().map(|&v| f64_to_host::<F>(v)).collect())
    }

    /// Host copy of `var_` (length `n_features`), the running per-feature
    /// population variance (ddof=0, matches sklearn's `var_`).
    pub fn var(&self, _pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        let s = self.fitted("var_")?;
        Ok(s.var_.iter().map(|&v| f64_to_host::<F>(v)).collect())
    }

    /// Borrow the fitted running state or return [`AlgoError::NotFitted`].
    fn fitted(&self, operation: &'static str) -> Result<&IncrementalSvdState, AlgoError> {
        self.state.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "incremental_pca",
            operation,
        })
    }

    /// Validate a single batch's hyperparameters/geometry BEFORE any merge
    /// (ASVS V5 / T-07-09): reject an out-of-range `n_components` and a malformed
    /// `(b, p)` geometry. `n_features` must agree with the running state after the
    /// first batch.
    fn validate_batch(&self, b: usize, p: usize, x_len: usize) -> Result<(), AlgoError> {
        // Geometry consistency before launch.
        if b == 0 || p == 0 || x_len != b * p {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: b,
                cols: p,
                len: x_len,
            }));
        }
        // n_features must agree with the running state after the first batch.
        if let Some(s) = self.state.as_ref() {
            if p != s.n_features {
                return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                    operand: "x",
                    rows: b,
                    cols: p,
                    len: x_len,
                }));
            }
        }
        // n_components must be 1..=min(batch_rows, n_features) — for the FIRST
        // batch the SVD is of the b×p batch alone, so n_components cannot exceed
        // min(b, p) (T-07-09).
        let max_nc = b.min(p);
        if self.n_components == 0 || self.n_components > max_nc {
            return Err(AlgoError::InvalidNComponents {
                estimator: "incremental_pca",
                requested: self.n_components,
                max: max_nc,
            });
        }
        Ok(())
    }
}

impl<F> PartialFit<F> for IncrementalPCA<F>
where
    F: Float + CubeElement + Pod,
{
    fn partial_fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        let (b, p) = shape;
        self.validate_batch(b, p, x.len())?;

        // Merge this batch into the running state (PRIM-07 incremental_svd). The
        // merge owns the Chan-Golub-LeVeque running mean/var update, the
        // stacked-matrix branch (first vs subsequent), the SVD-cap validation,
        // `align_rows`, and the ddof=1 / ratio finalize — we consume it (D-01).
        let new_state = merge::<F>(pool, self.state.take(), x, (b, p), self.n_components)?;
        self.n_features = new_state.n_features;
        self.state = Some(new_state);
        Ok(self)
    }
}

impl<F> Fit<F> for IncrementalPCA<F>
where
    F: Float + CubeElement + Pod,
{
    fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- ASVS V5 (T-07-09): reject a malformed geometry + an out-of-range
        //     n_components BEFORE any batch. n_components must be
        //     1..=min(n_samples, n_features) for the overall fit. ---
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        let max_nc = n_samples.min(n_features);
        if self.n_components == 0 || self.n_components > max_nc {
            return Err(AlgoError::InvalidNComponents {
                estimator: "incremental_pca",
                requested: self.n_components,
                max: max_nc,
            });
        }

        // --- batch_size = self.batch_size.unwrap_or(5 * n_features) (D-03). A
        //     caller-supplied batch_size must be >= 1 (ASVS V5 / T-07-09). ---
        let batch_size = match self.batch_size {
            Some(bs) => {
                if bs < 1 {
                    return Err(AlgoError::InvalidBatchSize {
                        estimator: "incremental_pca",
                        batch_size: bs,
                    });
                }
                bs
            }
            None => 5 * n_features,
        };

        // --- WR-01: reject n_components > batch_size up front, matching
        //     sklearn's "n_components=L must be <= batch number of samples B".
        //     Without this, gen_batches emits a leading size-`batch_size` batch
        //     that fails inside validate_batch with a `max` referencing the
        //     batch row count, not the n_components/batch_size relationship the
        //     caller got wrong. ---
        if self.n_components > batch_size {
            return Err(AlgoError::InvalidNComponents {
                estimator: "incremental_pca",
                requested: self.n_components,
                max: batch_size,
            });
        }

        // --- sklearn-faithful fit (D-02): RESET all fitted state, then loop
        //     partial_fit over gen_batches(n_samples, batch_size). The whole
        //     matrix is already on-device; slice each batch on the host and
        //     re-upload (the batches are tiny — the SVD merge is the cost). ---
        self.state = None;
        self.n_features = 0;

        let x_host = x.to_host(pool);
        for (start, end) in gen_batches(n_samples, batch_size, self.n_components) {
            let b = end - start;
            let batch_host: Vec<F> = x_host[start * n_features..end * n_features].to_vec();
            let batch_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &batch_host);
            let res = self.partial_fit(pool, &batch_dev, None, (b, n_features));
            batch_dev.release_into(pool);
            res?;
        }
        Ok(self)
    }
}

impl<F> Transform<F> for IncrementalPCA<F>
where
    F: Float + CubeElement + Pod,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_samples, n_features) = shape;
        let s = self.fitted("transform")?;
        if n_features != self.n_features || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        let nc = s.n_components;

        // components_ scaled for whitening: with whiten=True each component
        // direction is divided by sqrt(explained_variance_[i]) so the projection
        // has unit per-component variance (D-06). Build the (possibly whitened)
        // components in the estimator precision F.
        let comp64 = s.components_.to_host(pool);
        let scales = self.whiten_scales(s);
        let mut comp_host: Vec<F> = vec![F::from_int(0i64); nc * n_features];
        for j in 0..nc {
            for c in 0..n_features {
                comp_host[j * n_features + c] =
                    f64_to_host::<F>(comp64[j * n_features + c] * scales[j]);
            }
        }
        let comp_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &comp_host);

        // Center X on-host by the running mean_ (a tiny length-n_features vector).
        let mean64: Vec<f64> = s.mean_.clone();
        let x_host = x.to_host(pool);
        let mut x_centered: Vec<F> = vec![F::from_int(0i64); n_samples * n_features];
        for r in 0..n_samples {
            for c in 0..n_features {
                let v = host_to_f64(x_host[r * n_features + c]) - mean64[c];
                x_centered[r * n_features + c] = f64_to_host::<F>(v);
            }
        }
        let x_c_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_centered);

        // Z = X_c · components_ᵀ  (m × nc). components_ is (nc × n_features)
        // row-major; transb reads it as componentsᵀ — no transpose buffer (D-06).
        let z = gemm::<F>(
            pool,
            &x_c_dev,
            (n_samples, n_features),
            &comp_dev,
            (n_features, nc),
            false,
            true,
            None,
        )?;
        x_c_dev.release_into(pool);
        comp_dev.release_into(pool);
        Ok(z)
    }

    fn inverse_transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        z: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_samples, n_components) = shape;
        let s = self.fitted("inverse_transform")?;
        if n_components != s.n_components || z.len() != n_samples * n_components {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "z",
                rows: n_samples,
                cols: n_components,
                len: z.len(),
            }));
        }
        let n_features = self.n_features;
        let nc = n_components;

        // For whiten=True the latent Z has unit-variance columns; un-whiten by
        // MULTIPLYING each component direction back by sqrt(explained_variance_[i])
        // before the reconstruction GEMM (the inverse of the transform scale, D-06).
        let comp64 = s.components_.to_host(pool);
        let scales = self.whiten_scales(s); // 1/sqrt(ev) (or 1 for whiten=False)
        let mut comp_host: Vec<F> = vec![F::from_int(0i64); nc * n_features];
        for j in 0..nc {
            // un-whiten = divide the transform-scale back out = multiply by
            // 1/scales[j] = sqrt(ev[j]); for whiten=False scales[j]==1.
            let unwhiten = 1.0 / scales[j];
            for c in 0..n_features {
                comp_host[j * n_features + c] =
                    f64_to_host::<F>(comp64[j * n_features + c] * unwhiten);
            }
        }
        let comp_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &comp_host);

        // X̂_c = Z · components_  (m × n_features). components_ is
        // (nc × n_features) row-major; read as-is (no transpose).
        let recon = gemm::<F>(
            pool,
            z,
            (n_samples, nc),
            &comp_dev,
            (nc, n_features),
            false,
            false,
            None,
        )?;
        comp_dev.release_into(pool);

        // X̂ = X̂_c + mean_ (broadcast the length-n_features mean over the rows).
        let mean64: Vec<f64> = s.mean_.clone();
        let recon_host = recon.to_host(pool);
        let mut out_host: Vec<F> = vec![F::from_int(0i64); n_samples * n_features];
        for r in 0..n_samples {
            for c in 0..n_features {
                let v = host_to_f64(recon_host[r * n_features + c]) + mean64[c];
                out_host[r * n_features + c] = f64_to_host::<F>(v);
            }
        }
        recon.release_into(pool);
        Ok(DeviceArray::from_host(pool, &out_host))
    }
}

impl<F> IncrementalPCA<F>
where
    F: Float + CubeElement + Pod,
{
    /// Per-component whitening scale for the TRANSFORM direction:
    /// `1/sqrt(explained_variance_[i])` when `whiten=True`, else `1.0`. A
    /// (near-)zero variance is floored so the scale stays finite (it never occurs
    /// on a fitted component).
    ///
    /// WR-03: the `ev <= WHITEN_VAR_FLOOR` branch below is DEFENSIVE-ONLY and is
    /// unreachable on fitted data — every RETAINED component carries non-trivial
    /// explained variance (a component with ~0 variance would not be selected by
    /// the truncated SVD), so no committed oracle fixture exercises it. It guards
    /// purely against a non-finite `1/sqrt(0)` should a degenerate retained
    /// component ever arise.
    fn whiten_scales(&self, s: &IncrementalSvdState) -> Vec<f64> {
        if !self.whiten {
            return vec![1.0; s.n_components];
        }
        s.explained_variance_
            .iter()
            .map(|&ev| {
                if ev > WHITEN_VAR_FLOOR {
                    1.0 / ev.sqrt()
                } else {
                    1.0
                }
            })
            .collect()
    }
}

/// Yield sklearn `gen_batches(n, batch_size, min_batch_size=min_batch)`-style
/// `[start, end)` row ranges over `n` samples — a verbatim port of
/// `sklearn.utils.gen_batches`. sklearn `IncrementalPCA.fit` loops over
/// `gen_batches(n_samples, batch_size_, min_batch_size=self.n_components or 0)`,
/// so a trailing batch that would leave fewer than `min_batch` rows is folded
/// into the final `start..n` slice rather than emitted short.
///
/// Algorithm (sklearn): iterate `n // batch_size` full batches; for each, if
/// `end + min_batch_size > n` SKIP it (without advancing `start`); then emit a
/// final `start..n` slice if any rows remain. This is the exact loop in
/// `sklearn/utils/__init__.py::gen_batches`.
fn gen_batches(n: usize, batch_size: usize, min_batch: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for _ in 0..(n / batch_size) {
        let end = start + batch_size;
        if end + min_batch > n {
            continue;
        }
        out.push((start, end));
        start = end;
    }
    if start < n {
        out.push((start, n));
    }
    out
}
