//! `TruncatedSVD` (DECOMP-02) — truncated singular value decomposition of the
//! UNCENTERED design matrix, matching
//! `sklearn.decomposition.TruncatedSVD(algorithm='arpack')`.
//!
//! ## Same skeleton as PCA, THREE documented differences (RESEARCH Pitfall 2)
//! TruncatedSVD reuses the SAME validated Phase-3 thin [`svd`] + estimator-side
//! [`align_rows`] (`svd_flip(u_based_decision=False)`, D-03) skeleton as
//! [`Pca`](super::pca::Pca), differing only in:
//!
//! 1. **NO centering** — the thin SVD runs on the UNCENTERED `X` (there is no
//!    `mean_`). `components_ = flipped Vᵀ[:n_components]`.
//! 2. **`explained_variance_ = var(transform(X) columns)`** — the EMPIRICAL
//!    (population, ddof=0) variance of each transformed column, NOT PCA's
//!    `S²/(n−1)` (copying PCA's formula here is the Pitfall-2 anti-pattern). The
//!    transform is `transform(X) = X·components_ᵀ = U·S`.
//! 3. **`explained_variance_ratio_` denominator = total per-feature variance of
//!    the ORIGINAL X** (`Σ var(X[:, c], ddof=0)`), not the sum of the kept
//!    component variances.
//!
//! `singular_values_ = S[:n_components]`. `inverse_transform` is NOT implemented
//! (only PCA reconstructs in v1, D-01); the default `Transform::inverse_transform`
//! returns [`AlgoError::Unsupported`].
//!
//! ## Device residency (D-03)
//! Fitted `components_`/`explained_variance_`/`explained_variance_ratio_`/
//! `singular_values_` are device-resident [`DeviceArray`]s; `transform` runs the
//! heavy `gemm` on-device and materializes to the host only at a Rust accessor /
//! oracle-comparison boundary.
//!
//! Tests live in `crates/mlrs-algos/tests/truncated_svd_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::svd::svd;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::sign_flip::align_rows;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{Fit, Fitted, Transform, Unfit};

/// Truncated SVD (DECOMP-02) fitted by the thin SVD of the UNCENTERED `X`.
///
/// Construct with the zero-arg [`TruncatedSvd::new`] (sklearn-style default
/// `n_components = 2`) or [`TruncatedSvd::builder`] (`.n_components(usize)`), then
/// the consuming [`Fit::fit`] (returns the `Fitted`-tagged sibling) and
/// [`Transform::transform`]. Fitted attributes are device-resident (D-03); the
/// host accessors materialize them on demand and exist ONLY on
/// `TruncatedSvd<F, Fitted>` (the compile-time typestate replaces the old runtime
/// `NotFitted` guard, D-03). Unlike [`Pca`](super::pca::Pca), there is no `mean_`
/// (no centering) and `inverse_transform` is unsupported (the typestate
/// `Transform::inverse_transform` default → `AlgoError::Unsupported`).
pub struct TruncatedSvd<F, S = Unfit> {
    /// Number of components to keep (`1 ..= min(n_samples, n_features)`).
    n_components: usize,
    /// `components_` (`n_components × n_features`), row-major, device-resident.
    components_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `explained_variance_` (length `n_components`) = var(transform cols),
    /// device-resident.
    explained_variance_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `explained_variance_ratio_` (length `n_components`), device-resident.
    explained_variance_ratio_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `singular_values_` (length `n_components`), device-resident.
    singular_values_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `n_features` seen at fit, for `transform` geometry.
    n_features: usize,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> TruncatedSvd<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `TruncatedSVD` with the sklearn-style default
    /// `n_components = 2` directly in the `Unfit` state. This is the SINGLE source
    /// of truth for the default hyperparameter (D-08): the builder `Default`
    /// re-derives from here via [`TruncatedSvd::into_builder`]. The actual
    /// `n_components` is validated against `min(n_samples, n_features)` at `fit`
    /// (a data-DEPENDENT bound), so [`TruncatedSvdBuilder::build`] is infallible.
    pub fn new() -> Self {
        Self {
            n_components: 2,
            components_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            singular_values_: None,
            n_features: 0,
            _state: PhantomData,
        }
    }

    /// Start building a `TruncatedSVD` from the default `n_components` (D-08).
    pub fn builder() -> TruncatedSvdBuilder {
        TruncatedSvdBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying the
    /// hyperparameter. Used by [`TruncatedSvdBuilder::default`] to re-derive the
    /// default from [`TruncatedSvd::new`] (D-08).
    pub fn into_builder(self) -> TruncatedSvdBuilder {
        TruncatedSvdBuilder {
            n_components: self.n_components,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// device fields are excluded — all are `None` in any `Unfit` value). Used by
    /// the defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_components == other.n_components
    }
}

impl<F> Default for TruncatedSvd<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`TruncatedSvd`] (D-01). `Default` re-derives the default
/// `n_components` from [`TruncatedSvd::new`] (D-08 single source) rather than
/// holding a literal (Pitfall 1).
#[derive(Debug, Clone, Copy)]
pub struct TruncatedSvdBuilder {
    n_components: usize,
}

impl Default for TruncatedSvdBuilder {
    /// Re-derive the default `n_components` from [`TruncatedSvd::new`] (D-08).
    fn default() -> Self {
        TruncatedSvd::<f64, Unfit>::new().into_builder()
    }
}

impl TruncatedSvdBuilder {
    /// Set the number of components to keep (`1 ..= min(n_samples, n_features)`;
    /// validated against the data shape at `fit`).
    pub fn n_components(mut self, v: usize) -> Self {
        self.n_components = v;
        self
    }

    /// Build the (unfit) estimator. TruncatedSVD's `n_components` bound is
    /// `1 ..= min(n_samples, n_features)` — a data-DEPENDENT bound validated in
    /// [`Fit::fit`] (`AlgoError::InvalidNComponents`). This `build()` is therefore
    /// infallible-but-typed (`Result<_, BuildError>` that never errs), kept for
    /// family uniformity so the PyO3 `build_err_to_py` mapper stays shape-identical.
    pub fn build<F>(self) -> Result<TruncatedSvd<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        Ok(TruncatedSvd {
            n_components: self.n_components,
            components_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            singular_values_: None,
            n_features: 0,
            _state: PhantomData,
        })
    }
}

impl<F> TruncatedSvd<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of `components_` (`n_components × n_features`, row-major). `Some`
    /// by construction on the `Fitted` state (D-03).
    pub fn components(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.components_, pool)
    }

    /// Host copy of `explained_variance_` (length `n_components`).
    pub fn explained_variance(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.explained_variance_, pool)
    }

    /// Host copy of `explained_variance_ratio_` (length `n_components`).
    pub fn explained_variance_ratio(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.explained_variance_ratio_, pool)
    }

    /// Host copy of `singular_values_` (length `n_components`).
    pub fn singular_values(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.attr(&self.singular_values_, pool)
    }

    fn attr(
        &self,
        slot: &Option<DeviceArray<ActiveRuntime, F>>,
        pool: &BufferPool<ActiveRuntime>,
    ) -> Vec<F> {
        slot.as_ref()
            .expect("fitted attribute is Some by construction on TruncatedSvd<F, Fitted>")
            .to_host(pool)
    }
}

impl<F> Fit<F> for TruncatedSvd<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = TruncatedSvd<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<TruncatedSvd<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        let k = n_samples.min(n_features);

        // --- T-04-04-01 / ASVS V5: reject an out-of-range n_components BEFORE any
        //     prim launch (untrusted hyperparameter → typed error). ---
        if self.n_components == 0 || self.n_components > k {
            return Err(AlgoError::InvalidNComponents {
                estimator: "truncated_svd",
                requested: self.n_components,
                max: k,
            });
        }
        // --- T-04-04-03: variance is undefined for n_samples ≤ 1. ---
        if n_samples <= 1 {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        // --- T-04-04-02: geometry consistency before launch. ---
        if n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }

        // --- DIFFERENCE 1 (Pitfall 2): NO centering — thin SVD of the UNCENTERED
        //     X. X = U·diag(S)·Vᵀ via the validated Phase-3 svd primitive. ---
        let (u, s, vt) = svd::<F>(pool, x, (n_samples, n_features))?;
        let s_host = s.to_host(pool);
        let s64: Vec<f64> = s_host.iter().map(|&v| host_to_f64(v)).collect();
        let vt_host = vt.to_host(pool);

        // --- svd_flip(u_based_decision=False) on the Vᵀ rows (estimator-side;
        //     primitive stays raw — D-01/D-03). ---
        let vt_rows: Vec<Vec<f64>> = (0..k)
            .map(|j| {
                (0..n_features)
                    .map(|c| host_to_f64(vt_host[j * n_features + c]))
                    .collect()
            })
            .collect();
        let vt_flipped = align_rows(&vt_rows);

        let nc = self.n_components;

        // --- DIFFERENCE 2 (Pitfall 2): explained_variance_ = var(transform cols),
        //     NOT S²/(n−1). transform(X) = X·components_ᵀ = U·S. We form U·S for
        //     the kept components and take each column's POPULATION (ddof=0)
        //     variance. ---
        // transform column j (kept) = U[:, j] · S[j]. U is (m×k) row-major.
        let u_host = u.to_host(pool);
        let mut ev_kept: Vec<f64> = vec![0.0; nc];
        for j in 0..nc {
            let sj = s64[j];
            // z_rj = U[r, j] * S[j] but the flipped component carries the sign;
            // variance is sign-invariant, so the raw U·S column variance is the
            // svd_flip-invariant explained_variance_ (matches sklearn).
            let mut col: Vec<f64> = Vec::with_capacity(n_samples);
            for r in 0..n_samples {
                col.push(host_to_f64(u_host[r * k + j]) * sj);
            }
            let mean = col.iter().sum::<f64>() / n_samples as f64;
            let var = col.iter().map(|&z| (z - mean) * (z - mean)).sum::<f64>() / n_samples as f64; // ddof=0 (population), sklearn convention.
            ev_kept[j] = var;
        }

        // --- DIFFERENCE 3 (Pitfall 2): explained_variance_ratio_ denominator =
        //     total per-feature variance of the ORIGINAL X (Σ var(X[:, c], ddof=0)).
        let x_host = x.to_host(pool);
        let mut total_var = 0.0f64;
        for c in 0..n_features {
            let mut mean = 0.0f64;
            for r in 0..n_samples {
                mean += host_to_f64(x_host[r * n_features + c]);
            }
            mean /= n_samples as f64;
            let mut var = 0.0f64;
            for r in 0..n_samples {
                let d = host_to_f64(x_host[r * n_features + c]) - mean;
                var += d * d;
            }
            total_var += var / n_samples as f64;
        }
        let total_safe = if total_var.abs() > 0.0 {
            total_var
        } else {
            1.0
        };
        let ratio_kept: Vec<f64> = ev_kept.iter().map(|&ev| ev / total_safe).collect();

        // --- Truncate to n_components; build device-resident fitted state. ---
        let mut components_host: Vec<F> = vec![F::from_int(0i64); nc * n_features];
        for j in 0..nc {
            for c in 0..n_features {
                components_host[j * n_features + c] = f64_to_host::<F>(vt_flipped[j][c]);
            }
        }
        let ev_trunc: Vec<F> = ev_kept.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let ratio_trunc: Vec<F> = ratio_kept.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let sv_trunc: Vec<F> = s64[..nc].iter().map(|&v| f64_to_host::<F>(v)).collect();

        let components_dev: DeviceArray<ActiveRuntime, F> =
            DeviceArray::from_host(pool, &components_host);
        let ev_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &ev_trunc);
        let ratio_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &ratio_trunc);
        let sv_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &sv_trunc);

        // --- Release scratch; store device-resident fitted state (D-03). ---
        u.release_into(pool);
        s.release_into(pool);
        vt.release_into(pool);

        Ok(TruncatedSvd {
            n_components: self.n_components,
            components_: Some(components_dev),
            explained_variance_: Some(ev_dev),
            explained_variance_ratio_: Some(ratio_dev),
            singular_values_: Some(sv_dev),
            n_features,
            _state: PhantomData,
        })
    }
}

impl<F> Transform<F> for TruncatedSvd<F, Fitted>
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
        let components = self.components_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "truncated_svd",
            operation: "transform",
        })?;
        if n_features != self.n_features || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        let nc = self.n_components;

        // DIFFERENCE 1: NO centering. Z = X · components_ᵀ  (m × nc). components_
        // is (nc × n_features) row-major; transb reads it as componentsᵀ
        // (n_features × nc) — no transpose buffer (D-06).
        let z = gemm::<F>(
            pool,
            x,
            (n_samples, n_features),
            components,
            (n_features, nc),
            false,
            true, // components_ buffer is (nc × n_features); transb reads it as componentsᵀ.
            None,
        )?;
        Ok(z)
    }

    // inverse_transform: TruncatedSVD keeps the default Transform impl, which
    // returns AlgoError::Unsupported (only PCA reconstructs in v1, D-01).
}
