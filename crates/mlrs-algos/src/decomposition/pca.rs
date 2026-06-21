//! `PCA` (DECOMP-01) — principal component analysis by the SVD of the CENTERED
//! design matrix (D-01, NOT eig-of-covariance), matching
//! `sklearn.decomposition.PCA(svd_solver='full')`.
//!
//! ## Algorithm (sklearn `_fit_full`, RESEARCH-verified)
//! 1. `mean_` = column means of `X`; center `X_c = X − mean_`.
//! 2. Thin SVD of the CENTERED matrix: `X_c = U·diag(S)·Vᵀ` (`U` m×k, `S` k,
//!    `Vᵀ` k×n, `k = min(m, n)`) via the validated Phase-3 [`svd`] primitive —
//!    NO eig-of-covariance, NO bespoke matmul.
//! 3. `svd_flip(u_based_decision=False)` sign canonicalization, applied BY THE
//!    ESTIMATOR via [`align_rows`] on the `Vᵀ` rows (the primitive stays raw —
//!    D-01/D-03). `components_ = flipped Vᵀ[:n_components]`.
//! 4. `explained_variance_ = S²/(n_samples−1)` for ALL S; total variance is the
//!    sum over ALL `explained_variance_` BEFORE truncation;
//!    `explained_variance_ratio_ = explained_variance_ / total` (RESEARCH
//!    Pitfall 6 — the ratio denominator is the FULL spectrum, not the truncated
//!    one), then keep the top `n_components`.
//! 5. `singular_values_ = S[:n_components]`.
//!
//! ## Distinct from TruncatedSVD (D-01)
//! PCA centers `X` and uses `explained_variance_ = S²/(n−1)`; TruncatedSVD does
//! NOT center and uses `var(transform(X) columns)`. PCA also implements the
//! reconstruction `inverse_transform` (TruncatedSVD does not, D-01).
//!
//! ## Device residency (D-03)
//! Fitted `components_`/`explained_variance_`/`explained_variance_ratio_`/
//! `singular_values_`/`mean_` are stored as device-resident [`DeviceArray`]s;
//! `transform`/`inverse_transform` run the heavy `gemm` on-device and materialize
//! to the host only at a Rust accessor / oracle-comparison boundary.
//!
//! Tests live in `crates/mlrs-algos/tests/pca_test.rs` (AGENTS.md §2), never an
//! in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use mlrs_backend::prims::svd::svd;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::sign_flip::align_rows;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::AlgoError;
use crate::traits::{Fit, Transform};

/// Principal component analysis (DECOMP-01) fitted by the SVD of centered `X`.
///
/// Construct with [`Pca::new`] (`n_components`), then [`Fit::fit`] and
/// [`Transform::transform`] / [`Transform::inverse_transform`]. Fitted attributes
/// are device-resident (D-03); the host accessors materialize them on demand.
pub struct Pca<F> {
    /// Number of components to keep (`1 ..= min(n_samples, n_features)`).
    n_components: usize,
    /// `components_` (`n_components × n_features`), row-major, device-resident.
    components_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `explained_variance_` (length `n_components`), device-resident.
    explained_variance_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `explained_variance_ratio_` (length `n_components`), device-resident.
    explained_variance_ratio_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `singular_values_` (length `n_components`), device-resident.
    singular_values_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `mean_` (length `n_features`), device-resident.
    mean_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `n_features` seen at fit, for `transform`/`inverse_transform` geometry.
    n_features: usize,
}

impl<F> Pca<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `PCA` keeping `n_components` principal components
    /// (D-06 minimal surface — v1 takes an int `k ≤ min(m, n)`).
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components_: None,
            explained_variance_: None,
            explained_variance_ratio_: None,
            singular_values_: None,
            mean_: None,
            n_features: 0,
        }
    }

    /// Host copy of `components_` (`n_components × n_features`, row-major).
    pub fn components(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.attr(&self.components_, pool, "components_")
    }

    /// Host copy of `explained_variance_` (length `n_components`).
    pub fn explained_variance(
        &self,
        pool: &BufferPool<ActiveRuntime>,
    ) -> Result<Vec<F>, AlgoError> {
        self.attr(&self.explained_variance_, pool, "explained_variance_")
    }

    /// Host copy of `explained_variance_ratio_` (length `n_components`).
    pub fn explained_variance_ratio(
        &self,
        pool: &BufferPool<ActiveRuntime>,
    ) -> Result<Vec<F>, AlgoError> {
        self.attr(
            &self.explained_variance_ratio_,
            pool,
            "explained_variance_ratio_",
        )
    }

    /// Host copy of `singular_values_` (length `n_components`).
    pub fn singular_values(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.attr(&self.singular_values_, pool, "singular_values_")
    }

    /// Host copy of `mean_` (length `n_features`).
    pub fn mean(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.attr(&self.mean_, pool, "mean_")
    }

    fn attr(
        &self,
        slot: &Option<DeviceArray<ActiveRuntime, F>>,
        pool: &BufferPool<ActiveRuntime>,
        operation: &'static str,
    ) -> Result<Vec<F>, AlgoError> {
        slot.as_ref()
            .map(|a| a.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "pca",
                operation,
            })
    }
}

impl<F> Fit<F> for Pca<F>
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
        let k = n_samples.min(n_features);

        // --- T-04-04-01 / ASVS V5: reject an out-of-range n_components BEFORE any
        //     prim launch (untrusted hyperparameter → typed error). ---
        if self.n_components == 0 || self.n_components > k {
            return Err(AlgoError::InvalidNComponents {
                estimator: "pca",
                requested: self.n_components,
                max: k,
            });
        }
        // --- T-04-04-03: variance (S²/(n−1)) is undefined for n_samples ≤ 1. ---
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

        // --- 1. mean_ = column means via the Phase-2 column-mean reduction
        //        (key-link prim call, D-01). ---
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

        // --- 2. Center X on-host into a device buffer (the tiny per-column means
        //        and the descending S / ratio pass are already host work). ---
        let x_host = x.to_host(pool);
        let mut x_centered: Vec<F> = vec![F::from_int(0i64); n_samples * n_features];
        for r in 0..n_samples {
            for c in 0..n_features {
                let v = host_to_f64(x_host[r * n_features + c]) - mean64[c];
                x_centered[r * n_features + c] = f64_to_host::<F>(v);
            }
        }
        let x_c_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_centered);

        // --- 3. Thin SVD of the CENTERED design (D-01): X_c = U·diag(S)·Vᵀ. ---
        let (u, s, vt) = svd::<F>(pool, &x_c_dev, (n_samples, n_features))?;
        let s_host = s.to_host(pool);
        let s64: Vec<f64> = s_host.iter().map(|&v| host_to_f64(v)).collect();
        let vt_host = vt.to_host(pool);

        // --- 4. explained_variance_ = S²/(n−1) over ALL S; total = sum over ALL
        //        (BEFORE truncation, RESEARCH Pitfall 6); ratio = ev / total. ---
        let denom = (n_samples - 1) as f64;
        let ev_all: Vec<f64> = s64.iter().map(|&sigma| (sigma * sigma) / denom).collect();
        let total_var: f64 = ev_all.iter().sum();
        // Guard a degenerate zero total-variance denominator (T-04-04-03).
        let total_safe = if total_var.abs() > 0.0 {
            total_var
        } else {
            1.0
        };
        let ratio_all: Vec<f64> = ev_all.iter().map(|&ev| ev / total_safe).collect();

        // --- 5. svd_flip(u_based_decision=False) on the Vᵀ rows (estimator-side,
        //        primitive stays raw — D-01/D-03). align_rows == sklearn svd_flip.
        let vt_rows: Vec<Vec<f64>> = (0..k)
            .map(|j| {
                (0..n_features)
                    .map(|c| host_to_f64(vt_host[j * n_features + c]))
                    .collect()
            })
            .collect();
        let vt_flipped = align_rows(&vt_rows);

        // --- 6. Truncate to n_components and build device-resident fitted state.
        let nc = self.n_components;
        let mut components_host: Vec<F> = vec![F::from_int(0i64); nc * n_features];
        for j in 0..nc {
            for c in 0..n_features {
                components_host[j * n_features + c] = f64_to_host::<F>(vt_flipped[j][c]);
            }
        }
        let ev_trunc: Vec<F> = ev_all[..nc].iter().map(|&v| f64_to_host::<F>(v)).collect();
        let ratio_trunc: Vec<F> = ratio_all[..nc]
            .iter()
            .map(|&v| f64_to_host::<F>(v))
            .collect();
        let sv_trunc: Vec<F> = s64[..nc].iter().map(|&v| f64_to_host::<F>(v)).collect();

        let components_dev: DeviceArray<ActiveRuntime, F> =
            DeviceArray::from_host(pool, &components_host);
        let ev_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &ev_trunc);
        let ratio_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &ratio_trunc);
        let sv_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &sv_trunc);

        // --- 7. Release scratch; store device-resident fitted state (D-03). ---
        u.release_into(pool);
        s.release_into(pool);
        vt.release_into(pool);
        x_c_dev.release_into(pool);

        self.components_ = Some(components_dev);
        self.explained_variance_ = Some(ev_dev);
        self.explained_variance_ratio_ = Some(ratio_dev);
        self.singular_values_ = Some(sv_dev);
        self.mean_ = Some(mean_dev);
        self.n_features = n_features;
        Ok(self)
    }
}

impl<F> Transform<F> for Pca<F>
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
            estimator: "pca",
            operation: "transform",
        })?;
        let mean = self.mean_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "pca",
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

        // Center X on-host (mean_ is a tiny length-n_features vector).
        let mean_host = mean.to_host(pool);
        let mean64: Vec<f64> = mean_host.iter().map(|&v| host_to_f64(v)).collect();
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
        // row-major; transb reads it as componentsᵀ (n_features × nc) — no
        // transpose buffer (D-06).
        let z = gemm::<F>(
            pool,
            &x_c_dev,
            (n_samples, n_features),
            components,
            (n_features, nc),
            false,
            true, // components_ buffer is (nc × n_features); transb reads it as componentsᵀ.
            None,
        )?;
        x_c_dev.release_into(pool);
        Ok(z)
    }

    fn inverse_transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        z: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_samples, n_components) = shape;
        let components = self.components_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "pca",
            operation: "inverse_transform",
        })?;
        let mean = self.mean_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "pca",
            operation: "inverse_transform",
        })?;
        if n_components != self.n_components || z.len() != n_samples * n_components {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "z",
                rows: n_samples,
                cols: n_components,
                len: z.len(),
            }));
        }
        let n_features = self.n_features;

        // X̂_c = Z · components_  (m × n_features). components_ is
        // (nc × n_features) row-major; read as-is (no transpose).
        let recon = gemm::<F>(
            pool,
            z,
            (n_samples, n_components),
            components,
            (n_components, n_features),
            false,
            false,
            None,
        )?;

        // X̂ = X̂_c + mean_ (broadcast the length-n_features mean over the rows).
        let mean_host = mean.to_host(pool);
        let mean64: Vec<f64> = mean_host.iter().map(|&v| host_to_f64(v)).collect();
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
