//! `SparseRandomProjection` (PROJ-02) — Achlioptas sparse random projection,
//! stored DENSE (D-12).
//!
//! An `n_components × n_features` projection matrix whose entries are `0` with
//! probability `1 − density`, `+v` / `−v` each with probability `density/2`,
//! where `v = sqrt((1/density)/n_components)` (RESEARCH Pattern 4). Generated via
//! the Phase-7 [`mlrs_backend::prims::rng::sparse_achlioptas_matrix`] (PRIM-06,
//! host SplitMix64 → ONE upload, seed-reproducible — T-07-02). The matrix is
//! stored DENSE even though it is structurally sparse (D-12 — there is no sparse
//! device kernel in v2; at v2 matrix sizes a dense Achlioptas is acceptable), so
//! `transform == X · components_ᵀ` is the IDENTICAL single GEMM as the Gaussian
//! transformer ([`crate::projection::gaussian::project`]) — RandomProjection does
//! NOT center. Sparse INPUT densification happens at the Python ingress (Plan
//! 07), not here.
//!
//! ## `n_components='auto'` (PROJ-02)
//! [`NComponents::Auto`] resolves via [`johnson_lindenstrauss_min_dim`] exactly
//! like the Gaussian transformer (the ONE value-matched quantity — D-12); the
//! matrix/transform are property-gated, not 1e-5-valued.
//!
//! ## Hyperparameter guards (ASVS V5 / T-07-10)
//! `eps ∉ (0, 1)` → [`AlgoError::InvalidEpsDistortion`]; `density ∉ (0, 1]` →
//! [`AlgoError::InvalidDensity`]; `n_components < 1` →
//! [`AlgoError::InvalidNComponents`] — all rejected BEFORE any RNG matrix
//! allocation. `density = None` resolves to the sklearn default
//! `1/sqrt(n_features)`.
//!
//! Tests live in `crates/mlrs-algos/tests/random_projection_test.rs`
//! (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::rng;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::projection::gaussian::{johnson_lindenstrauss_min_dim, project, NComponents};
use crate::traits::{Fit, Transform};

/// Sparse (Achlioptas) random-projection transformer (PROJ-02), `components_`
/// stored DENSE (D-12).
///
/// Construct with [`SparseRandomProjection::new`] (`n_components`, `seed`, `eps`,
/// `density`), then [`Fit::fit`] to draw the Achlioptas `components_` and
/// [`Transform::transform`] (`X · components_ᵀ`, one GEMM, NO centering). Fitted
/// state is device-resident (D-03).
pub struct SparseRandomProjection<F> {
    /// Requested embedding dimension (`Auto` → resolved at fit via JL).
    n_components: NComponents,
    /// Documented `u64` seed driving SplitMix64 (T-07-02 — never `OsRng`).
    seed: u64,
    /// JL distortion bound for the `Auto` path (`eps ∈ (0, 1)`).
    eps: f64,
    /// Requested sparsity density (`None` → `1/sqrt(n_features)` at fit;
    /// otherwise must be `∈ (0, 1]`).
    density: Option<f64>,
    /// `components_` (`n_components × n_features`), row-major, device-resident.
    components_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Resolved embedding dimension after fit.
    n_components_: usize,
    /// Resolved density after fit (`None` → `1/sqrt(n_features)`).
    density_: f64,
    /// `n_features` seen at fit, for the `transform` geometry check.
    n_features: usize,
}

impl<F> SparseRandomProjection<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `SparseRandomProjection`.
    ///
    /// - `n_components`: [`NComponents::Auto`] (JL-sized) or
    ///   [`NComponents::Fixed`].
    /// - `seed`: documented `u64` driving SplitMix64 (never `OsRng`, T-07-02).
    /// - `eps`: JL distortion bound for the `Auto` path (`eps ∈ (0, 1)`).
    /// - `density`: `None` → sklearn default `1/sqrt(n_features)`; otherwise the
    ///   explicit density `∈ (0, 1]`.
    pub fn new(
        n_components: NComponents,
        seed: u64,
        eps: f64,
        density: Option<f64>,
    ) -> Self {
        Self {
            n_components,
            seed,
            eps,
            density,
            components_: None,
            n_components_: 0,
            density_: 0.0,
            n_features: 0,
        }
    }

    /// Host copy of `components_` (`n_components_ × n_features`, row-major, dense).
    pub fn components(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.components_
            .as_ref()
            .map(|a| a.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "sparse_random_projection",
                operation: "components_",
            })
    }

    /// The resolved embedding dimension (`Auto` → JL value) after fit.
    pub fn n_components_(&self) -> usize {
        self.n_components_
    }

    /// The resolved density (`None` → `1/sqrt(n_features)`) after fit.
    pub fn density_(&self) -> f64 {
        self.density_
    }
}

impl<F> Fit<F> for SparseRandomProjection<F>
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

        // --- T-07-10 / ASVS V5: geometry consistency BEFORE any RNG launch. ---
        if n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }

        // --- Resolve density: None → 1/sqrt(n_features) (sklearn default);
        //     validate density ∈ (0, 1] (ASVS V5 / T-07-10) BEFORE generation. ---
        let density = self
            .density
            .unwrap_or_else(|| 1.0 / (n_features as f64).sqrt());
        if !(density.is_finite() && density > 0.0 && density <= 1.0) {
            return Err(AlgoError::InvalidDensity {
                estimator: "sparse_random_projection",
                density,
            });
        }

        // --- Resolve n_components (Auto → JL; Fixed → caller's int). The Auto
        //     path validates eps ∈ (0,1); validate eps on the Fixed path too. ---
        let nc = match self.n_components {
            NComponents::Auto => johnson_lindenstrauss_min_dim(n_samples as f64, self.eps)?,
            NComponents::Fixed(k) => {
                if !(self.eps.is_finite() && self.eps > 0.0 && self.eps < 1.0) {
                    return Err(AlgoError::InvalidEpsDistortion {
                        estimator: "random_projection",
                        eps: self.eps,
                    });
                }
                k
            }
        };
        if nc < 1 {
            return Err(AlgoError::InvalidNComponents {
                estimator: "sparse_random_projection",
                requested: nc,
                max: n_features,
            });
        }

        // --- components_ = Achlioptas v=sqrt((1/density)/n_components), stored
        //     DENSE via PRIM-06 (host SplitMix64 → ONE upload — D-12 / T-07-02). ---
        let components_dev =
            rng::sparse_achlioptas_matrix::<F>(pool, self.seed, nc, n_features, density)?;

        self.components_ = Some(components_dev);
        self.n_components_ = nc;
        self.density_ = density;
        self.n_features = n_features;
        Ok(self)
    }
}

impl<F> Transform<F> for SparseRandomProjection<F>
where
    F: Float + CubeElement + Pod,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        // SAME single GEMM as Gaussian (D-12 — dense Achlioptas, no centering).
        project(
            "sparse_random_projection",
            self.components_.as_ref(),
            self.n_components_,
            self.n_features,
            pool,
            x,
            shape,
        )
    }
}
