//! `GaussianRandomProjection` (PROJ-01) + [`johnson_lindenstrauss_min_dim`].
//!
//! A dense random-projection transformer whose `components_` is an
//! `n_components × n_features` matrix drawn `N(0, 1/n_components)` via the
//! Phase-7 [`mlrs_backend::prims::rng::gaussian_matrix`] generator (PRIM-06,
//! host SplitMix64 → ONE device upload, seed-reproducible across cpu/rocm,
//! never `OsRng` — T-07-02). `transform == X · components_ᵀ` is the SAME single
//! GEMM as `Pca::transform` (D-12) — RandomProjection does **NOT** center
//! (no `mean_`, no centering pass).
//!
//! ## `n_components='auto'` (PROJ-01)
//! The [`NComponents::Auto`] path resolves the embedding dimension via
//! [`johnson_lindenstrauss_min_dim`] — the ONE value-matched quantity in the
//! whole random-projection family (D-12). `johnson_lindenstrauss_min_dim` is
//! value-matched to `sklearn.random_projection.johnson_lindenstrauss_min_dim`
//! at 1e-5; the projection MATRIX / transform are NOT (the SplitMix64 stream is
//! not numpy's MT19937), so their correctness is the structural PROPERTY gate
//! (JL distortion bound, matrix moments, seed-reproducibility,
//! `transform == X·componentsᵀ` self-consistency — D-12).
//!
//! ## Hyperparameter guards (ASVS V5 / T-07-10)
//! `eps ∉ (0, 1)` is rejected as [`AlgoError::InvalidEpsDistortion`] and
//! `n_components < 1` (the resolved or fixed value) as
//! [`AlgoError::InvalidNComponents`] — BEFORE any RNG matrix allocation.
//!
//! ## Device residency (D-03)
//! `components_` is a device-resident [`DeviceArray`]; the host accessor
//! materializes it on demand at a Rust / oracle boundary only.
//!
//! Tests live in `crates/mlrs-algos/tests/random_projection_test.rs`
//! (AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::rng;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::traits::{Fit, Transform};

/// `n_components` selector for the random-projection transformers (D-06 minimal
/// surface). [`NComponents::Auto`] sizes the embedding via
/// [`johnson_lindenstrauss_min_dim`] from the fit `n_samples` + the distortion
/// `eps`; [`NComponents::Fixed`] takes the caller's explicit integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NComponents {
    /// Size the embedding via the Johnson–Lindenstrauss lemma at fit time
    /// (`johnson_lindenstrauss_min_dim(n_samples, eps)`), exactly like
    /// sklearn's `n_components='auto'`.
    Auto,
    /// Use this fixed embedding dimension (must be `>= 1`).
    Fixed(usize),
}

/// The minimum embedding dimension that preserves pairwise distances within
/// `eps` for `n_samples` points, per the Johnson–Lindenstrauss lemma — the ONE
/// value-matched random-projection quantity (D-12). Matches
/// `sklearn.random_projection.johnson_lindenstrauss_min_dim(n_samples, eps)` at
/// 1e-5 (RESEARCH Pattern 5 / Code Example).
///
/// ```text
/// denom = eps²/2 − eps³/3
/// min_dim = floor(4 · ln(n_samples) / denom)
/// ```
///
/// `eps` MUST lie in the open interval `(0, 1)` (sklearn's contract); an
/// out-of-range `eps` is rejected as [`AlgoError::InvalidEpsDistortion`] BEFORE
/// the computation (ASVS V5 / T-07-10). `n_samples` is an `f64` so the `ln`
/// is taken directly on the sample count.
pub fn johnson_lindenstrauss_min_dim(n_samples: f64, eps: f64) -> Result<usize, AlgoError> {
    // ASVS V5 / T-07-10: reject eps ∉ (0, 1) BEFORE computing (a non-finite,
    // ≤ 0, or ≥ 1 eps makes the JL bound undefined / non-positive denom).
    if !(eps.is_finite() && eps > 0.0 && eps < 1.0) {
        return Err(AlgoError::InvalidEpsDistortion {
            estimator: "random_projection",
            eps,
        });
    }
    let denom = eps * eps / 2.0 - eps * eps * eps / 3.0;
    Ok((4.0 * n_samples.ln() / denom).floor() as usize)
}

/// Gaussian random-projection transformer (PROJ-01).
///
/// Construct with [`GaussianRandomProjection::new`] (`n_components`, `seed`,
/// `eps`), then [`Fit::fit`] to draw the `N(0, 1/n_components)` `components_`
/// and [`Transform::transform`] to project `X` (`X · components_ᵀ`, one GEMM,
/// NO centering). The fitted `components_` is device-resident (D-03).
pub struct GaussianRandomProjection<F> {
    /// Requested embedding dimension (`Auto` → resolved at fit via JL).
    n_components: NComponents,
    /// Documented `u64` seed driving the host SplitMix64 stream (T-07-02 —
    /// never `OsRng`); the seed-reproducibility source.
    seed: u64,
    /// JL distortion bound used by the `Auto` path (`eps ∈ (0, 1)`).
    eps: f64,
    /// `components_` (`n_components × n_features`), row-major, device-resident.
    components_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Resolved embedding dimension after fit (`Auto` → JL value).
    n_components_: usize,
    /// `n_features` seen at fit, for the `transform` geometry check.
    n_features: usize,
}

impl<F> GaussianRandomProjection<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `GaussianRandomProjection`.
    ///
    /// - `n_components`: [`NComponents::Auto`] (JL-sized) or
    ///   [`NComponents::Fixed`].
    /// - `seed`: the documented `u64` driving SplitMix64 (reproducibility
    ///   source — never `OsRng`, T-07-02).
    /// - `eps`: the JL distortion bound for the `Auto` path (`eps ∈ (0, 1)`);
    ///   ignored when `n_components` is `Fixed`, but still validated at fit so a
    ///   bad value never silently passes.
    pub fn new(n_components: NComponents, seed: u64, eps: f64) -> Self {
        Self {
            n_components,
            seed,
            eps,
            components_: None,
            n_components_: 0,
            n_features: 0,
        }
    }

    /// Host copy of `components_` (`n_components_ × n_features`, row-major).
    pub fn components(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.components_
            .as_ref()
            .map(|a| a.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "gaussian_random_projection",
                operation: "components_",
            })
    }

    /// The resolved embedding dimension (`Auto` → JL value) after fit.
    pub fn n_components_(&self) -> usize {
        self.n_components_
    }
}

impl<F> Fit<F> for GaussianRandomProjection<F>
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

        // --- T-07-10 / ASVS V5: geometry consistency BEFORE any RNG launch. ---
        if n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }

        // --- Resolve n_components: Auto → johnson_lindenstrauss_min_dim
        //     (which itself validates eps ∈ (0,1)); Fixed → the caller's int. ---
        let nc = match self.n_components {
            NComponents::Auto => johnson_lindenstrauss_min_dim(n_samples as f64, self.eps)?,
            NComponents::Fixed(k) => {
                // Validate eps even on the Fixed path so a bad eps never passes
                // silently (ASVS V5) — surfaces the same typed error.
                if !(self.eps.is_finite() && self.eps > 0.0 && self.eps < 1.0) {
                    return Err(AlgoError::InvalidEpsDistortion {
                        estimator: "random_projection",
                        eps: self.eps,
                    });
                }
                k
            }
        };

        // --- n_components >= 1 BEFORE generation (ASVS V5 / T-07-10). ---
        if nc < 1 {
            return Err(AlgoError::InvalidNComponents {
                estimator: "gaussian_random_projection",
                requested: nc,
                max: n_features,
            });
        }

        // --- components_ = N(0, 1/n_components) via PRIM-06 (host SplitMix64 →
        //     ONE upload, seed-reproducible — D-12 / T-07-02). ---
        let components_dev =
            rng::gaussian_matrix::<F>(pool, self.seed, nc, n_features)?;

        self.components_ = Some(components_dev);
        self.n_components_ = nc;
        self.n_features = n_features;
        Ok(self)
    }
}

impl<F> Transform<F> for GaussianRandomProjection<F>
where
    F: Float + CubeElement + Pod,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        project(
            "gaussian_random_projection",
            self.components_.as_ref(),
            self.n_components_,
            self.n_features,
            pool,
            x,
            shape,
        )
    }
}

/// Shared `transform == X · components_ᵀ` for the random-projection family
/// (Gaussian + Sparse share the SAME single GEMM — D-12; RandomProjection does
/// NOT center). `components_` is `(n_components × n_features)` row-major; `transb`
/// reads it as `componentsᵀ` (`n_features × n_components`) with no transpose
/// buffer (the exact pattern as `Pca::transform`, minus the centering pass).
///
/// `pub(crate)` so [`crate::projection::sparse::SparseRandomProjection`] reuses
/// the identical implementation.
pub(crate) fn project<F>(
    estimator: &'static str,
    components_: Option<&DeviceArray<ActiveRuntime, F>>,
    n_components_: usize,
    fitted_n_features: usize,
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let (n_samples, n_features) = shape;
    let components = components_.ok_or(AlgoError::NotFitted {
        estimator,
        operation: "transform",
    })?;
    if n_features != fitted_n_features || x.len() != n_samples * n_features {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "x",
            rows: n_samples,
            cols: n_features,
            len: x.len(),
        }));
    }
    let nc = n_components_;

    // Z = X · components_ᵀ  (m × nc). components_ is (nc × n_features) row-major;
    // transb reads it as componentsᵀ (n_features × nc) — no transpose buffer
    // (D-06), NO centering (RandomProjection does not center — D-12).
    let z = gemm::<F>(
        pool,
        x,
        (n_samples, n_features),
        components,
        (n_features, nc),
        false,
        true, // components_ buffer is (nc × n_features); transb reads componentsᵀ.
        None,
    )?;
    Ok(z)
}
