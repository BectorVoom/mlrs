//! `SpectralEmbedding` (SPECTRAL-01) — graph-Laplacian spectral embedding,
//! matching `sklearn.manifold.SpectralEmbedding`.
//!
//! ## Pipeline (RESEARCH System Diagram + Pattern 2)
//! `affinity A` → normalized Laplacian `(L, dd) = laplacian(A, n)` → the smallest
//! `n_components + 1` eigenvectors via v1 `eig` (full spectrum, reversed to
//! ascending) → `D^-1/2` recovery (`/ dd`, D-07) → deterministic sign flip → drop
//! the trivial ≈0 eigenvector (`drop_first = true`, D-08) → `embedding_`
//! (`n × n_components`).
//!
//! ## Affinity defaults & gamma (D-01 / D-04)
//! Default affinity is `"nearest_neighbors"` (sklearn `SpectralEmbedding`'s own
//! default — the kNN-connectivity graph, D-03), NOT `"rbf"`. `gamma = None`
//! resolves to `1/n_features` at `fit` (the [`KernelRidge`](crate::kernel_ridge)
//! at-fit gamma resolution precedent, D-04); the rbf affinity path uses
//! `kernel_matrix(Rbf { gamma })` (D-02).
//!
//! ## n_samples ≤ 64 (D-05 / D-06)
//! The Laplacian is `n_samples × n_samples` and v1 `eig` caps `n ≤ MAX_DIM = 64`.
//! `fit` rejects `n_samples > 64` with [`AlgoError::NSamplesExceedsMaxDim`] BEFORE
//! any affinity / Laplacian / eig launch (ASVS V5).
//!
//! ## Wave-0 scaffold status
//! This is the 09-01 Wave-0 COMPILING STUB: the struct + [`SpectralEmbedding::new`]
//! constructor are real (so the PyO3 wrapper + the test scaffolds compile today),
//! but the `fit` / `embedding_` bodies are `todo!()` pending the Wave-2 plan
//! (09-03). Do NOT write the eig recovery here — it is Wave 2.
//!
//! Tests live in `crates/mlrs-algos/tests/spectral_embedding_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;

use crate::error::AlgoError;

/// Spectral embedding (SPECTRAL-01) of an affinity graph onto the smallest
/// non-trivial eigenvectors of the normalized Laplacian.
///
/// Construct with [`SpectralEmbedding::new`] (`n_components`, `affinity`,
/// `gamma`, `n_neighbors`), then `fit` and read `embedding_` (Wave-2). Fitted
/// `embedding_` (`n × n_components`) is device-resident (D-03); the host accessor
/// materializes it on demand.
pub struct SpectralEmbedding<F>
where
    F: Float + CubeElement + Pod,
{
    /// Embedding dimensionality (sklearn default `2`, D-08). The smallest
    /// `n_components + 1` eigenvectors are computed and the trivial ≈0 one dropped.
    n_components: usize,
    /// Affinity construction (`"nearest_neighbors"` default, D-01 — the
    /// kNN-connectivity graph, D-03; `"rbf"` uses `kernel_matrix(Rbf)`, D-02).
    affinity: String,
    /// Kernel coefficient `γ` for the rbf affinity; `None` resolves to
    /// `1/n_features` at `fit` (D-04). Ignored for `"nearest_neighbors"`.
    gamma: Option<F>,
    /// Number of neighbors for the `"nearest_neighbors"` affinity (sklearn
    /// default `10`, D-03).
    n_neighbors: usize,
    /// Fitted `n × n_components` embedding, device-resident, `None` until `fit`.
    embedding_: Option<DeviceArray<ActiveRuntime, F>>,
}

impl<F> SpectralEmbedding<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `SpectralEmbedding`. `n_components` is the embedding
    /// dimensionality (sklearn default `2`, D-08), `affinity` selects the graph
    /// (`"nearest_neighbors"` default / `"rbf"`, D-01), `gamma` is the rbf kernel
    /// coefficient (`None` → `1/n_features` at `fit`, D-04), and `n_neighbors` is
    /// the kNN-connectivity neighbor count (sklearn default `10`, D-03). Invalid
    /// hyperparameters are rejected at `fit`, not construction.
    pub fn new(
        n_components: usize,
        affinity: String,
        gamma: Option<F>,
        n_neighbors: usize,
    ) -> Self {
        Self {
            n_components,
            affinity,
            gamma,
            n_neighbors,
            embedding_: None,
        }
    }

    /// Host copy of the fitted `embedding_` (`n × n_components` row-major). Errors
    /// with [`AlgoError::NotFitted`] before `fit`.
    ///
    /// **Wave-0 stub:** the `embedding_` accessor body is `todo!()` pending the
    /// Wave-2 plan (09-03). The field + signature are real (the PyO3 wrapper +
    /// test scaffolds compile against this surface today).
    pub fn embedding(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        // Reference the field so the Wave-0 stub keeps the fitted-state seam.
        let _ = (&self.embedding_, pool);
        todo!("SpectralEmbedding::embedding is filled by the Wave-2 plan 09-03 (SPECTRAL-01)")
    }

    /// Fit the spectral embedding to the affinity graph of `x`
    /// (`shape = (n_samples, n_features)`, row-major). Rejects `n_samples > 64`
    /// with [`AlgoError::NSamplesExceedsMaxDim`] BEFORE any launch (D-06).
    ///
    /// **Wave-0 stub:** the `fit` body (affinity → Laplacian → eig recovery → sign
    /// flip → drop-first) is `todo!()` pending the Wave-2 plan (09-03).
    pub fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        // Reference every field + arg so the Wave-0 stub fixes the full fit seam
        // (affinity/gamma/n_neighbors/n_components are consumed by the Wave-2 body).
        let _ = (
            &self.affinity,
            &self.gamma,
            self.n_neighbors,
            self.n_components,
            &mut self.embedding_,
            pool,
            x,
            shape,
        );
        todo!("SpectralEmbedding::fit is filled by the Wave-2 plan 09-03 (SPECTRAL-01)")
    }
}
