//! `SpectralClustering` (SPECTRAL-02) — spectral embedding → v1 KMeans, matching
//! `sklearn.cluster.SpectralClustering`.
//!
//! ## Pipeline (RESEARCH System Diagram + Pattern 2)
//! `affinity A` (rbf via `kernel_matrix(Rbf)` D-02, or kNN-connectivity D-03) →
//! normalized Laplacian `(L, dd) = laplacian(A, n)` → the smallest `n_components`
//! eigenvectors via v1 `eig` (reversed to ascending; `drop_first = false` for SC,
//! D-11) → `D^-1/2` recovery (`/ dd`, D-07) → [`KMeans::new`](crate::cluster::KMeans)
//! (default kmeans++, `n_init = 1`, D-10) → `labels_`. `labels_` matches sklearn
//! up to a label permutation (the exact-labels gate via a well-separated fixture,
//! D-10).
//!
//! ## Affinity defaults & gamma (D-01 / D-04)
//! Default affinity is `"rbf"` (sklearn `SpectralClustering`'s own default, D-01),
//! with `gamma` default `1.0` (literal, D-04 — NOT the `1/n_features` default of
//! `SpectralEmbedding`). `n_components` defaults to `n_clusters` (D-11).
//!
//! ## n_samples ≤ 64 (D-05 / D-06)
//! The Laplacian is `n_samples × n_samples` and v1 `eig` caps `n ≤ MAX_DIM = 64`.
//! `fit` rejects `n_samples > 64` with [`AlgoError::NSamplesExceedsMaxDim`] BEFORE
//! any affinity / Laplacian / eig launch (ASVS V5).
//!
//! ## Wave-0 scaffold status
//! This is the 09-01 Wave-0 COMPILING STUB: the struct + [`SpectralClustering::new`]
//! constructor are real (so the PyO3 wrapper + the test scaffolds compile today),
//! but the `fit` / `labels_` bodies are `todo!()` pending the Wave-3 plan (09-04).
//! Do NOT write the embedding + KMeans here — it is Wave 3.
//!
//! Tests live in `crates/mlrs-algos/tests/spectral_clustering_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;

use crate::error::AlgoError;

/// Spectral clustering (SPECTRAL-02): spectral embedding of an affinity graph
/// followed by v1 KMeans on the embedding (D-10).
///
/// Construct with [`SpectralClustering::new`] (`n_clusters`, `n_components`,
/// `affinity`, `gamma`, `n_neighbors`, `seed`), then `fit` and read `labels_`
/// (Wave-3). Fitted `labels_` (length `n`, `i32` — the KMeans i32 idiom) is
/// device-resident (D-03); the host accessor materializes it on demand.
pub struct SpectralClustering<F>
where
    F: Float + CubeElement + Pod,
{
    /// Number of clusters `k` (sklearn default `8`). Validated `1 ≤ k ≤ n_samples`
    /// at `fit` via the existing [`AlgoError::InvalidK`].
    n_clusters: usize,
    /// Embedding dimensionality; `None` resolves to `n_clusters` at `fit` (sklearn
    /// default `n_components = n_clusters`, D-11).
    n_components: Option<usize>,
    /// Affinity construction (`"rbf"` default, D-01 — `kernel_matrix(Rbf)`, D-02;
    /// `"nearest_neighbors"` uses the kNN-connectivity graph, D-03).
    affinity: String,
    /// Kernel coefficient `γ` for the rbf affinity (sklearn default `1.0` literal,
    /// D-04 — NOT the `1/n_features` default of `SpectralEmbedding`).
    gamma: F,
    /// Number of neighbors for the `"nearest_neighbors"` affinity (sklearn
    /// default `10`, D-03).
    n_neighbors: usize,
    /// Seed for the inner KMeans k-means++ host PRNG (D-10 — init-invariant on a
    /// well-separated fixture, so the exact seed is immaterial to the label gate).
    seed: u64,
    /// Fitted length-`n` integer labels (`i32`, the KMeans idiom), device-resident,
    /// `None` until `fit`.
    labels_: Option<DeviceArray<ActiveRuntime, i32>>,
}

impl<F> SpectralClustering<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `SpectralClustering`. `n_clusters` is the cluster count
    /// (sklearn default `8`), `n_components` the embedding dimension (`None` →
    /// `n_clusters` at `fit`, D-11), `affinity` selects the graph (`"rbf"` default
    /// / `"nearest_neighbors"`, D-01), `gamma` the rbf coefficient (default `1.0`,
    /// D-04), `n_neighbors` the kNN-connectivity neighbor count (default `10`,
    /// D-03), and `seed` the inner-KMeans PRNG seed. Invalid hyperparameters are
    /// rejected at `fit`, not construction.
    pub fn new(
        n_clusters: usize,
        n_components: Option<usize>,
        affinity: String,
        gamma: F,
        n_neighbors: usize,
        seed: u64,
    ) -> Self {
        Self {
            n_clusters,
            n_components,
            affinity,
            gamma,
            n_neighbors,
            seed,
            labels_: None,
        }
    }

    /// Host copy of the fitted `labels_` (length `n`, `i32`). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    ///
    /// **Wave-0 stub:** the `labels_` accessor body is `todo!()` pending the
    /// Wave-3 plan (09-04). The field + signature are real (the PyO3 wrapper +
    /// test scaffolds compile against this surface today).
    pub fn labels(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<i32>, AlgoError> {
        // Reference the field so the Wave-0 stub keeps the fitted-state seam.
        let _ = (&self.labels_, pool);
        todo!("SpectralClustering::labels is filled by the Wave-3 plan 09-04 (SPECTRAL-02)")
    }

    /// Fit the spectral clustering to the affinity graph of `x`
    /// (`shape = (n_samples, n_features)`, row-major). Rejects `n_samples > 64`
    /// with [`AlgoError::NSamplesExceedsMaxDim`] BEFORE any launch (D-06).
    ///
    /// **Wave-0 stub:** the `fit` body (affinity → Laplacian → eig recovery →
    /// KMeans) is `todo!()` pending the Wave-3 plan (09-04).
    pub fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        // Reference every field + arg so the Wave-0 stub fixes the full fit seam
        // (consumed by the Wave-3 body).
        let _ = (
            self.n_clusters,
            self.n_components,
            &self.affinity,
            self.gamma,
            self.n_neighbors,
            self.seed,
            &mut self.labels_,
            pool,
            x,
            shape,
        );
        todo!("SpectralClustering::fit is filled by the Wave-3 plan 09-04 (SPECTRAL-02)")
    }
}
