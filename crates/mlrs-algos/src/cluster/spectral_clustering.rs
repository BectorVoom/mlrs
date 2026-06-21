//! `SpectralClustering` (SPECTRAL-02) — spectral embedding → v1 KMeans, matching
//! `sklearn.cluster.SpectralClustering`.
//!
//! ## Pipeline (RESEARCH System Diagram + Pattern 2)
//! `affinity A` (rbf via `kernel_matrix(Rbf)` D-02, or kNN-connectivity D-03) →
//! normalized Laplacian `(L, dd) = laplacian(A, n)` → the smallest `n_components`
//! eigenvectors via v1 `eig` (reversed to ascending; `drop_first = false` for SC,
//! D-11 — the trivial ≈0 eigenvector is KEPT) → `D^-1/2` recovery (`/ dd`, D-07) →
//! [`KMeans::new`](crate::cluster::kmeans::KMeans) (default kmeans++, `n_init = 1`,
//! D-10) → `labels_`. `labels_` matches sklearn up to a label permutation (the
//! exact-labels gate via a well-separated fixture, D-10).
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
//! Tests live in `crates/mlrs-algos/tests/spectral_clustering_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::kernel_matrix::{kernel_matrix, Kernel};
use mlrs_backend::prims::laplacian::laplacian;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::cluster::kmeans::KMeans;
// WR-06 / IN-02: shared spectral host recovery math + kNN-connectivity affinity
// builder (formerly duplicated in this file).
use crate::cluster::spectral::{host_to_f64, recover};
use crate::error::AlgoError;
use crate::traits::Fit;

/// The v1 dense-eig MAX_DIM cap (`eig.rs` `MAX_DIM = 64`). The normalized
/// Laplacian is `n_samples × n_samples`, so `n_samples ≤ 64` is the documented
/// spectral problem-size ceiling (D-05). Rejected at `fit` with the
/// spectral-domain [`AlgoError::NSamplesExceedsMaxDim`] (D-06).
const MAX_DIM: usize = 64;

/// Spectral clustering (SPECTRAL-02): spectral embedding of an affinity graph
/// followed by v1 KMeans on the embedding (D-10).
///
/// Construct with [`SpectralClustering::new`] (`n_clusters`, `n_components`,
/// `affinity`, `gamma`, `n_neighbors`, `seed`), then `fit` and read `labels_`.
/// Fitted `labels_` (length `n`, `i32` — the KMeans i32 idiom) is device-resident
/// (D-03); the host accessor materializes it on demand.
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
    /// WR-03: the host copy of `labels_` produced by the most recent `fit`, kept so
    /// `fit_predict` can build its returned device buffer WITHOUT the extra
    /// device→host read-back that calling `self.labels(pool)` would incur. `fit`
    /// already materializes these labels on the host before uploading to
    /// `labels_`, so retaining them is free; the only cost is `n` i32s of host
    /// memory, which is reclaimed on the next `fit`.
    labels_host_: Option<Vec<i32>>,
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
            labels_host_: None,
        }
    }

    /// Host copy of the fitted `labels_` (length `n`, `i32`). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn labels(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<i32>, AlgoError> {
        self.labels_
            .as_ref()
            .map(|l| l.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "spectral_clustering",
                operation: "labels_",
            })
    }

    /// Fit the spectral clustering to the affinity graph of `x`
    /// (`shape = (n_samples, n_features)`, row-major). Rejects `n_samples > 64`
    /// with [`AlgoError::NSamplesExceedsMaxDim`] BEFORE any launch (D-06).
    ///
    /// Pipeline (D-11, pinned to sklearn `SpectralClustering.fit`): affinity (rbf
    /// via `kernel_matrix(Rbf{gamma})`, D-02/D-04 literal gamma; OR the
    /// kNN-connectivity builder) → `laplacian` `(L, dd)` → `eig(L)` (DESCENDING,
    /// reversed to ascending) → slice the smallest `n_components` columns →
    /// `/dd` recovery → `_deterministic_vector_sign_flip` → KEEP row 0
    /// (`drop_first = FALSE`, D-11) → `maps` (`n × n_components`) →
    /// `KMeans::new(n_clusters, seed).fit(maps)` (kmeans++, `n_init = 1`, D-10) →
    /// `labels_`.
    pub fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-9-VAL / ASVS V5: validate the untrusted hyperparameters +
        //     geometry BEFORE any affinity/Laplacian/eig/KMeans device work. The
        //     n_samples > 64 guard names the SPECTRAL cap (D-06), NOT eig's
        //     generic PrimError::NotSquare. Mirrors kmeans.rs:238 /
        //     spectral_embedding.rs:141. ---
        if n_samples > MAX_DIM {
            return Err(AlgoError::NSamplesExceedsMaxDim {
                estimator: "spectral_clustering",
                n_samples,
                max: MAX_DIM,
            });
        }
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        // n_clusters: 1 ≤ k ≤ n_samples (the KMeans gate, surfaced here pre-launch).
        if self.n_clusters < 1 || self.n_clusters > n_samples {
            return Err(AlgoError::InvalidK {
                estimator: "spectral_clustering",
                k: self.n_clusters,
                n_samples,
            });
        }
        // n_components defaults to n_clusters (D-11). drop_first=FALSE keeps ALL
        // n_components eigenvectors (including the trivial ≈0 one), so we need the
        // smallest n_components eigenvectors to exist (n_components ≤ n_samples).
        let n_components = self.n_components.unwrap_or(self.n_clusters);
        if n_components < 1 || n_components > n_samples {
            return Err(AlgoError::InvalidNComponents {
                estimator: "spectral_clustering",
                requested: n_components,
                max: n_samples,
            });
        }

        // --- Build the affinity matrix A (n×n) by the `affinity` string. ---
        let a = match self.affinity.as_str() {
            // rbf (D-02/D-04): A = kernel_matrix(X, X, Rbf{gamma}). For SC the
            // gamma default is the LITERAL 1.0 (D-04 — NO 1/n_features fork); the
            // typed `self.gamma` is the resolved coefficient, validated finite
            // (T-9-VAL / InvalidGamma).
            "rbf" => {
                let gamma64 = host_to_f64(self.gamma);
                // WR-04: sklearn's kernel-coefficient contract is
                // Interval(Real, 0, None, closed='neither') — STRICTLY positive.
                // gamma == 0 yields exp(0) = 1 for all pairs (a constant all-ones
                // affinity → degenerate graph); a negative gamma blows the affinity
                // up monotonically with distance. Reject gamma <= 0, not just the
                // non-finite case (the finiteness check is subsumed: NaN and ±inf
                // both fail `> 0.0`).
                if !(gamma64 > 0.0) {
                    return Err(AlgoError::InvalidGamma {
                        estimator: "spectral_clustering",
                        gamma: gamma64,
                    });
                }
                kernel_matrix::<F>(
                    pool,
                    x,
                    (n_samples, n_features),
                    x,
                    (n_samples, n_features),
                    Kernel::Rbf { gamma: self.gamma },
                    None,
                )?
            }
            // nearest_neighbors (D-03): the sklearn-exact binary connectivity
            // builder (RESEARCH Pattern 3), shared with SpectralEmbedding.
            "nearest_neighbors" => {
                // WR-03: sklearn's kneighbors_graph does NOT error when
                // n_neighbors > n_samples — NearestNeighbors silently caps at
                // n_samples. Clamp to min(n_neighbors, n_samples); only reject
                // k < 1. Mirrors SpectralEmbedding.
                let k = self.n_neighbors.min(n_samples);
                if k < 1 {
                    return Err(AlgoError::InvalidK {
                        estimator: "spectral_clustering",
                        k: self.n_neighbors,
                        n_samples,
                    });
                }
                // IN-02: shared free function (was a verbatim per-estimator method).
                crate::cluster::spectral::knn_connectivity_affinity::<F>(
                    pool, x, n_samples, n_features, k,
                )?
            }
            // Any other affinity string is out of scope (CONTEXT — precomputed /
            // precomputed_nearest_neighbors deferred). Fail loud with a typed error.
            other => {
                return Err(AlgoError::InvalidKernel {
                    estimator: "spectral_clustering",
                    kernel: other.to_string(),
                });
            }
        };

        // --- Normalized Laplacian (L, dd) = laplacian(A, n) (PRIM-09). `dd` is
        //     the D^(1/2) degree vector used for the /dd recovery (D-07). ---
        let (l, dd) = laplacian::<F>(pool, &a, n_samples)?;
        a.release_into(pool);
        let dd_host: Vec<f64> = dd.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        dd.release_into(pool);

        // --- Full symmetric spectrum via v1 eig (DESCENDING, V col-major). Thread
        //     the Laplacian buffer through `out` so eig reuses it as its working
        //     input — saving one n² allocation. Mirrors spectral_embedding.rs.
        //
        //     WR-05: `&l` (the eig `a` input) and `l_out` (the eig `out`) wrap the
        //     SAME ref-counted cubecl handle (l.handle().clone()). This aliasing is
        //     SOUND only because of two load-bearing, eig-internal invariants:
        //       (1) eig READS `a_in` (= the `out` handle) and NEVER writes it — it
        //           writes its separate w/V/info outputs (eig.rs jacobi_eig_sweep);
        //       (2) eig ACQUIRES w/V/info from the pool BEFORE it releases the `out`
        //           working buffer (eig.rs: acquire happens before the
        //           `a_in_owned.release_into(pool)` post-launch), so the freed
        //           handle is never re-handed mid-call.
        //     If eig ever writes its working input in place, or reorders the
        //     acquire/release, this reuse becomes an aliased-write / use-after-free
        //     with NO compile-time signal — keep those invariants if eig changes.
        let l_out = DeviceArray::<ActiveRuntime, F>::from_raw(
            l.handle().clone(),
            n_samples * n_samples,
        );
        let (w_desc, v_desc) = eig::<F>(pool, &l, n_samples, Some(l_out))?;
        // eig released the CLONE threaded through `out`; this drops `l`'s remaining
        // handle clone (the ref-counted buffer's last owner returns it to the pool).
        // `l` is not read again afterwards.
        drop(l);
        w_desc.release_into(pool);
        let v_host: Vec<f64> = v_desc
            .to_host(pool)
            .iter()
            .map(|&v| host_to_f64(v))
            .collect();
        v_desc.release_into(pool);

        // --- Post-eig recovery host math (D-07/D-11): slice the smallest
        //     n_components → /dd → sign-flip → KEEP row 0 (drop_first = FALSE for
        //     SC) → transpose into the n × n_components `maps`. ---
        // WR-06: drop_first = FALSE for SpectralClustering (D-11) — KEEP the trivial
        // ≈0 eigenvector as a KMeans feature. Shared recovery helper (was recover_maps).
        let maps_host = recover::<F>(&v_host, &dd_host, n_samples, n_components, false);
        let maps_dev = DeviceArray::from_host(pool, &maps_host);

        // --- v1 KMeans on the embedding (D-10): KMeans::new (kmeans++, n_init=1;
        //     NOT with_init — init-injection is rejected for SC). The
        //     well-separated fixture makes the partition unique up to permutation,
        //     so the SplitMix64-vs-MT19937 RNG gap is immaterial to the labels. ---
        let mut kmeans = KMeans::<F>::new(self.n_clusters, self.seed);
        kmeans.fit(pool, &maps_dev, None, (n_samples, n_components))?;
        maps_dev.release_into(pool);

        let labels_host = kmeans.labels(pool)?;
        // WR-01: the function-local KMeans owns fitted device buffers
        // (cluster_centers_ k×n_components, labels_ i32 length n). DeviceArray has
        // no Drop, so return them to the pool before `kmeans` falls out of scope —
        // otherwise their acquired bytes leak the pool accounting (live_bytes grows
        // monotonically across re-fits, the FOUND-05 invariant). Done AFTER copying
        // the labels to the host.
        kmeans.release_into(pool);
        let labels_dev: DeviceArray<ActiveRuntime, i32> =
            DeviceArray::from_host(pool, &labels_host);

        // --- Re-fit buffer reuse (WR-07): release a prior labels allocation back
        //     to the pool free-list before reassigning. ---
        if let Some(old) = self.labels_.take() {
            old.release_into(pool);
        }
        self.labels_ = Some(labels_dev);
        // WR-03: retain the host labels (already materialized above) so a following
        // `fit_predict` can rebuild a fresh device buffer without a redundant
        // device→host read-back of `labels_`.
        self.labels_host_ = Some(labels_host);
        Ok(self)
    }

    /// Convenience `fit_predict` (sklearn `ClusterMixin`): fit to `x` then return
    /// the fitted `labels_` as a fresh device-resident `i32` buffer.
    ///
    /// WR-03: builds the returned buffer directly from the host labels `fit` just
    /// materialized (`labels_host_`), avoiding the extra device→host→device round
    /// trip that calling `self.labels(pool)` (a fresh read-back of `labels_`) would
    /// incur. The returned buffer is an INDEPENDENT device allocation — it does not
    /// alias `self.labels_`, so the caller may `release_into` it freely.
    pub fn fit_predict(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        self.fit(pool, x, shape)?;
        // `fit` always sets `labels_host_` on success; the `ok_or` is a defensive
        // fallback that cannot trigger on the post-`fit` path.
        let labels = self
            .labels_host_
            .as_ref()
            .ok_or(AlgoError::NotFitted {
                estimator: "spectral_clustering",
                operation: "labels_",
            })?;
        Ok(DeviceArray::from_host(pool, labels))
    }
}

// WR-06: `recover_maps` (now `recover(.., drop_first = false)`), `host_to_f64`, and
// `f64_to_host` moved to the shared `crate::cluster::spectral` module (imported
// above) so the embedding and clustering recovery paths stay bit-identical.
// IN-02: `knn_connectivity_affinity` likewise moved to the shared module (was a
// verbatim copy of the SpectralEmbedding builder) so the two cannot drift.
