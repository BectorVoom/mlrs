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
//! The exact operation ORDER is load-bearing (RESEARCH §D-07/D-08, Pitfall 2):
//! slice smallest → `/dd` → sign-flip → drop-first. `dd` is the SAME vector the
//! Laplacian returned.
//!
//! ## Affinity defaults & gamma (D-01 / D-04)
//! Default affinity is `"nearest_neighbors"` (sklearn `SpectralEmbedding`'s own
//! default — the kNN-connectivity graph, D-03), NOT `"rbf"`. `gamma = None`
//! resolves to `1/n_features` at `fit` (the [`KernelRidge`](crate::kernel_ridge)
//! at-fit gamma resolution precedent, D-04); the rbf affinity path uses
//! `kernel_matrix(Rbf { gamma })` (D-02). The kNN-connectivity affinity is the
//! sklearn-exact binary connectivity graph (`include_self=True`,
//! `mode='connectivity'`, symmetrized `0.5·(A + Aᵀ)`, D-03).
//!
//! ## n_samples ≤ 64 (D-05 / D-06)
//! The Laplacian is `n_samples × n_samples` and v1 `eig` caps `n ≤ MAX_DIM = 64`.
//! `fit` rejects `n_samples > 64` with [`AlgoError::NSamplesExceedsMaxDim`] BEFORE
//! any affinity / Laplacian / eig launch (ASVS V5).
//!
//! Tests live in `crates/mlrs-algos/tests/spectral_embedding_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)] mod tests`).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::kernel_matrix::{kernel_matrix, Kernel};
use mlrs_backend::prims::laplacian::laplacian;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64};

// WR-06: shared spectral host recovery math (formerly duplicated in this file).
use crate::cluster::spectral::recover;
use crate::error::{AlgoError, BuildError};
// SHAPE A' (RESEARCH Open Q3): SpectralEmbedding had an INHERENT `fit`/accessor
// and NO legacy-trait import. The Phase-16 retrofit ADOPTS the typestate `Fit`
// trait (consuming-self) so the estimator joins the SINGLE trait surface and the
// traits.rs-gone grep (Plan 11) stays clean. SpectralEmbedding is non-transductive
// (like DBSCAN / sklearn's own SpectralEmbedding): it exposes the fitted
// `embedding_` accessor but NO `transform` for new points, so it does not adopt
// `Transform` (there is no inherent transform to adopt).
use crate::typestate::{validate_geometry, Fit, Fitted, Unfit};

/// The v1 dense-eig MAX_DIM cap (`eig.rs` `MAX_DIM = 64`). The normalized
/// Laplacian is `n_samples × n_samples`, so `n_samples ≤ 64` is the documented
/// spectral problem-size ceiling (D-05). Rejected at `fit` with the
/// spectral-domain [`AlgoError::NSamplesExceedsMaxDim`] (D-06), not deferred to
/// eig's generic `PrimError::NotSquare`.
const MAX_DIM: usize = 64;

/// Spectral embedding (SPECTRAL-01) of an affinity graph onto the smallest
/// non-trivial eigenvectors of the normalized Laplacian.
///
/// Construct with the zero-arg [`SpectralEmbedding::new`] (sklearn defaults:
/// `n_components = 2`, `affinity = "nearest_neighbors"`, `gamma = None`,
/// `n_neighbors = 10`) or [`SpectralEmbedding::builder`], then the consuming
/// [`Fit::fit`] (returns the `Fitted`-tagged sibling) and read `embedding_`.
/// Fitted `embedding_` (`n × n_components`) is device-resident; the host accessor
/// materializes it on demand and exists ONLY on `SpectralEmbedding<F, Fitted>`
/// (the compile-time typestate replaces the old runtime `NotFitted` guard, D-03).
/// SpectralEmbedding is non-transductive: there is NO `transform` for new points
/// (sklearn's own `SpectralEmbedding` likewise exposes only `fit_transform` /
/// `embedding_`).
pub struct SpectralEmbedding<F, S = Unfit> {
    /// Embedding dimensionality (sklearn default `2`, D-08). The smallest
    /// `n_components + 1` eigenvectors are computed and the trivial ≈0 one dropped.
    n_components: usize,
    /// Affinity construction (`"nearest_neighbors"` default, D-01 — the
    /// kNN-connectivity graph, D-03; `"rbf"` uses `kernel_matrix(Rbf)`, D-02).
    affinity: String,
    /// Kernel coefficient `γ` for the rbf affinity; `None` resolves to
    /// `1/n_features` at `fit` (D-04). Ignored for `"nearest_neighbors"`.
    gamma: Option<F>,
    /// Number of neighbors for the `"nearest_neighbors"` affinity (D-03).
    n_neighbors: usize,
    /// Fitted `n × n_components` embedding, device-resident, `None` until `fit`.
    embedding_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> SpectralEmbedding<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfitted `SpectralEmbedding` with sklearn's
    /// `SpectralEmbedding` defaults (`n_components = 2`,
    /// `affinity = "nearest_neighbors"`, `gamma = None`, `n_neighbors = 10`)
    /// directly in the `Unfit` state. SINGLE source of truth for the defaults
    /// (D-08): the builder `Default` re-derives via
    /// [`SpectralEmbedding::into_builder`]. Defaults are trusted valid, so this
    /// bypasses [`SpectralEmbeddingBuilder::build`]'s validation.
    pub fn new() -> Self {
        Self {
            n_components: 2,
            affinity: "nearest_neighbors".to_string(),
            gamma: None,
            n_neighbors: 10,
            embedding_: None,
            _state: PhantomData,
        }
    }

    /// Start building a `SpectralEmbedding` from sklearn's defaults (D-08 single
    /// source).
    pub fn builder() -> SpectralEmbeddingBuilder {
        SpectralEmbeddingBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`SpectralEmbeddingBuilder::default`] to re-derive
    /// the defaults from [`SpectralEmbedding::new`] (D-08).
    pub fn into_builder(self) -> SpectralEmbeddingBuilder {
        SpectralEmbeddingBuilder {
            n_components: self.n_components,
            affinity: self.affinity,
            gamma: self.gamma.map(host_to_f64),
            n_neighbors: self.n_neighbors,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `embedding_` is excluded — `None` in any `Unfit` value). Used by the
    /// defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_components == other.n_components
            && self.affinity == other.affinity
            && self.gamma.map(host_to_f64) == other.gamma.map(host_to_f64)
            && self.n_neighbors == other.n_neighbors
    }
}

impl<F> Default for SpectralEmbedding<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`SpectralEmbedding`] (D-01). `gamma` is `Option<f64>` (A5: the
/// scalar narrows to `Option<F>` at `build::<F>()`); `affinity` (`String`) takes
/// its value directly. `Default` re-derives the sklearn defaults from
/// [`SpectralEmbedding::new`] (D-08 single source).
#[derive(Debug, Clone)]
pub struct SpectralEmbeddingBuilder {
    n_components: usize,
    affinity: String,
    gamma: Option<f64>,
    n_neighbors: usize,
}

impl Default for SpectralEmbeddingBuilder {
    /// Re-derive the sklearn defaults from [`SpectralEmbedding::new`] (D-08 single
    /// source). `f64` is pinned only to read the F-independent scalar defaults.
    fn default() -> Self {
        SpectralEmbedding::<f64, Unfit>::new().into_builder()
    }
}

impl SpectralEmbeddingBuilder {
    /// Set the embedding dimensionality.
    pub fn n_components(mut self, v: usize) -> Self {
        self.n_components = v;
        self
    }

    /// Set the affinity construction (`"nearest_neighbors"` / `"rbf"`).
    pub fn affinity(mut self, v: String) -> Self {
        self.affinity = v;
        self
    }

    /// Set the rbf kernel coefficient `γ` (`None` → `1/n_features` at fit). The
    /// `Option<f64>` narrows to `Option<F>` at `build::<F>()` (A5).
    pub fn gamma(mut self, v: Option<f64>) -> Self {
        self.gamma = v;
        self
    }

    /// Set the neighbor count for the `"nearest_neighbors"` affinity.
    pub fn n_neighbors(mut self, v: usize) -> Self {
        self.n_neighbors = v;
        self
    }

    /// Build the (unfit) estimator, narrowing the stored `Option<f64>` `gamma` to
    /// the target float `Option<F>` (A5). SpectralEmbedding has no purely
    /// data-INDEPENDENT hyperparameter that is unconditionally validated: the
    /// `gamma > 0` check is affinity-branch-coupled (only the `"rbf"` path uses
    /// gamma) and the `n_components` check is data-DEPENDENT (it compares against
    /// `n_samples`), so both stay in the fit body (D-03 byte-identical). The
    /// `Result` is kept for family uniformity so the `build_err_to_py` PyO3 mapper
    /// is shape-identical across the Phase-16 builders.
    pub fn build<F>(self) -> Result<SpectralEmbedding<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        Ok(SpectralEmbedding {
            n_components: self.n_components,
            affinity: self.affinity,
            gamma: self.gamma.map(f64_to_host::<F>),
            n_neighbors: self.n_neighbors,
            embedding_: None,
            _state: PhantomData,
        })
    }
}

impl<F> SpectralEmbedding<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `embedding_` (`n × n_components` row-major). `Some`
    /// by construction on the `Fitted` state, so no `NotFitted` branch is needed
    /// (the compile-time typestate replaces the runtime guard, D-03).
    pub fn embedding(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.embedding_
            .as_ref()
            .expect("embedding_ is Some by construction on SpectralEmbedding<F, Fitted>")
            .to_host(pool)
    }
}

impl<F> Fit<F> for SpectralEmbedding<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = SpectralEmbedding<F, Fitted>;

    /// Fit the spectral embedding to the affinity graph of `x`
    /// (`shape = (n_samples, n_features)`, row-major), CONSUMING `self`. Rejects
    /// `n_samples > 64` with [`AlgoError::NSamplesExceedsMaxDim`] BEFORE any launch
    /// (D-06).
    ///
    /// Pipeline (RESEARCH System Diagram, pinned to sklearn `_spectral_embedding`
    /// order, D-07/D-08): affinity (rbf via `kernel_matrix(Rbf)` OR the
    /// kNN-connectivity builder) → `laplacian` `(L, dd)` → `eig(L)` (DESCENDING,
    /// reversed to ascending) → slice the smallest `n_components + 1` columns →
    /// `/dd` recovery → `_deterministic_vector_sign_flip` → drop the trivial
    /// row 0 → transpose → `embedding_` (`n × n_components`).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<SpectralEmbedding<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-9-VAL / ASVS V5: validate the untrusted hyperparameters +
        //     geometry BEFORE any affinity/Laplacian/eig device work. The
        //     n_samples > 64 guard names the SPECTRAL cap (D-06), NOT eig's
        //     generic PrimError::NotSquare. Mirrors kmeans.rs:234-252. ---
        if n_samples > MAX_DIM {
            return Err(AlgoError::NSamplesExceedsMaxDim {
                estimator: "spectral_embedding",
                n_samples,
                max: MAX_DIM,
            });
        }
        validate_geometry(x, shape)?;
        // The smallest `n_components + 1` eigenvectors must exist (drop_first).
        if self.n_components < 1 || self.n_components + 1 > n_samples {
            return Err(AlgoError::InvalidNComponents {
                estimator: "spectral_embedding",
                requested: self.n_components,
                max: n_samples.saturating_sub(1),
            });
        }

        // --- Build the affinity matrix A (n×n) by the `affinity` string. ---
        let a = match self.affinity.as_str() {
            // rbf (D-02/D-04): A = kernel_matrix(X, X, Rbf{gamma}). gamma=None →
            // 1/n_features resolved at fit (copy kernel_ridge's at-fit resolution,
            // D-04); the resolved gamma is validated finite (T-9-VAL / InvalidGamma).
            "rbf" => {
                let gamma = match self.gamma {
                    Some(g) => g,
                    None => f64_to_host::<F>(1.0 / n_features as f64),
                };
                let gamma64 = host_to_f64(gamma);
                // WR-04: sklearn's kernel-coefficient contract is
                // Interval(Real, 0, None, closed='neither') — STRICTLY positive.
                // gamma == 0 yields exp(0) = 1 for all pairs (a constant all-ones
                // affinity → degenerate graph); a negative gamma blows the affinity
                // up monotonically with distance. Reject gamma <= 0, not just the
                // non-finite case (the finiteness check is subsumed: NaN and ±inf
                // both fail `> 0.0`). The None → 1/n_features default is always
                // positive; this guards a user-supplied gamma.
                if !(gamma64 > 0.0) {
                    return Err(AlgoError::InvalidGamma {
                        estimator: "spectral_embedding",
                        gamma: gamma64,
                    });
                }
                // IN-03 (explicit parity decision): only `gamma <= 0` / non-finite
                // is rejected. A finite-positive gamma so tiny that `exp(-gamma*dist)`
                // underflows to an effective-constant all-ones affinity is ACCEPTED
                // — this matches sklearn, which likewise gates solely on
                // `Interval(Real, 0, None, closed='neither')` and does NOT guard the
                // effective-zero underflow boundary. Intentional sklearn parity, not
                // an oversight.
                kernel_matrix::<F>(
                    pool,
                    x,
                    (n_samples, n_features),
                    x,
                    (n_samples, n_features),
                    Kernel::Rbf { gamma },
                    None,
                )?
            }
            // nearest_neighbors (DEFAULT, D-03): the sklearn-exact binary
            // connectivity builder (RESEARCH Pattern 3).
            "nearest_neighbors" => {
                // WR-03: sklearn's kneighbors_graph does NOT error when
                // n_neighbors > n_samples — NearestNeighbors silently caps at
                // n_samples (effectively "use all available neighbors"). With the
                // SE default n_neighbors=10, ANY n_samples <= 10 (well within the
                // n<=64 cap) would otherwise raise on the default constructor.
                // Clamp to min(n_neighbors, n_samples); only reject k < 1.
                let k = self.n_neighbors.min(n_samples);
                if k < 1 {
                    return Err(AlgoError::InvalidK {
                        estimator: "spectral_embedding",
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
                    estimator: "spectral_embedding",
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
        //     input — saving one n² allocation (RESEARCH Anti-Pattern).
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

        // --- Post-eig recovery host math, reproducing the pinned sklearn
        //     _spectral_embedding order EXACTLY (RESEARCH §D-07/D-08): slice
        //     smallest → /dd → sign-flip → drop-first → transpose. ---
        // WR-06: drop_first = TRUE for SpectralEmbedding (D-08) — drop the trivial
        // ≈0 eigenvector. Shared recovery helper (was the local recover_embedding).
        let embedding_host =
            recover::<F>(&v_host, &dd_host, n_samples, self.n_components, true, true);
        let embedding_dev = DeviceArray::from_host(pool, &embedding_host);

        Ok(SpectralEmbedding {
            n_components: self.n_components,
            affinity: self.affinity,
            gamma: self.gamma,
            n_neighbors: self.n_neighbors,
            embedding_: Some(embedding_dev),
            _state: PhantomData,
        })
    }
}

// WR-06: `recover_embedding`, `host_to_f64`, and `f64_to_host` moved to the shared
// `crate::cluster::spectral` module (imported above) so the embedding and
// clustering recovery paths stay bit-identical by construction.
