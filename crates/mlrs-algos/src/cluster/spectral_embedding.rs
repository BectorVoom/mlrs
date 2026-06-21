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

use std::mem::size_of;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::kernel_matrix::{kernel_matrix, Kernel};
use mlrs_backend::prims::laplacian::laplacian;
use mlrs_backend::prims::topk::top_k;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;

/// The v1 dense-eig MAX_DIM cap (`eig.rs` `MAX_DIM = 64`). The normalized
/// Laplacian is `n_samples × n_samples`, so `n_samples ≤ 64` is the documented
/// spectral problem-size ceiling (D-05). Rejected at `fit` with the
/// spectral-domain [`AlgoError::NSamplesExceedsMaxDim`] (D-06), not deferred to
/// eig's generic `PrimError::NotSquare`.
const MAX_DIM: usize = 64;

/// Spectral embedding (SPECTRAL-01) of an affinity graph onto the smallest
/// non-trivial eigenvectors of the normalized Laplacian.
///
/// Construct with [`SpectralEmbedding::new`] (`n_components`, `affinity`,
/// `gamma`, `n_neighbors`), then `fit` and read `embedding_`. Fitted
/// `embedding_` (`n × n_components`) is device-resident; the host accessor
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
    /// Number of neighbors for the `"nearest_neighbors"` affinity (D-03).
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
    /// the kNN-connectivity neighbor count (D-03). Invalid hyperparameters are
    /// rejected at `fit`, not construction.
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
    pub fn embedding(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.embedding_
            .as_ref()
            .map(|e| e.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "spectral_embedding",
                operation: "embedding_",
            })
    }

    /// Fit the spectral embedding to the affinity graph of `x`
    /// (`shape = (n_samples, n_features)`, row-major). Rejects `n_samples > 64`
    /// with [`AlgoError::NSamplesExceedsMaxDim`] BEFORE any launch (D-06).
    ///
    /// Pipeline (RESEARCH System Diagram, pinned to sklearn `_spectral_embedding`
    /// order, D-07/D-08): affinity (rbf via `kernel_matrix(Rbf)` OR the
    /// kNN-connectivity builder) → `laplacian` `(L, dd)` → `eig(L)` (DESCENDING,
    /// reversed to ascending) → slice the smallest `n_components + 1` columns →
    /// `/dd` recovery → `_deterministic_vector_sign_flip` → drop the trivial
    /// row 0 → transpose → `embedding_` (`n × n_components`).
    pub fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
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
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
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
                if !gamma64.is_finite() {
                    return Err(AlgoError::InvalidGamma {
                        estimator: "spectral_embedding",
                        gamma: gamma64,
                    });
                }
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
                if self.n_neighbors < 1 || self.n_neighbors > n_samples {
                    return Err(AlgoError::InvalidK {
                        estimator: "spectral_embedding",
                        k: self.n_neighbors,
                        n_samples,
                    });
                }
                self.knn_connectivity_affinity(pool, x, n_samples, n_features)?
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
        //     input (RESEARCH Anti-Pattern — `out` is consumed/released after
        //     launch, eig.rs:149-151); `l` is not read again afterwards. ---
        let l_out = DeviceArray::<ActiveRuntime, F>::from_raw(
            l.handle().clone(),
            n_samples * n_samples,
        );
        let (w_desc, v_desc) = eig::<F>(pool, &l, n_samples, Some(l_out))?;
        drop(l); // the cloned handle was threaded through `out` and released by eig.
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
        let embedding_host =
            recover_embedding::<F>(&v_host, &dd_host, n_samples, self.n_components);
        let embedding_dev = DeviceArray::from_host(pool, &embedding_host);

        // --- Re-fit buffer reuse (WR-07): release a prior embedding allocation
        //     back to the pool free-list before reassigning. ---
        if let Some(old) = self.embedding_.take() {
            old.release_into(pool);
        }
        self.embedding_ = Some(embedding_dev);
        Ok(self)
    }

    /// Build the sklearn-exact binary kNN-connectivity affinity (D-03, RESEARCH
    /// Pattern 3): `distance(X, X, sqrt=false)` → `top_k(k = n_neighbors)` → set
    /// `A[i, j] = 1` for the `k` smallest-distance columns of row `i` (the self
    /// `d(i, i) = 0` is the row minimum, so `include_self=True` is automatic) →
    /// symmetrize `A = 0.5·(A + Aᵀ)`. Binary weights `0/1`, NOT distance weights.
    ///
    /// The top-k indices are read back to the host for the small `n × k` binarize
    /// + symmetrize; the resulting `n × n` affinity is uploaded device-resident
    /// for the Laplacian (which consumes it on-device).
    fn knn_connectivity_affinity(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        n: usize,
        d: usize,
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let k = self.n_neighbors;

        // Squared-euclidean distance (sqrt=false is order-preserving for top-k).
        let dist = distance::<F>(pool, x, (n, d), x, (n, d), false, None)?;
        // k smallest per row + their column indices (lowest-index tie-break).
        let (vals, idx) = top_k::<F>(pool, &dist, n, n, k, false, None, None)?;
        dist.release_into(pool);
        let idx_host = idx.to_host(pool);
        idx.release_into(pool);
        vals.release_into(pool);

        // Binarize: A[i, j] = 1 for the k nearest columns of row i (self included).
        let one = F::from_int(1i64);
        let zero = F::from_int(0i64);
        let mut a = vec![zero; n * n];
        for i in 0..n {
            for t in 0..k {
                let j = idx_host[i * k + t] as usize;
                a[i * n + j] = one;
            }
        }
        // Symmetrize A = 0.5·(A + Aᵀ): one-directional edges → 0.5, mutual → 1.0.
        let mut sym = vec![zero; n * n];
        for i in 0..n {
            for j in 0..n {
                let s = host_to_f64(a[i * n + j]) + host_to_f64(a[j * n + i]);
                sym[i * n + j] = if s == 0.0 {
                    zero
                } else {
                    f64_to_host::<F>(0.5 * s)
                };
            }
        }

        Ok(DeviceArray::from_host(pool, &sym))
    }
}

/// Reproduce the pinned sklearn `_spectral_embedding` recovery on the host
/// (RESEARCH §D-07/D-08, Code Examples). `v_host` is the DESCENDING eig `V`
/// column-major (`v_host[col*n + r] = V[r, col]`); `dd` is the length-`n` degree
/// vector the Laplacian returned. Returns the row-major `n × n_components`
/// embedding.
///
/// ORDER (load-bearing — a wrong order fails the value match):
///   1. slice the smallest `m = n_components + 1` eigenvectors (ascending; the
///      `r`-th smallest is descending column `n - 1 - r`) into an `m × n` array;
///   2. `emb[r][i] /= dd[i]` — the `D^-1/2` recovery, BEFORE the sign flip (D-07);
///   3. `_deterministic_vector_sign_flip` per ROW (argmax|row| → sign → multiply);
///   4. drop the trivial row 0 (drop_first); transpose → `n × n_components`.
fn recover_embedding<F>(v_host: &[f64], dd: &[f64], n: usize, n_components: usize) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    let m = n_components + 1; // drop_first = TRUE for SpectralEmbedding (D-08).

    // 1. Slice the m smallest eigenvectors into an m × n row-major array. v1 eig
    //    sorts w DESCENDING, so the r-th SMALLEST eigenvector is descending column
    //    (n - 1 - r) (RESEARCH Pitfall 3 / Eig column snippet).
    let mut emb = vec![0.0f64; m * n];
    for r in 0..m {
        let col = n - 1 - r;
        for i in 0..n {
            emb[r * n + i] = v_host[col * n + i];
        }
    }

    // 2. /dd recovery (D-07) — BEFORE the sign flip. dd is the Laplacian's
    //    returned degree vector (NOT a fresh sqrt).
    for r in 0..m {
        for i in 0..n {
            emb[r * n + i] /= dd[i];
        }
    }

    // 3. _deterministic_vector_sign_flip on the m × n array (per ROW): the
    //    largest-magnitude element of each eigenvector is made positive
    //    (sklearn extmath, exact). Lowest-index tie-break on equal magnitudes.
    for r in 0..m {
        let row = &emb[r * n..(r + 1) * n];
        let mut max_idx = 0usize;
        let mut max_abs = row[0].abs();
        for (i, &val) in row.iter().enumerate().skip(1) {
            if val.abs() > max_abs {
                max_abs = val.abs();
                max_idx = i;
            }
        }
        let sign = if emb[r * n + max_idx] < 0.0 { -1.0 } else { 1.0 };
        if sign < 0.0 {
            for i in 0..n {
                emb[r * n + i] = -emb[r * n + i];
            }
        }
    }

    // 4. Drop the trivial row 0 (drop_first) and transpose the kept rows
    //    1..=n_components into a row-major n × n_components embedding.
    let mut out = vec![F::from_int(0i64); n * n_components];
    for c in 0..n_components {
        let r = c + 1; // kept rows are 1..=n_components.
        for i in 0..n {
            out[i * n_components + c] = f64_to_host::<F>(emb[r * n + i]);
        }
    }
    out
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine (mirrors the
/// `kernel_ridge.rs` helper).
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("spectral_embedding is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("spectral_embedding is f32/f64 only"),
    }
}
