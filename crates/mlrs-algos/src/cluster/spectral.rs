//! Shared spectral-family host recovery math (WR-06).
//!
//! `SpectralEmbedding` (SPECTRAL-01) and `SpectralClustering` (SPECTRAL-02) run
//! the IDENTICAL post-eig host recovery — slice the smallest eigenvectors →
//! `/dd` (`D^-1/2` diffusion-map recovery, D-07) → `_deterministic_vector_sign_flip`
//! → transpose — differing ONLY in `drop_first`:
//!
//! - `SpectralEmbedding` (`drop_first = true`, D-08): compute the smallest
//!   `n_components + 1` eigenvectors and DROP the trivial ≈0 one (row 0), keeping
//!   rows `1..=n_components`.
//! - `SpectralClustering` (`drop_first = false`, D-11): keep ALL `n_components`
//!   smallest eigenvectors (the trivial ≈0 one is KEPT as a KMeans feature).
//!
//! The exact operation ORDER is load-bearing (RESEARCH §D-07/D-08, Pitfall 2):
//! slice smallest (ascending) → `/dd` → sign-flip → drop-first/transpose. A wrong
//! order fails the sklearn value/label match. Factoring the two former verbatim
//! copies into [`recover`] keeps the embedding and clustering paths bit-identical
//! by construction (the 09-04 file-disjointness rationale expired once both files
//! landed — they must not silently desynchronize on a future fix).
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::topk::top_k;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64};

use crate::error::AlgoError;

/// Reproduce the pinned sklearn `_spectral_embedding` recovery on the host
/// (RESEARCH §D-07/D-08). `v_host` is the DESCENDING eig `V` column-major
/// (`v_host[col*n + r] = V[r, col]`); `dd` is the length-`n` degree vector the
/// Laplacian returned. Returns the row-major `n × n_components` recovered matrix
/// (the `embedding_` for SE, the KMeans `maps` for SC).
///
/// `drop_first` selects the per-estimator slice (the ONLY real difference, WR-06):
///   - `true`  → slice the smallest `n_components + 1` eigenvectors, drop the
///     trivial ≈0 row 0, keep rows `1..=n_components` (SpectralEmbedding, D-08).
///   - `false` → slice the smallest `n_components` eigenvectors, keep ALL rows
///     `0..n_components` (SpectralClustering, D-11).
///
/// `diffusion_recover` selects the post-slice transform family (Plan 14-03):
///   - `true`  → the sklearn `_spectral_embedding` recovery: `/dd` diffusion-map
///     scaling (D-07) followed by the deterministic per-row sign flip. This is
///     what `SpectralEmbedding`/`SpectralClustering` need (their pinned sklearn
///     reference applies BOTH).
///   - `false` → return the RAW eigenvectors of the symmetric-normalized
///     Laplacian unchanged (no `/dd`, no sign flip). This is what umap-learn's
///     `spectral_layout` returns — it decomposes the SAME `I − D^-1/2 A D^-1/2`
///     but uses `eigenvectors[:, order]` directly, with no diffusion recovery and
///     no sign convention (verified by dump-diff: the `/dd` path mismatches umap
///     by ~0.2, the raw path matches to ≤1e-6 — RESEARCH Q3/A3 confirmed empirically).
///
/// ORDER (load-bearing — a wrong order fails the value/label match):
///   1. slice the smallest `m` eigenvectors (ascending; the `r`-th smallest is
///      descending column `n - 1 - r`) into an `m × n` array;
///   2. (`diffusion_recover` only) `emb[r][i] /= dd[i]` — the `D^-1/2` recovery,
///      BEFORE the sign flip (D-07);
///   3. (`diffusion_recover` only) `_deterministic_vector_sign_flip` per ROW
///      (argmax|row| → sign → multiply);
///   4. keep rows per `drop_first`; transpose → row-major `n × n_components`.
pub fn recover<F>(
    v_host: &[f64],
    dd: &[f64],
    n: usize,
    n_components: usize,
    drop_first: bool,
    diffusion_recover: bool,
) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    // drop_first = TRUE (SE, D-08) needs one EXTRA eigenvector (the trivial ≈0 one
    // that is then dropped); drop_first = FALSE (SC, D-11) keeps all n_components.
    let m = if drop_first {
        n_components + 1
    } else {
        n_components
    };

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

    // Steps 2-3 are the sklearn `_spectral_embedding` DIFFUSION recovery, applied
    // only when `diffusion_recover` is set. umap-learn's `spectral_layout` skips
    // BOTH (it returns the raw symmetric-Laplacian eigenvectors) — gating them
    // keeps the spectral-family callers bit-identical while letting the UMAP path
    // reuse the slice + drop-first + transpose (RESEARCH Q3/A3).
    if diffusion_recover {
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
    }

    // 4. Keep the per-estimator rows and transpose into a row-major
    //    n × n_components matrix. drop_first = TRUE drops row 0 (kept rows are
    //    1..=n_components); drop_first = FALSE keeps rows 0..n_components.
    let row_offset = usize::from(drop_first);
    let mut out = vec![F::from_int(0i64); n * n_components];
    for c in 0..n_components {
        let r = c + row_offset;
        for i in 0..n {
            out[i * n_components + c] = f64_to_host::<F>(emb[r * n + i]);
        }
    }
    out
}

/// Build the sklearn-exact binary kNN-connectivity affinity (D-03, RESEARCH
/// Pattern 3), shared by both spectral estimators (IN-02 — formerly duplicated
/// verbatim in `spectral_embedding.rs` and `spectral_clustering.rs`).
///
/// `distance(X, X, sqrt=false)` → `top_k(k = n_neighbors)` → set `A[i, j] = 1` for
/// the `k` smallest-distance columns of row `i` (the self `d(i, i) = 0` is the row
/// minimum, so `include_self=True` is automatic) → symmetrize `A = 0.5·(A + Aᵀ)`.
/// Binary weights `0/1`, NOT distance weights.
///
/// The top-k indices are read back to the host for the small `n × k` binarize +
/// symmetrize; the resulting `n × n` affinity is uploaded device-resident for the
/// Laplacian (which consumes it on-device). `k` is the CLAMPED neighbor count
/// (`min(n_neighbors, n_samples)`, WR-03), passed in by the caller rather than read
/// from `self.n_neighbors`. Factoring this into one function keeps the two affinity
/// builders bit-identical by construction (they must not silently desync on a
/// future fix — the same rationale as [`recover`]).
pub(crate) fn knn_connectivity_affinity<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    k: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
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
