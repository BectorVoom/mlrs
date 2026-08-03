//! Host affinity builders for the spectral family — the sklearn
//! `_get_affinity_matrix` branches, reproduced on the host.
//!
//! `SpectralEmbedding` and `SpectralClustering` accept overlapping but NOT
//! identical affinity sets, and this module implements the union:
//!
//! | affinity | SE | SC | layout |
//! |---|---|---|---|
//! | `nearest_neighbors` | ✓ (default) | ✓ | SPARSE CSR |
//! | `precomputed_nearest_neighbors` | ✓ | ✓ | SPARSE CSR |
//! | `precomputed` | ✓ | ✓ | dense, pass-through |
//! | `rbf` | ✓ | ✓ (default) | dense |
//! | `linear` / `poly` / `polynomial` / `sigmoid` / `laplacian` / `cosine` / `chi2` / `additive_chi2` | ✗ | ✓ | dense |
//!
//! `SpectralEmbedding` reaches `rbf_kernel` DIRECTLY (only `gamma`, no
//! `degree`/`coef0`), while `SpectralClustering` routes through
//! `pairwise_kernels(..., filter_params=True)` and so accepts the whole kernel
//! family plus `degree`/`coef0`. Keeping both here means the two estimators
//! cannot drift on the shared branches.
//!
//! ## Why the kNN graph stays sparse
//! The `nearest_neighbors` affinity has at most `2·n·k` nonzeros. Densifying it
//! to `n²` — which the device path does, because `laplacian`/`eig` take a dense
//! `DeviceArray` — turns an `O(n·k)` matvec into an `O(n²)` one and makes the
//! memory, not the arithmetic, the limit: `n = 50 000` is 20 GB dense and 8 MB
//! sparse. The CSR built here feeds
//! [`NormAdj`](super::spectral_host::NormAdj) directly.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use super::spectral_host::{Csr, HostAffinity};
use crate::manifold::umap::Metric;
use crate::manifold::umap_host_knn::host_knn;

/// A dense kernel affinity, with sklearn's `pairwise_kernels` parameters
/// already resolved (a `None` `gamma` becomes `1/n_features` at the call site,
/// as sklearn does).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelSpec {
    /// `exp(-γ‖x−y‖²)` — `rbf_kernel`.
    Rbf { gamma: f64 },
    /// `x·y` — `linear_kernel`.
    Linear,
    /// `(γ·x·y + coef0)^degree` — `polynomial_kernel`.
    Poly {
        /// Kernel coefficient.
        gamma: f64,
        /// Polynomial degree (sklearn default `3`).
        degree: f64,
        /// Independent term (sklearn default `1`).
        coef0: f64,
    },
    /// `tanh(γ·x·y + coef0)` — `sigmoid_kernel`.
    Sigmoid {
        /// Kernel coefficient.
        gamma: f64,
        /// Independent term.
        coef0: f64,
    },
    /// `exp(-γ‖x−y‖₁)` — `laplacian_kernel`.
    Laplacian { gamma: f64 },
    /// `x·y / (‖x‖·‖y‖)` — `cosine_similarity`.
    Cosine,
    /// `exp(γ · additive_chi2(x, y))` — `chi2_kernel` (sklearn's `gamma`
    /// default here is `1.0`, NOT `1/n_features`).
    Chi2 { gamma: f64 },
    /// `−Σ (xᵢ−yᵢ)²/(xᵢ+yᵢ)` — `additive_chi2_kernel`.
    AdditiveChi2,
}

/// Every affinity string this module can build, parsed once so an unknown
/// string fails at the estimator boundary rather than deep in a builder.
#[derive(Debug, Clone, PartialEq)]
pub enum AffinityKind {
    /// The kNN connectivity graph over the raw feature matrix.
    NearestNeighbors,
    /// The kNN connectivity graph over a PRECOMPUTED `n × n` distance matrix.
    PrecomputedNearestNeighbors,
    /// `X` is already the affinity — pass through unchanged.
    Precomputed,
    /// A dense kernel over the raw feature matrix.
    Kernel(KernelSpec),
}

/// Resolve an affinity string into an [`AffinityKind`].
///
/// `gamma` is the ALREADY-RESOLVED coefficient (the caller applies sklearn's
/// `None → 1/n_features` default, since only it knows `n_features`); `degree`
/// and `coef0` are the `pairwise_kernels` extras. `allow_kernels` is `false`
/// for `SpectralEmbedding`, which supports `rbf` ONLY — it calls `rbf_kernel`
/// directly and never reaches `pairwise_kernels`.
pub fn parse_affinity(
    name: &str,
    gamma: f64,
    degree: f64,
    coef0: f64,
    allow_kernels: bool,
) -> Option<AffinityKind> {
    Some(match name {
        "nearest_neighbors" => AffinityKind::NearestNeighbors,
        "precomputed_nearest_neighbors" => AffinityKind::PrecomputedNearestNeighbors,
        "precomputed" => AffinityKind::Precomputed,
        "rbf" => AffinityKind::Kernel(KernelSpec::Rbf { gamma }),
        _ if !allow_kernels => return None,
        "linear" => AffinityKind::Kernel(KernelSpec::Linear),
        "poly" | "polynomial" => AffinityKind::Kernel(KernelSpec::Poly {
            gamma,
            degree,
            coef0,
        }),
        "sigmoid" => AffinityKind::Kernel(KernelSpec::Sigmoid { gamma, coef0 }),
        "laplacian" => AffinityKind::Kernel(KernelSpec::Laplacian { gamma }),
        "cosine" => AffinityKind::Kernel(KernelSpec::Cosine),
        // `chi2_kernel`'s own gamma default is 1.0, not 1/n_features — but
        // `SpectralClustering` always forwards its `gamma` parameter (default
        // 1.0) through `pairwise_kernels`, so the caller's value is used
        // verbatim, exactly as in sklearn.
        "chi2" => AffinityKind::Kernel(KernelSpec::Chi2 { gamma }),
        "additive_chi2" => AffinityKind::Kernel(KernelSpec::AdditiveChi2),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// kNN connectivity
// ---------------------------------------------------------------------------

/// The symmetrized binary kNN connectivity graph, in CSR.
///
/// Reproduces
/// `A = kneighbors_graph(X, k, include_self=True); A = 0.5·(A + Aᵀ)`:
/// each row keeps its `k` nearest rows INCLUDING itself, entries are binary
/// `1`, and the symmetrization leaves `1.0` on a mutual pair, `0.5` on a
/// one-directional one, and `1.0` on the diagonal.
///
/// `include_self=True` means the query set is `X` itself, so the self-match at
/// distance 0 always occupies the first slot and the row holds only `k − 1`
/// GENUINE neighbors. That is why the search below asks
/// [`host_knn`] — which is self-dropped — for `k − 1` and adds the diagonal
/// back explicitly: the two constructions select the identical set, including
/// under exact-duplicate rows, where both resolve the distance-0 tie by lowest
/// index.
pub fn knn_connectivity(x: &[f64], n: usize, d: usize, k: usize) -> Csr {
    let k = k.clamp(1, n.max(1));
    let k_other = k - 1;
    let (idx, _dist) = if k_other > 0 && n > 1 {
        host_knn(x, n, d, k_other, Metric::Euclidean)
    } else {
        (Vec::new(), Vec::new())
    };
    let stride = if idx.is_empty() { 0 } else { idx.len() / n };
    directed_to_symmetric_csr(n, stride, |i, t| idx[i * stride + t] as usize)
}

/// The `precomputed_nearest_neighbors` twin: the same connectivity graph, but
/// the `k − 1` nearest columns are read out of a PRECOMPUTED `n × n` distance
/// matrix rather than searched in feature space.
///
/// Ties break on the lowest column index, matching the `top_k` /
/// `NearestNeighbors` convention.
pub fn precomputed_knn_connectivity(dist: &[f64], n: usize, k: usize) -> Csr {
    let k = k.clamp(1, n.max(1));
    let k_other = k - 1;
    let stride = k_other;
    let mut idx = vec![0usize; n * stride.max(1)];
    if k_other > 0 {
        // Bounded insertion of the `k-1` smallest off-diagonal entries per row.
        for i in 0..n {
            let row = &dist[i * n..(i + 1) * n];
            let mut best: Vec<(f64, usize)> = Vec::with_capacity(k_other + 1);
            for (j, &v) in row.iter().enumerate() {
                if j == i {
                    continue;
                }
                if best.len() < k_other || v < best[best.len() - 1].0 {
                    let pos = best.partition_point(|&(bv, _)| bv <= v);
                    best.insert(pos, (v, j));
                    if best.len() > k_other {
                        best.pop();
                    }
                }
            }
            for (t, &(_, j)) in best.iter().enumerate() {
                idx[i * stride + t] = j;
            }
        }
    }
    directed_to_symmetric_csr(n, stride, |i, t| idx[i * stride + t])
}

/// Turn a directed adjacency (`row i → itself plus `stride` neighbors given by
/// `nbr(i, t)`) into the symmetrized `0.5·(D + Dᵀ)` CSR.
///
/// Every directed edge deposits `0.5` in BOTH triangles; duplicate deposits on
/// a mutual pair are then merged by summation, which is what turns a mutual
/// edge into `1.0` and a one-directional edge into `0.5`. The placement is a
/// counting sort over row indices — `O(n·k)` — rather than a global sort of the
/// `2·n·k` pairs.
fn directed_to_symmetric_csr<F>(n: usize, stride: usize, nbr: F) -> Csr
where
    F: Fn(usize, usize) -> usize,
{
    // Pass 1: how many raw entries land in each row.
    let mut counts = vec![0u32; n + 1];
    for i in 0..n {
        // The self edge (i, i) deposits into row i twice.
        counts[i] += 2;
        for t in 0..stride {
            let j = nbr(i, t);
            counts[i] += 1;
            counts[j] += 1;
        }
    }
    let mut starts = vec![0u32; n + 1];
    let mut acc = 0u32;
    for i in 0..n {
        starts[i] = acc;
        acc += counts[i];
    }
    starts[n] = acc;

    // Pass 2: place.
    let mut cursor = starts.clone();
    let mut cols = vec![0u32; acc as usize];
    for i in 0..n {
        let p = cursor[i] as usize;
        cols[p] = i as u32;
        cols[p + 1] = i as u32;
        cursor[i] += 2;
        for t in 0..stride {
            let j = nbr(i, t);
            cols[cursor[i] as usize] = j as u32;
            cursor[i] += 1;
            cols[cursor[j] as usize] = i as u32;
            cursor[j] += 1;
        }
    }

    // Pass 3: per-row sort + merge, summing 0.5 per deposit.
    let mut indptr = vec![0u32; n + 1];
    let mut indices: Vec<u32> = Vec::with_capacity(acc as usize / 2 + n);
    let mut data: Vec<f64> = Vec::with_capacity(acc as usize / 2 + n);
    for i in 0..n {
        let (lo, hi) = (starts[i] as usize, starts[i + 1] as usize);
        let row = &mut cols[lo..hi];
        row.sort_unstable();
        let mut t = 0usize;
        while t < row.len() {
            let j = row[t];
            let mut run = 0usize;
            while t + run < row.len() && row[t + run] == j {
                run += 1;
            }
            indices.push(j);
            data.push(0.5 * run as f64);
            t += run;
        }
        indptr[i + 1] = indices.len() as u32;
    }
    Csr {
        indptr,
        indices,
        data,
    }
}

// ---------------------------------------------------------------------------
// Dense kernels
// ---------------------------------------------------------------------------

/// The dense `n × n` kernel affinity, row-major.
///
/// Parallel over disjoint row blocks — every entry is an independent reduction
/// over the `d` features, so splitting never changes a value.
pub fn kernel_affinity(x: &[f64], n: usize, d: usize, spec: KernelSpec) -> Vec<f64> {
    // Cosine works on L2-normalized rows, so the per-pair work collapses to a
    // plain dot product (sklearn's `cosine_similarity` normalizes once too,
    // rather than dividing by the norms per pair).
    let normalized;
    let data: &[f64] = match spec {
        KernelSpec::Cosine => {
            normalized = l2_normalize_rows(x, n, d);
            &normalized
        }
        _ => x,
    };

    let mut out = vec![0.0f64; n * n];
    // Each output row costs `n` pair evaluations of `d` features each.
    super::spectral_host::par_row_blocks(&mut out, n, n.saturating_mul(d.max(1)), |i0, blk| {
        for (r, row_out) in blk.chunks_mut(n).enumerate() {
            let i = i0 + r;
            let xi = &data[i * d..(i + 1) * d];
            for (j, o) in row_out.iter_mut().enumerate() {
                let xj = &data[j * d..(j + 1) * d];
                *o = pair(xi, xj, spec);
            }
        }
    });
    out
}

/// One kernel evaluation. The `match` is on a `Copy` enum the optimizer can
/// hoist out of the caller's inner loop.
#[inline]
fn pair(a: &[f64], b: &[f64], spec: KernelSpec) -> f64 {
    match spec {
        KernelSpec::Rbf { gamma } => {
            let mut s = 0.0;
            for (&p, &q) in a.iter().zip(b.iter()) {
                let t = p - q;
                s += t * t;
            }
            (-gamma * s).exp()
        }
        KernelSpec::Linear | KernelSpec::Cosine => {
            let mut s = 0.0;
            for (&p, &q) in a.iter().zip(b.iter()) {
                s += p * q;
            }
            s
        }
        KernelSpec::Poly {
            gamma,
            degree,
            coef0,
        } => {
            let mut s = 0.0;
            for (&p, &q) in a.iter().zip(b.iter()) {
                s += p * q;
            }
            (gamma * s + coef0).powf(degree)
        }
        KernelSpec::Sigmoid { gamma, coef0 } => {
            let mut s = 0.0;
            for (&p, &q) in a.iter().zip(b.iter()) {
                s += p * q;
            }
            (gamma * s + coef0).tanh()
        }
        KernelSpec::Laplacian { gamma } => {
            let mut s = 0.0;
            for (&p, &q) in a.iter().zip(b.iter()) {
                s += (p - q).abs();
            }
            (-gamma * s).exp()
        }
        KernelSpec::Chi2 { gamma } => (gamma * additive_chi2(a, b)).exp(),
        KernelSpec::AdditiveChi2 => additive_chi2(a, b),
    }
}

/// `−Σ (aᵢ−bᵢ)² / (aᵢ+bᵢ)`, skipping a pair whose denominator is zero — the
/// `additive_chi2_kernel` convention (`0/0` contributes nothing rather than
/// poisoning the sum with a NaN).
#[inline]
fn additive_chi2(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (&p, &q) in a.iter().zip(b.iter()) {
        let den = p + q;
        if den != 0.0 {
            let num = p - q;
            s += num * num / den;
        }
    }
    -s
}

/// Row-wise L2 normalization; a zero row is left as zeros (sklearn's
/// `normalize` divides by 1 in that case, which is the same result).
fn l2_normalize_rows(x: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * d];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let mut s = 0.0;
        for &v in row {
            s += v * v;
        }
        let inv = if s > 0.0 { 1.0 / s.sqrt() } else { 0.0 };
        for (o, &v) in out[i * d..(i + 1) * d].iter_mut().zip(row.iter()) {
            *o = v * inv;
        }
    }
    out
}

/// Build the affinity for `kind` from the `n × d` design (or, for the
/// precomputed kinds, from the `n × n` matrix passed as `x`).
///
/// `n_neighbors` is the ALREADY-RESOLVED neighbor count (sklearn's
/// `SpectralEmbedding` default `None → max(n/10, 1)` is applied by the caller,
/// which is where `n_samples` is known).
pub fn build_affinity(
    kind: &AffinityKind,
    x: &[f64],
    n: usize,
    d: usize,
    n_neighbors: usize,
) -> HostAffinity {
    match kind {
        AffinityKind::NearestNeighbors => {
            HostAffinity::Sparse(knn_connectivity(x, n, d, n_neighbors))
        }
        AffinityKind::PrecomputedNearestNeighbors => {
            HostAffinity::Sparse(precomputed_knn_connectivity(x, n, n_neighbors))
        }
        // sklearn passes `X` through verbatim; `check_symmetric` inside
        // `_spectral_embedding` then averages it with its transpose if it is
        // asymmetric beyond 1e-10, which `symmetrize_dense` reproduces.
        AffinityKind::Precomputed => HostAffinity::Dense(symmetrize_dense(x, n)),
        AffinityKind::Kernel(spec) => HostAffinity::Dense(kernel_affinity(x, n, d, *spec)),
    }
}

/// sklearn's `check_symmetric(..., tol=1e-10)`: leave an already-symmetric
/// matrix untouched, otherwise replace it with `0.5·(A + Aᵀ)`.
fn symmetrize_dense(a: &[f64], n: usize) -> Vec<f64> {
    let mut sym = false;
    'outer: for i in 0..n {
        for j in (i + 1)..n {
            if (a[i * n + j] - a[j * n + i]).abs() > 1e-10 {
                sym = true;
                break 'outer;
            }
        }
    }
    if !sym {
        return a.to_vec();
    }
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    out
}
