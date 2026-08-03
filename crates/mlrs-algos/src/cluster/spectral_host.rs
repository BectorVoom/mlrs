//! Host (CPU) spectral solver — the sklearn `_spectral_embedding` pipeline run
//! entirely on the host, shared by `SpectralEmbedding` and `SpectralClustering`.
//!
//! ## Why this exists
//! The device path materializes a DENSE `n × n` affinity, a DENSE `n × n`
//! Laplacian, and then asks the cyclic-Jacobi `eig` kernel for the FULL
//! spectrum. That kernel stages `a_sh`/`v_sh` as comptime-sized shared memory at
//! `MAX_DIM × MAX_DIM`, which caps `n ≤ 64` — so the spectral family could only
//! ever be fitted on toy inputs, and even there it paid `O(n³)` to obtain the
//! `n_components + 1` eigenpairs the estimator actually wants.
//!
//! sklearn does neither. Its default `affinity="nearest_neighbors"` graph is
//! SPARSE (`~n·k` nonzeros) and its default `eigen_solver=None → "arpack"` asks
//! for just the smallest `k` eigenpairs. This module takes the same two
//! decisions, minus ARPACK's shift-invert sparse factorization:
//!
//! - the kNN affinity stays in CSR and is never densified;
//! - the eigenproblem is solved by a restarted BLOCK KRYLOV iteration whose only
//!   contact with the matrix is a matrix-vector product, so the cost is
//!   `O(iters · nnz)` rather than `O(n³)` — and there is no factorization at
//!   all, which is where ARPACK's `sigma=-1e-5` shift-invert spends its time.
//!
//! Two things make that iteration trustworthy on the graphs this family is
//! actually pointed at, both of which a naive single-vector Lanczos gets wrong:
//! the start is a BLOCK (a single-vector Krylov space holds only ONE direction
//! from any eigenspace, so a repeated eigenvalue is silently under-resolved —
//! see [`lanczos_largest`]), and a disconnected graph's null space is written
//! down in CLOSED FORM rather than iterated for (see [`null_space_basis`]).
//!
//! ## Exactly what is being decomposed (sklearn parity, verified line by line)
//! `scipy.sparse.csgraph.laplacian(A, normed=True, return_diag=True)` EXCLUDES
//! the diagonal of `A` from the degree (the dense arm zeroes it, the sparse arm
//! subtracts `A.diagonal()`), so with `dd[i] = sqrt(Σ_{j≠i} A[i,j])` — and
//! `dd[i] = 1` on an isolated node, scipy's `where(deg == 0, 1, sqrt(deg))` —
//! the Laplacian is
//!
//! ```text
//!   L[i,j] = -A[i,j] / (dd[i]·dd[j])   (i ≠ j)
//!   L[i,i] = 1
//! ```
//!
//! That `L[i,i] = 1` is NOT what scipy writes — scipy writes `1 - isolated[i]`,
//! i.e. `0` on an isolated node — but sklearn's `_set_diag(laplacian, 1, ...)`
//! then forces it to `1` unconditionally before the ARPACK call. The device
//! path never applied `_set_diag`, so it silently disagreed with sklearn on any
//! graph containing a zero-degree node; this path applies it.
//!
//! Writing `M = I − L` (zero diagonal, `M[i,j] = A[i,j]/(dd[i]·dd[j])`), the
//! `n_components + 1` SMALLEST eigenpairs of `L` are the `n_components + 1`
//! LARGEST of `S = M + I = 2·I − L`, whose spectrum lies in `[0, 2]` and is
//! therefore positive semi-definite. Lanczos converges fastest at the ends of a
//! spectrum, and the `+I` shift makes "largest algebraic" and "largest
//! magnitude" the same request — so a bipartite-ish graph, whose `M` has
//! eigenvalues near `−1`, cannot lure the iteration onto the wrong end.
//!
//! ## Solver routing
//! `n ≤ DENSE_N` (64) takes a direct dense symmetric eigendecomposition of the
//! full `L` via [`crate::linear::sym_eig::sym_eig`] (Householder +
//! implicit-shift QL, already oracle-tested and uncapped) — sklearn's own
//! small-graph reference is the same `scipy.linalg.eigh(L)`. That arm is
//! insurance for the smallest graphs, not a performance path: it is `O(n³)` and
//! loses to Lanczos from about `n = 100` upward. Above `DENSE_N`, and whenever
//! `nev` is a large fraction of `n` (where a Krylov method has no room to
//! help), Lanczos takes over.
//!
//! ## Label assignment (`SpectralClustering` only)
//! The tail of `sklearn.cluster.SpectralClustering.fit` — `assign_labels` ∈
//! {`kmeans`, `discretize`, `cluster_qr`} — also lives here rather than in the
//! estimator, because all three consume the SAME `n × n_components` embedding
//! this module produces and none of them touch a device buffer. See
//! [`AssignLabels`], [`host_kmeans`], [`discretize_labels`] and
//! [`cluster_qr_labels`].
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use mlrs_backend::capability;
use mlrs_backend::prims::rng::SplitMix64;

use crate::error::AlgoError;
use crate::linear::sym_eig::sym_eig;

/// Below this order [`smallest_laplacian_vectors`] takes the dense symmetric
/// eigensolver instead of Lanczos.
///
/// It is deliberately SMALL. The dense arm is `O(n³)` with an unvectorized
/// scalar constant, and MEASURED on this host it is already 4.6x slower than
/// Lanczos at `n = 120` (3.40 ms vs 0.73 ms) and 27x slower at `n = 200`
/// (24.4 ms vs 0.91 ms) — a full spectrum is simply the wrong thing to compute
/// when the estimator wants `n_components + 1` eigenpairs out of it. The dense
/// arm survives only as insurance on the smallest graphs, where the whole fit is
/// a fraction of a millisecond either way and an iterative solver has the least
/// room to converge: it cannot mis-handle the clustered or exactly-degenerate
/// spectra the tiny oracle fixtures are built from, because it does not iterate.
///
/// The two arms were A/B'd through the `MLRS_SPECTRAL_DENSE_N` knob across
/// `n = 65 … 511` on connected graphs and agree to **2.0e-14** worst case, so
/// this constant trades no accuracy — see `lanczos_matches_dense` and
/// `lanczos_matches_dense_across_orders`.
pub const DENSE_N: usize = 64;

/// Smallest amount of scalar work (multiply-accumulates) worth handing to its own
/// scoped thread.
///
/// The Lanczos loop is the reason this is expressed as WORK rather than as a row
/// count. A row-count gate (the `umap_host_knn` /
/// `host_core::par_row_chunks` precedent) is right for those callers, which make
/// one pass over a large matrix — but Lanczos makes thousands of tiny passes: at
/// `n = 500` a restart runs ~26 matvecs, each of which was spawning two threads
/// to divide ~50 000 multiply-accumulates between them. `std::thread::scope`
/// costs tens of microseconds per spawn, so the fit spent most of its time
/// creating and joining threads that had almost nothing to do — measured as a
/// 3-4x LOSS to sklearn at `n = 500` before this gate existed, on a rung mlrs
/// wins comfortably once the small passes run inline.
///
/// 65 536 MACs is roughly 20 microseconds of scalar work on this host, i.e. the
/// point where a spawn stops being the dominant term.
const MIN_PAR_WORK: usize = 1 << 16;

/// Eigenvalue-residual target for the Lanczos restart loop. `‖S‖ ≤ 2`, so this
/// is an absolute residual of ~2e-12 — tight enough that the recovered
/// eigenvectors agree with sklearn's `tol=0` ARPACK run well inside the 1e-5
/// oracle band, loose enough to converge in a bounded number of restarts.
const LANCZOS_TOL: f64 = 1e-12;

/// Restart budget. Each restart runs `m − nlock` matvecs; hitting this cap means
/// the spectrum is pathologically clustered, and the best Ritz pairs so far are
/// returned rather than failing the fit (ARPACK likewise returns whatever it
/// converged to).
const LANCZOS_MAX_RESTARTS: usize = 60;

// ---------------------------------------------------------------------------
// Affinity storage
// ---------------------------------------------------------------------------

/// A symmetric sparse matrix in CSR, with the diagonal STORED — the kNN
/// connectivity graph has `A[i,i] = 1` from `include_self=True`, and
/// `affinity_matrix_` must expose it even though the Laplacian excludes it.
#[derive(Debug, Clone)]
pub struct Csr {
    /// Row starts, length `n + 1`.
    pub indptr: Vec<u32>,
    /// Column index per stored entry, ascending within a row.
    pub indices: Vec<u32>,
    /// Value per stored entry.
    pub data: Vec<f64>,
}

/// The affinity matrix in whichever layout its builder produced. A kNN
/// connectivity graph is sparse and stays sparse all the way into the matvec; a
/// kernel affinity (`rbf`, `poly`, …) is dense by construction, as it is in
/// sklearn.
#[derive(Debug, Clone)]
pub enum HostAffinity {
    /// `n × n` CSR (the `nearest_neighbors` /
    /// `precomputed_nearest_neighbors` affinities).
    Sparse(Csr),
    /// `n × n` row-major dense (the kernel and `precomputed` affinities).
    Dense(Vec<f64>),
}

impl HostAffinity {
    /// Row-major dense materialization, for the `affinity_matrix_` accessor
    /// when the caller asked for a dense view.
    pub fn to_dense(&self, n: usize) -> Vec<f64> {
        match self {
            HostAffinity::Dense(d) => d.clone(),
            HostAffinity::Sparse(c) => {
                let mut out = vec![0.0f64; n * n];
                for i in 0..n {
                    let (lo, hi) = (c.indptr[i] as usize, c.indptr[i + 1] as usize);
                    for t in lo..hi {
                        out[i * n + c.indices[t] as usize] = c.data[t];
                    }
                }
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The normalized-adjacency operator
// ---------------------------------------------------------------------------

/// `S = M + I`, where `M[i,j] = A[i,j] / (dd[i]·dd[j])` off the diagonal and
/// `M[i,i] = 0` — the matrix whose LARGEST eigenpairs are the SMALLEST of the
/// normalized Laplacian `L = I − M` (module docs).
///
/// The `1/(dd[i]·dd[j])` scaling is folded into the stored values ONCE at
/// construction rather than applied per matvec: the Lanczos loop touches the
/// matrix hundreds of times, and a divide in that inner loop would dominate a
/// kernel which is otherwise pure multiply-accumulate.
pub struct NormAdj {
    /// Scaled `M`, diagonal already zeroed.
    m: HostAffinity,
    /// Order.
    pub n: usize,
    /// `dd[i] = sqrt(Σ_{j≠i} A[i,j])`, or `1` on an isolated node — scipy's
    /// `where(deg == 0, 1, sqrt(deg))`. This is the vector the `/dd` diffusion
    /// recovery divides by, so it is kept for the caller verbatim.
    pub dd: Vec<f64>,
}

impl NormAdj {
    /// Build the operator, consuming the affinity (the scaling is applied in
    /// place — the caller keeps its own copy for `affinity_matrix_`).
    pub fn new(mut aff: HostAffinity, n: usize) -> Self {
        // Degrees EXCLUDING the diagonal: scipy zeroes/subtracts it before the
        // reduction, and the kNN graph's `A[i,i] = 1` must not inflate `deg`.
        let mut deg = vec![0.0f64; n];
        match &aff {
            HostAffinity::Sparse(c) => {
                for (i, d) in deg.iter_mut().enumerate() {
                    let (lo, hi) = (c.indptr[i] as usize, c.indptr[i + 1] as usize);
                    let mut s = 0.0;
                    for t in lo..hi {
                        if c.indices[t] as usize != i {
                            s += c.data[t];
                        }
                    }
                    *d = s;
                }
            }
            HostAffinity::Dense(dm) => {
                for (i, d) in deg.iter_mut().enumerate() {
                    let row = &dm[i * n..(i + 1) * n];
                    let mut s = 0.0;
                    for (j, &v) in row.iter().enumerate() {
                        if j != i {
                            s += v;
                        }
                    }
                    *d = s;
                }
            }
        }
        let dd: Vec<f64> = deg
            .iter()
            .map(|&w| if w == 0.0 { 1.0 } else { w.sqrt() })
            .collect();
        let inv: Vec<f64> = dd.iter().map(|&w| 1.0 / w).collect();

        match &mut aff {
            HostAffinity::Sparse(c) => {
                for i in 0..n {
                    let (lo, hi) = (c.indptr[i] as usize, c.indptr[i + 1] as usize);
                    let si = inv[i];
                    for t in lo..hi {
                        let j = c.indices[t] as usize;
                        c.data[t] = if j == i { 0.0 } else { c.data[t] * si * inv[j] };
                    }
                }
            }
            HostAffinity::Dense(dm) => {
                for i in 0..n {
                    let si = inv[i];
                    for j in 0..n {
                        dm[i * n + j] = if j == i {
                            0.0
                        } else {
                            dm[i * n + j] * si * inv[j]
                        };
                    }
                }
            }
        }

        NormAdj { m: aff, n, dd }
    }

    /// `y = S·x = M·x + x`, parallel over disjoint row blocks of `y`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        let n = self.n;
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), n);
        let m = &self.m;
        // Per-row work: the row's nonzero count for CSR (amortized as the mean),
        // the full width for a dense affinity.
        let work_per_row = match m {
            HostAffinity::Sparse(c) => (c.data.len() / n.max(1)).max(1),
            HostAffinity::Dense(_) => n,
        };
        par_blocks(y, work_per_row, |row0, ychunk| {
            for (r, out) in ychunk.iter_mut().enumerate() {
                let i = row0 + r;
                let acc = match m {
                    HostAffinity::Sparse(c) => {
                        let (lo, hi) = (c.indptr[i] as usize, c.indptr[i + 1] as usize);
                        let mut a = 0.0;
                        for t in lo..hi {
                            a += c.data[t] * x[c.indices[t] as usize];
                        }
                        a
                    }
                    HostAffinity::Dense(d) => {
                        let row = &d[i * n..(i + 1) * n];
                        let mut a = 0.0;
                        for (j, &v) in row.iter().enumerate() {
                            a += v * x[j];
                        }
                        a
                    }
                };
                *out = acc + x[i];
            }
        });
    }

    /// Per-node "has at least one neighbour" flag, i.e. `deg > 0`.
    ///
    /// [`null_space_basis`] needs it to skip isolated nodes, whose Laplacian
    /// eigenvalue is 1 rather than 0.
    pub fn degree_positive(&self) -> Vec<bool> {
        // `dd` is `sqrt(deg)` except on an isolated node, where the scipy guard
        // substitutes exactly 1. A genuine `deg == 1` is indistinguishable from
        // the guard by `dd` alone, so recompute from the stored (already scaled)
        // operator instead: a node with no neighbours has an empty row.
        match &self.m {
            HostAffinity::Sparse(c) => (0..self.n)
                .map(|i| {
                    let (lo, hi) = (c.indptr[i] as usize, c.indptr[i + 1] as usize);
                    (lo..hi).any(|t| c.indices[t] as usize != i && c.data[t] != 0.0)
                })
                .collect(),
            HostAffinity::Dense(d) => (0..self.n)
                .map(|i| (0..self.n).any(|j| j != i && d[i * self.n + j] != 0.0))
                .collect(),
        }
    }

    /// Whether the operator's affinity is stored DENSE (a kernel or precomputed
    /// affinity) rather than as a sparse graph.
    ///
    /// This is a solver hint, not just a storage detail. A kernel affinity is a
    /// similarity that varies smoothly with distance, so its normalized
    /// Laplacian has a heavily CLUSTERED low spectrum — in the limit of a
    /// near-constant affinity (an rbf kernel on high-dimensional uniform data,
    /// where all pairwise distances are nearly equal) it approaches the complete
    /// graph, whose non-trivial eigenvalues are all identical. A kNN
    /// connectivity graph has no such structure. The eigensolver widens its
    /// Krylov block accordingly; see [`lanczos_largest_sized`].
    pub fn is_dense(&self) -> bool {
        matches!(self.m, HostAffinity::Dense(_))
    }

    /// The DENSE normalized Laplacian `L = I − M`, row-major, with the diagonal
    /// forced to `1` (sklearn `_set_diag`). Only reachable on the `n ≤ DENSE_N`
    /// route, where materializing `n²` is what the dense eigensolver wants
    /// anyway.
    ///
    /// `pub` so the solver-equivalence test in
    /// `crates/mlrs-algos/tests/spectral_embedding_test.rs` can hand the SAME
    /// Laplacian to the dense `sym_eig` and to [`lanczos_largest`] and compare
    /// the two eigenbases directly.
    pub fn dense_laplacian(&self) -> Vec<f64> {
        let n = self.n;
        let mut l = vec![0.0f64; n * n];
        match &self.m {
            HostAffinity::Sparse(c) => {
                for i in 0..n {
                    let (lo, hi) = (c.indptr[i] as usize, c.indptr[i + 1] as usize);
                    for t in lo..hi {
                        l[i * n + c.indices[t] as usize] = -c.data[t];
                    }
                }
            }
            HostAffinity::Dense(d) => {
                for (o, &v) in l.iter_mut().zip(d.iter()) {
                    *o = -v;
                }
            }
        }
        for i in 0..n {
            l[i * n + i] = 1.0;
        }
        l
    }
}

// ---------------------------------------------------------------------------
// Parallel helpers
// ---------------------------------------------------------------------------

/// Split `out` into contiguous blocks and run `f(offset, block)` on each in its
/// own scoped thread. Blocks are disjoint, so splitting never changes a value —
/// only which thread computes it.
///
/// `work_per_item` is the scalar work one element of `out` costs (the row's
/// nonzero count for a sparse matvec, `n` for a dense one, the basis width for a
/// projection). It gates the split against [`MIN_PAR_WORK`]: a pass too small to
/// pay for its own threads runs INLINE, and a pass that is worth splitting is
/// still never cut finer than `MIN_PAR_WORK` per thread. Passing the work
/// explicitly is what lets one helper serve both the `O(n·d)` affinity build and
/// the `O(nnz)` Lanczos matvec without either one guessing at the other's shape.
fn par_blocks<T: Send, F>(out: &mut [T], work_per_item: usize, f: F)
where
    F: Fn(usize, &mut [T]) + Sync,
{
    let n = out.len();
    let units = capability::cpu_launch_units().max(1) as usize;
    let w = work_per_item.max(1);
    // Whole-pass work below the floor: the spawn would cost more than the pass.
    if units == 1 || n == 0 || n.saturating_mul(w) < MIN_PAR_WORK {
        f(0, out);
        return;
    }
    let min_items = MIN_PAR_WORK.div_ceil(w).max(1);
    let per = n.div_ceil(units).max(min_items);
    if per >= n {
        f(0, out);
        return;
    }
    let fref = &f;
    std::thread::scope(|scope| {
        let mut rest: &mut [T] = out;
        let mut off = 0usize;
        while off < n {
            let take = per.min(n - off);
            let (blk, tail) = rest.split_at_mut(take);
            rest = tail;
            let start = off;
            scope.spawn(move || fref(start, blk));
            off += take;
        }
    });
}

/// Split a row-major `rows × row_width` output into contiguous ROW blocks and
/// run `f(row0, block)` on each in its own scoped thread. The affinity builders
/// use this; the cut is always on a row boundary, so a block is a whole number
/// of rows. `work_per_row` gates the split exactly as in [`par_blocks`].
pub fn par_row_blocks<T: Send, F>(out: &mut [T], row_width: usize, work_per_row: usize, f: F)
where
    F: Fn(usize, &mut [T]) + Sync,
{
    if row_width == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / row_width;
    let units = capability::cpu_launch_units().max(1) as usize;
    let w = work_per_row.max(1);
    if units == 1 || rows == 0 || rows.saturating_mul(w) < MIN_PAR_WORK {
        f(0, out);
        return;
    }
    let min_rows = MIN_PAR_WORK.div_ceil(w).max(1);
    let per = rows.div_ceil(units).max(min_rows);
    if per >= rows {
        f(0, out);
        return;
    }
    let fref = &f;
    std::thread::scope(|scope| {
        let mut rest: &mut [T] = out;
        let mut row0 = 0usize;
        while row0 < rows {
            let take = per.min(rows - row0);
            let (blk, tail) = rest.split_at_mut(take * row_width);
            rest = tail;
            let start = row0;
            scope.spawn(move || fref(start, blk));
            row0 += take;
        }
    });
}

/// `c[j] = ⟨V[j], w⟩` for `j < ncols`, with `V` stored column-major
/// (`V[j·n .. (j+1)·n]`). Parallel over `j`: each thread owns a contiguous slab
/// of `V` and a disjoint slice of `c`.
fn basis_dots(v: &[f64], ncols: usize, n: usize, w: &[f64], c: &mut [f64]) {
    par_blocks(&mut c[..ncols], n, |j0, cblk| {
        for (t, out) in cblk.iter_mut().enumerate() {
            let j = j0 + t;
            *out = dot(&v[j * n..(j + 1) * n], w);
        }
    });
}

/// `w -= Σ_j c[j]·V[j]`. Parallel over disjoint row blocks of `w`.
fn basis_axpy(v: &[f64], ncols: usize, n: usize, c: &[f64], w: &mut [f64]) {
    par_blocks(w, ncols, |i0, wblk| {
        let len = wblk.len();
        for (j, &cj) in c.iter().enumerate().take(ncols) {
            if cj == 0.0 {
                continue;
            }
            let col = &v[j * n + i0..j * n + i0 + len];
            for (o, &a) in wblk.iter_mut().zip(col.iter()) {
                *o -= cj * a;
            }
        }
    });
}

/// `out[c] = Σ_r V[r]·Y[r, c]` for `c < k` — the Lanczos restart projection and
/// the final Ritz-vector assembly. `V` and `out` are column-major over `n`; `y`
/// is `m × k` column-major (`y[c·m + r]`).
///
/// One parallel pass PER OUTPUT COLUMN: the row-block split has to be over a
/// single column's `n` entries, because a flat split of the `k·n` output would
/// hand a thread a boundary in the middle of a column.
fn project(v: &[f64], m: usize, n: usize, y: &[f64], k: usize, out: &mut [f64]) {
    for c in 0..k {
        let ocol = &mut out[c * n..(c + 1) * n];
        let ycol = &y[c * m..(c + 1) * m];
        par_blocks(ocol, m, |i0, blk| {
            for (t, o) in blk.iter_mut().enumerate() {
                let i = i0 + t;
                let mut acc = 0.0;
                for (r, &yr) in ycol.iter().enumerate() {
                    acc += v[r * n + i] * yr;
                }
                *o = acc;
            }
        });
    }
}

/// `⟨a, b⟩` over a slice pair (called on one column at a time, inside a block).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

// ---------------------------------------------------------------------------
// Thick-restart Lanczos
// ---------------------------------------------------------------------------

/// A deterministic `SplitMix64`-seeded uniform starting vector.
///
/// sklearn seeds ARPACK with `random_state.uniform(-1, 1, n)`. The converged
/// invariant subspace does not depend on that vector — sklearn's own behavior
/// bears this out, seeds 0 and 99 differing by 2e-16 — so matching its exact bit
/// stream buys nothing. What matters is that the start is GENERIC (no needed
/// eigenvector is orthogonal to it) and REPRODUCIBLE across runs.
fn seeded_start(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Uniform on [-1, 1).
        v.push(((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0);
    }
    v
}

/// Krylov depth: the basis holds this many BLOCKS of `b` vectors.
///
/// `MLRS_SPECTRAL_DEPTH` overrides it. Depth and width trade against each other
/// — both enlarge the basis, and the basis is what costs `O(m·nnz)` matvecs and
/// `O(m²·n)` reorthogonalization per restart — so the pair was swept together on
/// the hard case (a two-moons kNN graph, whose wanted/unwanted eigenvalue gap is
/// ~8e-5) rather than picked independently.
const BLOCK_KRYLOV_DEPTH: usize = 8;

/// Depth the iteration STARTS at, before any growth.
///
/// Most spectral embeddings are easy — a connected graph with well-separated
/// low eigenvalues converges at this depth in one or two restarts — and paying
/// the full [`BLOCK_KRYLOV_DEPTH`] basis up front would make the easy case
/// carry the hard case's cost. MEASURED on a 2000x16 uniform cloud: depth 3
/// converges in 237 ms against 160 ms... which is to say depth alone is not
/// monotone, and the growth schedule below is what actually keeps both ends
/// cheap.
const BLOCK_KRYLOV_INITIAL_DEPTH: usize = 3;

/// Residual a STALLED iteration must already have reached before its result is
/// accepted.
///
/// The stall exit exists so a run that has hit its round-off floor stops instead
/// of spending its whole restart budget re-deriving the same vectors. But
/// "stopped improving" and "correct" are different claims: at a basis too small
/// for the problem the residual also improves by under 10% per restart, and
/// MEASURED on a two-moons kNN graph at depth 3 that exit returned vectors
/// almost ORTHOGONAL to the true invariant subspace (9e-01 out-of-subspace
/// error) — fast, and completely wrong. A stalled run is therefore only
/// believed once its residual is small in absolute terms; otherwise the basis
/// grows and it tries again.
const LANCZOS_STALL_ACCEPT: f64 = 1e-9;

/// Guard columns added to the block on top of `nev`; see [`lanczos_largest`].
/// `MLRS_SPECTRAL_BLOCK` overrides the resulting width.
const BLOCK_GUARD: usize = 4;

/// The Krylov block width for `nev` wanted eigenpairs, given the largest
/// eigenvalue `multiplicity` the caller expects.
///
/// A block of width `w` resolves multiplicity up to `w`, so the width must cover
/// the expected multiplicity; the extra columns beyond that are GUARD vectors,
/// which move the convergence rate off the gap to eigenvalue `nev+1` and onto
/// the much larger gap to eigenvalue `w+1`.
///
/// A caller that knows no eigenvalue is tied (`multiplicity = 1`) gets a lean
/// block, because guards it does not need are pure cost: MEASURED on a 2000x16
/// connected graph, width `nev` converges in 105 ms against 279 ms for the wide
/// block, to the same answer.
fn block_width(nev: usize, multiplicity: usize) -> usize {
    if let Some(v) = mlrs_backend::abflag::var("MLRS_SPECTRAL_BLOCK")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 1)
    {
        return v.max(nev.min(multiplicity.max(1)));
    }
    if multiplicity <= 1 {
        // No eigenvalue is tied by construction: `nev` columns suffice for
        // multiplicity, and the stall growth covers a near-tie if one shows up.
        nev + 1
    } else {
        (2 * nev + BLOCK_GUARD).max(multiplicity)
    }
    .max(nev.min(multiplicity.max(1)))
}

/// The Krylov depth (blocks of [`block_width`] held in the basis).
fn block_depth() -> usize {
    mlrs_backend::abflag::var("MLRS_SPECTRAL_DEPTH")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 2)
        .unwrap_or(BLOCK_KRYLOV_DEPTH)
}

/// Hard ceiling on the basis width, bounding both the `O(m²·n)`
/// reorthogonalization and the `m·n` memory regardless of `n_components`.
const MAX_BASIS: usize = 192;

/// Orthonormalize column `c` of the column-major basis `v` against columns
/// `0..c`, twice, and renormalize.
///
/// Returns `false` if the column collapsed into the existing span (its norm fell
/// below the round-off floor), which the caller answers by substituting a fresh
/// random direction — the basis must stay full rank for the Rayleigh–Ritz
/// projection to be meaningful.
fn orthonormalize_against(
    v: &mut [f64],
    c: usize,
    n: usize,
    coef: &mut [f64],
    deflate: &[f64],
) -> bool {
    // "Twice is enough" (Kahan): one pass leaves a component of size
    // `O(ε·κ)` behind, two passes take it to round-off, and a third buys nothing.
    let nd = if n == 0 { 0 } else { deflate.len() / n };
    for _ in 0..2 {
        if nd > 0 {
            // Deflation: the caller already holds these directions exactly (the
            // analytic null space), so the Krylov space must be built in their
            // ORTHOGONAL COMPLEMENT or the iteration wastes its budget
            // rediscovering them.
            let col = &mut v[c * n..(c + 1) * n];
            basis_dots(deflate, nd, n, col, coef);
            basis_axpy(deflate, nd, n, &coef[..nd], col);
        }
        if c > 0 {
            let (head, tail) = v.split_at_mut(c * n);
            basis_dots(head, c, n, &tail[..n], coef);
            basis_axpy(head, c, n, &coef[..c], &mut tail[..n]);
        }
    }
    let col = &mut v[c * n..(c + 1) * n];
    let nrm = dot(col, col).sqrt();
    if nrm <= 1e-140 {
        return false;
    }
    let inv = 1.0 / nrm;
    for x in col.iter_mut() {
        *x *= inv;
    }
    true
}

/// Write a fresh deterministic random direction into column `c` and
/// orthonormalize it against `0..c`.
fn refill_random(v: &mut [f64], c: usize, n: usize, seed: u64, coef: &mut [f64], deflate: &[f64]) {
    for attempt in 0..8u64 {
        let r = seeded_start(n, seed ^ ((c as u64 + 1) << 20) ^ (attempt << 40));
        v[c * n..(c + 1) * n].copy_from_slice(&r);
        if orthonormalize_against(v, c, n, coef, deflate) {
            return;
        }
    }
    // The space is genuinely exhausted (`c >= n`); leave the zero column, which
    // contributes nothing to the projection.
}

/// The `nev` LARGEST eigenvectors of `S = op`, as `nev` columns of length `n`
/// (`x[c·n .. (c+1)·n]`), ordered by DESCENDING eigenvalue.
///
/// ## Why this is a BLOCK method
/// A Krylov space grown from a SINGLE start vector `v₀` contains exactly one
/// direction out of any eigenspace — the projection of `v₀` onto it. So a
/// repeated eigenvalue is recovered ONCE, and the remaining requested pairs get
/// filled from the next DISTINCT eigenvalues. Those are genuine eigenpairs with
/// genuinely tiny residuals, so a residual-based convergence test reports
/// success on an answer that is missing `multiplicity − 1` of the vectors asked
/// for. Full reorthogonalization makes this worse, not better: the round-off
/// "ghost" mechanism that would otherwise stumble onto the extra copies is
/// exactly what it suppresses.
///
/// That failure is not exotic here — it is the main case. The normalized
/// Laplacian of a graph with `c` connected components has eigenvalue `0` with
/// multiplicity exactly `c` (equivalently `S = 2I − L` has `2` with multiplicity
/// `c`), and a kNN graph over well-separated clusters is disconnected BY
/// CONSTRUCTION. That is sklearn's "Graph is not fully connected" case, which
/// ARPACK handles correctly, so a single-vector iteration here would return a
/// visibly different embedding — and, downstream in `SpectralClustering`, wrong
/// labels.
///
/// Starting from a BLOCK of `b = nev` orthonormal random vectors fixes it: the
/// block Krylov space contains `min(b, multiplicity)` independent directions
/// from each eigenspace, and since at most `nev` copies of any eigenvalue can be
/// among the `nev` largest, `b = nev` is always enough.
///
/// ## Shape
/// Basis `V` is `m = b · BLOCK_KRYLOV_DEPTH` columns, grown by
/// `V[j+b] ← orthonormalize(S·V[j])` with FULL reorthogonalization. The
/// projected matrix `H = VᵀSV` is assembled from the reorthogonalization
/// coefficients themselves (each is a column of `H`), so it costs no extra
/// matvecs; both `(i,j)` and `(j,i)` are written, which fills the entries a
/// single pass would miss when a breakdown forces a random refill and the band
/// structure no longer holds. Rayleigh–Ritz on `H` gives the Ritz pairs, the
/// residual `‖S x − θx‖` is measured DIRECTLY on the `nev` wanted vectors (no
/// band-structure shortcut that a refill could invalidate), and a restart
/// re-seeds the block with the best `b` Ritz vectors.
///
/// `pub` so the solver-equivalence tests in
/// `crates/mlrs-algos/tests/spectral_embedding_test.rs` can call it DIRECTLY,
/// bypassing the `DENSE_N` routing in [`smallest_laplacian_vectors`] and
/// checking it against the dense `sym_eig` on the same operator. Production
/// callers should go through [`smallest_laplacian_vectors`], which picks the
/// solver.
pub fn lanczos_largest(op: &NormAdj, nev: usize, seed: u64) -> Vec<f64> {
    lanczos_largest_sized(op, nev, seed, nev)
}

/// [`lanczos_largest`] with an explicit MULTIPLICITY BUDGET.
///
/// `multiplicity` is the largest number of copies of a single eigenvalue the
/// caller expects, and it sets the floor on the Krylov block width. [`run`]
/// knows this exactly: a graph with `c` connected components has Laplacian
/// eigenvalue `0` with multiplicity exactly `c`, and it has already counted
/// the components to reproduce sklearn's connectivity warning. Passing that
/// count means the ORDINARY connected graph — where no eigenvalue is tied by
/// construction — pays a lean block, while the disconnected graph gets the
/// wide one it genuinely needs. Growth on stall still covers a connected
/// graph that merely has a near-tie.
pub fn lanczos_largest_sized(
    op: &NormAdj,
    nev: usize,
    seed: u64,
    multiplicity: usize,
) -> Vec<f64> {
    lanczos_largest_deflated(op, nev, seed, multiplicity, &[])
}

/// [`lanczos_largest_sized`] with an ORTHONORMAL deflation basis.
///
/// Every Krylov vector is orthogonalized against `deflate` as well as against
/// the running basis, so the iteration searches only the orthogonal complement
/// of those directions. [`run`] uses it to hand over the Laplacian's null space,
/// which it knows in closed form (see [`null_space_basis`]) — the iteration then
/// spends its whole budget on the eigenvalues it actually has to compute
/// instead of re-deriving an exactly repeated zero.
pub fn lanczos_largest_deflated(
    op: &NormAdj,
    nev: usize,
    seed: u64,
    multiplicity: usize,
    deflate: &[f64],
) -> Vec<f64> {
    let n = op.n;
    let nev = nev.min(n).max(1);
    // Block width. Two jobs, and the wider of the two requirements wins:
    //
    //  * MULTIPLICITY — the block must be at least `nev` wide, since at most
    //    `nev` copies of any single eigenvalue can appear among the `nev`
    //    largest, and a block of width `w` resolves multiplicity up to `w`.
    //  * CONVERGENCE — the restart re-seeds from the best `b` Ritz vectors, so
    //    the `b − nev` extra columns are GUARD vectors. Without them the
    //    convergence rate is governed by the gap between wanted eigenvalue
    //    `nev` and unwanted `nev+1`, which for a spectral embedding is routinely
    //    ~1e-4 (a two-moons kNN graph has its 3rd and 4th Laplacian eigenvalues
    //    at 0.001065 and 0.001147). Oversampling moves the rate onto the far
    //    larger gap to eigenvalue `b+1` instead.
    // The basis GROWS on stall, in BOTH dimensions, and the two are not
    // interchangeable:
    //
    //  * WIDTH buys multiplicity and oversampling. MEASURED on a two-circles kNN
    //    graph (whose 3rd and 4th Laplacian eigenvalues sit at 0.00128 and
    //    0.00143), width 3 leaves 9e-01 of out-of-subspace error while width 10
    //    reaches 4e-11 — and no amount of depth rescues width 3.
    //  * DEPTH buys polynomial degree, which is what separates eigenvalues that
    //    are merely close rather than tied.
    //
    // Width is therefore fixed at the value the sweep showed sufficient, and
    // only DEPTH grows on stall. Starting narrow in WIDTH was tried and
    // rejected: an easy 2000x16 graph does converge at width `nev` (105 ms
    // against 279 ms for the identical answer), but a run that starts narrow and
    // widens on stall restarts from a block the narrow basis already degraded,
    // and it ended at 3e-04 on two-moons where the fixed width reaches 1e-11.
    // Paying the wide block always is the price of not being quietly wrong on a
    // disconnected graph.
    // Start at the width the caller's multiplicity budget demands, and allow
    // growth up to the wide value on stall. For a DISCONNECTED graph those are
    // the same number, so the exactly-tied case never starts narrow — which is
    // what makes growing the width safe here: an earlier version that started
    // narrow unconditionally restarted from a block the narrow basis had already
    // degraded, and ended at 3e-04 on two-moons.
    let b_start = block_width(nev, multiplicity).min(n);
    let b_cap = block_width(nev, nev.max(2)).min(n).max(b_start);
    let depth_cap = block_depth();
    let mut b = b_start;
    let mut depth = BLOCK_KRYLOV_INITIAL_DEPTH.min(depth_cap);
    let m_max = (b_cap * depth_cap).clamp(b_cap + 1, n).min(MAX_BASIS.max(b_cap + 1));
    let mut m = (b * depth).clamp(b + 1, n).min(m_max);

    let mut v = vec![0.0f64; (m_max + b_cap) * n];
    let mut work = vec![0.0f64; n];
    let mut coef = vec![0.0f64; (m_max + b_cap + 1).max(deflate.len() / n.max(1) + 1)];
    let mut out = vec![0.0f64; nev * n];
    // Best-so-far iterate and its residual, for the stagnation exit below.
    let mut best = vec![0.0f64; nev * n];
    let mut best_resid = f64::INFINITY;
    let mut stall = 0usize;

    // Initial block: `b` deterministic random directions, orthonormalized.
    for c in 0..b {
        let r = seeded_start(n, seed ^ ((c as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        v[c * n..(c + 1) * n].copy_from_slice(&r);
        if !orthonormalize_against(&mut v, c, n, &mut coef, deflate) {
            refill_random(&mut v, c, n, seed ^ 0xDEAD_BEEF, &mut coef, deflate);
        }
    }

    for _restart in 0..LANCZOS_MAX_RESTARTS {
        // Pin the basis size for THIS iteration. `m` may be grown by the stall
        // branch below, and every buffer built here — `h`, the Ritz matrix `y`,
        // the restart projection — is indexed by it, so reading the field
        // directly after a growth would index an `m_old x m_old` matrix with the
        // new stride.
        let mm = m;
        let mut h = vec![0.0f64; mm * mm];
        let mut ncols = b;

        for j in 0..mm {
            {
                let (head, tail) = v.split_at(j * n);
                let _ = head;
                op.apply(&tail[..n], &mut work);
            }
            // `coef[i] = ⟨V[i], S·V[j]⟩` IS column `j` of `H`. Writing the
            // symmetric partner too is what keeps `H` complete when a random
            // refill has broken the band structure.
            basis_dots(&v, ncols, n, &work, &mut coef);
            for i in 0..ncols.min(mm) {
                h[i * mm + j] = coef[i];
                h[j * mm + i] = coef[i];
            }
            // Extend the basis with the orthogonal remainder, while there is
            // still room for it.
            if ncols < mm + b {
                basis_axpy(&v, ncols, n, &coef[..ncols], &mut work);
                v[ncols * n..(ncols + 1) * n].copy_from_slice(&work);
                if !orthonormalize_against(&mut v, ncols, n, &mut coef, deflate) {
                    // Invariant subspace reached: the Krylov space is exhausted
                    // before the basis is full. A fresh random direction keeps
                    // the projection full rank so the remaining Ritz pairs are
                    // still well defined.
                    refill_random(&mut v, ncols, n, seed ^ 0x00C0_FFEE, &mut coef, deflate);
                }
                ncols += 1;
            }
        }

        // Rayleigh-Ritz. `sym_eig` is DESCENDING with eigenvectors in columns,
        // so Ritz pair `r` (largest first) is simply column `r`.
        let (_theta, y) = sym_eig(&h, mm);
        let mut ycols = vec![0.0f64; mm * nev];
        for r in 0..nev {
            for row in 0..mm {
                ycols[r * mm + row] = y[row * mm + r];
            }
        }
        project(&v, mm, n, &ycols, nev, &mut out);

        // Residual measured DIRECTLY on the wanted vectors: `‖S x − θ x‖` with
        // `θ` the Rayleigh quotient. This costs `nev` matvecs and, unlike the
        // `β·y[m-1]` shortcut of a single-vector tridiagonal iteration, stays
        // valid when a refill has disturbed the band structure.
        let mut worst = 0.0f64;
        for r in 0..nev {
            let x = &out[r * n..(r + 1) * n];
            op.apply(x, &mut work);
            let theta = dot(x, &work);
            let mut resid = 0.0;
            for (w, xi) in work.iter().zip(x.iter()) {
                let e = w - theta * xi;
                resid += e * e;
            }
            worst = worst.max(resid.sqrt());
        }
        // Whether this restart made MEANINGFUL progress must be decided against
        // the PREVIOUS best, so read it before `best_resid` is updated.
        let improved = worst < best_resid * 0.9;
        // Keep the best iterate, not merely the last: a restart can overshoot
        // once it is down at the round-off floor.
        if worst < best_resid {
            best_resid = worst;
            best.copy_from_slice(&out);
        }
        if worst <= LANCZOS_TOL * 2.0 || mm >= n {
            return out;
        }
        // STAGNATION. The target above is a round-off-floor tolerance, and on a
        // problem whose wanted and unwanted eigenvalues are separated by ~1e-4 —
        // the ordinary case for a spectral embedding, not a pathology — the
        // attainable residual bottoms out just short of it. Without this check
        // the iteration spent its entire restart budget re-deriving an answer it
        // already had: MEASURED at 253 ms for a 600-sample two-moons graph, all
        // of it after the result had stopped changing. Stopping when three
        // consecutive restarts fail to improve the residual by even 10% returns
        // the same vectors far sooner, and the accuracy that matters is bounded
        // by `residual / eigenvalue-gap`, which is already ~1e-8 here.
        if improved {
            stall = 0;
        } else {
            stall += 1;
            if stall >= 2 {
                if b < b_cap {
                    // Width first: it is the dimension that separates a tie or a
                    // near-tie, and depth does not substitute for it.
                    b = (b * 2).clamp(b + 1, b_cap);
                    m = (b * depth).clamp(b + 1, n).min(m_max);
                    stall = 0;
                } else if depth < depth_cap && m < m_max {
                    // Widest block allowed: buy polynomial degree instead.
                    depth = (depth * 2).clamp(depth + 1, depth_cap);
                    m = (b * depth).clamp(b + 1, n).min(m_max);
                    stall = 0;
                } else if best_resid <= LANCZOS_STALL_ACCEPT {
                    // At the largest basis allowed and no longer improving, with
                    // a residual small enough to trust. Anything larger is spent
                    // re-deriving vectors that have stopped changing.
                    //
                    // There is deliberately NO escape hatch for a stalled run
                    // whose residual is still LARGE: an earlier version bailed
                    // out after a few stalled restarts regardless, and on the
                    // two-circles graph that returned vectors almost orthogonal
                    // to the true invariant subspace. Such a run now spends its
                    // full restart budget instead — slow on a genuinely hard
                    // spectrum, but never quietly wrong.
                    return best;
                }
            }
        }

        // Restart: re-seed the block with the best `b` Ritz vectors. Any that
        // collapse (a converged pair repeated) are replaced by fresh random
        // directions, which is also what lets a still-unresolved multiplicity
        // pick up another copy on the next pass.
        // The re-seed block cannot be wider than the basis it is projected
        // out of.
        let bb = b.min(mm);
        let mut ycols_b = vec![0.0f64; mm * bb];
        for r in 0..bb {
            for row in 0..mm {
                ycols_b[r * mm + row] = y[row * mm + r];
            }
        }
        let mut block = vec![0.0f64; bb * n];
        project(&v, mm, n, &ycols_b, bb, &mut block);
        v[..bb * n].copy_from_slice(&block);
        for c in 0..bb {
            if !orthonormalize_against(&mut v, c, n, &mut coef, deflate) {
                refill_random(&mut v, c, n, seed ^ (0xBEEF << c.min(40)), &mut coef, deflate);
            }
        }
    }

    // Restart budget exhausted — return the best iterate seen rather than failing
    // the fit (ARPACK likewise returns whatever it converged to).
    best
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// The order below which [`smallest_laplacian_vectors`] takes the dense solver.
///
/// `MLRS_SPECTRAL_DENSE_N` overrides it so the crossover can be swept on the
/// target host — the two arms compute the same eigenspace, so this is a pure
/// perf knob and forcing either one is always numerically legitimate (which is
/// what lets `lanczos_matches_dense` drive both through one entry point).
fn dense_threshold() -> usize {
    mlrs_backend::abflag::var("MLRS_SPECTRAL_DENSE_N")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DENSE_N)
}

/// The `nev` smallest eigenvectors of the normalized Laplacian of `op`, as
/// `nev` ROWS of length `n` ordered ASCENDING by eigenvalue — the array sklearn
/// calls `embedding` before the `/dd` recovery.
///
/// Routes to the dense solver for small orders and for an `nev` that is a large
/// fraction of `n` (where Lanczos has no advantage and its restart heuristics
/// are least reliable), and to the restarted block Krylov iteration otherwise.
pub fn smallest_laplacian_vectors(op: &NormAdj, nev: usize, seed: u64) -> Vec<f64> {
    smallest_laplacian_vectors_hinted(op, nev, seed, false)
}

/// [`smallest_laplacian_vectors`] with an explicit `force_dense` override.
///
/// The override exists for tests and for a caller that wants the direct dense
/// spectrum regardless of size. It is NOT how the degenerate case is handled:
/// an earlier version of this module routed every disconnected graph here,
/// paying `O(n³)`, because the single-vector Krylov iteration it then had could
/// not resolve the repeated zero eigenvalue. Both halves of that problem are now
/// solved where they belong — [`lanczos_largest_deflated`] starts from a BLOCK
/// wide enough for the multiplicity, and [`run`] hands it the null space in
/// closed form via [`null_space_basis`] — so `run` never forces the dense arm.
pub fn smallest_laplacian_vectors_hinted(
    op: &NormAdj,
    nev: usize,
    seed: u64,
    force_dense: bool,
) -> Vec<f64> {
    smallest_laplacian_vectors_for(op, nev, seed, force_dense, nev)
}

/// [`smallest_laplacian_vectors_hinted`] with the multiplicity budget described
/// on [`lanczos_largest_sized`].
pub fn smallest_laplacian_vectors_for(
    op: &NormAdj,
    nev: usize,
    seed: u64,
    force_dense: bool,
    multiplicity: usize,
) -> Vec<f64> {
    let n = op.n;
    let nev = nev.min(n);
    let mut out = vec![0.0f64; nev * n];
    if nev == 0 || n == 0 {
        return out;
    }

    if force_dense || n <= dense_threshold() || nev * 4 >= n {
        let l = op.dense_laplacian();
        // `sym_eig` is DESCENDING with eigenvectors in columns, so the `r`-th
        // SMALLEST is column `n - 1 - r`.
        let (_w, v) = sym_eig(&l, n);
        for r in 0..nev {
            let c = n - 1 - r;
            for i in 0..n {
                out[r * n + i] = v[i * n + c];
            }
        }
        return out;
    }

    // The largest eigenvectors of `S` ARE the smallest of `L`, and descending in
    // `S` is ascending in `L` — no reordering needed, only a transpose from the
    // solver's column layout into the row layout the recovery wants.
    let x = lanczos_largest_sized(op, nev, seed, multiplicity);
    let got = x.len() / n;
    for r in 0..nev.min(got) {
        out[r * n..(r + 1) * n].copy_from_slice(&x[r * n..(r + 1) * n]);
    }
    out
}

/// The sklearn `_spectral_embedding` post-solve recovery applied to the `m × n`
/// eigenvector rows: `/dd` diffusion scaling → deterministic sign flip →
/// drop-first → transpose into a row-major `n × n_components` matrix.
///
/// The ORDER is load-bearing (sklearn applies the sign flip AFTER the `/dd`
/// division, so the argmax is taken over the `D^-1/2`-scaled vector) and the
/// argmax tie-break is lowest-index, matching `np.argmax`.
///
/// `diffusion_recover = false` skips BOTH the `/dd` and the sign flip — the
/// umap-learn `spectral_layout` convention (see [`crate::cluster::spectral`]).
pub fn recover_rows(
    mut emb: Vec<f64>,
    dd: &[f64],
    n: usize,
    n_components: usize,
    drop_first: bool,
    diffusion_recover: bool,
) -> Vec<f64> {
    let m = if n == 0 { 0 } else { emb.len() / n };
    if diffusion_recover {
        for r in 0..m {
            for i in 0..n {
                emb[r * n + i] /= dd[i];
            }
        }
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
            if emb[r * n + max_idx] < 0.0 {
                for v in emb[r * n..(r + 1) * n].iter_mut() {
                    *v = -*v;
                }
            }
        }
    }
    let row_offset = usize::from(drop_first);
    let mut out = vec![0.0f64; n * n_components];
    for c in 0..n_components {
        let r = c + row_offset;
        if r >= m {
            continue;
        }
        for i in 0..n {
            out[i * n_components + c] = emb[r * n + i];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The estimator-facing entry point
// ---------------------------------------------------------------------------

/// Everything the spectral pipeline needs from an estimator, already resolved
/// down to plain scalars. Both `SpectralEmbedding` and `SpectralClustering`
/// build one of these, so the two cannot drift on the shared stages.
#[derive(Debug, Clone)]
pub struct SpectralPlan<'a> {
    /// Estimator name, for typed errors.
    pub estimator: &'static str,
    /// The affinity string (`"nearest_neighbors"`, `"rbf"`, …).
    pub affinity: &'a str,
    /// `gamma`, or `None` for sklearn's `1/n_features` default.
    pub gamma: Option<f64>,
    /// `pairwise_kernels` polynomial degree (`SpectralClustering` only).
    pub degree: f64,
    /// `pairwise_kernels` independent term (`SpectralClustering` only).
    pub coef0: f64,
    /// `n_neighbors`, or `None` for sklearn `SpectralEmbedding`'s
    /// `max(n_samples / 10, 1)` default.
    pub n_neighbors: Option<usize>,
    /// Number of embedding columns to KEEP.
    pub n_components: usize,
    /// Drop the trivial ≈0 eigenvector (`true` for `SpectralEmbedding`, `false`
    /// for `SpectralClustering`).
    pub drop_first: bool,
    /// Deterministic Lanczos start seed (from `random_state`).
    pub seed: u64,
    /// Whether the non-`rbf` `pairwise_kernels` family is in scope.
    /// `SpectralEmbedding` calls `rbf_kernel` directly and so sets this `false`.
    pub allow_kernels: bool,
}

/// Everything a fitted spectral estimator needs to expose.
pub struct SpectralFit {
    /// Row-major `n × n_components` embedding (`embedding_`, or the KMeans
    /// `maps` for `SpectralClustering`).
    pub embedding: Vec<f64>,
    /// `affinity_matrix_`, in whichever layout its builder produced.
    pub affinity: HostAffinity,
    /// The RESOLVED neighbor count (`n_neighbors_`); `None` unless a kNN
    /// affinity was used.
    pub n_neighbors_used: Option<usize>,
    /// The RESOLVED kernel coefficient (`gamma_`); `None` unless a kernel
    /// affinity was used.
    pub gamma_used: Option<f64>,
    /// Number of connected components of the affinity graph. sklearn warns when
    /// this exceeds 1 and changes nothing else; the caller decides whether to
    /// surface it.
    pub n_graph_components: usize,
}

/// Run the whole sklearn `_spectral_embedding` pipeline on the host: affinity →
/// normalized Laplacian → smallest `n_components (+1)` eigenvectors → `/dd`
/// recovery → sign flip → drop-first → transpose.
///
/// `x` is the row-major `n × d` design, or the `n × n` matrix for the
/// `precomputed` / `precomputed_nearest_neighbors` affinities.
pub fn run(
    plan: &SpectralPlan<'_>,
    x: &[f64],
    n: usize,
    d: usize,
) -> Result<SpectralFit, AlgoError> {
    if plan.n_components < 1 {
        return Err(AlgoError::InvalidNComponents {
            estimator: plan.estimator,
            requested: plan.n_components,
            max: n.saturating_sub(usize::from(plan.drop_first)),
        });
    }
    // The `drop_first` estimator needs one EXTRA eigenvector (the trivial ≈0 one
    // it then discards), so it needs a strictly larger `n_samples`.
    let nev = plan.n_components + usize::from(plan.drop_first);
    if nev > n {
        return Err(AlgoError::InvalidNComponents {
            estimator: plan.estimator,
            requested: plan.n_components,
            max: n.saturating_sub(usize::from(plan.drop_first)),
        });
    }

    // sklearn `SpectralEmbedding`: `n_neighbors_ = n_neighbors if not None else
    // max(int(n_samples / 10), 1)` — TRUNCATING division, floored at 1.
    // `SpectralClustering` passes an int default of 10 and never reaches the
    // `None` branch.
    let k_resolved = plan.n_neighbors.unwrap_or_else(|| (n / 10).max(1));
    if k_resolved < 1 {
        return Err(AlgoError::InvalidK {
            estimator: plan.estimator,
            k: k_resolved,
            n_samples: n,
        });
    }
    // WR-03 (retained): sklearn's `NearestNeighbors` caps rather than errors
    // when the request exceeds the sample count.
    let k = k_resolved.min(n).max(1);

    // sklearn resolves `gamma=None` to `1/n_features` at fit, once `n_features`
    // is known. The constraint is `Interval(Real, 0, None, closed="left")` — a
    // gamma of exactly 0 is LEGAL (it yields an all-ones affinity), so only a
    // negative or non-finite value is rejected.
    let gamma = match plan.gamma {
        Some(g) => g,
        None => 1.0 / (d.max(1) as f64),
    };
    if !(gamma >= 0.0) || !gamma.is_finite() {
        return Err(AlgoError::InvalidGamma {
            estimator: plan.estimator,
            gamma,
        });
    }

    let kind = super::spectral_affinity::parse_affinity(
        plan.affinity,
        gamma,
        plan.degree,
        plan.coef0,
        plan.allow_kernels,
    )
    .ok_or_else(|| AlgoError::InvalidKernel {
        estimator: plan.estimator,
        kernel: plan.affinity.to_string(),
    })?;

    // The precomputed affinities consume an `n × n` matrix, not an `n × d`
    // design — check that BEFORE indexing into it (ASVS V5).
    let precomputed = matches!(
        kind,
        super::spectral_affinity::AffinityKind::Precomputed
            | super::spectral_affinity::AffinityKind::PrecomputedNearestNeighbors
    );
    let expect = if precomputed { n * n } else { n * d };
    if x.len() != expect {
        return Err(AlgoError::InvalidGraphInput {
            estimator: plan.estimator,
            reason: format!(
                "affinity '{}' expects a {} matrix ({} values), got {}",
                plan.affinity,
                if precomputed { "n x n" } else { "n x n_features" },
                expect,
                x.len()
            ),
        });
    }

    let n_neighbors_used = matches!(
        kind,
        super::spectral_affinity::AffinityKind::NearestNeighbors
            | super::spectral_affinity::AffinityKind::PrecomputedNearestNeighbors
    )
    .then_some(k);
    let gamma_used = matches!(kind, super::spectral_affinity::AffinityKind::Kernel(_))
        .then_some(gamma);

    // Stage timings behind `log::debug!` (`RUST_LOG=debug`). The stage split is
    // not guessable from the outside — the affinity build, the graph scan and
    // the eigensolver all scale differently in `n`, `d` and `n_neighbors` — and
    // inferring it from A/B'd knobs gave misleading answers, because forcing a
    // tiny solver basis makes the SOLVER slower (it stalls and restarts) rather
    // than isolating the affinity.
    let t_aff = std::time::Instant::now();
    let affinity = super::spectral_affinity::build_affinity(&kind, x, n, d, k);
    let aff_ms = t_aff.elapsed().as_secs_f64() * 1e3;
    let t_cc = std::time::Instant::now();
    let (n_graph_components, comp_labels) = component_labels(&affinity, n);
    let cc_ms = t_cc.elapsed().as_secs_f64() * 1e3;

    // `NormAdj` scales in place, so the stored `affinity_matrix_` needs its own
    // copy. For the default sparse graph that is `~2·n·k` values; for a dense
    // kernel affinity it is the same `n²` sklearn itself keeps on the estimator.
    let t_norm = std::time::Instant::now();
    let op = NormAdj::new(affinity.clone(), n);
    let norm_ms = t_norm.elapsed().as_secs_f64() * 1e3;

    // The normalized Laplacian is only defined for a NON-NEGATIVE affinity:
    // `dd[i] = sqrt(Σ_{j≠i} A[i,j])` is NaN as soon as a row's off-diagonal sum
    // goes negative, and that NaN then propagates through the whole
    // decomposition. `SpectralClustering` can reach this legitimately — sklearn
    // lets `affinity` name ANY `pairwise_kernels` metric, and `additive_chi2`
    // (a negated distance) or a `sigmoid` with a negative `coef0` produce
    // negative entries. sklearn documents that "only kernels that produce
    // similarity scores should be used" and then does not check, so it hands
    // scipy a NaN Laplacian and reports whatever ARPACK does with it. Failing
    // with a typed error is strictly more useful, and it is what keeps the host
    // eigensolvers — which assume finite input — from indexing out of bounds on
    // a NaN convergence test.
    if let Some(i) = op.dd.iter().position(|v| !v.is_finite()) {
        return Err(AlgoError::InvalidGraphInput {
            estimator: plan.estimator,
            reason: format!(
                "affinity '{}' produced a non-finite degree at row {i}; the \
                 normalized Laplacian requires a non-negative, finite affinity \
                 (only similarity kernels are meaningful here)",
                plan.affinity
            ),
        });
    }

    // A graph with `c` components has Laplacian eigenvalue 0 with multiplicity
    // `c`, which a SINGLE-vector Krylov iteration cannot resolve — it returns
    // one vector from that eigenspace and fills the rest from the next distinct
    // eigenvalues, reporting convergence either way. That is why
    // [`lanczos_largest`] starts from a BLOCK of `nev` vectors: the degenerate
    // case is handled in the solver, so no routing decision is needed here and
    // a disconnected graph does not have to fall back to the `O(n³)` dense arm.
    let t_eig = std::time::Instant::now();
    // The component count IS the exact multiplicity of the zero eigenvalue, and
    // it is already in hand from the connectivity scan above — so the solver is
    // told how wide its Krylov block has to be instead of always paying for the
    // worst case.
    // A DISCONNECTED graph's null space is known in closed form — one vector per
    // component, `u[i] = dd[i]` on that component and 0 elsewhere (see
    // `null_space_basis`). Writing those down instead of iterating for them
    // removes the case an iterative solver handles WORST (an exactly repeated
    // eigenvalue) and that spectral CLUSTERING hits by construction, since a kNN
    // graph over separated clusters is disconnected. When the null space already
    // supplies every vector asked for, no iteration runs at all.
    let null = if n_graph_components > 1 {
        null_space_basis(&comp_labels, n_graph_components, &op.dd, n, &op.degree_positive())
    } else {
        Vec::new()
    };
    let n_null = if n == 0 { 0 } else { null.len() / n };
    let take_null = n_null.min(nev);
    let vecs = if take_null >= nev {
        null[..nev * n].to_vec()
    } else {
        // A dense kernel affinity is treated as if it were tied even when
        // connected: its low spectrum is clustered by construction (see
        // `NormAdj::is_dense`), and MEASURED on a 2000x16 rbf fit the lean block
        // cost 466 ms against 255 ms for the wide one.
        let want = nev - take_null;
        let multiplicity = if op.is_dense() {
            nev
        } else {
            (n_graph_components - take_null).max(1)
        };
        let rest = if n <= dense_threshold() || nev * 4 >= n {
            // The dense arm computes the whole spectrum anyway, so the analytic
            // null space buys it nothing; take its answer whole.
            smallest_laplacian_vectors_for(&op, nev, plan.seed, false, multiplicity)
        } else {
            let tail = lanczos_largest_deflated(&op, want, plan.seed, multiplicity, &null);
            let mut all = Vec::with_capacity(nev * n);
            all.extend_from_slice(&null[..take_null * n]);
            all.extend_from_slice(&tail[..want.min(tail.len() / n.max(1)) * n]);
            all.resize(nev * n, 0.0);
            all
        };
        rest
    };
    let eig_ms = t_eig.elapsed().as_secs_f64() * 1e3;
    let embedding = recover_rows(vecs, &op.dd, n, plan.n_components, plan.drop_first, true);
    log::debug!(
        "spectral[{}] n={n} d={d} nev={nev} k={k} components={n_graph_components} \
         nnz={} nullspace={n_null} | affinity {aff_ms:.1}ms | components {cc_ms:.1}ms | \
         normalize+solve {norm_ms:.1}+{eig_ms:.1}ms",
        plan.estimator,
        match &affinity {
            HostAffinity::Sparse(c) => c.data.len(),
            HostAffinity::Dense(_) => n * n,
        },
    );

    Ok(SpectralFit {
        embedding,
        affinity,
        n_neighbors_used,
        gamma_used,
        n_graph_components,
    })
}

/// Per-node connected-component label, plus the component count.
///
/// The count alone reproduces sklearn's connectivity warning; the LABELS are
/// what let [`null_space_basis`] write the Laplacian's null space down in closed
/// form instead of iterating for it.
pub fn component_labels(aff: &HostAffinity, n: usize) -> (usize, Vec<u32>) {
    let mut label = vec![u32::MAX; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut comps = 0u32;
    for s in 0..n {
        if label[s] != u32::MAX {
            continue;
        }
        label[s] = comps;
        stack.push(s);
        while let Some(i) = stack.pop() {
            match aff {
                HostAffinity::Sparse(c) => {
                    let (lo, hi) = (c.indptr[i] as usize, c.indptr[i + 1] as usize);
                    for t in lo..hi {
                        let j = c.indices[t] as usize;
                        if c.data[t] != 0.0 && label[j] == u32::MAX {
                            label[j] = comps;
                            stack.push(j);
                        }
                    }
                }
                HostAffinity::Dense(d) => {
                    for j in 0..n {
                        if d[i * n + j] != 0.0 && label[j] == u32::MAX {
                            label[j] = comps;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        comps += 1;
    }
    (comps as usize, label)
}

/// The normalized Laplacian's null space, in CLOSED FORM, as orthonormal
/// columns (`out[c·n .. (c+1)·n]`).
///
/// For a connected component `C` whose nodes all have positive degree, the
/// vector `u[i] = dd[i]` for `i ∈ C` and `0` elsewhere is an EXACT eigenvector
/// of eigenvalue 0:
///
/// ```text
///   (L u)_i = u_i − Σ_{j≠i} A[i,j]·u_j /(dd_i·dd_j)
///           = dd_i − (1/dd_i)·Σ_{j≠i} A[i,j]
///           = dd_i − deg_i/dd_i = dd_i − dd_i = 0
/// ```
///
/// and vectors from different components have disjoint support, so they are
/// mutually orthogonal. A graph with `c` such components therefore has a
/// `c`-dimensional null space that needs no iteration at all — which matters
/// because that is precisely the case an iterative solver finds HARDEST (an
/// exactly repeated eigenvalue, needing a Krylov block at least `c` wide) and
/// precisely the case spectral CLUSTERING is normally applied to.
///
/// ISOLATED nodes are excluded. A zero-degree node takes the `dd = 1` guard and
/// keeps `L[i,i] = 1` after sklearn's `_set_diag`, so its eigenvalue is 1, not
/// 0 — counting it as a null direction would hand back a vector that is not an
/// eigenvector at all.
pub fn null_space_basis(labels: &[u32], ncomp: usize, dd: &[f64], n: usize, deg_positive: &[bool]) -> Vec<f64> {
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for c in 0..ncomp as u32 {
        let mut v = vec![0.0f64; n];
        let mut norm2 = 0.0;
        let mut any = false;
        for i in 0..n {
            if labels[i] == c {
                if !deg_positive[i] {
                    any = false;
                    break;
                }
                v[i] = dd[i];
                norm2 += dd[i] * dd[i];
                any = true;
            }
        }
        if !any || norm2 <= 0.0 {
            continue;
        }
        let inv = 1.0 / norm2.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
        cols.push(v);
    }
    let mut out = Vec::with_capacity(cols.len() * n);
    for c in cols {
        out.extend_from_slice(&c);
    }
    out
}

/// Number of connected components of the affinity graph, treating any NONZERO
/// entry as an edge — sklearn's `_graph_is_connected`. Used only to reproduce
/// its `"Graph is not fully connected"` warning; like sklearn's, it changes no
/// behavior.
pub fn connected_components(aff: &HostAffinity, n: usize) -> usize {
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut comps = 0usize;
    for s in 0..n {
        if seen[s] {
            continue;
        }
        comps += 1;
        seen[s] = true;
        stack.push(s);
        while let Some(i) = stack.pop() {
            match aff {
                HostAffinity::Sparse(c) => {
                    let (lo, hi) = (c.indptr[i] as usize, c.indptr[i + 1] as usize);
                    for t in lo..hi {
                        let j = c.indices[t] as usize;
                        if c.data[t] != 0.0 && !seen[j] {
                            seen[j] = true;
                            stack.push(j);
                        }
                    }
                }
                HostAffinity::Dense(d) => {
                    for j in 0..n {
                        if d[i * n + j] != 0.0 && !seen[j] {
                            seen[j] = true;
                            stack.push(j);
                        }
                    }
                }
            }
        }
    }
    comps
}

// ---------------------------------------------------------------------------
// Label assignment — sklearn `SpectralClustering.assign_labels`
// ---------------------------------------------------------------------------

/// sklearn's `assign_labels` — the three ways `SpectralClustering` turns the
/// real-valued spectral embedding into a discrete partition.
///
/// All three read the SAME `n × n_components` `maps` matrix that
/// [`run`] returns; they differ only in how they discretize it, so keeping them
/// together here (rather than in the estimator) means the estimator body stays a
/// dispatch rather than three inlined algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignLabels {
    /// `"kmeans"` (sklearn's default) — k-means on the embedding with `n_init`
    /// restarts, keeping the lowest-inertia run.
    KMeans,
    /// `"discretize"` — the Yu & Shi rotation search.
    Discretize,
    /// `"cluster_qr"` — column-pivoted QR followed by an argmax.
    ClusterQr,
}

impl AssignLabels {
    /// Resolve the sklearn string. Returns `None` for anything outside
    /// sklearn's `StrOptions({"kmeans", "discretize", "cluster_qr"})`, so the
    /// estimator can reject an unknown value exactly where sklearn's
    /// `_fit_context` validation does.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "kmeans" => Some(AssignLabels::KMeans),
            "discretize" => Some(AssignLabels::Discretize),
            "cluster_qr" => Some(AssignLabels::ClusterQr),
            _ => None,
        }
    }

    /// The sklearn spelling, for round-tripping the parameter back to a caller.
    pub fn as_str(self) -> &'static str {
        match self {
            AssignLabels::KMeans => "kmeans",
            AssignLabels::Discretize => "discretize",
            AssignLabels::ClusterQr => "cluster_qr",
        }
    }
}

// ---------------------------------------------------------------------------
// Host k-means (`assign_labels="kmeans"`)
// ---------------------------------------------------------------------------

/// `sklearn.cluster.KMeans`'s default iteration cap. `k_means(...)` is called
/// from `SpectralClustering` without `max_iter`, so the default is what runs.
const KMEANS_MAX_ITER: usize = 300;

/// `sklearn.cluster.KMeans`'s default `tol`. sklearn does NOT use it as an
/// absolute bound: `_tolerance(X, tol)` scales it by the MEAN of the per-feature
/// variances, so the stopping rule is invariant to the overall scale of the
/// embedding — which matters here because a spectral embedding's magnitude
/// depends on `1/sqrt(degree)` and is otherwise arbitrary.
const KMEANS_TOL: f64 = 1e-4;

/// The result of a host k-means run on the embedding.
pub struct HostKMeansFit {
    /// Length-`n` cluster assignment (the `labels_` sklearn returns).
    pub labels: Vec<i32>,
    /// `Σ_i ‖x_i − c_{labels_i}‖²` for the returned centers — the quantity the
    /// `n_init` restarts are ranked by.
    pub inertia: f64,
    /// Row-major `k × d` cluster centers.
    pub centers: Vec<f64>,
}

/// k-means on a row-major `n × d` host matrix, in `f64`, with `n_init` restarts
/// keeping the LOWEST-inertia run — `sklearn.cluster.k_means(X, k, n_init=...)`
/// with its defaults (`init="k-means++"`, `max_iter=300`, `tol=1e-4`,
/// `algorithm="lloyd"`).
///
/// ## Why this is not `crate::cluster::kmeans::KMeans`
/// The device estimator has no `n_init` (it is a single k-means++ draw), and on
/// the `cpu` backend every Lloyd step is a cubecl launch — a runtime that spawns
/// one OS thread per unit and JITs at `-O0`, which is pathological for the tiny
/// `d = n_components` geometry a spectral embedding produces. This is the whole
/// point of the CPU rewrite, so the assignment stage runs on the host too.
///
/// ## sklearn parity
/// The k-means++ seeding below is sklearn's `_kmeans_plusplus` verbatim in
/// structure — including the `2 + int(log(k))` LOCAL TRIALS per center, which is
/// not a detail: greedy k-means++ picks the candidate that minimizes the
/// resulting potential rather than the first draw, and a plain D²-weighted
/// sampler lands on visibly worse partitions on the same data. What is NOT
/// reproduced is the bit stream: sklearn draws from numpy's MT19937 and this
/// draws from [`SplitMix64`] (the repo's ASVS-V6 host PRNG), so the two explore
/// different candidate sets. On a well-separated embedding — which is what a
/// spectral pipeline is FOR — every restart converges to the same partition, so
/// the labels agree up to a permutation regardless.
pub fn host_kmeans(
    x: &[f64],
    n: usize,
    d: usize,
    k: usize,
    n_init: usize,
    seed: u64,
) -> HostKMeansFit {
    let k = k.clamp(1, n.max(1));
    let tol = KMEANS_TOL * mean_feature_variance(x, n, d);
    // ONE PRNG stream across all restarts, as sklearn threads one `random_state`
    // through its `n_init` loop — consecutive restarts must not repeat a draw.
    let mut rng = SplitMix64::new(seed);
    let mut best: Option<HostKMeansFit> = None;
    for _ in 0..n_init.max(1) {
        let centers = kmeanspp_host(x, n, d, k, &mut rng);
        let fit = lloyd_host(x, n, d, k, centers, tol);
        let better = match &best {
            None => true,
            Some(b) => fit.inertia < b.inertia,
        };
        if better {
            best = Some(fit);
        }
    }
    best.expect("n_init.max(1) >= 1 always produces a candidate")
}

/// sklearn's `_tolerance(X, tol)` scale factor: the MEAN over features of the
/// per-feature (population) variance. An empty or single-column input yields
/// `0`, which turns the shift test into "iterate until the labels stop moving" —
/// the same degenerate behavior sklearn's `tol == 0` short circuit produces.
fn mean_feature_variance(x: &[f64], n: usize, d: usize) -> f64 {
    if n == 0 || d == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for j in 0..d {
        let mut mean = 0.0;
        for i in 0..n {
            mean += x[i * d + j];
        }
        mean /= n as f64;
        let mut var = 0.0;
        for i in 0..n {
            let t = x[i * d + j] - mean;
            var += t * t;
        }
        total += var / n as f64;
    }
    total / d as f64
}

/// `‖a − b‖²` over `d` features.
#[inline]
fn sqdist(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (&p, &q) in a.iter().zip(b.iter()) {
        let t = p - q;
        s += t * t;
    }
    s
}

/// sklearn's `_kmeans_plusplus`: the greedy D²-weighted seeding, returning the
/// row-major `k × d` initial centers.
///
/// The first center is uniform over the rows. Each subsequent center is chosen
/// by drawing `2 + int(log(k))` candidates ∝ their squared distance to the
/// nearest already-chosen center and KEEPING THE ONE that minimizes the
/// resulting total potential — the "greedy" variant, which is what sklearn (and
/// Arthur & Vassilvitskii's own implementation) does.
fn kmeanspp_host(x: &[f64], n: usize, d: usize, k: usize, rng: &mut SplitMix64) -> Vec<f64> {
    let mut centers = vec![0.0f64; k * d];
    if n == 0 || d == 0 {
        return centers;
    }
    // `2 + int(np.log(n_clusters))` — natural log, truncated.
    let n_local_trials = 2 + (k as f64).ln().max(0.0) as usize;

    let first = rng.next_below(n as u64) as usize;
    centers[..d].copy_from_slice(&x[first * d..(first + 1) * d]);

    let mut closest: Vec<f64> = (0..n)
        .map(|i| sqdist(&x[i * d..(i + 1) * d], &x[first * d..(first + 1) * d]))
        .collect();
    let mut pot: f64 = closest.iter().sum();

    let mut cum = vec![0.0f64; n];
    let mut cand = vec![0.0f64; n];
    let mut best_closest = vec![0.0f64; n];
    for c in 1..k {
        // sklearn draws against `stable_cumsum(closest_dist_sq)` and locates the
        // candidate with `np.searchsorted(..., side="left")` — the first index
        // whose prefix sum reaches the draw. `partition_point` is exactly that.
        let mut acc = 0.0;
        for (o, &w) in cum.iter_mut().zip(closest.iter()) {
            acc += w;
            *o = acc;
        }
        let mut best_pot = f64::INFINITY;
        let mut best_idx = 0usize;
        for _ in 0..n_local_trials {
            let target = rng.next_f64() * pot;
            let idx = cum.partition_point(|&v| v < target).min(n - 1);
            let xi = &x[idx * d..(idx + 1) * d];
            let mut p = 0.0;
            for i in 0..n {
                let m = sqdist(&x[i * d..(i + 1) * d], xi).min(closest[i]);
                cand[i] = m;
                p += m;
            }
            if p < best_pot {
                best_pot = p;
                best_idx = idx;
                best_closest.copy_from_slice(&cand);
            }
        }
        centers[c * d..(c + 1) * d].copy_from_slice(&x[best_idx * d..(best_idx + 1) * d]);
        closest.copy_from_slice(&best_closest);
        pot = best_pot;
    }
    centers
}

/// Lloyd iterations from a given initialization, returning the converged
/// labels / inertia / centers.
///
/// The loop shape is sklearn's `_kmeans_single_lloyd`: the E-step labels are
/// computed against the CURRENT centers, the centers are then updated, and the
/// iteration stops on either an unchanged labeling (strict convergence) or a
/// squared center shift within `tol`. A FINAL E-step re-labels against the last
/// centers so the returned labels and centers are consistent — sklearn does the
/// same, and skipping it leaves labels that belong to the previous iterate.
fn lloyd_host(
    x: &[f64],
    n: usize,
    d: usize,
    k: usize,
    mut centers: Vec<f64>,
    tol: f64,
) -> HostKMeansFit {
    // `(label, squared distance to that label's center)`, filled by the E-step.
    let mut assign = vec![(0i32, 0.0f64); n];
    let mut labels_old = vec![-1i32; n];
    let mut counts = vec![0usize; k];
    let mut sums = vec![0.0f64; k * d];

    for _ in 0..KMEANS_MAX_ITER {
        e_step(x, n, d, k, &centers, &mut assign);
        let shift = m_step(x, n, d, k, &assign, &mut counts, &mut sums, &mut centers);
        if assign.iter().map(|&(l, _)| l).eq(labels_old.iter().copied()) {
            break;
        }
        if shift <= tol {
            break;
        }
        for (o, &(l, _)) in labels_old.iter_mut().zip(assign.iter()) {
            *o = l;
        }
    }
    e_step(x, n, d, k, &centers, &mut assign);

    let inertia = assign.iter().map(|&(_, dist)| dist).sum();
    let labels = assign.iter().map(|&(l, _)| l).collect();
    HostKMeansFit {
        labels,
        inertia,
        centers,
    }
}

/// The E-step: nearest center per row, with the LOWEST index winning a tie
/// (`np.argmin` semantics). Parallel over disjoint row blocks — each row's
/// assignment is independent, so the split cannot change a value.
fn e_step(x: &[f64], n: usize, d: usize, k: usize, centers: &[f64], out: &mut [(i32, f64)]) {
    debug_assert_eq!(out.len(), n);
    // Per-row work: one `d`-dimensional squared distance against each of the `k`
    // centers.
    par_blocks(out, k.saturating_mul(d), |i0, blk| {
        for (t, o) in blk.iter_mut().enumerate() {
            let i = i0 + t;
            let xi = &x[i * d..(i + 1) * d];
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..k {
                let dist = sqdist(xi, &centers[c * d..(c + 1) * d]);
                if dist < best_d {
                    best_d = dist;
                    best = c;
                }
            }
            *o = (best as i32, best_d);
        }
    });
}

/// The M-step: cluster means, plus sklearn's empty-cluster relocation. Returns
/// the SQUARED total center shift, which is what `_kmeans_single_lloyd` compares
/// against the scaled `tol`.
///
/// sklearn's `_relocate_empty_clusters_dense` moves each empty cluster onto the
/// sample FARTHEST from its own assigned center and removes that sample from the
/// donor cluster's running sum. The one deviation here is that a donor holding a
/// single sample is skipped rather than emptied — sklearn can leave the donor
/// empty and repair it on the next iteration, which costs an extra sweep for no
/// benefit. Ties on the distance ranking break on the lowest row index; sklearn
/// leaves them to `np.argpartition`, which does not promise an order.
fn m_step(
    x: &[f64],
    n: usize,
    d: usize,
    k: usize,
    assign: &[(i32, f64)],
    counts: &mut [usize],
    sums: &mut [f64],
    centers: &mut [f64],
) -> f64 {
    counts.iter_mut().for_each(|c| *c = 0);
    sums.iter_mut().for_each(|s| *s = 0.0);
    for (i, &(l, _)) in assign.iter().enumerate() {
        let c = l as usize;
        counts[c] += 1;
        for j in 0..d {
            sums[c * d + j] += x[i * d + j];
        }
    }

    let empty: Vec<usize> = (0..k).filter(|&c| counts[c] == 0).collect();
    if !empty.is_empty() {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            assign[b]
                .1
                .partial_cmp(&assign[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut cursor = 0usize;
        for &target in &empty {
            while cursor < n {
                let i = order[cursor];
                cursor += 1;
                let donor = assign[i].0 as usize;
                if counts[donor] <= 1 {
                    continue;
                }
                counts[donor] -= 1;
                counts[target] = 1;
                for j in 0..d {
                    sums[donor * d + j] -= x[i * d + j];
                    sums[target * d + j] = x[i * d + j];
                }
                break;
            }
        }
    }

    let mut shift = 0.0;
    for c in 0..k {
        if counts[c] == 0 {
            // Nothing to relocate onto — keep the previous center, which is what
            // leaving `centers_new` untouched amounts to.
            continue;
        }
        let inv = 1.0 / counts[c] as f64;
        for j in 0..d {
            let v = sums[c * d + j] * inv;
            let delta = v - centers[c * d + j];
            shift += delta * delta;
            centers[c * d + j] = v;
        }
    }
    shift
}

// ---------------------------------------------------------------------------
// A small dense SVD, shared by `cluster_qr` and `discretize`
// ---------------------------------------------------------------------------

/// Sweeps allowed in the one-sided Jacobi SVD below. It converges quadratically
/// on a `k × k` matrix and needs a handful; the cap only bounds a
/// NaN-poisoned input.
const JACOBI_SVD_SWEEPS: usize = 60;

/// SVD of a row-major `k × k` matrix: returns `(u, s, vt)` with
/// `m = u · diag(s) · vt`, `s` DESCENDING, and `u` / `vt` row-major and
/// orthogonal — `numpy.linalg.svd`'s contract, which is what both `cluster_qr`
/// and `discretize` consume.
///
/// One-sided Jacobi: rotate PAIRS of columns of a working copy of `m` until they
/// are mutually orthogonal. The working copy is then `m·V = U·Σ`, so the column
/// norms are the singular values and the normalized columns are `U`. This is
/// chosen over a bidiagonal reduction because `k` is `n_clusters` (single
/// digits, in practice), the method is unconditionally convergent, and it is
/// accurate to high relative precision on the small, badly-scaled matrices
/// `discretize` produces when a cluster is nearly empty.
///
/// A rank-deficient `m` leaves some `σ = 0`, where `U`'s corresponding column is
/// mathematically ARBITRARY (any orthonormal completion is a valid SVD). Those
/// columns are completed by Gram–Schmidt against a deterministic basis. LAPACK
/// makes its own arbitrary choice there, so a rank-deficient input is the one
/// case where this cannot agree with numpy — a fact that matters only for
/// `cluster_qr` / `discretize` on a degenerate embedding, where the partition
/// itself is ambiguous.
fn svd_square(m: &[f64], k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut a = m.to_vec();
    let mut v = vec![0.0f64; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }
    if k == 0 {
        return (v.clone(), Vec::new(), v);
    }

    for _ in 0..JACOBI_SVD_SWEEPS {
        let mut rotated = false;
        for p in 0..k.saturating_sub(1) {
            for q in (p + 1)..k {
                let mut app = 0.0;
                let mut aqq = 0.0;
                let mut apq = 0.0;
                for i in 0..k {
                    let xp = a[i * k + p];
                    let xq = a[i * k + q];
                    app += xp * xp;
                    aqq += xq * xq;
                    apq += xp * xq;
                }
                if apq == 0.0 || apq.abs() <= 1e-15 * (app * aqq).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (aqq - app) / (2.0 * apq);
                // The SMALLER root of `t² + 2ζt − 1 = 0`, the numerically stable
                // choice; `sign(0) = +1` (a plain `signum` would return 0 and
                // cancel the rotation entirely).
                let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..k {
                    let xp = a[i * k + p];
                    let xq = a[i * k + q];
                    a[i * k + p] = c * xp - s * xq;
                    a[i * k + q] = s * xp + c * xq;
                    let vp = v[i * k + p];
                    let vq = v[i * k + q];
                    v[i * k + p] = c * vp - s * vq;
                    v[i * k + q] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }

    // Column norms are the singular values; sort DESCENDING (numpy's order),
    // permuting `U` and `V` with them.
    let sigma: Vec<f64> = (0..k)
        .map(|j| (0..k).map(|i| a[i * k + j] * a[i * k + j]).sum::<f64>().sqrt())
        .collect();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&p, &q| {
        sigma[q]
            .partial_cmp(&sigma[p])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(p.cmp(&q))
    });

    let scale = sigma.iter().cloned().fold(0.0f64, f64::max);
    let zero_cut = 1e-13 * scale.max(f64::MIN_POSITIVE);
    let mut u = vec![0.0f64; k * k];
    let mut vt = vec![0.0f64; k * k];
    let mut s_out = vec![0.0f64; k];
    let mut deficient: Vec<usize> = Vec::new();
    for (c, &src) in order.iter().enumerate() {
        s_out[c] = sigma[src];
        for i in 0..k {
            vt[c * k + i] = v[i * k + src];
        }
        if sigma[src] > zero_cut {
            let inv = 1.0 / sigma[src];
            for i in 0..k {
                u[i * k + c] = a[i * k + src] * inv;
            }
        } else {
            s_out[c] = 0.0;
            deficient.push(c);
        }
    }

    // Complete the arbitrary `σ = 0` columns of `U` so it stays orthogonal.
    for &c in &deficient {
        let mut filled = false;
        for basis in 0..k {
            let mut w = vec![0.0f64; k];
            w[basis] = 1.0;
            for prev in 0..k {
                if prev == c || deficient.iter().any(|&t| t == prev && t > c) {
                    continue;
                }
                let dp: f64 = (0..k).map(|i| w[i] * u[i * k + prev]).sum();
                for i in 0..k {
                    w[i] -= dp * u[i * k + prev];
                }
            }
            let nrm = w.iter().map(|&t| t * t).sum::<f64>().sqrt();
            if nrm > 1e-8 {
                for i in 0..k {
                    u[i * k + c] = w[i] / nrm;
                }
                filled = true;
                break;
            }
        }
        if !filled {
            u[c * k + c] = 1.0;
        }
    }

    (u, s_out, vt)
}

/// `U · Vᵀ` for the SVD `m = U Σ Vᵀ` — the ORTHOGONAL POLAR FACTOR of a
/// row-major `k × k` matrix, plus `Σ`'s trace.
///
/// Both label-assignment strategies that use the SVD want exactly this product
/// (`cluster_qr` applies it, `discretize` applies its transpose), and
/// `discretize` additionally needs `S.sum()` for its n-cut objective — so both
/// come back together and neither caller has to re-multiply.
fn polar_factor(m: &[f64], k: usize) -> (Vec<f64>, f64) {
    let (u, s, vt) = svd_square(m, k);
    let mut out = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for t in 0..k {
                acc += u[i * k + t] * vt[t * k + j];
            }
            out[i * k + j] = acc;
        }
    }
    (out, s.iter().sum())
}

// ---------------------------------------------------------------------------
// `assign_labels="cluster_qr"`
// ---------------------------------------------------------------------------

/// sklearn's `cluster_qr(vectors)` (Damle, Minden & Ying 2019) on a row-major
/// `n × k` embedding.
///
/// The algorithm is: column-pivoted QR of `vectorsᵀ` to pick the `k` rows that
/// are "most independent" (they act as one representative per cluster), then
/// rotate the whole embedding by the orthogonal polar factor of that `k × k`
/// sub-block's transpose, then take a row-wise argmax of the absolute values. It
/// has no tuning parameters and runs no iterations, which is why sklearn
/// documents it as the fastest of the three.
///
/// The pivot search is Businger–Golub with the trailing column norms RECOMPUTED
/// each step rather than downdated. LAPACK's `dgeqp3` downdates and only
/// recomputes when the running norm has degraded, so on a matrix with a near-tie
/// between two columns the two can pick different pivots; recomputing is the
/// more accurate of the two, and `k` is small enough that the extra `O(n·k²)` is
/// free.
pub fn cluster_qr_labels(vectors: &[f64], n: usize, k: usize) -> Vec<i32> {
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let pivots = pivoted_qr_pivots(vectors, n, k);
    // `vectors[piv[:k], :].T` — a `k × k` block whose (r, c) entry is component
    // `r` of the `c`-th selected row.
    let take = k.min(pivots.len());
    let mut block = vec![0.0f64; k * k];
    for c in 0..take {
        let row = &vectors[pivots[c] * k..(pivots[c] + 1) * k];
        for (r, &v) in row.iter().enumerate() {
            block[r * k + c] = v;
        }
    }
    let (rot, _) = polar_factor(&block, k);

    let mut labels = vec![0i32; n];
    // Per-row work: a `k x k` rotation applied to the row, then an argmax.
    par_blocks(&mut labels, k.saturating_mul(k), |i0, blk| {
        for (t, out) in blk.iter_mut().enumerate() {
            let row = &vectors[(i0 + t) * k..(i0 + t + 1) * k];
            let mut best = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for c in 0..k {
                let mut acc = 0.0;
                for (r, &v) in row.iter().enumerate() {
                    acc += v * rot[r * k + c];
                }
                let acc = acc.abs();
                if acc > best_v {
                    best_v = acc;
                    best = c;
                }
            }
            *out = best as i32;
        }
    });
    labels
}

/// The first `k` column pivots of a Businger–Golub column-pivoted Householder QR
/// of `vectorsᵀ` (a `k × n` matrix), returned as row indices into the row-major
/// `n × k` `vectors`.
///
/// A row-major `n × k` buffer IS a column-major `k × n` one, so `vectors` can be
/// consumed as the transposed matrix with no copy beyond the working duplicate:
/// column `j` of `vectorsᵀ` is the contiguous slice `vectors[j·k .. (j+1)·k]`.
fn pivoted_qr_pivots(vectors: &[f64], n: usize, k: usize) -> Vec<usize> {
    let mut w = vectors.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();
    let steps = k.min(n);
    for s in 0..steps {
        // Pick the trailing column with the largest remaining norm; the FIRST
        // maximum wins a tie, matching `np.argmax`.
        let mut best = s;
        let mut best_n = f64::NEG_INFINITY;
        for j in s..n {
            let mut t = 0.0;
            for r in s..k {
                let v = w[j * k + r];
                t += v * v;
            }
            if t > best_n {
                best_n = t;
                best = j;
            }
        }
        if best != s {
            piv.swap(s, best);
            for r in 0..k {
                w.swap(s * k + r, best * k + r);
            }
        }
        // Householder reflector zeroing rows `s+1..k` of column `s`.
        let mut alpha = 0.0;
        for r in s..k {
            alpha += w[s * k + r] * w[s * k + r];
        }
        let alpha = alpha.sqrt();
        if alpha == 0.0 {
            continue;
        }
        let head = w[s * k + s];
        let sign = if head >= 0.0 { 1.0 } else { -1.0 };
        let mut v: Vec<f64> = (s..k).map(|r| w[s * k + r]).collect();
        v[0] = head + sign * alpha;
        let vn2: f64 = v.iter().map(|&t| t * t).sum();
        if vn2 == 0.0 {
            continue;
        }
        for j in s..n {
            let mut dp = 0.0;
            for (t, &vt) in v.iter().enumerate() {
                dp += vt * w[j * k + s + t];
            }
            let f = 2.0 * dp / vn2;
            for (t, &vt) in v.iter().enumerate() {
                w[j * k + s + t] -= f * vt;
            }
        }
    }
    piv.truncate(steps);
    piv
}

// ---------------------------------------------------------------------------
// `assign_labels="discretize"`
// ---------------------------------------------------------------------------

/// Iteration cap for the rotation search, sklearn's `n_iter_max=20`.
const DISCRETIZE_MAX_ITER: usize = 20;

/// sklearn's `discretize(vectors, random_state=...)` (Yu & Shi 2003) on a
/// row-major `n × k` embedding.
///
/// The search alternates two steps: given a rotation, the closest PARTITION
/// matrix is the row-wise argmax; given a partition, the closest rotation is the
/// orthogonal polar factor of `Pᵀ·V`. It stops when the n-cut objective
/// `2·(n − Σσ)` stops moving (to machine epsilon, sklearn's own test) or after
/// `n_iter_max` iterations.
///
/// The preprocessing is load-bearing and reproduced exactly: each column is
/// rescaled to norm `sqrt(n)` and then sign-flipped so its FIRST entry is
/// negative, and only then are the rows normalized onto the unit sphere. The
/// column sign convention is what keeps the argmax search in one quadrant; the
/// row normalization is what turns the embedding into a point set the partition
/// matrices live among.
///
/// ## Where this cannot match sklearn bit-for-bit
/// The initial rotation's first column is a RANDOMLY chosen row
/// (`random_state.randint(n_samples)`), drawn here from [`SplitMix64`] rather
/// than numpy's MT19937. The search is a local optimization, so a different
/// start can land on a different local optimum on an ambiguous embedding. On a
/// well-separated one, every start reaches the same partition.
///
/// sklearn also wraps the whole search in a `max_svd_restarts=30` loop that
/// re-randomizes when LAPACK's SVD fails to converge. The one-sided Jacobi SVD
/// used here is unconditionally convergent, so that loop would never take its
/// second trip and is not reproduced. A NON-FINITE embedding (which no finite
/// affinity can produce) is reported as
/// [`AlgoError::NotConverged`] instead of being silently discretized.
pub fn discretize_labels(
    vectors: &[f64],
    n: usize,
    k: usize,
    seed: u64,
) -> Result<Vec<i32>, AlgoError> {
    if n == 0 || k == 0 {
        return Ok(vec![0; n]);
    }
    let mut v = vectors.to_vec();

    // Column rescale to `sqrt(n)` + first-entry sign convention.
    let norm_ones = (n as f64).sqrt();
    for c in 0..k {
        let mut nrm = 0.0;
        for i in 0..n {
            nrm += v[i * k + c] * v[i * k + c];
        }
        let nrm = nrm.sqrt();
        if nrm > 0.0 {
            let scale = norm_ones / nrm;
            for i in 0..n {
                v[i * k + c] *= scale;
            }
        }
        let head = v[c];
        if head != 0.0 {
            let sign = if head > 0.0 { 1.0 } else { -1.0 };
            for i in 0..n {
                v[i * k + c] *= -sign;
            }
        }
    }
    // Row normalization onto the unit sphere. sklearn divides unconditionally
    // and would emit NaN on an all-zero row; a zero row is left alone here,
    // which is the same result for every downstream argmax.
    for i in 0..n {
        let row = &mut v[i * k..(i + 1) * k];
        let nrm = row.iter().map(|&t| t * t).sum::<f64>().sqrt();
        if nrm > 0.0 {
            for t in row.iter_mut() {
                *t /= nrm;
            }
        }
    }
    if v.iter().any(|t| !t.is_finite()) {
        return Err(AlgoError::NotConverged {
            estimator: "spectral_clustering",
            max_iter: DISCRETIZE_MAX_ITER,
        });
    }

    // Initial rotation: a random row, then the rows least aligned with the
    // columns already picked.
    let mut rng = SplitMix64::new(seed);
    let mut rotation = vec![0.0f64; k * k];
    let start = rng.next_below(n as u64) as usize;
    for r in 0..k {
        rotation[r * k] = v[start * k + r];
    }
    let mut c_acc = vec![0.0f64; n];
    for j in 1..k {
        for (i, acc) in c_acc.iter_mut().enumerate() {
            let mut dp = 0.0;
            for r in 0..k {
                dp += v[i * k + r] * rotation[r * k + (j - 1)];
            }
            *acc += dp.abs();
        }
        let mut best = 0usize;
        let mut best_v = f64::INFINITY;
        for (i, &acc) in c_acc.iter().enumerate() {
            if acc < best_v {
                best_v = acc;
                best = i;
            }
        }
        for r in 0..k {
            rotation[r * k + j] = v[best * k + r];
        }
    }

    let eps = f64::EPSILON;
    let mut last_objective = 0.0f64;
    let mut labels = vec![0i32; n];
    let mut t_svd = vec![0.0f64; k * k];
    for iter in 0..=DISCRETIZE_MAX_ITER {
        // Closest partition matrix to `V · rotation`.
        for i in 0..n {
            let mut best = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for c in 0..k {
                let mut acc = 0.0;
                for r in 0..k {
                    acc += v[i * k + r] * rotation[r * k + c];
                }
                if acc > best_v {
                    best_v = acc;
                    best = c;
                }
            }
            labels[i] = best as i32;
        }
        // `t_svd = Pᵀ · V` — one row per cluster, summing that cluster's points.
        t_svd.iter_mut().for_each(|t| *t = 0.0);
        for (i, &l) in labels.iter().enumerate() {
            let c = l as usize;
            for r in 0..k {
                t_svd[c * k + r] += v[i * k + r];
            }
        }
        let (polar, s_sum) = polar_factor(&t_svd, k);
        let ncut = 2.0 * (n as f64 - s_sum);
        if (ncut - last_objective).abs() < eps || iter >= DISCRETIZE_MAX_ITER {
            break;
        }
        last_objective = ncut;
        // sklearn: `rotation = Vh.T @ U.T`, which is `(U · Vh)ᵀ`.
        for r in 0..k {
            for c in 0..k {
                rotation[r * k + c] = polar[c * k + r];
            }
        }
    }
    Ok(labels)
}
