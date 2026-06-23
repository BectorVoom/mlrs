//! `umap_internals` — UMAP host numeric stages (Plan 02's home).
//!
//! This module is an EMPTY stub created in Plan 14-01 to pre-declare file
//! ownership so Plans 02 and 03 fill their own sibling files WITHOUT both
//! editing `manifold/mod.rs` (file-disjoint, parallel-safe Wave 2).
//!
//! Plan 02 fills this with the deterministic host numerics:
//! `smooth_knn_dist` (per-row ρ/σ binary search), `compute_membership_strengths`
//! (membership exp), and `fuzzy_union` (t-conorm). Plan 05 adds
//! `init_graph_transform` (the transform frozen-subset weighted average).
//!
//! Tests live in `crates/mlrs-algos/tests/umap_test.rs` (AGENTS.md §2).

// ===========================================================================
// Verified umap-learn 0.5.12 constants (umap/umap_.py)
// ===========================================================================

/// Binary-search convergence tolerance (umap `SMOOTH_K_TOLERANCE`).
const SMOOTH_K_TOLERANCE: f64 = 1e-5;
/// Per-row / global sigma floor scale (umap `MIN_K_DIST_SCALE`).
const MIN_K_DIST_SCALE: f64 = 1e-3;
/// Max binary-search iterations (umap `smooth_knn_dist` default `n_iter`).
const SMOOTH_N_ITER: usize = 64;

/// umap's `NPY_FLOATMAX = np.finfo(np.float32).max`. umap accumulates ρ/σ in
/// float32, so the search upper bound and the `hi >= NPY_FLOATMAX` doubling
/// branch use the f32 max (NOT `f64::MAX`/`f64::INFINITY`) — this is what the
/// committed fixtures were produced with, so we match it exactly in host f64.
/// (`hi = inf` is HOST-side only here regardless; the device `F::INFINITY` ban
/// applies only inside CubeCL kernels, of which there are none in this module.)
const NPY_FLOATMAX: f64 = f32::MAX as f64;

/// Per-row smooth-kNN ρ (local connectivity) and σ (bandwidth) — a faithful host
/// f64 port of umap-learn 0.5.12 `umap.umap_.smooth_knn_dist`.
///
/// `knn_dist` is the row-major `(n, k)` directed KNN distance matrix (self
/// already dropped, ascending per row — exactly the Phase-13 prim output).
/// `n_neighbors` is umap's `k` argument (`target = log2(n_neighbors)*bandwidth`,
/// bandwidth = 1.0). `local_connectivity` is the fuzzy local-connectivity knob
/// (1.0 by default → ρ = nearest non-zero-distance neighbour).
///
/// Returns `(sigmas, rhos)`, each length `n`. ORDER is load-bearing: ρ is
/// computed FIRST, then the per-row binary search runs on `d − ρ`
/// (RESEARCH Pattern 1).
///
/// Pure host numerics — no device launch, no `DeviceArray`. Bounded iteration
/// (`SMOOTH_N_ITER`) + umap's zero-guards (per-row & global `MIN_K_DIST_SCALE`
/// floor, ρ ≤ 0 fallback) → no NaN / non-termination on pathological input
/// (threats T-14-03 / T-14-04).
pub fn smooth_knn_dist(
    knn_dist: &[f64],
    n: usize,
    k: usize,
    n_neighbors: usize,
    local_connectivity: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(knn_dist.len(), n * k, "knn_dist must be exactly n*k");

    let target = (n_neighbors as f64).log2(); // bandwidth = 1.0

    // umap's `mean_distances = np.mean(distances)` over the WHOLE (n, k) block.
    let mean_distances = if knn_dist.is_empty() {
        0.0
    } else {
        knn_dist.iter().sum::<f64>() / knn_dist.len() as f64
    };

    let mut sigmas = vec![0.0f64; n];
    let mut rhos = vec![0.0f64; n];

    for i in 0..n {
        let row = &knn_dist[i * k..i * k + k];

        // --- ρ FIRST: local-connectivity interpolation over non-zero dists. ---
        let non_zero: Vec<f64> = row.iter().copied().filter(|&d| d > 0.0).collect();
        let mut rho = 0.0f64;
        if non_zero.len() as f64 >= local_connectivity {
            let index = local_connectivity.floor() as usize;
            let interpolation = local_connectivity - index as f64;
            if index > 0 {
                rho = non_zero[index - 1];
                if interpolation > SMOOTH_K_TOLERANCE {
                    rho += interpolation * (non_zero[index] - non_zero[index - 1]);
                }
            } else {
                rho = interpolation * non_zero[0];
            }
        } else if !non_zero.is_empty() {
            rho = non_zero.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        }

        // --- THEN binary search σ s.t. Σ_{j≥1} exp(-(max(0,d−ρ))/σ) = target. ---
        let mut lo = 0.0f64;
        let mut hi = NPY_FLOATMAX;
        let mut mid = 1.0f64;

        for _ in 0..SMOOTH_N_ITER {
            // umap iterates `for j in range(1, k)` — column 0 (nearest) is
            // skipped on purpose. ORDER load-bearing; replicated verbatim.
            let mut psum = 0.0f64;
            for j in 1..k {
                let d = row[j] - rho;
                if d > 0.0 {
                    psum += (-(d / mid)).exp();
                } else {
                    psum += 1.0;
                }
            }

            if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }

            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if hi >= NPY_FLOATMAX {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
        }

        let mut sigma = mid;

        // --- σ floor: per-row mean when ρ>0, else global-mean fallback. ---
        if rho > 0.0 {
            let mean_ith = row.iter().sum::<f64>() / k as f64;
            if sigma < MIN_K_DIST_SCALE * mean_ith {
                sigma = MIN_K_DIST_SCALE * mean_ith;
            }
        } else if sigma < MIN_K_DIST_SCALE * mean_distances {
            sigma = MIN_K_DIST_SCALE * mean_distances;
        }

        sigmas[i] = sigma;
        rhos[i] = rho;
    }

    (sigmas, rhos)
}

/// Membership strengths for the directed fuzzy 1-skeleton — host f64 port of
/// umap-learn 0.5.12 `compute_membership_strengths`.
///
/// `knn_idx` / `knn_dist` are the row-major `(n, k)` directed KNN neighbour
/// indices (float-encoded, as the fixtures store them — rounded to `usize` here)
/// and distances. `rhos` / `sigmas` are the per-row outputs of
/// [`smooth_knn_dist`].
///
/// Emits the directed COO `(rows, cols, vals)` of length `n*k` with
/// `rows[i*k+j] = i`, `cols[i*k+j] = knn_idx[i,j]`, and the verified membership
/// value
/// `val = if (d − ρ ≤ 0 || σ == 0) { 1.0 } else { exp(-(d − ρ)/σ) }`
/// (self edges — `knn_idx[i,j] == i` — get `val = 0.0`, but the Phase-13 KNN is
/// already self-dropped so that branch is inert here). Zeros are NOT pruned at
/// this stage — [`fuzzy_union`] performs the `eliminate_zeros` equivalent so the
/// directed→symmetric merge sees umap's exact entry set.
///
/// Bounds: `cols` come straight from the Phase-13 prim output (already validated
/// `< n`); host indexing would panic rather than OOB-read (threat T-14-05).
pub fn compute_membership_strengths(
    knn_idx: &[f64],
    knn_dist: &[f64],
    rhos: &[f64],
    sigmas: &[f64],
    n: usize,
    k: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    assert_eq!(knn_idx.len(), n * k, "knn_idx must be exactly n*k");
    assert_eq!(knn_dist.len(), n * k, "knn_dist must be exactly n*k");
    assert_eq!(rhos.len(), n, "one rho per row");
    assert_eq!(sigmas.len(), n, "one sigma per row");

    let mut rows = vec![0usize; n * k];
    let mut cols = vec![0usize; n * k];
    let mut vals = vec![0.0f64; n * k];

    for i in 0..n {
        for j in 0..k {
            let p = i * k + j;
            let col = knn_idx[p].round() as usize;
            let d = knn_dist[p];

            let val = if col == i {
                0.0
            } else if d - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                1.0
            } else {
                (-((d - rhos[i]) / sigmas[i])).exp()
            };

            rows[p] = i;
            cols[p] = col;
            vals[p] = val;
        }
    }

    (rows, cols, vals)
}

/// t-conorm fuzzy-set union (UMAP's symmetrization, D-04) — host f64 port of the
/// `fuzzy_simplicial_set` union step of umap-learn 0.5.12.
///
/// Takes the directed membership COO from [`compute_membership_strengths`] and
/// forms the symmetric graph
/// `G = mix*(A + Aᵀ − A∘Aᵀ) + (1 − mix)*(A∘Aᵀ)`
/// where `A` is the directed sparse membership matrix and `mix` is
/// `set_op_mix_ratio` (1.0 → pure union `A + Aᵀ − A∘Aᵀ`). Zero entries are
/// pruned (umap's `eliminate_zeros`, applied both before AND after the union).
///
/// The output COO is sorted ascending by `(row, col)` — scipy's canonical
/// CSR-backed COO order after the arithmetic — so it is byte-stable for the
/// value-gate and for downstream consumers (spectral init / layout).
///
/// Pure host f64; `n` is small at fixture scale so the HashMap merge is cheap.
pub fn fuzzy_union(
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    _n: usize,
    set_op_mix_ratio: f64,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    use std::collections::{BTreeSet, HashMap};

    assert_eq!(rows.len(), cols.len(), "COO rows/cols length mismatch");
    assert_eq!(rows.len(), vals.len(), "COO rows/vals length mismatch");

    // A = directed membership matrix; umap calls `eliminate_zeros()` BEFORE the
    // union, so we drop zero entries here. (Duplicate (r,c) keys cannot occur:
    // each directed row has distinct neighbour columns.)
    let mut a: HashMap<(usize, usize), f64> = HashMap::with_capacity(vals.len());
    for e in 0..vals.len() {
        if vals[e] != 0.0 {
            a.insert((rows[e], cols[e]), vals[e]);
        }
    }

    // The union touches every (r,c) that is non-zero in A OR in Aᵀ — i.e. for
    // every directed key (r,c) both (r,c) and its transpose (c,r) appear in G.
    let mut keys: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(r, c) in a.keys() {
        keys.insert((r, c));
        keys.insert((c, r));
    }

    let mut out_rows = Vec::with_capacity(keys.len());
    let mut out_cols = Vec::with_capacity(keys.len());
    let mut out_vals = Vec::with_capacity(keys.len());

    // BTreeSet iterates ascending by (row, col) → scipy CSR canonical order.
    for (r, c) in keys {
        let a_rc = a.get(&(r, c)).copied().unwrap_or(0.0);
        let a_cr = a.get(&(c, r)).copied().unwrap_or(0.0);
        let prod = a_rc * a_cr;
        let g = set_op_mix_ratio * (a_rc + a_cr - prod) + (1.0 - set_op_mix_ratio) * prod;
        // umap's trailing `eliminate_zeros()` after the union.
        if g != 0.0 {
            out_rows.push(r);
            out_cols.push(c);
            out_vals.push(g);
        }
    }

    (out_rows, out_cols, out_vals)
}

/// Neighbor-weighted-average initialization for the transform frozen-subset path
/// (UMAP-04, D-03) — host f64 port of umap-learn 0.5.12's `init_graph_transform`
/// (RESEARCH Pattern 7 step 3).
///
/// Each NEW point is initialized to the row-normalized weighted average of its
/// trained-neighbor embedding coordinates:
/// `init[new_i] = Σ_j (graph_ij / rowsum_i) · embedding_train[col_j]`,
/// where `(rows, cols, vals)` is the transform membership graph COO of the `m`
/// new points against the `n` training points (rows ∈ `0..m`, cols ∈ `0..n`) and
/// `embedding_train` is the row-major `(n, n_components)` FROZEN training
/// embedding. A new point with no membership edges (zero rowsum) is left at the
/// origin (all-zeros) — umap's behaviour for an isolated query point.
///
/// Returns the row-major `(m, n_components)` init buffer for the new points.
///
/// Pure host f64; `m`/`n` are small at transform scale so the dense accumulation
/// is cheap. Bounds: `cols` come from the validated query-vs-train KNN, so host
/// indexing panics rather than OOB-reading (threat T-14-16).
pub fn init_graph_transform(
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    embedding_train: &[f64],
    m: usize,
    n: usize,
    n_components: usize,
) -> Vec<f64> {
    assert_eq!(rows.len(), cols.len(), "transform COO rows/cols length mismatch");
    assert_eq!(rows.len(), vals.len(), "transform COO rows/vals length mismatch");
    assert_eq!(
        embedding_train.len(),
        n * n_components,
        "embedding_train must be n*n_components"
    );

    let mut init = vec![0.0f64; m * n_components];
    let mut rowsum = vec![0.0f64; m];

    // Accumulate the un-normalized weighted sum of neighbor coords + the per-row
    // membership total (the normalizer).
    for e in 0..vals.len() {
        let r = rows[e];
        let c = cols[e];
        let w = vals[e];
        rowsum[r] += w;
        let src = c * n_components;
        let dst = r * n_components;
        for d in 0..n_components {
            init[dst + d] += w * embedding_train[src + d];
        }
    }

    // Row-normalize (skip zero-rowsum points — they stay at the origin).
    for r in 0..m {
        if rowsum[r] > 0.0 {
            let dst = r * n_components;
            for d in 0..n_components {
                init[dst + d] /= rowsum[r];
            }
        }
    }

    init
}
