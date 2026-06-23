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
