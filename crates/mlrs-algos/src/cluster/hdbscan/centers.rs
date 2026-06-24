//! `store_centers` → `centroids_` / `medoids_` (HDBS-04, plan 15-06).
//!
//! A line-for-line host port of sklearn's
//! `cluster/_hdbscan/hdbscan.py::_weighted_cluster_center` (RESEARCH Pattern 8),
//! gated vs sklearn ≤1e-5 under the SAME label permutation (Pitfall 6). For each
//! cluster id `c` in ascending order `0..n_clusters` (clusters = the distinct
//! non-negative fitted labels):
//!
//! ```text
//! data     = X[labels == c]                          # the cluster's feature rows
//! strength = probabilities[labels == c]              # per-point membership weight
//! centroid[c] = np.average(data, weights=strength, axis=0)   # weighted mean
//! dist_mat = pairwise_distances(data, metric) * strength     # WEIGHT each column j by strength[j]
//! medoid[c]   = data[argmin(dist_mat.sum(axis=1))]           # min weighted total distance
//! ```
//!
//! ## Two subtleties (RESEARCH Pattern 8 / Pitfall 6)
//! - The **medoid weights the distance matrix by `strength`** as a ROW vector
//!   (`dist_mat[i, j] *= strength[j]`), THEN sums each row and argmins — NOT a
//!   weighted mean of distances. The centroid weights by `strength` directly.
//! - Centers iterate **ascending cluster id** `0..n_clusters`; the caller compares
//!   them under the same label permutation that maps fitted→sklearn ids.
//!
//! `store_centers` is **feature-array only**: requesting it with
//! `Metric::Precomputed` errors (there are no feature rows to average) — sklearn
//! parity, threat T-15-06-V5. That guard lives in `hdbscan.rs::fit`; this module
//! assumes `data` is the genuine `n×p` feature matrix.
//!
//! All scalar math is `f64` (the host bridging domain). The per-cluster pairwise
//! distance is computed host-side over the SMALL per-cluster `data` (Don't
//! Hand-Roll: a host pairwise matching `pairwise_distances` for the metric).
//!
//! Tests live in `crates/mlrs-algos/tests/hdbscan_test.rs` (AGENTS.md §2).

use super::super::hdbscan::Metric;
// `host_pairwise` lives in the shared `distance` submodule (IN-03), which
// includes the `Cosine` arm this caller needs.
use super::distance::host_pairwise;

/// Which centers to compute in [`weighted_cluster_center`] (mirrors the public
/// `StoreCenters` enum, decoupled so this module needs no estimator state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Centers {
    /// Only `centroids_`.
    Centroid,
    /// Only `medoids_`.
    Medoid,
    /// Both `centroids_` and `medoids_`.
    Both,
}

impl Centers {
    /// Whether the centroid (probability-weighted mean) is requested.
    fn wants_centroid(self) -> bool {
        matches!(self, Centers::Centroid | Centers::Both)
    }
    /// Whether the medoid (strength-weighted min-total-distance) is requested.
    fn wants_medoid(self) -> bool {
        matches!(self, Centers::Medoid | Centers::Both)
    }
}

/// Per-cluster centroids/medoids over the fitted `labels` + `probabilities`
/// (sklearn `_weighted_cluster_center`). `x` is the row-major `n×p` feature
/// matrix; `labels[i]` is point `i`'s cluster (`-1` = noise, excluded); `probs[i]`
/// is its membership strength. `metric` is the estimator's distance (for the
/// medoid's pairwise distances).
///
/// Returns `(centroids, medoids)` — each is `Some(Vec<f64>)` of length
/// `n_clusters * p` (row-major, cluster id `c` at rows `c*p..(c+1)*p`) when
/// requested by `which`, else `None`. `n_clusters` is the number of distinct
/// non-negative labels (the dense `0..n_clusters` range produced by selection).
///
/// A cluster with zero total strength (degenerate; should not occur for a
/// genuinely selected cluster) falls back to an UNWEIGHTED mean / total distance
/// so the result stays finite.
pub fn weighted_cluster_center(
    x: &[f64],
    labels: &[i32],
    probs: &[f64],
    p: usize,
    metric: Metric,
    which: Centers,
) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    let n = labels.len();
    debug_assert_eq!(x.len(), n * p, "x must be the n×p feature matrix");
    debug_assert_eq!(probs.len(), n, "one probability per point");

    // n_clusters = number of distinct non-negative labels. Selection emits a dense
    // 0..n_clusters range, so the max label + 1 is the count (0 clusters ⇒ all
    // noise ⇒ empty centers).
    let n_clusters = labels
        .iter()
        .filter(|&&l| l >= 0)
        .map(|&l| l as usize + 1)
        .max()
        .unwrap_or(0);

    // WR-06: emitting one row per `c in 0..n_clusters` is correct ONLY when the
    // non-negative label space is dense (`0..n_clusters` with no gaps); a sparse
    // range would leave all-zero "phantom" rows. `select::get_clusters` builds a
    // dense `cluster_map`, so this invariant holds for the `fit` pipeline — assert
    // it so a future selection change or direct caller cannot silently break it.
    debug_assert!(
        {
            use std::collections::BTreeSet;
            let distinct: BTreeSet<usize> =
                labels.iter().filter(|&&l| l >= 0).map(|&l| l as usize).collect();
            distinct.len() == n_clusters
        },
        "weighted_cluster_center requires a dense 0..n_clusters label range",
    );

    let mut centroids = which.wants_centroid().then(|| vec![0.0f64; n_clusters * p]);
    let mut medoids = which.wants_medoid().then(|| vec![0.0f64; n_clusters * p]);

    for c in 0..n_clusters {
        // The member point indices of cluster `c` (ascending — the natural order).
        let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c as i32).collect();
        if members.is_empty() {
            continue;
        }
        let strength: Vec<f64> = members.iter().map(|&i| probs[i]).collect();
        let strength_sum: f64 = strength.iter().sum();

        // --- centroid: probability-weighted mean per feature ---
        if let Some(cent) = centroids.as_mut() {
            for k in 0..p {
                let mut acc = 0.0f64;
                for (mi, &i) in members.iter().enumerate() {
                    acc += x[i * p + k] * strength[mi];
                }
                // np.average normalises by the weight sum; fall back to the plain
                // mean if every member has zero strength (degenerate).
                cent[c * p + k] = if strength_sum > 0.0 {
                    acc / strength_sum
                } else {
                    acc / members.len() as f64
                };
            }
        }

        // --- medoid: argmin over members of the STRENGTH-WEIGHTED total distance ---
        if let Some(med) = medoids.as_mut() {
            // weighted_total[i] = Σ_j dist(member_i, member_j) * strength[j]
            // (sklearn: `pairwise_distances(data) * strength` then sum over axis=1).
            let mut best_member = 0usize;
            let mut best_total = f64::INFINITY;
            for (ri, &i) in members.iter().enumerate() {
                let mut total = 0.0f64;
                for (rj, &j) in members.iter().enumerate() {
                    let d = host_pairwise(x, p, metric, i, j);
                    total += d * strength[rj];
                }
                if total < best_total {
                    best_total = total;
                    best_member = ri;
                }
            }
            let medoid_point = members[best_member];
            for k in 0..p {
                med[c * p + k] = x[medoid_point * p + k];
            }
        }
    }

    (centroids, medoids)
}

