//! `neighbors` — k-nearest-neighbor estimators (NEIGH-01 / NEIGH-02 / NEIGH-03).
//!
//! Module index for the three Phase-5 neighbor estimators. They consume the new
//! Phase-5 top-k select primitive (`prims::topk`, composed on the Phase-2
//! pairwise-distance prim) and expose the
//! [`KNeighbors`](crate::typestate::KNeighbors) /
//! [`PredictLabels`](crate::typestate::PredictLabels) /
//! [`PredictProba`](crate::typestate::PredictProba) surface (D-07), retrofitted
//! onto the typestate `<F, S = Unfit>` builder surface in Phase 16:
//!
//! - `NearestNeighbors` (NEIGH-01) — `kneighbors` returns the `k` nearest
//!   (distances `F`, indices `i32`) per query, matching sklearn within 1e-5.
//! - `KNeighborsClassifier` (NEIGH-02) — majority-vote `predict_labels` (i32) +
//!   neighbor-fraction `predict_proba` (F).
//! - `KNeighborsRegressor` (NEIGH-03) — neighbor-mean `predict` (F).
//!
//! All three are added by plan **05-10** (one fixture serves all three). Each
//! ADDS its own `pub mod <estimator>;` line here and creates the matching file;
//! the plan does NOT edit `lib.rs` (owned by the Wave-0 scaffold), keeping the
//! estimator plans file-disjoint and parallel-safe.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub mod classifier;
pub mod nearest;
pub mod regressor;

/// The distance metric a neighbor estimator searches under (KNN-REG-PARAMS).
///
/// This is a RE-EXPORT of [`mlrs_backend::prims::knn_graph::Metric`], not a
/// mirror of it. HDBSCAN and UMAP each define their own copy because their
/// public metric sets are deliberately narrower than the prim's; the neighbor
/// estimators expose exactly the prim's set, so a second enum here would be a
/// pure `match` that can only ever drift (PATTERNS Pitfall 4). The Minkowski
/// exponent travels INSIDE the `Minkowski { p }` payload — there is no
/// standalone `p` on the estimator, so the compute path and the reported
/// hyperparameter cannot disagree.
///
/// sklearn's aliases collapse onto these: `metric='l2'`/`'euclidean'` and
/// `metric='minkowski', p=2` are [`Metric::Euclidean`]; `'l1'`/`'cityblock'`/
/// `'manhattan'` and `p=1` are [`Metric::Manhattan`]; `'chebyshev'`/`'infinity'`
/// and `p=inf` are [`Metric::Chebyshev`]. Collapsing the p=1/p=2 special cases
/// onto the dedicated kernels is not just an optimization — `minkowski_dist`
/// evaluates `F::powf`, which is capability-gated at f64 on some backends and
/// less accurate than the direct forms even where it runs.
pub use mlrs_backend::prims::knn_graph::Metric;

/// How a neighbor's target contributes to a prediction (sklearn `weights=`).
///
/// The callable third option sklearn accepts is NOT represented here: an
/// arbitrary Python function cannot cross into a device kernel. The Python shim
/// serves it by calling `kneighbors` and doing the weighted average itself, so
/// the core surface stays total over what it can actually compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weights {
    /// `weights='uniform'` — every neighbor contributes `1/k`.
    Uniform,
    /// `weights='distance'` — neighbor `j` contributes in proportion to `1/d_j`.
    ///
    /// A query that coincides with one or more training points takes sklearn's
    /// degenerate branch instead: those neighbors get weight 1 and every other
    /// neighbor in that row gets 0 (otherwise `1/0 = inf` and the normalization
    /// `inf/inf` is NaN). See `mlrs_kernels::knn::knn_regress_gather`.
    Distance,
}
