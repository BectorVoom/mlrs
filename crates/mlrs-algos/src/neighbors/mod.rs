//! `neighbors` — k-nearest-neighbor estimators (NEIGH-01 / NEIGH-02 / NEIGH-03).
//!
//! Module index for the three Phase-5 neighbor estimators. They consume the new
//! Phase-5 top-k select primitive (`prims::topk`, composed on the Phase-2
//! pairwise-distance prim) and expose the
//! [`KNeighbors`](crate::traits::KNeighbors) /
//! [`PredictLabels`](crate::traits::PredictLabels) /
//! [`PredictProba`](crate::traits::PredictProba) surface (D-07):
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

pub mod nearest;
