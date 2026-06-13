//! `cluster` — distance-based clustering estimators (CLUSTER-01 / CLUSTER-02).
//!
//! Module index for the two Phase-5 clustering estimators. They consume the new
//! Phase-5 distance/clustering primitives (`prims::kmeans`, `prims::dbscan`) and
//! return integer labels via the [`PredictLabels`](crate::traits::PredictLabels)
//! surface (D-05/D-06):
//!
//! - `KMeans` (CLUSTER-01) — k-means++ init (injected for the oracle, D-09) +
//!   Lloyd updates; stores `cluster_centers_` (F), `labels_`/`inertia_`. Up to a
//!   label permutation vs sklearn (D-09). Added by plan **05-07**.
//! - `DBSCAN` (CLUSTER-02) — eps-neighborhood core mask + host DFS expansion;
//!   stores `labels_` (noise = `-1`) and `core_sample_indices_` (i32). Added by
//!   plan **05-08**.
//!
//! Each estimator plan ADDS its own `pub mod <estimator>;` line here and creates
//! the matching file; the plans do NOT edit `lib.rs` (owned by the Wave-0
//! scaffold), keeping the estimator plans file-disjoint and parallel-safe.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub mod kmeans;
