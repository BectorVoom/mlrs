//! `kmeans` — D² distance + centroid-sum-by-label + inertia kernel (CLUSTER-01).
//! **Wave-0 stub.**
//!
//! Filled by plan **05-03**: `#[cube]` kernels for the Lloyd iteration —
//! per-sample squared distance to each centroid (the k-means++ D² sampling
//! weight, analog `reduce.rs`'s `reduce_sumsq_shared`), the centroid sum-by-label
//! scatter (analog `elementwise.rs`'s `center_columns` `c = tid % cols` per-
//! element map), and the inertia accumulation.
//!
//! Until 05-03 fills it this file is an empty (compiling) module body; the Wave-0
//! scaffold owns the `pub mod kmeans;` registration in `lib.rs`, so 05-03 only
//! touches THIS file (file-disjoint, parallel-safe). No `#[cube]` kernel yet.
//!
//! Tests live in `crates/mlrs-backend/tests/{kmeanspp,lloyd}_test.rs`
//! (AGENTS.md §2).
