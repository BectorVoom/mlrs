//! `prims::kmeans` — host orchestration for the KMeans primitive (CLUSTER-01).
//! **Wave-0 stub.**
//!
//! Filled by plan **05-03**: the launch wrapper for k-means++ D² sampling
//! (host RNG) + the Lloyd update loop — composing the Phase-2 distance prim,
//! `prims::reduce::argmin_rows` (host loop over device segments for the nearest-
//! centroid assignment), and the new `mlrs_kernels::kmeans` sum-by-label /
//! inertia kernels. Validates `k` before launch, threads reused buffers (D-11),
//! returns device-resident `cluster_centers_` + `labels_` (i32) + `inertia_`.
//!
//! Until 05-03 fills it this file is an empty (compiling) module body; the Wave-0
//! scaffold owns the `pub mod kmeans;` registration in `prims/mod.rs`, so 05-03
//! only touches THIS file and adds its own `pub use` (file-disjoint,
//! parallel-safe).
//!
//! Tests live in `crates/mlrs-backend/tests/{kmeanspp,lloyd}_test.rs`
//! (AGENTS.md §2).
