//! `prims::dbscan` — host orchestration for the DBSCAN primitive (CLUSTER-02).
//! **Wave-0 stub.**
//!
//! Filled by plan **05-04**: the launch wrapper that builds the `n × n` pairwise
//! distance matrix (Phase-2 distance prim), launches the new
//! `mlrs_kernels::dbscan` eps-threshold + per-row core-count kernel, then reads
//! the core mask + adjacency back to host (the `prims::cholesky` tiny-readback
//! idiom; D-04 documents this DELIBERATE readback) for the index-ordered DFS
//! cluster expansion. Validates `eps`/`min_samples` before launch; the n²
//! distance matrix is the dominant, reused allocation (memory-gate bound, D-04).
//!
//! Until 05-04 fills it this file is an empty (compiling) module body; the Wave-0
//! scaffold owns the `pub mod dbscan;` registration in `prims/mod.rs`, so 05-04
//! only touches THIS file and adds its own `pub use` (file-disjoint,
//! parallel-safe).
//!
//! Tests live in `crates/mlrs-backend/tests/dbscan_mask_test.rs` (AGENTS.md §2).
