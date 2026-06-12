//! `dbscan` — eps-threshold + per-row core-count mask kernel (CLUSTER-02).
//! **Wave-0 stub.**
//!
//! Filled by plan **05-04**: a `#[cube]` kernel over the `n × n` pairwise
//! squared-distance matrix that thresholds `D[i,j] <= eps²` and counts each row's
//! eps-neighborhood (self included) to mark core points — the 2D `(i, j)` map
//! shape of `elementwise.rs`'s `dist_combine_clamp` with an
//! `if i < rows && j < cols` bounds-check. The host walks the resulting core mask
//! + adjacency with an index-ordered DFS (D-04 documented readback).
//!
//! Until 05-04 fills it this file is an empty (compiling) module body; the Wave-0
//! scaffold owns the `pub mod dbscan;` registration in `lib.rs`, so 05-04 only
//! touches THIS file (file-disjoint, parallel-safe). No `#[cube]` kernel yet.
//!
//! Tests live in `crates/mlrs-backend/tests/dbscan_mask_test.rs` (AGENTS.md §2).
