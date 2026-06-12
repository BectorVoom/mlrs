//! `coordinate` — coordinate-descent soft-threshold + residual axpy kernel
//! (LINEAR-03/04). **Wave-0 stub.**
//!
//! Filled by plan **05-05**: `#[cube]` kernels for one coordinate update of the
//! Lasso / ElasticNet coordinate descent — the soft-threshold
//! `w_j = sign(t)·max(|t| − l1_reg, 0) / (norm2_cols[j] + l2_reg)` (un-normalized
//! form, `l1_reg = α·l1_ratio·n`) and the residual axpy
//! `R += (w_j_old − w_j)·X[:,j]` (analog `elementwise.rs`'s `scale` scalar-`F`
//! per-element map; the column dot reuses `reduce.rs`). The outer host loop
//! reads back exactly one scalar (duality gap) per iteration (D-10).
//!
//! Until 05-05 fills it this file is an empty (compiling) module body; the Wave-0
//! scaffold owns the `pub mod coordinate;` registration in `lib.rs`, so 05-05
//! only touches THIS file (file-disjoint, parallel-safe). No `#[cube]` kernel
//! yet.
//!
//! Tests live in `crates/mlrs-backend/tests/cd_test.rs` (AGENTS.md §2).
