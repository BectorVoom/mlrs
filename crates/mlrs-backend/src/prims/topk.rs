//! `prims::topk` — host orchestration for the top-k select primitive (PRIM,
//! D-02). **Wave-0 stub.**
//!
//! Filled by plan **05-02**: the launch wrapper that composes the Phase-2
//! pairwise-distance prim with the new `mlrs_kernels::topk` select kernel —
//! validating geometry before any `unsafe` launch (ASVS V5), threading an
//! optional reused `out` buffer (D-11), and returning device-resident
//! `(distances, indices)` per query row (the `prims::distance` precedent). The
//! `u32` neighbor indices are re-uploaded as `i32` (D-06).
//!
//! Until 05-02 fills it this file is an empty (compiling) module body; the Wave-0
//! scaffold owns the `pub mod topk;` registration in `prims/mod.rs`, so 05-02
//! only touches THIS file and adds its own `pub use` of the symbols it creates
//! (file-disjoint, parallel-safe).
//!
//! Tests live in `crates/mlrs-backend/tests/topk_test.rs` (AGENTS.md §2).
