//! `topk` — partial select-k kernel (PRIM, D-02). **Wave-0 stub.**
//!
//! Filled by plan **05-02**: a `#[cube]` kernel that, per query row of an
//! `n_queries × n_train` distance matrix, selects the `k` smallest
//! `(value, index)` pairs with the lowest-index tie-break — generalizing
//! `reduce.rs`'s `argmin_shared` (value+index carry) from `k = 1` to `k`. It will
//! write two outputs (`out_val: &mut Array<F>`, `out_idx: &mut Array<u32>`), the
//! host re-uploading the `u32` indices as `i32` (D-06).
//!
//! Until 05-02 fills it this file is an empty (compiling) module body: the
//! Wave-0 scaffold owns the `pub mod topk;` registration in `lib.rs` so 05-02
//! only ever touches THIS file (file-disjoint, parallel-safe). No `#[cube]`
//! kernel yet.
//!
//! Tests live in `crates/mlrs-backend/tests/topk_test.rs` (AGENTS.md §2).
