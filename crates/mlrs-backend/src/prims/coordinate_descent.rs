//! `prims::coordinate_descent` — host orchestration for the coordinate-descent
//! primitive (LINEAR-03/04). **Wave-0 stub.**
//!
//! Filled by plan **05-05**: the host-driven CD loop — acquire the solver buffers
//! (`R`, `norm2_cols`, `w`) ONCE before the loop and reuse every iteration,
//! launch the new `mlrs_kernels::coordinate` soft-threshold + residual-axpy
//! kernel per coordinate, and read back exactly ONE scalar (duality gap) per
//! outer convergence check (D-10 iterative-solver memory exception; the
//! `prims::cholesky` validate→launch→scalar-readback shape). Convergence
//! constants pinned from RESEARCH (`tol·‖y‖²`, `max_iter = 1000`).
//!
//! Until 05-05 fills it this file is an empty (compiling) module body; the Wave-0
//! scaffold owns the `pub mod coordinate_descent;` registration in
//! `prims/mod.rs`, so 05-05 only touches THIS file and adds its own `pub use`
//! (file-disjoint, parallel-safe).
//!
//! Tests live in `crates/mlrs-backend/tests/cd_test.rs` (AGENTS.md §2).
