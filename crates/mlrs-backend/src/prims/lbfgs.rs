//! `prims::lbfgs` — host orchestration for the L-BFGS primitive (LINEAR-05).
//! **Wave-0 stub.**
//!
//! Filled by plan **05-06**: the host-driven L-BFGS two-loop recursion + strong-
//! Wolfe line search — acquire the gradient + history `(s, y) × m` buffers ONCE
//! and reuse, launch the new `mlrs_kernels::lbfgs` stable-softmax loss/grad
//! kernel (plus `prims::gemm` for `Xw`), and read back exactly ONE scalar (max
//! projected gradient) per outer iteration (D-10; the `prims::cholesky`
//! new-primitive wrapper shape). Constants pinned from RESEARCH (`m = 10`,
//! `gtol = 1e-4`, `ftol = 64·eps`, `maxiter = 100`). HIGHEST project risk —
//! validated standalone on a convex quadratic FIRST.
//!
//! Until 05-06 fills it this file is an empty (compiling) module body; the Wave-0
//! scaffold owns the `pub mod lbfgs;` registration in `prims/mod.rs`, so 05-06
//! only touches THIS file and adds its own `pub use` (file-disjoint,
//! parallel-safe).
//!
//! Tests live in `crates/mlrs-backend/tests/lbfgs_test.rs` (AGENTS.md §2).
