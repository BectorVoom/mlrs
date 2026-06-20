//! `kernel_ridge` — kernel ridge regression (KERNEL-01).
//!
//! Module index for the Phase-8 `KernelRidge` estimator, built on the new
//! Phase-8 `kernel_matrix` primitive (`mlrs-backend/src/prims/kernel_matrix.rs`)
//! plus the v1 Phase-4 Cholesky solve:
//!
//! - `KernelRidge` (KERNEL-01) — kernel ridge regression. Fits the dual
//!   coefficients by a multi-RHS Cholesky solve of `(K + αI)·dual_coef_ = y`
//!   where `K = kernel_matrix(X, X, kernel)` is the `n×n` training Gram (D-02);
//!   predicts `y = kernel_matrix(X_test, X_fit_, kernel) · dual_coef_`. Unlike
//!   v1 `Ridge`, KernelRidge fits RAW data with NO centering and NO intercept
//!   (sklearn KernelRidge — D-06 / RESEARCH Pitfall 1). `gamma=None` resolves to
//!   `1/n_features` at `fit` (D-05). Added by plan **08-03**.
//!
//! The estimator plan ADDS its own `pub mod <estimator>;` line here and creates
//! the matching file; it does NOT edit `lib.rs` (owned by the 08-01 Wave-0
//! scaffold), keeping the estimator plans file-disjoint and parallel-safe. This
//! is the 04-01 / 05-01 / 07-01 stub precedent: an empty doc-comment-only module
//! body is a valid compiling stub (Wave-2 fills the `pub mod` + `pub use`).
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2 — no in-source
//! `#[cfg(test)] mod tests`).

// Phase-8 kernel ridge estimator (plan 08-03 — file-disjoint): this module
// owns its own `pub mod` + `pub use`; it does NOT edit lib.rs (owned by the
// 08-01 Wave-0 scaffold).
pub mod kernel_ridge;
pub use kernel_ridge::{KernelKind, KernelRidge};
