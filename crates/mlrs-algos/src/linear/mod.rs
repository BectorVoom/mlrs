//! `linear` — closed-form linear models (LINEAR-01 / LINEAR-02).
//!
//! Module index for the two Phase-4 linear estimators. They deliberately use
//! DIFFERENT solvers and must not be unified (RESEARCH Anti-Patterns):
//!
//! - `LinearRegression` (LINEAR-01) — **SVD pseudo-inverse**
//!   `coef = V·diag(σ⁺)·Uᵀ·y` with sklearn's small-singular-value cutoff,
//!   matching sklearn's default `lstsq` (D-02). Added by plan **04-03**.
//! - `Ridge` (LINEAR-02) — **Cholesky normal-equations**
//!   `(XᵀX + αI)·coef = Xᵀy` via the new Cholesky/solve primitive (D-02). α
//!   never penalizes the intercept (center-then-solve, D-05). Added by plan
//!   **04-05**.
//!
//! The estimator plans UNCOMMENT/add their own `pub mod <estimator>;` line here
//! and create the matching file; they do NOT edit `lib.rs` (owned by 04-01),
//! keeping the estimator plans file-disjoint and parallel-safe.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

// 04-03 adds: pub mod linear_regression;
// 04-05 adds: pub mod ridge;
