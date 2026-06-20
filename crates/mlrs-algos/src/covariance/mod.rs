//! `covariance` — covariance-matrix estimators (COV-01 / COV-02).
//!
//! Module index for the two Phase-7 covariance estimators, both built on the
//! Phase-2 `covariance`/Gram primitive (with `ddof = 0` — the empirical /
//! biased estimator, RESEARCH Pitfall 1) plus a host-side finalize:
//!
//! - `EmpiricalCovariance` (COV-01) — `covariance_ = AᵀA / n` of the centered
//!   data (`ddof = 0`); `location_` = column means (or `0` when
//!   `assume_centered`); `precision_` = `pinvh(covariance_)` via the Phase-3
//!   symmetric `eig` (NOT Cholesky — must tolerate a singular/rank-deficient
//!   covariance, D-05). Added by plan **07-04**.
//! - `LedoitWolf` (COV-02) — the Ledoit–Wolf shrinkage estimator: the same
//!   empirical `covariance_` shrunk toward a scaled-identity target by the
//!   closed-form optimal `shrinkage_ ∈ [0, 1]` (RESEARCH Pattern 3). Added by
//!   plan **07-04**.
//!
//! The estimator plan ADDS its own `pub mod <estimator>;` line here and creates
//! the matching file; it does NOT edit `lib.rs` (owned by the 07-01 Wave-0
//! scaffold), keeping the estimator plans file-disjoint and parallel-safe. This
//! is the 04-01 / 05-01 stub precedent: an empty doc-comment-only module body is
//! a valid compiling stub.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2 — no in-source
//! `#[cfg(test)] mod tests`).

// Phase-7 covariance estimators (filled by plan 07-04 — file-disjoint):
// pub mod empirical_covariance; // EmpiricalCovariance (COV-01), plan 07-04
// pub mod ledoit_wolf;          // LedoitWolf (COV-02), plan 07-04
