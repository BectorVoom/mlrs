//! `decomposition` — closed-form matrix decompositions (DECOMP-01 / DECOMP-02).
//!
//! Module index for the two Phase-4 decomposition estimators, both built on the
//! Phase-3 thin SVD primitive + `sign_flip::align_rows` (the estimator applies
//! `svd_flip`; the primitive stays raw — D-01/D-03):
//!
//! - `PCA` (DECOMP-01) — **SVD of CENTERED X** (NOT eig-of-covariance, D-01):
//!   center by column means → thin SVD → `explained_variance_ = S²/(n−1)`,
//!   `components_ = Vᵀ` after flip, `mean_`/`singular_values_`/`transform`/
//!   `inverse_transform`. Added by plan **04-04**.
//! - `TruncatedSVD` (DECOMP-02) — thin SVD of **UNCENTERED X**, deterministic
//!   `algorithm='arpack'` oracle (NOT randomized, D-07);
//!   `explained_variance_ = var(transform columns)`. Added by plan **04-04**.
//!
//! The estimator plan UNCOMMENTS/adds its own `pub mod <estimator>;` line here
//! and creates the matching file; it does NOT edit `lib.rs` (owned by 04-01),
//! keeping the estimator plans file-disjoint and parallel-safe.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub mod pca;
pub mod truncated_svd;
