//! Matrix decompositions and host routines for sparse and non-negative models.
//!
//! PCA and TruncatedSVD both use a thin SVD followed by deterministic sign
//! alignment in the estimator:
//!
//! - `PCA` (DECOMP-01) — **SVD of CENTERED X** (NOT eig-of-covariance, D-01):
//!   center by column means → thin SVD → `explained_variance_ = S²/(n−1)`,
//!   `components_ = Vᵀ` after flip, `mean_`/`singular_values_`/`transform`/
//!   `inverse_transform`. Added by plan **04-04**.
//! - `TruncatedSVD` (DECOMP-02) — thin SVD of **UNCENTERED X**, deterministic
//!   `algorithm='arpack'` oracle (NOT randomized, D-07);
//!   `explained_variance_ = var(transform columns)`. Added by plan **04-04**.
//!
//! The host routines also provide coordinate descent for non-negative
//! factorization, sparse coding for dictionary learning, and variational
//! document inference for latent Dirichlet allocation.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub mod coordinate_descent;
/// The `mlrs-decomp` model-file container — the three estimators' `save`/`load`
/// share the spectral core it defines (DECOMP-PERSIST, prototype).
pub mod decomp_persist;

pub mod incremental_pca;
pub mod lda;
pub mod pca;
pub mod sparse_coding;
pub mod truncated_svd;

pub use incremental_pca::IncrementalPCA;
pub use pca::Pca;
pub use truncated_svd::TruncatedSvd;
