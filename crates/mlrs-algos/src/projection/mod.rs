//! `projection` — random-projection transformers (PROJ-01 / PROJ-02).
//!
//! Module index for the two Phase-7 random-projection estimators, both built on
//! the new Phase-7 `prims::rng` matrix generator + the Phase-2 `gemm`
//! primitive (`transform == X · componentsᵀ`, the same single GEMM as PCA —
//! RandomProjection does NOT center, D-12):
//!
//! - `GaussianRandomProjection` (PROJ-01) — a dense projection matrix drawn
//!   `N(0, 1/n_components)`; the JL `n_components='auto'` path sizes it via
//!   `johnson_lindenstrauss_min_dim`. Added by plan **07-06**.
//! - `SparseRandomProjection` (PROJ-02) — an Achlioptas sparse projection matrix
//!   (density `∈ (0, 1]`, value `±sqrt((1/density)/n_components)`) stored DENSE
//!   even when sparse (D-12). Added by plan **07-06**.
//!
//! The estimator gate is a STRUCTURAL PROPERTY set, NOT the 1e-5 oracle (D-12 —
//! the RNG is SplitMix64, not MT19937, so the matrix cannot match sklearn
//! element-wise); only `johnson_lindenstrauss_min_dim` is value-matched.
//!
//! The estimator plan ADDS its own `pub mod <estimator>;` line here and creates
//! the matching file; it does NOT edit `lib.rs` (owned by the 07-01 Wave-0
//! scaffold), keeping the estimator plans file-disjoint and parallel-safe. An
//! empty doc-comment-only module body is a valid compiling stub (04-01 / 05-01
//! precedent).
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2 — no in-source
//! `#[cfg(test)] mod tests`).

/// The `mlrs-proj` model-file container — both projections' `save`/`load` share
/// the single-matrix layout it defines (PROJ-PERSIST, prototype).
pub mod proj_persist;

// Phase-7 random-projection estimators (plan 07-06 — file-disjoint):
pub mod gaussian; // GaussianRandomProjection (PROJ-01) + johnson_lindenstrauss_min_dim
pub mod sparse; // SparseRandomProjection (PROJ-02, Achlioptas dense)

pub use gaussian::{
    johnson_lindenstrauss_min_dim, GaussianRandomProjection, NComponents,
};
pub use sparse::SparseRandomProjection;
