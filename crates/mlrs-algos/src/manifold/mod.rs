//! `manifold` — manifold-learning / dimensionality-reduction estimators.
//!
//! Module index for the Phase-12 convention-foundation manifold shell. The home
//! for the v3 builder + typestate UMAP shell (UMAP-01):
//!
//! - `Umap` (UMAP-01) — born builder-fronted (`Umap::builder().build::<F>()?`)
//!   with the full sklearn/umap-learn hyperparameter surface, a typestate
//!   `<F, S = Unfit>` shape, a NON-algorithmic trivial fit (zeros `embedding_`),
//!   and fitted accessors gated on `Umap<F, Fitted>`. The real UMAP algorithm
//!   lands in Phase 14; this is the convention-demonstration shell only.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2 — never an in-source
//! `#[cfg(test)] mod tests`).

pub mod umap;

pub use umap::Umap;
