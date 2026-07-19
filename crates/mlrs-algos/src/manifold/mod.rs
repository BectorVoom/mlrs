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

// TSNE-01 — exact-method t-SNE (device Q/gradient prim + sklearn-exact host
// P-matrix / gradient-descent ports).
pub mod tsne;
pub mod umap;

// Plan-02/03 homes, pre-declared EMPTY in Plan 14-01 so the two Wave-2 plans
// fill their own file WITHOUT both editing this `mod.rs` (file-disjoint,
// parallel-safe). `umap_internals` = host numerics (smooth-kNN/membership/union,
// + transform helper in Plan 05); `umap_init` = a/b LM fit + spectral/random
// init. `umap_init` stays `pub(crate)` (Plan 03 owns it); `umap_internals` is
// `pub` so the Plan-02 value-gate in `tests/umap_test.rs` can reach the host
// stage fns directly (the plan key_link `umap_test.rs → umap_internals::*`):
// an integration test is an external crate and cannot see `pub(crate)` items.
pub mod umap_init;
pub mod umap_internals;

pub use tsne::Tsne;
pub use umap::Umap;
