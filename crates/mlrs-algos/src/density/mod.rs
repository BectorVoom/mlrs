//! `density` — kernel density estimation (KERNEL-02).
//!
//! Module index for the Phase-8 `KernelDensity` estimator. KD is given its OWN
//! `density/` home rather than living under `neighbors/` (RESEARCH Open Q2): in
//! mlrs's trait sense it is NOT a neighbor estimator — it implements
//! [`ScoreSamples`](crate::typestate::ScoreSamples) (per-sample log-density), not
//! `KNeighbors` / `PredictLabels`.
//!
//! - `KernelDensity` (KERNEL-02) — kernel density estimation. Stores the fitted
//!   training matrix `X_fit_` and the resolved `bandwidth` (numeric or the
//!   `scott`/`silverman` host closed-form, D-09); `score_samples(Q)` composes the
//!   v1 `distance` prim + a SharedMemory-free per-element KD kernel-value map +
//!   a per-query (row) log-sum-exp over the v1 `reduce` prim, finalized with the
//!   host-side per-kernel log-normalization (D-08/D-10/D-11). Six kernels
//!   (gaussian/tophat/epanechnikov/exponential/linear/cosine). Added by plan
//!   **08-04**.
//!
//! The estimator plan ADDS its own `pub mod <estimator>;` line here and creates
//! the matching file; it does NOT edit `lib.rs` (owned by the 08-01 Wave-0
//! scaffold), keeping the estimator plans file-disjoint and parallel-safe. This
//! is the 04-01 / 05-01 / 07-01 stub precedent: an empty doc-comment-only module
//! body is a valid compiling stub (Wave-2 fills the `pub mod` + `pub use`).
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2 — no in-source
//! `#[cfg(test)] mod tests`).

// Phase-8 kernel density estimator (plan 08-04 — file-disjoint): the Wave-2 plan
// adds its `pub mod` + re-export here and creates the matching estimator file; it
// does NOT edit `lib.rs` (owned by the 08-01 Wave-0 scaffold).
pub mod kernel_density;
pub use kernel_density::{BandwidthSpec, KdKernel, KernelDensity};
