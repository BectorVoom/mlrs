//! `prims` — host-side orchestration for the Phase-2 compute primitives.
//!
//! Each primitive's host API (shape validation, pool-routed scratch/out
//! buffers, kernel launch, device-resident result) lives in its own module
//! here. The device kernels themselves stay in the feature-free `mlrs-kernels`
//! crate (D-13); this layer owns the concrete `ActiveRuntime` and the launch
//! wrappers.
//!
//! Tests live in `crates/mlrs-backend/tests/` (never an in-source
//! `#[cfg(test)]` module — AGENTS.md §2).

pub mod cholesky;
pub mod covariance;
// Phase-5 prim stubs (Wave-0 scaffold owns these registrations; plans
// 05-02..06 fill their own file body — file-disjoint, parallel-safe). Each is an
// empty compiling module until its plan adds the launch wrapper + a `pub use` of
// its symbol INSIDE that file.
pub mod coordinate_descent;
pub mod dbscan;
pub mod distance;
pub mod eig;
pub mod gemm;
pub mod kmeans;
pub mod lbfgs;
pub mod reduce;
pub mod svd;
pub mod topk;
