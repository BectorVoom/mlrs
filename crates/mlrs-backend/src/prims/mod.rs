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

pub mod covariance;
pub mod distance;
pub mod gemm;
pub mod reduce;
