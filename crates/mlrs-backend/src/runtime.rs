//! Active CubeCL runtime selection by Cargo feature (FOUND-03, Pattern 2).
//!
//! Exactly one backend feature (`cpu` / `wgpu` / `cuda` / `rocm`) must be
//! active; the matching `cfg` block re-exports `ActiveRuntime` / `ActiveDevice`.
//!
//! Wave 0 (Plan 01) Task 1 stands up the feature-gated re-exports; the
//! `active_client()` constructor and the `Client` type alias that insulate the
//! `ComputeClient` signature (assumption A6) land in Task 2.

#[cfg(feature = "cpu")]
pub use cubecl::cpu::{CpuDevice as ActiveDevice, CpuRuntime as ActiveRuntime};

#[cfg(feature = "wgpu")]
pub use cubecl::wgpu::{WgpuDevice as ActiveDevice, WgpuRuntime as ActiveRuntime};

#[cfg(feature = "cuda")]
pub use cubecl::cuda::{CudaDevice as ActiveDevice, CudaRuntime as ActiveRuntime};

#[cfg(feature = "rocm")]
pub use cubecl::rocm::{RocmDevice as ActiveDevice, RocmRuntime as ActiveRuntime};
