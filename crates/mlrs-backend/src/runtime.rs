//! Active CubeCL runtime selection by Cargo feature (FOUND-03, Pattern 2).
//!
//! Exactly one backend feature (`cpu` / `wgpu` / `cuda` / `rocm`) must be
//! active; the matching `cfg` block re-exports `ActiveRuntime` / `ActiveDevice`.
//!
//! The `Client` type alias and `active_client()` constructor insulate the rest
//! of the workspace from the concrete `ComputeClient` generic signature
//! (assumption A6, resolved here against cubecl 0.10).

#[cfg(feature = "cpu")]
pub use cubecl::cpu::{CpuDevice as ActiveDevice, CpuRuntime as ActiveRuntime};

#[cfg(feature = "wgpu")]
pub use cubecl::wgpu::{WgpuDevice as ActiveDevice, WgpuRuntime as ActiveRuntime};

#[cfg(feature = "cuda")]
pub use cubecl::cuda::{CudaDevice as ActiveDevice, CudaRuntime as ActiveRuntime};

// cubecl 0.10 re-exports `cubecl_hip` as `cubecl::hip` under `rocm = ["hip"]`;
// there is NO `cubecl::rocm` module and NO `HipDevice` alias. The device struct
// is `AmdDevice` (derives `Default`, so `ActiveDevice::default()` in
// `active_client()` still works). RESEARCH 03 CRITICAL FINDING 2 / Pattern 1.
#[cfg(feature = "rocm")]
pub use cubecl::hip::{AmdDevice as ActiveDevice, HipRuntime as ActiveRuntime};

/// The concrete CubeCL compute client for the active runtime.
///
/// A6 RESOLVED: in cubecl 0.10 `ComputeClient` is generic over a SINGLE
/// `<R: Runtime>` parameter (NOT the `<Server, Channel>` form some older
/// examples used). This alias is the single place that signature is written;
/// downstream code refers to `runtime::Client` and never spells out the
/// generics.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub type Client = cubecl::client::ComputeClient<ActiveRuntime>;

/// Construct a compute client for the active runtime's default device.
///
/// Exactly one backend feature must be enabled; with none, this function is
/// not compiled (and the workspace build would fail to resolve `ActiveRuntime`,
/// enforcing the exactly-one-feature contract — T-01-03).
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn active_client() -> Client {
    use cubecl::Runtime as _;
    let device = ActiveDevice::default();
    ActiveRuntime::client(&device)
}
