//! Runtime capability gating (FOUND-04).
//!
//! Resolves the f64 capability query against cubecl 0.10 and exposes a stable
//! facade so downstream call sites (Plan 03, Plan 05) never re-discover the
//! symbol.
//!
//! ## A1 RESOLVED (capability-query symbol)
//! cubecl 0.10 has NO `feature_enabled(Feature::Type(Elem::Float(..)))` form —
//! the `Feature` enum from older examples does not exist in this layout. The
//! real query is:
//!
//! ```ignore
//! client.properties().supports_type(FloatKind::F64)
//! ```
//!
//! `DeviceProperties::supports_type` takes `impl Into<Type>`, and
//! `FloatKind -> ElemType -> StorageType -> Type` conversions are provided, so
//! `FloatKind::F64` is accepted directly.
//!
//! ## A2 RESOLVED (wgpu SHADER_F64)
//! On the wgpu adapter in this environment (AMD Radeon RADV GFX1152, Vulkan)
//! the adapter feature set includes `SHADER_F64`, and `supports_type` returns
//! `true` for f64 on wgpu here. f64 oracle tests therefore RUN (not skip) on
//! this machine; the skip path still exists for adapters lacking it.

pub use cubecl::ir::FloatKind;

use cubecl::Runtime;
use cubecl::client::ComputeClient;

/// Query whether the given client's backend supports a given float type.
///
/// Generic over the runtime so it works for any backend (cpu / wgpu / cuda /
/// rocm) without naming a concrete client type.
pub fn supports_type<R: Runtime>(client: &ComputeClient<R>, kind: FloatKind) -> bool {
    client.properties().supports_type(kind)
}

/// Convenience: does the client's backend support f64?
pub fn supports_f64<R: Runtime>(client: &ComputeClient<R>) -> bool {
    supports_type(client, FloatKind::F64)
}

/// Stable facade over the active runtime's f64 capability (FOUND-04 wording).
///
/// Constructs a client for the active runtime's default device and reports
/// whether the requested float type is supported. Downstream code (Plan 03's
/// skip/xfail gate, Plan 05's oracle dtype logging) calls this and never spells
/// out the underlying `supports_type` query.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn feature_enabled(kind: FloatKind) -> bool {
    let client = crate::runtime::active_client();
    client.properties().supports_type(kind)
}
