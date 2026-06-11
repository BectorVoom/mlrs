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

/// Query whether the given client's backend supports plane (subgroup) ops.
///
/// Mirrors [`supports_type`] but for the plane/subgroup capability. cubecl 0.10
/// exposes this via `client.features().plane` — an `EnumSet<Plane>` of the
/// supported plane operations. We report support when the basic plane-op set
/// (`Plane::Ops`) is present, which is the prerequisite for the reduction
/// plane-path (Plan 02). The plane width is separately available via
/// `client.properties().hardware.plane_size_{min,max}`.
///
/// A3 RESOLVED (subgroup-query symbol): the stable query is
/// `client.features().plane.contains(Plane::Ops)` (NOT a `feature_enabled`
/// form). Downstream plane-path code calls this facade and never re-discovers
/// the symbol.
pub fn supports_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
    use cubecl::ir::features::Plane;
    client.features().plane.contains(Plane::Ops)
}

/// Stable facade over the active runtime's plane/subgroup capability.
///
/// Constructs a client for the active runtime's default device and reports
/// whether plane (subgroup) ops are supported. Plan 02's reduction plane-path
/// skip calls this and never spells out the underlying `features().plane`
/// query.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn plane_supported() -> bool {
    supports_plane(&crate::runtime::active_client())
}

/// Active runtime's plane (subgroup) width, used to size the plane-path
/// reduction's per-(cube, plane) partial output (Plan 02).
///
/// Reports `client.properties().hardware.plane_size_max` — the upper plane
/// size the adapter advertises (CUDA warp = 32; wgpu subgroups vary 4..128).
/// When the adapter reports no plane support the value may be `0`; callers
/// clamp to at least `1`. The min/max symbols were pinned in Plan 02-01
/// (`spike_subgroup_query_reports_support`).
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn active_plane_width() -> u32 {
    crate::runtime::active_client()
        .properties()
        .hardware
        .plane_size_max
}

/// Static name of the active backend, derived from the compiled-in Cargo
/// feature (FOUND-03: exactly one backend feature is active).
///
/// Used in the dtype×backend oracle log line (Criterion 4) so CI output shows
/// which backend a given oracle run executed on.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub const fn active_backend_name() -> &'static str {
    #[cfg(feature = "cpu")]
    {
        "cpu"
    }
    #[cfg(feature = "wgpu")]
    {
        "wgpu"
    }
    #[cfg(feature = "cuda")]
    {
        "cuda"
    }
    #[cfg(feature = "rocm")]
    {
        "rocm"
    }
}

/// Emit the canonical oracle dtype×backend log line at the start of an oracle
/// test (Criterion 4: "CI log shows which dtype ran on which backend").
///
/// Logs at `info` level. `adapter` is a free-form adapter/device descriptor
/// (e.g. the wgpu adapter name, or `"default"` for cpu) so the line is
/// self-describing in CI output.
pub fn log_oracle_dtype(dtype: FloatKind, backend: &str, adapter: &str) {
    log::info!("oracle dtype={dtype:?} backend={backend} adapter={adapter}");
}

/// f64 skip-with-log gate (FOUND-04, T-03-04). Returns `true` when the f64 path
/// should be **skipped** because the active backend lacks f64 support, after
/// logging the reason at `warn` level. Returns `false` when f64 is supported and
/// the caller should proceed.
///
/// This is the chosen skip/xfail mechanism (logged early-return — Claude's
/// discretion per CONTEXT D-06/FOUND-04): an f64-gated oracle test calls this
/// and `return`s early when it reports `true`, so the run is **skipped, not
/// failed**, and CI shows the logged reason. On this environment's wgpu adapter
/// (AMD RADV GFX1152, `SHADER_F64` present) it returns `false` and f64 runs.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn skip_f64_with_log() -> bool {
    if feature_enabled(FloatKind::F64) {
        return false;
    }
    let backend = active_backend_name();
    log::warn!("skipping f64 oracle on {backend}: SHADER_F64 / f64 unsupported on this adapter");
    true
}
