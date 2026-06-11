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
