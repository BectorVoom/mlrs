//! f64-on-incapable-backend guard (D-04) + the `backend_supports_f64` flag (D-05).
//!
//! This module wraps the backend capability layer
//! ([`mlrs_backend::capability::feature_enabled`]) so the binding surface never
//! re-discovers the cubecl `supports_type(FloatKind::F64)` query, and turns the
//! capability *query* into a hard **`PyValueError`** when float64 data is handed
//! to a backend that cannot run it — D-04: never a silent downcast (the 1e-5
//! contract trumps convenience).
//!
//! It also exposes [`supports_f64`] as the boolean the Python shim and pytest
//! consume (via the module-level `backend_supports_f64()` registered in `lib.rs`)
//! to pick the default dtype (D-05) and to `@pytest.mark.skipif` f64 cases on an
//! f64-incapable backend (e.g. `mlrs-rocm`).

use mlrs_backend::capability::{active_backend_name, feature_enabled, FloatKind};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Does the active backend support float64 compute?
///
/// Thin, stable wrapper over `mlrs_backend::capability::feature_enabled(F64)` so
/// the binding layer never spells out the underlying cubecl `supports_type`
/// query. Surfaced to Python as `mlrs._mlrs.backend_supports_f64()` (registered
/// in `lib.rs`) so the shim can choose the default dtype (D-05) and pytest can
/// skip f64 on an incapable backend.
pub fn supports_f64() -> bool {
    feature_enabled(FloatKind::F64)
}

/// Does the active backend evaluate f64 TRANSCENDENTALS (`exp`/`log`/`powf`/
/// `tanh`) in device code?
///
/// A strictly narrower question than [`supports_f64`], and the two genuinely
/// differ: **wgpu** reports f64 support (its adapter has the extension) but WGSL
/// has no `f64` type, so its transcendental coverage is the driver's business
/// and mlrs treats it as absent — see
/// `mlrs_backend::capability::f64_transcendental_supported`. A prim that needs
/// one (`metric_distance(minkowski)`, `knn_graph(minkowski)`) rejects the launch
/// there rather than returning wrong numbers.
///
/// Surfaced to Python as `mlrs._mlrs.backend_supports_f64_transcendental()` so
/// the oracle suite can skip exactly those cells — `backend_supports_f64` alone
/// is the wrong gate for them, and gating on it is what left two f64 Minkowski
/// cases FAILING on wgpu instead of skipping.
pub fn supports_f64_transcendental() -> bool {
    mlrs_backend::capability::f64_transcendental_supported()
}

/// Guard an f64 compute path against an f64-incapable backend (D-04).
///
/// Returns `Ok(())` when the active backend supports float64; otherwise returns
/// a clear `PyValueError` naming the backend and pointing at the f64-capable
/// `mlrs-cpu` wheel. This is called on the f64 ingress arm *before* any device
/// compute, so f64 input to (e.g.) `mlrs-rocm` raises a recognizable Python
/// error instead of being silently downcast to f32.
pub fn guard_f64() -> PyResult<()> {
    if supports_f64() {
        return Ok(());
    }
    let backend = active_backend_name();
    Err(PyValueError::new_err(format!(
        "backend '{backend}' does not support float64 — pass float32 or install mlrs-cpu"
    )))
}
