//! `mlrs-py` — PyO3 binding layer for mlrs (cdylib).
//!
//! This crate owns the process-wide `#[global_allocator]` (FOUND-09): mimalloc
//! is wired exactly once in [`allocator`], the single cdylib artifact, and never
//! in any library crate. The allocator activation proof lives in the separate
//! test file `crates/mlrs-py/tests/allocator_test.rs` (AGENTS.md §2 — no
//! in-source test module).
//!
//! ## Module surface
//! The `#[pymodule] _mlrs` (defined here) exposes the low-level binding the
//! pure-Python `mlrs` shim delegates to. This plan (06-02) builds the shared
//! primitives every `#[pyclass]` wrapper (Plan 03) consumes:
//!
//! - [`ingress`] — owned Arrow PyCapsule import → the *unchanged*
//!   `mlrs_backend::bridge` validation → a pooled `DeviceArray` (PY-03 / D-02).
//! - [`egress`] — device→host `Vec<F>`/`Vec<i32>` + shape (numpy/arrow wrap is
//!   shim-side, D-03).
//! - [`capability`] — `backend_supports_f64()` flag + the f64-on-incapable-backend
//!   guard (D-04 / D-05).
//! - [`errors`] — boundary `BridgeError`/`AlgoError`/`anyhow` → `PyErr` mapping.
//! - [`dispatch`] — the `any_estimator!` dtype-dispatch macro (D-06) Plan 03
//!   invokes per estimator.
//!
//! ## Concurrency model (Claude's-discretion, RESEARCH §BufferPool Lifecycle)
//! A SINGLE process-global [`BufferPool`] + cubecl client lives behind a
//! [`Mutex`] ([`global_pool`]). Under `Python::detach` two Python threads may both
//! try to compute; the mutex serializes device access — correct, and matches the
//! reality that a single device is one compute queue. This means mlrs does NOT
//! give intra-process GPU parallelism across estimators in v1 (joblib `n_jobs>1`
//! over mlrs estimators serializes on the device mutex). This is the accepted v1
//! single-device semantics; true parallelism (per-thread clients/streams) is out
//! of v1 scope.
//!
//! ## Import-time driver probe (D-08)
//! `mlrs_backend::runtime::active_client()` calls cubecl `Runtime::client`, which
//! returns `ComputeClient` directly and `.unwrap()`s internally — a missing /
//! incompatible driver **panics**, it does not return an `Err`. The `#[pymodule]`
//! init wraps the probe in [`std::panic::catch_unwind`] and translates a caught
//! panic into a clean [`PyImportError`], so `import mlrs` on a driver-less machine
//! raises a Python `ImportError` instead of aborting the CPython process
//! (T-06-05). The wheel profile keeps `panic = "unwind"` (no `panic = "abort"`),
//! which `catch_unwind` requires.

use std::sync::{Mutex, OnceLock};

use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{active_client, ActiveRuntime};
use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;

// The `#[global_allocator]` definition. Source-only; its activation test is in
// `tests/allocator_test.rs` (FOUND-09: source/test separation).
mod allocator;

// The shared binding primitives the `#[pyclass]` wrappers (Plan 03) consume.
pub mod capability;
pub mod dispatch;
pub mod egress;
pub mod errors;
pub mod ingress;

// The 12 estimator `#[pyclass]` wrappers (Plan 03). Registered on `_mlrs` below.
pub mod estimators;

/// Boundary errors use `anyhow` (D-10); this alias documents the boundary error
/// convention the binding surface uses (mapped to `PyErr` in [`errors`]).
pub type BoundaryResult<T> = anyhow::Result<T>;

/// The single process-global buffer pool + cubecl client, behind a mutex.
///
/// Initialized lazily on first [`global_pool`] call (only ever reached AFTER the
/// import-time probe in [`_mlrs`] has confirmed the driver is present, so the
/// `active_client()` here cannot be the panicking-on-missing-driver call). See
/// the module-level concurrency note.
///
/// `allow(dead_code)`: the only consumers are the Plan-03 `#[pyclass]` wrappers'
/// `fit`/`predict` bodies (locked inside `Python::detach`); this plan delivers
/// the pool + accessor for them to consume. Removing the allow once Plan 03 lands.
#[allow(dead_code)]
static GLOBAL_POOL: OnceLock<Mutex<BufferPool<ActiveRuntime>>> = OnceLock::new();

/// Borrow the process-global `Mutex<BufferPool>` (D-04 / FOUND-05).
///
/// The `#[pyclass]` wrappers (Plan 03) lock this inside `Python::detach` to run
/// device compute with the GIL released. A single shared pool maximizes buffer
/// reuse (the FOUND-05 memory-gate invariant); the mutex makes the module-global
/// access sound and gives the `detach` closure body its required `Send`.
///
/// `allow(dead_code)`: consumed by the Plan-03 wrapper `fit`/`predict` bodies;
/// delivered here as the shared primitive they lock inside `Python::detach`.
#[allow(dead_code)]
pub(crate) fn global_pool() -> &'static Mutex<BufferPool<ActiveRuntime>> {
    GLOBAL_POOL.get_or_init(|| Mutex::new(BufferPool::new(active_client())))
}

/// Lock the process-global pool, RECOVERING from mutex poisoning (WR-02).
///
/// A device fault / OOM / unsupported-op panic inside a `py.detach` closure that
/// holds the [`global_pool`] guard would otherwise POISON the mutex, after which
/// every plain `.lock().expect("pool mutex")` in the whole module panics —
/// converting one recoverable device error into a permanent process-wide brick
/// (a robustness/DoS-class regression the infinity-free/SharedMemory-free cpu-MLIR
/// kernels make more likely to surface). The pool data is NOT left torn by a
/// panicked compute call (the cubecl handles are ref-counted and the panic
/// unwinds before any half-written pool mutation), so `into_inner()` recovery is
/// safe: a single bad `fit` no longer kills the interpreter session.
///
/// ## This is the SANCTIONED lock path (WR-04)
/// `lock_pool` is the single authoritative lock path for the binding layer. The
/// poison recovery only delivers its benefit if EVERY lock site uses it: one
/// surviving `global_pool().lock().expect("pool mutex")` re-panics on a poisoned
/// mutex and re-bricks the interpreter, making the brick-prevention only partial.
/// The spectral wrappers ([`crate::estimators::spectral`]) and the kernel wrappers
/// ([`crate::estimators::kernel`]) use `lock_pool` exclusively; new estimators MUST
/// do the same. (The remaining `linear`/`cluster`/`decomposition`/`covariance`/
/// `neighbors`/`projection` wrappers still carry the legacy panicking form — a
/// pre-existing, tracked migration; mixing the two helpers defeats the recovery on
/// those estimators until they are converted.)
///
/// ## ACCOUNTING CAVEAT after a recovered poison (WR-01)
/// "Not left torn" is a **memory-safety** statement, NOT an accounting one. The
/// `BufferPool` counters (`live_bytes`/`peak_bytes`) are bumped at `acquire` and
/// decremented at `release` ([`BufferPool::acquire`]/[`BufferPool::release`]). A
/// `fit` acquires many buffers and releases them incrementally; if a panic unwinds
/// the `py.detach` closure mid-`fit` while this guard is held, every
/// acquired-but-not-yet-released buffer leaves its bytes permanently added to
/// `live_bytes`. After `into_inner()` recovery, `live_bytes` (and therefore
/// `peak_bytes`) may be monotonically INFLATED for the rest of the process.
///
/// The pool cannot self-reconcile this: its free-list only knows about *released*
/// handles, and the *live* handles are owned by `DeviceArray`s outside the pool —
/// so there is no in-pool truth source to recompute `live_bytes` from. Callers and
/// the FOUND-05 leak-detection gates MUST therefore treat the conservation
/// property (`live_bytes` returns to its pre-`fit` baseline) as VOID once a poison
/// has been recovered through this path. Memory safety holds; the accounting
/// counters do not.
#[allow(dead_code)]
pub(crate) fn lock_pool(
) -> std::sync::MutexGuard<'static, BufferPool<ActiveRuntime>> {
    match global_pool().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Is float64 compute supported on the backend this wheel was built for? (D-05)
///
/// Exposed to Python as `mlrs._mlrs.backend_supports_f64()` so the pure-Python
/// shim picks the default dtype (f32 on an f64-incapable backend) and pytest can
/// `@pytest.mark.skipif(not mlrs._mlrs.backend_supports_f64())` the f64 oracle
/// cases (D-05 / RESEARCH §pytest harness).
#[pyfunction]
fn backend_supports_f64() -> bool {
    capability::supports_f64()
}

/// The `_mlrs` extension module (PyO3 `abi3-py312`), a submodule of the
/// pure-Python `mlrs` package (`module-name = "mlrs._mlrs"`).
///
/// At import it runs the D-08 driver probe and, on success, registers the
/// low-level surface. Plan 03 registers the 12 estimator `#[pyclass]`es here.
#[pymodule]
fn _mlrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // D-08 / T-06-05: probe the driver ONCE at import. cubecl's `client()`
    // `.unwrap()`s internally, so a missing driver PANICS — `catch_unwind` turns
    // that into a clean `ImportError` instead of a process abort. We touch
    // `properties()` to force the device handshake (a lazy client would not fail
    // until first use otherwise).
    let probe = std::panic::catch_unwind(|| {
        let client = active_client();
        let _ = client.properties();
    });
    if probe.is_err() {
        return Err(PyImportError::new_err(format!(
            "mlrs-{0} requires the {0} runtime/driver; none was detected. \
             Install the {0} driver or use a different mlrs backend wheel.",
            mlrs_backend::capability::active_backend_name()
        )));
    }

    // The driver is present: register the low-level surface.
    m.add_function(wrap_pyfunction!(backend_supports_f64, m)?)?;

    // Register all 12 estimator `#[pyclass]` wrappers (PY-01). The pure-Python
    // `mlrs` shim (Plan 04) subclasses sklearn and delegates to these.
    use estimators::cluster::{PyDBSCAN, PyKMeans};
    use estimators::covariance::{PyEmpiricalCovariance, PyLedoitWolf};
    use estimators::decomposition::{PyIncrementalPCA, PyPCA, PyTruncatedSVD};
    use estimators::kernel::{PyKernelDensity, PyKernelRidge};
    use estimators::linear::{
        PyElasticNet, PyLasso, PyLinearRegression, PyLinearSVC, PyLinearSVR,
        PyLogisticRegression, PyMBSGDClassifier, PyMBSGDRegressor, PyRidge,
    };
    use estimators::neighbors::{
        PyKNeighborsClassifier, PyKNeighborsRegressor, PyNearestNeighbors,
    };
    use estimators::projection::{
        johnson_lindenstrauss_min_dim, PyGaussianRandomProjection, PySparseRandomProjection,
    };
    use estimators::spectral::{PySpectralClustering, PySpectralEmbedding};
    m.add_class::<PyLinearRegression>()?;
    m.add_class::<PyRidge>()?;
    m.add_class::<PyLasso>()?;
    m.add_class::<PyElasticNet>()?;
    m.add_class::<PyLogisticRegression>()?;
    m.add_class::<PyKMeans>()?;
    m.add_class::<PyDBSCAN>()?;
    m.add_class::<PyPCA>()?;
    m.add_class::<PyTruncatedSVD>()?;
    m.add_class::<PyNearestNeighbors>()?;
    m.add_class::<PyKNeighborsClassifier>()?;
    m.add_class::<PyKNeighborsRegressor>()?;

    // Phase-7 covariance / projection / IncrementalPCA wrappers (PY-06 incr.).
    m.add_class::<PyEmpiricalCovariance>()?;
    m.add_class::<PyLedoitWolf>()?;
    m.add_class::<PyIncrementalPCA>()?;
    m.add_class::<PyGaussianRandomProjection>()?;
    m.add_class::<PySparseRandomProjection>()?;
    m.add_function(wrap_pyfunction!(johnson_lindenstrauss_min_dim, m)?)?;

    // Phase-8 kernel-family wrappers (KERNEL-01 / KERNEL-02 — PY-06 incr.).
    m.add_class::<PyKernelRidge>()?;
    m.add_class::<PyKernelDensity>()?;

    // Phase-9 spectral-family wrappers (SPECTRAL-01 / SPECTRAL-02 — PY-06 incr.).
    m.add_class::<PySpectralEmbedding>()?;
    m.add_class::<PySpectralClustering>()?;

    // Phase-10 SGD / linear-SVM wrappers (SGDSVM-01..04 — PY-06 incr.).
    m.add_class::<PyMBSGDClassifier>()?;
    m.add_class::<PyMBSGDRegressor>()?;
    m.add_class::<PyLinearSVC>()?;
    m.add_class::<PyLinearSVR>()?;
    Ok(())
}
