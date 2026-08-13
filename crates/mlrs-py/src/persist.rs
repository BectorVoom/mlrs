//! `persist` — the PyO3 half of model save/load.
//!
//! The on-disk format, the containers and the per-estimator `SaveModel` /
//! `LoadModel` impls all live in `mlrs_algos::persist` and its family modules.
//! This module carries only what the BINDING adds: dtype-arm dispatch, the
//! Python-side metadata channel, and the error mapping.
//!
//! ## The three problems this layer solves
//!
//! **1. A `#[pyclass]` cannot be generic over `F`.** Every estimator wrapper
//! holds an `Any<Name>` enum with `Unfit` / `F32(..)` / `F64(..)` arms
//! ([`crate::dispatch`]). Saving means matching the fitted arm; loading means
//! choosing one BEFORE there is a model to ask. [`PersistableAny`] is that
//! dispatch, and [`impl_persistable_any!`] generates it — one line per
//! estimator, against enums whose shape the same macro system already fixed.
//!
//! The load arm comes from the FILE, via
//! [`model_float_width`](mlrs_algos::persist::model_float_width): an `F32` file
//! loads into the f32 arm. Widening is available in the format and deliberately
//! not used here, because it would silently stop a Python round-trip from being
//! bit-exact. A file with no float tensor at all (`Binarizer`, `Normalizer`)
//! has no width to match, so it takes the backend's default — the same rule
//! `mlrs._io.pick_dtype` applies to fresh data.
//!
//! **2. The Python shim owns state the Rust estimator does not.** `output_type`,
//! sklearn parity arguments like `copy`, and the class name a loader needs to
//! rebuild the right object are all invisible to `mlrs_algos`. They ride in the
//! same file's `__metadata__` under a `py:` prefix, through
//! [`SaveModelExt::save_with`](mlrs_algos::persist::SaveModelExt::save_with).
//! One file, still a valid safetensors, still readable by
//! `safetensors.numpy.load_file` — with a handful of extra string entries a
//! non-mlrs reader can ignore.
//!
//! **3. A loader has to identify a file before it can load it.** Every typed
//! reader validates its container discriminator first, by design, so none of
//! them can answer "what is this?". [`read_metadata`] is the container-agnostic
//! way in, and is what `mlrs.load(path)` uses to find the class.
//!
//! ## What is deliberately NOT here
//!
//! No estimator-name-to-class table. That mapping is data, it changes whenever
//! an estimator is added, and it belongs next to the classes it names — so it
//! lives in the Python shim (`mlrs/_persist.py`), keyed on the `estimator`
//! string the Rust files already write.

use std::collections::BTreeMap;
use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mlrs_algos::persist::PersistError;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;

/// Map a [`PersistError`] to a `PyErr`.
///
/// Every variant is a caller-supplied-path or file-contents fault, so they all
/// map to `PyValueError` with the typed error's `Display` preserved verbatim —
/// which is what carries the useful half ("model file is missing the required
/// tensor 'coef_'", "this is a 'gaussian_nb' model file; 'ridge' cannot load
/// it"). An `Io` variant keeps its path.
///
/// `PyIOError` was considered for the `Io` variant and rejected: a caller
/// catching one exception type around a `load` is the common case, and splitting
/// by cause would make them catch two for no gain in what they can DO about it.
/// The message already names which it was.
pub fn persist_err_to_py(err: PersistError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Save/load dispatch over one estimator's `Any<Name>` arms.
///
/// Implemented by [`impl_persistable_any!`] for every wrapper whose fitted arms
/// hold a `mlrs_algos` estimator implementing `SaveModel` + `LoadModel`. The
/// trait exists so the `#[pymethods]` bodies are two lines each: PyO3 has no
/// `multiple-pymethods` feature enabled here, so `save`/`load` must be written
/// into each estimator's EXISTING `#[pymethods]` block, and every line that
/// appears there is a line duplicated 50-odd times.
pub trait PersistableAny: Sized {
    /// The estimator name used in the not-fitted error.
    const NAME: &'static str;

    /// Serialize the fitted arm, merging `extra` into `__metadata__`.
    fn save_arm(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        path: &Path,
        extra: &BTreeMap<String, String>,
    ) -> PyResult<()>;

    /// Deserialize into the arm matching the file's stored float width.
    fn load_arm(pool: &mut BufferPool<ActiveRuntime>, path: &Path) -> PyResult<Self>;
}

/// Which arm a file should load into.
///
/// `Some(true)` means f64. The choice is the file's stored width where it has
/// one; where it does not, the backend's f64 capability decides, matching
/// `mlrs._io.pick_dtype`'s rule for fresh data so a tensorless model does not
/// arrive on an arm the backend cannot serve.
pub fn load_wants_f64(path: &Path) -> PyResult<bool> {
    match mlrs_algos::persist::model_float_width(path).map_err(persist_err_to_py)? {
        Some(8) => Ok(true),
        Some(_) => Ok(false),
        None => Ok(crate::capability::supports_f64()),
    }
}

/// Generate [`PersistableAny`] for one `Any<Name>` dispatch enum.
///
/// The emitted `save_arm` matches the two fitted arms and rejects everything
/// else as not-fitted; `load_arm` picks the arm from the file and applies the
/// same `guard_f64` the `fit` path does BEFORE constructing an f64 model, so an
/// f64 file on an f64-incapable backend fails the same way an f64 fit would
/// rather than part-way through the load.
///
/// Invocation:
///
/// ```ignore
/// impl_persistable_any! {
///     any:  AnyStandardScaler,
///     algo: mlrs_algos::preprocessing::standard_scaler::StandardScaler,
///     name: "standard_scaler",
/// }
/// ```
#[macro_export]
macro_rules! impl_persistable_any {
    (
        any:  $any:ident,
        algo: $algo:ident $( :: $algo_rest:ident )*,
        name: $name:literal $(,)?
    ) => {
        impl $crate::persist::PersistableAny for $any {
            const NAME: &'static str = $name;

            fn save_arm(
                &self,
                pool: &mlrs_backend::pool::BufferPool<mlrs_backend::runtime::ActiveRuntime>,
                path: &std::path::Path,
                extra: &std::collections::BTreeMap<String, String>,
            ) -> pyo3::PyResult<()> {
                use mlrs_algos::persist::SaveModelExt as _;
                match self {
                    Self::F32(m) => m.save_with(pool, path, extra),
                    Self::F64(m) => m.save_with(pool, path, extra),
                    _ => return Err($crate::errors::not_fitted($name, "save")),
                }
                .map_err($crate::persist::persist_err_to_py)
            }

            fn load_arm(
                pool: &mut mlrs_backend::pool::BufferPool<mlrs_backend::runtime::ActiveRuntime>,
                path: &std::path::Path,
            ) -> pyo3::PyResult<Self> {
                if $crate::persist::load_wants_f64(path)? {
                    $crate::capability::guard_f64()?;
                    let m = <$algo $( :: $algo_rest )* <f64, mlrs_algos::typestate::Fitted>
                        as mlrs_algos::persist::LoadModel>::load(pool, path)
                        .map_err($crate::persist::persist_err_to_py)?;
                    Ok(Self::F64(m))
                } else {
                    let m = <$algo $( :: $algo_rest )* <f32, mlrs_algos::typestate::Fitted>
                        as mlrs_algos::persist::LoadModel>::load(pool, path)
                        .map_err($crate::persist::persist_err_to_py)?;
                    Ok(Self::F32(m))
                }
            }
        }
    };
}

/// The body of every wrapper's `save` `#[pymethods]` method.
///
/// Takes the extra metadata as a plain `Vec<(String, String)>` rather than a
/// `HashMap` so the caller's ordering is irrelevant and the merge stays
/// deterministic — the underlying map is a `BTreeMap`, which is what keeps
/// saving the same model twice byte-identical.
///
/// The `Sync` / `Send` bounds are what [`Python::detach`] requires to release
/// the GIL around the device work, and they hold for every `Any<Name>` enum for
/// the same reason the `fit` paths can already detach: the arms hold device
/// handles and plain data, never a Python object.
pub fn save_impl<T: PersistableAny + Sync>(
    py: Python<'_>,
    inner: &T,
    path: &str,
    extra: Vec<(String, String)>,
) -> PyResult<()> {
    let extra: BTreeMap<String, String> = extra.into_iter().collect();
    let path = Path::new(path);
    py.detach(|| {
        let pool = crate::lock_pool();
        inner.save_arm(&pool, path, &extra)
    })
}

/// The body of every wrapper's `load` `#[pymethods]` method.
///
/// `load` is an INSTANCE method that replaces `inner`, mirroring `fit`, rather
/// than a `#[staticmethod]` returning a new wrapper. The wrapper stores its
/// constructor hyperparameters alongside `inner` (so a second `fit` on the same
/// handle re-fits with what the caller asked for, not with defaults), and a
/// static constructor would have to reconstruct those from the file — which the
/// Python shim has already done, since it rebuilt the object from its own saved
/// `get_params()` before calling this.
pub fn load_impl<T: PersistableAny + Send>(py: Python<'_>, path: &str) -> PyResult<T> {
    let path = Path::new(path);
    py.detach(|| {
        let mut pool = crate::lock_pool();
        T::load_arm(&mut pool, path)
    })
}

/// Read a model file's `__metadata__` without knowing what estimator wrote it.
///
/// The entry point for `mlrs.load(path)`: it needs `estimator` to pick a class
/// and the `py:` entries to rebuild it, both before any typed reader — each of
/// which validates its own container first — could be called.
///
/// Returns every entry verbatim, including the `format` / `version` /
/// `estimator` discriminators and the `param:` values. Also useful on its own
/// for inspecting a file without loading the model.
#[pyfunction]
pub fn read_metadata(path: &str) -> PyResult<BTreeMap<String, String>> {
    mlrs_algos::persist::read_raw_metadata(Path::new(path)).map_err(persist_err_to_py)
}

/// The float width a model file's arrays are stored at — `"f32"`, `"f64"`, or
/// `None` for a file that holds no float tensor.
///
/// Exposed for tooling and for the tests that gate the "an f32 model writes an
/// f32 file" claim from the Python side; the load path uses
/// [`load_wants_f64`] rather than this, so the two cannot disagree.
#[pyfunction]
pub fn model_dtype(path: &str) -> PyResult<Option<&'static str>> {
    Ok(
        match mlrs_algos::persist::model_float_width(Path::new(path))
            .map_err(persist_err_to_py)?
        {
            Some(8) => Some("f64"),
            Some(_) => Some("f32"),
            None => None,
        },
    )
}
