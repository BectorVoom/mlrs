//! PyO3 free-function surface for the stacking meta-estimator (STACK-BIND-01).
//!
//! Thin wrappers over [`mlrs_algos::ensemble::stacking`], the structural core of
//! `mlrs.StackingRegressor`: name validation, the `'drop'` bookkeeping, the
//! meta-feature column layout, the `get_feature_names_out` strings, and the
//! `cv="prefit"` string parameter. Everything crossing this boundary is host
//! bookkeeping — index lists and short string lists, never a sample matrix — so
//! these take plain `Vec<String>` / `Vec<usize>` rather than the Arrow capsule
//! (the same call shape [`crate::model_selection`] uses for `partition_inverse`
//! and friends).
//!
//! The one exception is [`stacking_concatenate`] (STACK-META-01), which DOES
//! carry the meta matrix across: it is the entry point for the two Rust arms of
//! the meta-feature copy — the host arm
//! (`mlrs_algos::ensemble::stacking::concatenate_predictions`) and the CubeCL
//! device arm (`mlrs_backend::prims::stacking_meta`). Which arm the shim uses,
//! or whether it stays in `np.hstack`, is [`stacking_meta_engine`]'s answer.
//! `np.hstack` remains the DEFAULT: the copy performs no arithmetic, so both
//! Rust arms start a capsule round-trip in debt, and the ladder in
//! `docs/stacking.md` is what settles it rather than an assumption.
//!
//! Every [`StackingError`] is a `ValueError` — sklearn raises nothing else from
//! its stacking composition layer — except the `cv` string, which sklearn
//! rejects through `@validate_params` as `InvalidParameterError`; that one is
//! re-raised by the Python shim, which owns the exception class (see
//! [`cv_is_prefit`]).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

use arrow::array::ArrayRef;
use mlrs_algos::ensemble::stacking as stk;
use mlrs_backend::prims::stacking_meta::{meta_engine, MetaEngine};

use crate::egress::{f32_vec_to_pyarrow, f64_vec_to_pyarrow};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, FloatDtype};
use crate::ingress::{host_slice_f32, host_slice_f64};

fn stacking_err_to_py(err: stk::StackingError) -> PyErr {
    let stk::StackingError::Value(msg) = err;
    PyValueError::new_err(msg)
}

/// The `'drop'` sentinel, surfaced so the shim compares against the SAME
/// literal the Rust rules do rather than repeating it.
#[pyfunction]
pub fn stacking_drop_sentinel() -> &'static str {
    stk::DROP
}

/// sklearn `_BaseComposition._validate_names`: duplicate names, names that
/// collide with the meta-estimator's own constructor arguments, and names
/// containing `__`. `ctor_params` is the caller's `get_params(deep=False)` keys.
#[pyfunction]
pub fn stacking_validate_names(names: Vec<String>, ctor_params: Vec<String>) -> PyResult<()> {
    stk::validate_names(&names, &ctor_params).map_err(stacking_err_to_py)
}

/// The positions of the non-`'drop'` entries, in list order.
///
/// `is_drop[i]` is the shim's `estimators[i][1] == "drop"` answer — the
/// comparison has to happen in Python (the value is an arbitrary object whose
/// `__eq__` may be overloaded), the consequences do not.
#[pyfunction]
pub fn stacking_kept_indices(is_drop: Vec<bool>) -> PyResult<Vec<usize>> {
    stk::kept_indices(&is_drop).map_err(stacking_err_to_py)
}

/// Whether a STRING `cv` selects the prefit route.
///
/// Returns `true` for `"prefit"` and raises for any other string, carrying
/// sklearn's `StrOptions` message. The shim re-raises it as
/// `InvalidParameterError` (which subclasses both `ValueError` and `TypeError`)
/// so `except TypeError` callers migrating from sklearn still catch it — the
/// same reasoning `ModelSelectionError` splits its two variants for.
#[pyfunction]
pub fn stacking_cv_is_prefit(cv: &str) -> PyResult<bool> {
    stk::cv_route_from_str(cv)
        .map(|route| route == stk::CvRoute::Prefit)
        .map_err(stacking_err_to_py)
}

/// The meta-feature column layout: `(n_feature_outs, offsets, n_meta, width)`.
///
/// `pred_cols[i]` is the column count of the `i`-th KEPT estimator's prediction
/// block; `n_features` is `X`'s width and is consulted only when `passthrough`.
#[pyfunction]
pub fn stacking_meta_layout(
    pred_cols: Vec<usize>,
    n_features: usize,
    passthrough: bool,
) -> PyResult<(Vec<usize>, Vec<usize>, usize, usize)> {
    stk::meta_layout(&pred_cols, n_features, passthrough)
        .map(|l| (l.n_feature_outs, l.offsets, l.n_meta, l.width))
        .map_err(stacking_err_to_py)
}

/// sklearn `_BaseStacking.get_feature_names_out`. `input_features` is `Some`
/// exactly when `passthrough` is set.
#[pyfunction]
#[pyo3(signature = (class_name, kept_names, n_feature_outs, input_features=None))]
pub fn stacking_feature_names(
    class_name: &str,
    kept_names: Vec<String>,
    n_feature_outs: Vec<usize>,
    input_features: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    stk::meta_feature_names(
        class_name,
        &kept_names,
        &n_feature_outs,
        input_features.as_deref(),
    )
    .map_err(stacking_err_to_py)
}

/// The meta-assembly arm this process resolves `MLRS_STACK_META_ENGINE` to:
/// `"numpy"` (default), `"host"`, or `"device"` (STACK-META-01).
///
/// The shim asks once per `fit`/`transform` and branches on the answer, so the
/// three arms are named in ONE place — here — rather than by a second
/// environment read on the Python side that could disagree with Rust's.
/// `"numpy"` means the shim keeps the copy in `np.hstack` and never calls
/// [`stacking_concatenate`] at all.
#[pyfunction]
pub fn stacking_meta_engine() -> &'static str {
    meta_engine().as_str()
}

/// Assemble the meta-feature matrix in Rust — the host or the device arm.
///
/// `blocks` are the kept estimators' prediction blocks as Arrow arrays, each
/// row-major and `n_rows`-tall with `pred_cols[i]` columns; `x` is the design,
/// required exactly when `passthrough`. Every array must carry the SAME float
/// dtype — the shim promotes them (mirroring what `np.hstack` would have done)
/// before crossing, because a per-block promotion here would be a second copy
/// of the very data this call exists to copy once.
///
/// Returns `(flat_arrow_array, width)`; the shim reshapes to
/// `(n_rows, width)`. `engine` overrides [`stacking_meta_engine`] for one call
/// (the benchmark harness A/Bs both arms in one process this way, which is what
/// keeps a sweep from silently comparing an arm against itself).
#[pyfunction]
#[pyo3(signature = (blocks, pred_cols, n_rows, n_features, passthrough, x=None, engine=None))]
pub fn stacking_concatenate<'py>(
    py: Python<'py>,
    blocks: Vec<Bound<'py, PyAny>>,
    pred_cols: Vec<usize>,
    n_rows: usize,
    n_features: usize,
    passthrough: bool,
    x: Option<Bound<'py, PyAny>>,
    engine: Option<&str>,
) -> PyResult<(Bound<'py, PyAny>, usize)> {
    let arm = match engine {
        None => meta_engine(),
        Some("host") => MetaEngine::Host,
        Some("device") => MetaEngine::Device,
        Some("numpy") => MetaEngine::Host,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown meta engine {other:?}; expected 'host' or 'device'"
            )))
        }
    };
    let layout = stk::meta_layout(&pred_cols, n_features, passthrough).map_err(stacking_err_to_py)?;
    let width = layout.width;

    // The Arrow arrays are held for the whole call: the host slices below
    // BORROW them (the `capsule_to_array` contract), and the device arm reads
    // through those same slices while uploading.
    let arrays: Vec<ArrayRef> = blocks
        .iter()
        .map(capsule_to_array)
        .collect::<PyResult<Vec<_>>>()?;
    let x_array: Option<ArrayRef> = match (passthrough, x.as_ref()) {
        (true, Some(v)) => Some(capsule_to_array(v)?),
        (true, None) => {
            return Err(PyValueError::new_err(
                "passthrough=True requires X, but none was given",
            ))
        }
        (false, _) => None,
    };

    let dtype = match arrays.first() {
        Some(first) => float_dtype(first)?,
        None => {
            return Err(PyValueError::new_err(
                "All estimators are dropped. At least one is required to be an estimator.",
            ))
        }
    };
    if arm == MetaEngine::Device
        && dtype == FloatDtype::F64
        && !mlrs_backend::capability::f64_device_kernels_available()
    {
        // NOT `capability::guard_f64` (the ADVERTISED flag): cubecl's rocm and
        // cuda backends decline to advertise f64 because their matmul rejects
        // f64 operands, yet they run an ordinary f64 kernel — and this kernel is
        // a strided copy, with no arithmetic to be picky about. Gating on the
        // advertised flag refused the device arm on rocm for the ordinary case
        // of sklearn members, whose predictions are always f64.
        return Err(PyValueError::new_err(
            "MLRS_STACK_META_ENGINE=device cannot assemble a float64 meta matrix \
             on this backend (no f64 device kernels); use the 'host' or 'numpy' \
             arm, or pass float32 blocks",
        ));
    }

    match dtype {
        FloatDtype::F32 => {
            let views = arrays.iter().map(as_f32).collect::<PyResult<Vec<_>>>()?;
            let mut slices = Vec::with_capacity(views.len());
            for (view, &cols) in views.iter().zip(&pred_cols) {
                slices.push((host_slice_f32(view)?, cols));
            }
            let x_view = x_array.as_ref().map(as_f32).transpose()?;
            let x_slice = match x_view {
                Some(v) => Some((host_slice_f32(v)?, n_features)),
                None => None,
            };
            let out = py.detach(|| {
                let mut pool = crate::lock_pool();
                stk::concatenate_meta(arm, &mut pool, &layout, &slices, n_rows, x_slice)
            });
            Ok((f32_vec_to_pyarrow(py, out.map_err(stacking_err_to_py)?)?, width))
        }
        FloatDtype::F64 => {
            let views = arrays.iter().map(as_f64).collect::<PyResult<Vec<_>>>()?;
            let mut slices = Vec::with_capacity(views.len());
            for (view, &cols) in views.iter().zip(&pred_cols) {
                slices.push((host_slice_f64(view)?, cols));
            }
            let x_view = x_array.as_ref().map(as_f64).transpose()?;
            let x_slice = match x_view {
                Some(v) => Some((host_slice_f64(v)?, n_features)),
                None => None,
            };
            let out = py.detach(|| {
                let mut pool = crate::lock_pool();
                stk::concatenate_meta(arm, &mut pool, &layout, &slices, n_rows, x_slice)
            });
            Ok((f64_vec_to_pyarrow(py, out.map_err(stacking_err_to_py)?)?, width))
        }
    }
}

/// Register every stacking binding on the `_mlrs` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stacking_drop_sentinel, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_validate_names, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_kept_indices, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_cv_is_prefit, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_meta_layout, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_meta_engine, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_concatenate, m)?)?;
    Ok(())
}
