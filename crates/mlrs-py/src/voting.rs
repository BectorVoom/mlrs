//! PyO3 free-function surface for the voting meta-estimators (VOTE-BIND-01).
//!
//! Thin wrappers over [`mlrs_algos::ensemble::voting`], the structural core of
//! `mlrs.VotingRegressor` and `mlrs.VotingClassifier`: the `weights` length
//! rule, the `'drop'` filter over `weights`, the `voting` constraint, the
//! `get_feature_names_out` strings, and — the part that carries data — the
//! aggregations behind `transform`, `predict` and `predict_proba`.
//!
//! ## The classifier's aggregations are four, not two (VOTE-CLF-01)
//!
//! `voting` splits the estimator in half and the two halves share no data path:
//!
//! | method | `voting='hard'` | `voting='soft'` |
//! |---|---|---|
//! | `predict` | [`voting_hard_predict`] | [`voting_soft_predict`] |
//! | `predict_proba` | *(absent)* | [`voting_soft_proba`] |
//! | `transform` | [`voting_aggregate`] `"transform"` | [`voting_hstack`] |
//!
//! The hard route carries INTEGER labels and so crosses as `uint32` rather than
//! through the float dispatch every other entry point here uses; the soft route
//! carries `n × n_classes` probability blocks and reuses the regressor's
//! reduction with `n · n_classes` in place of `n`.
//!
//! The structural rules voting shares with stacking are NOT re-exported here.
//! `_BaseComposition._validate_names`, the `'drop'` sentinel and the kept-index
//! bookkeeping are `sklearn.ensemble._base`'s own shared rules, and the shim
//! calls [`crate::stacking`]'s bindings for them; a second copy would be a
//! second place for the message text to drift.
//!
//! [`voting_aggregate`] is the one entry point that moves a sample-sized array,
//! and it exists for the same reason `stacking_concatenate` does: to give the
//! host and CubeCL arms a boundary to be measured against `numpy` across.
//! `np.average` / `np.asarray(...).T` remain the DEFAULT — see
//! [`voting_engine`] and `docs/voting.md`.
//!
//! Every [`VotingError::Value`] is a `ValueError`; [`VotingError::ZeroWeightSum`]
//! is a `ZeroDivisionError`, because that is what `np.average` raises for a zero
//! weight sum and a caller's `except` clause migrated from sklearn has to keep
//! catching it.

use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

use arrow::array::ArrayRef;
use mlrs_algos::ensemble::voting as vote;
use mlrs_backend::prims::voting::{vote_engine, VoteEngine};

use crate::egress::{f32_vec_to_pyarrow, f64_vec_to_pyarrow, u32_vec_to_pyarrow};
use crate::ingress::{as_f32, as_f64, as_u32, capsule_to_array, float_dtype, FloatDtype};
use crate::ingress::{host_slice_f32, host_slice_f64, host_slice_u32};

fn voting_err_to_py(err: vote::VotingError) -> PyErr {
    match err {
        vote::VotingError::Value(msg) => PyValueError::new_err(msg),
        vote::VotingError::ZeroWeightSum => {
            PyZeroDivisionError::new_err(err_zero_weight_sum_message())
        }
    }
}

/// numpy's own text for a zero weight sum, kept next to the mapping that uses
/// it so the two cannot drift apart.
fn err_zero_weight_sum_message() -> &'static str {
    "Weights sum to zero, can't be normalized"
}

/// sklearn `_BaseVoting.fit`'s length check: `len(weights) == len(estimators)`.
///
/// Returns `None` and raises on mismatch, rather than returning a bool — the
/// message is the whole point of the rule and a bool would leave the shim to
/// re-spell it.
#[pyfunction]
pub fn voting_check_weights(n_weights: usize, n_estimators: usize) -> PyResult<()> {
    vote::check_weights_len(n_weights, n_estimators).map_err(voting_err_to_py)
}

/// sklearn `_BaseVoting._weights_not_none`: the POSITIONS in `weights` whose
/// entries are not `'drop'`, in list order.
///
/// `is_drop[i]` is the shim's `estimators[i][1] == 'drop'` answer, for the reason
/// `stacking_kept_indices` takes the same argument: the comparison is on an
/// arbitrary Python object and only Python can make it.
///
/// Positions rather than values, so the shim indexes its OWN untouched weight
/// objects. Passing them through as `f64` would erase a `float32` weight array's
/// dtype, which numpy propagates into `predict`'s result — see
/// `mlrs_algos::ensemble::voting::active_weight_slots`.
#[pyfunction]
pub fn voting_active_weight_slots(n_weights: usize, is_drop: Vec<bool>) -> PyResult<Vec<usize>> {
    vote::active_weight_slots(n_weights, &is_drop).map_err(voting_err_to_py)
}

/// sklearn `VotingRegressor.get_feature_names_out`: `"{class}_{name}"` per kept
/// member.
#[pyfunction]
pub fn voting_feature_names(class_name: &str, kept_names: Vec<String>) -> Vec<String> {
    vote::transform_feature_names(class_name, &kept_names)
}

/// The aggregation arm this process resolves `MLRS_VOTING_ENGINE` to: `"numpy"`
/// (default), `"host"`, or `"device"` (VOTE-01).
///
/// The shim asks once per `predict`/`transform` and branches on the answer, so
/// the three arms are named in ONE place — here — rather than by a second
/// environment read on the Python side that could disagree with Rust's.
/// `"numpy"` means the shim keeps the work in `np.average` /
/// `np.asarray(...).T` and never calls [`voting_aggregate`] at all.
#[pyfunction]
pub fn voting_engine() -> &'static str {
    vote_engine().as_str()
}

/// Aggregate the members' prediction columns in Rust — the host or device arm.
///
/// `columns` are the kept members' predictions as Arrow arrays, each `n_rows`
/// long and all carrying the SAME float dtype (the shim promotes them, mirroring
/// what numpy would have done, because a per-column promotion here would be a
/// second pass over the very data this call exists to read once).
///
/// `mode` picks what to compute:
///
/// * `"transform"` — the `n_rows × k` matrix, column `j` being member `j`.
///   `weights` is ignored, exactly as sklearn's `transform` ignores it.
/// * `"predict"` — the `n_rows`-long weighted mean. `weights` is `None` for the
///   uniform case.
///
/// Returns `(flat_arrow_array, n_cols)`; the shim reshapes. `engine` overrides
/// [`voting_engine`] for one call, which is what lets the benchmark harness A/B
/// both arms in one process instead of silently comparing an arm against itself.
#[pyfunction]
#[pyo3(signature = (columns, n_rows, mode, weights=None, engine=None))]
pub fn voting_aggregate<'py>(
    py: Python<'py>,
    columns: Vec<Bound<'py, PyAny>>,
    n_rows: usize,
    mode: &str,
    weights: Option<Vec<f64>>,
    engine: Option<&str>,
) -> PyResult<(Bound<'py, PyAny>, usize)> {
    let arm = resolve_arm(engine)?;
    let average = match mode {
        "predict" => true,
        "transform" => false,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown voting mode {other:?}; expected 'predict' or 'transform'"
            )))
        }
    };

    // The Arrow arrays are held for the whole call: the host slices below BORROW
    // them (the `capsule_to_array` contract), and the device arm reads through
    // those same slices while uploading.
    let (arrays, dtype) = float_blocks(&columns, arm)?;
    let n_cols = if average { 1 } else { arrays.len() };

    match dtype {
        FloatDtype::F32 => {
            let cols = f32_slices(&arrays)?;
            let w: Option<Vec<f32>> = weights.map(|w| w.iter().map(|&v| v as f32).collect());
            let out = py.detach(|| {
                let mut pool = crate::lock_pool();
                if average {
                    vote::vote_average(arm, &mut pool, &cols, w.as_deref(), n_rows)
                } else {
                    vote::vote_transform(arm, &mut pool, &cols, n_rows)
                }
            });
            Ok((
                f32_vec_to_pyarrow(py, out.map_err(voting_err_to_py)?)?,
                n_cols,
            ))
        }
        FloatDtype::F64 => {
            let cols = f64_slices(&arrays)?;
            let out = py.detach(|| {
                let mut pool = crate::lock_pool();
                if average {
                    vote::vote_average(arm, &mut pool, &cols, weights.as_deref(), n_rows)
                } else {
                    vote::vote_transform(arm, &mut pool, &cols, n_rows)
                }
            });
            Ok((
                f64_vec_to_pyarrow(py, out.map_err(voting_err_to_py)?)?,
                n_cols,
            ))
        }
    }
}

// ------------------------------------------------------------------------- //
// VotingClassifier (VOTE-CLF-01)
// ------------------------------------------------------------------------- //

/// Validate `voting` and echo the canonical spelling back.
///
/// sklearn's constraint is `StrOptions({"hard", "soft"})`, so an unrecognized
/// value raises a `ValueError` that the shim re-raises as
/// `InvalidParameterError` — the same two-step stacking's `stack_method` uses,
/// and for the same reason: `InvalidParameterError` is a Python class this crate
/// cannot construct, but the MESSAGE (the part callers match on) is Rust's.
///
/// Returns the string rather than `None` so the shim's own branch reads off
/// Rust's parse instead of re-comparing the literal — one place decides what
/// `"soft"` means.
#[pyfunction]
pub fn voting_mode(value: &str) -> PyResult<&'static str> {
    vote::voting_mode(value)
        .map(|mode| mode.as_str())
        .map_err(voting_err_to_py)
}

/// sklearn `VotingClassifier.get_feature_names_out`.
///
/// `voting='hard'` gives one name per kept member; `voting='soft'` gives
/// `n_classes` per member, suffixed with the class index and NO separator
/// (`votingclassifier_lr0`). `n_classes` is ignored on the hard route and the
/// shim still passes it, so the two routes share one call site.
#[pyfunction]
pub fn voting_classifier_feature_names(
    class_name: &str,
    kept_names: Vec<String>,
    voting: &str,
    n_classes: usize,
) -> PyResult<Vec<String>> {
    let mode = vote::voting_mode(voting).map_err(voting_err_to_py)?;
    Ok(vote::classifier_feature_names(
        class_name,
        &kept_names,
        mode,
        n_classes,
    ))
}

/// sklearn's one `get_feature_names_out` rejection: `voting='soft'` with
/// `flatten_transform=False` names a 3-D output, which has no columns.
///
/// Raises `ValueError` with sklearn's text, or returns `None`.
#[pyfunction]
pub fn voting_check_feature_names(voting: &str, flatten_transform: bool) -> PyResult<()> {
    let mode = vote::voting_mode(voting).map_err(voting_err_to_py)?;
    vote::check_feature_names_supported(mode, flatten_transform).map_err(voting_err_to_py)
}

/// `voting='hard'` — the weighted majority label per row, in Rust.
///
/// `columns` are the kept members' ENCODED predictions as `uint32` Arrow arrays,
/// each `n_rows` long; `n_bins` is one past the largest label present, which the
/// shim computes on the numpy side (it has to scan for a negative label anyway,
/// so that the numpy fallback — not this function — reports `np.bincount`'s own
/// "must have no negative elements"). `weights` is `None` for the uniform case.
///
/// Returns the argmax INDICES as a `uint32` Arrow array; the shim maps them back
/// through `classes_`. `engine` overrides [`voting_engine`] for one call, which
/// is what lets the benchmark harness A/B both arms in one process.
///
/// **The tally is `f64` on the host arm and on any backend with f64 device
/// kernels**, matching `np.bincount(x, weights=w)`'s own accumulator. On a
/// backend without them the device arm narrows to `f32`, which is still EXACT
/// for the uniform case (a sum of `1.0`s) and for integral weights; it is
/// documented rather than refused because refusing would leave hard voting — the
/// sklearn DEFAULT — with no device arm at all on such a backend.
#[pyfunction]
#[pyo3(signature = (columns, n_rows, n_bins, weights=None, engine=None))]
pub fn voting_hard_predict<'py>(
    py: Python<'py>,
    columns: Vec<Bound<'py, PyAny>>,
    n_rows: usize,
    n_bins: usize,
    weights: Option<Vec<f64>>,
    engine: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let arm = resolve_arm(engine)?;
    let arrays: Vec<ArrayRef> = columns
        .iter()
        .map(capsule_to_array)
        .collect::<PyResult<Vec<_>>>()?;
    if arrays.is_empty() {
        return Err(PyValueError::new_err(NO_ESTIMATORS));
    }
    let views = arrays.iter().map(as_u32).collect::<PyResult<Vec<_>>>()?;
    let mut cols = Vec::with_capacity(views.len());
    for view in &views {
        cols.push(host_slice_u32(*view)?);
    }

    let out = py.detach(|| {
        let mut pool = crate::lock_pool();
        vote::vote_hard_predict(arm, &mut pool, &cols, weights.as_deref(), n_rows, n_bins)
    });
    u32_vec_to_pyarrow(py, out.map_err(voting_err_to_py)?)
}

/// `voting='soft'` — `predict_proba`'s weighted average, in Rust.
///
/// `blocks` are the kept members' `predict_proba` outputs, each flattened
/// row-major to `n_rows * n_cols` and all carrying the same float dtype (the
/// shim promotes them, mirroring numpy). Returns the flat `n_rows × n_cols`
/// average; the shim reshapes.
///
/// This is [`voting_aggregate`]'s `"predict"` mode with `n_rows * n_cols`
/// elements per column — the reduced axis is still the member axis — and it is a
/// separate entry point only so the shape crosses explicitly rather than being
/// reconstructed on the Python side after the fact.
#[pyfunction]
#[pyo3(signature = (blocks, n_rows, n_cols, weights=None, engine=None))]
pub fn voting_soft_proba<'py>(
    py: Python<'py>,
    blocks: Vec<Bound<'py, PyAny>>,
    n_rows: usize,
    n_cols: usize,
    weights: Option<Vec<f64>>,
    engine: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let arm = resolve_arm(engine)?;
    let (arrays, dtype) = float_blocks(&blocks, arm)?;
    match dtype {
        FloatDtype::F32 => {
            let cols = f32_slices(&arrays)?;
            let w: Option<Vec<f32>> = weights.map(|w| w.iter().map(|&v| v as f32).collect());
            let out = py.detach(|| {
                let mut pool = crate::lock_pool();
                vote::vote_soft_proba(arm, &mut pool, &cols, w.as_deref(), n_rows, n_cols)
            });
            f32_vec_to_pyarrow(py, out.map_err(voting_err_to_py)?)
        }
        FloatDtype::F64 => {
            let cols = f64_slices(&arrays)?;
            let out = py.detach(|| {
                let mut pool = crate::lock_pool();
                vote::vote_soft_proba(arm, &mut pool, &cols, weights.as_deref(), n_rows, n_cols)
            });
            f64_vec_to_pyarrow(py, out.map_err(voting_err_to_py)?)
        }
    }
}

/// `voting='soft'` — the LABELS that average implies, in Rust.
///
/// `argmax(np.average(probas, axis=0, weights=w), axis=1)`. On the `device` arm
/// the two halves are FUSED and the `n_rows × n_cols` average never crosses the
/// bus; on the `host` arm they run in sequence, which is what `predict_proba`
/// followed by `np.argmax` would have cost anyway.
///
/// Returns the argmax indices as a `uint32` Arrow array.
#[pyfunction]
#[pyo3(signature = (blocks, n_rows, n_cols, weights=None, engine=None))]
pub fn voting_soft_predict<'py>(
    py: Python<'py>,
    blocks: Vec<Bound<'py, PyAny>>,
    n_rows: usize,
    n_cols: usize,
    weights: Option<Vec<f64>>,
    engine: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let arm = resolve_arm(engine)?;
    let (arrays, dtype) = float_blocks(&blocks, arm)?;
    let out = match dtype {
        FloatDtype::F32 => {
            let cols = f32_slices(&arrays)?;
            let w: Option<Vec<f32>> = weights.map(|w| w.iter().map(|&v| v as f32).collect());
            py.detach(|| {
                let mut pool = crate::lock_pool();
                vote::vote_soft_predict(arm, &mut pool, &cols, w.as_deref(), n_rows, n_cols)
            })
        }
        FloatDtype::F64 => {
            let cols = f64_slices(&arrays)?;
            py.detach(|| {
                let mut pool = crate::lock_pool();
                vote::vote_soft_predict(arm, &mut pool, &cols, weights.as_deref(), n_rows, n_cols)
            })
        }
    };
    u32_vec_to_pyarrow(py, out.map_err(voting_err_to_py)?)
}

/// `voting='soft', flatten_transform=True` — `np.hstack(probas)`, in Rust.
///
/// Returns `(flat_arrow_array, n_cols_out)` where `n_cols_out` is `k * n_cols`;
/// the shim reshapes. `weights` plays no part — sklearn's `transform` ignores
/// them on both routes — and is deliberately not a parameter here so a mutated
/// `weights` cannot break a `transform` sklearn completes.
#[pyfunction]
#[pyo3(signature = (blocks, n_rows, n_cols, engine=None))]
pub fn voting_hstack<'py>(
    py: Python<'py>,
    blocks: Vec<Bound<'py, PyAny>>,
    n_rows: usize,
    n_cols: usize,
    engine: Option<&str>,
) -> PyResult<(Bound<'py, PyAny>, usize)> {
    let arm = resolve_arm(engine)?;
    let (arrays, dtype) = float_blocks(&blocks, arm)?;
    let out_cols = arrays.len() * n_cols;
    let flat = match dtype {
        FloatDtype::F32 => {
            let cols = f32_slices(&arrays)?;
            let out = py.detach(|| {
                let mut pool = crate::lock_pool();
                vote::vote_hstack(arm, &mut pool, &cols, n_rows, n_cols)
            });
            f32_vec_to_pyarrow(py, out.map_err(voting_err_to_py)?)?
        }
        FloatDtype::F64 => {
            let cols = f64_slices(&arrays)?;
            let out = py.detach(|| {
                let mut pool = crate::lock_pool();
                vote::vote_hstack(arm, &mut pool, &cols, n_rows, n_cols)
            });
            f64_vec_to_pyarrow(py, out.map_err(voting_err_to_py)?)?
        }
    };
    Ok((flat, out_cols))
}

/// sklearn's text for an ensemble with nothing left in it, kept in one place
/// because three entry points reject on it.
const NO_ESTIMATORS: &str =
    "All estimators are dropped. At least one is required to be an estimator.";

/// `engine` (a one-call override) or [`voting_engine`] (the process knob).
///
/// `"numpy"` arriving here means the caller asked for the reference arm
/// explicitly — the benchmark harness does — and it maps to the host loop, since
/// `np.average` cannot be run from Rust.
fn resolve_arm(engine: Option<&str>) -> PyResult<VoteEngine> {
    match engine {
        None => Ok(vote_engine()),
        Some("numpy") | Some("host") => Ok(VoteEngine::Host),
        Some("device") => Ok(VoteEngine::Device),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown voting engine {other:?}; expected 'host' or 'device'"
        ))),
    }
}

/// Import the probability blocks and agree on their float width.
///
/// Also carries the f64-on-a-device-without-f64-kernels rejection, which is
/// [`voting_aggregate`]'s verbatim — including the reason it tests
/// `f64_device_kernels_available` rather than the ADVERTISED f64 flag.
fn float_blocks(
    blocks: &[Bound<'_, PyAny>],
    arm: VoteEngine,
) -> PyResult<(Vec<ArrayRef>, FloatDtype)> {
    let arrays: Vec<ArrayRef> = blocks
        .iter()
        .map(capsule_to_array)
        .collect::<PyResult<Vec<_>>>()?;
    let dtype = match arrays.first() {
        Some(first) => float_dtype(first)?,
        None => return Err(PyValueError::new_err(NO_ESTIMATORS)),
    };
    if arm == VoteEngine::Device
        && dtype == FloatDtype::F64
        && !mlrs_backend::capability::f64_device_kernels_available()
    {
        return Err(PyValueError::new_err(F64_DEVICE_REFUSAL));
    }
    Ok((arrays, dtype))
}

/// The refusal text for an f64 device aggregation on a backend that has no f64
/// kernels, shared by every entry point so a caller sees one message.
const F64_DEVICE_REFUSAL: &str =
    "MLRS_VOTING_ENGINE=device cannot aggregate float64 predictions on \
     this backend (no f64 device kernels); use the 'host' or 'numpy' \
     arm, or pass float32 columns";

/// Borrow every array as `&[f32]`, validated.
///
/// The slices borrow the CALLER's `arrays` — `as_f32` yields a view whose
/// lifetime is the `ArrayRef`'s, and `host_slice_f32` passes that lifetime
/// through — so nothing here is self-referential and the caller only has to keep
/// `arrays` alive, which is the [`capsule_to_array`] contract anyway.
fn f32_slices(arrays: &[ArrayRef]) -> PyResult<Vec<&[f32]>> {
    let mut cols = Vec::with_capacity(arrays.len());
    for array in arrays {
        cols.push(host_slice_f32(as_f32(array)?)?);
    }
    Ok(cols)
}

/// f64 twin of [`f32_slices`].
fn f64_slices(arrays: &[ArrayRef]) -> PyResult<Vec<&[f64]>> {
    let mut cols = Vec::with_capacity(arrays.len());
    for array in arrays {
        cols.push(host_slice_f64(as_f64(array)?)?);
    }
    Ok(cols)
}

/// Register every voting binding on the `_mlrs` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(voting_check_weights, m)?)?;
    m.add_function(wrap_pyfunction!(voting_active_weight_slots, m)?)?;
    m.add_function(wrap_pyfunction!(voting_feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(voting_engine, m)?)?;
    m.add_function(wrap_pyfunction!(voting_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(voting_mode, m)?)?;
    m.add_function(wrap_pyfunction!(voting_classifier_feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(voting_check_feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(voting_hard_predict, m)?)?;
    m.add_function(wrap_pyfunction!(voting_soft_proba, m)?)?;
    m.add_function(wrap_pyfunction!(voting_soft_predict, m)?)?;
    m.add_function(wrap_pyfunction!(voting_hstack, m)?)?;
    Ok(())
}
