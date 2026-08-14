//! PyO3 free-function surface for the voting meta-estimator (VOTE-BIND-01).
//!
//! Thin wrappers over [`mlrs_algos::ensemble::voting`], the structural core of
//! `mlrs.VotingRegressor`: the `weights` length rule, the `'drop'` filter over
//! `weights`, the `get_feature_names_out` strings, and — the part that carries
//! data — the two aggregations behind `transform` and `predict`.
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

use crate::egress::{f32_vec_to_pyarrow, f64_vec_to_pyarrow};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, FloatDtype};
use crate::ingress::{host_slice_f32, host_slice_f64};

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
    let arm = match engine {
        None => vote_engine(),
        // `numpy` reaching this function at all means the caller asked for the
        // reference arm explicitly (the benchmark harness does); it is the host
        // loop, since `np.average` cannot be run from here.
        Some("numpy") | Some("host") => VoteEngine::Host,
        Some("device") => VoteEngine::Device,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown voting engine {other:?}; expected 'host' or 'device'"
            )))
        }
    };
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
    let arrays: Vec<ArrayRef> = columns
        .iter()
        .map(capsule_to_array)
        .collect::<PyResult<Vec<_>>>()?;
    let dtype = match arrays.first() {
        Some(first) => float_dtype(first)?,
        None => {
            return Err(PyValueError::new_err(
                "All estimators are dropped. At least one is required to be an estimator.",
            ))
        }
    };
    let k = arrays.len();
    let n_cols = if average { 1 } else { k };

    if arm == VoteEngine::Device
        && dtype == FloatDtype::F64
        && !mlrs_backend::capability::f64_device_kernels_available()
    {
        // NOT `capability::guard_f64` (the ADVERTISED flag): cubecl's rocm and
        // cuda backends decline to advertise f64 because their MATMUL rejects
        // f64 operands, yet they run an ordinary f64 kernel fine — and these
        // kernels are a multiply-add and a copy. Gating on the advertised flag
        // is what made the stacking device arm raise on rocm for the ordinary
        // case of sklearn members, whose predictions are always f64
        // (STACK-META-01).
        return Err(PyValueError::new_err(
            "MLRS_VOTING_ENGINE=device cannot aggregate float64 predictions on \
             this backend (no f64 device kernels); use the 'host' or 'numpy' \
             arm, or pass float32 columns",
        ));
    }

    match dtype {
        FloatDtype::F32 => {
            let views = arrays.iter().map(as_f32).collect::<PyResult<Vec<_>>>()?;
            let mut cols = Vec::with_capacity(views.len());
            for view in &views {
                cols.push(host_slice_f32(*view)?);
            }
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
            let views = arrays.iter().map(as_f64).collect::<PyResult<Vec<_>>>()?;
            let mut cols = Vec::with_capacity(views.len());
            for view in &views {
                cols.push(host_slice_f64(*view)?);
            }
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

/// Register every voting binding on the `_mlrs` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(voting_check_weights, m)?)?;
    m.add_function(wrap_pyfunction!(voting_active_weight_slots, m)?)?;
    m.add_function(wrap_pyfunction!(voting_feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(voting_engine, m)?)?;
    m.add_function(wrap_pyfunction!(voting_aggregate, m)?)?;
    Ok(())
}
