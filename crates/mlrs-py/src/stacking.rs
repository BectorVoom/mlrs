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
//! There is deliberately NO binding for
//! `mlrs_algos::ensemble::stacking::concatenate_predictions`: the shim performs
//! that copy with `np.hstack`, which is the identical single `n × width` write
//! without the FFI round-trip or an f64-only constraint. The Rust function stays
//! as the executable statement of the layout contract for Rust-native callers
//! and for `crates/mlrs-algos/tests/stacking_test.rs`.
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

use mlrs_algos::ensemble::stacking as stk;

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

/// Register every stacking binding on the `_mlrs` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stacking_drop_sentinel, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_validate_names, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_kept_indices, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_cv_is_prefit, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_meta_layout, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_feature_names, m)?)?;
    Ok(())
}
