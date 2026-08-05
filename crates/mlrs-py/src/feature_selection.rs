//! PyO3 surface for `mlrs.feature_selection` (FSEL-01).
//!
//! ## Why free functions and plain `Vec<f64>`, not the arrow capsule
//! Same reasoning as [`crate::metrics`], and for the same reason: this whole
//! surface is HOST `f64` by design (`mlrs_backend::prims::feature_score`'s module
//! docs give the accuracy argument), so there is no device buffer for an arrow
//! capsule to feed. Routing `X` through `capsule_to_array` →
//! `DeviceArray::from_host` → `fit` → `to_host` would add two full copies of the
//! design to a computation that never leaves the host — the "no-upload host
//! slice" pathology the mbsgd / NB / SVM cpu campaigns each had to undo after
//! the fact. This module skips it from the start.
//!
//! The one genuinely device-side piece of a selector, the `transform` column
//! gather, is NOT bound here either — deliberately. The Python shim does that
//! gather in the caller's own container (`mlrs._frame.take_columns`), because a
//! pandas or polars frame carries per-column dtypes and NAMES that a round-trip
//! through a flat `f64` device buffer would destroy. The compiled side returns the
//! support MASK; the mask is the model, and the gather is container bookkeeping.
//! That is the same split `mlrs.model_selection`'s splitters already use.
//!
//! ## What is bound
//! * the five closed-form score functions plus both `mutual_info_*` estimators,
//!   each returning `(scores, pvalues)` or just `scores`, matching sklearn's own
//!   return shapes;
//! * [`variance_threshold`] and [`univariate_select`], which return the fitted
//!   attributes AND the support mask in one call — a selector's `fit` has nothing
//!   else in it, so a second FFI crossing to read the mask back would be pure
//!   overhead.
//!
//! `AlgoError` maps to `PyValueError` through [`crate::errors::algo_err_to_py`].

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mlrs_algos::feature_selection::{
    self as fsel, DiscreteFeatures, GenericParam, MutualInfoParams, ScoreFunc, ScoreResult,
};

use crate::errors::algo_err_to_py;

/// Validate that `x` is a `rows × cols` row-major buffer and `y` (when present)
/// has `rows` entries, with sklearn's own message shape.
fn check_geometry(x_len: usize, y_len: Option<usize>, rows: usize, cols: usize) -> PyResult<()> {
    if rows == 0 || cols == 0 || x_len != rows * cols {
        return Err(PyValueError::new_err(format!(
            "mlrs.feature_selection: X has {x_len} values but rows={rows}, cols={cols}"
        )));
    }
    if let Some(n) = y_len {
        if n != rows {
            return Err(PyValueError::new_err(format!(
                "mlrs.feature_selection: y has {n} values but X has {rows} samples"
            )));
        }
    }
    Ok(())
}

/// Split a [`ScoreResult`] into the `(scores, pvalues)` tuple the shim expects.
fn split(res: ScoreResult) -> (Vec<f64>, Option<Vec<f64>>) {
    (res.scores, res.pvalues)
}

// ===========================================================================
// The closed-form score functions
// ===========================================================================

/// `sklearn.feature_selection.f_classif(X, y)`.
#[pyfunction]
pub fn f_classif(
    x: Vec<f64>,
    y: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
    check_geometry(x.len(), Some(y.len()), rows, cols)?;
    fsel::f_classif(&x, &y, rows, cols)
        .map(split)
        .map_err(algo_err_to_py)
}

/// `sklearn.feature_selection.f_oneway(*groups)`.
///
/// The variadic signature becomes `(flat, sizes, cols)`: the groups
/// concatenated row-major in class order, plus each group's row count. That is
/// the same information without a variadic crossing the FFI, and it is exactly
/// what the shim already has after grouping by label.
#[pyfunction]
pub fn f_oneway(
    flat: Vec<f64>,
    sizes: Vec<usize>,
    cols: usize,
) -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
    let total: usize = sizes.iter().sum();
    check_geometry(flat.len(), None, total.max(1), cols)?;
    let mut offset = 0usize;
    let mut groups: Vec<(usize, &[f64])> = Vec::with_capacity(sizes.len());
    for &n in sizes.iter() {
        let end = offset + n * cols;
        groups.push((n, &flat[offset..end]));
        offset = end;
    }
    fsel::f_oneway(&groups, cols)
        .map(split)
        .map_err(algo_err_to_py)
}

/// `sklearn.feature_selection.chi2(X, y)`.
#[pyfunction]
pub fn chi2(
    x: Vec<f64>,
    y: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
    check_geometry(x.len(), Some(y.len()), rows, cols)?;
    fsel::chi2(&x, &y, rows, cols)
        .map(split)
        .map_err(algo_err_to_py)
}

/// `sklearn.feature_selection.r_regression(X, y, center=, force_finite=)`.
#[pyfunction]
#[pyo3(signature = (x, y, rows, cols, center=true, force_finite=true))]
pub fn r_regression(
    x: Vec<f64>,
    y: Vec<f64>,
    rows: usize,
    cols: usize,
    center: bool,
    force_finite: bool,
) -> PyResult<Vec<f64>> {
    check_geometry(x.len(), Some(y.len()), rows, cols)?;
    fsel::r_regression(&x, &y, rows, cols, center, force_finite).map_err(algo_err_to_py)
}

/// `sklearn.feature_selection.f_regression(X, y, center=, force_finite=)`.
#[pyfunction]
#[pyo3(signature = (x, y, rows, cols, center=true, force_finite=true))]
pub fn f_regression(
    x: Vec<f64>,
    y: Vec<f64>,
    rows: usize,
    cols: usize,
    center: bool,
    force_finite: bool,
) -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
    check_geometry(x.len(), Some(y.len()), rows, cols)?;
    fsel::f_regression(&x, &y, rows, cols, center, force_finite)
        .map(split)
        .map_err(algo_err_to_py)
}

// ===========================================================================
// mutual_info_*
// ===========================================================================

/// Build [`MutualInfoParams`] from the shim's already-normalised arguments.
///
/// `discrete_features` arrives PRE-RESOLVED as `None` (sklearn's `"auto"`, which
/// for mlrs's dense-only ingress means all-continuous), `Some(bool)` for the
/// blanket form, or a mask — the shim resolves sklearn's index-array form into a
/// mask because it has numpy to hand, and resolving it twice would let the two
/// layers disagree.
fn mi_params(
    discrete_all: Option<bool>,
    discrete_mask: Option<Vec<bool>>,
    n_neighbors: usize,
    copy: bool,
    random_state: Option<u64>,
    n_jobs: Option<usize>,
) -> MutualInfoParams {
    let discrete_features = match (discrete_all, discrete_mask) {
        (_, Some(mask)) => DiscreteFeatures::Mask(mask),
        (Some(v), None) => DiscreteFeatures::All(v),
        (None, None) => DiscreteFeatures::Auto,
    };
    MutualInfoParams {
        discrete_features,
        n_neighbors,
        copy,
        random_state,
        n_jobs,
    }
}

/// `sklearn.feature_selection.mutual_info_classif(X, y, ...)`.
#[pyfunction]
#[pyo3(signature = (
    x, y, rows, cols,
    discrete_all=None, discrete_mask=None, n_neighbors=3, copy=true,
    random_state=None, n_jobs=None
))]
#[allow(clippy::too_many_arguments)]
pub fn mutual_info_classif(
    x: Vec<f64>,
    y: Vec<f64>,
    rows: usize,
    cols: usize,
    discrete_all: Option<bool>,
    discrete_mask: Option<Vec<bool>>,
    n_neighbors: usize,
    copy: bool,
    random_state: Option<u64>,
    n_jobs: Option<usize>,
) -> PyResult<Vec<f64>> {
    check_geometry(x.len(), Some(y.len()), rows, cols)?;
    let p = mi_params(
        discrete_all,
        discrete_mask,
        n_neighbors,
        copy,
        random_state,
        n_jobs,
    );
    fsel::mutual_info_classif(&x, &y, rows, cols, &p).map_err(algo_err_to_py)
}

/// `sklearn.feature_selection.mutual_info_regression(X, y, ...)`.
#[pyfunction]
#[pyo3(signature = (
    x, y, rows, cols,
    discrete_all=None, discrete_mask=None, n_neighbors=3, copy=true,
    random_state=None, n_jobs=None
))]
#[allow(clippy::too_many_arguments)]
pub fn mutual_info_regression(
    x: Vec<f64>,
    y: Vec<f64>,
    rows: usize,
    cols: usize,
    discrete_all: Option<bool>,
    discrete_mask: Option<Vec<bool>>,
    n_neighbors: usize,
    copy: bool,
    random_state: Option<u64>,
    n_jobs: Option<usize>,
) -> PyResult<Vec<f64>> {
    check_geometry(x.len(), Some(y.len()), rows, cols)?;
    let p = mi_params(
        discrete_all,
        discrete_mask,
        n_neighbors,
        copy,
        random_state,
        n_jobs,
    );
    fsel::mutual_info_regression(&x, &y, rows, cols, &p).map_err(algo_err_to_py)
}

// ===========================================================================
// The selectors' fit, returning fitted attributes AND the support mask
// ===========================================================================

/// `VarianceThreshold.fit(X)` → `(variances_, support_mask)`.
///
/// Reaches the same host sweep the Rust `VarianceThreshold` uses
/// (`prims::feature_score::col_moments`) through the module-level helpers rather
/// than through the typestate estimator, because the typestate `fit` takes a
/// `DeviceArray` and this path has no device operand — see the module docs. The
/// two therefore share the STATISTICS but not the wrapper; the Rust oracle test
/// covers the statistics.
///
/// NaN input is accepted here, alone among the selectors, because sklearn's
/// `VarianceThreshold` validates with `ensure_all_finite="allow-nan"` and
/// computes `np.nanvar`. The shim passes NaNs straight through for that reason.
#[pyfunction]
pub fn variance_threshold(
    x: Vec<f64>,
    rows: usize,
    cols: usize,
    threshold: f64,
) -> PyResult<(Vec<f64>, Vec<bool>)> {
    check_geometry(x.len(), None, rows, cols)?;
    fsel::variance_threshold::variances_and_support(&x, rows, cols, threshold)
        .map_err(algo_err_to_py)
}

/// Resolve the shim's `score_func` NAME (plus its options) into a [`ScoreFunc`].
///
/// A name rather than a callback: a Python callable crossing back into Rust per
/// `fit` would need the GIL re-acquired inside the estimator, and every built-in
/// sklearn `score_func` is one of these seven. A caller passing a genuinely
/// custom callable is handled ENTIRELY in the shim — it calls the callable
/// itself and uses [`univariate_select_from_scores`], so the compiled side never
/// needs to call Python.
#[allow(clippy::too_many_arguments)]
fn resolve_score_func(
    name: &str,
    center: bool,
    force_finite: bool,
    discrete_all: Option<bool>,
    discrete_mask: Option<Vec<bool>>,
    n_neighbors: usize,
    random_state: Option<u64>,
    n_jobs: Option<usize>,
) -> PyResult<ScoreFunc> {
    let mi = || {
        mi_params(
            discrete_all,
            discrete_mask.clone(),
            n_neighbors,
            true,
            random_state,
            n_jobs,
        )
    };
    match name {
        "f_classif" => Ok(ScoreFunc::FClassif),
        "chi2" => Ok(ScoreFunc::Chi2),
        "r_regression" => Ok(ScoreFunc::RRegression {
            center,
            force_finite,
        }),
        "f_regression" => Ok(ScoreFunc::FRegression {
            center,
            force_finite,
        }),
        "mutual_info_classif" => Ok(ScoreFunc::MutualInfoClassif(mi())),
        "mutual_info_regression" => Ok(ScoreFunc::MutualInfoRegression(mi())),
        other => Err(PyValueError::new_err(format!(
            "mlrs.feature_selection: unknown built-in score_func '{other}'"
        ))),
    }
}

/// Turn the shim's `(mode, param)` pair into a [`GenericParam`].
fn resolve_param(param: Option<f64>) -> GenericParam {
    match param {
        Some(v) => GenericParam::Value(v),
        // `None` is how the shim spells sklearn's `"all"`, which is the only
        // non-numeric `param`/`k` value sklearn accepts.
        None => GenericParam::All,
    }
}

/// The univariate filters' `fit` → `(scores_, pvalues_, support_mask)`.
///
/// `mode` is `GenericUnivariateSelect`'s mode string, and the five specific
/// sklearn classes are the shim passing their own mode — which is sklearn's own
/// factoring (`GenericUnivariateSelect._make_selector`), so nothing is lost.
#[pyfunction]
#[pyo3(signature = (
    x, y, rows, cols, mode, param, score_func,
    center=true, force_finite=true,
    discrete_all=None, discrete_mask=None, n_neighbors=3,
    random_state=None, n_jobs=None
))]
#[allow(clippy::too_many_arguments)]
pub fn univariate_select(
    x: Vec<f64>,
    y: Vec<f64>,
    rows: usize,
    cols: usize,
    mode: &str,
    param: Option<f64>,
    score_func: &str,
    center: bool,
    force_finite: bool,
    discrete_all: Option<bool>,
    discrete_mask: Option<Vec<bool>>,
    n_neighbors: usize,
    random_state: Option<u64>,
    n_jobs: Option<usize>,
) -> PyResult<(Vec<f64>, Option<Vec<f64>>, Vec<bool>)> {
    check_geometry(x.len(), Some(y.len()), rows, cols)?;
    let sf = resolve_score_func(
        score_func,
        center,
        force_finite,
        discrete_all,
        discrete_mask,
        n_neighbors,
        random_state,
        n_jobs,
    )?;
    fsel::univariate::fit_host(&x, &y, rows, cols, mode, resolve_param(param), sf)
        .map_err(algo_err_to_py)
}

/// The mask half alone, for a caller-supplied `score_func`.
///
/// The shim calls a custom Python callable itself and hands the resulting
/// `(scores, pvalues)` here, so the selection RULE — the part with sklearn's
/// stable-sort tie-breaking, percentile interpolation and Benjamini-Hochberg
/// step in it — is still the single Rust implementation the oracle tests cover,
/// while the callable never has to be invoked from Rust.
#[pyfunction]
#[pyo3(signature = (scores, pvalues, mode, param))]
pub fn univariate_select_from_scores(
    scores: Vec<f64>,
    pvalues: Option<Vec<f64>>,
    mode: &str,
    param: Option<f64>,
) -> PyResult<Vec<bool>> {
    fsel::univariate::mask_from_scores(&scores, pvalues.as_deref(), mode, resolve_param(param))
        .map_err(algo_err_to_py)
}
