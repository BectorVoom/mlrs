//! PyO3 free-function surface for `mlrs.metrics` (METR-BIND-01, TASK-15;
//! parameters extended by METR-PARAM-01).
//!
//! One `#[pyfunction]` per `mlrs_algos::metrics::{classification,regression}`
//! function. Every O(n) input (labels, targets, scores, probabilities,
//! `sample_weight`) crosses as a **pyarrow `float64` array capsule** through
//! the same [`crate::ingress`] gauntlet the estimators use; only the O(K)
//! parameters (`labels`, `classes`, per-output `multioutput` weights) cross as
//! plain `Vec`s, where the element-by-element cost is a rounding error.
//!
//! ## Why the capsule, and why labels are floats
//!
//! PyO3's `Vec<T>` extraction walks the Python sequence protocol ELEMENT BY
//! ELEMENT. Measured on this surface: **~44 ns/element**, i.e. 44 ms of
//! ingress for a one-million-sample `mean_squared_error` whose actual
//! reduction is ~1 ms — sklearn ran the same call in 2.8 ms, so the binding,
//! not the algorithm, was the entire performance story (the ingress sibling of
//! the egress pathology [`crate::egress::f32_vec_to_pyarrow`] documents). The
//! capsule is zero-copy: `pa.array(numpy_f64_array)` measures ~1 µs at ANY
//! length.
//!
//! Labels cross as `float64` rather than as an integer array because the
//! `mlrs_backend::bridge` ingress is float-only apart from the `uint32` label
//! column `VotingClassifier` uses, and metric labels can be NEGATIVE (a
//! `{-1, 1}` target, `pos_label=-1`). The values are integral, so the shim's
//! `astype(np.float64)` is exact and the `f64 -> i32` round on this side is
//! lossless — one cheap pass on top of a zero-copy transfer, versus the
//! per-element boxing it replaces.
//!
//! `average=None`'s per-class output has no existing polymorphic
//! (float-or-list) PyO3 return precedent in this codebase, so it is bound as
//! a SEPARATE `..._per_class` function (mirroring the `predict_proba_f32`/
//! `predict_proba_f64` dtype-suffix-split convention,
//! `crates/mlrs-py/src/estimators/neighbors.rs:257,270`) rather than an
//! invented union return type.
//!
//! `MetricError` maps to `PyValueError` via [`crate::errors::metric_err_to_py`]
//! (a sibling of `algo_err_to_py`, which only accepts `AlgoError` — a
//! distinct type).

use arrow::array::ArrayRef;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mlrs_algos::metrics::classification as cls;
use mlrs_algos::metrics::regression as reg;
use mlrs_algos::metrics::{
    Average, MetricOut, MultiClass, MultiOutput, Normalize, PrfOut, ZeroDivision,
};

use crate::egress::f64_vec_to_pyarrow;
use crate::errors::metric_err_to_py;
use crate::ingress::{as_f64, capsule_to_array, host_slice_f64};

// --------------------------------------------------------------------------- //
// ingress helpers
// --------------------------------------------------------------------------- //

/// Import a pyarrow array capsule (owned; offset/nulls/alignment hard-rejected
/// when its values are borrowed — see [`crate::ingress`]). The returned
/// [`ArrayRef`] must outlive every slice borrowed from it, which is why each
/// binding below holds it in a local before calling [`f64_slice`]/[`labels`].
fn capsule(x: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    capsule_to_array(x)
}

/// Optional-capsule twin of [`capsule`], for `sample_weight`.
fn capsule_opt(x: Option<&Bound<'_, PyAny>>) -> PyResult<Option<ArrayRef>> {
    x.map(capsule).transpose()
}

/// Borrow an imported capsule's `float64` values.
fn f64_slice(array: &ArrayRef) -> PyResult<&[f64]> {
    host_slice_f64(as_f64(array)?)
}

/// Borrow an imported capsule and round its (integral) `float64` values to the
/// `i32` labels the algos layer takes.
fn labels(array: &ArrayRef) -> PyResult<Vec<i32>> {
    Ok(f64_slice(array)?
        .iter()
        .map(|&v| v.round() as i32)
        .collect())
}

// --------------------------------------------------------------------------- //
// parameter parsing
// --------------------------------------------------------------------------- //

/// Parse the `average` string crossing the FFI boundary. `"none"` is
/// rejected here (callers wanting `average=None` use the `..._per_class`
/// sibling function instead, TASK-15 resolved-decision).
fn average_from_str(average: &str) -> PyResult<Average> {
    match average {
        "binary" => Ok(Average::Binary),
        "macro" => Ok(Average::Macro),
        "micro" => Ok(Average::Micro),
        "weighted" => Ok(Average::Weighted),
        "none" => Err(PyValueError::new_err(
            "average='none' is not valid here; call the '..._per_class' variant instead",
        )),
        other => Err(PyValueError::new_err(format!("unknown average '{other}'"))),
    }
}

/// `average` for multiclass `roc_auc_score` (METR-PARAM-01): `macro` /
/// `weighted` for both strategies, plus `micro` for OvR. `average=None` uses
/// the `..._per_class` sibling, so `"none"` is rejected here exactly as
/// [`average_from_str`] rejects it. Whether `micro` is actually legal for the
/// requested `multi_class` is enforced by the algos layer
/// (`MetricError::UnsupportedAverage`), so the rule lives in ONE place.
fn ovr_ovo_average_from_str(average: &str) -> PyResult<Average> {
    match average {
        "macro" => Ok(Average::Macro),
        "weighted" => Ok(Average::Weighted),
        "micro" => Ok(Average::Micro),
        "none" => Err(PyValueError::new_err(
            "average='none' is not valid here; call 'roc_auc_score_multiclass_per_class' instead",
        )),
        other => Err(PyValueError::new_err(format!(
            "unknown average '{other}' (multiclass roc_auc_score accepts 'macro'/'weighted'/'micro')"
        ))),
    }
}

/// Parse `confusion_matrix`'s `normalize` string (METR-PARAM-01). `None` (the
/// Python `normalize=None` default) stays `None` — it is not a string value.
fn normalize_from_str(normalize: Option<&str>) -> PyResult<Option<Normalize>> {
    match normalize {
        None => Ok(None),
        Some("true") => Ok(Some(Normalize::True_)),
        Some("pred") => Ok(Some(Normalize::Pred)),
        Some("all") => Ok(Some(Normalize::All)),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown normalize '{other}' (expected 'true', 'pred', 'all' or None)"
        ))),
    }
}

/// Parse the regression `multioutput` string, or wrap a caller-supplied
/// per-output weight vector (METR-PARAM-01). The weights are consulted only
/// for the `"weights"` sentinel the Python shim sends for sklearn's array-like
/// form.
fn multioutput_from<'a>(
    multioutput: &str,
    weights: Option<&'a [f64]>,
) -> PyResult<MultiOutput<'a>> {
    match multioutput {
        "raw_values" => Ok(MultiOutput::RawValues),
        "uniform_average" => Ok(MultiOutput::UniformAverage),
        "variance_weighted" => Ok(MultiOutput::VarianceWeighted),
        "weights" => weights
            .map(MultiOutput::Weights)
            .ok_or_else(|| PyValueError::new_err("multioutput='weights' requires a weight vector")),
        other => Err(PyValueError::new_err(format!(
            "unknown multioutput '{other}'"
        ))),
    }
}

/// Unwrap a regression [`MetricOut`] into the flat `Vec<f64>` the Python shim
/// reshapes: a one-element vector for the reduced (scalar) `multioutput`
/// values, the per-output vector for `raw_values`. The shim knows which form
/// it asked for, so no union return type is needed at the boundary — and the
/// vector is O(n_outputs), never O(n_samples), so a plain `Vec` egress is the
/// right shape here.
fn metric_out_to_vec(out: MetricOut) -> Vec<f64> {
    match out {
        MetricOut::Scalar(v) => vec![v],
        MetricOut::Raw(v) => v,
    }
}

fn multi_class_from_str(multi_class: &str) -> PyResult<MultiClass> {
    match multi_class {
        "ovr" => Ok(MultiClass::Ovr),
        "ovo" => Ok(MultiClass::Ovo),
        other => Err(PyValueError::new_err(format!(
            "unknown multi_class '{other}'"
        ))),
    }
}

/// `zero_division` crosses the FFI boundary as `f64`: `0.0` → `Zero`, `1.0`
/// → `One`, `NaN` → `Nan` (TASK-15 resolved-decision; any other finite value
/// falls back to `Zero`, sklearn's own default).
fn zero_division_from_f64(zero_division: f64) -> ZeroDivision {
    if zero_division.is_nan() {
        ZeroDivision::Nan
    } else if zero_division == 1.0 {
        ZeroDivision::One
    } else {
        ZeroDivision::Zero
    }
}

fn check_same_len(a: usize, b: usize, what: &str) -> PyResult<()> {
    if a != b {
        return Err(PyValueError::new_err(format!(
            "{what}: mismatched lengths ({a} vs {b})"
        )));
    }
    Ok(())
}

// ==================== accuracy_score (METR-CLS-01) ====================

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, normalize=true))]
pub fn accuracy_score(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    sample_weight: Option<&Bound<'_, PyAny>>,
    normalize: bool,
) -> PyResult<f64> {
    let (yt_a, yp_a, sw_a) = (
        capsule(y_true)?,
        capsule(y_pred)?,
        capsule_opt(sample_weight)?,
    );
    let (yt, yp) = (labels(&yt_a)?, labels(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    check_same_len(yt.len(), yp.len(), "accuracy_score")?;
    cls::accuracy_score(&yt, &yp, sw, normalize).map_err(metric_err_to_py)
}

// ==================== confusion_matrix (METR-CLS-02) ====================

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, class_labels=None, sample_weight=None, normalize=None))]
pub fn confusion_matrix(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    class_labels: Option<Vec<i32>>,
    sample_weight: Option<&Bound<'_, PyAny>>,
    normalize: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    let (yt_a, yp_a, sw_a) = (
        capsule(y_true)?,
        capsule(y_pred)?,
        capsule_opt(sample_weight)?,
    );
    let (yt, yp) = (labels(&yt_a)?, labels(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    check_same_len(yt.len(), yp.len(), "confusion_matrix")?;
    cls::confusion_matrix(
        &yt,
        &yp,
        class_labels.as_deref(),
        sw,
        normalize_from_str(normalize)?,
    )
    .map_err(metric_err_to_py)
}

// ==================== precision/recall/f1 (METR-CLS-03/04/05) ====================
//
// Both members of each pair return `(value, zero_division_hit)`: the flag is
// what lets the shim raise sklearn's `zero_division="warn"`
// UndefinedMetricWarning WITHOUT a second O(n) pass to discover that a
// denominator was zero.

macro_rules! prf_pyfunctions {
    ($scalar_fn:ident, $per_class_fn:ident, $algos_fn:path) => {
        #[pyfunction]
        #[pyo3(signature = (y_true, y_pred, class_labels=None, pos_label=1, average="binary", sample_weight=None, zero_division=0.0))]
        #[allow(clippy::too_many_arguments)]
        pub fn $scalar_fn(
            y_true: &Bound<'_, PyAny>,
            y_pred: &Bound<'_, PyAny>,
            class_labels: Option<Vec<i32>>,
            pos_label: i32,
            average: &str,
            sample_weight: Option<&Bound<'_, PyAny>>,
            zero_division: f64,
        ) -> PyResult<(f64, bool, Vec<i32>)> {
            let (yt_a, yp_a, sw_a) = (
                capsule(y_true)?,
                capsule(y_pred)?,
                capsule_opt(sample_weight)?,
            );
            let (yt, yp) = (labels(&yt_a)?, labels(&yp_a)?);
            let sw = sw_a.as_ref().map(f64_slice).transpose()?;
            check_same_len(yt.len(), yp.len(), stringify!($scalar_fn))?;
            let avg = average_from_str(average)?;
            let zd = zero_division_from_f64(zero_division);
            let got = $algos_fn(&yt, &yp, class_labels.as_deref(), pos_label, avg, sw, zd)
                .map_err(metric_err_to_py)?;
            match got.out {
                PrfOut::Scalar(v) => Ok((v, got.zero_division_hit, got.classes)),
                PrfOut::PerClass(_) => unreachable!(
                    "average_from_str rejects 'none'; PerClass cannot be produced here"
                ),
            }
        }

        #[pyfunction]
        #[pyo3(signature = (y_true, y_pred, class_labels=None, sample_weight=None, zero_division=0.0))]
        pub fn $per_class_fn(
            y_true: &Bound<'_, PyAny>,
            y_pred: &Bound<'_, PyAny>,
            class_labels: Option<Vec<i32>>,
            sample_weight: Option<&Bound<'_, PyAny>>,
            zero_division: f64,
        ) -> PyResult<(Vec<f64>, bool, Vec<i32>)> {
            let (yt_a, yp_a, sw_a) = (
                capsule(y_true)?,
                capsule(y_pred)?,
                capsule_opt(sample_weight)?,
            );
            let (yt, yp) = (labels(&yt_a)?, labels(&yp_a)?);
            let sw = sw_a.as_ref().map(f64_slice).transpose()?;
            check_same_len(yt.len(), yp.len(), stringify!($per_class_fn))?;
            let zd = zero_division_from_f64(zero_division);
            let got = $algos_fn(&yt, &yp, class_labels.as_deref(), 1, Average::None_, sw, zd)
                .map_err(metric_err_to_py)?;
            match got.out {
                PrfOut::PerClass(v) => Ok((v, got.zero_division_hit, got.classes)),
                PrfOut::Scalar(_) => {
                    unreachable!("Average::None_ always produces PerClass")
                }
            }
        }
    };
}

prf_pyfunctions!(
    precision_score,
    precision_score_per_class,
    cls::precision_score
);
prf_pyfunctions!(recall_score, recall_score_per_class, cls::recall_score);
prf_pyfunctions!(f1_score, f1_score_per_class, cls::f1_score);

// ==================== log_loss (METR-CLS-06) ====================

#[pyfunction]
#[pyo3(signature = (y_true, y_prob, n_classes, class_labels=None, sample_weight=None, eps=f64::EPSILON, normalize=true))]
#[allow(clippy::too_many_arguments)]
pub fn log_loss(
    y_true: &Bound<'_, PyAny>,
    y_prob: &Bound<'_, PyAny>,
    n_classes: usize,
    class_labels: Option<Vec<i32>>,
    sample_weight: Option<&Bound<'_, PyAny>>,
    eps: f64,
    normalize: bool,
) -> PyResult<f64> {
    let (yt_a, yp_a, sw_a) = (
        capsule(y_true)?,
        capsule(y_prob)?,
        capsule_opt(sample_weight)?,
    );
    let yt = labels(&yt_a)?;
    let prob = f64_slice(&yp_a)?;
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    if prob.len() != yt.len() * n_classes {
        return Err(PyValueError::new_err(format!(
            "log_loss: y_prob length {} != y_true.len() ({}) * n_classes ({})",
            prob.len(),
            yt.len(),
            n_classes
        )));
    }
    cls::log_loss(
        &yt,
        prob,
        n_classes,
        class_labels.as_deref(),
        sw,
        eps,
        normalize,
    )
    .map_err(metric_err_to_py)
}

// ==================== roc_auc_score (METR-CLS-07/08) ====================

#[pyfunction]
#[pyo3(signature = (y_true, y_score, pos_label=1, sample_weight=None, max_fpr=None))]
pub fn roc_auc_score_binary(
    y_true: &Bound<'_, PyAny>,
    y_score: &Bound<'_, PyAny>,
    pos_label: i32,
    sample_weight: Option<&Bound<'_, PyAny>>,
    max_fpr: Option<f64>,
) -> PyResult<f64> {
    let (yt_a, ys_a, sw_a) = (
        capsule(y_true)?,
        capsule(y_score)?,
        capsule_opt(sample_weight)?,
    );
    let yt = labels(&yt_a)?;
    let ys = f64_slice(&ys_a)?;
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    check_same_len(yt.len(), ys.len(), "roc_auc_score_binary")?;
    cls::roc_auc_score_binary(&yt, ys, pos_label, sw, max_fpr).map_err(metric_err_to_py)
}

/// Scalar multiclass `roc_auc_score` (`average ∈ {macro, weighted, micro}`).
/// `classes` is the resolved class order — column `c` of the row-major
/// `y_score` belongs to `classes[c]` (METR-PARAM-01: sklearn's `labels`, or
/// the sorted unique of `y_true`).
#[pyfunction]
#[pyo3(signature = (y_true, y_score, classes, multi_class="ovr", average="macro", sample_weight=None))]
pub fn roc_auc_score_multiclass(
    y_true: &Bound<'_, PyAny>,
    y_score: &Bound<'_, PyAny>,
    classes: Vec<i32>,
    multi_class: &str,
    average: &str,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (yt_a, ys_a, sw_a) = (
        capsule(y_true)?,
        capsule(y_score)?,
        capsule_opt(sample_weight)?,
    );
    let yt = labels(&yt_a)?;
    let ys = f64_slice(&ys_a)?;
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let mc = multi_class_from_str(multi_class)?;
    let avg = ovr_ovo_average_from_str(average)?;
    match cls::roc_auc_score_multiclass(&yt, ys, &classes, mc, avg, sw).map_err(metric_err_to_py)? {
        PrfOut::Scalar(v) => Ok(v),
        PrfOut::PerClass(_) => unreachable!(
            "ovr_ovo_average_from_str rejects 'none'; PerClass cannot be produced here"
        ),
    }
}

/// `average=None` multiclass `roc_auc_score`: the per-class OvR AUC vector, in
/// the `classes` order (METR-PARAM-01). OvO has no `average=None` form —
/// sklearn raises `NotImplementedError`, and the algos layer returns
/// `MetricError::UnsupportedAverage`, which surfaces here as a `ValueError`.
#[pyfunction]
#[pyo3(signature = (y_true, y_score, classes, multi_class="ovr", sample_weight=None))]
pub fn roc_auc_score_multiclass_per_class(
    y_true: &Bound<'_, PyAny>,
    y_score: &Bound<'_, PyAny>,
    classes: Vec<i32>,
    multi_class: &str,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<f64>> {
    let (yt_a, ys_a, sw_a) = (
        capsule(y_true)?,
        capsule(y_score)?,
        capsule_opt(sample_weight)?,
    );
    let yt = labels(&yt_a)?;
    let ys = f64_slice(&ys_a)?;
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let mc = multi_class_from_str(multi_class)?;
    match cls::roc_auc_score_multiclass(&yt, ys, &classes, mc, Average::None_, sw)
        .map_err(metric_err_to_py)?
    {
        PrfOut::PerClass(v) => Ok(v),
        PrfOut::Scalar(_) => unreachable!("Average::None_ always produces PerClass"),
    }
}

// ==================== precision_recall_curve (METR-CLS-09) ====================

/// The three curve columns come back as pyarrow arrays, not Python lists: they
/// are O(distinct scores) long — O(n) in the worst case — and boxing a
/// million-point curve into `PyFloat`s would cost more than computing it (the
/// pathology [`crate::egress::f32_vec_to_pyarrow`] documents). It also makes
/// `drop_intermediate=True` pay twice: a shorter curve is a smaller egress as
/// well as less arithmetic.
#[pyfunction]
#[pyo3(signature = (y_true, probas_pred, pos_label=1, sample_weight=None, drop_intermediate=false))]
pub fn precision_recall_curve<'py>(
    py: Python<'py>,
    y_true: &Bound<'py, PyAny>,
    probas_pred: &Bound<'py, PyAny>,
    pos_label: i32,
    sample_weight: Option<&Bound<'py, PyAny>>,
    drop_intermediate: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (yt_a, ps_a, sw_a) = (
        capsule(y_true)?,
        capsule(probas_pred)?,
        capsule_opt(sample_weight)?,
    );
    let yt = labels(&yt_a)?;
    let ps = f64_slice(&ps_a)?;
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    check_same_len(yt.len(), ps.len(), "precision_recall_curve")?;
    let (precision, recall, thresholds) =
        cls::precision_recall_curve(&yt, ps, pos_label, sw, drop_intermediate)
            .map_err(metric_err_to_py)?;
    Ok((
        f64_vec_to_pyarrow(py, precision)?,
        f64_vec_to_pyarrow(py, recall)?,
        f64_vec_to_pyarrow(py, thresholds)?,
    ))
}

// ==================== r2_score / mean_squared_error / mean_absolute_error ====================
// (METR-REG-01/02/03 + METR-PARAM-01's `multioutput`/`force_finite`.)
//
// Every regression binding takes the ROW-MAJOR `n_samples × n_outputs` buffer
// flat plus its `n_outputs`, and returns a flat `Vec<f64>` — one element for a
// reduced `multioutput`, `n_outputs` for `raw_values` (see
// [`metric_out_to_vec`]). `multioutput_weights` carries sklearn's array-like
// `multioutput=[w0, w1, ...]` form, which the shim signals with the
// `multioutput="weights"` sentinel.

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_outputs=1, sample_weight=None, multioutput="uniform_average", multioutput_weights=None, force_finite=true))]
#[allow(clippy::too_many_arguments)]
pub fn r2_score(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    n_outputs: usize,
    sample_weight: Option<&Bound<'_, PyAny>>,
    multioutput: &str,
    multioutput_weights: Option<Vec<f64>>,
    force_finite: bool,
) -> PyResult<Vec<f64>> {
    let (yt_a, yp_a, sw_a) = (
        capsule(y_true)?,
        capsule(y_pred)?,
        capsule_opt(sample_weight)?,
    );
    let (yt, yp) = (f64_slice(&yt_a)?, f64_slice(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    check_same_len(yt.len(), yp.len(), "r2_score")?;
    let mo = multioutput_from(multioutput, multioutput_weights.as_deref())?;
    reg::r2_score::<f64>(yt, yp, n_outputs, sw, mo, force_finite)
        .map(metric_out_to_vec)
        .map_err(metric_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_outputs=1, sample_weight=None, multioutput="uniform_average", multioutput_weights=None))]
pub fn mean_squared_error(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    n_outputs: usize,
    sample_weight: Option<&Bound<'_, PyAny>>,
    multioutput: &str,
    multioutput_weights: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let (yt_a, yp_a, sw_a) = (
        capsule(y_true)?,
        capsule(y_pred)?,
        capsule_opt(sample_weight)?,
    );
    let (yt, yp) = (f64_slice(&yt_a)?, f64_slice(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    check_same_len(yt.len(), yp.len(), "mean_squared_error")?;
    let mo = multioutput_from(multioutput, multioutput_weights.as_deref())?;
    reg::mean_squared_error::<f64>(yt, yp, n_outputs, sw, mo)
        .map(metric_out_to_vec)
        .map_err(metric_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_outputs=1, sample_weight=None, multioutput="uniform_average", multioutput_weights=None))]
pub fn mean_absolute_error(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    n_outputs: usize,
    sample_weight: Option<&Bound<'_, PyAny>>,
    multioutput: &str,
    multioutput_weights: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let (yt_a, yp_a, sw_a) = (
        capsule(y_true)?,
        capsule(y_pred)?,
        capsule_opt(sample_weight)?,
    );
    let (yt, yp) = (f64_slice(&yt_a)?, f64_slice(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    check_same_len(yt.len(), yp.len(), "mean_absolute_error")?;
    let mo = multioutput_from(multioutput, multioutput_weights.as_deref())?;
    reg::mean_absolute_error::<f64>(yt, yp, n_outputs, sw, mo)
        .map(metric_out_to_vec)
        .map_err(metric_err_to_py)
}
