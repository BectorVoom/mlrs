//! PyO3 free-function surface for `mlrs.metrics` (METR-BIND-01, TASK-15).
//!
//! One `#[pyfunction]` per `mlrs_algos::metrics::{classification,regression}`
//! function, taking plain `Vec<i32>`/`Vec<f64>` (labels / targets / proba /
//! scores / sample_weight) — NOT the arrow capsule (host-only + integer
//! labels; the capsule ingress is float-only, `crates/mlrs-py/src/ingress.rs:112-118`).
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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mlrs_algos::metrics::classification as cls;
use mlrs_algos::metrics::regression as reg;
use mlrs_algos::metrics::{Average, MultiClass, ZeroDivision};

use crate::errors::metric_err_to_py;

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

/// `average` restricted to `{macro, weighted}` for multiclass `roc_auc_score`
/// (SPEC §4: `average ∈ {macro, weighted}` for multiclass roc_auc_score).
fn ovr_ovo_average_from_str(average: &str) -> PyResult<Average> {
    match average {
        "macro" => Ok(Average::Macro),
        "weighted" => Ok(Average::Weighted),
        other => Err(PyValueError::new_err(format!(
            "unknown average '{other}' (multiclass roc_auc_score accepts 'macro'/'weighted' only)"
        ))),
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
    y_true: Vec<i32>,
    y_pred: Vec<i32>,
    sample_weight: Option<Vec<f64>>,
    normalize: bool,
) -> PyResult<f64> {
    check_same_len(y_true.len(), y_pred.len(), "accuracy_score")?;
    cls::accuracy_score(&y_true, &y_pred, sample_weight.as_deref(), normalize)
        .map_err(metric_err_to_py)
}

// ==================== confusion_matrix (METR-CLS-02) ====================

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, labels=None, sample_weight=None))]
pub fn confusion_matrix(
    y_true: Vec<i32>,
    y_pred: Vec<i32>,
    labels: Option<Vec<i32>>,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    check_same_len(y_true.len(), y_pred.len(), "confusion_matrix")?;
    cls::confusion_matrix(&y_true, &y_pred, labels.as_deref(), sample_weight.as_deref())
        .map_err(metric_err_to_py)
}

// ==================== precision/recall/f1 (METR-CLS-03/04/05) ====================

macro_rules! prf_pyfunctions {
    ($scalar_fn:ident, $per_class_fn:ident, $algos_fn:path) => {
        #[pyfunction]
        #[pyo3(signature = (y_true, y_pred, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division=0.0))]
        pub fn $scalar_fn(
            y_true: Vec<i32>,
            y_pred: Vec<i32>,
            labels: Option<Vec<i32>>,
            pos_label: i32,
            average: &str,
            sample_weight: Option<Vec<f64>>,
            zero_division: f64,
        ) -> PyResult<f64> {
            check_same_len(y_true.len(), y_pred.len(), stringify!($scalar_fn))?;
            let avg = average_from_str(average)?;
            let zd = zero_division_from_f64(zero_division);
            match $algos_fn(&y_true, &y_pred, labels.as_deref(), pos_label, avg, sample_weight.as_deref(), zd)
                .map_err(metric_err_to_py)?
            {
                mlrs_algos::metrics::PrfOut::Scalar(v) => Ok(v),
                mlrs_algos::metrics::PrfOut::PerClass(_) => unreachable!(
                    "average_from_str rejects 'none'; PerClass cannot be produced here"
                ),
            }
        }

        #[pyfunction]
        #[pyo3(signature = (y_true, y_pred, labels=None, sample_weight=None, zero_division=0.0))]
        pub fn $per_class_fn(
            y_true: Vec<i32>,
            y_pred: Vec<i32>,
            labels: Option<Vec<i32>>,
            sample_weight: Option<Vec<f64>>,
            zero_division: f64,
        ) -> PyResult<Vec<f64>> {
            check_same_len(y_true.len(), y_pred.len(), stringify!($per_class_fn))?;
            let zd = zero_division_from_f64(zero_division);
            match $algos_fn(&y_true, &y_pred, labels.as_deref(), 1, Average::None_, sample_weight.as_deref(), zd)
                .map_err(metric_err_to_py)?
            {
                mlrs_algos::metrics::PrfOut::PerClass(v) => Ok(v),
                mlrs_algos::metrics::PrfOut::Scalar(_) => {
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
#[pyo3(signature = (y_true, y_prob, n_classes, labels=None, sample_weight=None, eps=f64::EPSILON, normalize=true))]
#[allow(clippy::too_many_arguments)]
pub fn log_loss(
    y_true: Vec<i32>,
    y_prob: Vec<f64>,
    n_classes: usize,
    labels: Option<Vec<i32>>,
    sample_weight: Option<Vec<f64>>,
    eps: f64,
    normalize: bool,
) -> PyResult<f64> {
    if y_prob.len() != y_true.len() * n_classes {
        return Err(PyValueError::new_err(format!(
            "log_loss: y_prob length {} != y_true.len() ({}) * n_classes ({})",
            y_prob.len(),
            y_true.len(),
            n_classes
        )));
    }
    cls::log_loss(
        &y_true,
        &y_prob,
        n_classes,
        labels.as_deref(),
        sample_weight.as_deref(),
        eps,
        normalize,
    )
    .map_err(metric_err_to_py)
}

// ==================== roc_auc_score (METR-CLS-07/08) ====================

#[pyfunction]
#[pyo3(signature = (y_true, y_score, pos_label=1, sample_weight=None))]
pub fn roc_auc_score_binary(
    y_true: Vec<i32>,
    y_score: Vec<f64>,
    pos_label: i32,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<f64> {
    check_same_len(y_true.len(), y_score.len(), "roc_auc_score_binary")?;
    cls::roc_auc_score_binary(&y_true, &y_score, pos_label, sample_weight.as_deref())
        .map_err(metric_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_score, n_classes, multi_class="ovr", average="macro", sample_weight=None))]
pub fn roc_auc_score_multiclass(
    y_true: Vec<i32>,
    y_score: Vec<f64>,
    n_classes: usize,
    multi_class: &str,
    average: &str,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<f64> {
    let mc = multi_class_from_str(multi_class)?;
    let avg = ovr_ovo_average_from_str(average)?;
    cls::roc_auc_score_multiclass(
        &y_true,
        &y_score,
        n_classes,
        mc,
        avg,
        sample_weight.as_deref(),
    )
    .map_err(metric_err_to_py)
}

// ==================== precision_recall_curve (METR-CLS-09) ====================

#[pyfunction]
#[pyo3(signature = (y_true, probas_pred, pos_label=1, sample_weight=None))]
pub fn precision_recall_curve(
    y_true: Vec<i32>,
    probas_pred: Vec<f64>,
    pos_label: i32,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    check_same_len(y_true.len(), probas_pred.len(), "precision_recall_curve")?;
    cls::precision_recall_curve(&y_true, &probas_pred, pos_label, sample_weight.as_deref())
        .map_err(metric_err_to_py)
}

// ==================== r2_score / mean_squared_error / mean_absolute_error (METR-REG-01/02/03) ====================

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None))]
pub fn r2_score(
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<f64> {
    check_same_len(y_true.len(), y_pred.len(), "r2_score")?;
    reg::r2_score::<f64>(&y_true, &y_pred, sample_weight.as_deref()).map_err(metric_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None))]
pub fn mean_squared_error(
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<f64> {
    check_same_len(y_true.len(), y_pred.len(), "mean_squared_error")?;
    reg::mean_squared_error::<f64>(&y_true, &y_pred, sample_weight.as_deref())
        .map_err(metric_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None))]
pub fn mean_absolute_error(
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
    sample_weight: Option<Vec<f64>>,
) -> PyResult<f64> {
    check_same_len(y_true.len(), y_pred.len(), "mean_absolute_error")?;
    reg::mean_absolute_error::<f64>(&y_true, &y_pred, sample_weight.as_deref())
        .map_err(metric_err_to_py)
}
