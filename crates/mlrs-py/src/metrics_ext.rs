//! PyO3 bindings for the rest of `sklearn.metrics` — everything beyond the
//! eleven METR-PARAM-01 metrics in [`crate::metrics`]: the remaining
//! classification and regression scores, the ranking/threshold metrics, the
//! clustering metrics, and the pairwise distance/kernel engine.
//!
//! Same conventions as [`crate::metrics`]:
//!
//! * every O(n) input crosses as a zero-copy pyarrow `float64` capsule (labels
//!   included, rounded back to integers here — exact for any integral value
//!   below 2⁵³);
//! * O(K)/O(n_outputs) parameters cross as plain `Vec`s;
//! * O(n) or O(n²) OUTPUTS (curves, pairwise matrices, per-sample scores) leave
//!   as pyarrow arrays, never as boxed Python lists;
//! * the Python shim (`mlrs/metrics/`) owns sklearn's messages, warnings and
//!   output shapes; every VALUE is computed in `mlrs_algos::metrics`.
//!
//! The multilabel ("indicator") forms take the row-major `n × n_labels` 0/1
//! matrix flat plus its `n_labels`; `n_labels = 0` selects the 1-D form.
//!
//! [`register`] adds every function here to the `_mlrs` module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mlrs_algos::metrics::classification as cls;
use mlrs_algos::metrics::cluster as clu;
use mlrs_algos::metrics::pairwise::{self as pw, Distance, Kernel};
use mlrs_algos::metrics::prfs::{self as prf_core, SetAverage};
use mlrs_algos::metrics::ranking::{self as rk, ScoreAverage};
use mlrs_algos::metrics::regression as reg;
use mlrs_algos::metrics::{MetricOut, PrfOut};

use crate::egress::{f64_vec_to_pyarrow, i64_vec_to_pyarrow};
use crate::errors::metric_err_to_py;
use crate::metrics::{
    capsule, capsule_opt, f64_slice, labels, metric_out_to_vec, multioutput_from, zero_division_from_f64,
};

type Py3<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>);

fn set_average_from_str(average: &str) -> PyResult<SetAverage> {
    Ok(match average {
        "none" => SetAverage::None_,
        "binary" => SetAverage::Binary,
        "micro" => SetAverage::Micro,
        "macro" => SetAverage::Macro,
        "weighted" => SetAverage::Weighted,
        "samples" => SetAverage::Samples,
        other => return Err(PyValueError::new_err(format!("unknown average '{other}'"))),
    })
}

fn score_average_from_str(average: &str) -> PyResult<ScoreAverage> {
    Ok(match average {
        "none" => ScoreAverage::None_,
        "micro" => ScoreAverage::Micro,
        "macro" => ScoreAverage::Macro,
        "weighted" => ScoreAverage::Weighted,
        "samples" => ScoreAverage::Samples,
        other => return Err(PyValueError::new_err(format!("unknown average '{other}'"))),
    })
}

fn prf_vec(out: PrfOut) -> Vec<f64> {
    match out {
        PrfOut::Scalar(v) => vec![v],
        PrfOut::PerClass(v) => v,
    }
}

/// Integral labels (cluster ids may exceed `i32`) from an f64 capsule.
fn labels_i64(array: &arrow::array::ArrayRef) -> PyResult<Vec<i64>> {
    Ok(f64_slice(array)?.iter().map(|&v| v.round() as i64).collect())
}

// ==================== set-wise classification ====================

/// `precision_recall_fscore_support` over a 1-D (`n_labels = 0`) or
/// multilabel-indicator target. Returns `(precision, recall, fscore, support,
/// (p_zd, r_zd, f_zd), classes)`; each ratio is a 1-element vector for an
/// averaged call and the per-label vector for `average='none'`.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_labels=0, beta=1.0, class_labels=None, label_cols=None, pos_label=1, average="binary", sample_weight=None, zero_division=0.0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prfs(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    n_labels: usize,
    beta: f64,
    class_labels: Option<Vec<i32>>,
    label_cols: Option<Vec<usize>>,
    pos_label: i32,
    average: &str,
    sample_weight: Option<&Bound<'_, PyAny>>,
    zero_division: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Option<Vec<f64>>, (bool, bool, bool), Vec<i32>)> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let (yt, yp) = (labels(&yt_a)?, labels(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let avg = set_average_from_str(average)?;
    let zd = zero_division_from_f64(zero_division);
    let r = if n_labels == 0 {
        prf_core::prfs_multiclass(&yt, &yp, beta, class_labels.as_deref(), pos_label, avg, sw, zd)
    } else {
        prf_core::prfs_indicator(&yt, &yp, n_labels, beta, label_cols.as_deref(), avg, sw, zd)
    }
    .map_err(metric_err_to_py)?;
    Ok((
        prf_vec(r.precision),
        prf_vec(r.recall),
        prf_vec(r.fscore),
        r.support,
        (r.precision_zd, r.recall_zd, r.fscore_zd),
        r.classes,
    ))
}

/// `jaccard_score`: `(values, zero_division_hit, classes)`.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_labels=0, class_labels=None, label_cols=None, pos_label=1, average="binary", sample_weight=None, zero_division=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn jaccard_score(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    n_labels: usize,
    class_labels: Option<Vec<i32>>,
    label_cols: Option<Vec<usize>>,
    pos_label: i32,
    average: &str,
    sample_weight: Option<&Bound<'_, PyAny>>,
    zero_division: f64,
) -> PyResult<(Vec<f64>, bool, Vec<i32>)> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let (yt, yp) = (labels(&yt_a)?, labels(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let avg = set_average_from_str(average)?;
    let zd = zero_division_from_f64(zero_division);
    if n_labels == 0 {
        let (v, hit, classes) =
            prf_core::jaccard_multiclass(&yt, &yp, class_labels.as_deref(), pos_label, avg, sw, zd)
                .map_err(metric_err_to_py)?;
        Ok((prf_vec(v), hit, classes))
    } else {
        let (v, hit) = prf_core::jaccard_indicator(&yt, &yp, n_labels, label_cols.as_deref(), avg, sw, zd)
            .map_err(metric_err_to_py)?;
        Ok((prf_vec(v), hit, Vec::new()))
    }
}

/// `multilabel_confusion_matrix`, flat `[tn, fp, fn, tp]` per label/sample.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_labels=0, sample_weight=None, class_labels=None, label_cols=None, samplewise=false))]
#[allow(clippy::too_many_arguments)]
pub fn multilabel_confusion_matrix<'py>(
    py: Python<'py>,
    y_true: &Bound<'py, PyAny>,
    y_pred: &Bound<'py, PyAny>,
    n_labels: usize,
    sample_weight: Option<&Bound<'py, PyAny>>,
    class_labels: Option<Vec<i32>>,
    label_cols: Option<Vec<usize>>,
    samplewise: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let (yt, yp) = (labels(&yt_a)?, labels(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let flat = if n_labels == 0 {
        prf_core::multilabel_confusion_matrix(&yt, &yp, sw, class_labels.as_deref())
    } else {
        prf_core::multilabel_confusion_matrix_indicator(&yt, &yp, n_labels, sw, label_cols.as_deref(), samplewise)
    }
    .map_err(metric_err_to_py)?;
    f64_vec_to_pyarrow(py, flat)
}

// ==================== label-based classification ====================

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None, adjusted=false))]
pub fn balanced_accuracy_score(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    sample_weight: Option<&Bound<'_, PyAny>>,
    adjusted: bool,
) -> PyResult<(f64, bool)> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    cls::balanced_accuracy_score(&labels(&yt_a)?, &labels(&yp_a)?, sw, adjusted).map_err(metric_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, sample_weight=None))]
pub fn matthews_corrcoef(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    cls::matthews_corrcoef(&labels(&yt_a)?, &labels(&yp_a)?, sw).map_err(metric_err_to_py)
}

/// `cohen_kappa_score` → `(value, status)`: status 0 = defined, 1 = no shared
/// labels, 2 = one shared label (the shim substitutes `replace_undefined_by`).
#[pyfunction]
#[pyo3(signature = (y1, y2, class_labels=None, weights=None, sample_weight=None))]
pub fn cohen_kappa_score(
    y1: &Bound<'_, PyAny>,
    y2: &Bound<'_, PyAny>,
    class_labels: Option<Vec<i32>>,
    weights: Option<&str>,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<(f64, u8)> {
    let (a, b, sw_a) = (capsule(y1)?, capsule(y2)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let w = match weights {
        None => cls::KappaWeights::None_,
        Some("linear") => cls::KappaWeights::Linear,
        Some("quadratic") => cls::KappaWeights::Quadratic,
        Some(o) => return Err(PyValueError::new_err(format!("unknown weights '{o}'"))),
    };
    match cls::cohen_kappa_score(&labels(&a)?, &labels(&b)?, class_labels.as_deref(), w, sw)
        .map_err(metric_err_to_py)?
    {
        cls::KappaOut::Value(v) => Ok((v, 0)),
        cls::KappaOut::NoSharedLabels => Ok((f64::NAN, 1)),
        cls::KappaOut::OneSharedLabel => Ok((f64::NAN, 2)),
    }
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_labels=0, sample_weight=None))]
pub fn hamming_loss(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    n_labels: usize,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let (yt, yp) = (labels(&yt_a)?, labels(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    if n_labels == 0 {
        cls::hamming_loss(&yt, &yp, sw)
    } else {
        cls::hamming_loss_indicator(&yt, &yp, n_labels, sw)
    }
    .map_err(metric_err_to_py)
}

/// Subset accuracy (`accuracy_score` on a multilabel indicator target).
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_labels, sample_weight=None, normalize=true))]
pub fn accuracy_score_indicator(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    n_labels: usize,
    sample_weight: Option<&Bound<'_, PyAny>>,
    normalize: bool,
) -> PyResult<f64> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    cls::accuracy_score_indicator(&labels(&yt_a)?, &labels(&yp_a)?, n_labels, sw, normalize)
        .map_err(metric_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_labels=0, sample_weight=None, normalize=true))]
pub fn zero_one_loss(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    n_labels: usize,
    sample_weight: Option<&Bound<'_, PyAny>>,
    normalize: bool,
) -> PyResult<f64> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let (yt, yp) = (labels(&yt_a)?, labels(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    if n_labels == 0 {
        cls::zero_one_loss(&yt, &yp, sw, normalize)
    } else {
        cls::zero_one_loss_indicator(&yt, &yp, n_labels, sw, normalize)
    }
    .map_err(metric_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y_signed, pred_decision, sample_weight=None))]
pub fn hinge_loss_binary(
    y_signed: &Bound<'_, PyAny>,
    pred_decision: &Bound<'_, PyAny>,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (y_a, d_a, sw_a) = (capsule(y_signed)?, capsule(pred_decision)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    cls::hinge_loss_binary(f64_slice(&y_a)?, f64_slice(&d_a)?, sw).map_err(metric_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y_true, pred_decision, classes, sample_weight=None))]
pub fn hinge_loss_multiclass(
    y_true: &Bound<'_, PyAny>,
    pred_decision: &Bound<'_, PyAny>,
    classes: Vec<i32>,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (y_a, d_a, sw_a) = (capsule(y_true)?, capsule(pred_decision)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    cls::hinge_loss_multiclass(&labels(&y_a)?, f64_slice(&d_a)?, &classes, sw).map_err(metric_err_to_py)
}

/// `mode`: 0 = brier, 1 = d2_brier, 2 = d2_log_loss.
#[pyfunction]
#[pyo3(signature = (mode, y_true, y_prob, classes, sample_weight=None, scale_by_half=false, eps=f64::EPSILON))]
#[allow(clippy::too_many_arguments)]
pub fn probabilistic_score(
    mode: u8,
    y_true: &Bound<'_, PyAny>,
    y_prob: &Bound<'_, PyAny>,
    classes: Vec<i32>,
    sample_weight: Option<&Bound<'_, PyAny>>,
    scale_by_half: bool,
    eps: f64,
) -> PyResult<f64> {
    let (y_a, p_a, sw_a) = (capsule(y_true)?, capsule(y_prob)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let (yt, p) = (labels(&y_a)?, f64_slice(&p_a)?);
    match mode {
        0 => cls::brier_score_loss(&yt, p, &classes, sw, scale_by_half),
        1 => cls::d2_brier_score(&yt, p, &classes, sw),
        2 => cls::d2_log_loss_score(&yt, p, &classes, sw, eps),
        _ => return Err(PyValueError::new_err("unknown probabilistic score mode")),
    }
    .map_err(metric_err_to_py)
}

/// `class_likelihood_ratios` → `(lr_pos, lr_neg, no_positive, pos_undefined,
/// no_predicted_positive, neg_undefined)`.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, class_labels=None, sample_weight=None))]
pub fn class_likelihood_ratios(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    class_labels: Option<Vec<i32>>,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<(f64, f64, bool, bool, bool, bool)> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let r = cls::class_likelihood_ratios(&labels(&yt_a)?, &labels(&yp_a)?, class_labels.as_deref(), sw)
        .map_err(metric_err_to_py)?;
    Ok((
        r.lr_pos,
        r.lr_neg,
        r.no_positive,
        r.pos_undefined,
        r.no_predicted_positive,
        r.neg_undefined,
    ))
}

/// `top_k_accuracy_score`; `classes = None` selects the binary form (`y_true`
/// is then the 0/1 encoding and `y_score` the positive-class score).
#[pyfunction]
#[pyo3(signature = (y_true, y_score, k, classes=None, normalize=true, sample_weight=None))]
pub fn top_k_accuracy_score(
    y_true: &Bound<'_, PyAny>,
    y_score: &Bound<'_, PyAny>,
    k: usize,
    classes: Option<Vec<i32>>,
    normalize: bool,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (y_a, s_a, sw_a) = (capsule(y_true)?, capsule(y_score)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let (yt, s) = (labels(&y_a)?, f64_slice(&s_a)?);
    match classes {
        Some(c) => cls::top_k_accuracy_score(&yt, s, &c, k, normalize, sw),
        None => cls::top_k_accuracy_score_binary(&yt, s, k, normalize, sw),
    }
    .map_err(metric_err_to_py)
}

// ==================== regression ====================

/// The regression metrics that share `(y_true, y_pred, n_outputs,
/// sample_weight, multioutput, multioutput_weights)` and return a flat
/// `Vec<f64>` (one element when reduced). `name` selects the metric; `alpha`
/// is read by the pinball metrics and `force_finite` by explained variance.
#[pyfunction]
#[pyo3(signature = (name, y_true, y_pred, n_outputs=1, sample_weight=None, multioutput="uniform_average", multioutput_weights=None, alpha=0.5, force_finite=true))]
#[allow(clippy::too_many_arguments)]
pub fn regression_metric(
    name: &str,
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    n_outputs: usize,
    sample_weight: Option<&Bound<'_, PyAny>>,
    multioutput: &str,
    multioutput_weights: Option<Vec<f64>>,
    alpha: f64,
    force_finite: bool,
) -> PyResult<Vec<f64>> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let (yt, yp) = (f64_slice(&yt_a)?, f64_slice(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let mo = multioutput_from(multioutput, multioutput_weights.as_deref())?;
    let out: MetricOut = match name {
        "explained_variance" => reg::explained_variance_score::<f64>(yt, yp, n_outputs, sw, mo, force_finite),
        "mape" => reg::mean_absolute_percentage_error::<f64>(yt, yp, n_outputs, sw, mo),
        "msle" => reg::mean_squared_log_error::<f64>(yt, yp, n_outputs, sw, mo),
        "rmse" => reg::root_mean_squared_error::<f64>(yt, yp, n_outputs, sw, mo),
        "rmsle" => reg::root_mean_squared_log_error::<f64>(yt, yp, n_outputs, sw, mo),
        "median_ae" => reg::median_absolute_error::<f64>(yt, yp, n_outputs, sw, mo),
        "pinball" => reg::mean_pinball_loss::<f64>(yt, yp, n_outputs, sw, alpha, mo),
        "d2_pinball" => reg::d2_pinball_score::<f64>(yt, yp, n_outputs, sw, alpha, mo),
        other => return Err(PyValueError::new_err(format!("unknown regression metric '{other}'"))),
    }
    .map_err(metric_err_to_py)?;
    Ok(metric_out_to_vec(out))
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_outputs=1))]
pub fn max_error(y_true: &Bound<'_, PyAny>, y_pred: &Bound<'_, PyAny>, n_outputs: usize) -> PyResult<f64> {
    let (yt_a, yp_a) = (capsule(y_true)?, capsule(y_pred)?);
    reg::max_error::<f64>(f64_slice(&yt_a)?, f64_slice(&yp_a)?, n_outputs).map_err(metric_err_to_py)
}

/// `mean_tweedie_deviance` (`d2 = false`) or `d2_tweedie_score` (`d2 = true`).
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, n_outputs=1, sample_weight=None, power=0.0, d2=false))]
pub fn tweedie(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
    n_outputs: usize,
    sample_weight: Option<&Bound<'_, PyAny>>,
    power: f64,
    d2: bool,
) -> PyResult<f64> {
    let (yt_a, yp_a, sw_a) = (capsule(y_true)?, capsule(y_pred)?, capsule_opt(sample_weight)?);
    let (yt, yp) = (f64_slice(&yt_a)?, f64_slice(&yp_a)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    if d2 {
        reg::d2_tweedie_score::<f64>(yt, yp, n_outputs, sw, power)
    } else {
        reg::mean_tweedie_deviance::<f64>(yt, yp, n_outputs, sw, power)
    }
    .map_err(metric_err_to_py)
}

// ==================== ranking / thresholds ====================

#[pyfunction]
#[pyo3(signature = (y_true, y_score, pos_label=1, sample_weight=None))]
pub fn confusion_matrix_at_thresholds<'py>(
    py: Python<'py>,
    y_true: &Bound<'py, PyAny>,
    y_score: &Bound<'py, PyAny>,
    pos_label: i32,
    sample_weight: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (y_a, s_a, sw_a) = (capsule(y_true)?, capsule(y_score)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let (tns, fps, fns, tps, thr) =
        rk::confusion_matrix_at_thresholds(&labels(&y_a)?, f64_slice(&s_a)?, pos_label, sw).map_err(metric_err_to_py)?;
    Ok((
        f64_vec_to_pyarrow(py, tns)?,
        f64_vec_to_pyarrow(py, fps)?,
        f64_vec_to_pyarrow(py, fns)?,
        f64_vec_to_pyarrow(py, tps)?,
        f64_vec_to_pyarrow(py, thr)?,
    ))
}

/// `roc_curve` → `((fpr, tpr, thresholds), no_negative, no_positive)`.
#[pyfunction]
#[pyo3(signature = (y_true, y_score, pos_label=1, sample_weight=None, drop_intermediate=true))]
pub fn roc_curve<'py>(
    py: Python<'py>,
    y_true: &Bound<'py, PyAny>,
    y_score: &Bound<'py, PyAny>,
    pos_label: i32,
    sample_weight: Option<&Bound<'py, PyAny>>,
    drop_intermediate: bool,
) -> PyResult<(Py3<'py>, bool, bool)> {
    let (y_a, s_a, sw_a) = (capsule(y_true)?, capsule(y_score)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let r = rk::roc_curve(&labels(&y_a)?, f64_slice(&s_a)?, pos_label, sw, drop_intermediate)
        .map_err(metric_err_to_py)?;
    Ok((
        (
            f64_vec_to_pyarrow(py, r.fpr)?,
            f64_vec_to_pyarrow(py, r.tpr)?,
            f64_vec_to_pyarrow(py, r.thresholds)?,
        ),
        r.no_negative,
        r.no_positive,
    ))
}

#[pyfunction]
#[pyo3(signature = (y_true, y_score, pos_label=1, sample_weight=None, drop_intermediate=false))]
pub fn det_curve<'py>(
    py: Python<'py>,
    y_true: &Bound<'py, PyAny>,
    y_score: &Bound<'py, PyAny>,
    pos_label: i32,
    sample_weight: Option<&Bound<'py, PyAny>>,
    drop_intermediate: bool,
) -> PyResult<Py3<'py>> {
    let (y_a, s_a, sw_a) = (capsule(y_true)?, capsule(y_score)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let (fpr, fnr, thr) = rk::det_curve(&labels(&y_a)?, f64_slice(&s_a)?, pos_label, sw, drop_intermediate)
        .map_err(metric_err_to_py)?;
    Ok((
        f64_vec_to_pyarrow(py, fpr)?,
        f64_vec_to_pyarrow(py, fnr)?,
        f64_vec_to_pyarrow(py, thr)?,
    ))
}

#[pyfunction]
pub fn auc(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<f64> {
    let (x_a, y_a) = (capsule(x)?, capsule(y)?);
    rk::auc(f64_slice(&x_a)?, f64_slice(&y_a)?).map_err(metric_err_to_py)
}

/// `average_precision_score`: binary (`n_labels = 0`, scalar in a 1-element
/// vector) or multilabel/binarized-multiclass (`n_labels > 0`).
#[pyfunction]
#[pyo3(signature = (y_true, y_score, n_labels=0, pos_label=1, average="macro", sample_weight=None))]
pub fn average_precision_score(
    y_true: &Bound<'_, PyAny>,
    y_score: &Bound<'_, PyAny>,
    n_labels: usize,
    pos_label: i32,
    average: &str,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<f64>> {
    let (y_a, s_a, sw_a) = (capsule(y_true)?, capsule(y_score)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let (yt, s) = (labels(&y_a)?, f64_slice(&s_a)?);
    if n_labels == 0 {
        return rk::average_precision_binary(&yt, s, pos_label, sw)
            .map(|v| vec![v])
            .map_err(metric_err_to_py);
    }
    rk::average_precision_indicator(&yt, s, n_labels, score_average_from_str(average)?, sw)
        .map(prf_vec)
        .map_err(metric_err_to_py)
}

/// The multilabel ranking metrics over an `n × n_labels` indicator: `name ∈
/// {coverage_error, lrap, label_ranking_loss}`.
#[pyfunction]
#[pyo3(signature = (name, y_true, y_score, n_labels, sample_weight=None))]
pub fn label_ranking_metric(
    name: &str,
    y_true: &Bound<'_, PyAny>,
    y_score: &Bound<'_, PyAny>,
    n_labels: usize,
    sample_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (y_a, s_a, sw_a) = (capsule(y_true)?, capsule(y_score)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let (yt, s) = (labels(&y_a)?, f64_slice(&s_a)?);
    match name {
        "coverage_error" => rk::coverage_error(&yt, s, n_labels, sw),
        "lrap" => rk::label_ranking_average_precision_score(&yt, s, n_labels, sw),
        "label_ranking_loss" => rk::label_ranking_loss(&yt, s, n_labels, sw),
        other => return Err(PyValueError::new_err(format!("unknown ranking metric '{other}'"))),
    }
    .map_err(metric_err_to_py)
}

/// `dcg_score` (`normalized = false`) / `ndcg_score` (`true`, `log_base`
/// ignored — sklearn fixes it at 2).
#[pyfunction]
#[pyo3(signature = (y_true, y_score, n_labels, k=None, log_base=2.0, sample_weight=None, ignore_ties=false, normalized=false))]
#[allow(clippy::too_many_arguments)]
pub fn dcg(
    y_true: &Bound<'_, PyAny>,
    y_score: &Bound<'_, PyAny>,
    n_labels: usize,
    k: Option<usize>,
    log_base: f64,
    sample_weight: Option<&Bound<'_, PyAny>>,
    ignore_ties: bool,
    normalized: bool,
) -> PyResult<f64> {
    let (y_a, s_a, sw_a) = (capsule(y_true)?, capsule(y_score)?, capsule_opt(sample_weight)?);
    let sw = sw_a.as_ref().map(f64_slice).transpose()?;
    let (yt, s) = (f64_slice(&y_a)?, f64_slice(&s_a)?);
    if normalized {
        rk::ndcg_score(yt, s, n_labels, k, sw, ignore_ties)
    } else {
        rk::dcg_score(yt, s, n_labels, k, log_base, sw, ignore_ties)
    }
    .map_err(metric_err_to_py)
}

// ==================== clustering ====================

#[pyfunction]
pub fn contingency_matrix<'py>(
    py: Python<'py>,
    labels_true: &Bound<'py, PyAny>,
    labels_pred: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, usize, usize)> {
    let (a, b) = (capsule(labels_true)?, capsule(labels_pred)?);
    let (flat, r, c) = clu::contingency_matrix(&labels_i64(&a)?, &labels_i64(&b)?).map_err(metric_err_to_py)?;
    Ok((i64_vec_to_pyarrow(py, flat.into_iter().map(|v| v as i64).collect())?, r, c))
}

/// The two-labelling scores that return one float: `name ∈ {mutual_info,
/// adjusted_mutual_info, normalized_mutual_info, fowlkes_mallows, rand,
/// adjusted_rand}`; `average_method` is read by the normalized/adjusted MI.
#[pyfunction]
#[pyo3(signature = (name, labels_true, labels_pred, average_method="arithmetic"))]
pub fn cluster_score(
    name: &str,
    labels_true: &Bound<'_, PyAny>,
    labels_pred: &Bound<'_, PyAny>,
    average_method: &str,
) -> PyResult<f64> {
    let (a, b) = (capsule(labels_true)?, capsule(labels_pred)?);
    let (lt, lp) = (labels_i64(&a)?, labels_i64(&b)?);
    let method = match average_method {
        "min" => clu::AverageMethod::Min,
        "geometric" => clu::AverageMethod::Geometric,
        "arithmetic" => clu::AverageMethod::Arithmetic,
        "max" => clu::AverageMethod::Max,
        o => return Err(PyValueError::new_err(format!("unknown average_method '{o}'"))),
    };
    match name {
        "mutual_info" => clu::mutual_info_score(&lt, &lp),
        "adjusted_mutual_info" => clu::adjusted_mutual_info_score(&lt, &lp, method),
        "normalized_mutual_info" => clu::normalized_mutual_info_score(&lt, &lp, method),
        "fowlkes_mallows" => clu::fowlkes_mallows_score(&lt, &lp),
        "rand" => clu::rand_score(&lt, &lp),
        "adjusted_rand" => clu::adjusted_rand_score(&lt, &lp),
        o => return Err(PyValueError::new_err(format!("unknown cluster score '{o}'"))),
    }
    .map_err(metric_err_to_py)
}

#[pyfunction]
pub fn mutual_info_score_contingency(counts: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<f64> {
    let a = capsule(counts)?;
    clu::mutual_info_score_contingency(f64_slice(&a)?, rows, cols).map_err(metric_err_to_py)
}

#[pyfunction]
pub fn expected_mutual_information(counts: &Bound<'_, PyAny>, rows: usize, cols: usize, n_samples: u64) -> PyResult<f64> {
    let a = capsule(counts)?;
    clu::expected_mutual_information_dense(f64_slice(&a)?, rows, cols, n_samples).map_err(metric_err_to_py)
}

#[pyfunction]
pub fn cluster_entropy(labels: &Bound<'_, PyAny>) -> PyResult<f64> {
    let a = capsule(labels)?;
    Ok(clu::entropy(&labels_i64(&a)?))
}

#[pyfunction]
#[pyo3(signature = (labels_true, labels_pred, beta=1.0))]
pub fn homogeneity_completeness_v_measure(
    labels_true: &Bound<'_, PyAny>,
    labels_pred: &Bound<'_, PyAny>,
    beta: f64,
) -> PyResult<(f64, f64, f64)> {
    let (a, b) = (capsule(labels_true)?, capsule(labels_pred)?);
    clu::homogeneity_completeness_v_measure(&labels_i64(&a)?, &labels_i64(&b)?, beta).map_err(metric_err_to_py)
}

#[pyfunction]
pub fn pair_confusion_matrix(labels_true: &Bound<'_, PyAny>, labels_pred: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let (a, b) = (capsule(labels_true)?, capsule(labels_pred)?);
    let m = clu::pair_confusion_matrix(&labels_i64(&a)?, &labels_i64(&b)?).map_err(metric_err_to_py)?;
    m.iter()
        .map(|&v| i64::try_from(v).map_err(|_| PyValueError::new_err("pair count overflows int64")))
        .collect()
}

/// `silhouette_samples`. `precomputed = true` reads `x` as the `n × n`
/// distance matrix; otherwise the distances use the parsed metric.
#[pyfunction]
#[pyo3(signature = (x, d, labels, metric="euclidean", precomputed=false, p=None, w=None, v=None, vi=None))]
#[allow(clippy::too_many_arguments)]
pub fn silhouette_samples<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    d: usize,
    labels: &Bound<'py, PyAny>,
    metric: &str,
    precomputed: bool,
    p: Option<f64>,
    w: Option<Vec<f64>>,
    v: Option<Vec<f64>>,
    vi: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let (x_a, l_a) = (capsule(x)?, capsule(labels)?);
    let (xs, ls) = (f64_slice(&x_a)?, labels_i64(&l_a)?);
    let out = if precomputed {
        py.detach(|| clu::silhouette_samples_from_distances(xs, &ls))
    } else {
        let m = distance_from(metric, p, w, v, vi, f64::NAN)?;
        py.detach(|| clu::silhouette_samples(xs, d, &ls, &m))
    }
    .map_err(metric_err_to_py)?;
    f64_vec_to_pyarrow(py, out)
}

/// `calinski_harabasz_score` (`name = "calinski_harabasz"`) or
/// `davies_bouldin_score` (`"davies_bouldin"`).
#[pyfunction]
pub fn cluster_dispersion(name: &str, x: &Bound<'_, PyAny>, labels: &Bound<'_, PyAny>) -> PyResult<f64> {
    let (x_a, l_a) = (capsule(x)?, capsule(labels)?);
    let (xs, ls) = (f64_slice(&x_a)?, labels_i64(&l_a)?);
    match name {
        "calinski_harabasz" => clu::calinski_harabasz_score(xs, &ls),
        "davies_bouldin" => clu::davies_bouldin_score(xs, &ls),
        o => return Err(PyValueError::new_err(format!("unknown dispersion score '{o}'"))),
    }
    .map_err(metric_err_to_py)
}

/// `consensus_score` from a precomputed `n_a × n_b` similarity matrix (the
/// shim computes it for a callable `similarity`).
#[pyfunction]
pub fn consensus_from_similarity(sim: Vec<f64>, n_a: usize, n_b: usize) -> PyResult<f64> {
    clu::consensus_from_similarity(&sim, n_a, n_b).map_err(metric_err_to_py)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn consensus_score_jaccard(
    a_rows: &Bound<'_, PyAny>,
    a_cols: &Bound<'_, PyAny>,
    b_rows: &Bound<'_, PyAny>,
    b_cols: &Bound<'_, PyAny>,
    n_a: usize,
    n_b: usize,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<f64> {
    let (ar, ac, br, bc) = (capsule(a_rows)?, capsule(a_cols)?, capsule(b_rows)?, capsule(b_cols)?);
    clu::consensus_score_jaccard(
        f64_slice(&ar)?,
        f64_slice(&ac)?,
        f64_slice(&br)?,
        f64_slice(&bc)?,
        n_a,
        n_b,
        n_rows,
        n_cols,
    )
    .map_err(metric_err_to_py)
}

// ==================== pairwise ====================

/// Build a [`Distance`] from its name and the optional scipy parameters.
fn distance_from(
    metric: &str,
    p: Option<f64>,
    w: Option<Vec<f64>>,
    v: Option<Vec<f64>>,
    vi: Option<Vec<f64>>,
    missing: f64,
) -> PyResult<Distance> {
    Ok(match metric {
        "minkowski" => Distance::Minkowski { p: p.unwrap_or(2.0), w },
        "seuclidean" => Distance::SEuclidean {
            v: v.ok_or_else(|| PyValueError::new_err("seuclidean requires V"))?,
        },
        "mahalanobis" => Distance::Mahalanobis {
            vi: vi.ok_or_else(|| PyValueError::new_err("mahalanobis requires VI"))?,
        },
        "nan_euclidean" => Distance::NanEuclidean { missing },
        other => Distance::from_name(other).map_err(metric_err_to_py)?,
    })
}

/// `pairwise_distances`, flat row-major. `y = None` means `Y = X`.
#[pyfunction]
#[pyo3(signature = (x, y, d, metric="euclidean", p=None, w=None, v=None, vi=None, missing=f64::NAN))]
#[allow(clippy::too_many_arguments)]
pub fn pairwise_distances<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: Option<&Bound<'py, PyAny>>,
    d: usize,
    metric: &str,
    p: Option<f64>,
    w: Option<Vec<f64>>,
    v: Option<Vec<f64>>,
    vi: Option<Vec<f64>>,
    missing: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let x_a = capsule(x)?;
    let y_a = capsule_opt(y)?;
    let xs = f64_slice(&x_a)?;
    let ys = y_a.as_ref().map(f64_slice).transpose()?;
    let m = distance_from(metric, p, w, v, vi, missing)?;
    let out = py.detach(|| pw::pairwise_distances(xs, ys, d, &m)).map_err(metric_err_to_py)?;
    f64_vec_to_pyarrow(py, out)
}

/// `pairwise_kernels`, flat row-major. `gamma`/`degree`/`coef0` are already
/// resolved by the shim (sklearn's `gamma=None -> 1/n_features`).
#[pyfunction]
#[pyo3(signature = (x, y, d, kernel="linear", gamma=1.0, degree=3.0, coef0=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn pairwise_kernels<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: Option<&Bound<'py, PyAny>>,
    d: usize,
    kernel: &str,
    gamma: f64,
    degree: f64,
    coef0: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let x_a = capsule(x)?;
    let y_a = capsule_opt(y)?;
    let xs = f64_slice(&x_a)?;
    let ys = y_a.as_ref().map(f64_slice).transpose()?;
    let k = match kernel {
        "linear" => Kernel::Linear,
        "poly" | "polynomial" => Kernel::Poly { gamma, degree, coef0 },
        "rbf" => Kernel::Rbf { gamma },
        "sigmoid" => Kernel::Sigmoid { gamma, coef0 },
        "laplacian" => Kernel::Laplacian { gamma },
        "cosine" => Kernel::Cosine,
        "chi2" => Kernel::Chi2 { gamma },
        "additive_chi2" => Kernel::AdditiveChi2,
        o => return Err(PyValueError::new_err(format!("unknown kernel '{o}'"))),
    };
    let out = py.detach(|| pw::pairwise_kernels(xs, ys, d, k)).map_err(metric_err_to_py)?;
    f64_vec_to_pyarrow(py, out)
}

#[pyfunction]
#[pyo3(signature = (x, y, d, metric="euclidean", p=None))]
pub fn paired_distances<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    d: usize,
    metric: &str,
    p: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (x_a, y_a) = (capsule(x)?, capsule(y)?);
    let m = distance_from(metric, p, None, None, None, f64::NAN)?;
    let out = pw::paired_distances(f64_slice(&x_a)?, f64_slice(&y_a)?, d, &m).map_err(metric_err_to_py)?;
    f64_vec_to_pyarrow(py, out)
}

/// `pairwise_distances_argmin_min(X, Y, axis=1)` → `(indices, distances)`.
#[pyfunction]
#[pyo3(signature = (x, y, d, metric="euclidean", p=None, w=None, v=None, vi=None))]
#[allow(clippy::too_many_arguments)]
pub fn argmin_min<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    d: usize,
    metric: &str,
    p: Option<f64>,
    w: Option<Vec<f64>>,
    v: Option<Vec<f64>>,
    vi: Option<Vec<f64>>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (x_a, y_a) = (capsule(x)?, capsule(y)?);
    let (xs, ys) = (f64_slice(&x_a)?, f64_slice(&y_a)?);
    let m = distance_from(metric, p, w, v, vi, f64::NAN)?;
    let (idx, val) = py.detach(|| pw::argmin_min(xs, ys, d, &m)).map_err(metric_err_to_py)?;
    Ok((
        i64_vec_to_pyarrow(py, idx.into_iter().map(|i| i as i64).collect())?,
        f64_vec_to_pyarrow(py, val)?,
    ))
}

/// Column variances (ddof=1) and the INVERSE covariance of `x` — the
/// `seuclidean` `V` / `mahalanobis` `VI` defaults sklearn derives when `Y` is
/// `None`.
#[pyfunction]
pub fn distance_defaults(x: &Bound<'_, PyAny>, d: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let a = capsule(x)?;
    let (var, cov) = pw::column_var_cov(f64_slice(&a)?, d).map_err(metric_err_to_py)?;
    let vi = pw::invert(&cov, d).map_err(metric_err_to_py)?;
    Ok((var, vi))
}

/// Register every binding in this file on the `_mlrs` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prfs, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard_score, m)?)?;
    m.add_function(wrap_pyfunction!(multilabel_confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(balanced_accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(matthews_corrcoef, m)?)?;
    m.add_function(wrap_pyfunction!(cohen_kappa_score, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_loss, m)?)?;
    m.add_function(wrap_pyfunction!(accuracy_score_indicator, m)?)?;
    m.add_function(wrap_pyfunction!(zero_one_loss, m)?)?;
    m.add_function(wrap_pyfunction!(hinge_loss_binary, m)?)?;
    m.add_function(wrap_pyfunction!(hinge_loss_multiclass, m)?)?;
    m.add_function(wrap_pyfunction!(probabilistic_score, m)?)?;
    m.add_function(wrap_pyfunction!(class_likelihood_ratios, m)?)?;
    m.add_function(wrap_pyfunction!(top_k_accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(regression_metric, m)?)?;
    m.add_function(wrap_pyfunction!(max_error, m)?)?;
    m.add_function(wrap_pyfunction!(tweedie, m)?)?;
    m.add_function(wrap_pyfunction!(confusion_matrix_at_thresholds, m)?)?;
    m.add_function(wrap_pyfunction!(roc_curve, m)?)?;
    m.add_function(wrap_pyfunction!(det_curve, m)?)?;
    m.add_function(wrap_pyfunction!(auc, m)?)?;
    m.add_function(wrap_pyfunction!(average_precision_score, m)?)?;
    m.add_function(wrap_pyfunction!(label_ranking_metric, m)?)?;
    m.add_function(wrap_pyfunction!(dcg, m)?)?;
    m.add_function(wrap_pyfunction!(contingency_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_score, m)?)?;
    m.add_function(wrap_pyfunction!(mutual_info_score_contingency, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(expected_mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(homogeneity_completeness_v_measure, m)?)?;
    m.add_function(wrap_pyfunction!(pair_confusion_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(silhouette_samples, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(consensus_from_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(consensus_score_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_distances, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_kernels, m)?)?;
    m.add_function(wrap_pyfunction!(paired_distances, m)?)?;
    m.add_function(wrap_pyfunction!(argmin_min, m)?)?;
    m.add_function(wrap_pyfunction!(distance_defaults, m)?)?;
    Ok(())
}
