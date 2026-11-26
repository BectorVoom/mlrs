//! The shared precision / recall / F-beta / Jaccard core, built on the
//! per-label confusion counts sklearn calls the *multilabel confusion matrix*
//! (MCM).
//!
//! Every set-wise classification metric in sklearn 1.9 —
//! `precision_recall_fscore_support`, `precision_score`, `recall_score`,
//! `f1_score`, `fbeta_score`, `jaccard_score`, `multilabel_confusion_matrix`
//! and `classification_report` — is the same two steps:
//!
//! 1. **Count.** Per label (or, for `average='samples'`, per sample) the
//!    weighted `tp`, `pred_sum = tp + fp` and `true_sum = tp + fn`.
//!    [`mcm_multiclass`] does this for a 1-D target in ONE pass over the
//!    samples (via [`super::class_bookkeeping`]'s O(1) class lookup);
//!    [`mcm_indicator`] does it for a 2-D 0/1 indicator target, column-wise or
//!    sample-wise.
//! 2. **Divide and average.** [`prfs_from_counts`] applies sklearn's
//!    `_prf_divide` (a zero denominator yields the `zero_division` value and
//!    raises a flag the shim turns into `UndefinedMetricWarning`) and then
//!    `_nanaverage` — which DROPS NaN entries before averaging and falls back to
//!    an unweighted mean when the weights sum to zero.
//!
//! That second rule is why this module exists separately from the original
//! `average_ratio`: with `zero_division=np.nan`, sklearn's macro average skips
//! the undefined classes (`[1.0, 1.0, NaN] -> 1.0`), while a plain mean returns
//! NaN. `precision_score`/`recall_score`/`f1_score` now all route through here,
//! so the rule is stated once.
//!
//! Tests live in `crates/mlrs-algos/tests/metrics_prfs_test.rs` (AGENTS.md §2).

use super::{class_bookkeeping, validate_weight, MetricError, PrfOut, ZeroDivision};

/// Per-label (or per-sample) confusion counts. `tn` is only materialized by the
/// callers that need it ([`multilabel_confusion_matrix`]); the ratio metrics
/// read the other three.
#[derive(Debug, Clone, PartialEq)]
pub struct Mcm {
    pub tp: Vec<f64>,
    pub fp: Vec<f64>,
    pub fnn: Vec<f64>,
    pub tn: Vec<f64>,
}

/// The `average` modes of the set-wise metrics. `Binary` is resolved by the
/// caller to a one-element count vector (the `pos_label` column), after which
/// it averages exactly like `Macro` — which is what sklearn does too
/// (`labels=[pos_label]`, then `_nanaverage` over one value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetAverage {
    None_,
    Binary,
    Micro,
    Macro,
    Weighted,
    Samples,
}

/// `precision_recall_fscore_support`'s result. Each of the three ratios is a
/// scalar for an averaged call and the per-label vector for `average=None`;
/// `support` is the per-label `true_sum` for `average=None` and `None`
/// otherwise (sklearn returns `None` there too). The three `*_zd` flags say
/// whether that ratio consulted `zero_division` — what decides sklearn's
/// `UndefinedMetricWarning` under the default `zero_division="warn"`.
#[derive(Debug, Clone, PartialEq)]
pub struct PrfsResult {
    pub precision: PrfOut,
    pub recall: PrfOut,
    pub fscore: PrfOut,
    pub support: Option<Vec<f64>>,
    pub precision_zd: bool,
    pub recall_zd: bool,
    pub fscore_zd: bool,
    /// The resolved class order the counts are indexed by (the caller's
    /// `labels`, or the sorted unique of `y_true ∪ y_pred`; for `Binary`, the
    /// sorted unique — the shim's two binary guards read it).
    pub classes: Vec<i32>,
}

pub(crate) fn zd_value(zero_division: ZeroDivision) -> f64 {
    match zero_division {
        ZeroDivision::Zero => 0.0,
        ZeroDivision::One => 1.0,
        ZeroDivision::Nan => f64::NAN,
    }
}

/// sklearn's `_prf_divide`: `num / den` elementwise, with `zero_division`
/// substituted (and the returned flag set) wherever `den == 0`.
pub(crate) fn prf_divide(num: &[f64], den: &[f64], zero_division: ZeroDivision) -> (Vec<f64>, bool) {
    let mut hit = false;
    let out = num
        .iter()
        .zip(den.iter())
        .map(|(&n, &d)| {
            if d == 0.0 {
                hit = true;
                zd_value(zero_division)
            } else {
                n / d
            }
        })
        .collect();
    (out, hit)
}

/// sklearn's `_average(a, weights)` (`np.average` semantics): unweighted mean
/// for `None`, weighted mean otherwise. A zero weight-total returns `None` —
/// sklearn raises `ZeroDivisionError` there, and each caller decides what that
/// means (`_nanaverage` falls back to the plain mean).
pub(crate) fn average(values: &[f64], weights: Option<&[f64]>) -> Option<f64> {
    match weights {
        None => Some(values.iter().sum::<f64>() / values.len() as f64),
        Some(w) => {
            let total: f64 = w.iter().sum();
            if total == 0.0 {
                return None;
            }
            Some(values.iter().zip(w).map(|(&v, &wi)| v * wi).sum::<f64>() / total)
        }
    }
}

/// sklearn's `_nanaverage`: an average that IGNORES NaN entries (and their
/// weights), is NaN when every entry is NaN or there are none, and falls back
/// to the unweighted mean of the kept entries when their weights sum to zero.
pub(crate) fn nanaverage(values: &[f64], weights: Option<&[f64]>) -> f64 {
    if values.is_empty() || values.iter().all(|v| v.is_nan()) {
        return f64::NAN;
    }
    let keep: Vec<usize> = (0..values.len()).filter(|&i| !values[i].is_nan()).collect();
    let kept: Vec<f64> = keep.iter().map(|&i| values[i]).collect();
    match weights {
        None => kept.iter().sum::<f64>() / kept.len() as f64,
        Some(w) => {
            let kw: Vec<f64> = keep.iter().map(|&i| w[i]).collect();
            average(&kept, Some(&kw)).unwrap_or_else(|| kept.iter().sum::<f64>() / kept.len() as f64)
        }
    }
}

/// Per-label counts for a 1-D (binary/multiclass) target, in the order of
/// `labels` (or the sorted unique of `y_true ∪ y_pred`). One pass over the
/// samples. `tn` is the remaining weight: a sample whose true AND predicted
/// labels are both outside a given label still counts as that label's
/// true-negative, exactly as sklearn's `tn = n - tp - fp - fn`.
pub fn mcm_multiclass(
    y_true: &[i32],
    y_pred: &[i32],
    sample_weight: Option<&[f64]>,
    labels: Option<&[i32]>,
) -> Result<(Vec<i32>, Mcm), MetricError> {
    let bk = class_bookkeeping(y_true, y_pred, sample_weight, labels)?;
    let total: f64 = match sample_weight {
        Some(w) => w.iter().sum(),
        None => y_true.len() as f64,
    };
    let tn = (0..bk.classes.len())
        .map(|i| total - bk.tp[i] - bk.fp[i] - bk.fnn[i])
        .collect();
    Ok((
        bk.classes,
        Mcm {
            tp: bk.tp,
            fp: bk.fp,
            fnn: bk.fnn,
            tn,
        },
    ))
}

/// Per-label (`samplewise = false`) or per-sample (`samplewise = true`) counts
/// for a row-major `n_samples × n_labels` multilabel INDICATOR target (any
/// nonzero entry is a positive). `label_cols` selects (and orders) the label
/// columns, sklearn's `labels` for multilabel input; every index must be
/// `< n_labels`.
///
/// Sample weights scale each sample's contribution; in the samplewise case a
/// sample's `tn` is `w_i * n_selected - tp - fp - fn`, sklearn's formula.
pub fn mcm_indicator(
    y_true: &[i32],
    y_pred: &[i32],
    n_labels: usize,
    sample_weight: Option<&[f64]>,
    label_cols: Option<&[usize]>,
    samplewise: bool,
) -> Result<Mcm, MetricError> {
    if y_true.len() != y_pred.len() {
        return Err(MetricError::LengthMismatch);
    }
    if n_labels == 0 || y_true.len() % n_labels != 0 {
        return Err(MetricError::BadShape);
    }
    let n = y_true.len() / n_labels;
    validate_weight(n, sample_weight)?;
    let cols: Vec<usize> = match label_cols {
        Some(c) => {
            if c.iter().any(|&j| j >= n_labels) {
                return Err(MetricError::LabelNotInLabels);
            }
            c.to_vec()
        }
        None => (0..n_labels).collect(),
    };
    let m = cols.len();
    let len = if samplewise { n } else { m };
    let (mut tp, mut ps, mut ts) = (vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for i in 0..n {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        let row_t = &y_true[i * n_labels..(i + 1) * n_labels];
        let row_p = &y_pred[i * n_labels..(i + 1) * n_labels];
        for (k, &j) in cols.iter().enumerate() {
            let (t, p) = (row_t[j] != 0, row_p[j] != 0);
            let slot = if samplewise { i } else { k };
            if t && p {
                tp[slot] += w;
            }
            if p {
                ps[slot] += w;
            }
            if t {
                ts[slot] += w;
            }
        }
    }
    let fp: Vec<f64> = (0..len).map(|i| ps[i] - tp[i]).collect();
    let fnn: Vec<f64> = (0..len).map(|i| ts[i] - tp[i]).collect();
    let tn: Vec<f64> = if samplewise {
        (0..len)
            .map(|i| {
                let w = sample_weight.map_or(1.0, |sw| sw[i]);
                w * m as f64 - tp[i] - fp[i] - fnn[i]
            })
            .collect()
    } else {
        let total: f64 = match sample_weight {
            Some(w) => w.iter().sum(),
            None => n as f64,
        };
        (0..len).map(|i| total - tp[i] - fp[i] - fnn[i]).collect()
    };
    Ok(Mcm { tp, fp, fnn, tn })
}

/// Divide-and-average over already-collected counts (sklearn's
/// `precision_recall_fscore_support` after its MCM call).
///
/// `beta = +inf` makes F the recall and `beta = 0` the precision (sklearn's two
/// short-circuits); otherwise `F = (1+β²)·tp / (β²·true_sum + pred_sum)`, which
/// is algebraically the harmonic form but computed from counts, so it never
/// double-rounds through the two ratios. `samples_weight` is the per-sample
/// weight for `SetAverage::Samples` (sklearn averages the per-sample scores
/// with it).
pub fn prfs_from_counts(
    mcm: &Mcm,
    beta: f64,
    average: SetAverage,
    samples_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> PrfsResult {
    let mut tp = mcm.tp.clone();
    let mut pred_sum: Vec<f64> = (0..tp.len()).map(|i| mcm.tp[i] + mcm.fp[i]).collect();
    let mut true_sum: Vec<f64> = (0..tp.len()).map(|i| mcm.tp[i] + mcm.fnn[i]).collect();
    if average == SetAverage::Micro {
        tp = vec![tp.iter().sum()];
        pred_sum = vec![pred_sum.iter().sum()];
        true_sum = vec![true_sum.iter().sum()];
    }
    let (precision, precision_zd) = prf_divide(&tp, &pred_sum, zero_division);
    let (recall, recall_zd) = prf_divide(&tp, &true_sum, zero_division);
    let (fscore, fscore_zd) = if beta.is_infinite() && beta > 0.0 {
        (recall.clone(), recall_zd)
    } else if beta == 0.0 {
        (precision.clone(), precision_zd)
    } else {
        let b2 = beta * beta;
        let num: Vec<f64> = tp.iter().map(|&t| (1.0 + b2) * t).collect();
        let den: Vec<f64> = (0..tp.len()).map(|i| b2 * true_sum[i] + pred_sum[i]).collect();
        prf_divide(&num, &den, zero_division)
    };

    if average == SetAverage::None_ {
        return PrfsResult {
            precision: PrfOut::PerClass(precision),
            recall: PrfOut::PerClass(recall),
            fscore: PrfOut::PerClass(fscore),
            support: Some(true_sum),
            precision_zd,
            recall_zd,
            fscore_zd,
            classes: Vec::new(),
        };
    }
    let weights: Option<&[f64]> = match average {
        SetAverage::Weighted => Some(&true_sum),
        SetAverage::Samples => samples_weight,
        _ => None,
    };
    PrfsResult {
        precision: PrfOut::Scalar(nanaverage(&precision, weights)),
        recall: PrfOut::Scalar(nanaverage(&recall, weights)),
        fscore: PrfOut::Scalar(nanaverage(&fscore, weights)),
        support: None,
        precision_zd,
        recall_zd,
        fscore_zd,
        classes: Vec::new(),
    }
}

/// `precision_recall_fscore_support` for a 1-D target.
///
/// `SetAverage::Binary` reads the `pos_label` class out of the counts over
/// the PRESENT labels (sklearn's `labels=[pos_label]`; a `pos_label` absent
/// from the data is all-zero counts, i.e. a zero-division case). `labels` is
/// ignored for `Binary`, as in sklearn. `Samples` is a multilabel-only mode and
/// returns `Err(UnsupportedTarget)` here.
#[allow(clippy::too_many_arguments)]
pub fn prfs_multiclass(
    y_true: &[i32],
    y_pred: &[i32],
    beta: f64,
    labels: Option<&[i32]>,
    pos_label: i32,
    average: SetAverage,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<PrfsResult, MetricError> {
    if !(beta >= 0.0) {
        return Err(MetricError::InvalidParameter);
    }
    match average {
        SetAverage::Samples => Err(MetricError::UnsupportedTarget),
        SetAverage::Binary => {
            let (classes, mcm) = mcm_multiclass(y_true, y_pred, sample_weight, None)?;
            let one = match classes.iter().position(|&c| c == pos_label) {
                Some(i) => Mcm {
                    tp: vec![mcm.tp[i]],
                    fp: vec![mcm.fp[i]],
                    fnn: vec![mcm.fnn[i]],
                    tn: vec![mcm.tn[i]],
                },
                None => Mcm {
                    tp: vec![0.0],
                    fp: vec![0.0],
                    fnn: vec![0.0],
                    tn: vec![0.0],
                },
            };
            let mut r = prfs_from_counts(&one, beta, SetAverage::Binary, None, zero_division);
            r.classes = classes;
            Ok(r)
        }
        _ => {
            let (classes, mcm) = mcm_multiclass(y_true, y_pred, sample_weight, labels)?;
            let mut r = prfs_from_counts(&mcm, beta, average, None, zero_division);
            r.classes = classes;
            Ok(r)
        }
    }
}

/// `precision_recall_fscore_support` for a row-major multilabel indicator
/// target. `Binary` is not a multilabel mode (`Err(UnsupportedTarget)`);
/// `Samples` averages the per-sample scores weighted by `sample_weight`.
#[allow(clippy::too_many_arguments)]
pub fn prfs_indicator(
    y_true: &[i32],
    y_pred: &[i32],
    n_labels: usize,
    beta: f64,
    label_cols: Option<&[usize]>,
    average: SetAverage,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<PrfsResult, MetricError> {
    if !(beta >= 0.0) {
        return Err(MetricError::InvalidParameter);
    }
    if average == SetAverage::Binary {
        return Err(MetricError::UnsupportedTarget);
    }
    let samplewise = average == SetAverage::Samples;
    let mcm = mcm_indicator(y_true, y_pred, n_labels, sample_weight, label_cols, samplewise)?;
    Ok(prfs_from_counts(&mcm, beta, average, sample_weight, zero_division))
}

/// Jaccard over already-collected counts: `tp / (tp + fp + fn)`, with
/// sklearn's `jaccard_score` averaging — which (unlike
/// `precision_recall_fscore_support`) uses the plain `_average`, so a NaN from
/// `zero_division=np.nan` PROPAGATES, and `average='weighted'` with an all-zero
/// support falls back to the unweighted mean.
pub fn jaccard_from_counts(
    mcm: &Mcm,
    average: SetAverage,
    samples_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> (PrfOut, bool) {
    let mut num = mcm.tp.clone();
    let mut den: Vec<f64> = (0..num.len()).map(|i| mcm.tp[i] + mcm.fp[i] + mcm.fnn[i]).collect();
    if average == SetAverage::Micro {
        num = vec![num.iter().sum()];
        den = vec![den.iter().sum()];
    }
    let (j, hit) = prf_divide(&num, &den, zero_division);
    if average == SetAverage::None_ {
        return (PrfOut::PerClass(j), hit);
    }
    let support: Vec<f64> = (0..mcm.tp.len()).map(|i| mcm.tp[i] + mcm.fnn[i]).collect();
    let weights: Option<&[f64]> = match average {
        SetAverage::Weighted if support.iter().any(|&s| s != 0.0) => Some(&support),
        SetAverage::Samples => samples_weight,
        _ => None,
    };
    let v = average_or_mean(&j, weights);
    (PrfOut::Scalar(v), hit)
}

fn average_or_mean(values: &[f64], weights: Option<&[f64]>) -> f64 {
    average(values, weights).unwrap_or(f64::NAN)
}

/// `jaccard_score` for a 1-D target (`Binary` reads the `pos_label` class, as
/// in [`prfs_multiclass`]). Returns the value, the zero-division flag, and the
/// resolved class order.
pub fn jaccard_multiclass(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: SetAverage,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<(PrfOut, bool, Vec<i32>), MetricError> {
    match average {
        SetAverage::Samples => Err(MetricError::UnsupportedTarget),
        SetAverage::Binary => {
            let (classes, mcm) = mcm_multiclass(y_true, y_pred, sample_weight, None)?;
            let one = match classes.iter().position(|&c| c == pos_label) {
                Some(i) => Mcm {
                    tp: vec![mcm.tp[i]],
                    fp: vec![mcm.fp[i]],
                    fnn: vec![mcm.fnn[i]],
                    tn: vec![mcm.tn[i]],
                },
                None => Mcm {
                    tp: vec![0.0],
                    fp: vec![0.0],
                    fnn: vec![0.0],
                    tn: vec![0.0],
                },
            };
            let (v, hit) = jaccard_from_counts(&one, SetAverage::Binary, None, zero_division);
            Ok((v, hit, classes))
        }
        _ => {
            let (classes, mcm) = mcm_multiclass(y_true, y_pred, sample_weight, labels)?;
            let (v, hit) = jaccard_from_counts(&mcm, average, None, zero_division);
            Ok((v, hit, classes))
        }
    }
}

/// `jaccard_score` for a row-major multilabel indicator target.
pub fn jaccard_indicator(
    y_true: &[i32],
    y_pred: &[i32],
    n_labels: usize,
    label_cols: Option<&[usize]>,
    average: SetAverage,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<(PrfOut, bool), MetricError> {
    if average == SetAverage::Binary {
        return Err(MetricError::UnsupportedTarget);
    }
    let samplewise = average == SetAverage::Samples;
    let mcm = mcm_indicator(y_true, y_pred, n_labels, sample_weight, label_cols, samplewise)?;
    Ok(jaccard_from_counts(&mcm, average, sample_weight, zero_division))
}

/// `multilabel_confusion_matrix` as a flat `n × 4` row-major `[tn, fp, fn, tp]`
/// buffer per label (sklearn's `(n, 2, 2)` array, raveled), for a 1-D target.
pub fn multilabel_confusion_matrix(
    y_true: &[i32],
    y_pred: &[i32],
    sample_weight: Option<&[f64]>,
    labels: Option<&[i32]>,
) -> Result<Vec<f64>, MetricError> {
    let (_, mcm) = mcm_multiclass(y_true, y_pred, sample_weight, labels)?;
    Ok(ravel_mcm(&mcm))
}

/// [`multilabel_confusion_matrix`] for a multilabel indicator target,
/// column-wise or (`samplewise`) row-wise.
pub fn multilabel_confusion_matrix_indicator(
    y_true: &[i32],
    y_pred: &[i32],
    n_labels: usize,
    sample_weight: Option<&[f64]>,
    label_cols: Option<&[usize]>,
    samplewise: bool,
) -> Result<Vec<f64>, MetricError> {
    let mcm = mcm_indicator(y_true, y_pred, n_labels, sample_weight, label_cols, samplewise)?;
    Ok(ravel_mcm(&mcm))
}

fn ravel_mcm(mcm: &Mcm) -> Vec<f64> {
    let mut out = Vec::with_capacity(mcm.tp.len() * 4);
    for i in 0..mcm.tp.len() {
        out.extend_from_slice(&[mcm.tn[i], mcm.fp[i], mcm.fnn[i], mcm.tp[i]]);
    }
    out
}
