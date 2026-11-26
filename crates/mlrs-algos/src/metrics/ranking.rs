//! Ranking / threshold metrics: the binary classifier curve and everything
//! sklearn builds on it (`confusion_matrix_at_thresholds`, `roc_curve`,
//! `det_curve`, `average_precision_score`), `auc`, and the multilabel ranking
//! metrics (`coverage_error`, `label_ranking_average_precision_score`,
//! `label_ranking_loss`, `dcg_score`, `ndcg_score`).
//!
//! ## One curve, sklearn's exact thresholds
//! [`binary_clf_curve`] reproduces sklearn's
//! `_sort_inputs_and_compute_classification_thresholds` + cumulative counts:
//! zero-weight samples are REMOVED first (so a score carried only by
//! zero-weight samples is not a threshold), then a stable descending sort, and
//! one threshold per distinct score at the LAST index of its run. The curve
//! functions are pure O(m) post-processing of that one O(n log n) pass.
//!
//! Multilabel inputs are row-major `n_samples × n_labels` buffers. Tests live
//! in `crates/mlrs-algos/tests/metrics_ranking_test.rs` (AGENTS.md §2).

use super::prfs::average as weighted_average;
use super::{validate_weight, MetricError, PrfOut};

/// The cumulative counts at each distinct threshold, in DESCENDING threshold
/// order: `tps[i]`/`fps[i]` are the weighted positives/negatives with
/// `score >= thresholds[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ClfCurve {
    pub fps: Vec<f64>,
    pub tps: Vec<f64>,
    pub thresholds: Vec<f64>,
}

/// sklearn's binary classifier curve over `y_true == pos_label`.
///
/// `Err(NaNScore)` for a NaN or infinite score (sklearn's `assert_all_finite`),
/// `Err(LengthMismatch)`/`Err(InvalidWeight)` for a bad `sample_weight`.
pub fn binary_clf_curve(
    y_true: &[i32],
    y_score: &[f64],
    pos_label: i32,
    sample_weight: Option<&[f64]>,
) -> Result<ClfCurve, MetricError> {
    if y_true.len() != y_score.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_true.len(), sample_weight)?;
    if y_score.iter().any(|v| !v.is_finite()) {
        return Err(MetricError::NaNScore);
    }
    let mut idx: Vec<usize> = (0..y_true.len())
        .filter(|&i| sample_weight.is_none_or(|w| w[i] != 0.0))
        .collect();
    // Stable sort, descending. Ties keep their input order, which does not
    // matter for the counts (a threshold sits at the END of its run) but keeps
    // the pass deterministic.
    idx.sort_by(|&a, &b| y_score[b].total_cmp(&y_score[a]));

    let mut curve = ClfCurve {
        fps: Vec::new(),
        tps: Vec::new(),
        thresholds: Vec::new(),
    };
    let (mut tp, mut fp) = (0.0f64, 0.0f64);
    for (k, &i) in idx.iter().enumerate() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        if y_true[i] == pos_label {
            tp += w;
        } else {
            fp += w;
        }
        let last_of_run = k + 1 == idx.len() || y_score[idx[k + 1]] != y_score[i];
        if last_of_run {
            curve.tps.push(tp);
            curve.fps.push(fp);
            curve.thresholds.push(y_score[i]);
        }
    }
    Ok(curve)
}

/// `confusion_matrix_at_thresholds`: `(tns, fps, fns, tps, thresholds)`, one
/// entry per distinct threshold, descending.
#[allow(clippy::type_complexity)]
pub fn confusion_matrix_at_thresholds(
    y_true: &[i32],
    y_score: &[f64],
    pos_label: i32,
    sample_weight: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), MetricError> {
    let c = binary_clf_curve(y_true, y_score, pos_label, sample_weight)?;
    let (fp_tot, tp_tot) = (
        c.fps.last().copied().unwrap_or(0.0),
        c.tps.last().copied().unwrap_or(0.0),
    );
    let tns = c.fps.iter().map(|&f| fp_tot - f).collect();
    let fns = c.tps.iter().map(|&t| tp_tot - t).collect();
    Ok((tns, c.fps, fns, c.tps, c.thresholds))
}

/// `roc_curve`'s result. `no_negative`/`no_positive` mark sklearn's two
/// degenerate cases (the matching rate is all-NaN and sklearn warns).
#[derive(Debug, Clone, PartialEq)]
pub struct RocCurve {
    pub fpr: Vec<f64>,
    pub tpr: Vec<f64>,
    pub thresholds: Vec<f64>,
    pub no_negative: bool,
    pub no_positive: bool,
}

/// Indices of a curve kept by sklearn's `drop_intermediate`: the endpoints,
/// plus every interior point where `keep(i)` holds.
fn kept(len: usize, keep: impl Fn(usize) -> bool) -> Vec<usize> {
    (0..len)
        .filter(|&i| i == 0 || i + 1 == len || keep(i))
        .collect()
}

/// `roc_curve`: `(fpr, tpr, thresholds)` with sklearn's leading `(0, 0, +inf)`
/// point. `drop_intermediate` removes the collinear interior points (those
/// whose second differences in BOTH `fps` and `tps` are zero), before the
/// leading point is added.
pub fn roc_curve(
    y_true: &[i32],
    y_score: &[f64],
    pos_label: i32,
    sample_weight: Option<&[f64]>,
    drop_intermediate: bool,
) -> Result<RocCurve, MetricError> {
    let c = binary_clf_curve(y_true, y_score, pos_label, sample_weight)?;
    let m = c.fps.len();
    let keep: Vec<usize> = if drop_intermediate && m > 2 {
        kept(m, |i| {
            c.fps[i - 1] - 2.0 * c.fps[i] + c.fps[i + 1] != 0.0
                || c.tps[i - 1] - 2.0 * c.tps[i] + c.tps[i + 1] != 0.0
        })
    } else {
        (0..m).collect()
    };
    let mut fps = vec![0.0];
    let mut tps = vec![0.0];
    let mut thresholds = vec![f64::INFINITY];
    for &i in &keep {
        fps.push(c.fps[i]);
        tps.push(c.tps[i]);
        thresholds.push(c.thresholds[i]);
    }
    let (fp_tot, tp_tot) = (*fps.last().unwrap(), *tps.last().unwrap());
    let no_negative = fp_tot <= 0.0;
    let no_positive = tp_tot <= 0.0;
    let fpr = fps
        .iter()
        .map(|&f| if no_negative { f64::NAN } else { f / fp_tot })
        .collect();
    let tpr = tps
        .iter()
        .map(|&t| if no_positive { f64::NAN } else { t / tp_tot })
        .collect();
    Ok(RocCurve {
        fpr,
        tpr,
        thresholds,
        no_negative,
        no_positive,
    })
}

/// `det_curve`: `(fpr, fnr, thresholds)` in INCREASING threshold order,
/// trimmed exactly as sklearn trims it (from the last point still at the
/// minimum false-positive count to the first point reaching the full
/// true-positive count). `Err(TooFewPoints)` unless `y_true` holds exactly two
/// distinct labels.
pub fn det_curve(
    y_true: &[i32],
    y_score: &[f64],
    pos_label: i32,
    sample_weight: Option<&[f64]>,
    drop_intermediate: bool,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), MetricError> {
    let c = binary_clf_curve(y_true, y_score, pos_label, sample_weight)?;
    let mut fps = vec![0.0];
    let mut tps = vec![0.0];
    let mut thresholds = vec![f64::INFINITY];
    fps.extend_from_slice(&c.fps);
    tps.extend_from_slice(&c.tps);
    thresholds.extend_from_slice(&c.thresholds);
    if drop_intermediate && fps.len() > 2 {
        let keep = kept(fps.len(), |i| tps[i] != tps[i - 1] || tps[i + 1] != tps[i]);
        fps = keep.iter().map(|&i| fps[i]).collect();
        tps = keep.iter().map(|&i| tps[i]).collect();
        thresholds = keep.iter().map(|&i| thresholds[i]).collect();
    }
    let mut distinct = y_true.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != 2 {
        return Err(MetricError::TooFewPoints);
    }
    let p_count = *tps.last().unwrap();
    let n_count = *fps.last().unwrap();
    // `searchsorted(fps, fps[0], 'right') - 1`: the last index still at fps[0]
    // (fps is non-decreasing).
    let first = fps.partition_point(|&f| f <= fps[0]) - 1;
    // `searchsorted(tps, tps[-1], 'left') + 1`.
    let last = tps.partition_point(|&t| t < p_count) + 1;
    let range = first..last.max(first);
    let fpr = range.clone().rev().map(|i| fps[i] / n_count).collect();
    let fnr = range.clone().rev().map(|i| (p_count - tps[i]) / p_count).collect();
    let thr = range.rev().map(|i| thresholds[i]).collect();
    Ok((fpr, fnr, thr))
}

/// `auc`: trapezoidal area under `(x, y)`, for `x` monotonically increasing OR
/// decreasing (the latter negates the integral, as sklearn's `direction`).
pub fn auc(x: &[f64], y: &[f64]) -> Result<f64, MetricError> {
    if x.len() != y.len() {
        return Err(MetricError::LengthMismatch);
    }
    if x.len() < 2 {
        return Err(MetricError::TooFewPoints);
    }
    let dx: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let mut direction = 1.0;
    if dx.iter().any(|&d| d < 0.0) {
        if dx.iter().all(|&d| d <= 0.0) {
            direction = -1.0;
        } else {
            return Err(MetricError::NonMonotonic);
        }
    }
    let area: f64 = dx
        .iter()
        .enumerate()
        .map(|(i, &d)| d * (y[i + 1] + y[i]) / 2.0)
        .sum();
    Ok(direction * area)
}

/// Binary (uninterpolated) average precision: `Σ (R_j - R_{j-1}) · P_j` over
/// the precision-recall curve in descending-threshold order, clamped at 0 —
/// sklearn's `max(0, -Σ diff(recall) · precision[:-1])`. A target with no
/// positive sample yields 0 (its recall column is all ones).
pub fn average_precision_binary(
    y_true: &[i32],
    y_score: &[f64],
    pos_label: i32,
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    let c = binary_clf_curve(y_true, y_score, pos_label, sample_weight)?;
    let tp_tot = c.tps.last().copied().unwrap_or(0.0);
    let mut prev_recall = 0.0f64;
    let mut ap = 0.0f64;
    for i in 0..c.tps.len() {
        let ps = c.tps[i] + c.fps[i];
        let precision = if ps != 0.0 { c.tps[i] / ps } else { 0.0 };
        let recall = if tp_tot > 0.0 { c.tps[i] / tp_tot } else { 1.0 };
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
    }
    Ok(ap.max(0.0))
}

/// The averaging modes of sklearn's `_average_binary_score` (multilabel
/// ranking scores). `None` returns the per-label vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreAverage {
    None_,
    Micro,
    Macro,
    Weighted,
    Samples,
}

/// sklearn's `_average_binary_score` over a row-major `n × L` multilabel
/// indicator `y_true` and score matrix: the binary `metric` per label column
/// (or per sample row for `Samples`, or ONCE over the raveled pairs for
/// `Micro`), then the requested average. A label whose average weight is zero
/// is forced to score 0 before averaging; a `Weighted` call whose total
/// weight is (close to) zero returns `0`.
pub fn average_binary_score(
    metric: impl Fn(&[i32], &[f64], Option<&[f64]>) -> Result<f64, MetricError>,
    y_true: &[i32],
    y_score: &[f64],
    n_labels: usize,
    average: ScoreAverage,
    sample_weight: Option<&[f64]>,
) -> Result<PrfOut, MetricError> {
    if y_true.len() != y_score.len() {
        return Err(MetricError::LengthMismatch);
    }
    if n_labels == 0 || y_true.len() % n_labels != 0 {
        return Err(MetricError::BadShape);
    }
    let n = y_true.len() / n_labels;
    validate_weight(n, sample_weight)?;
    let bin: Vec<i32> = y_true.iter().map(|&v| i32::from(v != 0)).collect();

    if average == ScoreAverage::Micro {
        let w_rep: Option<Vec<f64>> = sample_weight.map(|sw| {
            sw.iter()
                .flat_map(|&w| std::iter::repeat_n(w, n_labels))
                .collect()
        });
        return Ok(PrfOut::Scalar(metric(&bin, y_score, w_rep.as_deref())?));
    }

    let mut average_weight: Option<Vec<f64>> = None;
    if average == ScoreAverage::Weighted {
        let mut aw = vec![0.0f64; n_labels];
        for i in 0..n {
            let w = sample_weight.map_or(1.0, |sw| sw[i]);
            for (j, a) in aw.iter_mut().enumerate() {
                *a += f64::from(bin[i * n_labels + j]) * w;
            }
        }
        let total: f64 = aw.iter().sum();
        if total.abs() <= 1e-8 {
            return Ok(PrfOut::Scalar(0.0));
        }
        average_weight = Some(aw);
    }

    let scores: Vec<f64> = if average == ScoreAverage::Samples {
        average_weight = sample_weight.map(<[f64]>::to_vec);
        (0..n)
            .map(|i| {
                let r = i * n_labels..(i + 1) * n_labels;
                metric(&bin[r.clone()], &y_score[r], None)
            })
            .collect::<Result<_, _>>()?
    } else {
        let mut col_t = vec![0i32; n];
        let mut col_s = vec![0.0f64; n];
        (0..n_labels)
            .map(|j| {
                for i in 0..n {
                    col_t[i] = bin[i * n_labels + j];
                    col_s[i] = y_score[i * n_labels + j];
                }
                metric(&col_t, &col_s, sample_weight)
            })
            .collect::<Result<_, _>>()?
    };

    if average == ScoreAverage::None_ {
        return Ok(PrfOut::PerClass(scores));
    }
    let scores: Vec<f64> = match &average_weight {
        Some(aw) => scores
            .iter()
            .zip(aw)
            .map(|(&s, &w)| if w == 0.0 { 0.0 } else { s })
            .collect(),
        None => scores,
    };
    weighted_average(&scores, average_weight.as_deref())
        .map(PrfOut::Scalar)
        .ok_or(MetricError::ZeroWeightSum)
}

/// Multilabel / multiclass-binarized `average_precision_score`.
pub fn average_precision_indicator(
    y_true: &[i32],
    y_score: &[f64],
    n_labels: usize,
    average: ScoreAverage,
    sample_weight: Option<&[f64]>,
) -> Result<PrfOut, MetricError> {
    average_binary_score(
        |t, s, w| average_precision_binary(t, s, 1, w),
        y_true,
        y_score,
        n_labels,
        average,
        sample_weight,
    )
}

fn check_ranking_shape(
    y_true_len: usize,
    y_score_len: usize,
    n_labels: usize,
    sample_weight: Option<&[f64]>,
) -> Result<usize, MetricError> {
    if y_true_len != y_score_len {
        return Err(MetricError::LengthMismatch);
    }
    if n_labels == 0 || y_true_len % n_labels != 0 {
        return Err(MetricError::BadShape);
    }
    let n = y_true_len / n_labels;
    validate_weight(n, sample_weight)?;
    Ok(n)
}

fn weighted_mean(values: &[f64], sample_weight: Option<&[f64]>) -> Result<f64, MetricError> {
    weighted_average(values, sample_weight).ok_or(MetricError::ZeroWeightSum)
}

/// `coverage_error`: per sample, how far down the score ranking one must go to
/// cover every relevant label — the count of labels scoring at least the
/// LOWEST relevant score (0 for a sample with no relevant label).
pub fn coverage_error(
    y_true: &[i32],
    y_score: &[f64],
    n_labels: usize,
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    let n = check_ranking_shape(y_true.len(), y_score.len(), n_labels, sample_weight)?;
    let cov: Vec<f64> = (0..n)
        .map(|i| {
            let t = &y_true[i * n_labels..(i + 1) * n_labels];
            let s = &y_score[i * n_labels..(i + 1) * n_labels];
            let min_rel = (0..n_labels)
                .filter(|&j| t[j] != 0)
                .map(|j| s[j])
                .fold(f64::INFINITY, f64::min);
            if min_rel == f64::INFINITY && !t.iter().any(|&v| v != 0) {
                0.0
            } else {
                s.iter().filter(|&&v| v >= min_rel).count() as f64
            }
        })
        .collect();
    weighted_mean(&cov, sample_weight)
}

/// `label_ranking_average_precision_score`: per sample, the mean over its
/// relevant labels `j` of `|{relevant l : s_l >= s_j}| / |{l : s_l >= s_j}|`
/// (both "max" ranks); a sample whose relevant set is empty or complete scores
/// 1.
pub fn label_ranking_average_precision_score(
    y_true: &[i32],
    y_score: &[f64],
    n_labels: usize,
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    let n = check_ranking_shape(y_true.len(), y_score.len(), n_labels, sample_weight)?;
    let mut out = 0.0f64;
    for i in 0..n {
        let t = &y_true[i * n_labels..(i + 1) * n_labels];
        let s = &y_score[i * n_labels..(i + 1) * n_labels];
        let relevant: Vec<usize> = (0..n_labels).filter(|&j| t[j] != 0).collect();
        let aux = if relevant.is_empty() || relevant.len() == n_labels {
            1.0
        } else {
            relevant
                .iter()
                .map(|&j| {
                    let rank = s.iter().filter(|&&v| v >= s[j]).count() as f64;
                    let l = relevant.iter().filter(|&&r| s[r] >= s[j]).count() as f64;
                    l / rank
                })
                .sum::<f64>()
                / relevant.len() as f64
        };
        out += aux * sample_weight.map_or(1.0, |sw| sw[i]);
    }
    let total = sample_weight.map_or(n as f64, |sw| sw.iter().sum());
    Ok(out / total)
}

/// `label_ranking_loss`: per sample, the fraction of (relevant, irrelevant)
/// label pairs that are mis-ordered — the irrelevant label scoring AT LEAST as
/// high — over `|relevant| · |irrelevant|`; 0 when either set is empty.
pub fn label_ranking_loss(
    y_true: &[i32],
    y_score: &[f64],
    n_labels: usize,
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    let n = check_ranking_shape(y_true.len(), y_score.len(), n_labels, sample_weight)?;
    let loss: Vec<f64> = (0..n)
        .map(|i| {
            let t = &y_true[i * n_labels..(i + 1) * n_labels];
            let s = &y_score[i * n_labels..(i + 1) * n_labels];
            let n_pos = t.iter().filter(|&&v| v != 0).count();
            if n_pos == 0 || n_pos == n_labels {
                return 0.0;
            }
            let mut bad = 0usize;
            for a in 0..n_labels {
                if t[a] == 0 {
                    continue;
                }
                for b in 0..n_labels {
                    if t[b] == 0 && s[b] >= s[a] {
                        bad += 1;
                    }
                }
            }
            bad as f64 / ((n_labels - n_pos) * n_pos) as f64
        })
        .collect();
    weighted_mean(&loss, sample_weight)
}

/// Per-sample DCG, sklearn's `_dcg_sample_scores`. With `ignore_ties = false`
/// a group of tied scores shares the AVERAGE gain of its members across the
/// discounts of the positions it spans (`_tie_averaged_dcg`); with `true` the
/// order is a reversed stable ascending sort.
fn dcg_samples(
    y_true: &[f64],
    y_score: &[f64],
    n_labels: usize,
    k: Option<usize>,
    log_base: f64,
    ignore_ties: bool,
) -> Vec<f64> {
    let n = y_true.len() / n_labels;
    let ln_base = log_base.ln();
    let discount: Vec<f64> = (0..n_labels)
        .map(|r| {
            if k.is_some_and(|k| r >= k) {
                0.0
            } else {
                1.0 / (((r + 2) as f64).ln() / ln_base)
            }
        })
        .collect();
    let mut cum = Vec::with_capacity(n_labels);
    let mut run = 0.0f64;
    for &d in &discount {
        run += d;
        cum.push(run);
    }
    (0..n)
        .map(|i| {
            let t = &y_true[i * n_labels..(i + 1) * n_labels];
            let s = &y_score[i * n_labels..(i + 1) * n_labels];
            let mut order: Vec<usize> = (0..n_labels).collect();
            if ignore_ties {
                order.sort_by(|&a, &b| s[a].total_cmp(&s[b]));
                order.reverse();
                return order
                    .iter()
                    .enumerate()
                    .map(|(r, &j)| discount[r] * t[j])
                    .sum();
            }
            order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
            let mut gain = 0.0f64;
            let mut start = 0usize;
            while start < n_labels {
                let mut end = start;
                let mut sum = 0.0f64;
                while end < n_labels && s[order[end]] == s[order[start]] {
                    sum += t[order[end]];
                    end += 1;
                }
                let count = (end - start) as f64;
                let disc = cum[end - 1] - if start == 0 { 0.0 } else { cum[start - 1] };
                gain += sum / count * disc;
                start = end;
            }
            gain
        })
        .collect()
}

/// `dcg_score`: the (weighted) mean per-sample discounted cumulative gain.
#[allow(clippy::too_many_arguments)]
pub fn dcg_score(
    y_true: &[f64],
    y_score: &[f64],
    n_labels: usize,
    k: Option<usize>,
    log_base: f64,
    sample_weight: Option<&[f64]>,
    ignore_ties: bool,
) -> Result<f64, MetricError> {
    check_ranking_shape(y_true.len(), y_score.len(), n_labels, sample_weight)?;
    if !(log_base > 0.0) || log_base == 1.0 {
        return Err(MetricError::InvalidParameter);
    }
    let g = dcg_samples(y_true, y_score, n_labels, k, log_base, ignore_ties);
    weighted_mean(&g, sample_weight)
}

/// `ndcg_score`: per-sample DCG over the ideal DCG (the true gains sorted by
/// themselves), 0 for a sample whose ideal DCG is 0, then the weighted mean.
pub fn ndcg_score(
    y_true: &[f64],
    y_score: &[f64],
    n_labels: usize,
    k: Option<usize>,
    sample_weight: Option<&[f64]>,
    ignore_ties: bool,
) -> Result<f64, MetricError> {
    check_ranking_shape(y_true.len(), y_score.len(), n_labels, sample_weight)?;
    if y_true.iter().any(|&v| v < 0.0) {
        return Err(MetricError::DomainError);
    }
    let gain = dcg_samples(y_true, y_score, n_labels, k, 2.0, ignore_ties);
    let ideal = dcg_samples(y_true, y_true, n_labels, k, 2.0, true);
    let g: Vec<f64> = gain
        .iter()
        .zip(&ideal)
        .map(|(&g, &z)| if z == 0.0 { 0.0 } else { g / z })
        .collect();
    weighted_mean(&g, sample_weight)
}
