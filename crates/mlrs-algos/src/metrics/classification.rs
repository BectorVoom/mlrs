//! Classification metrics (METR-CLS-01..09).
//!
//! Built on [`super::class_bookkeeping`]'s shared weighted TP/FP/FN
//! accumulation for the label-based metrics (`accuracy_score`,
//! `confusion_matrix`, `precision_score`/`recall_score`/`f1_score`) and a
//! shared sort-by-descending-score sweep ([`sweep`]) for the rank-based
//! metrics (`roc_auc_score_binary`/`_multiclass`, `precision_recall_curve`).
//!
//! Tests live in `crates/mlrs-algos/tests/metrics_classification_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)] mod tests`).

use super::{
    class_bookkeeping, validate_weight, Average, MetricError, MultiClass, PrfOut, ZeroDivision,
};

// ==================== TASK-03 — METR-CLS-01: accuracy_score ====================

/// The fraction of exact matches between `y_true` and `y_pred` (weighted, or
/// weighted count if `normalize=false`). `sample_weight=None` uses unit
/// weights. Empty input yields `0.0/0.0 = NaN` (IEEE-754), matching
/// `nb_common::accuracy_score`'s documented empty-input contract without a
/// special-cased branch.
///
/// NOTE: `nb_common::accuracy_score(pred, y_true)` (existing, opposite arg
/// order) is now a thin delegate to this function (TASK-03, SPEC §5
/// CLS-01) — ONE source of truth.
pub fn accuracy_score(
    y_true: &[i32],
    y_pred: &[i32],
    sample_weight: Option<&[f64]>,
    normalize: bool,
) -> f64 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "accuracy_score: length mismatch y_true={} y_pred={}",
        y_true.len(),
        y_pred.len()
    );
    let mut correct = 0.0f64;
    let mut total = 0.0f64;
    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        total += w;
        if y_true[i] == y_pred[i] {
            correct += w;
        }
    }
    if normalize {
        correct / total
    } else {
        correct
    }
}

// ==================== TASK-04 — METR-CLS-02: confusion_matrix ====================

/// The `C×C` (weighted) confusion matrix: `matrix[i][j]` is the weighted
/// count of samples with true label `classes[i]` and predicted label
/// `classes[j]`, in the resolved class order (sorted unique of `y_true ∪
/// y_pred` when `labels=None`, else `labels` verbatim — including a class
/// absent from the data, which gets a full zero row/column).
pub fn confusion_matrix(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    sample_weight: Option<&[f64]>,
) -> Vec<Vec<f64>> {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "confusion_matrix: length mismatch"
    );
    let classes: Vec<i32> = match labels {
        Some(ls) => ls.to_vec(),
        None => {
            let mut set: Vec<i32> = y_true.iter().chain(y_pred.iter()).copied().collect();
            set.sort_unstable();
            set.dedup();
            set
        }
    };
    let n = classes.len();
    let mut matrix = vec![vec![0.0f64; n]; n];
    let index_of = |c: i32| classes.iter().position(|&x| x == c);
    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        if let (Some(ti), Some(pi)) = (index_of(y_true[i]), index_of(y_pred[i])) {
            matrix[ti][pi] += w;
        }
    }
    matrix
}

// ==================== TASK-05/06/07 — precision/recall/f1 ====================

/// Shared per-`average` dispatch over a per-class ratio (precision =
/// `tp/(tp+fp)`, recall = `tp/(tp+fn)`, f1 = `2*tp/(2*tp+fp+fn)`), reused by
/// `precision_score`/`recall_score`/`f1_score` so all three share ONE
/// average-dispatch implementation (TASK-05 Refactor step).
fn average_ratio(
    classes: &[i32],
    numerators: &[f64],
    denominators: &[f64],
    supports: &[f64],
    pos_label: i32,
    average: Average,
    zero_division: ZeroDivision,
) -> PrfOut {
    let zd = |zero_division: ZeroDivision| match zero_division {
        ZeroDivision::Zero => 0.0,
        ZeroDivision::One => 1.0,
        ZeroDivision::Nan => f64::NAN,
    };
    let per_class: Vec<f64> = (0..classes.len())
        .map(|i| {
            if denominators[i] > 0.0 {
                numerators[i] / denominators[i]
            } else {
                zd(zero_division)
            }
        })
        .collect();

    match average {
        Average::None_ => PrfOut::PerClass(per_class),
        Average::Binary => {
            // A pos_label absent from BOTH y_true and y_pred (e.g. the f1
            // zero-division degenerate, TASK-07) is not in `classes` at
            // all — its (tp, fp, fn) are all implicitly zero, so this is a
            // zero-division case (matches sklearn's own behavior on this
            // input, empirically confirmed at TASK-02 fixture generation).
            match classes.iter().position(|&c| c == pos_label) {
                Some(idx) => PrfOut::Scalar(per_class[idx]),
                None => PrfOut::Scalar(zd(zero_division)),
            }
        }
        Average::Macro => {
            let sum: f64 = per_class.iter().sum();
            PrfOut::Scalar(sum / per_class.len() as f64)
        }
        Average::Micro => {
            let num_sum: f64 = numerators.iter().sum();
            let den_sum: f64 = denominators.iter().sum();
            PrfOut::Scalar(if den_sum > 0.0 {
                num_sum / den_sum
            } else {
                zd(zero_division)
            })
        }
        Average::Weighted => {
            let support_sum: f64 = supports.iter().sum();
            if support_sum <= 0.0 {
                return PrfOut::Scalar(zd(zero_division));
            }
            let weighted: f64 = per_class
                .iter()
                .zip(supports.iter())
                .map(|(&r, &s)| r * s)
                .sum();
            PrfOut::Scalar(weighted / support_sum)
        }
    }
}

/// `precision = tp / (tp + fp)` per class, dispatched over `average` (SPEC
/// §5 CLS-03).
pub fn precision_score(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: Average,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> PrfOut {
    let bk = class_bookkeeping(y_true, y_pred, sample_weight, labels)
        .expect("precision_score: invalid input");
    let denom: Vec<f64> = bk
        .tp
        .iter()
        .zip(bk.fp.iter())
        .map(|(&tp, &fp)| tp + fp)
        .collect();
    let support: Vec<f64> = bk
        .tp
        .iter()
        .zip(bk.fnn.iter())
        .map(|(&tp, &fnv)| tp + fnv)
        .collect();
    average_ratio(
        &bk.classes,
        &bk.tp,
        &denom,
        &support,
        pos_label,
        average,
        zero_division,
    )
}

/// `recall = tp / (tp + fn)` per class, dispatched over `average` (SPEC §5
/// CLS-04).
pub fn recall_score(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: Average,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> PrfOut {
    let bk = class_bookkeeping(y_true, y_pred, sample_weight, labels)
        .expect("recall_score: invalid input");
    let denom: Vec<f64> = bk
        .tp
        .iter()
        .zip(bk.fnn.iter())
        .map(|(&tp, &fnv)| tp + fnv)
        .collect();
    let support = denom.clone();
    average_ratio(
        &bk.classes,
        &bk.tp,
        &denom,
        &support,
        pos_label,
        average,
        zero_division,
    )
}

/// `f1 = 2*tp / (2*tp + fp + fn)` per class, computed DIRECTLY from the
/// shared weighted TP/FP/FN (harmonic mean) — NOT from
/// `precision_score(...) × recall_score(...)` floats, to avoid
/// double-rounding (SPEC §5 CLS-05 note, TASK-07).
pub fn f1_score(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: Average,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> PrfOut {
    let bk =
        class_bookkeeping(y_true, y_pred, sample_weight, labels).expect("f1_score: invalid input");
    let numer: Vec<f64> = bk.tp.iter().map(|&tp| 2.0 * tp).collect();
    let denom: Vec<f64> = (0..bk.classes.len())
        .map(|i| 2.0 * bk.tp[i] + bk.fp[i] + bk.fnn[i])
        .collect();
    let support: Vec<f64> = bk
        .tp
        .iter()
        .zip(bk.fnn.iter())
        .map(|(&tp, &fnv)| tp + fnv)
        .collect();
    average_ratio(
        &bk.classes,
        &numer,
        &denom,
        &support,
        pos_label,
        average,
        zero_division,
    )
}

// ==================== TASK-08 — METR-CLS-06: log_loss ====================

/// Weighted multiclass cross-entropy: `-mean_i w_i * ln(p_i[y_true_i])`,
/// with every probability clipped to `[eps, 1-eps]` first (NO
/// renormalization — empirically resolved against `scikit-learn==1.9.0`,
/// TASK-02's degenerate-fixture probe: a row that does not sum to 1
/// produces the CLIP-ONLY value, not the row-renormalized one).
///
/// `labels` (when given) defines the accepted class SET — resolved to its
/// SORTED order for column indexing, exactly matching sklearn's own
/// behavior (empirically probed, TASK-02): passing a non-lexicographic
/// `labels` order (e.g. `[1, 0]`) produces the IDENTICAL value to the
/// sorted order (sklearn warns but does not remap columns). `y_prob` is
/// row-major `n_rows × n_classes`, column `j` corresponding to the `j`-th
/// smallest class in the resolved set.
pub fn log_loss(
    y_true: &[i32],
    y_prob: &[f64],
    n_classes: usize,
    labels: Option<&[i32]>,
    sample_weight: Option<&[f64]>,
    eps: f64,
    normalize: bool,
) -> f64 {
    assert_eq!(
        y_prob.len(),
        y_true.len() * n_classes,
        "log_loss: y_prob shape mismatch"
    );
    let classes: Vec<i32> = match labels {
        Some(ls) => {
            let mut v = ls.to_vec();
            v.sort_unstable();
            v
        }
        None => {
            let mut v: Vec<i32> = y_true.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        }
    };
    let index_of = |c: i32| {
        classes
            .iter()
            .position(|&x| x == c)
            .expect("log_loss: y_true label not in resolved class set")
    };

    let mut sum = 0.0f64;
    let mut weight_total = 0.0f64;
    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        let col = index_of(y_true[i]);
        let p = y_prob[i * n_classes + col].clamp(eps, 1.0 - eps);
        sum += -w * p.ln();
        weight_total += w;
    }
    if normalize {
        sum / weight_total
    } else {
        sum
    }
}

// ==================== Shared rank-based sweep (TASK-09/10/11) ====================

/// Cumulative sweep over samples grouped by exact score value, sorted
/// DESCENDING (highest score first). `cum_tp[i]`/`cum_fp[i]` are the
/// weighted count of positives/negatives with `score >= scores_desc[i]`
/// (i.e. through and including group `i`). Reused by
/// `roc_auc_score_binary`/`_multiclass` and `precision_recall_curve` so the
/// sort+cumulative-count machinery is written exactly once (TASK-09
/// Refactor step).
struct Sweep {
    scores_desc: Vec<f64>,
    cum_tp: Vec<f64>,
    cum_fp: Vec<f64>,
    total_pos: f64,
    total_neg: f64,
}

fn sweep(y_true: &[i32], scores: &[f64], pos_label: i32, sample_weight: Option<&[f64]>) -> Sweep {
    let n = y_true.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .expect("scores must not be NaN")
    });

    let mut scores_desc = Vec::new();
    let mut cum_tp = Vec::new();
    let mut cum_fp = Vec::new();
    let mut run_tp = 0.0f64;
    let mut run_fp = 0.0f64;
    let mut i = 0usize;
    while i < n {
        let s = scores[idx[i]];
        let mut j = i;
        while j < n && scores[idx[j]] == s {
            let w = sample_weight.map_or(1.0, |sw| sw[idx[j]]);
            if y_true[idx[j]] == pos_label {
                run_tp += w;
            } else {
                run_fp += w;
            }
            j += 1;
        }
        scores_desc.push(s);
        cum_tp.push(run_tp);
        cum_fp.push(run_fp);
        i = j;
    }

    let mut total_pos = 0.0f64;
    let mut total_neg = 0.0f64;
    for i in 0..n {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        if y_true[i] == pos_label {
            total_pos += w;
        } else {
            total_neg += w;
        }
    }

    Sweep {
        scores_desc,
        cum_tp,
        cum_fp,
        total_pos,
        total_neg,
    }
}

/// Trapezoidal-integrate the ROC curve implied by a [`Sweep`]: `Σ
/// (fpr[i]-fpr[i-1]) * (tpr[i]+tpr[i-1])/2`, starting from `(0,0)`.
fn auc_from_sweep(sw: &Sweep) -> f64 {
    let mut auc = 0.0f64;
    let mut prev_fpr = 0.0f64;
    let mut prev_tpr = 0.0f64;
    for i in 0..sw.scores_desc.len() {
        let fpr = sw.cum_fp[i] / sw.total_neg;
        let tpr = sw.cum_tp[i] / sw.total_pos;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_fpr = fpr;
        prev_tpr = tpr;
    }
    auc
}

// ==================== TASK-09 — METR-CLS-07: roc_auc_score (binary) ====================

/// Rank-based binary AUC (stable descending sort, average-rank tie
/// handling via [`sweep`]'s exact-score grouping) + trapezoidal
/// integration. Returns `Err(MetricError::SingleClassRocAuc)` when fewer
/// than 2 classes are present in `y_true`, or when either the positive or
/// negative weighted total is zero (an equivalent degenerate case under
/// `sample_weight`) — mlrs deliberately signals this as a typed `Err`
/// rather than mirroring sklearn's own (NaN + `UndefinedMetricWarning`)
/// behavior on this specific input (documented divergence, TASK-02
/// docstring / PLAN.md TASK-09).
pub fn roc_auc_score_binary(
    y_true: &[i32],
    y_score: &[f64],
    pos_label: i32,
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    if y_true.len() != y_score.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_true.len(), sample_weight)?;

    let mut distinct: Vec<i32> = y_true.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() < 2 {
        return Err(MetricError::SingleClassRocAuc);
    }

    let sw = sweep(y_true, y_score, pos_label, sample_weight);
    if sw.total_pos <= 0.0 || sw.total_neg <= 0.0 {
        return Err(MetricError::SingleClassRocAuc);
    }
    Ok(auc_from_sweep(&sw))
}

// ==================== TASK-10 — METR-CLS-08: roc_auc_score (multiclass) ====================

/// Multiclass `roc_auc_score` (OvR/OvO, macro/weighted averages), reusing
/// [`roc_auc_score_binary`]'s sweep helper per class (OvR) or per class-pair
/// (OvO, bidirectional Hand & Till average) — no re-implemented sweep
/// (TASK-10 Refactor step).
///
/// `sample_weight` on the **OvR** path has no carve-out — it always
/// computes a value. On the **OvO** path, `sample_weight.is_some()`
/// returns `Err(MetricError::WeightedOvoUnsupported)` immediately, BEFORE
/// any pairwise sweep, matching the pinned `scikit-learn==1.9.0`'s own
/// rejection of `roc_auc_score(multi_class='ovo', sample_weight=...)`
/// (empirically probed at TASK-02 Green time — Branch A, SPEC §2/§4 Q10,
/// Plan-Check Issue 2).
pub fn roc_auc_score_multiclass(
    y_true: &[i32],
    y_score: &[f64],
    n_classes: usize,
    multi_class: MultiClass,
    average: Average,
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    if y_true.len() * n_classes != y_score.len() {
        return Err(MetricError::BadShape);
    }
    validate_weight(y_true.len(), sample_weight)?;

    match multi_class {
        MultiClass::Ovr => {
            let n = y_true.len();
            let mut per_class_auc = Vec::with_capacity(n_classes);
            let mut prevalence = Vec::with_capacity(n_classes);
            for c in 0..n_classes {
                let y_bin: Vec<i32> = y_true
                    .iter()
                    .map(|&t| if t == c as i32 { 1 } else { 0 })
                    .collect();
                let scores_c: Vec<f64> = (0..n).map(|i| y_score[i * n_classes + c]).collect();
                let auc = roc_auc_score_binary(&y_bin, &scores_c, 1, sample_weight)?;
                per_class_auc.push(auc);
                let prev: f64 = (0..n)
                    .filter(|&i| y_true[i] == c as i32)
                    .map(|i| sample_weight.map_or(1.0, |sw| sw[i]))
                    .sum();
                prevalence.push(prev);
            }
            Ok(match average {
                Average::Weighted => {
                    let total: f64 = prevalence.iter().sum();
                    per_class_auc
                        .iter()
                        .zip(prevalence.iter())
                        .map(|(&a, &p)| a * p)
                        .sum::<f64>()
                        / total
                }
                _ => per_class_auc.iter().sum::<f64>() / n_classes as f64,
            })
        }
        MultiClass::Ovo => {
            if sample_weight.is_some() {
                return Err(MetricError::WeightedOvoUnsupported);
            }
            let n = y_true.len();
            let mut pair_aucs = Vec::new();
            let mut pair_weights = Vec::new();
            let prevalence: Vec<f64> = (0..n_classes)
                .map(|c| y_true.iter().filter(|&&t| t == c as i32).count() as f64)
                .collect();
            for i in 0..n_classes {
                for j in (i + 1)..n_classes {
                    let idxs: Vec<usize> = (0..n)
                        .filter(|&k| y_true[k] == i as i32 || y_true[k] == j as i32)
                        .collect();
                    let y_sub: Vec<i32> = idxs.iter().map(|&k| y_true[k]).collect();
                    let sc_i: Vec<f64> = idxs.iter().map(|&k| y_score[k * n_classes + i]).collect();
                    let sc_j: Vec<f64> = idxs.iter().map(|&k| y_score[k * n_classes + j]).collect();
                    let auc_i_vs_j = roc_auc_score_binary(&y_sub, &sc_i, i as i32, None)?;
                    let auc_j_vs_i = roc_auc_score_binary(&y_sub, &sc_j, j as i32, None)?;
                    pair_aucs.push((auc_i_vs_j + auc_j_vs_i) / 2.0);
                    pair_weights.push(prevalence[i] + prevalence[j]);
                }
            }
            Ok(match average {
                Average::Weighted => {
                    let total: f64 = pair_weights.iter().sum();
                    pair_aucs
                        .iter()
                        .zip(pair_weights.iter())
                        .map(|(&a, &w)| a * w)
                        .sum::<f64>()
                        / total
                }
                _ => pair_aucs.iter().sum::<f64>() / pair_aucs.len() as f64,
            })
        }
    }
}

// ==================== TASK-11 — METR-CLS-09: precision_recall_curve ====================

/// Threshold sweep (reusing [`sweep`]/[`Sweep`]) producing sklearn's
/// `precision_recall_curve` convention: `precision`/`recall` length =
/// `thresholds.len()+1` with a trailing `(1.0, 0.0)` sentinel (the
/// "threshold = +infinity, predict nothing positive" point), `thresholds`
/// strictly ascending (the distinct score values, ascending).
pub fn precision_recall_curve(
    y_true: &[i32],
    probas_pred: &[f64],
    pos_label: i32,
    sample_weight: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(
        y_true.len(),
        probas_pred.len(),
        "precision_recall_curve: length mismatch"
    );
    let sw = sweep(y_true, probas_pred, pos_label, sample_weight);
    let k = sw.scores_desc.len();
    let mut thresholds = Vec::with_capacity(k);
    let mut precision = Vec::with_capacity(k + 1);
    let mut recall = Vec::with_capacity(k + 1);
    for i in (0..k).rev() {
        thresholds.push(sw.scores_desc[i]);
        let tp = sw.cum_tp[i];
        let fp = sw.cum_fp[i];
        precision.push(if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 });
        recall.push(if sw.total_pos > 0.0 {
            tp / sw.total_pos
        } else {
            0.0
        });
    }
    precision.push(1.0);
    recall.push(0.0);
    (precision, recall, thresholds)
}
