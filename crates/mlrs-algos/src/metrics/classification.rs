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
    class_bookkeeping, validate_weight, Average, ClassIndex, MetricError, MultiClass, Normalize,
    PrfOut, PrfResult, ZeroDivision,
};

// ==================== TASK-03 — METR-CLS-01: accuracy_score ====================

/// The fraction of exact matches between `y_true` and `y_pred` (weighted, or
/// weighted count if `normalize=false`). `sample_weight=None` uses unit
/// weights. Empty input yields `0.0/0.0 = NaN` (IEEE-754), matching
/// `nb_common::accuracy_score`'s documented empty-input contract without a
/// special-cased branch.
///
/// Returns `Err(MetricError::LengthMismatch)` if `y_true`/`y_pred`/
/// `sample_weight` lengths disagree, and `Err(MetricError::InvalidWeight)` on
/// a negative/NaN weight entry — no panic (code-review fix: a too-short
/// `sample_weight` previously indexed out of bounds and panicked, and a
/// too-long one was silently truncated with no error).
///
/// NOTE: `nb_common::accuracy_score(pred, y_true)` (existing, opposite arg
/// order) is now a thin delegate to this function (TASK-03, SPEC §5
/// CLS-01) — ONE source of truth.
pub fn accuracy_score(
    y_true: &[i32],
    y_pred: &[i32],
    sample_weight: Option<&[f64]>,
    normalize: bool,
) -> Result<f64, MetricError> {
    if y_true.len() != y_pred.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_true.len(), sample_weight)?;

    let mut correct = 0.0f64;
    let mut total = 0.0f64;
    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        total += w;
        if y_true[i] == y_pred[i] {
            correct += w;
        }
    }
    Ok(if normalize { correct / total } else { correct })
}

// ==================== TASK-04 — METR-CLS-02: confusion_matrix ====================

/// The `C×C` (weighted) confusion matrix: `matrix[i][j]` is the weighted
/// count of samples with true label `classes[i]` and predicted label
/// `classes[j]`, in the resolved class order (sorted unique of `y_true ∪
/// y_pred` when `labels=None`, else `labels` verbatim — including a class
/// absent from the data, which gets a full zero row/column).
///
/// `normalize` (METR-PARAM-01) divides the finished counts by the row sum
/// ([`Normalize::True_`]), column sum ([`Normalize::Pred`]) or grand total
/// ([`Normalize::All`]). A ZERO divisor yields `0.0`, not NaN — sklearn
/// divides under `np.errstate(all="ignore")` and then `np.nan_to_num`s the
/// result, so a class with no true (or no predicted) samples contributes an
/// all-zero row (or column) rather than an all-NaN one.
///
/// Returns `Err(MetricError::LengthMismatch)`/`Err(MetricError::InvalidWeight)`
/// on a bad `sample_weight` — no panic (code-review fix, same class of bug
/// as `accuracy_score`).
pub fn confusion_matrix(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    sample_weight: Option<&[f64]>,
    normalize: Option<Normalize>,
) -> Result<Vec<Vec<f64>>, MetricError> {
    if y_true.len() != y_pred.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_true.len(), sample_weight)?;

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
    // O(1) per lookup, so the tabulation is O(n) rather than the O(n·K) a
    // `classes.iter().position(...)` scan per sample would cost — see
    // [`ClassIndex`] for the measurement that motivated it (METR-PARAM-02).
    let index = ClassIndex::new(&classes);
    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        if let (Some(ti), Some(pi)) = (index.get(y_true[i]), index.get(y_pred[i])) {
            matrix[ti][pi] += w;
        }
    }

    // `nan_to_num`-equivalent: a zero divisor leaves the (already zero) cells
    // at 0.0 instead of producing NaN.
    let divide = |cell: &mut f64, denom: f64| {
        *cell = if denom == 0.0 { 0.0 } else { *cell / denom };
    };
    match normalize {
        None => {}
        Some(Normalize::True_) => {
            for row in matrix.iter_mut() {
                let denom: f64 = row.iter().sum();
                for cell in row.iter_mut() {
                    divide(cell, denom);
                }
            }
        }
        Some(Normalize::Pred) => {
            for j in 0..n {
                let denom: f64 = (0..n).map(|i| matrix[i][j]).sum();
                for row in matrix.iter_mut() {
                    divide(&mut row[j], denom);
                }
            }
        }
        Some(Normalize::All) => {
            let denom: f64 = matrix.iter().flat_map(|row| row.iter()).sum();
            for row in matrix.iter_mut() {
                for cell in row.iter_mut() {
                    divide(cell, denom);
                }
            }
        }
    }
    Ok(matrix)
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
) -> PrfResult {
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

    // Whether the REPORTED value consulted the `zero_division` policy — the
    // input to sklearn's `zero_division="warn"` UndefinedMetricWarning
    // (METR-PARAM-01). Scoped per `average`: `Binary` only cares about the
    // `pos_label` class, `Micro` only about the summed denominator.
    let any_zero_denominator = denominators.iter().any(|&d| d <= 0.0);

    match average {
        Average::None_ => PrfResult {
            out: PrfOut::PerClass(per_class),
            zero_division_hit: any_zero_denominator,
            classes: classes.to_vec(),
        },
        Average::Binary => {
            // A pos_label absent from BOTH y_true and y_pred (e.g. the f1
            // zero-division degenerate, TASK-07) is not in `classes` at
            // all — its (tp, fp, fn) are all implicitly zero, so this is a
            // zero-division case (matches sklearn's own behavior on this
            // input, empirically confirmed at TASK-02 fixture generation).
            match classes.iter().position(|&c| c == pos_label) {
                Some(idx) => PrfResult {
                    out: PrfOut::Scalar(per_class[idx]),
                    zero_division_hit: denominators[idx] <= 0.0,
                    classes: classes.to_vec(),
                },
                None => PrfResult {
                    out: PrfOut::Scalar(zd(zero_division)),
                    zero_division_hit: true,
                    classes: classes.to_vec(),
                },
            }
        }
        Average::Macro => {
            let sum: f64 = per_class.iter().sum();
            PrfResult {
                out: PrfOut::Scalar(sum / per_class.len() as f64),
                zero_division_hit: any_zero_denominator,
                classes: classes.to_vec(),
            }
        }
        Average::Micro => {
            let num_sum: f64 = numerators.iter().sum();
            let den_sum: f64 = denominators.iter().sum();
            PrfResult {
                out: PrfOut::Scalar(if den_sum > 0.0 {
                    num_sum / den_sum
                } else {
                    zd(zero_division)
                }),
                zero_division_hit: den_sum <= 0.0,
                classes: classes.to_vec(),
            }
        }
        Average::Weighted => {
            let support_sum: f64 = supports.iter().sum();
            if support_sum <= 0.0 {
                return PrfResult {
                    out: PrfOut::Scalar(zd(zero_division)),
                    zero_division_hit: true,
                    classes: classes.to_vec(),
                };
            }
            let weighted: f64 = per_class
                .iter()
                .zip(supports.iter())
                .map(|(&r, &s)| r * s)
                .sum();
            PrfResult {
                out: PrfOut::Scalar(weighted / support_sum),
                zero_division_hit: any_zero_denominator,
                classes: classes.to_vec(),
            }
        }
    }
}

/// `precision = tp / (tp + fp)` per class, dispatched over `average` (SPEC
/// §5 CLS-03).
///
/// Propagates [`class_bookkeeping`]'s `Result` (code-review fix: this
/// previously `.expect()`-ed the Result, turning a documented graceful
/// length-mismatch/invalid-weight `Err` into a Rust panic).
pub fn precision_score(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: Average,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<PrfResult, MetricError> {
    let bk = class_bookkeeping(y_true, y_pred, sample_weight, labels)?;
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
    Ok(average_ratio(
        &bk.classes,
        &bk.tp,
        &denom,
        &support,
        pos_label,
        average,
        zero_division,
    ))
}

/// `recall = tp / (tp + fn)` per class, dispatched over `average` (SPEC §5
/// CLS-04).
///
/// Propagates [`class_bookkeeping`]'s `Result` (code-review fix, same class
/// of bug as `precision_score`).
pub fn recall_score(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: Average,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<PrfResult, MetricError> {
    let bk = class_bookkeeping(y_true, y_pred, sample_weight, labels)?;
    let denom: Vec<f64> = bk
        .tp
        .iter()
        .zip(bk.fnn.iter())
        .map(|(&tp, &fnv)| tp + fnv)
        .collect();
    let support = denom.clone();
    Ok(average_ratio(
        &bk.classes,
        &bk.tp,
        &denom,
        &support,
        pos_label,
        average,
        zero_division,
    ))
}

/// `f1 = 2*tp / (2*tp + fp + fn)` per class, computed DIRECTLY from the
/// shared weighted TP/FP/FN (harmonic mean) — NOT from
/// `precision_score(...) × recall_score(...)` floats, to avoid
/// double-rounding (SPEC §5 CLS-05 note, TASK-07).
///
/// Propagates [`class_bookkeeping`]'s `Result` (code-review fix, same class
/// of bug as `precision_score`).
pub fn f1_score(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: Average,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<PrfResult, MetricError> {
    let bk = class_bookkeeping(y_true, y_pred, sample_weight, labels)?;
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
    Ok(average_ratio(
        &bk.classes,
        &numer,
        &denom,
        &support,
        pos_label,
        average,
        zero_division,
    ))
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
///
/// Returns `Err(MetricError::BadShape)` if `y_prob`'s length isn't
/// `y_true.len() * n_classes`, and `Err(MetricError::LengthMismatch)`/
/// `Err(MetricError::InvalidWeight)` on a bad `sample_weight` — no panic
/// (code-review fix: a too-short `sample_weight` previously indexed out of
/// bounds and panicked).
pub fn log_loss(
    y_true: &[i32],
    y_prob: &[f64],
    n_classes: usize,
    labels: Option<&[i32]>,
    sample_weight: Option<&[f64]>,
    eps: f64,
    normalize: bool,
) -> Result<f64, MetricError> {
    if y_prob.len() != y_true.len() * n_classes {
        return Err(MetricError::BadShape);
    }
    validate_weight(y_true.len(), sample_weight)?;

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
    // A y_true label absent from the resolved class set only happens when the
    // caller passed an explicit `labels` omitting a class present in y_true
    // (with `labels = None` the set is DERIVED from y_true, so every label is
    // present). sklearn raises `ValueError("y_true contains values ... not
    // belonging to the passed labels ...")`; return a typed error rather than
    // panicking (code-review fix — a panic across the PyO3 boundary aborts the
    // interpreter, whereas sklearn's ValueError is catchable).
    //
    // O(1) per lookup (METR-PARAM-02) — the scan it replaces cost O(K) on top
    // of the O(K) probability row this loop already walks past.
    let index = ClassIndex::new(&classes);

    let mut sum = 0.0f64;
    let mut weight_total = 0.0f64;
    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        let col = index.get(y_true[i]).ok_or(MetricError::LabelNotInLabels)?;
        let p = y_prob[i * n_classes + col].clamp(eps, 1.0 - eps);
        sum += -w * p.ln();
        weight_total += w;
    }
    Ok(if normalize { sum / weight_total } else { sum })
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
    // `total_cmp` (a total order over ALL f64 incl. NaN) rather than
    // `partial_cmp(...).expect(...)`: the public callers
    // (`roc_auc_score_binary` / `precision_recall_curve`) reject NaN scores up
    // front with `MetricError::NaNScore` (sklearn's own "Input contains NaN."
    // ValueError), so no NaN reaches here; `total_cmp` makes the sort
    // panic-proof regardless (code-review fix — the old `.expect` panicked the
    // interpreter across the PyO3 boundary on any NaN that slipped through).
    // For the finite, non-NaN scores that DO reach here, `total_cmp` orders
    // identically to `partial_cmp`, so the sweep result is unchanged.
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));

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

/// McClish-corrected PARTIAL AUC over `fpr ∈ [0, max_fpr]` (METR-PARAM-01,
/// sklearn's `max_fpr`).
///
/// Integrates the same `(0,0)`-anchored ROC polyline as [`auc_from_sweep`] up
/// to `max_fpr`, linearly interpolating the final point, then standardizes:
/// `0.5 * (1 + (partial - min_area) / (max_area - min_area))` with `min_area =
/// max_fpr²/2` (the chance diagonal) and `max_area = max_fpr` (a perfect
/// classifier) — so a non-discriminant score still reads `0.5` and a perfect
/// one `1.0`.
///
/// sklearn computes this from `roc_curve(...)`, whose default
/// `drop_intermediate=True` removes points whose SECOND differences in both
/// `fps` and `tps` are zero. Those points are exactly the collinear interior
/// ones, so the polyline — and hence both the integral and the interpolated
/// endpoint — is unchanged by working from the full sweep here.
fn partial_auc_from_sweep(sw: &Sweep, max_fpr: f64) -> f64 {
    // The ROC polyline, `(0,0)` first (sklearn's `roc_curve` prepends the same
    // origin point).
    let mut fpr = Vec::with_capacity(sw.scores_desc.len() + 1);
    let mut tpr = Vec::with_capacity(sw.scores_desc.len() + 1);
    fpr.push(0.0);
    tpr.push(0.0);
    for i in 0..sw.scores_desc.len() {
        fpr.push(sw.cum_fp[i] / sw.total_neg);
        tpr.push(sw.cum_tp[i] / sw.total_pos);
    }

    // `np.searchsorted(fpr, max_fpr, "right")`: the first index whose fpr is
    // strictly greater than `max_fpr`. `max_fpr < 1 = fpr.last()` (the
    // `max_fpr == 1` case is short-circuited by the caller), so this index
    // always exists and is ≥ 1.
    let stop = fpr
        .iter()
        .position(|&f| f > max_fpr)
        .unwrap_or(fpr.len() - 1);
    let (x0, x1) = (fpr[stop - 1], fpr[stop]);
    let (y0, y1) = (tpr[stop - 1], tpr[stop]);
    // x0 <= max_fpr < x1 by construction, so the span is strictly positive.
    let y_at_max = y0 + (y1 - y0) * (max_fpr - x0) / (x1 - x0);

    let mut partial = 0.0f64;
    for i in 1..stop {
        partial += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0;
    }
    partial += (max_fpr - x0) * (y_at_max + y0) / 2.0;

    let min_area = 0.5 * max_fpr * max_fpr;
    let max_area = max_fpr;
    0.5 * (1.0 + (partial - min_area) / (max_area - min_area))
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
/// `max_fpr` (METR-PARAM-01) restricts the integral to `fpr ∈ [0, max_fpr]`
/// and applies the McClish standardization — see [`partial_auc_from_sweep`].
/// `None` and `Some(1.0)` both mean the full AUC (sklearn short-circuits
/// `max_fpr == 1` before its own range check); anything outside `(0, 1]`
/// returns `Err(MetricError::InvalidMaxFpr)`.
pub fn roc_auc_score_binary(
    y_true: &[i32],
    y_score: &[f64],
    pos_label: i32,
    sample_weight: Option<&[f64]>,
    max_fpr: Option<f64>,
) -> Result<f64, MetricError> {
    if let Some(m) = max_fpr {
        if !(m > 0.0 && m <= 1.0) {
            return Err(MetricError::InvalidMaxFpr);
        }
    }
    if y_true.len() != y_score.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_true.len(), sample_weight)?;
    // sklearn raises `ValueError("Input contains NaN.")` on a NaN score;
    // reject up front with a typed error rather than reaching the sort (which
    // used to panic on NaN) — code-review fix. Covers the multiclass OvR/OvO
    // paths too, since they funnel their per-class scores through here.
    if y_score.iter().any(|v| v.is_nan()) {
        return Err(MetricError::NaNScore);
    }

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
    Ok(match max_fpr {
        None => auc_from_sweep(&sw),
        // sklearn returns the FULL auc for `max_fpr == 1` rather than routing
        // it through the (mathematically equal, but differently rounded)
        // McClish formula.
        Some(m) if m == 1.0 => auc_from_sweep(&sw),
        Some(m) => partial_auc_from_sweep(&sw, m),
    })
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
/// `classes` is the RESOLVED class order (sklearn's `labels`, or the sorted
/// unique of `y_true` when `labels=None`) — column `c` of the row-major
/// `y_score` belongs to `classes[c]`, and `y_true` is encoded against it, so
/// arbitrary integer labels work (the previous version hard-coded
/// `y_true ∈ {0..n_classes-1}`). A `y_true` value missing from `classes`
/// returns `Err(MetricError::LabelNotInLabels)`.
///
/// `average` (METR-PARAM-01) now covers sklearn's full multiclass set:
/// `Macro`, `Weighted` and `None_` (per-class vector) for OvR, plus `Micro`
/// for OvR only — OvO accepts `Macro`/`Weighted` alone
/// (`Err(MetricError::UnsupportedAverage)` otherwise, mirroring sklearn's
/// `average must be one of ...` / `average=None is not implemented for
/// multi_class='ovo'`).
///
/// `Micro` binarizes `y_true` into the `n_samples × n_classes` indicator
/// matrix and runs ONE binary sweep over the raveled `(indicator, score)`
/// pairs with each sample weight repeated `n_classes` times — sklearn's
/// `_average_binary_score` micro path, exactly.
pub fn roc_auc_score_multiclass(
    y_true: &[i32],
    y_score: &[f64],
    classes: &[i32],
    multi_class: MultiClass,
    average: Average,
    sample_weight: Option<&[f64]>,
) -> Result<PrfOut, MetricError> {
    let n_classes = classes.len();
    let n = y_true.len();
    if n * n_classes != y_score.len() {
        return Err(MetricError::BadShape);
    }
    validate_weight(n, sample_weight)?;
    if average == Average::Binary {
        return Err(MetricError::UnsupportedAverage);
    }

    // Encode y_true against the resolved class order once, O(1) per sample
    // ([`ClassIndex`], METR-PARAM-02) rather than a K-long scan each.
    let index = ClassIndex::new(classes);
    let encoded: Vec<usize> = y_true
        .iter()
        .map(|&t| index.get(t).ok_or(MetricError::LabelNotInLabels))
        .collect::<Result<_, _>>()?;

    match multi_class {
        MultiClass::Ovr => {
            if average == Average::Micro {
                // One binary problem over the raveled indicator matrix.
                let mut y_bin = Vec::with_capacity(n * n_classes);
                for &e in encoded.iter() {
                    for c in 0..n_classes {
                        y_bin.push(if e == c { 1 } else { 0 });
                    }
                }
                let sw_rep: Option<Vec<f64>> = sample_weight.map(|sw| {
                    sw.iter()
                        .flat_map(|&w| std::iter::repeat_n(w, n_classes))
                        .collect()
                });
                let auc = roc_auc_score_binary(&y_bin, y_score, 1, sw_rep.as_deref(), None)?;
                return Ok(PrfOut::Scalar(auc));
            }

            let mut per_class_auc = Vec::with_capacity(n_classes);
            let mut prevalence = Vec::with_capacity(n_classes);
            let mut scores_c = vec![0.0f64; n];
            let mut y_bin = vec![0i32; n];
            for c in 0..n_classes {
                for i in 0..n {
                    y_bin[i] = i32::from(encoded[i] == c);
                    scores_c[i] = y_score[i * n_classes + c];
                }
                per_class_auc.push(roc_auc_score_binary(
                    &y_bin,
                    &scores_c,
                    1,
                    sample_weight,
                    None,
                )?);
                prevalence.push(
                    (0..n)
                        .filter(|&i| encoded[i] == c)
                        .map(|i| sample_weight.map_or(1.0, |sw| sw[i]))
                        .sum::<f64>(),
                );
            }
            match average {
                Average::None_ => Ok(PrfOut::PerClass(per_class_auc)),
                Average::Weighted => {
                    let total: f64 = prevalence.iter().sum();
                    // sklearn returns a bare 0 when the weights sum to zero
                    // (every class empty under the weights) rather than
                    // dividing by zero.
                    if total == 0.0 {
                        return Ok(PrfOut::Scalar(0.0));
                    }
                    // "Scores with 0 weights are forced to be 0" — sklearn's
                    // guard against a NaN from an empty class polluting the
                    // average.
                    let weighted: f64 = per_class_auc
                        .iter()
                        .zip(prevalence.iter())
                        .map(|(&a, &p)| if p == 0.0 { 0.0 } else { a * p })
                        .sum();
                    Ok(PrfOut::Scalar(weighted / total))
                }
                _ => Ok(PrfOut::Scalar(
                    per_class_auc.iter().sum::<f64>() / n_classes as f64,
                )),
            }
        }
        MultiClass::Ovo => {
            if sample_weight.is_some() {
                return Err(MetricError::WeightedOvoUnsupported);
            }
            if matches!(average, Average::Micro | Average::None_) {
                return Err(MetricError::UnsupportedAverage);
            }
            let mut pair_aucs = Vec::new();
            let mut pair_weights = Vec::new();
            let prevalence: Vec<f64> = (0..n_classes)
                .map(|c| encoded.iter().filter(|&&e| e == c).count() as f64)
                .collect();
            for i in 0..n_classes {
                for j in (i + 1)..n_classes {
                    let idxs: Vec<usize> = (0..n)
                        .filter(|&k| encoded[k] == i || encoded[k] == j)
                        .collect();
                    let y_sub: Vec<i32> = idxs.iter().map(|&k| encoded[k] as i32).collect();
                    let sc_i: Vec<f64> = idxs.iter().map(|&k| y_score[k * n_classes + i]).collect();
                    let sc_j: Vec<f64> = idxs.iter().map(|&k| y_score[k * n_classes + j]).collect();
                    let auc_i_vs_j = roc_auc_score_binary(&y_sub, &sc_i, i as i32, None, None)?;
                    let auc_j_vs_i = roc_auc_score_binary(&y_sub, &sc_j, j as i32, None, None)?;
                    pair_aucs.push((auc_i_vs_j + auc_j_vs_i) / 2.0);
                    pair_weights.push(prevalence[i] + prevalence[j]);
                }
            }
            Ok(PrfOut::Scalar(match average {
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
            }))
        }
    }
}

// ==================== TASK-11 — METR-CLS-09: precision_recall_curve ====================

/// Threshold sweep (reusing [`sweep`]/[`Sweep`]) producing sklearn's
/// `precision_recall_curve` convention: `precision`/`recall` length =
/// `thresholds.len()+1` with a trailing `(1.0, 0.0)` sentinel (the
/// "threshold = +infinity, predict nothing positive" point), `thresholds`
/// strictly ascending (the distinct score values, ascending).
///
/// `drop_intermediate` (METR-PARAM-01, sklearn ≥1.3) drops every threshold
/// whose true-positive count is unchanged from BOTH its neighbours — the
/// interior points of a vertical run on the PR plot, which carry no extra
/// recall. The first and last thresholds are always kept, and the sweep is
/// left untouched when it has 2 or fewer points (sklearn's `fps.shape[0] > 2`
/// guard).
///
/// When `y_true` contains NO positive sample the recall column is all `1.0`
/// (sklearn's "No positive class found in y_true, recall is set to one for all
/// thresholds" branch, which the Python shim accompanies with the matching
/// warning) rather than the all-`0.0` an earlier version produced.
///
/// Returns `Err(MetricError::LengthMismatch)`/`Err(MetricError::InvalidWeight)`
/// on a bad `sample_weight` — no panic (code-review fix: unlike
/// `roc_auc_score_binary`/`_multiclass`, this function called [`sweep`]
/// without first validating `sample_weight`, so a too-short weight vector
/// indexed out of bounds).
pub fn precision_recall_curve(
    y_true: &[i32],
    probas_pred: &[f64],
    pos_label: i32,
    sample_weight: Option<&[f64]>,
    drop_intermediate: bool,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), MetricError> {
    if y_true.len() != probas_pred.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_true.len(), sample_weight)?;
    // sklearn raises `ValueError("Input contains NaN.")` on a NaN score;
    // reject up front rather than reaching the sort (which used to panic on
    // NaN) — code-review fix, same as `roc_auc_score_binary`.
    if probas_pred.iter().any(|v| v.is_nan()) {
        return Err(MetricError::NaNScore);
    }

    let sw = sweep(y_true, probas_pred, pos_label, sample_weight);
    let m = sw.scores_desc.len();

    // Indices INTO the sweep (descending threshold), after the optional drop.
    let kept: Vec<usize> = if drop_intermediate && m > 2 {
        (0..m)
            .filter(|&k| {
                k == 0
                    || k == m - 1
                    || sw.cum_tp[k] != sw.cum_tp[k - 1]
                    || sw.cum_tp[k + 1] != sw.cum_tp[k]
            })
            .collect()
    } else {
        (0..m).collect()
    };

    let mut thresholds = Vec::with_capacity(kept.len());
    let mut precision = Vec::with_capacity(kept.len() + 1);
    let mut recall = Vec::with_capacity(kept.len() + 1);
    for &i in kept.iter().rev() {
        thresholds.push(sw.scores_desc[i]);
        let tp = sw.cum_tp[i];
        let fp = sw.cum_fp[i];
        // sklearn 1.9.0: `precision = where(ps != 0, tps / ps, 0.0)` — the
        // zero-denominator cell is 0.0, not the 1.0 an earlier version used.
        precision.push(if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 });
        recall.push(if sw.total_pos > 0.0 {
            tp / sw.total_pos
        } else {
            1.0
        });
    }
    precision.push(1.0);
    recall.push(0.0);
    Ok((precision, recall, thresholds))
}
