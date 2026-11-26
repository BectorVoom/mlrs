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

use super::prfs::{prfs_multiclass, SetAverage};
use super::{
    validate_weight, Average, ClassIndex, MetricError, MultiClass, Normalize, PrfOut, PrfResult,
    ZeroDivision,
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
//
// All three are views of ONE `precision_recall_fscore_support` computation
// ([`super::prfs`]), so they share sklearn's exact divide-then-`_nanaverage`
// rules — including the one the original per-metric `average_ratio` got wrong:
// with `zero_division=np.nan`, a macro/weighted average SKIPS the undefined
// classes rather than returning NaN.

/// Map the original [`Average`] onto the set-wise averaging modes.
fn set_average(average: Average) -> SetAverage {
    match average {
        Average::Binary => SetAverage::Binary,
        Average::Macro => SetAverage::Macro,
        Average::Micro => SetAverage::Micro,
        Average::Weighted => SetAverage::Weighted,
        Average::None_ => SetAverage::None_,
    }
}

/// Which of the three ratios a scalar metric reports.
#[derive(Clone, Copy)]
enum Ratio {
    Precision,
    Recall,
    FScore,
}

#[allow(clippy::too_many_arguments)]
fn prf_view(
    ratio: Ratio,
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: Average,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<PrfResult, MetricError> {
    let r = prfs_multiclass(
        y_true,
        y_pred,
        1.0,
        labels,
        pos_label,
        set_average(average),
        sample_weight,
        zero_division,
    )?;
    let (out, zero_division_hit) = match ratio {
        Ratio::Precision => (r.precision, r.precision_zd),
        Ratio::Recall => (r.recall, r.recall_zd),
        Ratio::FScore => (r.fscore, r.fscore_zd),
    };
    Ok(PrfResult {
        out,
        zero_division_hit,
        classes: r.classes,
    })
}

/// `precision = tp / (tp + fp)` per class, dispatched over `average` (SPEC
/// §5 CLS-03). A view of [`super::prfs::prfs_multiclass`].
pub fn precision_score(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: Average,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<PrfResult, MetricError> {
    prf_view(
        Ratio::Precision,
        y_true,
        y_pred,
        labels,
        pos_label,
        average,
        sample_weight,
        zero_division,
    )
}

/// `recall = tp / (tp + fn)` per class, dispatched over `average` (SPEC §5
/// CLS-04). A view of [`super::prfs::prfs_multiclass`].
pub fn recall_score(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: Average,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<PrfResult, MetricError> {
    prf_view(
        Ratio::Recall,
        y_true,
        y_pred,
        labels,
        pos_label,
        average,
        sample_weight,
        zero_division,
    )
}

/// `f1 = 2*tp / (2*tp + fp + fn)` per class, computed DIRECTLY from the
/// weighted counts (never from the two rounded ratios), dispatched over
/// `average` (SPEC §5 CLS-05). A view of [`super::prfs::prfs_multiclass`] at
/// `beta = 1`.
pub fn f1_score(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    pos_label: i32,
    average: Average,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<PrfResult, MetricError> {
    prf_view(
        Ratio::FScore,
        y_true,
        y_pred,
        labels,
        pos_label,
        average,
        sample_weight,
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

    // sklearn drops zero-weight samples BEFORE sorting, so a score carried
    // only by zero-weight samples is not a threshold. Filter only when some
    // weight is actually zero, so the common path borrows the inputs as-is.
    let filtered: Option<(Vec<i32>, Vec<f64>, Vec<f64>)> = sample_weight
        .filter(|w| w.contains(&0.0))
        .map(|w| {
            let keep: Vec<usize> = (0..w.len()).filter(|&i| w[i] != 0.0).collect();
            (
                keep.iter().map(|&i| y_true[i]).collect(),
                keep.iter().map(|&i| probas_pred[i]).collect(),
                keep.iter().map(|&i| w[i]).collect(),
            )
        });
    let sw = match &filtered {
        Some((t, p, w)) => sweep(t, p, pos_label, Some(w)),
        None => sweep(y_true, probas_pred, pos_label, sample_weight),
    };
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

// ==================== The rest of sklearn's classification surface ====================
//
// Everything below reads either the weighted confusion matrix
// ([`confusion_matrix`], one O(n) pass) or the per-row probability/decision
// matrix, and then does O(K)/O(K²) bookkeeping — the same shape as the metrics
// above. Label ENCODING (which column is which class) is done here, through
// [`ClassIndex`], never in the shim.

/// Weighted mean of a per-sample quantity (`np.average(values, weights)`), or
/// `Err(ZeroWeightSum)` when the weights sum to zero (numpy raises
/// `ZeroDivisionError` there).
fn weighted_mean(values: impl Iterator<Item = f64>, sample_weight: Option<&[f64]>) -> Result<f64, MetricError> {
    let mut sum = 0.0f64;
    let mut total = 0.0f64;
    for (i, v) in values.enumerate() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        sum += w * v;
        total += w;
    }
    if total == 0.0 {
        return Err(MetricError::ZeroWeightSum);
    }
    Ok(sum / total)
}

/// `balanced_accuracy_score`: the mean per-class recall over the classes
/// present in `y_true`. Returns `(score, dropped)` where `dropped` is `true`
/// when some class appears only in `y_pred` — sklearn drops that class's NaN
/// recall and warns "y_pred contains classes not in y_true".
///
/// `adjusted = true` rescales so chance is 0 and perfect is 1:
/// `(score - 1/K) / (1 - 1/K)` over the `K` kept classes.
pub fn balanced_accuracy_score(
    y_true: &[i32],
    y_pred: &[i32],
    sample_weight: Option<&[f64]>,
    adjusted: bool,
) -> Result<(f64, bool), MetricError> {
    let c = confusion_matrix(y_true, y_pred, None, sample_weight, None)?;
    let mut per_class = Vec::with_capacity(c.len());
    let mut dropped = false;
    for (i, row) in c.iter().enumerate() {
        let support: f64 = row.iter().sum();
        let r = row[i] / support;
        if r.is_nan() {
            dropped = true;
        } else {
            per_class.push(r);
        }
    }
    let mut score = per_class.iter().sum::<f64>() / per_class.len() as f64;
    if adjusted {
        let chance = 1.0 / per_class.len() as f64;
        score = (score - chance) / (1.0 - chance);
    }
    Ok((score, dropped))
}

/// `matthews_corrcoef` (the multiclass `R_K` statistic of Gorodkin 2004),
/// from the weighted confusion matrix over the labels present in either
/// vector. A zero covariance product returns `0.0`, as sklearn does.
pub fn matthews_corrcoef(
    y_true: &[i32],
    y_pred: &[i32],
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    let c = confusion_matrix(y_true, y_pred, None, sample_weight, None)?;
    let k = c.len();
    let t_sum: Vec<f64> = c.iter().map(|row| row.iter().sum()).collect();
    let p_sum: Vec<f64> = (0..k).map(|j| (0..k).map(|i| c[i][j]).sum()).collect();
    let n_correct: f64 = (0..k).map(|i| c[i][i]).sum();
    let n: f64 = p_sum.iter().sum();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let cov_ytyp = n_correct * n - dot(&t_sum, &p_sum);
    let cov_ypyp = n * n - dot(&p_sum, &p_sum);
    let cov_ytyt = n * n - dot(&t_sum, &t_sum);
    let prod = cov_ypyp * cov_ytyt;
    Ok(if prod == 0.0 { 0.0 } else { cov_ytyp / prod.sqrt() })
}

/// `cohen_kappa_score`'s disagreement weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KappaWeights {
    /// Unweighted: every off-diagonal cell counts 1.
    None_,
    /// `|i - j|`.
    Linear,
    /// `(i - j)²`.
    Quadratic,
}

/// The outcome of [`cohen_kappa_score`]: a value, or one of sklearn's two
/// undefined cases (the shim substitutes `replace_undefined_by` and emits the
/// matching `UndefinedMetricWarning`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KappaOut {
    Value(f64),
    /// The confusion matrix is empty (no sample has both labels in `labels`).
    NoSharedLabels,
    /// The expected-disagreement denominator is zero (one label in common).
    OneSharedLabel,
}

/// `cohen_kappa_score`: `1 - Σ w·C / Σ w·E`, where `E` is the outer product of
/// the confusion matrix's marginals over its total.
pub fn cohen_kappa_score(
    y1: &[i32],
    y2: &[i32],
    labels: Option<&[i32]>,
    weights: KappaWeights,
    sample_weight: Option<&[f64]>,
) -> Result<KappaOut, MetricError> {
    let c = confusion_matrix(y1, y2, labels, sample_weight, None)?;
    let k = c.len();
    let sum0: Vec<f64> = (0..k).map(|j| (0..k).map(|i| c[i][j]).sum()).collect();
    let sum1: Vec<f64> = c.iter().map(|row| row.iter().sum()).collect();
    let total: f64 = sum0.iter().sum();
    if total == 0.0 {
        return Ok(KappaOut::NoSharedLabels);
    }
    let w = |i: usize, j: usize| -> f64 {
        let d = i as f64 - j as f64;
        match weights {
            KappaWeights::None_ => f64::from(i != j),
            KappaWeights::Linear => d.abs(),
            KappaWeights::Quadratic => d * d,
        }
    };
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..k {
        for j in 0..k {
            num += w(i, j) * c[i][j];
            den += w(i, j) * sum0[i] * sum1[j] / total;
        }
    }
    if den == 0.0 {
        return Ok(KappaOut::OneSharedLabel);
    }
    Ok(KappaOut::Value(1.0 - num / den))
}

/// `hamming_loss` for a 1-D target: the weighted fraction of mismatches.
pub fn hamming_loss(
    y_true: &[i32],
    y_pred: &[i32],
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    if y_true.len() != y_pred.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_true.len(), sample_weight)?;
    weighted_mean(
        y_true.iter().zip(y_pred).map(|(t, p)| f64::from(t != p)),
        sample_weight,
    )
}

/// Validate a row-major `n × n_labels` indicator pair and return `n`.
fn indicator_rows(
    y_true: &[i32],
    y_pred: &[i32],
    n_labels: usize,
    sample_weight: Option<&[f64]>,
) -> Result<usize, MetricError> {
    if y_true.len() != y_pred.len() {
        return Err(MetricError::LengthMismatch);
    }
    if n_labels == 0 || y_true.len() % n_labels != 0 {
        return Err(MetricError::BadShape);
    }
    let n = y_true.len() / n_labels;
    validate_weight(n, sample_weight)?;
    Ok(n)
}

/// `hamming_loss` for a multilabel indicator target: weighted count of
/// differing cells over `n_samples * n_labels * mean(weight)`.
pub fn hamming_loss_indicator(
    y_true: &[i32],
    y_pred: &[i32],
    n_labels: usize,
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    let n = indicator_rows(y_true, y_pred, n_labels, sample_weight)?;
    let mut diff = 0.0f64;
    for i in 0..n {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        let d = (0..n_labels)
            .filter(|&j| y_true[i * n_labels + j] != y_pred[i * n_labels + j])
            .count();
        diff += w * d as f64;
    }
    let weight_average = match sample_weight {
        Some(sw) => sw.iter().sum::<f64>() / n as f64,
        None => 1.0,
    };
    Ok(diff / (n as f64 * n_labels as f64 * weight_average))
}

/// Subset accuracy for a multilabel indicator target (`accuracy_score` on 2-D
/// input): a sample counts as correct only when EVERY label matches.
pub fn accuracy_score_indicator(
    y_true: &[i32],
    y_pred: &[i32],
    n_labels: usize,
    sample_weight: Option<&[f64]>,
    normalize: bool,
) -> Result<f64, MetricError> {
    let n = indicator_rows(y_true, y_pred, n_labels, sample_weight)?;
    let (mut correct, mut total) = (0.0f64, 0.0f64);
    for i in 0..n {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        let row = i * n_labels..(i + 1) * n_labels;
        total += w;
        if y_true[row.clone()] == y_pred[row] {
            correct += w;
        }
    }
    Ok(if normalize { correct / total } else { correct })
}

/// `zero_one_loss`: `1 - accuracy` (normalized) or `total_weight - correct`.
pub fn zero_one_loss(
    y_true: &[i32],
    y_pred: &[i32],
    sample_weight: Option<&[f64]>,
    normalize: bool,
) -> Result<f64, MetricError> {
    let score = accuracy_score(y_true, y_pred, sample_weight, normalize)?;
    Ok(zero_one_from_accuracy(score, y_true.len(), sample_weight, normalize))
}

/// [`zero_one_loss`] for a multilabel indicator target (subset accuracy).
pub fn zero_one_loss_indicator(
    y_true: &[i32],
    y_pred: &[i32],
    n_labels: usize,
    sample_weight: Option<&[f64]>,
    normalize: bool,
) -> Result<f64, MetricError> {
    let score = accuracy_score_indicator(y_true, y_pred, n_labels, sample_weight, normalize)?;
    Ok(zero_one_from_accuracy(score, y_true.len() / n_labels, sample_weight, normalize))
}

fn zero_one_from_accuracy(score: f64, n: usize, sample_weight: Option<&[f64]>, normalize: bool) -> f64 {
    if normalize {
        1.0 - score
    } else {
        let total = sample_weight.map_or(n as f64, |sw| sw.iter().sum());
        total - score
    }
}

/// Binary `hinge_loss`: `mean(max(0, 1 - y·f(x)))` with `y ∈ {-1, +1}`
/// (`y_signed`, which the caller derives with sklearn's
/// `LabelBinarizer(neg_label=-1)` rule: the larger class is `+1`).
pub fn hinge_loss_binary(
    y_signed: &[f64],
    pred_decision: &[f64],
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    if y_signed.len() != pred_decision.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_signed.len(), sample_weight)?;
    weighted_mean(
        y_signed
            .iter()
            .zip(pred_decision)
            .map(|(&y, &d)| (1.0 - y * d).max(0.0)),
        sample_weight,
    )
}

/// Multiclass `hinge_loss` (Crammer–Singer): the margin is the true class's
/// decision value minus the largest OTHER class's, over the row-major
/// `n × K` `pred_decision`, whose column `c` belongs to `classes[c]`.
pub fn hinge_loss_multiclass(
    y_true: &[i32],
    pred_decision: &[f64],
    classes: &[i32],
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    let k = classes.len();
    let n = y_true.len();
    if k < 2 || pred_decision.len() != n * k {
        return Err(MetricError::BadShape);
    }
    validate_weight(n, sample_weight)?;
    let index = ClassIndex::new(classes);
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let yi = index.get(y_true[i]).ok_or(MetricError::LabelNotInLabels)?;
        let row = &pred_decision[i * k..(i + 1) * k];
        let other = row
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != yi)
            .map(|(_, &v)| v)
            .fold(f64::NEG_INFINITY, f64::max);
        losses.push((1.0 - (row[yi] - other)).max(0.0));
    }
    weighted_mean(losses.into_iter(), sample_weight)
}

/// Encode `y_true` against `classes` into column indices.
fn encode(y_true: &[i32], classes: &[i32]) -> Result<Vec<usize>, MetricError> {
    let index = ClassIndex::new(classes);
    y_true
        .iter()
        .map(|&t| index.get(t).ok_or(MetricError::LabelNotInLabels))
        .collect()
}

/// Per-sample `Σ_k (onehot_k - p_k)²` over a row-major `n × K` probability
/// matrix whose column `c` belongs to `classes[c]`.
fn brier_rows(enc: &[usize], y_prob: &[f64], k: usize, null: Option<&[f64]>) -> Vec<f64> {
    (0..enc.len())
        .map(|i| {
            (0..k)
                .map(|c| {
                    let p = null.map_or(y_prob[i * k + c], |q| q[c]);
                    let t = f64::from(enc[i] == c);
                    (t - p) * (t - p)
                })
                .sum()
        })
        .collect()
}

/// The (weighted) class prevalence vector — the "null model" probabilities the
/// D² scores compare against.
fn prevalence(enc: &[usize], k: usize, sample_weight: Option<&[f64]>) -> Result<Vec<f64>, MetricError> {
    let mut p = vec![0.0f64; k];
    let mut total = 0.0f64;
    for (i, &e) in enc.iter().enumerate() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        p[e] += w;
        total += w;
    }
    if total == 0.0 {
        return Err(MetricError::ZeroWeightSum);
    }
    for v in p.iter_mut() {
        *v /= total;
    }
    Ok(p)
}

fn check_prob_geometry(
    y_true: &[i32],
    y_prob: &[f64],
    classes: &[i32],
    sample_weight: Option<&[f64]>,
) -> Result<Vec<usize>, MetricError> {
    if classes.is_empty() || y_prob.len() != y_true.len() * classes.len() {
        return Err(MetricError::BadShape);
    }
    validate_weight(y_true.len(), sample_weight)?;
    if y_prob.iter().any(|&p| !(0.0..=1.0).contains(&p)) {
        return Err(MetricError::DomainError);
    }
    encode(y_true, classes)
}

/// `brier_score_loss`: the weighted mean over samples of
/// `Σ_k (onehot_k - p_k)²`, halved when `scale_by_half` (sklearn's
/// `scale_by_half='auto'` halves for binary input, which is what makes the
/// binary value the familiar `mean((y - p)²)`).
///
/// A binary call passes the two-column `[1-p, p]` matrix with `classes =
/// [neg, pos]` — the caller resolves `pos_label`.
pub fn brier_score_loss(
    y_true: &[i32],
    y_prob: &[f64],
    classes: &[i32],
    sample_weight: Option<&[f64]>,
    scale_by_half: bool,
) -> Result<f64, MetricError> {
    let enc = check_prob_geometry(y_true, y_prob, classes, sample_weight)?;
    let rows = brier_rows(&enc, y_prob, classes.len(), None);
    let b = weighted_mean(rows.into_iter(), sample_weight)?;
    Ok(if scale_by_half { 0.5 * b } else { b })
}

/// `d2_brier_score`: `1 - brier / brier_null`, the null model predicting the
/// weighted class prevalence for every sample.
pub fn d2_brier_score(
    y_true: &[i32],
    y_prob: &[f64],
    classes: &[i32],
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    let enc = check_prob_geometry(y_true, y_prob, classes, sample_weight)?;
    let k = classes.len();
    let null = prevalence(&enc, k, sample_weight)?;
    let b = weighted_mean(brier_rows(&enc, y_prob, k, None).into_iter(), sample_weight)?;
    let b0 = weighted_mean(brier_rows(&enc, y_prob, k, Some(&null)).into_iter(), sample_weight)?;
    Ok(1.0 - b / b0)
}

/// `d2_log_loss_score`: `1 - log_loss / log_loss_null` (both UNnormalized
/// weighted sums, so the ratio is the same either way), the null model
/// predicting the weighted class prevalence. Probabilities are clipped to
/// `[eps, 1 - eps]` before the log, as `log_loss` does.
pub fn d2_log_loss_score(
    y_true: &[i32],
    y_prob: &[f64],
    classes: &[i32],
    sample_weight: Option<&[f64]>,
    eps: f64,
) -> Result<f64, MetricError> {
    let enc = check_prob_geometry(y_true, y_prob, classes, sample_weight)?;
    let k = classes.len();
    let null = prevalence(&enc, k, sample_weight)?;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, &e) in enc.iter().enumerate() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        num -= w * y_prob[i * k + e].clamp(eps, 1.0 - eps).ln();
        den -= w * null[e].clamp(eps, 1.0 - eps).ln();
    }
    Ok(1.0 - num / den)
}

/// `class_likelihood_ratios`' raw result. `lr_pos`/`lr_neg` are the plain
/// ratios (`NaN`/`inf` where they divide by zero); the flags mark sklearn's
/// three undefined cases, for which the shim substitutes
/// `replace_undefined_by` and warns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LikelihoodRatios {
    pub lr_pos: f64,
    pub lr_neg: f64,
    /// No positive sample in `y_true`.
    pub no_positive: bool,
    /// `fp == 0`: `LR+` is undefined.
    pub pos_undefined: bool,
    /// `tp == 0` (with `fp == 0`): no sample was predicted positive at all.
    pub no_predicted_positive: bool,
    /// `tn == 0`: `LR-` is undefined.
    pub neg_undefined: bool,
}

/// `class_likelihood_ratios` over the binary weighted confusion matrix (in the
/// `labels` order, or the sorted unique — so the SECOND class is positive).
/// `LR+ = sensitivity / (1 - specificity)`, `LR- = (1 - sensitivity) /
/// specificity`.
pub fn class_likelihood_ratios(
    y_true: &[i32],
    y_pred: &[i32],
    labels: Option<&[i32]>,
    sample_weight: Option<&[f64]>,
) -> Result<LikelihoodRatios, MetricError> {
    let c = confusion_matrix(y_true, y_pred, labels, sample_weight, None)?;
    if c.len() != 2 {
        return Err(MetricError::UnsupportedTarget);
    }
    let (tn, fp, fnn, tp) = (c[0][0], c[0][1], c[1][0], c[1][1]);
    let support_pos = tp + fnn;
    let support_neg = tn + fp;
    Ok(LikelihoodRatios {
        lr_pos: (tp * support_neg) / (fp * support_pos),
        lr_neg: (fnn * support_neg) / (tn * support_pos),
        no_positive: support_pos == 0.0,
        pos_undefined: fp == 0.0,
        no_predicted_positive: fp == 0.0 && tp == 0.0,
        neg_undefined: tn == 0.0,
    })
}

/// Multiclass `top_k_accuracy_score`: a sample is a hit when its true class is
/// among the `k` highest-scoring columns of the row-major `n × K` `y_score`
/// (column `c` = `classes[c]`).
///
/// Ties follow sklearn's `argsort(kind='mergesort')[:, ::-1]`: a stable
/// ascending sort reversed, so among EQUAL scores the HIGHER column index ranks
/// first. The rank of class `c` is therefore `#{c': s[c'] > s[c]} + #{c' > c:
/// s[c'] == s[c]}`, an O(K) count per sample with no sort.
pub fn top_k_accuracy_score(
    y_true: &[i32],
    y_score: &[f64],
    classes: &[i32],
    k: usize,
    normalize: bool,
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    let n_classes = classes.len();
    let n = y_true.len();
    if n_classes == 0 || y_score.len() != n * n_classes {
        return Err(MetricError::BadShape);
    }
    if k == 0 {
        return Err(MetricError::InvalidParameter);
    }
    validate_weight(n, sample_weight)?;
    let enc = encode(y_true, classes)?;
    let hits = (0..n).map(|i| {
        let row = &y_score[i * n_classes..(i + 1) * n_classes];
        let c = enc[i];
        let s = row[c];
        let rank = (0..n_classes)
            .filter(|&j| row[j] > s || (row[j] == s && j > c))
            .count();
        f64::from(rank < k)
    });
    hits_reduce(hits, n, normalize, sample_weight)
}

/// Binary `top_k_accuracy_score`. `y_true` is the 0/1 encoding against the
/// sorted classes and `y_score` the positive-class score. With `k == 1` the
/// prediction is `score > threshold` (0.5 for scores inside `[0, 1]`, else 0);
/// with `k >= 2` every sample is a hit, as in sklearn.
pub fn top_k_accuracy_score_binary(
    y_true: &[i32],
    y_score: &[f64],
    k: usize,
    normalize: bool,
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    let n = y_true.len();
    if y_score.len() != n {
        return Err(MetricError::LengthMismatch);
    }
    if k == 0 {
        return Err(MetricError::InvalidParameter);
    }
    validate_weight(n, sample_weight)?;
    let (lo, hi) = y_score
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let threshold = if lo >= 0.0 && hi <= 1.0 { 0.5 } else { 0.0 };
    let hits = (0..n).map(|i| {
        if k == 1 {
            f64::from(i32::from(y_score[i] > threshold) == y_true[i])
        } else {
            1.0
        }
    });
    hits_reduce(hits, n, normalize, sample_weight)
}

fn hits_reduce(
    hits: impl Iterator<Item = f64>,
    n: usize,
    normalize: bool,
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    if normalize {
        return weighted_mean(hits, sample_weight);
    }
    let _ = n;
    Ok(hits
        .enumerate()
        .map(|(i, h)| h * sample_weight.map_or(1.0, |sw| sw[i]))
        .sum())
}
