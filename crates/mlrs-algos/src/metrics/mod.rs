//! `mlrs.metrics` — host-only sklearn-compatible metrics surface (Phase 24
//! METR-01/02/03; SPEC `.planning/plans/metrics-surface/SPEC.md`).
//!
//! This module owns the shared substrate every classification metric
//! (`classification` submodule, TASK-03..11) builds on: the [`Average`] /
//! [`ZeroDivision`] / [`MultiClass`] / [`MetricError`] / [`PrfOut`] types and
//! [`class_bookkeeping`] — sorted unique-class discovery (or an explicit
//! `labels` order, including classes absent from the data) plus per-class
//! weighted TP/FP/FN accumulation. No metric VALUE logic (accuracy,
//! confusion, precision, …) lives here — see [`classification`] /
//! [`regression`].
//!
//! Every metric is a small O(n) host reduction over already-materialized
//! label/target vectors — none needs a device kernel, `BufferPool`, or the
//! f64 capability guard (host-only by design, SPEC §3).
//!
//! `metrics/mod.rs` is edited EXACTLY ONCE in the whole metrics-surface plan
//! (Plan-Check Issue 4, TASK-01): every later classification/regression task
//! only appends functions to its own already-registered `classification.rs`/
//! `regression.rs` file.
//!
//! Tests live in `crates/mlrs-algos/tests/metrics_infra_test.rs` (AGENTS.md
//! §2 — no in-source `#[cfg(test)] mod tests`).

pub mod classification;
pub mod regression;

/// The averaging strategy for `precision_score`/`recall_score`/`f1_score`
/// (and multiclass `roc_auc_score`'s `average` parameter, which only uses
/// `Macro`/`Weighted`). `None_` (sklearn's `average=None`) returns the
/// per-class vector via [`PrfOut::PerClass`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Average {
    Binary,
    Macro,
    Micro,
    Weighted,
    None_,
}

/// The `zero_division` policy applied when a per-class ratio's denominator is
/// zero (e.g. no predicted positives for `precision_score`). sklearn's
/// `"warn"` string maps to `Zero` at the PyO3/shim boundary (Rust never sees
/// the warning string itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroDivision {
    Zero,
    One,
    Nan,
}

/// Multiclass `roc_auc_score` reduction strategy: one-vs-rest or
/// one-vs-one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiClass {
    Ovr,
    Ovo,
}

/// Typed metrics-surface error — mapped to `PyValueError` at the PyO3
/// boundary by a NEW `metric_err_to_py` (a sibling of `algo_err_to_py`, which
/// only accepts [`crate::AlgoError`] — `MetricError` is a distinct type, TASK-15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricError {
    /// Two label/weight vectors that should be the same length are not.
    LengthMismatch,
    /// An input that must be non-empty is empty.
    EmptyInput,
    /// `roc_auc_score` (binary) was called with only one class present in
    /// `y_true`.
    SingleClassRocAuc,
    /// A shape precondition (e.g. `y_prob` not `n_rows * n_classes` long)
    /// was violated.
    BadShape,
    /// A `sample_weight` entry is negative, NaN, or otherwise not a valid
    /// non-negative finite weight.
    InvalidWeight,
    /// `roc_auc_score(multi_class='ovo', sample_weight=Some(_))` was
    /// requested but the pinned sklearn version (probed at TASK-02 fixture
    /// generation) rejects that combination — the Rust OvO branch matches
    /// sklearn's own rejection rather than silently ignoring the weight
    /// (SPEC §2/§4 Q10 carve-out, Plan-Check Issue 2). Unused until TASK-10
    /// wires the OvO branch, but present here so `mod.rs` is touched exactly
    /// once for the whole plan.
    WeightedOvoUnsupported,
}

impl std::fmt::Display for MetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            MetricError::LengthMismatch => "mlrs.metrics: mismatched input lengths",
            MetricError::EmptyInput => "mlrs.metrics: empty input",
            MetricError::SingleClassRocAuc => {
                "mlrs.metrics: roc_auc_score requires at least two classes in y_true"
            }
            MetricError::BadShape => {
                "mlrs.metrics: input shape does not match the declared geometry"
            }
            MetricError::InvalidWeight => {
                "mlrs.metrics: sample_weight must be finite and non-negative"
            }
            MetricError::WeightedOvoUnsupported => {
                "mlrs.metrics: roc_auc_score(multi_class='ovo', sample_weight=...) is not supported"
            }
        };
        f.write_str(msg)
    }
}

impl std::error::Error for MetricError {}

/// The return shape shared by `precision_score`/`recall_score`/`f1_score`:
/// a single scalar for `average ∈ {binary,macro,micro,weighted}`, or the
/// per-class vector (in the resolved class order) for `average=None`.
#[derive(Debug, Clone, PartialEq)]
pub enum PrfOut {
    Scalar(f64),
    PerClass(Vec<f64>),
}

/// The shared label/weight bookkeeping result: the resolved class order
/// (sorted unique of `y_true ∪ y_pred`, or the caller's `labels` verbatim
/// including labels absent from the data) plus each class's weighted
/// `(tp, fp, fn)` triple in that same order.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassBookkeeping {
    /// The resolved class order.
    pub classes: Vec<i32>,
    /// `tp[i]` is the weighted count of samples where `y_true == y_pred ==
    /// classes[i]`.
    pub tp: Vec<f64>,
    /// `fp[i]` is the weighted count of samples where `y_pred == classes[i]`
    /// but `y_true != classes[i]`.
    pub fp: Vec<f64>,
    /// `fnn[i]` is the weighted count of samples where `y_true ==
    /// classes[i]` but `y_pred != classes[i]`. Named `fnn` (not `fn`, a Rust
    /// keyword).
    pub fnn: Vec<f64>,
}

/// Validate `sample_weight` (if given): must be the same length as the
/// labels and every entry finite and `>= 0.0`.
fn validate_weight(len: usize, sample_weight: Option<&[f64]>) -> Result<(), MetricError> {
    if let Some(w) = sample_weight {
        if w.len() != len {
            return Err(MetricError::LengthMismatch);
        }
        if w.iter().any(|&wi| !wi.is_finite() || wi < 0.0) {
            return Err(MetricError::InvalidWeight);
        }
    }
    Ok(())
}

/// Shared label/weight bookkeeping (METR-INFRA-01): given equal-length
/// `y_true`/`y_pred` (+ an optional same-length `sample_weight`), resolves
/// the class order — sorted unique of `y_true ∪ y_pred` when `labels` is
/// `None`, else `labels` verbatim (including a class absent from the data,
/// which gets `(tp,fp,fn) = (0,0,0)`) — and accumulates each class's
/// weighted TP/FP/FN in that order.
///
/// With `sample_weight = None`, weighted counts equal the unweighted integer
/// counts (unit weights). Returns `Err(MetricError::LengthMismatch)` on a
/// `y_true`/`y_pred` (or weight) length mismatch, and
/// `Err(MetricError::InvalidWeight)` on a negative/NaN weight entry — no
/// panic (SPEC §5 behavior clause).
pub fn class_bookkeeping(
    y_true: &[i32],
    y_pred: &[i32],
    sample_weight: Option<&[f64]>,
    labels: Option<&[i32]>,
) -> Result<ClassBookkeeping, MetricError> {
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

    let mut tp = vec![0.0f64; classes.len()];
    let mut fp = vec![0.0f64; classes.len()];
    let mut fnn = vec![0.0f64; classes.len()];

    // class -> index in `classes` (only classes we track contribute).
    let index_of = |c: i32| classes.iter().position(|&x| x == c);

    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        let t = y_true[i];
        let p = y_pred[i];
        if t == p {
            if let Some(idx) = index_of(t) {
                tp[idx] += w;
            }
        } else {
            if let Some(idx) = index_of(p) {
                fp[idx] += w;
            }
            if let Some(idx) = index_of(t) {
                fnn[idx] += w;
            }
        }
    }

    Ok(ClassBookkeeping {
        classes,
        tp,
        fp,
        fnn,
    })
}
