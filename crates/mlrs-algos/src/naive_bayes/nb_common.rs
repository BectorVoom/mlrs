//! `nb_common` — shared Naive Bayes free functions (D-03 — NO struct, NO trait).
//!
//! The five NB estimators (`GaussianNB` / `MultinomialNB` / `BernoulliNB` /
//! `ComplementNB` / `CategoricalNB`) are fully independent structs; the math they
//! share lives HERE as free functions they CALL, not as a base struct or trait
//! object. This is the D-03 "DRY at the function level" decision.
//!
//! ## What's shared
//!
//! - [`log_sum_exp_normalize`] — the per-row log-sum-exp that turns a row of
//!   joint log-likelihoods into `(proba, log_proba)` for
//!   `predict_proba` / `predict_log_proba` (Pattern 3: host f64, per-row
//!   max-shift, a SINGLE terminal log — never `±∞` / `F::INFINITY` mid-pipeline,
//!   so it stays cpu-MLIR-safe, Pitfall 9).
//! - [`empirical_class_log_prior`] — `log(count_c / Σ count)` from `class_count_`
//!   when the user supplies no explicit prior.
//! - [`argmax_decode`] / [`argmin_decode`] — map each row's argmax / argmin joint
//!   log-likelihood through the sorted `classes_` table to the predicted label
//!   (`ComplementNB` uses argmin internally, D-08).
//! - [`accuracy_score`] — the fraction of exact matches, for the shared `score`
//!   (D-07, sklearn `ClassifierMixin.score`).
//! - [`class_grouped_sum`] — the one-owner-per-`(class, feature)` GATHER helper:
//!   composes the validated v1 `column_reduce` (`ScalarOp::Sum`) prim over
//!   host-grouped per-class row blocks. It is a GATHER, NEVER a scatter-add: NO
//!   new `#[cube]` kernel, NO `SharedMemory`, NO atomics, NO `F::INFINITY`
//!   (Pitfall 1/2, the cubecl-cpu SharedMemory constraint). For the GaussianNB
//!   per-class sum-of-squares the sibling [`class_grouped_sumsq`] composes
//!   `column_reduce` with `ScalarOp::SumSq` (resolves RESEARCH assumption A5:
//!   a per-axis SumSq IS exposed by the reduce prim, so no squared-host-copy is
//!   needed).
//!
//! All host math is f64 (`mlrs_core::host_to_f64`) regardless of the estimator's
//! `F`, because the class-conditional sums and the log-sum-exp are accumulation-
//! heavy and the oracle gate is ≤ 1e-5 vs sklearn. The device touch is ONLY the
//! reduce-prim launch inside the two GATHER helpers.
//!
//! Tests live in `crates/mlrs-algos/tests/nb_common_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{host_to_f64, PrimError};

/// Normalize a SINGLE row of `n_classes` joint log-likelihoods into
/// `(proba, log_proba)` (Pattern 3 — host f64, per-row max-shift, single terminal
/// log).
///
/// Given `joint_ll = [ll_0, …, ll_{n_classes-1}]` this computes
/// `m = max_c ll_c`, `lse = m + log(Σ_c exp(ll_c − m))`,
/// `log_proba_c = ll_c − lse`, and `proba_c = exp(log_proba_c)`. The returned
/// `proba` sums to `1.0 ± 1e-12` and `log_proba == joint_ll − lse` element-wise.
/// The max-shift keeps `exp` from overflowing and the single terminal `log`
/// keeps the small probabilities from underflowing to `0` (Pitfall 9); the
/// pipeline never produces `±∞` (cpu-MLIR-safe).
///
/// Panics only on `n_classes == 0` (a degenerate row with no classes) — callers
/// pass `classes_.len() >= 1` from a fitted estimator.
pub fn log_sum_exp_normalize(joint_ll: &[f64], n_classes: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        joint_ll.len(),
        n_classes,
        "log_sum_exp_normalize: joint_ll length {} != n_classes {}",
        joint_ll.len(),
        n_classes
    );
    assert!(n_classes > 0, "log_sum_exp_normalize: n_classes must be > 0");

    let m = joint_ll.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // m is finite for any finite joint_ll (n_classes > 0); the shifted sum's
    // largest term is exp(0) = 1, so sum_exp >= 1 and log(sum_exp) is finite.
    let sum_exp: f64 = joint_ll.iter().map(|&ll| (ll - m).exp()).sum();
    let lse = m + sum_exp.ln();

    let log_proba: Vec<f64> = joint_ll.iter().map(|&ll| ll - lse).collect();
    let proba: Vec<f64> = log_proba.iter().map(|&lp| lp.exp()).collect();
    (proba, log_proba)
}

/// The empirical class log-prior `log(count_c / Σ count)` from `class_count_`
/// (used when the user supplies no explicit `priors` / `class_prior`).
///
/// A uniform `[10.0, 10.0]` input yields `[ln 0.5, ln 0.5]`. Panics only on an
/// empty input or a non-positive total (a fitted estimator always has at least
/// one sample per observed class).
pub fn empirical_class_log_prior(class_count: &[f64]) -> Vec<f64> {
    assert!(
        !class_count.is_empty(),
        "empirical_class_log_prior: empty class_count"
    );
    let total: f64 = class_count.iter().sum();
    assert!(
        total > 0.0,
        "empirical_class_log_prior: non-positive total count {total}"
    );
    class_count.iter().map(|&c| (c / total).ln()).collect()
}

/// Decode per-row argmax over the `n_rows × n_classes` row-major joint
/// log-likelihood matrix into the predicted label via the sorted `classes_`
/// table. Lowest-index tie-break (sklearn / the reduce-prim convention).
///
/// `joint_ll.len()` must equal `n_rows * classes_.len()`.
pub fn argmax_decode(joint_ll: &[f64], classes_: &[i64]) -> Vec<i32> {
    decode(joint_ll, classes_, true)
}

/// Decode per-row argmin (the ComplementNB decision rule, D-08) over the
/// `n_rows × n_classes` joint log-likelihood matrix into the predicted label via
/// the sorted `classes_` table. Lowest-index tie-break.
pub fn argmin_decode(joint_ll: &[f64], classes_: &[i64]) -> Vec<i32> {
    decode(joint_ll, classes_, false)
}

fn decode(joint_ll: &[f64], classes_: &[i64], take_max: bool) -> Vec<i32> {
    let n_classes = classes_.len();
    assert!(n_classes > 0, "decode: empty classes_");
    assert_eq!(
        joint_ll.len() % n_classes,
        0,
        "decode: joint_ll length {} not a multiple of n_classes {}",
        joint_ll.len(),
        n_classes
    );
    let n_rows = joint_ll.len() / n_classes;
    let mut out: Vec<i32> = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let row = &joint_ll[r * n_classes..(r + 1) * n_classes];
        let mut best_idx = 0usize;
        let mut best_val = row[0];
        for (c, &v) in row.iter().enumerate().skip(1) {
            let better = if take_max { v > best_val } else { v < best_val };
            if better {
                best_val = v;
                best_idx = c;
            }
        }
        out.push(classes_[best_idx] as i32);
    }
    out
}

/// The fraction of exact matches `Σ[pred_i == y_true_i] / n` (the shared `score`,
/// D-07). `[1,1,0]` vs `[1,0,0]` → `2/3`. Panics on a length mismatch (a real
/// caller passes equal-length vectors).
pub fn accuracy_score(pred: &[i32], y_true: &[i32]) -> f64 {
    assert_eq!(
        pred.len(),
        y_true.len(),
        "accuracy_score: length mismatch pred={} y_true={}",
        pred.len(),
        y_true.len()
    );
    if pred.is_empty() {
        return 0.0;
    }
    let correct = pred
        .iter()
        .zip(y_true.iter())
        .filter(|(p, t)| p == t)
        .count();
    correct as f64 / pred.len() as f64
}

/// The one-owner-per-`(class, feature)` GATHER (Pitfall 1/2, ROADMAP #1):
/// `out[c][j] = Σ_{i : class_of_row[i] == c} x[i][j]`, an `n_classes × n_features`
/// host f64 matrix.
///
/// Host-groups the rows by class (one owner per class — a GATHER, NEVER a
/// scatter-add), uploads each class's contiguous row block via
/// [`DeviceArray::from_host`], runs the validated `column_reduce`
/// (`ScalarOp::Sum`) prim over it to sum each feature column, and
/// `release_into(pool)`s the scratch buffer (WR-07 — the per-class scratch is
/// transient and conserves `live_bytes`). NO new `#[cube]` kernel, NO
/// `SharedMemory`, NO atomics, NO `F::INFINITY` — only the v1 reduce prim.
///
/// `x` is the flat `n_samples × n_features` row-major matrix `(shape)`;
/// `class_of_row[i] ∈ [0, n_classes)` is the dense class index of row `i`. A
/// class with no rows contributes an all-zero row. Returns a `PrimError` only if
/// the reduce prim's geometry guard trips (it `u32::try_from`-guards the grid).
pub fn class_grouped_sum<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    class_of_row: &[usize],
    n_classes: usize,
) -> Result<Vec<Vec<f64>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    grouped_reduce::<F>(pool, x, shape, class_of_row, n_classes, ScalarOp::Sum)
}

/// The sum-of-SQUARES sibling of [`class_grouped_sum`] (resolves A5):
/// `out[c][j] = Σ_{i : class_of_row[i] == c} x[i][j]²`. Composes the same
/// per-class GATHER over `column_reduce` but with [`ScalarOp::SumSq`], so the
/// per-axis squared sum is computed by the reduce prim directly (no
/// squared-host-copy). GaussianNB uses `theta_cj = sum_cj / n_c` and
/// `var_cj = sumsq_cj / n_c − theta_cj²` from these two GATHERs.
pub fn class_grouped_sumsq<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    class_of_row: &[usize],
    n_classes: usize,
) -> Result<Vec<Vec<f64>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    grouped_reduce::<F>(pool, x, shape, class_of_row, n_classes, ScalarOp::SumSq)
}

/// Shared GATHER body for [`class_grouped_sum`] / [`class_grouped_sumsq`]:
/// host-group rows by class, `column_reduce` each per-class block with `op`,
/// release the scratch.
fn grouped_reduce<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    class_of_row: &[usize],
    n_classes: usize,
    op: ScalarOp,
) -> Result<Vec<Vec<f64>>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (n_samples, n_features) = shape;
    // Geometry guard BEFORE any launch (T-11-02 / ASVS V5).
    if x.len() != n_samples * n_features {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n_samples,
            cols: n_features,
            len: x.len(),
        });
    }
    assert_eq!(
        class_of_row.len(),
        n_samples,
        "grouped_reduce: class_of_row length {} != n_samples {}",
        class_of_row.len(),
        n_samples
    );

    // Read the full host matrix ONCE; host-group the row indices by class (one
    // owner per class — the GATHER).
    let host = x.to_host(pool);
    let mut out: Vec<Vec<f64>> = vec![vec![0.0f64; n_features]; n_classes];

    for c in 0..n_classes {
        // Collect this class's contiguous row block into a fresh host buffer.
        let rows: Vec<usize> = (0..n_samples).filter(|&i| class_of_row[i] == c).collect();
        let n_c = rows.len();
        if n_c == 0 {
            // No rows for this class → all-zero row (already initialized).
            continue;
        }
        let mut block: Vec<F> = Vec::with_capacity(n_c * n_features);
        for &i in &rows {
            block.extend_from_slice(&host[i * n_features..(i + 1) * n_features]);
        }
        let block_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &block);

        // column_reduce sums each of the n_features columns over the n_c rows of
        // this class block — the per-(class, feature) owner. ReducePath::Shared is
        // always available (cpu-MLIR-safe; the plane path is capability-gated).
        let reduced = column_reduce::<F>(pool, &block_dev, n_c, n_features, op, ReducePath::Shared)?
            .expect("shared-path column_reduce is always available");
        let reduced_host = reduced.to_host(pool);
        for (j, &v) in reduced_host.iter().enumerate() {
            out[c][j] = host_to_f64(v);
        }

        // WR-07: both per-class scratch buffers are transient — release them so
        // the free-list serves the same-shape next class, conserving live_bytes.
        reduced.release_into(pool);
        block_dev.release_into(pool);
    }

    Ok(out)
}
