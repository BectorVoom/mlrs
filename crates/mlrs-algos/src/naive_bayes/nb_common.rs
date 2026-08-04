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
//! - `class_grouped_stats_host` — the per-class column `Σ x` / `Σ x²` sweep that
//!   every NB fit runs (NB-FIT-CPU). ONE row-major pass over the host design
//!   matrix, chunked over rows across a scoped worker pool, with the caller's
//!   per-element validation fused in. `host_workers` / `chunk_rows` /
//!   `PAR_*` are its shared worker-count policy, also used by the
//!   CategoricalNB / BernoulliNB tabulation sweeps.
//! - [`class_grouped_sum`] — the one-owner-per-`(class, feature)` GATHER helper:
//!   composes the validated v1 `column_reduce` (`ScalarOp::Sum`) prim over
//!   host-grouped per-class row blocks. It is a GATHER, NEVER a scatter-add: NO
//!   new `#[cube]` kernel, NO `SharedMemory`, NO atomics, NO `F::INFINITY`
//!   (Pitfall 1/2, the cubecl-cpu SharedMemory constraint). For the GaussianNB
//!   per-class sum-of-squares the sibling [`class_grouped_sumsq`] composes
//!   `column_reduce` with `ScalarOp::SumSq` (resolves RESEARCH assumption A5:
//!   a per-axis SumSq IS exposed by the reduce prim, so no squared-host-copy is
//!   needed). NO FIT CALLS THESE TWO ANY MORE — see `class_grouped_stats_host`
//!   for what they cost and why; they stay as `pub` API for a device-resident
//!   caller.
//!
//! All host math is f64 (`mlrs_core::host_to_f64`) regardless of the estimator's
//! `F`, because the class-conditional sums and the log-sum-exp are accumulation-
//! heavy and the oracle gate is ≤ 1e-5 vs sklearn. No NB fit touches the device
//! at all now except to upload the fitted operand `predict` needs.
//!
//! Tests live in `crates/mlrs-algos/tests/nb_common_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use crate::error::AlgoError;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{host_to_f64, PrimError};

/// Shared tolerance for the integer round-trip check applied to NB labels and
/// categorical feature values (IN-02). A host-read f32/f64 integer-encoded value
/// is treated as the integer `v.round()` when `(v.round() - v).abs() <=
/// NB_LABEL_INT_TOL`. Extracted here so the four variants (`GaussianNB` /
/// `MultinomialNB` / `CategoricalNB` label decode + the categorical category
/// index check) stay consistent if the tolerance is ever tuned.
pub const NB_LABEL_INT_TOL: f64 = 1e-6;

/// The count-based variants' `X`-domain rejection, worded the way scikit-learn
/// words it.
///
/// `MultinomialNB`/`ComplementNB`/`CategoricalNB` read `X` as occurrence counts
/// and call `check_non_negative` on it, which raises
/// `"Negative values in data passed to {name} (input X)"`. sklearn's
/// `check_positive_only_tag_during_fit` asserts on that exact phrase, and more
/// importantly a user who catches sklearn's message should catch ours — so the
/// phrase is reproduced verbatim and the offending value appended, rather than
/// invented afresh.
///
/// A NON-FINITE entry gets its own wording: it is a different fault with a
/// different sklearn message (`check_array`'s "Input contains NaN"), and the
/// Python shim's `check_array` normally rejects it before Rust ever sees it, so
/// this arm is the defence-in-depth path rather than the expected one.
///
/// `sklearn_name` is the sklearn CLASS name (`"MultinomialNB"`), not the
/// snake-case estimator id — the message is the user-facing one.
pub(crate) fn non_negative_x_error(
    estimator: &'static str,
    sklearn_name: &str,
    v: f64,
) -> AlgoError {
    let reason = if v.is_finite() {
        format!("Negative values in data passed to {sklearn_name} (input X) (got {v})")
    } else {
        format!("Input X contains an infinity or a value too large for its dtype (got {v})")
    };
    AlgoError::InvalidFeatureInput { estimator, reason }
}

/// Below this many `n_samples · n_features` elements a chunked host fit pass
/// stays single-threaded: spawning a scoped worker costs ~30 µs, which dwarfs a
/// scan over a few tens of thousands of elements.
///
/// Measured for [`class_grouped_stats_host`] on a 16-core box (MultinomialNB,
/// `C = 4`, min of 9, wall / cpu ms), 1 worker vs 8:
///
/// | n·d     | 160 k     | 640 k     | 1.6 M     | 6.4 M      | 12.8 M     |
/// |---------|-----------|-----------|-----------|------------|------------|
/// | 1w      | 0.38/0.38 | 1.15/1.15 | 2.71/2.73 | 9.41/9.47  | 23.9/23.9  |
/// | 8w      | 0.32/0.68 | 0.49/1.72 | 1.47/5.17 | 3.13/13.56 | 5.86/29.8  |
///
/// So this threshold is the point where parallelism stops LOSING, not where it
/// starts paying: at `160 k` it buys 16 % of wall for 80 % more CPU, and only by
/// `640 k` is it a clear 2.3× win. Raising it to ~`1 << 19` would be defensible
/// for THIS sweep — it is left alone because the constant is shared with the
/// CategoricalNB / BernoulliNB tabulation sweeps, whose per-element work is
/// heavier (a strided scatter, not a column add) and whose crossover is
/// therefore lower; re-measure all three before moving it. NOTE that a
/// heavily-loaded box makes the small rungs look far worse than the table above
/// (8 fresh threads must each win a slot in a deep run queue), which is a
/// benchmarking artifact, not a cost this threshold should be tuned against.
pub(crate) const PAR_MIN_ELEMS: usize = 1 << 15;

/// Ceiling on the worker count for a chunked host fit pass. These passes stream
/// the design matrix, so they are DRAM-bandwidth-bound, not core-bound: the wall
/// clock stops improving long before the cores run out, while CPU time keeps
/// climbing linearly with every worker added. Measured on a 16-core box with the
/// CategoricalNB `100 000 × 128` fit (wall / cpu ms, min of 5):
///
/// | workers | 1     | 2     | 4     | 8     | 16    |
/// |---------|-------|-------|-------|-------|-------|
/// | wall ms | 67.8  | 53.3  | 41.6  | 36.0  | 35.2  |
/// | cpu ms  | 67.4  | 77.5  | 74.5  | 88.0  | 132.4 |
///
/// 8 is the knee: it takes the last real wall-clock gain (13 % over 4), and
/// doubling again buys 2 % for half as much CPU again. Spending a whole machine
/// to shave 2 % off one fit is the wrong trade in a library, so the pool is
/// capped here and the rest of the box is left for the caller's other work.
pub(crate) const PAR_MAX_WORKERS: usize = 8;

/// Per-worker accumulator budget, in entries. A chunked host fit pass replicates
/// its `n_classes · <axis>` table PER worker to stay lock-free, so a fit whose
/// table is already huge runs on ONE table (serial) rather than allocating a
/// copy per core. 1 Mi entries is 4 MiB as `u32` / 8 MiB as `f64`; one
/// un-replicated table is at most the size of the `feature_log_prob_` the fit
/// must return anyway, so it can never dominate the estimator it builds.
pub(crate) const PAR_TABLE_MAX_ENTRIES: usize = 1 << 20;

/// Worker count for a row-chunked host pass over `n_elems` elements: `1` below
/// [`PAR_MIN_ELEMS`], else the machine's parallelism capped at
/// [`PAR_MAX_WORKERS`].
///
/// `env_key` (e.g. `MLRS_CATNB_WORKERS` / `MLRS_BERNNB_WORKERS`) forces the
/// count when set, read through [`mlrs_backend::abflag`] so a test can scope the
/// override to its own thread rather than racing `environ`. Forcing `1` pins the
/// fully serial arm, which is what makes a serial-vs-parallel agreement test
/// possible. Override on new hardware to re-measure the table above.
pub(crate) fn host_workers(env_key: &'static str, n_elems: usize) -> usize {
    if let Some(forced) = mlrs_backend::abflag::var(env_key)
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v >= 1)
    {
        return forced;
    }
    if n_elems < PAR_MIN_ELEMS {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .clamp(1, PAR_MAX_WORKERS)
}

/// Row-chunk size (in ROWS) for a `workers`-way split of `n_samples`, capped at
/// `u32::MAX` rows so a per-chunk `u32` count can never overflow (a count is
/// bounded by its chunk's row count).
pub(crate) fn chunk_rows(n_samples: usize, workers: usize) -> usize {
    n_samples
        .div_ceil(workers.max(1))
        .max(1)
        .min(u32::MAX as usize)
}

/// Normalize a SINGLE row of `n_classes` joint log-likelihoods into
/// `(proba, log_proba)` (Pattern 3 — host f64, per-row max-shift, single terminal
/// log).
///
/// Given `joint_ll = [ll_0, …, ll_{n_classes-1}]` this computes
/// `m = max_c ll_c`, `lse = m + log(Σ_c exp(ll_c − m))`,
/// `log_proba_c = ll_c − lse`, and `proba_c = exp(log_proba_c)`. The returned
/// `proba` sums to `1.0 ± 1e-12` and `log_proba == joint_ll − lse` element-wise.
/// The max-shift keeps `exp` from overflowing; the single terminal `log` keeps
/// the `log_proba` output underflow-safe (Pitfall 9). NOTE (IN-01): only
/// `log_proba` avoids underflow — `proba_c = exp(log_proba_c)` may still flush a
/// tiny `log_proba` to `0.0`, which is the expected, correct behavior. The
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
    assert!(
        n_classes > 0,
        "log_sum_exp_normalize: n_classes must be > 0"
    );

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
/// D-07). `[1,1,0]` vs `[1,0,0]` → `2/3`. Returns `f64::NAN` for an empty input
/// (accuracy is undefined with no samples — sklearn raises; WR-07). Panics on a
/// length mismatch (a real caller passes equal-length vectors).
///
/// TASK-03 (METR-CLS-01): a thin delegate to
/// [`crate::metrics::classification::accuracy_score`] — ONE source of truth
/// for the accuracy computation. This function's own signature/argument
/// order (`pred` first, `y_true` second — opposite sklearn's own
/// `accuracy_score(y_true, y_pred)` convention) and doc-comment are
/// UNCHANGED; only the body changed, so this crate's one caller needs no
/// edit. The empty-input `NaN` contract above is preserved for free by the
/// new implementation's `0.0/0.0 = NaN` division (no special-cased branch;
/// regression-locked by `nb_common_test.rs::nb_common_accuracy_score_empty_input_is_nan`).
pub fn accuracy_score(pred: &[i32], y_true: &[i32]) -> f64 {
    crate::metrics::classification::accuracy_score(y_true, pred, None, true)
        .expect("accuracy_score: pred/y_true length mismatch")
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
/// NOTE (NB-FIT-CPU): **no estimator fit calls this any more.** Every NB fit now
/// runs [`class_grouped_stats_host`] instead — see its docs for the measured
/// reason. This device GATHER is retained as the validated reduce-prim
/// composition (it is `pub` API with its own launch-witness tests) for a caller
/// that genuinely wants the reduction to stay device-resident.
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
/// squared-host-copy). GaussianNB used `theta_cj = sum_cj / n_c` and
/// `var_cj = sumsq_cj / n_c − theta_cj²` from these two GATHERs; it now gets
/// both from ONE [`class_grouped_stats_host`] sweep. Same retention note as
/// [`class_grouped_sum`].
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
        // WR-02: release `block_dev` on EVERY exit (Ok / None / Err) before the `?`
        // propagation so a `column_reduce` failure does not leak the scratch buffer
        // (WR-07 "conserve live_bytes" contract).
        let reduced =
            match column_reduce::<F>(pool, &block_dev, n_c, n_features, op, ReducePath::Shared) {
                Ok(Some(r)) => r,
                Ok(None) => {
                    block_dev.release_into(pool);
                    unreachable!("shared-path column_reduce is always available");
                }
                Err(e) => {
                    block_dev.release_into(pool);
                    return Err(e);
                }
            };
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

// ===========================================================================
// The HOST class-grouped sweep (NB-FIT-CPU) — what every NB fit uses instead of
// the device GATHER above.
// ===========================================================================

/// Per-element validation applied by [`class_grouped_stats_host`] as it sweeps.
///
/// The check is fused into the accumulate loop rather than run as its own pass:
/// the sweep already reads every element, and `check_array`'s finite scan on the
/// Python side is a second single-threaded trip over the whole matrix. Every
/// caller therefore hands `ensure_all_finite=False` to its shim and lets the
/// PyO3 arm re-raise `check_array`'s exact `ValueError` from this verdict.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HostScanCheck {
    /// Reject a non-finite value (`NaN` / `±inf`). GaussianNB's check: it models
    /// real-valued features, so a negative value is perfectly valid.
    Finite,
    /// Reject a non-finite OR negative value — sklearn's `check_non_negative`
    /// parity for the count-based discrete variants, whose
    /// `((count + alpha) / denom).ln()` would otherwise go silently `NaN`/`-inf`.
    NonNegative,
}

impl HostScanCheck {
    /// Whether `xf` fails this check. Written so a `NaN` — for which EVERY
    /// ordering comparison is false — is REJECTED, not silently accumulated.
    #[inline(always)]
    fn rejects(self, xf: f64) -> bool {
        match self {
            HostScanCheck::Finite => !xf.is_finite(),
            HostScanCheck::NonNegative => !xf.is_finite() || xf < 0.0,
        }
    }

    /// The same verdict as [`Self::rejects`], re-expressed as an inclusive
    /// `[lo, hi]` interval so the sweep can fold validity BRANCH-FREE.
    ///
    /// `accepts(xf) == (xf >= lo) & (xf <= hi)`, with both bounds loop-invariant.
    /// This is what lets the accumulate loop vectorize: a `bool` fold over two
    /// compares has no early exit, whereas the equivalent `if rejects { return }`
    /// is a data-dependent branch out of the loop body that blocks every
    /// vectorizer (and mispredicts on real data for the `NonNegative` arm).
    ///
    /// The equivalence rests on `hi = f64::MAX` rather than `f64::INFINITY`:
    /// every FINITE value lies within `±f64::MAX`, while `+inf`, `-inf` and
    /// `NaN` all fail at least one comparison (`NaN` fails both — every ordering
    /// comparison against it is false, which is the property
    /// [`Self::rejects`] is written for too). `-0.0 >= 0.0` holds, so a negative
    /// zero is accepted by both spellings.
    ///
    /// [`Self::rejects`] stays the single source of truth for the REPORTED
    /// offender: the fold only says a row contains one, and the scalar locate
    /// pass then finds it with `rejects`.
    #[inline(always)]
    fn bounds(self) -> (f64, f64) {
        match self {
            HostScanCheck::Finite => (-f64::MAX, f64::MAX),
            HostScanCheck::NonNegative => (0.0, f64::MAX),
        }
    }
}

/// What [`class_grouped_stats_host`] should compute — grouped into a struct
/// because the four call sites differ along three independent axes and a
/// positional `bool, bool, bool` argument list at each of them would be
/// unreadable.
#[derive(Clone, Copy)]
pub(crate) struct StatsRequest {
    /// Per-element validation, fused into the accumulate loop.
    pub check: HostScanCheck,
    /// Also accumulate the per-class `Σ w x²` (GaussianNB's second sufficient
    /// statistic — free here, a whole second traversal if asked for separately).
    pub sumsq: bool,
    /// Also accumulate the UNWEIGHTED whole-column `Σ x` / `Σ x²` (length
    /// `n_features`).
    ///
    /// Only GaussianNB needs this, and only when weights are present: its
    /// `epsilon_` is `var_smoothing · max_j Var(X[:,j])` over the UNWEIGHTED
    /// design (sklearn computes it before it ever looks at `sample_weight`), and
    /// once the per-class totals carry weights the unweighted column variance is
    /// no longer recoverable from them. Unweighted, the caller reduces
    /// [`ClassGroupedStats::sum`] over `c` instead and leaves this off.
    pub global_unweighted: bool,
    /// The caller's `MLRS_*_WORKERS` override key (see [`host_workers`]).
    pub env_key: &'static str,
}

/// The per-class column statistics [`class_grouped_stats_host`] returns, flat
/// `n_classes × n_features` row-major (`[c * n_features + j]`).
///
/// With `weights = Some(w)` every per-class accumulator carries `w_i` as a
/// factor: `sum` is `Σ w x`, `sumsq` is `Σ w x²`, and [`Self::class_weight`] is
/// `Σ w` — which is exactly sklearn's weighted `class_count_`. Unweighted,
/// `class_weight[c]` is the plain row count.
pub(crate) struct ClassGroupedStats {
    /// `sum[c * n_features + j] = Σ_{i : class_of_row[i] == c} w_i · x[i][j]`.
    pub sum: Vec<f64>,
    /// The matching `Σ w_i · x[i][j]²`, EMPTY when the caller did not ask.
    pub sumsq: Vec<f64>,
    /// `class_weight[c] = Σ_{i : class_of_row[i] == c} w_i` — sklearn's
    /// `class_count_` (the plain row count when unweighted).
    pub class_weight: Vec<f64>,
    /// UNWEIGHTED whole-column `Σ x` (length `n_features`), EMPTY unless
    /// [`StatsRequest::global_unweighted`].
    pub global_sum: Vec<f64>,
    /// UNWEIGHTED whole-column `Σ x²` (length `n_features`), EMPTY unless
    /// [`StatsRequest::global_unweighted`].
    pub global_sumsq: Vec<f64>,
    /// Flat index (in the whole matrix) and value of the first element that
    /// failed the [`HostScanCheck`], in ROW-MAJOR order — independent of how the
    /// rows were split across workers. `None` when every element passed.
    pub first_invalid: Option<(usize, f64)>,
}

/// One worker's private accumulators.
struct ChunkAcc {
    sum: Vec<f64>,
    sumsq: Vec<f64>,
    class_weight: Vec<f64>,
    global_sum: Vec<f64>,
    global_sumsq: Vec<f64>,
    first_invalid: Option<(usize, f64)>,
}

/// One worker's share of the sweep: validate + accumulate every requested
/// statistic for every row of `chunk`.
///
/// ## Why this is written as four monomorphized row loops
/// The obvious spelling — one loop that tests `want_sumsq` / `want_global` and
/// `check.rejects(xf)` per element — costs far more than the adds it performs,
/// for two separate reasons:
///
/// 1. **The fused reject is an early `return` out of the inner loop.** A
///    data-dependent exit is an unconditional vectorization barrier, so the
///    whole sweep ran one scalar element at a time. It also MISPREDICTS: for
///    the `NonNegative` arm the predicate reads the data, and BernoulliNB's
///    `x > threshold` twin (a ~30 %-true branch) measured 2.16x the unweighted
///    arm's cpu time purely from mispredicts. Validity is now folded
///    branch-free through [`HostScanCheck::bounds`] and inspected ONCE per row;
///    only a row that actually contains an offender pays the scalar locate
///    pass that pins the exact index.
/// 2. **The optional accumulators were runtime `bool`s.** Perfectly predicted,
///    but they still sit inside the loop body where they keep LLVM from proving
///    a fixed store pattern. Hoisting them into `const` generic parameters
///    gives each of the three live shapes its own straight-line body.
///
/// Accumulating a row that turns out to be invalid is deliberate and harmless:
/// the `NaN` it folds in is discarded with the whole [`ChunkAcc`] the moment
/// `first_invalid` is set, and paying for it unconditionally is what removes the
/// branch.
fn stats_chunk<F>(
    chunk: &[F],
    class_of_row: &[usize],
    weights: Option<&[f64]>,
    n_features: usize,
    check: HostScanCheck,
    flat_base: usize,
    acc: &mut ChunkAcc,
) where
    F: Float + CubeElement + Pod,
{
    // Only three of the four combinations are reachable — `global_unweighted`
    // is GaussianNB's, and GaussianNB always asks for `sumsq` too — but the
    // fourth is spelled out rather than `unreachable!()`d so a future caller
    // cannot turn a request shape into a panic.
    match (!acc.sumsq.is_empty(), !acc.global_sum.is_empty()) {
        (false, false) => stats_rows::<F, false, false>(
            chunk,
            class_of_row,
            weights,
            n_features,
            check,
            flat_base,
            acc,
        ),
        (true, false) => stats_rows::<F, true, false>(
            chunk,
            class_of_row,
            weights,
            n_features,
            check,
            flat_base,
            acc,
        ),
        (false, true) => stats_rows::<F, false, true>(
            chunk,
            class_of_row,
            weights,
            n_features,
            check,
            flat_base,
            acc,
        ),
        (true, true) => stats_rows::<F, true, true>(
            chunk,
            class_of_row,
            weights,
            n_features,
            check,
            flat_base,
            acc,
        ),
    }
}

/// [`stats_chunk`] with the two optional accumulators resolved at COMPILE time.
///
/// `SQ` selects the per-class `Σ w x²`; `GLOB` selects the unweighted whole-column
/// `Σ x` / `Σ x²`. See [`stats_chunk`] for why they are const rather than runtime
/// flags.
fn stats_rows<F, const SQ: bool, const GLOB: bool>(
    chunk: &[F],
    class_of_row: &[usize],
    weights: Option<&[f64]>,
    n_features: usize,
    check: HostScanCheck,
    flat_base: usize,
    acc: &mut ChunkAcc,
) where
    F: Float + CubeElement + Pod,
{
    // Destructured ONCE: the four accumulators are distinct fields, so this both
    // proves them disjoint to the borrow checker inside the loop and stops each
    // iteration re-loading `acc`'s `Vec` pointers through the `&mut`.
    let ChunkAcc {
        sum,
        sumsq,
        class_weight,
        global_sum,
        global_sumsq,
        first_invalid,
    } = acc;
    let (lo, hi) = check.bounds();

    for (r, (row, &c)) in chunk
        .chunks_exact(n_features)
        .zip(class_of_row.iter())
        .enumerate()
    {
        let w = weights.map_or(1.0, |w| w[r]);
        class_weight[c] += w;
        // A zero-weight row contributes nothing to any per-class accumulator,
        // but it still has to be VALIDATED (sklearn's `check_array` scans the
        // whole matrix regardless of the weights) and it still counts toward the
        // unweighted global column totals. So the loop runs either way.
        let base = c * n_features;
        // `ok` is folded with `&`, NOT `&&`: the bitwise operator has no
        // short-circuit, so there is no branch and no early exit in the body.
        let mut ok = true;

        if GLOB && SQ {
            // GaussianNB's weighted arm: per-class Σwx / Σwx² AND the unweighted
            // whole-column Σx / Σx².
            //
            // Deliberately TWO three-stream loops rather than one fused
            // five-stream one. Fusing them reads the row once instead of twice,
            // which looks like the obvious win — but each row re-creates the
            // whole zip, and at SMALL `d` that per-row setup is not amortized:
            // the fused spelling measured a 7 % LOSS at `d = 8` (500 000 × 8)
            // against the code this replaced, while still winning at `d = 128`.
            // Split, both loops win at every width. The second read is nearly
            // free — the row is a few hundred bytes and is still in L1 from the
            // first loop, whereas the ROW's arrival from DRAM (what this sweep
            // is actually bound by) is paid once either way.
            let (sum_row, sq_row) = (
                &mut sum[base..base + n_features],
                &mut sumsq[base..base + n_features],
            );
            for ((&xv, s), ss) in row
                .iter()
                .zip(sum_row.iter_mut())
                .zip(sq_row.iter_mut())
            {
                let xf = host_to_f64(xv);
                ok &= (xf >= lo) & (xf <= hi);
                let wx = w * xf;
                *s += wx;
                *ss += wx * xf;
            }
            for ((&xv, gs), gss) in row
                .iter()
                .zip(global_sum.iter_mut())
                .zip(global_sumsq.iter_mut())
            {
                let xf = host_to_f64(xv);
                *gs += xf;
                *gss += xf * xf;
            }
        } else if SQ {
            let (sum_row, sq_row) = (
                &mut sum[base..base + n_features],
                &mut sumsq[base..base + n_features],
            );
            for ((&xv, s), ss) in row
                .iter()
                .zip(sum_row.iter_mut())
                .zip(sq_row.iter_mut())
            {
                let xf = host_to_f64(xv);
                ok &= (xf >= lo) & (xf <= hi);
                let wx = w * xf;
                *s += wx;
                *ss += wx * xf;
            }
        } else {
            // MultinomialNB / ComplementNB: one multiply-add per element.
            let sum_row = &mut sum[base..base + n_features];
            for (&xv, s) in row.iter().zip(sum_row.iter_mut()) {
                let xf = host_to_f64(xv);
                ok &= (xf >= lo) & (xf <= hi);
                *s += w * xf;
            }
        }

        if !ok {
            // Cold: this row holds the first offender in the chunk. Re-walk it
            // scalar-wise with `rejects` — the single source of truth for the
            // verdict — to report the exact flat index and value.
            for (j, &xv) in row.iter().enumerate() {
                let xf = host_to_f64(xv);
                if check.rejects(xf) {
                    *first_invalid = Some((flat_base + r * n_features + j, xf));
                    return;
                }
            }
            unreachable!("stats_rows: branch-free fold rejected a row `rejects` accepts");
        }
    }
}

/// `out.sum[c, j] = Σ_{i : class_of_row[i] == c} w_i · x[i][j]` (plus the other
/// statistics [`StatsRequest`] asks for), computed in ONE row-major sweep over
/// the HOST design matrix, chunked over rows across a scoped worker pool.
///
/// `weights` is the validated `sample_weight`, host f64, length `n_samples`
/// (`None` = every weight 1). It multiplies each row's contribution to every
/// per-class accumulator, which is exactly what sklearn's
/// `Y *= sample_weight.T` before `feature_count_ += Y.T @ X` does — so
/// [`ClassGroupedStats::sum`] IS sklearn's weighted `feature_count_` and
/// [`ClassGroupedStats::class_weight`] its weighted `class_count_`.
///
/// ## Why this replaced the device GATHER (PERF, NB-FIT-CPU)
///
/// [`class_grouped_sum`] read the matrix back to the host and then, for EACH
/// class, gathered that class's rows into a fresh `n_c × d` block, uploaded the
/// block, launched `column_reduce` over it, and read the result back — so a fit
/// moved the whole design matrix `2 + n_classes` extra times and paid
/// `n_classes` kernel launches. On the cpu backend every launch is a cubecl-cpu
/// kernel with an OS thread per unit, and a default `1000 × 8` BernoulliNB fit
/// cost 96.4 s against sklearn's 3.5 ms. GaussianNB paid it TWICE (sum and
/// sumsq, `2 · n_classes` launches over the same data).
///
/// This sweep reads `x` exactly once, writes `n_classes · n_features`
/// accumulators, creates NO device buffer, and folds the caller's per-element
/// validation in for free. `sum` and `sumsq` come out of the SAME pass, so
/// GaussianNB's two GATHERs become one traversal.
///
/// Each worker accumulates into private tables and the driver sums them, so the
/// sweep is lock-free; a table too large to replicate (`n_classes · n_features >
/// PAR_TABLE_MAX_ENTRIES`) drops to the serial arm rather than allocating a copy
/// per core. Forcing `req.env_key` to `1` pins the serial arm, which is what
/// makes a serial-vs-parallel agreement test possible.
///
/// A class with no rows contributes an all-zero row, matching the GATHER.
/// Callers guard geometry themselves; this asserts only the invariants the
/// indexing depends on.
pub(crate) fn class_grouped_stats_host<F>(
    x: &[F],
    shape: (usize, usize),
    class_of_row: &[usize],
    weights: Option<&[f64]>,
    n_classes: usize,
    req: StatsRequest,
) -> ClassGroupedStats
where
    F: Float + CubeElement + Pod,
{
    let (n_samples, n_features) = shape;
    assert_eq!(
        class_of_row.len(),
        n_samples,
        "class_grouped_stats_host: class_of_row length {} != n_samples {n_samples}",
        class_of_row.len()
    );
    if let Some(w) = weights {
        assert_eq!(
            w.len(),
            n_samples,
            "class_grouped_stats_host: weights length {} != n_samples {n_samples}",
            w.len()
        );
    }
    let table_len = n_classes * n_features;
    let workers = if table_len > PAR_TABLE_MAX_ENTRIES {
        1
    } else {
        host_workers(req.env_key, n_samples * n_features)
    };
    let rows_per = chunk_rows(n_samples, workers);
    let elems_per = rows_per * n_features;
    let sq_len = if req.sumsq { table_len } else { 0 };
    let glob_len = if req.global_unweighted { n_features } else { 0 };

    // One worker's share, returning its private accumulators.
    let run = |chunk: &[F], cls: &[usize], w: Option<&[f64]>, flat_base: usize| {
        let mut acc = ChunkAcc {
            sum: vec![0.0; table_len],
            sumsq: vec![0.0; sq_len],
            class_weight: vec![0.0; n_classes],
            global_sum: vec![0.0; glob_len],
            global_sumsq: vec![0.0; glob_len],
            first_invalid: None,
        };
        stats_chunk::<F>(chunk, cls, w, n_features, req.check, flat_base, &mut acc);
        acc
    };

    let parts: Vec<ChunkAcc> = if workers == 1 {
        vec![run(x, class_of_row, weights, 0)]
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = x
                .chunks(elems_per)
                .zip(class_of_row.chunks(rows_per))
                .enumerate()
                .map(|(ci, (chunk, cls))| {
                    let run = &run;
                    // The weight slice is chunked the same way as the rows, so a
                    // worker indexes it with its LOCAL row index.
                    let w = weights.map(|w| &w[ci * rows_per..ci * rows_per + cls.len()]);
                    scope.spawn(move || run(chunk, cls, w, ci * elems_per))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("class_grouped_stats_host: worker panicked"))
                .collect()
        })
    };

    // The FIRST offender in row-major order — independent of worker count, so
    // the caller's error message does not depend on the machine's core count.
    let first_invalid = parts
        .iter()
        .filter_map(|p| p.first_invalid)
        .min_by(|(a, _), (b, _)| a.cmp(b));

    let mut out = ClassGroupedStats {
        sum: vec![0.0; table_len],
        sumsq: vec![0.0; sq_len],
        class_weight: vec![0.0; n_classes],
        global_sum: vec![0.0; glob_len],
        global_sumsq: vec![0.0; glob_len],
        first_invalid,
    };
    for p in &parts {
        for (a, &v) in out.sum.iter_mut().zip(p.sum.iter()) {
            *a += v;
        }
        for (a, &v) in out.sumsq.iter_mut().zip(p.sumsq.iter()) {
            *a += v;
        }
        for (a, &v) in out.class_weight.iter_mut().zip(p.class_weight.iter()) {
            *a += v;
        }
        for (a, &v) in out.global_sum.iter_mut().zip(p.global_sum.iter()) {
            *a += v;
        }
        for (a, &v) in out.global_sumsq.iter_mut().zip(p.global_sumsq.iter()) {
            *a += v;
        }
    }
    out
}
