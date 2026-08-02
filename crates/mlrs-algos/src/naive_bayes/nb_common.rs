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
}

/// The per-class column statistics [`class_grouped_stats_host`] returns, flat
/// `n_classes × n_features` row-major (`[c * n_features + j]`).
pub(crate) struct ClassGroupedStats {
    /// `sum[c * n_features + j] = Σ_{i : class_of_row[i] == c} x[i][j]`.
    pub sum: Vec<f64>,
    /// The matching `Σ x[i][j]²`, EMPTY when the caller did not ask for it.
    pub sumsq: Vec<f64>,
    /// Flat index (in the whole matrix) and value of the first element that
    /// failed the [`HostScanCheck`], in ROW-MAJOR order — independent of how the
    /// rows were split across workers. `None` when every element passed.
    pub first_invalid: Option<(usize, f64)>,
}

/// One worker's share of the sweep: validate + accumulate `sum` (and `sumsq`
/// when non-empty) for every row of `chunk`.
///
/// `want_sumsq` is hoisted OUT of the inner loop so the common sum-only case
/// (the three discrete variants) does not pay a branch per element.
fn stats_chunk<F>(
    chunk: &[F],
    class_of_row: &[usize],
    n_features: usize,
    check: HostScanCheck,
    flat_base: usize,
    sum: &mut [f64],
    sumsq: &mut [f64],
) -> Option<(usize, f64)>
where
    F: Float + CubeElement + Pod,
{
    let want_sumsq = !sumsq.is_empty();
    for (r, (row, &c)) in chunk
        .chunks_exact(n_features)
        .zip(class_of_row.iter())
        .enumerate()
    {
        let base = c * n_features;
        if want_sumsq {
            let s = &mut sum[base..base + n_features];
            let q = &mut sumsq[base..base + n_features];
            for (j, ((&xv, sa), qa)) in row.iter().zip(s.iter_mut()).zip(q.iter_mut()).enumerate() {
                let xf = host_to_f64(xv);
                if check.rejects(xf) {
                    return Some((flat_base + r * n_features + j, xf));
                }
                *sa += xf;
                *qa += xf * xf;
            }
        } else {
            let s = &mut sum[base..base + n_features];
            for (j, (&xv, sa)) in row.iter().zip(s.iter_mut()).enumerate() {
                let xf = host_to_f64(xv);
                if check.rejects(xf) {
                    return Some((flat_base + r * n_features + j, xf));
                }
                *sa += xf;
            }
        }
    }
    None
}

/// `out.sum[c, j] = Σ_{i : class_of_row[i] == c} x[i][j]` (plus `out.sumsq` when
/// `want_sumsq`), computed in ONE row-major sweep over the HOST design matrix,
/// chunked over rows across a scoped worker pool.
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
/// Each worker accumulates into a private table and the driver sums them, so the
/// sweep is lock-free; a table too large to replicate (`n_classes · n_features >
/// PAR_TABLE_MAX_ENTRIES`) drops to the serial arm rather than allocating a copy
/// per core. `env_key` names the caller's `MLRS_*_WORKERS` override (see
/// [`host_workers`]); forcing it to `1` pins the serial arm, which is what makes
/// a serial-vs-parallel agreement test possible.
///
/// A class with no rows contributes an all-zero row, matching the GATHER.
/// Callers guard geometry themselves; this asserts only the invariant the
/// indexing depends on.
pub(crate) fn class_grouped_stats_host<F>(
    x: &[F],
    shape: (usize, usize),
    class_of_row: &[usize],
    n_classes: usize,
    check: HostScanCheck,
    want_sumsq: bool,
    env_key: &'static str,
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
    let table_len = n_classes * n_features;
    let workers = if table_len > PAR_TABLE_MAX_ENTRIES {
        1
    } else {
        host_workers(env_key, n_samples * n_features)
    };
    let rows_per = chunk_rows(n_samples, workers);
    let elems_per = rows_per * n_features;
    let sq_len = if want_sumsq { table_len } else { 0 };

    // One worker's share, returning its private tables and the first invalid
    // element it saw (flat index + value).
    let run = |chunk: &[F], cls: &[usize], flat_base: usize| {
        let mut sum = vec![0.0f64; table_len];
        let mut sumsq = vec![0.0f64; sq_len];
        let bad = stats_chunk::<F>(
            chunk,
            cls,
            n_features,
            check,
            flat_base,
            &mut sum,
            &mut sumsq,
        );
        (sum, sumsq, bad)
    };

    let parts: Vec<(Vec<f64>, Vec<f64>, Option<(usize, f64)>)> = if workers == 1 {
        vec![run(x, class_of_row, 0)]
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = x
                .chunks(elems_per)
                .zip(class_of_row.chunks(rows_per))
                .enumerate()
                .map(|(ci, (chunk, cls))| {
                    let run = &run;
                    scope.spawn(move || run(chunk, cls, ci * elems_per))
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
        .filter_map(|(_, _, e)| *e)
        .min_by(|(a, _), (b, _)| a.cmp(b));

    let mut sum = vec![0.0f64; table_len];
    let mut sumsq = vec![0.0f64; sq_len];
    for (s, q, _) in &parts {
        for (acc, &v) in sum.iter_mut().zip(s.iter()) {
            *acc += v;
        }
        for (acc, &v) in sumsq.iter_mut().zip(q.iter()) {
            *acc += v;
        }
    }
    ClassGroupedStats {
        sum,
        sumsq,
        first_invalid,
    }
}
