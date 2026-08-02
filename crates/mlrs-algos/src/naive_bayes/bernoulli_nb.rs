//! `BernoulliNB` (NB-03) — Bernoulli Naive Bayes,
//! ≈ `sklearn.naive_bayes.BernoulliNB`.
//!
//! Wave-0 SCAFFOLD: struct + [`BernoulliNBBuilder`] (D-02 sklearn defaults) +
//! `build()` (data-INDEPENDENT validation incl. the D-06 `force_alpha` clip+warn)
//! are SHIPPED; the `Fit` impl carries a REAL geometry guard but a `todo!()`
//! compute body filled in Wave 1. Analog: `multinomial_nb.rs` (discrete builder
//! shape) + the `Option<f64>` knob precedent from `density/kernel_density.rs`.
//! SEPARATE struct (D-03 — no shared base).
//!
//! The D-04 `binarize: Option<f64>` knob — `None` disables binarization
//! (assumes already-binary input); `Some(t)` thresholds `x > t → 1`. The
//! `(1 − x)·log(1 − p)` non-occurrence term folds into the Wave-1 GEMM via
//! `flp = log p − log(1 − p)` + a per-class constant `Σ_j log(1 − p_cj)`
//! (Pitfall 5) — set up there, not here.
//!
//! ## Fit shape (BERNNB-FIT-CPU)
//!
//! The fit is ENTIRELY host-side — there is no `#[cube]` kernel on this path,
//! only ONE row-major sweep that validates, binarizes, and accumulates the
//! per-`(class, feature)` occurrence counts at the same time
//! ([`BernoulliNB::fit_host`] documents what the sweep replaced and why).
//! [`BernoulliNB::fit_from_host_slice`] is the entry point the PyO3 bridge uses,
//! so the operands are never round-tripped through a `DeviceArray` just to be
//! read straight back; the only device buffer a fit creates is the tiny
//! `n_classes × n_features` GEMM operand it returns for `predict`.
//!
//! Tests live in `crates/mlrs-algos/tests/bernoulli_nb_test.rs` (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::naive_bayes::multinomial_nb::{
    decode_classes_host, resolve_class_log_prior, validate_discrete_alpha,
    validate_non_negative_counts,
};
use crate::naive_bayes::nb_common::{
    argmax_decode, chunk_rows, host_workers, log_sum_exp_normalize,
};
// Phase 16 (D-02 shape-B trait-swap): builder UNTOUCHED; `<F, S = Unfit>` state
// param + migration to the consuming-self `typestate` surface. fit/predict math
// BYTE-IDENTICAL (D-03).
use crate::typestate::{
    validate_geometry, Fit, Fitted, PredictLabels, PredictLogProba, PredictProba, Unfit,
};

/// Bernoulli Naive Bayes (NB-03). Construct via [`BernoulliNB::builder`], then
/// [`Fit::fit`] + (Wave-1) the predict surface. Fitted `feature_log_prob_` /
/// `class_log_prior_` are device-resident / host f64 (D-03), `None` until `fit`.
pub struct BernoulliNB<F, S = Unfit> {
    /// Additive smoothing (D-02 default `1.0`).
    alpha: f64,
    /// Keep `alpha` as-is when `< 1e-10` (D-02 default `true`); else clip (D-06).
    /// Retained as fitted-config provenance (exposed via [`BernoulliNB::force_alpha`]);
    /// the clip already applied at `build()` (WR-08).
    force_alpha: bool,
    /// Threshold for binarizing the input; `None` disables binarization (assumes
    /// already-binary), `Some(t)` maps `x > t → 1` (D-02 default `Some(0.0)`).
    binarize: Option<f64>,
    /// Learn class priors from the data (D-02 default `true`).
    fit_prior: bool,
    /// User-supplied class priors, or `None` → empirical (D-02 default `None`).
    class_prior: Option<Vec<f64>>,
    /// DISTINCT sorted class labels inferred at `fit`.
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Fitted `feature_log_prob_` (`n_classes × n_features`), device-resident.
    /// This is the GEMM operand `log p − log(1 − p)` (Pitfall 5), NOT the raw
    /// `log p` — the non-occurrence term is folded in so the device matvec is a
    /// single GEMM. The raw `log p` is recoverable but never needed at predict.
    feature_log_prob_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Per-class non-occurrence constant `Σ_j log(1 − p_cj)` (sklearn `neg_prob`
    /// row-sum), host f64, length `n_classes`, `None` until `fit`. Added to the
    /// joint LL bias alongside `class_log_prior_` (Pitfall 5).
    neg_prob_sum_: Option<Vec<f64>>,
    /// Per-class log-prior (host f64), `None` until `fit`.
    class_log_prior_: Option<Vec<f64>>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> BernoulliNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `BernoulliNB` with sklearn's defaults (D-02).
    pub fn builder() -> BernoulliNBBuilder {
        BernoulliNBBuilder::default()
    }
}

impl<F> BernoulliNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The inferred class labels (empty until `fit`).
    pub fn classes(&self) -> &[i64] {
        &self.classes_
    }

    /// The stored `force_alpha` config provenance (WR-08). The D-06 alpha clip is
    /// already applied at `build()`; this exposes whether the clip was suppressed.
    pub fn force_alpha(&self) -> bool {
        self.force_alpha
    }

    /// The per-class log-prior (`None` until `fit`).
    pub fn class_log_prior(&self) -> Option<&[f64]> {
        self.class_log_prior_.as_deref()
    }

    /// Host-materialized GEMM operand `log p − log(1 − p)` (`n_classes ×
    /// n_features` row-major), `None` until `fit`. NOTE this is the folded
    /// operand, not the raw `feature_log_prob_` (= `log p`).
    pub fn feature_log_prob_delta(&self, pool: &BufferPool<ActiveRuntime>) -> Option<Vec<f64>> {
        self.feature_log_prob_
            .as_ref()
            .map(|t| t.to_host(pool).iter().map(|&v| host_to_f64(v)).collect())
    }
}

/// Apply the D-04 binarization to a host f64 buffer: `Some(t)` maps `x > t → 1.0`
/// else `0.0`; `None` assumes the input is already binary and passes it through.
///
/// PREDICT-side only. The fit never materializes a binarized copy — it folds the
/// threshold into the accumulate step ([`count_chunk_binarized`]).
fn binarize_host(buf: &mut [f64], binarize: Option<f64>) {
    if let Some(t) = binarize {
        for v in buf.iter_mut() {
            *v = if *v > t { 1.0 } else { 0.0 };
        }
    }
}

/// The `MLRS_BERNNB_WORKERS` override key handed to
/// [`crate::naive_bayes::nb_common::host_workers`] — see there for the
/// worker-count policy. `MLRS_BERNNB_WORKERS=1` pins the fully serial arm, which
/// is what makes the serial-vs-parallel agreement test possible.
const WORKERS_ENV: &str = "MLRS_BERNNB_WORKERS";

/// Per-worker accumulator budget, in entries. The fit replicates the
/// `n_classes · n_features` occurrence table PER worker to stay lock-free, so a
/// fit whose table is already huge runs on ONE table (serial) rather than
/// allocating a copy per core. 1 Mi entries is 4 MiB as `u32`; the un-replicated
/// table is at most the same size as the `feature_log_prob_` the fit must return
/// anyway, so it can never dominate the estimator it builds.
const PAR_TABLE_MAX_ENTRIES: usize = 1 << 20;

/// Fused validate + binarize + tabulate over ONE row-chunk, the `binarize =
/// Some(threshold)` arm: `table[c · n_features + j] += (x[i,j] > threshold)` for
/// every row `i` of the chunk, where `c = class_of_row[i]`.
///
/// Returns the flat index (in the WHOLE matrix — `flat_base` is the chunk's
/// offset) and value of the first element that is not finite and non-negative,
/// or `None`. Reporting the flat index lets the parallel driver pick the FIRST
/// offender in row-major order, so the error message does not depend on the
/// worker count. The predicate is written `!xf.is_finite() || xf < 0.0` — the
/// `is_finite` arm is what rejects a `NaN`, for which every ordering comparison
/// (including `xf > threshold`) is false and which would otherwise be silently
/// counted as a non-occurrence.
///
/// Counts are `u32`: a count is bounded by the chunk's row count, and
/// [`chunk_rows`] caps a chunk at `u32::MAX` rows. That halves the accumulator
/// traffic against an `f64` table, and the occurrence count is EXACT — the
/// per-class row block sum this replaced accumulated in floating point.
fn count_chunk_binarized<F>(
    chunk: &[F],
    class_of_row: &[usize],
    n_features: usize,
    threshold: f64,
    flat_base: usize,
    table: &mut [u32],
) -> Option<(usize, f64)>
where
    F: Float + CubeElement + Pod,
{
    for (r, (row, &c)) in chunk
        .chunks_exact(n_features)
        .zip(class_of_row.iter())
        .enumerate()
    {
        let acc = &mut table[c * n_features..(c + 1) * n_features];
        for (j, (&xv, a)) in row.iter().zip(acc.iter_mut()).enumerate() {
            let xf = host_to_f64(xv);
            if !xf.is_finite() || xf < 0.0 {
                return Some((flat_base + r * n_features + j, xf));
            }
            *a += u32::from(xf > threshold);
        }
    }
    None
}

/// The `binarize = None` twin of [`count_chunk_binarized`]: the input is assumed
/// already binary, so the raw value is summed (sklearn's `binarize=None` feeds
/// `X` to the count GEMM unchanged). Accumulates in `f64` because a pass-through
/// value need not be an integer.
fn count_chunk_raw<F>(
    chunk: &[F],
    class_of_row: &[usize],
    n_features: usize,
    flat_base: usize,
    table: &mut [f64],
) -> Option<(usize, f64)>
where
    F: Float + CubeElement + Pod,
{
    for (r, (row, &c)) in chunk
        .chunks_exact(n_features)
        .zip(class_of_row.iter())
        .enumerate()
    {
        let acc = &mut table[c * n_features..(c + 1) * n_features];
        for (j, (&xv, a)) in row.iter().zip(acc.iter_mut()).enumerate() {
            let xf = host_to_f64(xv);
            if !xf.is_finite() || xf < 0.0 {
                return Some((flat_base + r * n_features + j, xf));
            }
            *a += xf;
        }
    }
    None
}

/// Builder for [`BernoulliNB`] (D-01). Defaults (D-02): `alpha=1.0`,
/// `force_alpha=true`, `binarize=Some(0.0)`, `fit_prior=true`,
/// `class_prior=None`. Setter names mirror sklearn (D-09).
#[derive(Debug, Clone)]
pub struct BernoulliNBBuilder {
    alpha: f64,
    force_alpha: bool,
    binarize: Option<f64>,
    fit_prior: bool,
    class_prior: Option<Vec<f64>>,
}

impl Default for BernoulliNBBuilder {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            force_alpha: true,
            binarize: Some(0.0),
            fit_prior: true,
            class_prior: None,
        }
    }
}

impl BernoulliNBBuilder {
    /// Set the additive smoothing `alpha`.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    /// Set whether to keep a tiny `alpha` as-is (else clip to `1e-10`, D-06).
    pub fn force_alpha(mut self, force_alpha: bool) -> Self {
        self.force_alpha = force_alpha;
        self
    }
    /// Set the binarization threshold (`None` disables binarization).
    pub fn binarize(mut self, binarize: Option<f64>) -> Self {
        self.binarize = binarize;
        self
    }
    /// Set whether to learn class priors from the data.
    pub fn fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }
    /// Set explicit class priors (`None` → empirical / uniform).
    pub fn class_prior(mut self, class_prior: Option<Vec<f64>>) -> Self {
        self.class_prior = class_prior;
        self
    }

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters at
    /// `build()` BEFORE any data is seen (D-05): `alpha >= 0`, finite+non-negative
    /// `class_prior` entries, and the D-06 `force_alpha` clip+warn (shared
    /// [`validate_discrete_alpha`]). `binarize` needs no validation (any finite or
    /// `None` threshold is valid).
    pub fn build<F>(self) -> Result<BernoulliNB<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let alpha = validate_discrete_alpha(
            "bernoulli_nb",
            self.alpha,
            self.force_alpha,
            self.class_prior.as_deref(),
        )?;
        Ok(BernoulliNB {
            alpha,
            force_alpha: self.force_alpha,
            binarize: self.binarize,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            classes_: Vec::new(),
            n_features: 0,
            feature_log_prob_: None,
            neg_prob_sum_: None,
            class_log_prior_: None,
            _state: PhantomData,
        })
    }
}

impl<F> BernoulliNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Fit directly from HOST slices — the no-upload twin of [`Fit::fit`].
    ///
    /// A `BernoulliNB` fit reads every element of `x` and `y` on the host (the
    /// counts are validated, thresholded, and smoothed into host-f64 tables) and
    /// touches the device exactly once, to upload the `n_classes × n_features`
    /// GEMM operand `predict` needs. Routing the OPERANDS through a `DeviceArray`
    /// therefore bought a round trip and nothing else: `from_host` copied `n·d`
    /// floats into a pool buffer and `to_host` copied them straight back out. The
    /// PyO3 bridge hands the Arrow values here instead, so a fit touches the
    /// caller's buffer once. Precedent: `Ridge::fit_from_host_slice` /
    /// `CategoricalNB::fit_from_host_slice`.
    ///
    /// `pool` is still required — the fitted `feature_log_prob_` is
    /// device-resident (D-03) — but only that one small upload uses it.
    ///
    /// `shape` is `(n_samples, n_features)` and `x` is row-major, exactly as for
    /// [`Fit::fit`]; the geometry guard is the slice twin of `validate_geometry`.
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y: &[F],
        shape: (usize, usize),
    ) -> Result<BernoulliNB<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        if y.len() != n_samples {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: 1,
                len: y.len(),
            }));
        }
        self.fit_host(pool, x, y, shape)
    }

    /// The shared host fit body behind [`Fit::fit`] and
    /// [`BernoulliNB::fit_from_host_slice`] — both geometry guards run in the
    /// caller, so this is the math only.
    ///
    /// ## Why ONE fused sweep (PERF)
    ///
    /// The previous body did the same work in seven full passes over the design
    /// matrix plus `n_classes` device launches: read `x` to host, map it into an
    /// `n·d` `Vec<f64>`, scan that for finiteness, scan it again to binarize,
    /// map it back into an `n·d` `Vec<F>`, upload THAT, and then hand it to
    /// `class_grouped_sum` — which read it back to the host, and for EACH class
    /// gathered that class's rows into a fresh `n_c·d` block, uploaded the block,
    /// launched `column_reduce` over it, and read the result back. On the cpu
    /// backend every one of those launches is a cubecl-cpu kernel with a thread
    /// per unit, so a fit cost seconds where the arithmetic is one pass of adds.
    ///
    /// Now: [`count_chunk_binarized`] validates, thresholds, and accumulates
    /// `feature_count_[c, j]` in ONE row-major sweep, chunked over rows across a
    /// scoped worker pool with a replicated (lock-free) `u32` accumulator per
    /// worker. No binarized copy is materialized, no per-class row block is
    /// gathered, and no device buffer is created for the design matrix at all —
    /// the sweep reads `x` exactly once and writes `n_classes · n_features`
    /// counts. The only device touch left is the fitted operand upload.
    fn fit_host(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x_host: &[F],
        y_host: &[F],
        shape: (usize, usize),
    ) -> Result<BernoulliNB<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // Labels first: the previous body decoded `classes_` BEFORE it scanned
        // `X`, so a fit with both a bad label and a bad feature value reports the
        // label error. Keep that precedence — the X validation now lives inside
        // the count sweep below.
        let (classes_, class_of_row, n_classes) = decode_classes_host::<F>("bernoulli_nb", y_host)?;

        // --- The fused sweep: validate the RAW input is finite and non-negative
        //     (CR-01 / T-11-02 — a NaN would otherwise be silently counted as a
        //     non-occurrence and a negative count is rejected for parity), apply
        //     the D-04 `x > t → 1` threshold, and accumulate
        //     feature_count_[c,j] = Σ over class-c rows of the binarized x[i,j],
        //     all in one pass. `binarize=None` assumes the input is already
        //     binary (pass-through). ---
        let table_len = n_classes * n_features;
        let workers = if table_len > PAR_TABLE_MAX_ENTRIES {
            1
        } else {
            host_workers(WORKERS_ENV, n_samples * n_features)
        };
        let rows_per = chunk_rows(n_samples, workers);
        let elems_per = rows_per * n_features;
        let binarize = self.binarize;

        // One worker's share of the sweep, returning its private count table and
        // the first invalid element it saw (flat index + value).
        let run = |chunk: &[F], cls: &[usize], flat_base: usize| -> (Vec<f64>, Option<(usize, f64)>) {
            match binarize {
                Some(t) => {
                    let mut table = vec![0u32; table_len];
                    let bad = count_chunk_binarized::<F>(
                        chunk, cls, n_features, t, flat_base, &mut table,
                    );
                    (table.iter().map(|&v| f64::from(v)).collect(), bad)
                }
                None => {
                    let mut table = vec![0.0f64; table_len];
                    let bad = count_chunk_raw::<F>(chunk, cls, n_features, flat_base, &mut table);
                    (table, bad)
                }
            }
        };

        // The `n_samples <= u32::MAX` arm is not about parallelism: a single
        // table over MORE rows than that could overflow its `u32` counters, so
        // such a fit takes the chunked branch (whose chunks are capped at
        // `u32::MAX` rows by `chunk_rows`) even at one worker.
        let parts: Vec<(Vec<f64>, Option<(usize, f64)>)> =
            if workers == 1 && n_samples <= u32::MAX as usize {
                vec![run(x_host, &class_of_row, 0)]
            } else {
                std::thread::scope(|scope| {
                    let handles: Vec<_> = x_host
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
                        .map(|h| h.join().expect("bernoulli_nb: fit worker panicked"))
                        .collect()
                })
            };

        // The FIRST offender in row-major order — independent of worker count,
        // and the same value the whole-matrix scan this replaced reported.
        if let Some((_, xf)) = parts
            .iter()
            .filter_map(|(_, e)| *e)
            .min_by(|(a, _), (b, _)| a.cmp(b))
        {
            return Err(AlgoError::InvalidLabels {
                estimator: "bernoulli_nb",
                reason: format!("input X must be finite and non-negative (got {xf})"),
            });
        }
        let mut feature_count: Vec<f64> = vec![0.0; table_len];
        for (table, _) in &parts {
            for (acc, &v) in feature_count.iter_mut().zip(table.iter()) {
                *acc += v;
            }
        }

        let mut class_count_: Vec<f64> = vec![0.0; n_classes];
        for &c in &class_of_row {
            class_count_[c] += 1.0;
        }

        // --- feature_log_prob_[c,j] = log((count+alpha)/(class_count[c]+2·alpha))
        //     (Pitfall 4: the Bernoulli denominator smoothing is 2·alpha). The GEMM
        //     operand is the DELTA log p − log(1−p) and the per-class const
        //     Σ_j log(1−p_cj) becomes the bias (Pitfall 5). ---
        let alpha = self.alpha;
        let mut flp_delta: Vec<f64> = vec![0.0; table_len];
        let mut neg_prob_sum: Vec<f64> = vec![0.0; n_classes];
        for c in 0..n_classes {
            let denom = class_count_[c] + 2.0 * alpha;
            for j in 0..n_features {
                let p = (feature_count[c * n_features + j] + alpha) / denom;
                let log_p = p.ln();
                let log_1mp = (1.0 - p).ln();
                flp_delta[c * n_features + j] = log_p - log_1mp;
                neg_prob_sum[c] += log_1mp;
            }
        }

        let class_log_prior_ = resolve_class_log_prior(
            "bernoulli_nb",
            self.fit_prior,
            &self.class_prior,
            &class_count_,
            n_classes,
        )?;

        // The consuming-self transition carries no prior fitted state (fresh
        // `Unfit` has feature_log_prob_ = None) — the old WR-07 re-fit release is
        // vacuous and dropped; reuse across re-CONSTRUCT+fit flows via the pool.
        let flp_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(
            pool,
            &flp_delta.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<F>>(),
        );

        Ok(BernoulliNB {
            alpha: self.alpha,
            force_alpha: self.force_alpha,
            binarize: self.binarize,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            classes_,
            n_features,
            feature_log_prob_: Some(flp_dev),
            neg_prob_sum_: Some(neg_prob_sum),
            class_log_prior_: Some(class_log_prior_),
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for BernoulliNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = BernoulliNB<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<BernoulliNB<F, Fitted>, AlgoError> {
        let (n_samples, _n_features) = shape;
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "bernoulli_nb",
            operation: "fit (requires y)",
        })?;
        if y.len() != n_samples {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: 1,
                len: y.len(),
            }));
        }
        // The device buffers exist only because `Fit` is the shared typestate
        // surface — this fit's math is entirely host-side, so read both operands
        // once and run the same body `fit_from_host_slice` runs. A caller that
        // already holds host slices should use that entry point instead and skip
        // the upload these two reads undo.
        let x_host = x.to_host(pool);
        let y_host = y.to_host(pool);
        self.fit_host(pool, &x_host, &y_host, shape)
    }
}

impl<F> BernoulliNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Per-query-row joint log-likelihood (`n_query × n_classes`, host f64). The
    /// query X is binarized the SAME way as fit, then
    /// `LL[i,c] = class_log_prior_[c] + Σ_j log(1−p_cj)
    ///          + Σ_j x_ij·(log p_cj − log(1−p_cj))` — the Σ_j x·delta term is the
    /// device `gemm(X_bin @ flp_delta.T)` (Pitfall 5), the rest the per-class bias.
    fn joint_log_likelihood(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_query, n_features) = shape;
        let flp = self.feature_log_prob_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "bernoulli_nb",
            operation: "predict (call fit first)",
        })?;
        let neg_prob_sum = self.neg_prob_sum_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "bernoulli_nb",
            operation: "predict (call fit first)",
        })?;
        let class_log_prior = self.class_log_prior_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "bernoulli_nb",
            operation: "predict (call fit first)",
        })?;
        if n_query == 0 || n_features != self.n_features || x.len() != n_query * n_features {
            return Err(AlgoError::InvalidLabels {
                estimator: "bernoulli_nb",
                reason: format!(
                    "predict geometry: got {n_query}x{n_features}, fitted n_features={}",
                    self.n_features
                ),
            });
        }
        let n_classes = self.classes_.len();

        // Binarize the query the same way as fit BEFORE the GEMM.
        let mut xq_bin: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        // CR-01 / T-11-02: a negative / NaN query row is equally invalid — reject it
        // before binarization + the GEMM (sklearn rejects at predict too).
        validate_non_negative_counts("bernoulli_nb", &xq_bin)?;
        binarize_host(&mut xq_bin, self.binarize);
        let xq_bin_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(
            pool,
            &xq_bin.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<F>>(),
        );

        let raw = gemm::<F>(
            pool,
            &xq_bin_dev,
            (n_query, n_features),
            flp,
            (n_features, n_classes),
            false,
            true,
            None,
        )?;
        let raw_host = raw.to_host(pool);
        raw.release_into(pool);
        xq_bin_dev.release_into(pool);

        let mut jll = vec![0.0f64; n_query * n_classes];
        for i in 0..n_query {
            for c in 0..n_classes {
                jll[i * n_classes + c] = class_log_prior[c]
                    + neg_prob_sum[c]
                    + host_to_f64(raw_host[i * n_classes + c]);
            }
        }
        Ok(jll)
    }
}

impl<F> PredictLabels<F> for BernoulliNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let jll = self.joint_log_likelihood(pool, x, shape)?;
        let labels = argmax_decode(&jll, &self.classes_);
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

impl<F> PredictProba<F> for BernoulliNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, _n_features) = shape;
        let jll = self.joint_log_likelihood(pool, x, shape)?;
        let n_classes = self.classes_.len();
        let mut proba: Vec<F> = vec![f64_to_host::<F>(0.0); n_query * n_classes];
        for r in 0..n_query {
            let row = &jll[r * n_classes..(r + 1) * n_classes];
            let (p, _lp) = log_sum_exp_normalize(row, n_classes);
            for (c, &pv) in p.iter().enumerate() {
                proba[r * n_classes + c] = f64_to_host::<F>(pv);
            }
        }
        Ok(DeviceArray::from_host(pool, &proba))
    }
}

impl<F> PredictLogProba<F> for BernoulliNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_log_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, _n_features) = shape;
        let jll = self.joint_log_likelihood(pool, x, shape)?;
        let n_classes = self.classes_.len();
        let mut log_proba: Vec<F> = vec![f64_to_host::<F>(0.0); n_query * n_classes];
        for r in 0..n_query {
            let row = &jll[r * n_classes..(r + 1) * n_classes];
            let (_p, lp) = log_sum_exp_normalize(row, n_classes);
            for (c, &lpv) in lp.iter().enumerate() {
                log_proba[r * n_classes + c] = f64_to_host::<F>(lpv);
            }
        }
        Ok(DeviceArray::from_host(pool, &log_proba))
    }
}
