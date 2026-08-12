//! `ComplementNB` (NB-04) — Complement Naive Bayes,
//! ≈ `sklearn.naive_bayes.ComplementNB`.
//!
//! Wave-0 SCAFFOLD: struct + [`ComplementNBBuilder`] (D-02 sklearn defaults) +
//! `build()` (data-INDEPENDENT validation incl. the D-06 `force_alpha` clip+warn)
//! are SHIPPED; the `Fit` impl carries a REAL geometry guard but a `todo!()`
//! compute body filled in Wave 1. Analog: `multinomial_nb.rs` (discrete builder
//! shape). SEPARATE struct (D-03). ComplementNB carries the extra `norm: bool`
//! knob and decodes with `argmin` INTERNALLY (D-08 — it picks the class whose
//! complement fits worst; the sign flips). Its complement-weighted
//! `feature_log_prob_` + optional L1 `norm` is a DIFFERENT formula from
//! MultinomialNB — implement it verbatim from FEATURES.md in Wave 1, do NOT copy
//! Multinomial (Pitfall 6).
//!
//! ## Fit shape (NB-FIT-CPU)
//!
//! The fit is ENTIRELY host-side — one
//! [`class_grouped_stats_host`](crate::naive_bayes::nb_common) sweep that
//! validates and accumulates `feature_count_` in a single row-major pass, then
//! host f64 complement weighting. It touches the device once, to upload the
//! `n_classes × n_features` GEMM operand `predict` needs;
//! [`ComplementNB::fit_from_host_slice`] is the entry point the PyO3 bridge uses
//! so the operands are never round-tripped through a `DeviceArray` just to be
//! read straight back.
//!
//! Tests live in `crates/mlrs-algos/tests/complement_nb_test.rs` (AGENTS.md §2).

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::linear::ridge::validate_sample_weight;
use crate::naive_bayes::multinomial_nb::{
    decode_classes_host, resolve_class_log_prior, validate_discrete_alpha,
    validate_non_negative_counts,
};
use crate::naive_bayes::nb_common::{
    argmin_decode, class_grouped_stats_host, log_sum_exp_normalize, ClassGroupedStats,
    HostScanCheck, StatsRequest, non_negative_x_error,
};
// NB-PERSIST: the safetensors container. The fitted state is the shared discrete
// core plus the `norm` provenance scalar.
use crate::naive_bayes::nb_persist::{
    read_discrete_core, AlignedBytes, DiscreteCoreRef, LoadModel, NbFile, NbWriter, PersistError,
    SaveModel,
};
// Phase 16 (D-02 shape-B trait-swap): builder UNTOUCHED; `<F, S = Unfit>` state
// param + migration to the consuming-self `typestate` surface. fit/predict math
// BYTE-IDENTICAL (D-03).
use crate::typestate::{
    validate_geometry, Fit, Fitted, PredictLabels, PredictLogProba, PredictProba, Unfit,
};

/// Complement Naive Bayes (NB-04). Construct via [`ComplementNB::builder`], then
/// [`Fit::fit`] + (Wave-1) the predict surface (argmin decode internally, D-08).
/// Fitted `feature_log_prob_` / `class_log_prior_` are device-resident / host f64
/// (D-03), `None` until `fit`.
pub struct ComplementNB<F, S = Unfit> {
    /// Additive smoothing (D-02 default `1.0`).
    alpha: f64,
    /// Keep `alpha` as-is when `< 1e-10` (D-02 default `true`); else clip (D-06).
    /// Retained as fitted-config provenance (exposed via [`ComplementNB::force_alpha`]);
    /// the clip already applied at `build()` (WR-08).
    force_alpha: bool,
    /// Learn class priors from the data (D-02 default `true`).
    fit_prior: bool,
    /// User-supplied class priors, or `None` → empirical (D-02 default `None`).
    class_prior: Option<Vec<f64>>,
    /// Apply a second L1 normalization to the complement weights (D-02 default
    /// `false`).
    norm: bool,
    /// DISTINCT sorted class labels inferred at `fit`.
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Fitted complement-weighted `feature_log_prob_` (`n_classes × n_features`),
    /// device-resident, `None` until `fit`.
    feature_log_prob_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Per-class log-prior (host f64), `None` until `fit`.
    class_log_prior_: Option<Vec<f64>>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> ComplementNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `ComplementNB` with sklearn's defaults (D-02).
    pub fn builder() -> ComplementNBBuilder {
        ComplementNBBuilder::default()
    }
}

impl<F> ComplementNB<F, Fitted>
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

    /// Host-materialized complement-weighted `feature_log_prob_` (`n_classes ×
    /// n_features` row-major), `None` until `fit`. This is the sklearn weights
    /// (`-logged`, or `logged/summed` under `norm`).
    pub fn feature_log_prob(&self, pool: &BufferPool<ActiveRuntime>) -> Option<Vec<f64>> {
        self.feature_log_prob_
            .as_ref()
            .map(|t| t.to_host(pool).iter().map(|&v| host_to_f64(v)).collect())
    }
}

/// The `estimator` discriminator written into every `ComplementNB` model file
/// (see [`nb_persist`](crate::naive_bayes::nb_persist)).
const PERSIST_TAG: &str = "complement_nb";

impl<F> SaveModel for ComplementNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted model to `path` as a safetensors file.
    ///
    /// The shared discrete core plus one scalar: `norm`, which selects the
    /// second L1 normalization of the complement weights. `norm` is baked into
    /// the fitted `feature_log_prob_` already, so it is stored as provenance —
    /// the same reason `alpha` and `force_alpha` are.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        let feature_log_prob = self
            .feature_log_prob_
            .as_ref()
            .ok_or_else(|| absent("feature_log_prob_"))?
            .to_host(pool);
        let class_log_prior = self
            .class_log_prior_
            .as_deref()
            .ok_or_else(|| absent("class_log_prior_"))?;

        let mut w = NbWriter::new(PERSIST_TAG);
        w.scalar_bool("param:norm", self.norm);
        DiscreteCoreRef {
            matrix_name: "feature_log_prob_",
            alpha: self.alpha,
            force_alpha: self.force_alpha,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior.as_deref(),
            classes: &self.classes_,
            class_log_prior,
            feature_log_prob: &feature_log_prob,
            n_features: self.n_features,
        }
        .write_into(&mut w)?;
        w.write(path)
    }
}

impl<F> LoadModel for ComplementNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read a model back from `path`, re-uploading `feature_log_prob_` to
    /// `pool`.
    ///
    /// ```ignore
    /// let clf: ComplementNB<f32, Fitted> = ComplementNB::load(&mut pool, path)?;
    /// ```
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<ComplementNB<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = NbFile::parse(&raw, PERSIST_TAG)?;
        let core = read_discrete_core::<F>(&file, "feature_log_prob_")?;

        Ok(ComplementNB {
            alpha: core.alpha,
            force_alpha: core.force_alpha,
            fit_prior: core.fit_prior,
            class_prior: core.class_prior,
            norm: file.scalar_bool("param:norm")?,
            classes_: core.classes,
            n_features: core.n_features,
            feature_log_prob_: Some(DeviceArray::from_host(pool, &core.feature_log_prob)),
            class_log_prior_: Some(core.class_log_prior),
            _state: PhantomData,
        })
    }
}

/// Builder for [`ComplementNB`] (D-01). Defaults (D-02): `alpha=1.0`,
/// `force_alpha=true`, `fit_prior=true`, `class_prior=None`, `norm=false`. Setter
/// names mirror sklearn (D-09).
#[derive(Debug, Clone)]
pub struct ComplementNBBuilder {
    alpha: f64,
    force_alpha: bool,
    fit_prior: bool,
    class_prior: Option<Vec<f64>>,
    norm: bool,
}

impl Default for ComplementNBBuilder {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            force_alpha: true,
            fit_prior: true,
            class_prior: None,
            norm: false,
        }
    }
}

impl ComplementNBBuilder {
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
    /// Set whether to apply the second L1 normalization to the weights.
    pub fn norm(mut self, norm: bool) -> Self {
        self.norm = norm;
        self
    }

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters at
    /// `build()` (D-05): `alpha >= 0`, finite+non-negative `class_prior`, and the
    /// D-06 `force_alpha` clip+warn (shared [`validate_discrete_alpha`]). `norm`
    /// needs no validation.
    pub fn build<F>(self) -> Result<ComplementNB<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let alpha = validate_discrete_alpha(
            "complement_nb",
            self.alpha,
            self.force_alpha,
            self.class_prior.as_deref(),
        )?;
        Ok(ComplementNB {
            alpha,
            force_alpha: self.force_alpha,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            norm: self.norm,
            classes_: Vec::new(),
            n_features: 0,
            feature_log_prob_: None,
            class_log_prior_: None,
            _state: PhantomData,
        })
    }
}

/// The `MLRS_CNB_WORKERS` override key handed to
/// [`crate::naive_bayes::nb_common::host_workers`] — see there for the
/// worker-count policy. `MLRS_CNB_WORKERS=1` pins the fully serial arm, which is
/// what makes the serial-vs-parallel agreement test possible.
const WORKERS_ENV: &str = "MLRS_CNB_WORKERS";

impl<F> ComplementNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Fit directly from HOST slices — the no-upload twin of [`Fit::fit`].
    ///
    /// A `ComplementNB` fit reads every element of `x` and `y` on the host (the
    /// counts are validated and accumulated, then complement-weighted into
    /// host-f64 tables) and touches the device exactly once, to upload the
    /// `n_classes × n_features` GEMM operand `predict` needs. Routing the
    /// OPERANDS through a `DeviceArray` therefore bought a round trip and nothing
    /// else. Precedent: `Ridge::fit_from_host_slice` /
    /// `CategoricalNB::fit_from_host_slice`.
    ///
    /// `shape` is `(n_samples, n_features)` and `x` is row-major, exactly as for
    /// [`Fit::fit`]; the geometry guard is the slice twin of `validate_geometry`.
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y: &[F],
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<ComplementNB<F, Fitted>, AlgoError> {
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
        self.fit_host(pool, x, y, shape, sample_weight)
    }

    /// The shared host fit body behind [`Fit::fit`] and
    /// [`ComplementNB::fit_from_host_slice`] — both geometry guards run in the
    /// caller, so this is the math only.
    ///
    /// ## Why ONE fused sweep (PERF, NB-FIT-CPU)
    ///
    /// The previous body read `x` to host, mapped it into an `n·d` `Vec<f64>`,
    /// scanned that for the non-negative check, and then called
    /// `class_grouped_sum` — which read `x` back to the host AGAIN and, for EACH
    /// class, gathered that class's rows into a fresh block, uploaded the block,
    /// launched `column_reduce` over it, and read the result back. On the cpu
    /// backend every launch is a cubecl-cpu kernel with a thread per unit.
    ///
    /// Now `class_grouped_stats_host` validates and accumulates
    /// `feature_count_[c, j]` in ONE row-major sweep, chunked over rows across a
    /// scoped worker pool with a replicated (lock-free) table per worker. The
    /// design matrix is read once and never becomes a device buffer.
    fn fit_host(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x_host: &[F],
        y_host: &[F],
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<ComplementNB<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // Weights are a pure ARGUMENT check (length, finite, non-negative, not
        // all zero) — cheapest and most basic, so it runs before the label
        // decode and the data sweep.
        let sw = validate_sample_weight::<F>("complement_nb", sample_weight, n_samples)?;

        // ORDER NOTE: the count sweep below needs `class_of_row`, so the label
        // decode now runs BEFORE the X validation that used to be its own pass.
        // See `MultinomialNB::fit_host` for what that does and does not change —
        // same reasoning verbatim.
        let (classes_, class_of_row, n_classes) =
            decode_classes_host::<F>("complement_nb", y_host)?;

        // feature_count_[c,j] in ONE sweep, with the CR-01 / T-11-02
        // finite-and-non-negative check fused in (sklearn's `check_non_negative`
        // parity; a negative count drives comp_sum / the log to NaN/-inf
        // silently).
        let ClassGroupedStats {
            sum: feature_count,
            class_weight: class_count_,
            first_invalid,
            ..
        } = class_grouped_stats_host::<F>(
            x_host,
            shape,
            &class_of_row,
            sw.as_deref(),
            n_classes,
            StatsRequest {
                check: HostScanCheck::NonNegative,
                sumsq: false,
                global_unweighted: false,
                env_key: WORKERS_ENV,
            },
        );
        if let Some((_, v)) = first_invalid {
            return Err(non_negative_x_error("complement_nb", "ComplementNB", v));
        }


        // --- ComplementNB weights (Pitfall 6 — DIFFERENT formula from
        //     MultinomialNB; do NOT copy it). feature_all_[j] = Σ_c count[c,j];
        //     comp_count[c,j] = feature_all_[j] + alpha − count[c,j] (sklearn folds
        //     the +alpha per-element so the row denominator already carries the
        //     alpha·n_features smoothing); logged[c,j] = log(comp_count[c,j] /
        //     Σ_j comp_count[c,j]). The stored feature_log_prob_ is sklearn's exact
        //     weights: `-logged` (default) or `logged / Σ_j logged` (norm). ---
        let alpha = self.alpha;
        let mut feature_all: Vec<f64> = vec![0.0; n_features];
        for c in 0..n_classes {
            for j in 0..n_features {
                feature_all[j] += feature_count[c * n_features + j];
            }
        }

        let mut flp: Vec<f64> = vec![0.0; n_classes * n_features];
        for c in 0..n_classes {
            // comp_count row and its sum (per-element +alpha already folded in).
            let comp: Vec<f64> = (0..n_features)
                .map(|j| feature_all[j] + alpha - feature_count[c * n_features + j])
                .collect();
            let comp_sum: f64 = comp.iter().sum();
            let logged: Vec<f64> = comp.iter().map(|&cc| (cc / comp_sum).ln()).collect();
            if self.norm {
                // Second L1 normalization: feature_log_prob_ = logged / Σ_j logged.
                let summed: f64 = logged.iter().sum();
                for j in 0..n_features {
                    flp[c * n_features + j] = logged[j] / summed;
                }
            } else {
                // feature_log_prob_ = −logged (the complement weights).
                for j in 0..n_features {
                    flp[c * n_features + j] = -logged[j];
                }
            }
        }

        // class_log_prior_ resolved as the discrete sibling (only used in the
        // single-class edge case at predict, but kept for the accessor surface).
        let class_log_prior_ = resolve_class_log_prior(
            "complement_nb",
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
            &flp.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<F>>(),
        );

        Ok(ComplementNB {
            alpha: self.alpha,
            force_alpha: self.force_alpha,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            norm: self.norm,
            classes_,
            n_features,
            feature_log_prob_: Some(flp_dev),
            class_log_prior_: Some(class_log_prior_),
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for ComplementNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = ComplementNB<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<ComplementNB<F, Fitted>, AlgoError> {
        let (n_samples, _n_features) = shape;
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "complement_nb",
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
        // once and run the same body `fit_from_host_slice` runs.
        let x_host = x.to_host(pool);
        let y_host = y.to_host(pool);
        self.fit_host(pool, &x_host, &y_host, shape, None)
    }
}

impl<F> ComplementNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Per-query-row joint log-likelihood (`n_query × n_classes`, host f64) =
    /// `X @ feature_log_prob_.T` (+ `class_log_prior_` only in the single-class
    /// edge case, per sklearn). The device matvec is `gemm` (transb=true) over the
    /// stored `(n_classes, n_features)` weights. Labels decode with `argmin` over
    /// `−jll` (D-08 — argmax over feature_log_prob_ == argmin over `−`), proba
    /// log-sum-exp-normalizes `jll` directly (sklearn `predict_proba` convention).
    fn joint_log_likelihood(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_query, n_features) = shape;
        let flp = self.feature_log_prob_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "complement_nb",
            operation: "predict (call fit first)",
        })?;
        let class_log_prior = self.class_log_prior_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "complement_nb",
            operation: "predict (call fit first)",
        })?;
        if n_query == 0 || n_features != self.n_features || x.len() != n_query * n_features {
            return Err(AlgoError::InvalidLabels {
                estimator: "complement_nb",
                reason: format!(
                    "predict geometry: got {n_query}x{n_features}, fitted n_features={}",
                    self.n_features
                ),
            });
        }
        // CR-01 / T-11-02: a negative / NaN query row is equally invalid for the
        // count model — reject it before the GEMM (sklearn rejects at predict too).
        let x_host: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        validate_non_negative_counts("complement_nb", "ComplementNB", &x_host)?;
        let n_classes = self.classes_.len();
        let raw = gemm::<F>(
            pool,
            x,
            (n_query, n_features),
            flp,
            (n_features, n_classes),
            false,
            true,
            None,
        )?;
        let raw_host = raw.to_host(pool);
        raw.release_into(pool);

        // sklearn adds class_log_prior_ only when there is a single class.
        let single = n_classes == 1;
        let mut jll = vec![0.0f64; n_query * n_classes];
        for i in 0..n_query {
            for c in 0..n_classes {
                let mut v = host_to_f64(raw_host[i * n_classes + c]);
                if single {
                    v += class_log_prior[c];
                }
                jll[i * n_classes + c] = v;
            }
        }
        Ok(jll)
    }
}

impl<F> PredictLabels<F> for ComplementNB<F, Fitted>
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
        // sklearn predicts argmax over feature_log_prob_; that equals argmin over
        // the negated jll (D-08 — the ComplementNB internal argmin convention).
        let neg: Vec<f64> = jll.iter().map(|&v| -v).collect();
        let labels = argmin_decode(&neg, &self.classes_);
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

impl<F> PredictProba<F> for ComplementNB<F, Fitted>
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

impl<F> PredictLogProba<F> for ComplementNB<F, Fitted>
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
