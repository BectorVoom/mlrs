//! `MBSGDClassifier` (SGDSVM-01) — minibatch-SGD linear classifier, ≈
//! `sklearn.linear_model.SGDClassifier`.
//!
//! The struct, the [`MBSGDClassifierBuilder`] (D-01/D-03 — sklearn-default field
//! initializers), the `build() -> Result<MBSGDClassifier<F>, BuildError>`
//! validation, and the `fit`/`predict` bodies are all SHIPPED: `fit` lowers the
//! validated `SgdConfig` into the flat `SgdParams` and drives the PRIM-10
//! `sgd_solve` minibatch-SGD solver; `predict_labels`/`predict_proba` run the
//! on-device decision-margin matvec. The closest analog is `logistic.rs`
//! (classifier: `classes_` remap + `PredictLabels` + `PredictProba`); the
//! construction surface switches from `new()`/`with_opts()` to the builder (D-01).
//!
//! Tests live in `crates/mlrs-algos/tests/mbsgd_classifier_test.rs`
//! (AGENTS.md §2), never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device::Device;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::sgd::{
    sgd_solve, sgd_solve_host_slice, SgdLoss, SgdParams, SgdSchedule,
};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::linear::sgd_config::{LearningRate, Loss, Penalty, SgdConfig};
use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, PredictProba, Unfit};

/// Minibatch-SGD linear classifier (SGDSVM-01). Construct via
/// [`MBSGDClassifier::builder`], then the consuming [`Fit::fit`] (returns the
/// `Fitted`-tagged sibling) + [`PredictLabels::predict_labels`] /
/// [`PredictProba::predict_proba`]. Binary targets get a ONE-solve fit with a
/// length-`n_features` `coef_` / length-1 `intercept_`; 3+ classes get a
/// one-vs-rest fit with an `n_classes × n_features` `coef_` / length-`n_classes`
/// `intercept_` (sklearn's own `coef_` shape rule — see [`n_coef_rows`]).
/// Fitted state is device-resident (D-03); the host accessors exist ONLY on
/// `MBSGDClassifier<F, Fitted>` (the compile-time typestate replaces the old
/// runtime `NotFitted` guard, D-03).
///
/// [`n_coef_rows`]: MBSGDClassifier::n_coef_rows
pub struct MBSGDClassifier<F, S = Unfit> {
    /// Where to run the heavy phase (DEVICE-PARAM-01).
    device: Device,
    /// The lowered, validated hyperparameter bundle (D-06).
    config: SgdConfig,
    /// DISTINCT sorted class labels inferred at `fit` (Pitfall 4 — ±1 encoding
    /// for a binary fit, one-vs-rest for 3+ classes).
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Rows in `coef_`: ONE for a binary fit, `classes_.len()` for a
    /// one-vs-rest multiclass fit — sklearn's `coef_` shape rule (`(1, d)`
    /// binary, `(n_classes, d)` otherwise), stored explicitly so `predict`
    /// does not have to re-derive it from `classes_.len()` and get the binary
    /// case wrong (2 classes, 1 row).
    n_coef_rows: usize,
    /// Fitted coefficients (device-resident), `n_coef_rows × n_features`
    /// row-major, `None` until `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (device-resident), length `n_coef_rows`, `None` until
    /// `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> MBSGDClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Start building an `MBSGDClassifier` with sklearn's `SGDClassifier`
    /// defaults (D-03).
    pub fn builder() -> MBSGDClassifierBuilder {
        MBSGDClassifierBuilder::default()
    }

    /// The lowered configuration (D-06).
    pub fn config(&self) -> &SgdConfig {
        &self.config
    }
}

impl<F> MBSGDClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The lowered configuration (D-06).
    pub fn config(&self) -> &SgdConfig {
        &self.config
    }

    /// The inferred class labels (length 2 for the binary fit, `n_classes`
    /// for a one-vs-rest multiclass fit).
    pub fn classes(&self) -> &[i64] {
        &self.classes_
    }

    /// Rows in [`coef`](Self::coef): `1` for a binary fit, `n_classes` for a
    /// one-vs-rest multiclass fit (sklearn's `coef_` shape rule). The flat
    /// `coef` buffer is this many rows of `n_features`, row-major.
    pub fn n_coef_rows(&self) -> usize {
        self.n_coef_rows
    }

    /// Host copy of the fitted `coef_`, `n_coef_rows × n_features` row-major
    /// (so length `n_features` for the binary fit). `Some` by construction on
    /// the `Fitted` state, so no `NotFitted` branch is needed (the
    /// compile-time typestate replaces the runtime guard, D-03).
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on MBSGDClassifier<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_`'s FIRST entry. Kept for the
    /// binary fit, where sklearn's `intercept_` is a single value; use
    /// [`intercepts`](Self::intercepts) for the one-vs-rest vector.
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> F {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on MBSGDClassifier<F, Fitted>")
            .to_host(pool)[0]
    }

    /// Host copy of the fitted `intercept_`, length
    /// [`n_coef_rows`](Self::n_coef_rows) — one per solved sub-problem.
    pub fn intercepts(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on MBSGDClassifier<F, Fitted>")
            .to_host(pool)
    }
}

/// Builder for [`MBSGDClassifier`] (D-01). Default field initializers encode the
/// sklearn `SGDClassifier` defaults (D-03): `loss=hinge`, `penalty=l2`,
/// `alpha=1e-4`, `l1_ratio=0.15`, `max_iter=1000`, `tol=1e-3`,
/// `learning_rate=optimal`, `eta0=0.01`, `power_t=0.5`, `n_iter_no_change=5`.
#[derive(Debug, Clone, Copy)]
pub struct MBSGDClassifierBuilder {
    device: Device,
    loss: Loss,
    penalty: Penalty,
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    learning_rate: LearningRate,
    eta0: f64,
    power_t: f64,
    batch_size: usize,
    shuffle: bool,
    seed: u64,
    n_iter_no_change: usize,
}

impl Default for MBSGDClassifierBuilder {
    fn default() -> Self {
        Self {
            device: Device::Auto,
            loss: Loss::Hinge,
            penalty: Penalty::L2,
            alpha: 1e-4,
            l1_ratio: 0.15,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            learning_rate: LearningRate::Optimal,
            eta0: 0.01,
            power_t: 0.5,
            batch_size: 1,
            shuffle: true,
            seed: 0,
            n_iter_no_change: 5,
        }
    }
}

impl MBSGDClassifierBuilder {

    /// Pin the execution arm (DEVICE-PARAM-01). [`Device::Auto`] keeps the
    /// existing gate and its `MLRS_*` A/B flag; `Cpu`/`Gpu` override its PERF
    /// half only — each prim keeps its own capability checks inside.
    pub fn device(mut self, v: Device) -> Self {
        self.device = v;
        self
    }
    /// Set the loss family.
    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }
    /// Set the penalty family.
    pub fn penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }
    /// Set the penalty strength `alpha`.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    /// Set the ElasticNet mixing `l1_ratio`.
    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }
    /// Set whether to fit an intercept.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
    /// Set the epoch cap.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    /// Set the stopping tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Set the learning-rate schedule.
    pub fn learning_rate(mut self, learning_rate: LearningRate) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    /// Set the initial learning rate `eta0`.
    pub fn eta0(mut self, eta0: f64) -> Self {
        self.eta0 = eta0;
        self
    }
    /// Set the inverse-scaling exponent `power_t`.
    pub fn power_t(mut self, power_t: f64) -> Self {
        self.power_t = power_t;
        self
    }
    /// Set the minibatch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    /// Set whether to shuffle each epoch.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    /// Set the RNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Set the loss-plateau patience (`tol > 0` only): consecutive
    /// non-improving epochs before stopping (sklearn `n_iter_no_change`).
    pub fn n_iter_no_change(mut self, n_iter_no_change: usize) -> Self {
        self.n_iter_no_change = n_iter_no_change;
        self
    }

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters
    /// (D-08, T-10-03-01). The data-INDEPENDENT predicates are checked HERE,
    /// BEFORE any data is seen (the data-DEPENDENT geometry / label checks live in
    /// [`Fit::fit`], D-08):
    ///
    /// - `alpha >= 0` ([`BuildError::InvalidAlpha`]) — a negative penalty is
    ///   undefined.
    /// - `l1_ratio ∈ [0, 1]` ([`BuildError::InvalidL1Ratio`]) when the penalty is
    ///   `ElasticNet` (the mixing parameter blends L1/L2).
    /// - `eta0 > 0` ([`BuildError::InvalidEta0`]) unless the schedule is `Optimal`
    ///   (the Bottou schedule does not read `eta0`).
    /// - the loss must be valid for a CLASSIFIER ({`Hinge`, `Log`,
    ///   `SquaredHinge`}); a regression loss (`EpsilonInsensitive` /
    ///   `SquaredEpsilonInsensitive`) is [`BuildError::InvalidLossForEstimator`].
    ///
    /// On success the lowered [`SgdConfig`] is stored and the fitted state is
    /// `None`.
    pub fn build<F>(self) -> Result<MBSGDClassifier<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        // --- T-10-03-01 / ASVS V5: validate the data-INDEPENDENT hyperparameters
        //     at build() BEFORE any data is seen (D-08). ---
        if !(self.alpha >= 0.0) {
            return Err(BuildError::InvalidAlpha {
                estimator: "mbsgd_classifier",
                alpha: self.alpha,
            });
        }
        if self.penalty == Penalty::ElasticNet
            && !(self.l1_ratio >= 0.0 && self.l1_ratio <= 1.0)
        {
            return Err(BuildError::InvalidL1Ratio {
                estimator: "mbsgd_classifier",
                l1_ratio: self.l1_ratio,
            });
        }
        if self.learning_rate != LearningRate::Optimal && !(self.eta0 > 0.0) {
            return Err(BuildError::InvalidEta0 {
                estimator: "mbsgd_classifier",
                eta0: self.eta0,
            });
        }
        // WR-04: reject a non-finite `power_t` (NaN / ±inf) — it feeds the
        // `invscaling` schedule `eta0 / t^power_t` and would drive the step rate
        // to NaN/inf. A negative finite `power_t` is accepted (documented
        // divergence — it makes the rate grow with t).
        if !self.power_t.is_finite() {
            return Err(BuildError::InvalidPowerT {
                estimator: "mbsgd_classifier",
                power_t: self.power_t,
            });
        }
        match self.loss {
            Loss::Hinge | Loss::Log | Loss::SquaredHinge => {}
            other => {
                return Err(BuildError::InvalidLossForEstimator {
                    estimator: "mbsgd_classifier",
                    loss: other.name().to_string(),
                });
            }
        }
        let config = SgdConfig {
            loss: self.loss,
            penalty: self.penalty,
            alpha: self.alpha,
            l1_ratio: self.l1_ratio,
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            learning_rate: self.learning_rate,
            eta0: self.eta0,
            power_t: self.power_t,
            epsilon: 0.0,
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            seed: self.seed,
            n_iter_no_change: self.n_iter_no_change,
        };
        Ok(MBSGDClassifier {
            device: self.device,
            config,
            classes_: Vec::new(),
            n_features: 0,
            n_coef_rows: 0,
            coef_: None,
            intercept_: None,
            _state: PhantomData,
        })
    }
}

/// Derive `classes_` and the per-solve ±1 margin target(s) from a
/// host-materialized `y`.
///
/// Shared by [`Fit::fit`] and [`MBSGDClassifier::fit_from_host_slice`] so the
/// two ingress paths cannot drift on label validation. Returns
/// `(classes_, targets)`:
///
/// - a BINARY target (exactly 2 distinct classes) is ONE target vector, with
///   `classes_[1] → +1` (sklearn maps the HIGHER class to `+1`);
/// - 3+ classes are `classes_.len()` INDEPENDENT one-vs-rest target vectors,
///   `targets[c][i] = +1` iff sample `i`'s label is `classes_[c]`, else `−1` —
///   exactly `sklearn.linear_model.SGDClassifier`'s own multiclass strategy
///   (`BaseSGDClassifier._fit_multiclass`).
///
/// The binary case is deliberately NOT expressed as two OvR solves: that would
/// double the work and return a `(2, d)` `coef_` sklearn never produces (the
/// `LinearSVC` OvR precedent's exact rule).
fn prepare_labels<F>(y_host: &[F], n_samples: usize) -> Result<(Vec<i64>, Vec<Vec<F>>), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let mut raw_labels: Vec<i64> = Vec::with_capacity(n_samples);
    for &yv in y_host.iter() {
        let lf = host_to_f64(yv);
        let li = lf.round();
        if (li - lf).abs() > 1e-6 {
            return Err(AlgoError::InvalidLabels {
                estimator: "mbsgd_classifier",
                reason: format!("labels must be integers (got {lf})"),
            });
        }
        raw_labels.push(li as i64);
    }
    let mut classes_: Vec<i64> = raw_labels.clone();
    classes_.sort_unstable();
    classes_.dedup();
    if classes_.len() < 2 {
        // sklearn's own wording for the degenerate single-class fit.
        return Err(AlgoError::InvalidLabels {
            estimator: "mbsgd_classifier",
            reason: format!(
                "this solver needs samples of at least 2 classes in the data, \
                 but the data contains only {} class",
                classes_.len()
            ),
        });
    }
    // WR-02: `predict_labels` emits class ids as `i32`; a class id that fits an
    // `f64` mantissa but exceeds `i32` range would be SILENTLY TRUNCATED (`as
    // i32` wraps) into a wrong predicted label. Validate the distinct class ids
    // against `i32` range at fit so an out-of-range label is a typed error, not
    // a silent wrong prediction.
    for &cls in classes_.iter() {
        if i32::try_from(cls).is_err() {
            return Err(AlgoError::InvalidLabels {
                estimator: "mbsgd_classifier",
                reason: format!(
                    "class label {cls} does not fit in i32 \
                     (predicted labels are i32)"
                ),
            });
        }
    }
    let targets = if classes_.len() == 2 {
        vec![raw_labels
            .iter()
            .map(|&l| f64_to_host::<F>(if l == classes_[1] { 1.0 } else { -1.0 }))
            .collect()]
    } else {
        classes_
            .iter()
            .map(|&cls| {
                raw_labels
                    .iter()
                    .map(|&l| f64_to_host::<F>(if l == cls { 1.0 } else { -1.0 }))
                    .collect()
            })
            .collect()
    };
    Ok((classes_, targets))
}

impl<F> MBSGDClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// [`Fit::fit`] over HOST slices — the no-upload ingress for backends whose
    /// SGD solve runs on the host anyway ([`sgd_host_available`]).
    ///
    /// `x` is the `n × d` row-major design and `y` the length-`n` raw label
    /// vector, both borrowed from host memory (at the Python boundary, the
    /// Arrow values themselves). The fitted `coef_`/`intercept_` are still
    /// device-resident, so the returned estimator is indistinguishable from one
    /// produced by [`Fit::fit`] — only the ingress differs, and only two
    /// `d`-element uploads happen instead of three `n·d` passes.
    ///
    /// Returns [`PrimError::UnsupportedCapability`] on a backend without a host
    /// arm; callers must branch on [`sgd_host_available`] rather than treating
    /// this as a universal entry point.
    ///
    /// [`sgd_host_available`]: mlrs_backend::prims::sgd::sgd_host_available
    ///
    /// ## One-vs-rest fan-out (the multiclass speed lever)
    /// Each of the `n_coef_rows` sub-problems is an INDEPENDENT binary SGD
    /// solve over the SAME read-only `x` — no shared mutable state at all
    /// (unlike the LinearSVC L-BFGS OvR precedent, which reuses one worker
    /// pool sequentially because its objective evaluator owns a `&mut`
    /// scratch buffer). `sgd_solve_host_slice` takes plain borrowed slices, so
    /// the classes are solved on real OS threads (`std::thread::scope`,
    /// `mlrs_backend::capability::cpu_launch_units()` workers, each a
    /// disjoint row band of the output — no merge step). A binary fit stays
    /// the single synchronous call it always was; threading one class is pure
    /// overhead.
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y: &[F],
        shape: (usize, usize),
    ) -> Result<MBSGDClassifier<F, Fitted>, AlgoError>
    where
        F: Send + Sync,
    {
        let (n_samples, n_features) = shape;

        // --- The slice twin of the D-08 geometry guard: `validate_geometry`
        //     reads a DeviceArray's length, which we do not have here. ---
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

        let (classes_, targets) = prepare_labels::<F>(y, n_samples)?;
        let params = lower_config(&self.config);
        let n_coef_rows = targets.len();

        let mut coef_host = vec![f64_to_host::<F>(0.0); n_coef_rows * n_features];
        let mut intercept_host = vec![f64_to_host::<F>(0.0); n_coef_rows];

        if n_coef_rows == 1 {
            let (coef, intercept) = sgd_solve_host_slice::<F>(x, &targets[0], shape, &params)?;
            coef_host.copy_from_slice(&coef);
            intercept_host[0] = intercept;
        } else {
            let workers = mlrs_backend::capability::cpu_launch_units()
                .max(1)
                .min(n_coef_rows as u32) as usize;
            let rows_per_worker = n_coef_rows.div_ceil(workers);
            let mut worker_err: Option<PrimError> = None;
            std::thread::scope(|scope| {
                let handles: Vec<_> = coef_host
                    .chunks_mut(rows_per_worker * n_features)
                    .zip(intercept_host.chunks_mut(rows_per_worker))
                    .zip(targets.chunks(rows_per_worker))
                    .map(|((coef_band, intercept_band), target_band)| {
                        let params_ref = &params;
                        scope.spawn(move || -> Result<(), PrimError> {
                            for (i, t) in target_band.iter().enumerate() {
                                let (coef_c, intercept_c) =
                                    sgd_solve_host_slice::<F>(x, t, shape, params_ref)?;
                                coef_band[i * n_features..(i + 1) * n_features]
                                    .copy_from_slice(&coef_c);
                                intercept_band[i] = intercept_c;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                for h in handles {
                    if let Err(e) = h.join().expect("OvR worker thread panicked") {
                        worker_err = Some(e);
                    }
                }
            });
            if let Some(e) = worker_err {
                return Err(AlgoError::Prim(e));
            }
        }

        Ok(MBSGDClassifier {
            device: self.device,
            config: self.config,
            classes_,
            n_features,
            n_coef_rows,
            coef_: Some(DeviceArray::from_host(pool, &coef_host)),
            intercept_: Some(DeviceArray::from_host(pool, &intercept_host)),
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for MBSGDClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = MBSGDClassifier<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<MBSGDClassifier<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-10-03-02 / ASVS V5: data-DEPENDENT geometry guard BEFORE any
        //     launch (D-08 — the data-INDEPENDENT params were validated at
        //     build()). ---
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "mbsgd_classifier",
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

        // --- Pitfall 4: distinct-sorted classes_ (logistic.rs precedent),
        //     binary ±1 remap for the margin loss, one-vs-rest for 3+ classes. ---
        let y_host = y.to_host(pool);
        let (classes_, targets) = prepare_labels::<F>(&y_host, n_samples)?;

        // --- Lower the validated SgdConfig into the prim-local flat SgdParams
        //     (D-06; the prim cannot take the algos SgdConfig — circular
        //     dependency). The classifier never uses epsilon (regression-only). ---
        let params = lower_config(&self.config);
        let n_coef_rows = targets.len();

        // A BINARY fit stays exactly the old single-solve path — fully
        // device-resident, no host round-trip.
        if n_coef_rows == 1 {
            let yp_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &targets[0]);
            let (coef, intercept) = sgd_solve::<F>(pool, x, &yp_dev, shape, &params, self.device)?;
            yp_dev.release_into(pool);
            return Ok(MBSGDClassifier {
                device: self.device,
                config: self.config,
                classes_,
                n_features,
                n_coef_rows,
                coef_: Some(coef),
                intercept_: Some(intercept),
                _state: PhantomData,
            });
        }

        // --- One-vs-rest solves run SEQUENTIALLY here, unlike the host-slice
        //     arm: `sgd_solve` takes `&mut BufferPool`, so unlike the
        //     independent host slices there is one mutable allocator shared
        //     across every solve and it cannot be borrowed from multiple
        //     threads at once (the same constraint the LinearSVC OvR L-BFGS
        //     precedent hit). Each class still costs only ONE upload (`yp_dev`,
        //     released before the next) and one device solve. ---
        let mut coef_host: Vec<F> = Vec::with_capacity(n_coef_rows * n_features);
        let mut intercept_host: Vec<F> = Vec::with_capacity(n_coef_rows);
        for t in &targets {
            let yp_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, t);
            // Delegate to the validated PRIM-10 prim (10-02). A device failure
            // is a typed PrimError, wrapped into AlgoError::Prim via `?`
            // (never a panic across the estimator boundary — T-10-03-03).
            let (coef, intercept) = sgd_solve::<F>(pool, x, &yp_dev, shape, &params, self.device)?;
            // The ±1 target buffer is only needed during the solve (WR-07
            // re-fit buffer release).
            yp_dev.release_into(pool);
            coef_host.extend(coef.to_host(pool));
            coef.release_into(pool);
            intercept_host.push(intercept.to_host(pool)[0]);
            intercept.release_into(pool);
        }

        Ok(MBSGDClassifier {
            device: self.device,
            config: self.config,
            classes_,
            n_features,
            n_coef_rows,
            coef_: Some(DeviceArray::from_host(pool, &coef_host)),
            intercept_: Some(DeviceArray::from_host(pool, &intercept_host)),
            _state: PhantomData,
        })
    }
}

impl<F> PredictLabels<F> for MBSGDClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let (n_query, _n_features) = shape;
        let k = self.n_coef_rows;

        // The decision margin(s) per query row (X·coef + intercept).
        let margins = self.decision_margin(pool, x, shape)?;

        let mut labels: Vec<i32> = vec![0i32; n_query];
        if k == 1 {
            // sign of the margin selects the class: >= 0 → classes_[1] (the +1
            // class), else classes_[0] (the −1 class) — sklearn's
            // `decision >= 0 → +1`.
            for (r, label) in labels.iter_mut().enumerate() {
                *label = if margins[r] >= 0.0 {
                    self.classes_[1] as i32
                } else {
                    self.classes_[0] as i32
                };
            }
        } else {
            // One-vs-rest: the argmax column of each row, through `classes_`.
            // Strict-`>`, so a tie goes to the LOWEST class index — the
            // `numpy.argmax` rule sklearn's `predict` inherits.
            for (r, label) in labels.iter_mut().enumerate() {
                let row = &margins[r * k..(r + 1) * k];
                let mut best = 0usize;
                for (j, &v) in row.iter().enumerate().skip(1) {
                    if v > row[best] {
                        best = j;
                    }
                }
                *label = self.classes_[best] as i32;
            }
        }
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

impl<F> PredictProba<F> for MBSGDClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Per-class probabilities from the log-loss sigmoid `1/(1 + exp(−margin))`
    /// (sklearn's `SGDClassifier(loss="log_loss").predict_proba`, sklearn's
    /// `_predict_proba_lr`). For a non-log loss this sigmoid is NOT a
    /// calibrated probability (sklearn raises); mlrs returns the same sigmoid
    /// shape over the decision margin (the caller pins the log-loss fixture).
    ///
    /// Binary: `n_query × 2` (`[P(class₀), P(class₁)]` per row); `P(class₁) =
    /// σ(margin)`, `P(class₀) = 1 − σ(margin)`.
    ///
    /// One-vs-rest (`n_coef_rows > 1`): `n_query × n_classes`, sklearn's OvR
    /// normalization — sigmoid EVERY column, then divide each row by its own
    /// sum so the row sums to 1 (NOT a softmax: no exponentiation, just an
    /// L1 row-normalize of the independently-sigmoided margins).
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, _n_features) = shape;
        let k = self.n_coef_rows;
        let margins = self.decision_margin(pool, x, shape)?;

        // Numerically-stable logistic sigmoid σ(m) = 1/(1 + exp(−m)).
        let sigmoid = |m: f64| -> f64 {
            if m >= 0.0 {
                1.0 / (1.0 + (-m).exp())
            } else {
                let e = m.exp();
                e / (1.0 + e)
            }
        };

        let proba: Vec<F> = if k == 1 {
            let mut proba: Vec<F> = vec![F::from_int(0i64); n_query * 2];
            for (r, &m) in margins.iter().enumerate() {
                let p1 = sigmoid(m);
                proba[r * 2] = f64_to_host::<F>(1.0 - p1);
                proba[r * 2 + 1] = f64_to_host::<F>(p1);
            }
            proba
        } else {
            let mut proba: Vec<F> = vec![F::from_int(0i64); n_query * k];
            for r in 0..n_query {
                let row = &margins[r * k..(r + 1) * k];
                let mut sum = 0.0f64;
                let sig: Vec<f64> = row
                    .iter()
                    .map(|&m| {
                        let p = sigmoid(m);
                        sum += p;
                        p
                    })
                    .collect();
                // sklearn does NOT guard a zero row-sum (every entry is a
                // sigmoid output in `(0, 1)`, so `sum` is always `> 0`).
                for (j, &p) in sig.iter().enumerate() {
                    proba[r * k + j] = f64_to_host::<F>(p / sum);
                }
            }
            proba
        };
        Ok(DeviceArray::from_host(pool, &proba))
    }
}

impl<F> MBSGDClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host-materialized decision matrix `X·coef_ᵀ + intercept_`, `n_query ×
    /// n_coef_rows` row-major (length `n_query` for a binary fit — the raw
    /// signed margin), shared by `predict_labels` (sign / argmax) and
    /// `predict_proba` (sigmoid). ONE on-device GEMM computes every column at
    /// once (`coef_` stored `n_coef_rows × n_features` row-major, read
    /// TRANSPOSED via `transb`, so `n_coef_rows == 1` is the exact same call
    /// the old single-column path made — no special case needed), then the
    /// per-row intercepts are broadcast host-side (the small predict
    /// geometry; the fitted state stays device-resident until here).
    /// Validates geometry / fitted-`n_features` (ASVS V5).
    fn decision_margin(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_query, n_features) = shape;
        let k = self.n_coef_rows;

        // `coef_`/`intercept_` are `Some` by construction on the `Fitted` state
        // (the compile-time typestate replaces the old runtime `NotFitted`
        // guard, D-03).
        let coef = self
            .coef_
            .as_ref()
            .expect("coef_ is Some by construction on MBSGDClassifier<F, Fitted>");
        let intercept = self
            .intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on MBSGDClassifier<F, Fitted>");

        if n_query == 0 || n_features == 0 || x.len() != n_query * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_query,
                cols: n_features,
                len: x.len(),
            }));
        }
        if n_features != self.n_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: n_features,
                rhs: self.n_features,
            }));
        }

        let raw = gemm::<F>(
            pool,
            x,
            (n_query, n_features),
            coef,
            (n_features, k),
            false,
            true,
            None,
        )?;
        let bias = intercept.to_host(pool);
        let raw_host = raw.to_host(pool);
        raw.release_into(pool);

        Ok((0..n_query * k)
            .map(|i| host_to_f64(raw_host[i]) + host_to_f64(bias[i % k]))
            .collect())
    }
}

/// Lower a validated [`SgdConfig`] into the prim-local flat [`SgdParams`] the
/// PRIM-10 `sgd_solve` consumes (D-06; the prim cannot take the algos
/// `SgdConfig` — circular dependency, so the estimator lowers at the call site,
/// the cd_solve flat-scalar precedent). Shared by both SGD estimators.
pub(crate) fn lower_config(cfg: &SgdConfig) -> SgdParams {
    let loss = match cfg.loss {
        Loss::Hinge => SgdLoss::Hinge,
        Loss::Log => SgdLoss::Log,
        Loss::SquaredHinge => SgdLoss::SquaredHinge,
        Loss::SquaredLoss => SgdLoss::SquaredError,
        Loss::EpsilonInsensitive => SgdLoss::EpsilonInsensitive,
        Loss::SquaredEpsilonInsensitive => SgdLoss::SquaredEpsilonInsensitive,
    };
    let schedule = match cfg.learning_rate {
        LearningRate::Optimal => SgdSchedule::Optimal,
        LearningRate::InvScaling => SgdSchedule::InvScaling,
        LearningRate::Constant => SgdSchedule::Constant,
        LearningRate::Adaptive => SgdSchedule::Adaptive,
    };
    // The host applies the L1 cumulative soft-shrink only when the penalty
    // includes an L1 term (L1 or ElasticNet with l1_ratio > 0).
    let apply_l1 = match cfg.penalty {
        Penalty::L1 => true,
        Penalty::ElasticNet => true,
        Penalty::L2 => false,
    };
    // L2-only / ElasticNet lower `l1_ratio` straight through; a pure-L1 penalty is
    // the `l1_ratio = 1` case of the elastic-net shrink math the prim runs.
    let l1_ratio = match cfg.penalty {
        Penalty::L1 => 1.0,
        Penalty::L2 => 0.0,
        Penalty::ElasticNet => cfg.l1_ratio,
    };
    SgdParams {
        loss,
        schedule,
        alpha: cfg.alpha,
        l1_ratio,
        apply_l1,
        fit_intercept: cfg.fit_intercept,
        eta0: cfg.eta0,
        power_t: cfg.power_t,
        epsilon: cfg.epsilon,
        batch_size: cfg.batch_size,
        max_iter: cfg.max_iter,
        tol: cfg.tol,
        n_iter_no_change: cfg.n_iter_no_change,
    }
}
