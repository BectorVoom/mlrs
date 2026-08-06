//! Nearest-neighbor `#[pyclass]` wrappers (PY-01/PY-02/PY-05):
//! `PyNearestNeighbors`, `PyKNeighborsClassifier`, `PyKNeighborsRegressor`.
//!
//! `NearestNeighbors` is `Fit` + `KNeighbors` (returns `(distances, indices)`,
//! the latter `i32`) — it has NO `predict`. `KNeighborsClassifier` adds the
//! weighted vote (`predict_labels` i32 + `predict_proba`) and its own
//! `kneighbors`; `KNeighborsRegressor` adds `predict` (the weighted neighbour
//! mean). All neighbor indices are `i32` at egress (D-06).
//!
//! The classifier's vote is read from the core's HOST entry points
//! (`predict_labels_host` / `predict_proba_host`) rather than the
//! `DeviceArray`-returning trait methods: the vote is `O(n_query · k)` and formed
//! host-side, so the trait spelling would upload a buffer only for the next line
//! to read it straight back.
//!
//! All three DEFER their training-matrix upload from `fit` to the first query
//! (KNN-REG-FIT). Brute-force k-NN builds nothing at `fit`, so an eager upload
//! is a copy sklearn never makes — see [`PyKNeighborsRegressor`] for the
//! rationale and the measurement. Consequently none of them uses the borrowing
//! [`mlrs_algos::typestate::Fit`]: each core estimator's `fit_owned` takes the
//! device buffer by value, which also removes the `device_copy` that the
//! borrowing form is obliged to make.

use arrow::array::ArrayRef;
use pyo3::prelude::*;

use mlrs_algos::neighbors::classifier::{prepare_labels, KNeighborsClassifier, PreparedLabels};
use mlrs_algos::neighbors::nearest::NearestNeighbors;
use mlrs_algos::neighbors::regressor::KNeighborsRegressor;
use mlrs_algos::neighbors::{Metric, Weights};
// All three estimators in this file are on the typestate surface — the legacy
// trait glob is fully removed. The lifecycle/accessor traits are
// imported under `Typestate*` aliases and called via UFCS at each migrated arm
// (the typestate module-doc warns against globbing the fit/predict/kneighbors
// method-name collisions; aliasing + UFCS resolves it).
use mlrs_algos::typestate::{KNeighbors as TypestateKNeighbors, Predict as TypestatePredict};

use mlrs_backend::device_array::DeviceArray;

use crate::errors::{algo_err_to_py, build_err_to_py, nonfinite_input_err, not_fitted};
use crate::ingress::{
    all_finite_f32, all_finite_f64, as_f32, as_f64, capsule_to_array, float_dtype, host_slice_f32,
    host_slice_f64, validated_f32, validated_f64, FloatDtype,
};

// ---------------------------------------------------------------------------
// NearestNeighbors — Fit + KNeighbors (distances + i32 indices); NO predict
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyNearestNeighbors,
    algo:  mlrs_algos::neighbors::nearest::NearestNeighbors,
    unfit: { n_neighbors: usize },
}

/// A validated training matrix that has NOT been uploaded to the device yet.
///
/// The unsupervised twin of [`PendingFit`] — see that type and
/// [`PyKNeighborsRegressor`] for the rationale, which applies verbatim: a
/// brute-force neighbour index has nothing to build at `fit`, so the upload
/// belongs to the first query.
struct NnPendingFit {
    x: ArrayRef,
    rows: usize,
    cols: usize,
    dt: FloatDtype,
}

/// sklearn-compatible `NearestNeighbors` (unsupervised neighbor index).
#[pyclass(name = "NearestNeighbors")]
pub struct PyNearestNeighbors {
    inner: AnyNearestNeighbors,
    /// WR-02: the hyperparameter lives on the WRAPPER, not in the `Unfit` enum
    /// payload. After a `fit` the enum is in a fitted (or pending) arm and that
    /// payload is gone, so a second `fit` on the same object — which is what
    /// every `clone`-and-refit path does — would silently fall back to the
    /// default.
    n_neighbors: usize,
    /// `Some` between `fit` and the first query — see [`NnPendingFit`].
    pending: Option<NnPendingFit>,
}

impl PyNearestNeighbors {
    /// Rust-callable default constructor for the smoke test. See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyNearestNeighbors::Unfit { n_neighbors: 5 },
            n_neighbors: 5,
            pending: None,
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        self.pending.is_none() && matches!(self.inner, AnyNearestNeighbors::Unfit { .. })
    }

    /// Upload a deferred `fit`'s training matrix and build the fitted core
    /// estimator. Idempotent, so every query method can call it unconditionally.
    fn materialize(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        let n_neighbors = self.n_neighbors;
        let built = py.detach(|| -> PyResult<AnyNearestNeighbors> {
            let mut pool = crate::lock_pool();
            match p.dt {
                FloatDtype::F32 => {
                    let est = NearestNeighbors::<f32>::builder()
                        .n_neighbors(n_neighbors)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let xd = DeviceArray::from_host(&mut pool, host_slice_f32(as_f32(&p.x)?)?);
                    Ok(AnyNearestNeighbors::F32(
                        est.fit_owned(&mut pool, xd, (p.rows, p.cols))
                            .map_err(algo_err_to_py)?,
                    ))
                }
                FloatDtype::F64 => {
                    let est = NearestNeighbors::<f64>::builder()
                        .n_neighbors(n_neighbors)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let xd = DeviceArray::from_host(&mut pool, host_slice_f64(as_f64(&p.x)?)?);
                    Ok(AnyNearestNeighbors::F64(
                        est.fit_owned(&mut pool, xd, (p.rows, p.cols))
                            .map_err(algo_err_to_py)?,
                    ))
                }
            }
        });
        // Restore the pending data on failure, or the estimator would report
        // `NotFittedError` for what was really an upload/geometry error.
        match built {
            Ok(f) => {
                self.inner = f;
                Ok(())
            }
            Err(e) => {
                self.pending = Some(p);
                Err(e)
            }
        }
    }
}

#[pymethods]
impl PyNearestNeighbors {
    /// `NearestNeighbors(n_neighbors=5)`.
    #[new]
    #[pyo3(signature = (n_neighbors = 5))]
    fn new(n_neighbors: usize) -> Self {
        Self {
            inner: AnyNearestNeighbors::Unfit { n_neighbors },
            n_neighbors,
            pending: None,
        }
    }

    /// Fit (validate + store the training matrix). Unsupervised — no `y`. GIL
    /// released (PY-03); f64 guarded on an f64-incapable backend (D-04).
    ///
    /// Validation only — the device upload is deferred to the first
    /// `kneighbors` (KNN-REG-FIT; see [`PyKNeighborsRegressor`] for the
    /// measurement). The NaN/inf rejection moved here from numpy's
    /// `check_array` along with it, so the shim passes `ensure_all_finite=False`
    /// and [`nonfinite_input_err`] reproduces `check_array`'s exact message.
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        if matches!(dt, FloatDtype::F64) {
            crate::capability::guard_f64()?;
        }
        let n_neighbors = self.n_neighbors;
        py.detach(|| -> PyResult<()> {
            match dt {
                FloatDtype::F32 => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    if !all_finite_f32(xh) {
                        return Err(nonfinite_input_err(xh, "float32"));
                    }
                    NearestNeighbors::<f32>::builder()
                        .n_neighbors(n_neighbors)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                }
                FloatDtype::F64 => {
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    if !all_finite_f64(xh) {
                        return Err(nonfinite_input_err(xh, "float64"));
                    }
                    NearestNeighbors::<f64>::builder()
                        .n_neighbors(n_neighbors)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                }
            }
            Ok(())
        })?;
        self.inner = AnyNearestNeighbors::Unfit { n_neighbors };
        self.pending = Some(NnPendingFit { x: xa, rows, cols, dt });
        Ok(())
    }

    /// `kneighbors(x, k)` → `(distances, indices)` each `rows × k` row-major; the
    /// distances are `f32`, the indices `i32` (D-06).
    fn kneighbors_f32(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize, k: usize) -> PyResult<(Vec<f32>, Vec<i32>)> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyNearestNeighbors::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let (d, i) = TypestateKNeighbors::kneighbors(est, &mut pool, &xd, (rows, cols), k)
                        .map_err(algo_err_to_py)?;
                    Ok((d.to_host_metered(&mut pool), i.to_host_metered(&mut pool)))
                }
                _ => Err(not_fitted("nearest_neighbors", "kneighbors (f32 path)")),
            }
        })
    }
    fn kneighbors_f64(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize, k: usize) -> PyResult<(Vec<f64>, Vec<i32>)> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyNearestNeighbors::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let (d, i) = TypestateKNeighbors::kneighbors(est, &mut pool, &xd, (rows, cols), k)
                        .map_err(algo_err_to_py)?;
                    Ok((d.to_host_metered(&mut pool), i.to_host_metered(&mut pool)))
                }
                _ => Err(not_fitted("nearest_neighbors", "kneighbors (f64 path)")),
            }
        })
    }
    /// A DEFERRED fit counts as fitted: the training data is validated and held,
    /// only its upload is outstanding.
    fn is_fitted(&self) -> bool {
        self.pending.is_some() || !matches!(self.inner, AnyNearestNeighbors::Unfit { .. })
    }
    /// The fitted float dtype — known from the pending array before the upload,
    /// because the shim reads it (via `_suffix`) to pick which `kneighbors_*` to
    /// call and so must not itself force materialization.
    fn dtype(&self) -> Option<&'static str> {
        if let Some(p) = &self.pending {
            return Some(match p.dt {
                FloatDtype::F32 => "f32",
                FloatDtype::F64 => "f64",
            });
        }
        match &self.inner {
            AnyNearestNeighbors::Unfit { .. } => None,
            AnyNearestNeighbors::F32(_) => Some("f32"),
            AnyNearestNeighbors::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// KNeighborsClassifier — Fit + KNeighbors + PredictLabels (i32) + PredictProba
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyKNeighborsClassifier,
    algo:  mlrs_algos::neighbors::classifier::KNeighborsClassifier,
    unfit: { n_neighbors: usize },
}

/// A validated training set whose MATRIX has not been uploaded yet.
///
/// The labels are NOT deferred with it. They are prepared eagerly, for two
/// reasons: the work is `O(n_train)` over a label vector rather than over the
/// `n_train x n_features` matrix, so it is not what makes `fit` expensive; and
/// sklearn's `classes_` must be readable the instant `fit` returns, so deferring
/// it would force materialization on the very next line the shim executes.
struct ClfPendingFit {
    x: ArrayRef,
    labels: PreparedLabels,
    rows: usize,
    cols: usize,
    dt: FloatDtype,
}

/// The hyperparameters `fit` rebuilds the core classifier from.
///
/// WR-02: held on the WRAPPER, not read back out of the `Unfit` enum arm — see
/// [`KnnRegParams`] for why (sklearn's `clone` + refit path would otherwise
/// silently fall back to the defaults).
#[derive(Clone, Copy)]
struct KnnClfParams {
    n_neighbors: usize,
    weights: Weights,
    metric: Metric,
}

impl Default for KnnClfParams {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            weights: Weights::Uniform,
            metric: Metric::Euclidean,
        }
    }
}

impl KnnClfParams {
    /// Build the unfit f32 core estimator, mapping a rejected hyperparameter to
    /// its Python exception. Called BOTH from `fit` (for its validation effect,
    /// discarding the result) and from `materialize` (for the estimator itself),
    /// so the two cannot disagree about which parameter sets are acceptable.
    fn build_f32(&self) -> PyResult<KNeighborsClassifier<f32>> {
        KNeighborsClassifier::<f32>::builder()
            .n_neighbors(self.n_neighbors)
            .weights(self.weights)
            .metric(self.metric)
            .build::<f32>()
            .map_err(build_err_to_py)
    }

    /// f64 twin of [`KnnClfParams::build_f32`].
    fn build_f64(&self) -> PyResult<KNeighborsClassifier<f64>> {
        KNeighborsClassifier::<f64>::builder()
            .n_neighbors(self.n_neighbors)
            .weights(self.weights)
            .metric(self.metric)
            .build::<f64>()
            .map_err(build_err_to_py)
    }
}

/// sklearn-compatible `KNeighborsClassifier` (weighted neighbor vote).
#[pyclass(name = "KNeighborsClassifier")]
pub struct PyKNeighborsClassifier {
    inner: AnyKNeighborsClassifier,
    /// WR-02: see [`PyNearestNeighbors::n_neighbors`].
    params: KnnClfParams,
    /// `Some` between `fit` and the first query — see [`ClfPendingFit`].
    pending: Option<ClfPendingFit>,
}

impl PyKNeighborsClassifier {
    /// Rust-callable default constructor for the smoke test. See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyKNeighborsClassifier::Unfit { n_neighbors: 5 },
            params: KnnClfParams::default(),
            pending: None,
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        self.pending.is_none() && matches!(self.inner, AnyKNeighborsClassifier::Unfit { .. })
    }

    /// Upload a deferred `fit`'s training matrix and build the fitted core
    /// estimator. Idempotent, so every query method can call it unconditionally.
    fn materialize(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        let params = self.params;
        let shape = (p.rows, p.cols);
        // `fit_owned` CONSUMES the labels, but a failure has to leave the
        // estimator exactly as it found it (see the restore below), so it gets a
        // clone. That is two `Vec<i32>` of length `n_train`; next to the matrix
        // upload happening in the same call it is free, and it buys a failure
        // path that needs no ownership plumbing at all.
        let labels = p.labels.clone();
        let built = py.detach(|| -> PyResult<AnyKNeighborsClassifier> {
            let mut pool = crate::lock_pool();
            match p.dt {
                FloatDtype::F32 => {
                    let est = params.build_f32()?;
                    let xd = DeviceArray::from_host(&mut pool, host_slice_f32(as_f32(&p.x)?)?);
                    Ok(AnyKNeighborsClassifier::F32(
                        est.fit_owned(&mut pool, xd, labels, shape)
                            .map_err(algo_err_to_py)?,
                    ))
                }
                FloatDtype::F64 => {
                    let est = params.build_f64()?;
                    let xd = DeviceArray::from_host(&mut pool, host_slice_f64(as_f64(&p.x)?)?);
                    Ok(AnyKNeighborsClassifier::F64(
                        est.fit_owned(&mut pool, xd, labels, shape)
                            .map_err(algo_err_to_py)?,
                    ))
                }
            }
        });
        match built {
            Ok(f) => {
                self.inner = f;
                Ok(())
            }
            Err(e) => {
                self.pending = Some(p);
                Err(e)
            }
        }
    }
}

#[pymethods]
impl PyKNeighborsClassifier {
    /// `KNeighborsClassifier(n_neighbors=5, weights='uniform',
    /// metric='minkowski', p=2)`.
    ///
    /// `weights=<callable>` / `metric=<callable>` are NOT accepted here — an
    /// arbitrary Python function cannot cross into a device kernel. The shim
    /// serves them from `kneighbors` output instead and only ever constructs this
    /// wrapper with a built-in weighting and metric.
    #[new]
    #[pyo3(signature = (n_neighbors = 5, weights = "uniform", metric = "minkowski", p = 2.0))]
    fn new(n_neighbors: usize, weights: &str, metric: &str, p: f64) -> PyResult<Self> {
        let weights = weights_from_str(weights)?;
        let metric = metric_from_str(metric, p)?;
        Ok(Self {
            inner: AnyKNeighborsClassifier::Unfit { n_neighbors },
            params: KnnClfParams {
                n_neighbors,
                weights,
                metric,
            },
            pending: None,
        })
    }

    /// Fit (validate + prepare labels + store the training matrix).
    ///
    /// Validation and label preparation only — the matrix upload is deferred to
    /// the first query (KNN-REG-FIT; see [`PyKNeighborsRegressor`] for the
    /// measurement). `y` is never uploaded AT ALL: the vote gather works on host
    /// `i32` class indices, so the old path's `y` upload existed only to be read
    /// straight back by the core's `y.to_host`.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        if matches!(dt, FloatDtype::F64) {
            crate::capability::guard_f64()?;
        }
        let params = self.params;
        let labels = py.detach(|| -> PyResult<PreparedLabels> {
            match dt {
                FloatDtype::F32 => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    if !all_finite_f32(xh) {
                        return Err(nonfinite_input_err(xh, "float32"));
                    }
                    params.build_f32()?;
                    // `prepare_labels` rejects a non-finite label itself (with
                    // the InvalidLabels wording), so no `all_finite` pass on `y`.
                    prepare_labels::<f32>(host_slice_f32(as_f32(&ya)?)?, rows)
                        .map_err(algo_err_to_py)
                }
                FloatDtype::F64 => {
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    if !all_finite_f64(xh) {
                        return Err(nonfinite_input_err(xh, "float64"));
                    }
                    params.build_f64()?;
                    prepare_labels::<f64>(host_slice_f64(as_f64(&ya)?)?, rows)
                        .map_err(algo_err_to_py)
                }
            }
        })?;
        self.inner = AnyKNeighborsClassifier::Unfit {
            n_neighbors: params.n_neighbors,
        };
        self.pending = Some(ClfPendingFit { x: xa, labels, rows, cols, dt });
        Ok(())
    }

    /// `predict(x)` → length-`rows` host `Vec<i32>` class labels (D-06).
    ///
    /// Goes through the core's HOST vote (`predict_labels_host`) rather than the
    /// `DeviceArray`-returning trait method. The caller here wants a host `Vec`,
    /// and the vote is formed host-side (see the core's module docs for why), so
    /// the trait spelling would upload the labels only to read them straight back.
    fn predict_labels(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKNeighborsClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    est.predict_labels_host(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)
                }
                AnyKNeighborsClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    est.predict_labels_host(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)
                }
                _ => Err(not_fitted("kneighbors_classifier", "predict")),
            }
        })
    }

    /// `predict_proba(x)` → `rows × n_classes` host floats (per-class weight
    /// shares). Same host-vote rationale as [`Self::predict_labels`].
    fn predict_proba_f32(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKNeighborsClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    est.predict_proba_host(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)
                }
                _ => Err(not_fitted("kneighbors_classifier", "predict_proba (f32 path)")),
            }
        })
    }
    fn predict_proba_f64(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKNeighborsClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    est.predict_proba_host(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)
                }
                _ => Err(not_fitted("kneighbors_classifier", "predict_proba (f64 path)")),
            }
        })
    }

    /// `kneighbors(x, k)` → `(distances, indices)` each `rows × k` row-major,
    /// under the FITTED metric (sklearn's `KNeighborsMixin` surface). Backs the
    /// shim's `kneighbors` / `kneighbors_graph` and its `weights=<callable>`,
    /// `metric=<callable>` and MULTI-OUTPUT vote paths.
    fn kneighbors_f32(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize, k: usize) -> PyResult<(Vec<f32>, Vec<i32>)> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKNeighborsClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let (d, i) = TypestateKNeighbors::kneighbors(est, &mut pool, &xd, (rows, cols), k)
                        .map_err(algo_err_to_py)?;
                    Ok((d.to_host_metered(&mut pool), i.to_host_metered(&mut pool)))
                }
                _ => Err(not_fitted("kneighbors_classifier", "kneighbors (f32 path)")),
            }
        })
    }
    fn kneighbors_f64(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize, k: usize) -> PyResult<(Vec<f64>, Vec<i32>)> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKNeighborsClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let (d, i) = TypestateKNeighbors::kneighbors(est, &mut pool, &xd, (rows, cols), k)
                        .map_err(algo_err_to_py)?;
                    Ok((d.to_host_metered(&mut pool), i.to_host_metered(&mut pool)))
                }
                _ => Err(not_fitted("kneighbors_classifier", "kneighbors (f64 path)")),
            }
        })
    }

    /// Training-set size (sklearn's `n_samples_fit_`). Answerable from the
    /// PENDING arrays, so reading it does not force the deferred upload.
    fn n_samples_fit(&self) -> PyResult<usize> {
        if let Some(p) = &self.pending {
            return Ok(p.rows);
        }
        match &self.inner {
            AnyKNeighborsClassifier::F32(e) => Ok(e.train_shape().0),
            AnyKNeighborsClassifier::F64(e) => Ok(e.train_shape().0),
            _ => Err(not_fitted("kneighbors_classifier", "n_samples_fit_")),
        }
    }
    /// The RESOLVED metric name (sklearn's `effective_metric_`): the alias and
    /// `minkowski`-`p` collapse [`metric_from_str`] applied, read back from the
    /// object that actually computes the distance.
    fn effective_metric(&self) -> String {
        metric_to_str(self.params.metric).to_string()
    }
    /// The RESOLVED Minkowski exponent (sklearn's `effective_metric_params_`
    /// `{'p': ...}` entry), `None` for the metrics that do not take one.
    fn effective_p(&self) -> Option<f64> {
        match self.params.metric {
            Metric::Minkowski { p } => Some(p),
            _ => None,
        }
    }

    /// Number of classes inferred at fit (errors before fit).
    ///
    /// Answered from the PENDING labels when the upload has not happened yet —
    /// they were prepared eagerly precisely so this and `classes_` need no
    /// device work (see [`ClfPendingFit`]).
    fn n_classes(&self) -> PyResult<usize> {
        if let Some(p) = &self.pending {
            return Ok(p.labels.n_classes());
        }
        match &self.inner {
            AnyKNeighborsClassifier::F32(e) => Ok(e.n_classes()),
            AnyKNeighborsClassifier::F64(e) => Ok(e.n_classes()),
            _ => Err(not_fitted("kneighbors_classifier", "n_classes")),
        }
    }
    /// The DISTINCT sorted training labels (`classes_`). The shim MUST use these
    /// rather than a fabricated `0..n_classes` range so a non-contiguous target
    /// (e.g. `{0, 2}`) round-trips through `predict` (WR-01).
    ///
    /// The shim reads this on the line after `fit` returns, so it must be
    /// answerable from the pending labels — otherwise it would force the upload
    /// immediately and the deferral would buy nothing.
    fn classes_(&self) -> PyResult<Vec<i64>> {
        if let Some(p) = &self.pending {
            return Ok(p.labels.classes().iter().map(|&c| c as i64).collect());
        }
        match &self.inner {
            AnyKNeighborsClassifier::F32(e) => {
                Ok(e.classes().iter().map(|&c| c as i64).collect())
            }
            AnyKNeighborsClassifier::F64(e) => {
                Ok(e.classes().iter().map(|&c| c as i64).collect())
            }
            _ => Err(not_fitted("kneighbors_classifier", "classes_")),
        }
    }
    /// A DEFERRED fit counts as fitted: the training data is validated and held,
    /// only its upload is outstanding.
    fn is_fitted(&self) -> bool {
        self.pending.is_some() || !matches!(self.inner, AnyKNeighborsClassifier::Unfit { .. })
    }
    /// The fitted float dtype — known from the pending array before the upload,
    /// because the shim reads it (via `_suffix`) to pick which `predict_proba_*`
    /// to call and so must not itself force materialization.
    fn dtype(&self) -> Option<&'static str> {
        if let Some(p) = &self.pending {
            return Some(match p.dt {
                FloatDtype::F32 => "f32",
                FloatDtype::F64 => "f64",
            });
        }
        match &self.inner {
            AnyKNeighborsClassifier::Unfit { .. } => None,
            AnyKNeighborsClassifier::F32(_) => Some("f32"),
            AnyKNeighborsClassifier::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// KNeighborsRegressor — Fit + Predict (continuous neighbor mean)
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyKNeighborsRegressor,
    algo:  mlrs_algos::neighbors::regressor::KNeighborsRegressor,
    unfit: { n_neighbors: usize },
}

/// Resolve sklearn's `(metric, p)` pair onto the core [`Metric`] enum.
///
/// This lives in Rust rather than the shim so `_mlrs` is self-consistent on its
/// own: the object that COMPUTES the distance is the object that reports which
/// distance it computed (`effective_metric`), and the shim's `effective_metric_`
/// is a read-back rather than a parallel reimplementation that could drift.
///
/// The `minkowski` collapse is not an optimization. `p = 1` / `p = 2` / `p = ∞`
/// route to the dedicated `manhattan_dist` / the Euclidean kernel family /
/// `chebyshev_dist` because `minkowski_dist` evaluates `F::powf`, which is
/// capability-gated at f64 on backends without f64 transcendentals and is less
/// accurate than the direct forms even where it runs — `Σ|d|^2` then `^(1/2)`
/// through `exp2`/`log2` is not `Σd²` then `sqrt`.
///
/// Unknown names are rejected here as well as in the shim: this is a public
/// `#[pyclass]` constructor, so the shim's validation is not the only way in.
/// Resolve sklearn's built-in `weights=` names onto the core [`Weights`] enum.
///
/// Shared by the classifier and the regressor so the two cannot drift on which
/// spellings they accept. A CALLABLE never reaches here — it cannot cross into a
/// device kernel, and the shim applies it host-side, constructing the wrapper
/// with the (then unused) uniform weighting.
fn weights_from_str(name: &str) -> PyResult<Weights> {
    Ok(match name {
        "uniform" => Weights::Uniform,
        "distance" => Weights::Distance,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "weights '{other}' is not supported"
            )))
        }
    })
}

fn metric_from_str(name: &str, p: f64) -> PyResult<Metric> {
    Ok(match name {
        "euclidean" | "l2" => Metric::Euclidean,
        "manhattan" | "l1" | "cityblock" => Metric::Manhattan,
        "chebyshev" | "infinity" => Metric::Chebyshev,
        "cosine" => Metric::Cosine,
        "minkowski" => {
            if p == 1.0 {
                Metric::Manhattan
            } else if p == 2.0 {
                Metric::Euclidean
            } else if p == f64::INFINITY {
                Metric::Chebyshev
            } else {
                Metric::Minkowski { p }
            }
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "metric '{other}' is not supported"
            )))
        }
    })
}

/// The sklearn name of a resolved [`Metric`] — the inverse of
/// [`metric_from_str`] up to aliasing, for `effective_metric_`.
fn metric_to_str(m: Metric) -> &'static str {
    match m {
        Metric::Euclidean => "euclidean",
        Metric::Manhattan => "manhattan",
        Metric::Chebyshev => "chebyshev",
        Metric::Cosine => "cosine",
        Metric::Minkowski { .. } => "minkowski",
    }
}

/// The hyperparameters `fit` rebuilds the core estimator from.
///
/// WR-02: held on the WRAPPER, not read back out of the `Unfit` enum arm. Once
/// `fit` has run, `self.inner` is an `F32`/`F64` arm and the `Unfit` payload is
/// gone — a second `fit` on the same object would silently fall back to the
/// defaults if it sourced its hyperparameters from there. sklearn's `clone` +
/// refit path (used by every `cross_val_score` / `GridSearchCV`) does exactly
/// that.
#[derive(Clone, Copy)]
struct KnnRegParams {
    n_neighbors: usize,
    weights: Weights,
    metric: Metric,
}

impl Default for KnnRegParams {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            weights: Weights::Uniform,
            metric: Metric::Euclidean,
        }
    }
}

impl KnnRegParams {
    /// Build the unfit f32 core estimator, mapping a rejected hyperparameter to
    /// its Python exception. Called BOTH from `fit` (for its validation effect,
    /// discarding the result) and from `materialize` (for the estimator itself),
    /// so the two cannot disagree about which parameter sets are acceptable.
    fn build_f32(&self) -> PyResult<KNeighborsRegressor<f32>> {
        KNeighborsRegressor::<f32>::builder()
            .n_neighbors(self.n_neighbors)
            .weights(self.weights)
            .metric(self.metric)
            .build::<f32>()
            .map_err(build_err_to_py)
    }

    /// f64 twin of [`KnnRegParams::build_f32`].
    fn build_f64(&self) -> PyResult<KNeighborsRegressor<f64>> {
        KNeighborsRegressor::<f64>::builder()
            .n_neighbors(self.n_neighbors)
            .weights(self.weights)
            .metric(self.metric)
            .build::<f64>()
            .map_err(build_err_to_py)
    }
}

/// Validated training data that has NOT been uploaded to the device yet
/// (KNN-REG-FIT).
///
/// The `arrow` arrays are OWNED — [`capsule_to_array`] takes the exported
/// array's release callback, so these buffers outlive the Python handles and
/// nothing reaches back into memory Python could free. Holding them costs no
/// extra memory either: the export shares the buffer with the `pyarrow` object
/// the shim already keeps.
struct PendingFit {
    x: ArrayRef,
    y: ArrayRef,
    rows: usize,
    cols: usize,
    dt: FloatDtype,
}

/// sklearn-compatible `KNeighborsRegressor` (weighted neighbor-mean regression).
///
/// ## `fit` does not touch the device (KNN-REG-FIT)
/// Brute-force k-NN has no model to solve for: `fit` validates the training set
/// and stores it, and every neighbour is computed at query time. sklearn's `fit`
/// is correspondingly a validation pass over a reference it keeps — it makes no
/// copy at all.
///
/// mlrs matches that. `fit` validates and parks the host-resident `arrow`
/// buffers in [`PendingFit`]; the device upload happens on the first query, via
/// [`PyKNeighborsRegressor::materialize`]. Uploading eagerly instead measured
/// ~4.3 ms of a 5.6 ms fit at the 200 000 x 32 f32 rung — `DeviceArray::from_host`
/// both meters a full-size allocation through the pool and makes an owned `Vec`
/// copy before handing it to CubeCL, so it costs about twice a `memcpy` — against
/// sklearn's 1.6 ms. Deferring it moves that cost onto the query path, where the
/// `O(n_query x n_train x d)` search dwarfs it (it is under 1% of a `predict` at
/// the same rung), and leaves `fit` bounded by the same NaN/inf scan sklearn is.
///
/// This changes no result. The upload is an implementation detail of the device
/// search; whether it happens in `fit` or in the first `predict` is invisible
/// except in where the time is attributed — which is why the deferral is honest
/// rather than a benchmark trick: the work is genuinely part of querying.
#[pyclass(name = "KNeighborsRegressor")]
pub struct PyKNeighborsRegressor {
    inner: AnyKNeighborsRegressor,
    params: KnnRegParams,
    /// `Some` between `fit` and the first query — see [`PendingFit`].
    pending: Option<PendingFit>,
}

impl PyKNeighborsRegressor {
    /// Rust-callable default constructor for the smoke test. See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyKNeighborsRegressor::Unfit { n_neighbors: 5 },
            params: KnnRegParams::default(),
            pending: None,
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        self.pending.is_none() && matches!(self.inner, AnyKNeighborsRegressor::Unfit { .. })
    }

    /// Upload a deferred `fit`'s training data and build the fitted core
    /// estimator — the second half of the split `fit` (see the type docs).
    ///
    /// Idempotent and a no-op once materialized, so every query method can call
    /// it unconditionally. The GIL is released for the upload, matching every
    /// other device call in this file (PY-03).
    fn materialize(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        let params = self.params;
        let fitted = py.detach(|| -> PyResult<AnyKNeighborsRegressor> {
            let mut pool = crate::lock_pool();
            match p.dt {
                FloatDtype::F32 => {
                    let est = params.build_f32()?;
                    let xd = DeviceArray::from_host(&mut pool, host_slice_f32(as_f32(&p.x)?)?);
                    let yd = DeviceArray::from_host(&mut pool, host_slice_f32(as_f32(&p.y)?)?);
                    Ok(AnyKNeighborsRegressor::F32(
                        est.fit_owned(&mut pool, xd, yd, (p.rows, p.cols))
                            .map_err(algo_err_to_py)?,
                    ))
                }
                FloatDtype::F64 => {
                    let est = params.build_f64()?;
                    let xd = DeviceArray::from_host(&mut pool, host_slice_f64(as_f64(&p.x)?)?);
                    let yd = DeviceArray::from_host(&mut pool, host_slice_f64(as_f64(&p.y)?)?);
                    Ok(AnyKNeighborsRegressor::F64(
                        est.fit_owned(&mut pool, xd, yd, (p.rows, p.cols))
                            .map_err(algo_err_to_py)?,
                    ))
                }
            }
        });
        // On failure the pending data is already taken, so the estimator would
        // silently fall back to "unfitted" and report `NotFittedError` for a
        // failure that was really an upload/geometry error. Put it back.
        match fitted {
            Ok(f) => {
                self.inner = f;
                Ok(())
            }
            Err(e) => {
                self.pending = Some(p);
                Err(e)
            }
        }
    }
}

#[pymethods]
impl PyKNeighborsRegressor {
    /// `KNeighborsRegressor(n_neighbors=5, weights='uniform',
    /// metric='minkowski', p=2)`.
    ///
    /// `weights=<callable>` is NOT accepted here — an arbitrary Python function
    /// cannot cross into a device kernel. The shim serves it from `kneighbors`
    /// output instead and only ever constructs this wrapper with a built-in
    /// weighting.
    #[new]
    #[pyo3(signature = (n_neighbors = 5, weights = "uniform", metric = "minkowski", p = 2.0))]
    fn new(n_neighbors: usize, weights: &str, metric: &str, p: f64) -> PyResult<Self> {
        let weights = weights_from_str(weights)?;
        let metric = metric_from_str(metric, p)?;
        Ok(Self {
            inner: AnyKNeighborsRegressor::Unfit { n_neighbors },
            params: KnnRegParams {
                n_neighbors,
                weights,
                metric,
            },
            pending: None,
        })
    }

    /// Fit on `(x, y)`. `y` is row-major `rows × n_outputs` — the width is
    /// inferred core-side from `y.len() / rows`, so a 1-D and a single-column
    /// 2-D target are the same call.
    ///
    /// ## What `fit` actually does (KNN-REG-FIT)
    /// Validate, and nothing else — see the type docs for why the device upload
    /// is deferred to the first query.
    ///
    /// The validation it does do is one pass per operand:
    ///
    /// * the NaN/inf rejection is [`all_finite_f32`] over the host slice rather
    ///   than a separate `check_array` pass in numpy (the shim passes
    ///   `ensure_all_finite=False` for exactly this reason, and
    ///   [`nonfinite_input_err`] reproduces `check_array`'s message so the
    ///   Python-visible contract is unchanged);
    /// * the hyperparameters are validated by building the core estimator, and
    ///   the result is DISCARDED — the built estimator is cheap and stateless
    ///   before `fit_owned`, and doing it here is what keeps a bad
    ///   `n_neighbors` / `p` an error at `fit` (where sklearn raises it) rather
    ///   than surfacing later from the first `predict`.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        if matches!(dt, FloatDtype::F64) {
            crate::capability::guard_f64()?;
        }
        let params = self.params;
        py.detach(|| -> PyResult<()> {
            match dt {
                FloatDtype::F32 => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    let yh = host_slice_f32(as_f32(&ya)?)?;
                    // X before y, matching the order sklearn's `validate_data`
                    // reports them in.
                    if !all_finite_f32(xh) {
                        return Err(nonfinite_input_err(xh, "float32"));
                    }
                    if !all_finite_f32(yh) {
                        return Err(nonfinite_input_err(yh, "float32"));
                    }
                    params.build_f32()?;
                }
                FloatDtype::F64 => {
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    let yh = host_slice_f64(as_f64(&ya)?)?;
                    if !all_finite_f64(xh) {
                        return Err(nonfinite_input_err(xh, "float64"));
                    }
                    if !all_finite_f64(yh) {
                        return Err(nonfinite_input_err(yh, "float64"));
                    }
                    params.build_f64()?;
                }
            }
            Ok(())
        })?;
        // Geometry is validated at materialization by `fit_owned`, but the two
        // cheap shape facts the shim reads back before any query (`n_outputs`,
        // `n_samples_fit`) are derivable now, so check them here too rather than
        // letting a ragged `y` reach the first `predict`.
        let y_len = ya.len();
        if rows == 0 || cols == 0 || y_len == 0 || y_len % rows != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "KNeighborsRegressor.fit: y of length {y_len} is not a whole \
                 multiple of n_samples = {rows}"
            )));
        }
        self.inner = AnyKNeighborsRegressor::Unfit {
            n_neighbors: params.n_neighbors,
        };
        self.pending = Some(PendingFit {
            x: xa,
            y: ya,
            rows,
            cols,
            dt,
        });
        Ok(())
    }

    /// `predict(x)` → `rows × n_outputs()` host floats, row-major.
    ///
    /// Takes `&mut self` because the FIRST query is what uploads the training
    /// set (see the type docs); subsequent calls find it already materialized
    /// and the extra borrow costs nothing.
    fn predict_f32(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKNeighborsRegressor::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestatePredict::predict(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kneighbors_regressor", "predict (f32 path)")),
            }
        })
    }
    fn predict_f64(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKNeighborsRegressor::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestatePredict::predict(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kneighbors_regressor", "predict (f64 path)")),
            }
        })
    }

    /// `kneighbors(x, k)` → `(distances, indices)` each `rows × k` row-major,
    /// under the FITTED metric (sklearn's `KNeighborsMixin` surface). Backs the
    /// shim's `kneighbors` / `kneighbors_graph` and its `weights=<callable>` and
    /// `metric=<callable>` paths.
    fn kneighbors_f32(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize, k: usize) -> PyResult<(Vec<f32>, Vec<i32>)> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKNeighborsRegressor::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let (d, i) = TypestateKNeighbors::kneighbors(est, &mut pool, &xd, (rows, cols), k)
                        .map_err(algo_err_to_py)?;
                    Ok((d.to_host_metered(&mut pool), i.to_host_metered(&mut pool)))
                }
                _ => Err(not_fitted("kneighbors_regressor", "kneighbors (f32 path)")),
            }
        })
    }
    fn kneighbors_f64(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize, k: usize) -> PyResult<(Vec<f64>, Vec<i32>)> {
        self.materialize(py)?;
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKNeighborsRegressor::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let (d, i) = TypestateKNeighbors::kneighbors(est, &mut pool, &xd, (rows, cols), k)
                        .map_err(algo_err_to_py)?;
                    Ok((d.to_host_metered(&mut pool), i.to_host_metered(&mut pool)))
                }
                _ => Err(not_fitted("kneighbors_regressor", "kneighbors (f64 path)")),
            }
        })
    }

    /// Target width inferred at fit (sklearn's `n_outputs_`) — the shim needs it
    /// to reshape the flat `predict` buffer.
    ///
    /// Answered from the PENDING arrays when the upload has not happened yet:
    /// it is `y.len() / n_samples`, a shape fact that needs no device work.
    /// Materializing to answer it would defeat the deferral, because the shim
    /// reads it on the path to every `predict`.
    fn n_outputs(&self) -> PyResult<usize> {
        if let Some(p) = &self.pending {
            return Ok(p.y.len() / p.rows);
        }
        match &self.inner {
            AnyKNeighborsRegressor::F32(e) => Ok(e.n_outputs()),
            AnyKNeighborsRegressor::F64(e) => Ok(e.n_outputs()),
            _ => Err(not_fitted("kneighbors_regressor", "n_outputs")),
        }
    }
    /// Training-set size (sklearn's `n_samples_fit_`). Also answerable from the
    /// pending arrays.
    fn n_samples_fit(&self) -> PyResult<usize> {
        if let Some(p) = &self.pending {
            return Ok(p.rows);
        }
        match &self.inner {
            AnyKNeighborsRegressor::F32(e) => Ok(e.train_shape().0),
            AnyKNeighborsRegressor::F64(e) => Ok(e.train_shape().0),
            _ => Err(not_fitted("kneighbors_regressor", "n_samples_fit_")),
        }
    }
    /// The RESOLVED metric name (sklearn's `effective_metric_`): the alias and
    /// `minkowski`-`p` collapse [`metric_from_str`] applied, read back from the
    /// object that actually computes the distance.
    fn effective_metric(&self) -> String {
        metric_to_str(self.params.metric).to_string()
    }
    /// The RESOLVED Minkowski exponent (sklearn's `effective_metric_params_`
    /// `{'p': ...}` entry), `None` for the metrics that do not take one.
    fn effective_p(&self) -> Option<f64> {
        match self.params.metric {
            Metric::Minkowski { p } => Some(p),
            _ => None,
        }
    }
    /// A DEFERRED fit counts as fitted: the training data is validated and held,
    /// only its upload is outstanding. Reporting `False` here would make the
    /// shim raise `NotFittedError` after a successful `fit`.
    fn is_fitted(&self) -> bool {
        self.pending.is_some() || !matches!(self.inner, AnyKNeighborsRegressor::Unfit { .. })
    }
    /// The fitted float dtype. Known from the pending arrays before the upload —
    /// the shim reads it (via `_suffix`) to pick which `predict_*` to call, so
    /// it must not itself force materialization.
    fn dtype(&self) -> Option<&'static str> {
        if let Some(p) = &self.pending {
            return Some(match p.dt {
                FloatDtype::F32 => "f32",
                FloatDtype::F64 => "f64",
            });
        }
        match &self.inner {
            AnyKNeighborsRegressor::Unfit { .. } => None,
            AnyKNeighborsRegressor::F32(_) => Some("f32"),
            AnyKNeighborsRegressor::F64(_) => Some("f64"),
        }
    }
}
