//! Nearest-neighbor `#[pyclass]` wrappers (PY-01/PY-02/PY-05):
//! `PyNearestNeighbors`, `PyKNeighborsClassifier`, `PyKNeighborsRegressor`.
//!
//! `NearestNeighbors` is `Fit` + [`KNeighbors`] (returns `(distances, indices)`,
//! the latter `i32`) — it has NO `predict`. `KNeighborsClassifier` adds
//! [`PredictLabels`] (i32 votes) + [`PredictProba`]; `KNeighborsRegressor` adds
//! [`Predict`] (continuous mean). All neighbor indices are `i32` at egress (D-06).

use pyo3::prelude::*;

use mlrs_algos::neighbors::classifier::KNeighborsClassifier;
use mlrs_algos::neighbors::nearest::NearestNeighbors;
use mlrs_algos::neighbors::regressor::KNeighborsRegressor;
// Legacy accessor traits still consumed by the not-yet-migrated regressor arm in
// this file (removed once all three estimators migrate). `Fit` is kept as the
// legacy method-call surface for the regressor arm; the migrated NearestNeighbors
// and KNeighborsClassifier arms call the typestate forms via UFCS aliases below.
use mlrs_algos::traits::{Fit, Predict};
// Typestate forms for the migrated arms; aliased so they do not collide by path
// with the legacy `traits::*` glob above (typestate module-doc warning) — called
// via UFCS at the migrated arms only.
use mlrs_algos::typestate::{
    Fit as TypestateFit, KNeighbors as TypestateKNeighbors,
    PredictLabels as TypestatePredictLabels, PredictProba as TypestatePredictProba,
};

use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

// ---------------------------------------------------------------------------
// NearestNeighbors — Fit + KNeighbors (distances + i32 indices); NO predict
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyNearestNeighbors,
    algo:  mlrs_algos::neighbors::nearest::NearestNeighbors,
    unfit: { n_neighbors: usize },
}

/// sklearn-compatible `NearestNeighbors` (unsupervised neighbor index).
#[pyclass(name = "NearestNeighbors")]
pub struct PyNearestNeighbors {
    inner: AnyNearestNeighbors,
}

impl PyNearestNeighbors {
    /// Rust-callable default constructor for the smoke test. See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self { inner: AnyNearestNeighbors::Unfit { n_neighbors: 5 } }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyNearestNeighbors::Unfit { .. })
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
        }
    }

    /// Fit (store training matrix). Unsupervised — no `y`. GIL released (PY-03);
    /// f64 guarded on an f64-incapable backend (D-04).
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let n_neighbors = match &self.inner {
            AnyNearestNeighbors::Unfit { n_neighbors } => *n_neighbors,
            _ => 5,
        };
        let fitted = py.detach(|| -> PyResult<AnyNearestNeighbors> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = NearestNeighbors::<f32>::builder()
                        .n_neighbors(n_neighbors)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyNearestNeighbors::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = NearestNeighbors::<f64>::builder()
                        .n_neighbors(n_neighbors)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyNearestNeighbors::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `kneighbors(x, k)` → `(distances, indices)` each `rows × k` row-major; the
    /// distances are `f32`, the indices `i32` (D-06).
    fn kneighbors_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize, k: usize) -> PyResult<(Vec<f32>, Vec<i32>)> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
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
    fn kneighbors_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize, k: usize) -> PyResult<(Vec<f64>, Vec<i32>)> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
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
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyNearestNeighbors::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
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

/// sklearn-compatible `KNeighborsClassifier` (majority neighbor vote).
#[pyclass(name = "KNeighborsClassifier")]
pub struct PyKNeighborsClassifier {
    inner: AnyKNeighborsClassifier,
}

impl PyKNeighborsClassifier {
    /// Rust-callable default constructor for the smoke test. See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self { inner: AnyKNeighborsClassifier::Unfit { n_neighbors: 5 } }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyKNeighborsClassifier::Unfit { .. })
    }
}

#[pymethods]
impl PyKNeighborsClassifier {
    /// `KNeighborsClassifier(n_neighbors=5)`.
    #[new]
    #[pyo3(signature = (n_neighbors = 5))]
    fn new(n_neighbors: usize) -> Self {
        Self {
            inner: AnyKNeighborsClassifier::Unfit { n_neighbors },
        }
    }

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
        let n_neighbors = match &self.inner {
            AnyKNeighborsClassifier::Unfit { n_neighbors } => *n_neighbors,
            _ => 5,
        };
        let fitted = py.detach(|| -> PyResult<AnyKNeighborsClassifier> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = KNeighborsClassifier::<f32>::builder()
                        .n_neighbors(n_neighbors)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyKNeighborsClassifier::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = KNeighborsClassifier::<f64>::builder()
                        .n_neighbors(n_neighbors)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyKNeighborsClassifier::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → length-`rows` host `Vec<i32>` class labels (D-06).
    fn predict_labels(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyKNeighborsClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                AnyKNeighborsClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kneighbors_classifier", "predict")),
            }
        })
    }

    /// `predict_proba(x)` → `rows × n_classes` host floats (neighbor-vote fractions).
    fn predict_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyKNeighborsClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kneighbors_classifier", "predict_proba (f32 path)")),
            }
        })
    }
    fn predict_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyKNeighborsClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kneighbors_classifier", "predict_proba (f64 path)")),
            }
        })
    }

    /// Number of classes inferred at fit (errors before fit).
    fn n_classes(&self) -> PyResult<usize> {
        match &self.inner {
            AnyKNeighborsClassifier::F32(e) => Ok(e.n_classes()),
            AnyKNeighborsClassifier::F64(e) => Ok(e.n_classes()),
            _ => Err(not_fitted("kneighbors_classifier", "n_classes")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyKNeighborsClassifier::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
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

crate::any_estimator! {
    any:   AnyKNeighborsRegressor,
    algo:  mlrs_algos::neighbors::regressor::KNeighborsRegressor,
    unfit: { n_neighbors: usize },
}

/// sklearn-compatible `KNeighborsRegressor` (neighbor-mean regression).
#[pyclass(name = "KNeighborsRegressor")]
pub struct PyKNeighborsRegressor {
    inner: AnyKNeighborsRegressor,
}

impl PyKNeighborsRegressor {
    /// Rust-callable default constructor for the smoke test. See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self { inner: AnyKNeighborsRegressor::Unfit { n_neighbors: 5 } }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyKNeighborsRegressor::Unfit { .. })
    }
}

#[pymethods]
impl PyKNeighborsRegressor {
    /// `KNeighborsRegressor(n_neighbors=5)`.
    #[new]
    #[pyo3(signature = (n_neighbors = 5))]
    fn new(n_neighbors: usize) -> Self {
        Self {
            inner: AnyKNeighborsRegressor::Unfit { n_neighbors },
        }
    }

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
        let n_neighbors = match &self.inner {
            AnyKNeighborsRegressor::Unfit { n_neighbors } => *n_neighbors,
            _ => 5,
        };
        let fitted = py.detach(|| -> PyResult<AnyKNeighborsRegressor> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = KNeighborsRegressor::<f32>::new(n_neighbors);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyKNeighborsRegressor::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = KNeighborsRegressor::<f64>::new(n_neighbors);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyKNeighborsRegressor::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn predict_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyKNeighborsRegressor::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.predict(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kneighbors_regressor", "predict (f32 path)")),
            }
        })
    }
    fn predict_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyKNeighborsRegressor::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.predict(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kneighbors_regressor", "predict (f64 path)")),
            }
        })
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyKNeighborsRegressor::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyKNeighborsRegressor::Unfit { .. } => None,
            AnyKNeighborsRegressor::F32(_) => Some("f32"),
            AnyKNeighborsRegressor::F64(_) => Some("f64"),
        }
    }
}
