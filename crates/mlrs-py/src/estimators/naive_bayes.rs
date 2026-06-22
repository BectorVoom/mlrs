//! Naive-Bayes `#[pyclass]` wrappers (PY-06): `PyGaussianNB`, `PyMultinomialNB`,
//! `PyBernoulliNB`, `PyComplementNB`, `PyCategoricalNB`.
//!
//! Each is the `Fit` + `PredictLabels` + `PredictProba` + `PredictLogProba`
//! surface of its `mlrs_algos::naive_bayes` estimator, dtype-dispatched (D-06)
//! through the macro-emitted `Any<Name>` enum. The five expose the full sklearn
//! surface — `fit` / `predict_labels` / `predict_proba_{f32,f64}` /
//! `predict_log_proba_{f32,f64}` / `score` — with **sklearn-mirrored hyperparameter
//! names** (D-09 — zero translation): GaussianNB carries `var_smoothing` / `priors`
//! (NO `alpha`); the four discrete variants carry `alpha` / `force_alpha` /
//! `fit_prior` / `class_prior` plus per-variant knobs (`binarize`, `norm`,
//! `min_categories`).
//!
//! Every device-compute body honors the two load-bearing contracts documented on
//! [`crate::dispatch`]:
//!
//! 1. **GIL release (PY-03).** The trait call runs inside `py.detach(|| { … })`
//!    around the poison-recovering [`crate::lock_pool`] (WR-04 — NOT
//!    `global_pool().lock().expect()`).
//! 2. **f64 guard (D-04).** On the `FloatDtype::F64` arm,
//!    [`crate::capability::guard_f64`]`()?` runs BEFORE any upload.
//!
//! Construction (TryFrom/builder) failures from a bad hyperparameter (negative
//! `alpha` / `var_smoothing`, a non-finite prior) surface as a Python `ValueError`
//! via the existing [`crate::errors::build_err_to_py`]; a fit-time data/geometry
//! failure (mismatched prior length, non-integer categorical input) via
//! [`crate::errors::algo_err_to_py`] — ZERO new mapper (T-11-05-01/04).
//!
//! `MultinomialNB` densifies sparse input at the PyO3 ingress (NB-02): the ingress
//! bridge already materializes a dense `Float32Array`/`Float64Array` from the
//! pyarrow capsule, so a sparse caller densifies before crossing into this module
//! (the densify-at-ingress precedent, PROJ-02).
//!
//! D-10 (scope exclusion): these five wrappers expose `Fit` ONLY — there is NO
//! `partial_fit` method (PY-06 scopes `partial_fit` to IncrementalPCA / MBSGD per
//! the ROADMAP success criterion). ComplementNB's argmin decode stays internal to
//! the algos layer (D-08 — no PyO3 special-case; `score`/`predict_labels` are
//! identical across the five).
//!
//! Tests live in `crates/mlrs-py/tests/` (AGENTS.md §2 — never an in-source
//! `#[cfg(test)] mod tests`).

use arrow::array::ArrayRef;
use pyo3::prelude::*;

use mlrs_algos::naive_bayes::bernoulli_nb::BernoulliNB;
use mlrs_algos::naive_bayes::categorical_nb::{CategoricalNB, MinCategories};
use mlrs_algos::naive_bayes::complement_nb::ComplementNB;
use mlrs_algos::naive_bayes::gaussian_nb::GaussianNB;
use mlrs_algos::naive_bayes::multinomial_nb::MultinomialNB;
use mlrs_algos::naive_bayes::nb_common::accuracy_score;
use mlrs_algos::traits::{Fit, PredictLabels, PredictLogProba, PredictProba};

use crate::errors::{algo_err_to_py, build_err_to_py, dtype_mismatch, not_fitted};
use crate::ingress::{
    as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype,
};

// ===========================================================================
// dtype-dispatch enums (D-06) — one per estimator.
// ===========================================================================

crate::any_estimator! {
    any:   AnyGaussianNB,
    algo:  mlrs_algos::naive_bayes::gaussian_nb::GaussianNB,
    unfit: { var_smoothing: f64, priors: Option<Vec<f64>> },
}

crate::any_estimator! {
    any:   AnyMultinomialNB,
    algo:  mlrs_algos::naive_bayes::multinomial_nb::MultinomialNB,
    unfit: {
        alpha: f64, force_alpha: bool, fit_prior: bool,
        class_prior: Option<Vec<f64>>,
    },
}

crate::any_estimator! {
    any:   AnyBernoulliNB,
    algo:  mlrs_algos::naive_bayes::bernoulli_nb::BernoulliNB,
    unfit: {
        alpha: f64, force_alpha: bool, binarize: Option<f64>,
        fit_prior: bool, class_prior: Option<Vec<f64>>,
    },
}

crate::any_estimator! {
    any:   AnyComplementNB,
    algo:  mlrs_algos::naive_bayes::complement_nb::ComplementNB,
    unfit: {
        alpha: f64, force_alpha: bool, fit_prior: bool,
        class_prior: Option<Vec<f64>>, norm: bool,
    },
}

crate::any_estimator! {
    any:   AnyCategoricalNB,
    algo:  mlrs_algos::naive_bayes::categorical_nb::CategoricalNB,
    unfit: {
        alpha: f64, force_alpha: bool, fit_prior: bool,
        class_prior: Option<Vec<f64>>, min_categories: MinCategories,
    },
}

// ===========================================================================
// Shared Python-facing predict/proba/log_proba/score/classes_/is_fitted/dtype
// surface — emitted as METHOD ITEMS inlined into each estimator's SINGLE
// `#[pymethods]` block (PyO3 forbids two `#[pymethods]` impls per pyclass without
// the `multiple-pymethods` feature; the surface is identical across all five, so
// one macro factors it). Uploads are HOISTED to a `let` binding before the trait
// call (the `&mut pool` borrow cannot be re-taken inside the same call, E0499).
// ===========================================================================

/// Emit the shared NB predict surface as **free functions** keyed by `$fns`
/// (a module-unique prefix), operating on the estimator's `&Any<Name>` enum.
/// PyO3 forbids a second `#[pymethods]` impl per pyclass (no `multiple-pymethods`
/// — v2 adds zero binding infra), and a `macro_rules!` cannot expand as items
/// inside a `#[pymethods]` block; so the device-touching logic lives in these
/// free functions and each estimator's single `#[pymethods]` block delegates via
/// the thin [`nb_thin_methods!`] wrappers. Uploads are HOISTED to a `let` binding
/// before the trait call (the `&mut pool` borrow cannot be re-taken, E0499).
macro_rules! nb_surface_fns {
    ($any:ident, $name:literal,
     $labels:ident, $pf32:ident, $pf64:ident, $lpf32:ident, $lpf64:ident,
     $score:ident, $classes:ident, $fitted:ident, $dtype:ident) => {
        fn $labels(
            inner: &$any,
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            rows: usize,
            cols: usize,
        ) -> PyResult<Vec<i32>> {
            let xa = capsule_to_array(x)?;
            py.detach(|| {
                let mut pool = crate::lock_pool();
                match inner {
                    $any::F32(est) => {
                        let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                        Ok(est
                            .predict_labels(&mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool))
                    }
                    $any::F64(est) => {
                        let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                        Ok(est
                            .predict_labels(&mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool))
                    }
                    _ => Err(not_fitted($name, "predict")),
                }
            })
        }

        fn $pf32(
            inner: &$any,
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            rows: usize,
            cols: usize,
        ) -> PyResult<Vec<f32>> {
            let xa = capsule_to_array(x)?;
            py.detach(|| {
                let mut pool = crate::lock_pool();
                match inner {
                    $any::F32(est) => {
                        let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                        Ok(est
                            .predict_proba(&mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool))
                    }
                    // WR-04: an estimator fitted as f64 is FITTED — distinguish the
                    // wrong-dtype case (dtype-mismatch error naming the fitted dtype)
                    // from the genuinely-unfit case (not_fitted).
                    $any::F64(_) => Err(dtype_mismatch($name, "f32", "f64")),
                    _ => Err(not_fitted($name, "predict_proba (f32 path)")),
                }
            })
        }
        fn $pf64(
            inner: &$any,
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            rows: usize,
            cols: usize,
        ) -> PyResult<Vec<f64>> {
            let xa = capsule_to_array(x)?;
            py.detach(|| {
                let mut pool = crate::lock_pool();
                match inner {
                    $any::F64(est) => {
                        let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                        Ok(est
                            .predict_proba(&mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool))
                    }
                    // WR-04: distinguish the wrong-dtype case from genuinely-unfit.
                    $any::F32(_) => Err(dtype_mismatch($name, "f64", "f32")),
                    _ => Err(not_fitted($name, "predict_proba (f64 path)")),
                }
            })
        }

        fn $lpf32(
            inner: &$any,
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            rows: usize,
            cols: usize,
        ) -> PyResult<Vec<f32>> {
            let xa = capsule_to_array(x)?;
            py.detach(|| {
                let mut pool = crate::lock_pool();
                match inner {
                    $any::F32(est) => {
                        let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                        Ok(est
                            .predict_log_proba(&mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool))
                    }
                    // WR-04: distinguish the wrong-dtype case from genuinely-unfit.
                    $any::F64(_) => Err(dtype_mismatch($name, "f32", "f64")),
                    _ => Err(not_fitted($name, "predict_log_proba (f32 path)")),
                }
            })
        }
        fn $lpf64(
            inner: &$any,
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            rows: usize,
            cols: usize,
        ) -> PyResult<Vec<f64>> {
            let xa = capsule_to_array(x)?;
            py.detach(|| {
                let mut pool = crate::lock_pool();
                match inner {
                    $any::F64(est) => {
                        let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                        Ok(est
                            .predict_log_proba(&mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool))
                    }
                    // WR-04: distinguish the wrong-dtype case from genuinely-unfit.
                    $any::F32(_) => Err(dtype_mismatch($name, "f64", "f32")),
                    _ => Err(not_fitted($name, "predict_log_proba (f64 path)")),
                }
            })
        }

        fn $score(
            inner: &$any,
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            y: &Bound<'_, PyAny>,
            rows: usize,
            cols: usize,
        ) -> PyResult<f64> {
            let ya = capsule_to_array(y)?;
            let pred = $labels(inner, py, x, rows, cols)?;
            let y_true = labels_as_i32(&ya)?;
            Ok(accuracy_score(&pred, &y_true))
        }

        fn $classes(inner: &$any) -> Vec<i64> {
            match inner {
                $any::F32(e) => e.classes().to_vec(),
                $any::F64(e) => e.classes().to_vec(),
                _ => Vec::new(),
            }
        }
        fn $fitted(inner: &$any) -> bool {
            !matches!(inner, $any::Unfit { .. })
        }
        fn $dtype(inner: &$any) -> Option<&'static str> {
            match inner {
                $any::Unfit { .. } => None,
                $any::F32(_) => Some("f32"),
                $any::F64(_) => Some("f64"),
            }
        }
    };
}

// Generate the shared free functions for all five estimators (module-unique
// prefixes keep them non-colliding).
nb_surface_fns!(
    AnyGaussianNB, "gaussian_nb",
    g_labels, g_pf32, g_pf64, g_lpf32, g_lpf64, g_score, g_classes, g_fitted, g_dtype
);
nb_surface_fns!(
    AnyMultinomialNB, "multinomial_nb",
    m_labels, m_pf32, m_pf64, m_lpf32, m_lpf64, m_score, m_classes, m_fitted, m_dtype
);
nb_surface_fns!(
    AnyBernoulliNB, "bernoulli_nb",
    b_labels, b_pf32, b_pf64, b_lpf32, b_lpf64, b_score, b_classes, b_fitted, b_dtype
);
nb_surface_fns!(
    AnyComplementNB, "complement_nb",
    c_labels, c_pf32, c_pf64, c_lpf32, c_lpf64, c_score, c_classes, c_fitted, c_dtype
);
nb_surface_fns!(
    AnyCategoricalNB, "categorical_nb",
    k_labels, k_pf32, k_pf64, k_lpf32, k_lpf64, k_score, k_classes, k_fitted, k_dtype
);

// ===========================================================================
// GaussianNB — Fit + PredictLabels (i32) + PredictProba + PredictLogProba.
// sklearn-named: var_smoothing / priors (NO alpha — D-09).
// ===========================================================================

/// sklearn-compatible `GaussianNB`. The sklearn-named `var_smoothing` / `priors`
/// are stored verbatim in the `Unfit` arm; the builder `build()` runs at the first
/// `fit` (a bad `var_smoothing` surfaces as a `ValueError` there, D-05/D-09).
#[pyclass(name = "GaussianNB")]
pub struct PyGaussianNB {
    inner: AnyGaussianNB,
}

impl PyGaussianNB {
    /// Rust-callable default constructor (smoke-test seam).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyGaussianNB::Unfit {
                var_smoothing: 1e-9,
                priors: None,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyGaussianNB::Unfit { .. })
    }
}

#[pymethods]
impl PyGaussianNB {
    /// `GaussianNB(var_smoothing=1e-9, priors=None)` (D-02 sklearn defaults).
    #[new]
    #[pyo3(signature = (var_smoothing = 1e-9, priors = None))]
    fn new(var_smoothing: f64, priors: Option<Vec<f64>>) -> Self {
        Self {
            inner: AnyGaussianNB::Unfit {
                var_smoothing,
                priors,
            },
        }
    }

    /// Fit on `x` (`rows × cols`, row-major) + label vector `y`. The builder
    /// validates the data-independent params (`build()` → `ValueError`, D-09)
    /// BEFORE the device launch; GIL released (PY-03); f64 guarded (D-04).
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
        let (var_smoothing, priors) = match &self.inner {
            AnyGaussianNB::Unfit {
                var_smoothing,
                priors,
            } => (*var_smoothing, priors.clone()),
            _ => return Err(not_fitted("gaussian_nb", "re-fit")),
        };
        let fitted = py.detach(|| -> PyResult<AnyGaussianNB> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = GaussianNB::<f32>::builder()
                        .var_smoothing(var_smoothing)
                        .priors(priors)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyGaussianNB::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = GaussianNB::<f64>::builder()
                        .var_smoothing(var_smoothing)
                        .priors(priors)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyGaussianNB::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn predict_labels(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        g_labels(&self.inner, py, x, rows, cols)
    }
    fn predict_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        g_pf32(&self.inner, py, x, rows, cols)
    }
    fn predict_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        g_pf64(&self.inner, py, x, rows, cols)
    }
    fn predict_log_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        g_lpf32(&self.inner, py, x, rows, cols)
    }
    fn predict_log_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        g_lpf64(&self.inner, py, x, rows, cols)
    }
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<f64> {
        g_score(&self.inner, py, x, y, rows, cols)
    }
    fn classes_(&self) -> Vec<i64> {
        g_classes(&self.inner)
    }
    fn is_fitted(&self) -> bool {
        g_fitted(&self.inner)
    }
    fn dtype(&self) -> Option<&'static str> {
        g_dtype(&self.inner)
    }
}

// ===========================================================================
// Shared host helper — materialize the i32 label vector for `score`.
// ===========================================================================

/// Materialize a pyarrow float `y` capsule to a host `Vec<i32>` (round-to-nearest)
/// for the accuracy comparison in `score`. The targets are integer class labels
/// passed as a float Arrow array (the same ingress convention `fit` uses).
fn labels_as_i32(ya: &ArrayRef) -> PyResult<Vec<i32>> {
    match float_dtype(ya)? {
        FloatDtype::F32 => Ok(as_f32(ya)?.values().iter().map(|v| v.round() as i32).collect()),
        FloatDtype::F64 => Ok(as_f64(ya)?.values().iter().map(|v| v.round() as i32).collect()),
    }
}

// ===========================================================================
// MultinomialNB — sklearn-named: alpha / force_alpha / fit_prior / class_prior.
// Densifies sparse input at ingress (NB-02).
// ===========================================================================

/// sklearn-compatible `MultinomialNB`. Sparse input is densified at the PyO3
/// ingress (NB-02) — the ingress bridge materializes a dense float Arrow array.
#[pyclass(name = "MultinomialNB")]
pub struct PyMultinomialNB {
    inner: AnyMultinomialNB,
}

impl PyMultinomialNB {
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyMultinomialNB::Unfit {
                alpha: 1.0,
                force_alpha: true,
                fit_prior: true,
                class_prior: None,
            },
        }
    }
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyMultinomialNB::Unfit { .. })
    }
}

#[pymethods]
impl PyMultinomialNB {
    /// `MultinomialNB(alpha=1.0, force_alpha=True, fit_prior=True,
    /// class_prior=None)` (D-02 sklearn defaults).
    #[new]
    #[pyo3(signature = (alpha = 1.0, force_alpha = true, fit_prior = true, class_prior = None))]
    fn new(alpha: f64, force_alpha: bool, fit_prior: bool, class_prior: Option<Vec<f64>>) -> Self {
        Self {
            inner: AnyMultinomialNB::Unfit {
                alpha,
                force_alpha,
                fit_prior,
                class_prior,
            },
        }
    }

    /// Fit on `x` (already-dense `rows × cols`, row-major — sparse densified at
    /// ingress, NB-02) + label vector `y`.
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
        let (alpha, force_alpha, fit_prior, class_prior) = match &self.inner {
            AnyMultinomialNB::Unfit {
                alpha,
                force_alpha,
                fit_prior,
                class_prior,
            } => (*alpha, *force_alpha, *fit_prior, class_prior.clone()),
            _ => return Err(not_fitted("multinomial_nb", "re-fit")),
        };
        let fitted = py.detach(|| -> PyResult<AnyMultinomialNB> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = MultinomialNB::<f32>::builder()
                        .alpha(alpha)
                        .force_alpha(force_alpha)
                        .fit_prior(fit_prior)
                        .class_prior(class_prior)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyMultinomialNB::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = MultinomialNB::<f64>::builder()
                        .alpha(alpha)
                        .force_alpha(force_alpha)
                        .fit_prior(fit_prior)
                        .class_prior(class_prior)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyMultinomialNB::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn predict_labels(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        m_labels(&self.inner, py, x, rows, cols)
    }
    fn predict_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        m_pf32(&self.inner, py, x, rows, cols)
    }
    fn predict_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        m_pf64(&self.inner, py, x, rows, cols)
    }
    fn predict_log_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        m_lpf32(&self.inner, py, x, rows, cols)
    }
    fn predict_log_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        m_lpf64(&self.inner, py, x, rows, cols)
    }
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<f64> {
        m_score(&self.inner, py, x, y, rows, cols)
    }
    fn classes_(&self) -> Vec<i64> {
        m_classes(&self.inner)
    }
    fn is_fitted(&self) -> bool {
        m_fitted(&self.inner)
    }
    fn dtype(&self) -> Option<&'static str> {
        m_dtype(&self.inner)
    }
}

// ===========================================================================
// BernoulliNB — sklearn-named: alpha / force_alpha / binarize / fit_prior /
// class_prior.
// ===========================================================================

/// sklearn-compatible `BernoulliNB`.
#[pyclass(name = "BernoulliNB")]
pub struct PyBernoulliNB {
    inner: AnyBernoulliNB,
}

impl PyBernoulliNB {
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyBernoulliNB::Unfit {
                alpha: 1.0,
                force_alpha: true,
                binarize: Some(0.0),
                fit_prior: true,
                class_prior: None,
            },
        }
    }
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyBernoulliNB::Unfit { .. })
    }
}

#[pymethods]
impl PyBernoulliNB {
    /// `BernoulliNB(alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True,
    /// class_prior=None)` (D-02 sklearn defaults).
    #[new]
    #[pyo3(signature = (
        alpha = 1.0, force_alpha = true, binarize = Some(0.0),
        fit_prior = true, class_prior = None,
    ))]
    fn new(
        alpha: f64,
        force_alpha: bool,
        binarize: Option<f64>,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
    ) -> Self {
        Self {
            inner: AnyBernoulliNB::Unfit {
                alpha,
                force_alpha,
                binarize,
                fit_prior,
                class_prior,
            },
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
        let (alpha, force_alpha, binarize, fit_prior, class_prior) = match &self.inner {
            AnyBernoulliNB::Unfit {
                alpha,
                force_alpha,
                binarize,
                fit_prior,
                class_prior,
            } => (*alpha, *force_alpha, *binarize, *fit_prior, class_prior.clone()),
            _ => return Err(not_fitted("bernoulli_nb", "re-fit")),
        };
        let fitted = py.detach(|| -> PyResult<AnyBernoulliNB> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = BernoulliNB::<f32>::builder()
                        .alpha(alpha)
                        .force_alpha(force_alpha)
                        .binarize(binarize)
                        .fit_prior(fit_prior)
                        .class_prior(class_prior)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyBernoulliNB::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = BernoulliNB::<f64>::builder()
                        .alpha(alpha)
                        .force_alpha(force_alpha)
                        .binarize(binarize)
                        .fit_prior(fit_prior)
                        .class_prior(class_prior)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyBernoulliNB::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn predict_labels(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        b_labels(&self.inner, py, x, rows, cols)
    }
    fn predict_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        b_pf32(&self.inner, py, x, rows, cols)
    }
    fn predict_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        b_pf64(&self.inner, py, x, rows, cols)
    }
    fn predict_log_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        b_lpf32(&self.inner, py, x, rows, cols)
    }
    fn predict_log_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        b_lpf64(&self.inner, py, x, rows, cols)
    }
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<f64> {
        b_score(&self.inner, py, x, y, rows, cols)
    }
    fn classes_(&self) -> Vec<i64> {
        b_classes(&self.inner)
    }
    fn is_fitted(&self) -> bool {
        b_fitted(&self.inner)
    }
    fn dtype(&self) -> Option<&'static str> {
        b_dtype(&self.inner)
    }
}

// ===========================================================================
// ComplementNB — sklearn-named: alpha / force_alpha / fit_prior / class_prior /
// norm. argmin decode stays internal (D-08).
// ===========================================================================

/// sklearn-compatible `ComplementNB`.
#[pyclass(name = "ComplementNB")]
pub struct PyComplementNB {
    inner: AnyComplementNB,
}

impl PyComplementNB {
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyComplementNB::Unfit {
                alpha: 1.0,
                force_alpha: true,
                fit_prior: true,
                class_prior: None,
                norm: false,
            },
        }
    }
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyComplementNB::Unfit { .. })
    }
}

#[pymethods]
impl PyComplementNB {
    /// `ComplementNB(alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None,
    /// norm=False)` (D-02 sklearn defaults).
    #[new]
    #[pyo3(signature = (
        alpha = 1.0, force_alpha = true, fit_prior = true,
        class_prior = None, norm = false,
    ))]
    fn new(
        alpha: f64,
        force_alpha: bool,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
        norm: bool,
    ) -> Self {
        Self {
            inner: AnyComplementNB::Unfit {
                alpha,
                force_alpha,
                fit_prior,
                class_prior,
                norm,
            },
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
        let (alpha, force_alpha, fit_prior, class_prior, norm) = match &self.inner {
            AnyComplementNB::Unfit {
                alpha,
                force_alpha,
                fit_prior,
                class_prior,
                norm,
            } => (*alpha, *force_alpha, *fit_prior, class_prior.clone(), *norm),
            _ => return Err(not_fitted("complement_nb", "re-fit")),
        };
        let fitted = py.detach(|| -> PyResult<AnyComplementNB> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = ComplementNB::<f32>::builder()
                        .alpha(alpha)
                        .force_alpha(force_alpha)
                        .fit_prior(fit_prior)
                        .class_prior(class_prior)
                        .norm(norm)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyComplementNB::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = ComplementNB::<f64>::builder()
                        .alpha(alpha)
                        .force_alpha(force_alpha)
                        .fit_prior(fit_prior)
                        .class_prior(class_prior)
                        .norm(norm)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyComplementNB::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn predict_labels(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        c_labels(&self.inner, py, x, rows, cols)
    }
    fn predict_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        c_pf32(&self.inner, py, x, rows, cols)
    }
    fn predict_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        c_pf64(&self.inner, py, x, rows, cols)
    }
    fn predict_log_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        c_lpf32(&self.inner, py, x, rows, cols)
    }
    fn predict_log_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        c_lpf64(&self.inner, py, x, rows, cols)
    }
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<f64> {
        c_score(&self.inner, py, x, y, rows, cols)
    }
    fn classes_(&self) -> Vec<i64> {
        c_classes(&self.inner)
    }
    fn is_fitted(&self) -> bool {
        c_fitted(&self.inner)
    }
    fn dtype(&self) -> Option<&'static str> {
        c_dtype(&self.inner)
    }
}

// ===========================================================================
// CategoricalNB — sklearn-named: alpha / force_alpha / fit_prior / class_prior /
// min_categories (None | int | list → MinCategories::{Infer,Uniform,PerFeature}).
// ===========================================================================

/// sklearn-compatible `CategoricalNB`.
#[pyclass(name = "CategoricalNB")]
pub struct PyCategoricalNB {
    inner: AnyCategoricalNB,
}

impl PyCategoricalNB {
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyCategoricalNB::Unfit {
                alpha: 1.0,
                force_alpha: true,
                fit_prior: true,
                class_prior: None,
                min_categories: MinCategories::Infer,
            },
        }
    }
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyCategoricalNB::Unfit { .. })
    }
}

/// Map sklearn's `min_categories=None | int | array-like` to [`MinCategories`].
/// `None` → `Infer`; a single non-negative int → `Uniform`; a list of ints →
/// `PerFeature`. Anything else is a Python `ValueError`.
fn resolve_min_categories(min_categories: Option<&Bound<'_, PyAny>>) -> PyResult<MinCategories> {
    use pyo3::exceptions::PyValueError;
    let obj = match min_categories {
        None => return Ok(MinCategories::Infer),
        Some(o) if o.is_none() => return Ok(MinCategories::Infer),
        Some(o) => o,
    };
    // A single scalar int → Uniform; a sequence → PerFeature. Try the scalar
    // first (a Python int extracts to usize), then fall back to a Vec<usize>.
    if let Ok(u) = obj.extract::<usize>() {
        return Ok(MinCategories::Uniform(u));
    }
    if let Ok(v) = obj.extract::<Vec<usize>>() {
        return Ok(MinCategories::PerFeature(v));
    }
    Err(PyValueError::new_err(
        "CategoricalNB: min_categories must be None, a non-negative int, or a \
         list of non-negative ints",
    ))
}

#[pymethods]
impl PyCategoricalNB {
    /// `CategoricalNB(alpha=1.0, force_alpha=True, fit_prior=True,
    /// class_prior=None, min_categories=None)` (D-02 sklearn defaults).
    #[new]
    #[pyo3(signature = (
        alpha = 1.0, force_alpha = true, fit_prior = true,
        class_prior = None, min_categories = None,
    ))]
    fn new(
        alpha: f64,
        force_alpha: bool,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
        min_categories: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let min_categories = resolve_min_categories(min_categories)?;
        Ok(Self {
            inner: AnyCategoricalNB::Unfit {
                alpha,
                force_alpha,
                fit_prior,
                class_prior,
                min_categories,
            },
        })
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
        let (alpha, force_alpha, fit_prior, class_prior, min_categories) = match &self.inner {
            AnyCategoricalNB::Unfit {
                alpha,
                force_alpha,
                fit_prior,
                class_prior,
                min_categories,
            } => (
                *alpha,
                *force_alpha,
                *fit_prior,
                class_prior.clone(),
                min_categories.clone(),
            ),
            _ => return Err(not_fitted("categorical_nb", "re-fit")),
        };
        let fitted = py.detach(|| -> PyResult<AnyCategoricalNB> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = CategoricalNB::<f32>::builder()
                        .alpha(alpha)
                        .force_alpha(force_alpha)
                        .fit_prior(fit_prior)
                        .class_prior(class_prior)
                        .min_categories(min_categories)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyCategoricalNB::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = CategoricalNB::<f64>::builder()
                        .alpha(alpha)
                        .force_alpha(force_alpha)
                        .fit_prior(fit_prior)
                        .class_prior(class_prior)
                        .min_categories(min_categories)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyCategoricalNB::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn predict_labels(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        k_labels(&self.inner, py, x, rows, cols)
    }
    fn predict_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        k_pf32(&self.inner, py, x, rows, cols)
    }
    fn predict_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        k_pf64(&self.inner, py, x, rows, cols)
    }
    fn predict_log_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        k_lpf32(&self.inner, py, x, rows, cols)
    }
    fn predict_log_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        k_lpf64(&self.inner, py, x, rows, cols)
    }
    fn score(&self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<f64> {
        k_score(&self.inner, py, x, y, rows, cols)
    }
    fn classes_(&self) -> Vec<i64> {
        k_classes(&self.inner)
    }
    fn is_fitted(&self) -> bool {
        k_fitted(&self.inner)
    }
    fn dtype(&self) -> Option<&'static str> {
        k_dtype(&self.inner)
    }
}
