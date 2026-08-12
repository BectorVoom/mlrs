//! `RANSACRegressor`'s `#[pyclass]` wrapper (RANSAC-01).
//!
//! Its own file rather than another 400 lines of
//! [`estimators::linear`](super::linear) because it is the only linear-family
//! wrapper that is a META-estimator: it borrows the caller's numpy
//! `RandomState` for the whole fit, and it calls BACK into Python once per
//! trial when the user supplied `is_data_valid` / `is_model_valid`. Neither is
//! a shape the shared dense-regressor helpers in that file model.
//!
//! ## The RNG is a borrow, not a copy
//! sklearn resolves `random_state` through `check_random_state` and then draws
//! from the resulting LIVE `RandomState`, so a caller who passes their own
//! generator observes it advance. The shim therefore hands in the
//! `_mlrs.NumpyRandomState` handle `mlrs.model_selection` already defines
//! (lifted from `rs.get_state()`), the fit advances it, and the shim writes the
//! advanced words back with `rs.set_state(...)`. Same mechanism, same
//! guarantee, one implementation.
//!
//! ## The two validity callbacks, and why they cross the GIL from Rust
//! sklearn's `is_data_valid(X_subset, y_subset)` and
//! `is_model_valid(model, X_subset, y_subset)` fire at most once per trial —
//! a hundred crossings against a hundred `n × d` scans, which is free. So the
//! Rust loop keeps driving and simply calls up: the wrapper takes two
//! LOW-LEVEL bridge callables with flat-list signatures and the shim adapts
//! them to sklearn's, because reshaping to numpy and building the estimator
//! object `is_model_valid` expects is Python's job, not Rust's.
//!
//! A bridge that RAISES stashes its `PyErr` and answers
//! [`RansacVerdict::Abort`]; the fit unwinds with
//! [`AlgoError::CallbackAborted`] and this wrapper re-raises the ORIGINAL
//! exception, so a user's `ValueError` reaches them as their `ValueError`.
//!
//! Tests live in `crates/mlrs-py/tests/` and
//! `crates/mlrs-algos/tests/ransac_test.rs` (AGENTS.md §2).

use std::cell::RefCell;

use arrow::array::ArrayRef;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use mlrs_algos::error::AlgoError;
use mlrs_algos::linear::ransac::{
    MinSamples, RansacCallbacks, RansacLoss, RansacModel, RansacRegressor, RansacVerdict,
};
use mlrs_algos::typestate::{Fitted as AlgoFitted, Unfit as AlgoUnfit};

use crate::egress::{f32_vec_to_pyarrow, f64_vec_to_pyarrow};
use crate::errors::{algo_err_to_py, build_err_to_py, nonfinite_input_err, not_fitted};
use crate::ingress::{
    as_f32, as_f64, capsule_to_array, float_dtype, host_slice_f32, host_slice_f64, FloatDtype,
};
use crate::model_selection::PyNumpyRandomState;

/// The ctor hyperparameters, kept OUTSIDE the fitted arms so `get_params`-style
/// round-tripping and a second `fit` both still see them (the `PyHuberRegressor`
/// precedent).
#[derive(Debug, Clone, Copy)]
struct RansacParams {
    min_samples: MinSamples,
    residual_threshold: Option<f64>,
    max_trials: usize,
    max_skips: f64,
    stop_n_inliers: f64,
    stop_score: f64,
    stop_probability: f64,
    loss: RansacLoss,
    base_fit_intercept: bool,
}

/// Dtype-dispatched fitted state. Unlike the other linear wrappers there is no
/// `Unfit { .. }` arm holding hyperparameters: they live in
/// [`RansacParams`] on the wrapper, because the estimator is rebuilt from them
/// on every `fit` anyway (the typestate `fit` consumes it).
enum AnyRansac {
    /// Never fitted.
    Unfit,
    F32(Box<RansacRegressor<f32, AlgoFitted>>),
    F64(Box<RansacRegressor<f64, AlgoFitted>>),
}

/// The fitted attributes a `#[pyclass]` getter cannot reach through the dtype
/// dispatch generically. All are `f64`/`usize`/`bool` on both arms — the
/// sub-sample solve runs in `f64` whatever the design's width.
#[derive(Debug, Clone, Default)]
struct RansacSnapshot {
    coef: Vec<f64>,
    intercept: Vec<f64>,
    inlier_mask: Vec<bool>,
    n_trials: usize,
    n_skips_no_inliers: usize,
    n_skips_invalid_data: usize,
    n_skips_invalid_model: usize,
    exceeded_max_skips: bool,
    min_samples_used: usize,
    residual_threshold_used: f64,
    n_targets: usize,
}

/// sklearn-compatible `RANSACRegressor` over a `LinearRegression` base.
#[pyclass(name = "RANSACRegressor")]
pub struct PyRANSACRegressor {
    inner: AnyRansac,
    params: RansacParams,
    snap: RansacSnapshot,
}

/// Lower the shim's three-way `min_samples` into the typed enum.
///
/// The shim forwards sklearn's value verbatim (`None`, an int, or a float);
/// mirroring sklearn's own branch, an int-valued float `>= 1` is an ABSOLUTE
/// count, not a fraction.
fn lower_min_samples(v: Option<f64>) -> MinSamples {
    match v {
        None => MinSamples::Auto,
        Some(f) if f >= 1.0 => MinSamples::Absolute(f as usize),
        Some(f) => MinSamples::Fraction(f),
    }
}

/// Build the unfit estimator from the ctor params. Monomorphized per float
/// width by the macro so the nine builder setters are written once.
macro_rules! ransac_build {
    ($float:ty, $p:expr) => {
        RansacRegressor::<$float>::builder()
            .min_samples($p.min_samples)
            .residual_threshold($p.residual_threshold)
            .max_trials($p.max_trials)
            .max_skips($p.max_skips)
            .stop_n_inliers($p.stop_n_inliers)
            .stop_score($p.stop_score)
            .stop_probability($p.stop_probability)
            .loss($p.loss)
            .base_fit_intercept($p.base_fit_intercept)
            .build::<$float>()
            .map_err(build_err_to_py)?
    };
}

/// Snapshot the fitted attributes. Written once, invoked on both float arms.
macro_rules! ransac_snapshot {
    ($fitted:expr) => {
        RansacSnapshot {
            coef: $fitted.coef().to_vec(),
            intercept: $fitted.intercept().to_vec(),
            inlier_mask: $fitted.inlier_mask().to_vec(),
            n_trials: $fitted.n_trials(),
            n_skips_no_inliers: $fitted.n_skips_no_inliers(),
            n_skips_invalid_data: $fitted.n_skips_invalid_data(),
            n_skips_invalid_model: $fitted.n_skips_invalid_model(),
            exceeded_max_skips: $fitted.exceeded_max_skips(),
            min_samples_used: $fitted.min_samples_used(),
            residual_threshold_used: $fitted.residual_threshold_used(),
            n_targets: $fitted.n_targets(),
        }
    };
}

#[pymethods]
impl PyRANSACRegressor {
    /// `RANSACRegressor(min_samples=None, residual_threshold=None,
    /// max_trials=100, max_skips=inf, stop_n_inliers=inf, stop_score=inf,
    /// stop_probability=0.99, loss="absolute_error", base_fit_intercept=True)`.
    ///
    /// `loss` is the estimator's ONE string-valued parameter; a callable `loss`
    /// never reaches here (the shim's Python trial loop owns that case).
    /// `base_fit_intercept` is the inner `LinearRegression`'s `fit_intercept` —
    /// it is not a `RANSACRegressor` parameter in sklearn, it is a parameter of
    /// the object passed as `estimator=`.
    #[new]
    #[pyo3(signature = (
        min_samples = None,
        residual_threshold = None,
        max_trials = 100,
        max_skips = f64::INFINITY,
        stop_n_inliers = f64::INFINITY,
        stop_score = f64::INFINITY,
        stop_probability = 0.99,
        loss = "absolute_error",
        base_fit_intercept = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        min_samples: Option<f64>,
        residual_threshold: Option<f64>,
        max_trials: usize,
        max_skips: f64,
        stop_n_inliers: f64,
        stop_score: f64,
        stop_probability: f64,
        loss: &str,
        base_fit_intercept: bool,
    ) -> PyResult<Self> {
        let loss = match loss {
            "absolute_error" => RansacLoss::AbsoluteError,
            "squared_error" => RansacLoss::SquaredError,
            other => {
                return Err(PyValueError::new_err(format!(
                    "The 'loss' parameter of RANSACRegressor must be a str among \
                     {{'absolute_error', 'squared_error'}} or a callable. Got {other:?} instead."
                )))
            }
        };
        Ok(Self {
            inner: AnyRansac::Unfit,
            params: RansacParams {
                min_samples: lower_min_samples(min_samples),
                residual_threshold,
                max_trials,
                max_skips,
                stop_n_inliers,
                stop_score,
                stop_probability,
                loss,
                base_fit_intercept,
            },
            snap: RansacSnapshot::default(),
        })
    }

    /// `fit(x, y, rows, cols, n_targets, rng, sample_weight=None,
    /// is_data_valid=None, is_model_valid=None)`.
    ///
    /// `x` / `y` are Arrow float arrays in the SAME dtype (`y` flattened
    /// row-major over `n_targets`); `sample_weight` is an optional length-`rows`
    /// **float64** array regardless of the design's width (the sub-sample solve
    /// runs in `f64`). `rng` is the caller's borrowed
    /// [`PyNumpyRandomState`], advanced in place.
    ///
    /// ## Why the GIL is only released when there are no callbacks
    /// With neither bridge supplied the fit touches nothing Python and runs
    /// under `py.detach`, exactly like every other estimator here. With a
    /// bridge, the loop calls INTO Python from inside itself, so the GIL is
    /// held for the fit's duration — releasing and re-acquiring it a hundred
    /// times around callbacks that are themselves Python would cost more than
    /// it returns, and the scan's own parallelism is Rust worker threads that
    /// never wanted the GIL in the first place.
    #[pyo3(signature = (
        x, y, rows, cols, n_targets, rng,
        sample_weight = None, is_data_valid = None, is_model_valid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
        n_targets: usize,
        rng: &Bound<'_, PyNumpyRandomState>,
        sample_weight: Option<&Bound<'_, PyAny>>,
        is_data_valid: Option<&Bound<'_, PyAny>>,
        is_model_valid: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let swa = sample_weight.map(capsule_to_array).transpose()?;
        let dt = float_dtype(&xa)?;
        let p = self.params;

        // The caller's MT19937 words, taken out for the fit and written back
        // afterwards — so a Python-side draw before or after this call sees the
        // stream sklearn would have left.
        let mut core_rng = rng.borrow().inner.clone();

        let sw: Option<Vec<f64>> = match swa.as_ref() {
            Some(a) => Some(host_slice_f64(as_f64(a)?)?.to_vec()),
            None => None,
        };

        // Where a raising bridge parks its real exception (module docs).
        let stash: RefCell<Option<PyErr>> = RefCell::new(None);

        let result = ransac_fit_dispatch(
            py,
            dt,
            &p,
            &xa,
            &ya,
            rows,
            cols,
            n_targets,
            sw.as_deref(),
            &mut core_rng,
            is_data_valid,
            is_model_valid,
            &stash,
        );

        // The generator advances whether or not the fit succeeded — sklearn's
        // would have too.
        rng.borrow_mut().inner = core_rng;

        match result {
            Ok((inner, snap)) => {
                self.inner = inner;
                self.snap = snap;
                Ok(())
            }
            Err(e) => {
                // A bridge that raised parked its real exception here; re-raise
                // THAT rather than the core's `CallbackAborted` placeholder.
                if let Some(orig) = stash.borrow_mut().take() {
                    return Err(orig);
                }
                Err(e)
            }
        }
    }

    /// `predict(x, rows, cols)` → an `rows·n_targets` **pyarrow** float array
    /// (D-03), row-major over `(row, target)`.
    fn predict_f32<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let xa = capsule_to_array(x)?;
        let out = match &self.inner {
            AnyRansac::F32(est) => py.detach(|| -> PyResult<Vec<f32>> {
                let xh = host_slice_f32(as_f32(&xa)?)?;
                let pred = est
                    .predict_from_host(xh, (rows, cols))
                    .map_err(algo_err_to_py)?;
                // The matvec read every element of `x`, so it already knows
                // whether the operand was finite; relaying that verdict is what
                // lets the shim pass `ensure_all_finite=False` without dropping
                // the check (the dense-regressor contract in `linear.rs`).
                if !pred.operand_finite {
                    return Err(nonfinite_input_err(xh, "float32"));
                }
                Ok(pred.values)
            })?,
            _ => return Err(not_fitted("ransac", "predict (f32 path)")),
        };
        f32_vec_to_pyarrow(py, out)
    }

    fn predict_f64<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let xa = capsule_to_array(x)?;
        let out = match &self.inner {
            AnyRansac::F64(est) => py.detach(|| -> PyResult<Vec<f64>> {
                let xh = host_slice_f64(as_f64(&xa)?)?;
                let pred = est
                    .predict_from_host(xh, (rows, cols))
                    .map_err(algo_err_to_py)?;
                // The matvec read every element of `x`, so it already knows
                // whether the operand was finite; relaying that verdict is what
                // lets the shim pass `ensure_all_finite=False` without dropping
                // the check (the dense-regressor contract in `linear.rs`).
                if !pred.operand_finite {
                    return Err(nonfinite_input_err(xh, "float64"));
                }
                Ok(pred.values)
            })?,
            _ => return Err(not_fitted("ransac", "predict (f64 path)")),
        };
        f64_vec_to_pyarrow(py, out)
    }

    /// sklearn `estimator_.coef_`, FLAT `n_targets × n_features` row-major.
    fn coef(&self) -> PyResult<Vec<f64>> {
        self.fitted_guard("coef_")?;
        Ok(self.snap.coef.clone())
    }
    /// sklearn `estimator_.intercept_`, length `n_targets`.
    fn intercept(&self) -> PyResult<Vec<f64>> {
        self.fitted_guard("intercept_")?;
        Ok(self.snap.intercept.clone())
    }
    /// sklearn `inlier_mask_`.
    fn inlier_mask(&self) -> PyResult<Vec<bool>> {
        self.fitted_guard("inlier_mask_")?;
        Ok(self.snap.inlier_mask.clone())
    }
    /// sklearn `n_trials_`.
    fn n_trials(&self) -> PyResult<usize> {
        self.fitted_guard("n_trials_")?;
        Ok(self.snap.n_trials)
    }
    /// sklearn `n_skips_no_inliers_`.
    fn n_skips_no_inliers(&self) -> PyResult<usize> {
        self.fitted_guard("n_skips_no_inliers_")?;
        Ok(self.snap.n_skips_no_inliers)
    }
    /// sklearn `n_skips_invalid_data_`.
    fn n_skips_invalid_data(&self) -> PyResult<usize> {
        self.fitted_guard("n_skips_invalid_data_")?;
        Ok(self.snap.n_skips_invalid_data)
    }
    /// sklearn `n_skips_invalid_model_`.
    fn n_skips_invalid_model(&self) -> PyResult<usize> {
        self.fitted_guard("n_skips_invalid_model_")?;
        Ok(self.snap.n_skips_invalid_model)
    }
    /// Whether the skips exceeded `max_skips` even though a consensus set WAS
    /// found — the shim turns this into sklearn's `ConvergenceWarning`.
    fn exceeded_max_skips(&self) -> bool {
        self.snap.exceeded_max_skips
    }
    /// The RESOLVED sub-sample size, for diagnosis.
    fn min_samples_used(&self) -> PyResult<usize> {
        self.fitted_guard("min_samples_used")?;
        Ok(self.snap.min_samples_used)
    }
    /// The RESOLVED inlier cut-off (the target MAD when `residual_threshold`
    /// was `None`).
    fn residual_threshold_used(&self) -> PyResult<f64> {
        self.fitted_guard("residual_threshold_used")?;
        Ok(self.snap.residual_threshold_used)
    }
    /// Target columns the fit saw.
    fn n_targets(&self) -> usize {
        self.snap.n_targets
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyRansac::Unfit)
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyRansac::Unfit => None,
            AnyRansac::F32(_) => Some("f32"),
            AnyRansac::F64(_) => Some("f64"),
        }
    }
}

impl PyRANSACRegressor {
    fn fitted_guard(&self, what: &'static str) -> PyResult<()> {
        if matches!(self.inner, AnyRansac::Unfit) {
            return Err(not_fitted("ransac", what));
        }
        Ok(())
    }
}

/// Run the fit on whichever float arm the design's dtype selects.
///
/// A free function rather than a `macro_rules!` body inside `#[pymethods]`,
/// because the two arms differ in more than a type name: the callback bridges
/// close over `&[f32]` / `&[f64]` sub-samples and each closure has to be built
/// at the concrete width.
#[allow(clippy::too_many_arguments)]
fn ransac_fit_dispatch(
    py: Python<'_>,
    dt: FloatDtype,
    p: &RansacParams,
    xa: &ArrayRef,
    ya: &ArrayRef,
    rows: usize,
    cols: usize,
    n_targets: usize,
    sw: Option<&[f64]>,
    rng: &mut mlrs_algos::model_selection::rng::NumpyRandomState,
    is_data_valid: Option<&Bound<'_, PyAny>>,
    is_model_valid: Option<&Bound<'_, PyAny>>,
    stash: &RefCell<Option<PyErr>>,
) -> PyResult<(AnyRansac, RansacSnapshot)> {
    macro_rules! arm {
        ($float:ty, $as:ident, $host:ident, $variant:ident) => {{
            let est: RansacRegressor<$float, AlgoUnfit> = ransac_build!($float, p);
            let xh = $host($as(xa)?)?;
            let yh = $host($as(ya)?)?;

            // The sub-sample crosses as a flat list of `f64`: it is
            // `min_samples x n_features` (eleven rows of ten, by default), and
            // reshaping it — plus building the estimator object sklearn's
            // `is_model_valid` expects — is the shim's job, not Rust's.
            let data_cb = |xs: &[$float], ys: &[$float], m: usize| -> RansacVerdict {
                let f = is_data_valid.expect("installed only when Some");
                verdict(
                    stash,
                    (|| {
                        let (xl, yl) = (widen_list(py, xs)?, widen_list(py, ys)?);
                        f.call1((xl, yl, m, cols, n_targets))?.is_truthy()
                    })(),
                )
            };
            let model_cb =
                |model: RansacModel<'_>, xs: &[$float], ys: &[$float], m: usize| -> RansacVerdict {
                    let f = is_model_valid.expect("installed only when Some");
                    verdict(
                        stash,
                        (|| {
                            let coef = PyList::new(py, model.coef)?;
                            let icept = PyList::new(py, model.intercept)?;
                            let (xl, yl) = (widen_list(py, xs)?, widen_list(py, ys)?);
                            f.call1((coef, icept, xl, yl, m, cols, n_targets))?
                                .is_truthy()
                        })(),
                    )
                };
            type DataFn<'f, T> = &'f dyn Fn(&[T], &[T], usize) -> RansacVerdict;
            type ModelFn<'f, T> = &'f dyn Fn(RansacModel<'_>, &[T], &[T], usize) -> RansacVerdict;
            let callbacks = RansacCallbacks::<$float> {
                is_data_valid: is_data_valid.map(|_| &data_cb as DataFn<'_, $float>),
                is_model_valid: is_model_valid.map(|_| &model_cb as ModelFn<'_, $float>),
            };
            // GIL held only when a bridge can fire (method docs).
            let fitted = if is_data_valid.is_some() || is_model_valid.is_some() {
                est.fit_from_host_slice(xh, yh, (rows, cols), n_targets, sw, rng, &callbacks)
            } else {
                py.detach(|| {
                    est.fit_from_host_slice(
                        xh,
                        yh,
                        (rows, cols),
                        n_targets,
                        sw,
                        rng,
                        &RansacCallbacks::default(),
                    )
                })
            }
            .map_err(ransac_err_to_py)?;
            let snap = ransac_snapshot!(fitted);
            Ok((AnyRansac::$variant(Box::new(fitted)), snap))
        }};
    }

    match dt {
        FloatDtype::F32 => arm!(f32, as_f32, host_slice_f32, F32),
        FloatDtype::F64 => {
            crate::capability::guard_f64()?;
            arm!(f64, as_f64, host_slice_f64, F64)
        }
    }
}

/// A flat sub-sample slice as a Python list of `f64`.
///
/// `F` is opaque arithmetic-wise at this layer, so the widening goes through
/// the byte view — the `linear_predict_host` idiom, spelled at the two concrete
/// widths the PyO3 surface ever instantiates.
fn widen_list<'py, T: bytemuck::Pod>(py: Python<'py>, xs: &[T]) -> PyResult<Bound<'py, PyList>> {
    match size_of::<T>() {
        4 => PyList::new(
            py,
            bytemuck::cast_slice::<T, f32>(xs).iter().map(|&v| v as f64),
        ),
        8 => PyList::new(py, bytemuck::cast_slice::<T, f64>(xs).iter().copied()),
        other => unreachable!("ransac is f32/f64 only, got a {other}-byte element"),
    }
}

/// Turn a bridge callable's answer into a verdict, parking any exception in
/// `stash` so the wrapper can re-raise the ORIGINAL error after the fit unwinds
/// (module docs).
fn verdict(stash: &RefCell<Option<PyErr>>, outcome: PyResult<bool>) -> RansacVerdict {
    match outcome {
        Ok(true) => RansacVerdict::Valid,
        Ok(false) => RansacVerdict::Invalid,
        Err(e) => {
            *stash.borrow_mut() = Some(e);
            RansacVerdict::Abort
        }
    }
}

/// [`algo_err_to_py`] with sklearn's two verbatim "no consensus set" messages
/// substituted, so a caller who greps for sklearn's wording still finds it.
fn ransac_err_to_py(e: AlgoError) -> PyErr {
    match &e {
        AlgoError::MinSamplesExceedsNSamples { n_samples, .. } => PyValueError::new_err(format!(
            "`min_samples` may not be larger than number of samples: \
             n_samples = {n_samples}."
        )),
        AlgoError::NoValidConsensusSet { skipped_out, .. } => {
            if *skipped_out {
                PyValueError::new_err(
                    "RANSAC skipped more iterations than `max_skips` without \
                     finding a valid consensus set. Iterations were skipped \
                     because each randomly chosen sub-sample failed the \
                     passing criteria. See estimator attributes for \
                     diagnostics (n_skips*).",
                )
            } else {
                PyValueError::new_err(
                    "RANSAC could not find a valid consensus set. All \
                     `max_trials` iterations were skipped because each \
                     randomly chosen sub-sample failed the passing criteria. \
                     See estimator attributes for diagnostics (n_skips*).",
                )
            }
        }
        _ => algo_err_to_py(e),
    }
}
