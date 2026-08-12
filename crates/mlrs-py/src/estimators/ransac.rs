//! `RANSACRegressor`'s `#[pyclass]` wrapper (RANSAC-01 / RANSAC-02).
//!
//! Its own file rather than another 400 lines of
//! [`estimators::linear`](super::linear) because it is the only linear-family
//! wrapper that is a META-estimator: it borrows the caller's numpy
//! `RandomState` for the whole fit, and it calls BACK into Python during the
//! trial loop — for the two validity predicates, and (since RANSAC-02) for the
//! base estimator itself when that base is not a `LinearRegression`. Neither is
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
//! ## ONE Python object, one call per trial (RANSAC-02)
//! Everything Python-side is reached through a single `bridge` object the shim
//! builds per fit, with three methods and four flags. That shape exists to make
//! the per-trial crossing as cheap as it can be:
//!
//! | method | when | what crosses |
//! |---|---|---|
//! | `data_valid(idx)` | `is_data_valid` installed | one `int32` arrow array of `min_samples` |
//! | `model_valid(idx, coef, icept)` | `is_model_valid` installed | that, plus `t·d + t` floats |
//! | `run_trial(idx)` | base is not a `LinearRegression` | that, and `n·t` predictions back |
//! | `refit(idx)` | once, at the end | the consensus indices |
//!
//! Two things are deliberate here. The sub-sample crosses as INDICES, not as
//! values: the shim holds `X` already, so `X[idx]` is one numpy fancy index,
//! where the previous shape materialized `min_samples × n_features` boxed Python
//! floats per trial. And the predictions come back as a **pyarrow** array, whose
//! buffer this side reads without a per-element conversion — the egress-list
//! pathology `egress.rs` documents, avoided on the ingress direction too.
//!
//! What is NOT done is batching the foreign base's trials into one call. It
//! would cut the crossings further, and it is rejected on purpose: a batch is
//! SPECULATIVE (the loop may stop before reaching its later trials), and
//! speculating means running a user's `fit` — and any side effect it has — for a
//! trial that never happens. The native base speculates freely because there it
//! is pure arithmetic; see `mlrs_algos::linear::ransac`'s module docs.
//!
//! A bridge call that RAISES stashes its `PyErr` and answers
//! [`RansacVerdict::Abort`] / `Err(())`; the fit unwinds with
//! [`AlgoError::CallbackAborted`] and this wrapper re-raises the ORIGINAL
//! exception, so a user's `ValueError` reaches them as their `ValueError`.
//!
//! Tests live in `crates/mlrs-py/tests/` and
//! `crates/mlrs-algos/tests/ransac_test.rs` (AGENTS.md §2).

use std::cell::RefCell;

use arrow::array::{Array, ArrayRef, UInt32Array};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mlrs_algos::error::AlgoError;
use mlrs_algos::linear::ransac::{
    MinSamples, RansacBase, RansacCallbacks, RansacDriver, RansacLoss, RansacModel,
    RansacRegressor, RansacTrialBridge, RansacVerdict, TrialStatus,
};
use mlrs_algos::model_selection::rng::NumpyRandomState;
use mlrs_algos::typestate::{Fitted as AlgoFitted, Unfit as AlgoUnfit};
use mlrs_backend::device::Device;

use crate::egress::{f32_vec_to_pyarrow, f64_vec_to_pyarrow, i32_vec_to_pyarrow, u32_vec_to_pyarrow};
use crate::errors::{algo_err_to_py, build_err_to_py, nonfinite_input_err, not_fitted};
use crate::ingress::{
    as_f32, as_f64, capsule_to_array, float_dtype, host_slice_f32, host_slice_f64, FloatDtype,
};
use crate::model_selection::PyNumpyRandomState;

/// `bridge.run_trial` status codes, mirroring
/// [`TrialStatus`](mlrs_algos::linear::ransac::TrialStatus). Named here because
/// they are a wire format shared with `mlrs/linear.py`, which spells the same
/// three constants.
const STATUS_FITTED: u8 = 0;
const STATUS_INVALID_DATA: u8 = 1;
const STATUS_INVALID_MODEL: u8 = 2;

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
    device: Device,
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
    device_used: &'static str,
    batch_width: usize,
    has_model: bool,
}

/// sklearn-compatible `RANSACRegressor` over ANY base regressor.
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
/// width by the macro so the ten builder setters are written once.
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
            .device($p.device)
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
            device_used: $fitted.device_arm(),
            batch_width: $fitted.batch_width(),
            has_model: $fitted.has_linear_model(),
        }
    };
}

#[pymethods]
impl PyRANSACRegressor {
    /// `RANSACRegressor(min_samples=None, residual_threshold=None,
    /// max_trials=100, max_skips=inf, stop_n_inliers=inf, stop_score=inf,
    /// stop_probability=0.99, loss="absolute_error", base_fit_intercept=True,
    /// device="auto")`.
    ///
    /// `loss` is the estimator's ONE string-valued sklearn parameter; the
    /// CALLABLE spelling never reaches here as a string — the shim's bridge
    /// evaluates it and hands the residual over per trial.
    /// `base_fit_intercept` is the inner `LinearRegression`'s `fit_intercept` —
    /// it is not a `RANSACRegressor` parameter in sklearn, it is a parameter of
    /// the object passed as `estimator=`, and it is ignored entirely when that
    /// object is driven through the bridge.
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
        device = "auto",
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
        device: &str,
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
        let device = Device::from_name(device).ok_or_else(|| {
            PyValueError::new_err(format!(
                "RANSACRegressor: device={device:?} is not one of ('auto', 'cpu', 'gpu')."
            ))
        })?;
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
                device,
            },
            snap: RansacSnapshot::default(),
        })
    }

    /// `fit(x, y, rows, cols, n_targets, rng, sample_weight=None, bridge=None)`.
    ///
    /// `x` / `y` are Arrow float arrays in the SAME dtype (`y` flattened
    /// row-major over `n_targets`); `sample_weight` is an optional length-`rows`
    /// **float64** array regardless of the design's width (the sub-sample solve
    /// runs in `f64`). `rng` is the caller's borrowed
    /// [`PyNumpyRandomState`], advanced in place.
    ///
    /// `bridge` is `None` for the pure-native fit — sklearn's default
    /// `LinearRegression` base with neither validity predicate — and otherwise
    /// the shim's per-fit bridge object (module docs), whose four flags say
    /// which of its methods the loop should call.
    ///
    /// ## Why the GIL is only released when there is no bridge
    /// With no bridge the fit touches nothing Python and runs under `py.detach`,
    /// exactly like every other estimator here. With one, the loop calls INTO
    /// Python from inside itself, so the GIL is held for the fit's duration —
    /// releasing and re-acquiring it around calls that are themselves Python
    /// would cost more than it returns, and the scan's own parallelism is Rust
    /// worker threads that never wanted the GIL in the first place.
    #[pyo3(signature = (
        x, y, rows, cols, n_targets, rng, sample_weight = None, bridge = None,
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
        bridge: Option<&Bound<'_, PyAny>>,
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
        let flags = match bridge {
            Some(b) => BridgeFlags::read(b)?,
            None => BridgeFlags::default(),
        };

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
            bridge,
            flags,
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
    ///
    /// Only reachable on the native base: a foreign `estimator_` predicts
    /// through its own Python object, and the shim delegates there instead of
    /// calling this.
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
    /// Empty on the foreign base arm, where `estimator_` is the caller's object.
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
    /// The scan arm that ACTUALLY ran — `MlrsBase.device_`'s contract.
    fn device_used(&self) -> PyResult<&'static str> {
        self.fitted_guard("device_")?;
        Ok(self.snap.device_used)
    }
    /// Trials scanned per launch (`1` on every host fit). Read by the perf
    /// probes to tell an honoured `device='gpu'` from an unhonoured one.
    fn batch_width(&self) -> PyResult<usize> {
        self.fitted_guard("batch_width")?;
        Ok(self.snap.batch_width)
    }
    /// Whether the fit produced a native linear model (false when the base
    /// estimator was driven through the bridge).
    fn has_model(&self) -> bool {
        self.snap.has_model
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

/// Which of the bridge's methods this fit should call, read ONCE from the
/// object's flags rather than probed per trial.
#[derive(Debug, Clone, Copy, Default)]
struct BridgeFlags {
    /// The base estimator is not one this crate hosts — drive it through
    /// `run_trial`.
    foreign: bool,
    /// `is_data_valid` was supplied.
    data_valid: bool,
    /// `is_model_valid` was supplied.
    model_valid: bool,
    /// `loss` is a CALLABLE, so `run_trial` also returns the residual.
    supplies_residual: bool,
    /// The base estimator accepted `set_params(random_state=...)`, so it may
    /// DRAW from the shared generator and the state has to round-trip through
    /// every bridge call ([`RansacTrialBridge::run_trial`]).
    seeded: bool,
}

impl BridgeFlags {
    fn read(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            foreign: obj.getattr("foreign")?.extract()?,
            data_valid: obj.getattr("has_data_valid")?.extract()?,
            model_valid: obj.getattr("has_model_valid")?.extract()?,
            supplies_residual: obj.getattr("supplies_residual")?.extract()?,
            seeded: obj.getattr("seeded")?.extract()?,
        })
    }
}

/// The Python side of [`RansacTrialBridge`] — one call per trial, no
/// speculation (module docs).
struct PyTrialBridge<'py, 'a> {
    py: Python<'py>,
    obj: &'a Bound<'py, PyAny>,
    stash: &'a RefCell<Option<PyErr>>,
    supplies_residual: bool,
    seeded: bool,
    /// The design's geometry, so a bridge that returns the wrong SHAPE is a
    /// `ValueError` here rather than an out-of-range read inside the scan
    /// (ASVS V5 — the length is checked, never trusted).
    n: usize,
    t: usize,
}

impl PyTrialBridge<'_, '_> {
    /// Park a raising bridge's real exception so the wrapper can re-raise it
    /// after the fit unwinds, and answer the core's foreign-free error.
    fn park<T>(&self, outcome: PyResult<T>) -> Result<T, ()> {
        match outcome {
            Ok(v) => Ok(v),
            Err(e) => {
                *self.stash.borrow_mut() = Some(e);
                Err(())
            }
        }
    }
}

impl RansacTrialBridge for PyTrialBridge<'_, '_> {
    fn run_trial(
        &self,
        idxs: &[i64],
        rng: &mut NumpyRandomState,
        scan: &mut dyn FnMut(&[f64], Option<&[f64]>),
    ) -> Result<TrialStatus, ()> {
        self.park((|| -> PyResult<TrialStatus> {
            let idx = idx_to_pyarrow(self.py, idxs)?;
            let out = self
                .obj
                .call_method1("run_trial", (idx, self.state_out(rng)?))?;
            let status: u8 = out.get_item(0)?.extract()?;
            self.state_in(rng, &out.get_item(3)?)?;
            match status {
                STATUS_INVALID_DATA => return Ok(TrialStatus::InvalidData),
                STATUS_INVALID_MODEL => return Ok(TrialStatus::InvalidModel),
                STATUS_FITTED => {}
                other => {
                    return Err(PyValueError::new_err(format!(
                        "RANSACRegressor: bridge returned an unknown trial status {other}"
                    )))
                }
            }
            // The arrow arrays own their buffers for the length of this scope,
            // and the scan runs INSIDE it — so the predictions are read where
            // pyarrow put them and never copied (the `RansacTrialBridge`
            // callback contract).
            let pred_arr = capsule_to_array(&out.get_item(1)?)?;
            let pred = host_slice_f64(as_f64(&pred_arr)?)?;
            check_len(pred.len(), self.n * self.t, "run_trial predictions")?;
            let resid_arr = match self.supplies_residual {
                true => Some(capsule_to_array(&out.get_item(2)?)?),
                false => None,
            };
            let resid = match resid_arr.as_ref() {
                Some(a) => {
                    let r = host_slice_f64(as_f64(a)?)?;
                    check_len(r.len(), self.n, "run_trial residuals")?;
                    Some(r)
                }
                None => None,
            };
            scan(pred, resid);
            Ok(TrialStatus::Fitted)
        })())
    }

    fn supplies_residual(&self) -> bool {
        self.supplies_residual
    }

    fn refit(&self, idxs: &[i64], rng: &mut NumpyRandomState) -> Result<(), ()> {
        self.park((|| -> PyResult<()> {
            let idx = idx_to_pyarrow(self.py, idxs)?;
            let out = self.obj.call_method1("refit", (idx, self.state_out(rng)?))?;
            self.state_in(rng, &out)?;
            Ok(())
        })())
    }
}

impl PyTrialBridge<'_, '_> {
    /// The draw stream's 624 key words plus its read position, as a `uint32`
    /// pyarrow array — or `None` when the base estimator cannot consume from it.
    ///
    /// 2.5 KB per trial, paid ONLY by a randomized base. It buys the property
    /// `test_a_randomized_base_consumes_the_same_random_state` pins: the
    /// estimator's own draws land between the trial draws, exactly where
    /// sklearn's do.
    fn state_out(&self, rng: &NumpyRandomState) -> PyResult<Option<Bound<'_, PyAny>>> {
        if !self.seeded {
            return Ok(None);
        }
        let mut words: Vec<u32> = rng.key().to_vec();
        words.push(rng.pos() as u32);
        Ok(Some(u32_vec_to_pyarrow(self.py, words)?))
    }

    /// The inverse of [`state_out`](Self::state_out): adopt the words the base
    /// estimator advanced the generator to.
    fn state_in(&self, rng: &mut NumpyRandomState, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        if !self.seeded || obj.is_none() {
            return Ok(());
        }
        let arr = capsule_to_array(obj)?;
        let words = arr
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| PyValueError::new_err("RANSACRegressor: bridge state must be uint32"))?;
        if words.len() != MT_STATE_WORDS + 1 {
            return Err(PyValueError::new_err(format!(
                "RANSACRegressor: bridge state has {} words, expected {}",
                words.len(),
                MT_STATE_WORDS + 1
            )));
        }
        let mut key = [0u32; MT_STATE_WORDS];
        for (i, slot) in key.iter_mut().enumerate() {
            *slot = words.value(i);
        }
        *rng = NumpyRandomState::from_key(key, words.value(MT_STATE_WORDS) as usize);
        Ok(())
    }
}

/// numpy's MT19937 key length — the `624` of its `get_state()` tuple.
const MT_STATE_WORDS: usize = 624;

/// Run the fit on whichever float arm the design's dtype selects.
///
/// The driver — base arm plus the two predicates — is built ONCE here rather
/// than per arm: since RANSAC-02 the predicates take row INDICES, so nothing
/// about them depends on the design's width.
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
    bridge: Option<&Bound<'_, PyAny>>,
    flags: BridgeFlags,
    stash: &RefCell<Option<PyErr>>,
) -> PyResult<(AnyRansac, RansacSnapshot)> {
    /// Turn a bridge predicate's answer into a verdict, parking any exception
    /// in `stash` (module docs).
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

    let data_cb = |idxs: &[i64]| -> RansacVerdict {
        let obj = bridge.expect("installed only when the bridge exists");
        verdict(
            stash,
            (|| {
                let idx = idx_to_pyarrow(py, idxs)?;
                obj.call_method1("data_valid", (idx,))?.is_truthy()
            })(),
        )
    };
    let model_cb = |model: RansacModel<'_>, idxs: &[i64]| -> RansacVerdict {
        let obj = bridge.expect("installed only when the bridge exists");
        verdict(
            stash,
            (|| {
                let idx = idx_to_pyarrow(py, idxs)?;
                let coef = f64_vec_to_pyarrow(py, model.coef.to_vec())?;
                let icept = f64_vec_to_pyarrow(py, model.intercept.to_vec())?;
                obj.call_method1("model_valid", (idx, coef, icept))?
                    .is_truthy()
            })(),
        )
    };
    let trial_bridge = bridge.map(|obj| PyTrialBridge {
        py,
        obj,
        stash,
        supplies_residual: flags.supplies_residual,
        seeded: flags.seeded,
        n: rows,
        t: n_targets.max(1),
    });

    type DataFn<'f> = &'f dyn Fn(&[i64]) -> RansacVerdict;
    type ModelFn<'f> = &'f dyn Fn(RansacModel<'_>, &[i64]) -> RansacVerdict;
    let driver = match (flags.foreign, trial_bridge.as_ref()) {
        (true, Some(b)) => RansacDriver {
            base: RansacBase::Foreign(b),
            callbacks: RansacCallbacks::default(),
        },
        _ => RansacDriver::with_callbacks(RansacCallbacks {
            is_data_valid: flags.data_valid.then_some(&data_cb as DataFn<'_>),
            is_model_valid: flags.model_valid.then_some(&model_cb as ModelFn<'_>),
        }),
    };

    macro_rules! arm {
        ($float:ty, $as:ident, $host:ident, $variant:ident) => {{
            let est: RansacRegressor<$float, AlgoUnfit> = ransac_build!($float, p);
            let xh = $host($as(xa)?)?;
            let yh = $host($as(ya)?)?;
            // GIL held only when a bridge can fire (method docs).
            let fitted = match bridge {
                Some(_) => {
                    let mut pool = crate::lock_pool();
                    est.fit_from_host_slice(
                        &mut pool,
                        xh,
                        yh,
                        (rows, cols),
                        n_targets,
                        sw,
                        rng,
                        &driver,
                    )
                }
                None => py.detach(|| {
                    let mut pool = crate::lock_pool();
                    est.fit_from_host_slice(
                        &mut pool,
                        xh,
                        yh,
                        (rows, cols),
                        n_targets,
                        sw,
                        rng,
                        &RansacDriver::default(),
                    )
                }),
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

/// Row indices as an `int32` **pyarrow** array — the cheapest sub-sample handle
/// that crosses (module docs).
///
/// `i32` rather than `i64` because these index a design whose row count already
/// fits an `i32` on every path that reaches here, and because it halves what
/// crosses on the one call a trial makes.
fn idx_to_pyarrow<'py>(py: Python<'py>, idxs: &[i64]) -> PyResult<Bound<'py, PyAny>> {
    i32_vec_to_pyarrow(py, idxs.iter().map(|&v| v as i32).collect())
}

/// Reject a bridge result whose length is not the one the design implies.
fn check_len(got: usize, want: usize, what: &str) -> PyResult<()> {
    if got != want {
        return Err(PyValueError::new_err(format!(
            "RANSACRegressor: bridge {what} has length {got}, expected {want}"
        )));
    }
    Ok(())
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
