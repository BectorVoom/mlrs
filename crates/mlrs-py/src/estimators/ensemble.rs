//! Ensemble `#[pyclass]` wrappers (PY-ENS-01/02/03/04, RF-IMP-02, RF-OOB-02):
//! `PyRandomForestClassifier` (TASK-08), `PyRandomForestRegressor`
//! (TASK-09), `PyHistGradientBoostingClassifier` (TASK-18), and
//! `PyHistGradientBoostingRegressor` (TASK-19, appended below — both HGB
//! classes are structural only, no `feature_importances_`/`oob_score_`: not
//! applicable to HGB, SPEC §2 non-goal).
//!
//! Both wrap the TYPESTATE `mlrs_algos::ensemble::random_forest_{classifier,
//! regressor}` estimators (`S = Unfit` default — D-04/[`crate::any_estimator_typestate`]
//! is the CORRECT macro here, not [`crate::any_estimator`]: the plain macro would
//! resolve the `F32`/`F64` fitted arms as `RandomForestClassifier<f32,
//! mlrs_algos::typestate::Unfit>` instead of the `Fitted` sibling the consuming
//! `fit` returns — the exact trap `crate::dispatch`'s doc comment warns about).
//! Every device-compute body honors the two load-bearing contracts documented on
//! [`crate::dispatch`]: GIL release (`py.detach`, PY-03) and the f64 guard
//! (`crate::capability::guard_f64()?` BEFORE any f64 upload, D-04). The exact
//! `fit` shape mirrors [`crate::estimators::naive_bayes::PyGaussianNB::fit`]
//! (typestate-fit-with-y template).
//!
//! ## `max_features` (sklearn-named, heterogeneous shape)
//! sklearn accepts `"sqrt"`, `"log2"`, an int, a float, or `None` for
//! `max_features`. This is parsed EAGERLY at `#[new]` time into the plain Rust
//! [`MaxFeaturesArg`] enum (mirrors
//! [`crate::estimators::naive_bayes::resolve_min_categories`]'s
//! parse-immediately precedent for a `PyAny`-shaped constructor argument,
//! deliberately avoiding storing a raw `PyObject` in the macro-emitted `Unfit`
//! arm — a `PyObject` field would need a live GIL token merely to construct the
//! Rust-callable `unfit_default()` smoke-test helper, which every OTHER
//! estimator in this crate builds without one). Only the `Frac` variant's
//! fraction→feature-count arithmetic is deferred to `fit()` time, since it needs
//! `n_features` (`cols`), which is not known until then.
//!
//! ## `feature_importances_` / `oob_score_` (RF-IMP-02 / RF-OOB-02)
//! Both are thin readbacks of the already-computed `Fitted`-state accessors
//! (`RandomForestClassifier::<F, Fitted>::{feature_importances, oob_score}`,
//! RF-IMP-01/RF-OOB-01) — no new device work, no `py.detach` needed (pure host
//! `Vec` clones). `oob_score_f32`/`_f64` return `PyResult<Option<f32|f64>>`:
//! `None` when the estimator was fitted with `oob_score=False` (the common
//! case); `not_fitted(..)` (a `PyValueError`) when called before `fit`.
//!
//! Tests live in `crates/mlrs-py/tests/` (AGENTS.md §2 — never an in-source
//! `#[cfg(test)] mod tests`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mlrs_algos::ensemble::hist_gradient_boosting_classifier::HistGradientBoostingClassifier;
use mlrs_algos::ensemble::hist_gradient_boosting_regressor::HistGradientBoostingRegressor;
use mlrs_algos::ensemble::random_forest_classifier::RandomForestClassifier;
use mlrs_algos::ensemble::random_forest_regressor::RandomForestRegressor;
use mlrs_algos::ensemble::MaxFeatures;
use mlrs_algos::typestate::{
    Fit as TypestateFit, Predict as TypestatePredict, PredictLabels as TypestatePredictLabels,
    PredictProba as TypestatePredictProba,
};

use crate::errors::{algo_err_to_py, build_err_to_py, dtype_mismatch, not_fitted};
use crate::ingress::{
    as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype,
};

// ===========================================================================
// max_features (sklearn-named) — parsed at construction time (see module doc).
// ===========================================================================

/// The parsed SHAPE of the sklearn `max_features` hyperparameter. `Frac`'s
/// fraction→count arithmetic is resolved against `n_features` at `fit()` time
/// via [`MaxFeaturesArg::resolve`] — the only piece of this value that is
/// genuinely data-dependent.
#[derive(Debug, Clone, Copy)]
enum MaxFeaturesArg {
    Sqrt,
    Log2,
    All,
    Count(usize),
    Frac(f64),
}

impl MaxFeaturesArg {
    /// Resolve against the fitted feature count. sklearn's forests compute a
    /// float `max_features` as `max(1, int(max_features * n_features_in_))` —
    /// i.e. TRUNCATION toward zero (`int()`), NOT `ceil`. Truncation also keeps
    /// this path consistent with `MaxFeatures::Sqrt`/`Log2`, which floor
    /// (`mlrs_algos::ensemble::MaxFeatures::resolve`).
    fn resolve(self, n_features: usize) -> MaxFeatures {
        match self {
            MaxFeaturesArg::Sqrt => MaxFeatures::Sqrt,
            MaxFeaturesArg::Log2 => MaxFeatures::Log2,
            MaxFeaturesArg::All => MaxFeatures::All,
            MaxFeaturesArg::Count(v) => MaxFeatures::Value(v),
            MaxFeaturesArg::Frac(f) => {
                MaxFeatures::Value(((f * n_features as f64) as usize).max(1))
            }
        }
    }
}

/// Parse the sklearn-named `max_features` constructor argument: `"sqrt"` /
/// `"log2"` (case-sensitive, sklearn convention) / `None` (→ all features,
/// matching sklearn's `max_features=None` semantics) / `"all"` (an mlrs
/// convenience alias for "use all features", always expressible even where the
/// FFI cannot round-trip an explicit Python `None`) / a positive Python `int`
/// (→ an explicit per-node count) / a Python `float` in `(0.0, 1.0]` (→ a
/// fraction, resolved to a count at `fit()`). Any other shape (including an
/// out-of-range int/float or an unrecognized string) is a `PyValueError`
/// naming the offending value (mirrors the
/// [`crate::estimators::manifold::parse_metric`] string-parse-to-`ValueError`
/// precedent).
fn parse_max_features(v: &Bound<'_, PyAny>) -> PyResult<MaxFeaturesArg> {
    if v.is_none() {
        return Ok(MaxFeaturesArg::All);
    }
    if let Ok(s) = v.extract::<String>() {
        return match s.as_str() {
            "sqrt" => Ok(MaxFeaturesArg::Sqrt),
            "log2" => Ok(MaxFeaturesArg::Log2),
            "all" => Ok(MaxFeaturesArg::All),
            other => Err(PyValueError::new_err(format!(
                "random_forest: unknown max_features string {other:?}; expected \
                 \"sqrt\", \"log2\", \"all\", an int, a float, or None"
            ))),
        };
    }
    // NOTE: Python `bool` extracts as an integer (bool subclasses int); this is
    // an accepted, harmless edge case (max_features=True/False would resolve to
    // Count(1)/reject as < 1) — sklearn itself does not special-case bool here
    // either.
    if let Ok(i) = v.extract::<i64>() {
        if i < 1 {
            return Err(PyValueError::new_err(format!(
                "random_forest: max_features integer must be >= 1 (got {i})"
            )));
        }
        return Ok(MaxFeaturesArg::Count(i as usize));
    }
    if let Ok(f) = v.extract::<f64>() {
        if !(f > 0.0 && f <= 1.0) {
            return Err(PyValueError::new_err(format!(
                "random_forest: max_features float must be in (0.0, 1.0] (got {f})"
            )));
        }
        return Ok(MaxFeaturesArg::Frac(f));
    }
    Err(PyValueError::new_err(
        "random_forest: max_features must be \"sqrt\", \"log2\", \"all\", an int, a float, or None",
    ))
}

// ===========================================================================
// RandomForestClassifier — Fit + PredictLabels + PredictProba + classes_ +
// feature_importances_ (RF-IMP-02) + oob_score_ (RF-OOB-02).
// ===========================================================================

crate::any_estimator_typestate! {
    any:   AnyRandomForestClassifier,
    algo:  mlrs_algos::ensemble::random_forest_classifier::RandomForestClassifier,
    unfit: {
        n_estimators: usize, max_depth: usize, n_bins: usize,
        max_features: MaxFeaturesArg, min_samples_split: f64,
        min_samples_leaf: f64, bootstrap: bool, oob_score: bool, seed: u64,
    },
}

/// sklearn-compatible `RandomForestClassifier`. Defaults mirror
/// `RandomForestClassifierBuilder`'s own defaults verbatim (D-08 single
/// source): `n_estimators=100, max_depth=10, n_bins=32, max_features="sqrt",
/// min_samples_split=2.0, min_samples_leaf=1.0, bootstrap=True,
/// oob_score=False, seed=42`.
#[pyclass(name = "RandomForestClassifier")]
pub struct PyRandomForestClassifier {
    inner: AnyRandomForestClassifier,
}

impl PyRandomForestClassifier {
    /// Rust-callable default constructor (smoke-test seam — see
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`]).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyRandomForestClassifier::Unfit {
                n_estimators: 100,
                max_depth: 10,
                n_bins: 32,
                max_features: MaxFeaturesArg::Sqrt,
                min_samples_split: 2.0,
                min_samples_leaf: 1.0,
                bootstrap: true,
                oob_score: false,
                seed: 42,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyRandomForestClassifier::Unfit { .. })
    }
}

#[pymethods]
impl PyRandomForestClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        max_depth = 10,
        n_bins = 32,
        max_features = None,
        min_samples_split = 2.0,
        min_samples_leaf = 1.0,
        bootstrap = true,
        oob_score = false,
        seed = 42,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        max_depth: usize,
        n_bins: usize,
        max_features: Option<&Bound<'_, PyAny>>,
        min_samples_split: f64,
        min_samples_leaf: f64,
        bootstrap: bool,
        oob_score: bool,
        seed: u64,
    ) -> PyResult<Self> {
        // FFI-level `max_features` handling. PyO3's `Option<&Bound<PyAny>>`
        // cannot distinguish an OMITTED argument from an explicit Python
        // `None` (both arrive as Rust `None`), so at THIS low-level `_mlrs`
        // boundary both resolve to the estimator's sklearn OMITTED default
        // (`"sqrt"` for the classifier). Full sklearn `max_features=None`
        // ("use all features") parity is provided one layer up by the Python
        // shim (`mlrs.RandomForestClassifier`), which CAN distinguish the two
        // (its own `__init__` default is the `"sqrt"` string, so an explicit
        // `None` is visible) and forwards it as the `"all"` sentinel.
        // All-features is also expressible directly at this FFI boundary via
        // `max_features="all"` or `1.0`.
        let max_features = match max_features {
            Some(v) => parse_max_features(v)?,
            None => MaxFeaturesArg::Sqrt,
        };
        Ok(Self {
            inner: AnyRandomForestClassifier::Unfit {
                n_estimators,
                max_depth,
                n_bins,
                max_features,
                min_samples_split,
                min_samples_leaf,
                bootstrap,
                oob_score,
                seed,
            },
        })
    }

    /// Fit on `x` (`rows × cols`, row-major) + integer-valued class label
    /// vector `y`. The builder validates data-INDEPENDENT hyperparameters at
    /// `build()` (a bad `min_samples_split`/`oob_score`-without-`bootstrap`
    /// surfaces as a `ValueError` there, D-05/D-09/RF-OOB-01); GIL released
    /// (PY-03); f64 guarded (D-04). `max_features` is resolved against `cols`
    /// (the fitted feature count) here, BEFORE `py.detach` (it touches no
    /// device state, but resolving it inside the detached closure would need
    /// a `Python<'_>` token the closure does not capture by design).
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
        let (
            n_estimators,
            max_depth,
            n_bins,
            max_features,
            min_samples_split,
            min_samples_leaf,
            bootstrap,
            oob_score,
            seed,
        ) = match &self.inner {
            AnyRandomForestClassifier::Unfit {
                n_estimators,
                max_depth,
                n_bins,
                max_features,
                min_samples_split,
                min_samples_leaf,
                bootstrap,
                oob_score,
                seed,
            } => (
                *n_estimators,
                *max_depth,
                *n_bins,
                max_features.resolve(cols),
                *min_samples_split,
                *min_samples_leaf,
                *bootstrap,
                *oob_score,
                *seed,
            ),
            _ => return Err(not_fitted("random_forest_classifier", "re-fit")),
        };
        let fitted = py.detach(|| -> PyResult<AnyRandomForestClassifier> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = RandomForestClassifier::<f32>::builder()
                        .n_estimators(n_estimators)
                        .max_depth(max_depth)
                        .n_bins(n_bins)
                        .max_features(max_features)
                        .min_samples_split(min_samples_split)
                        .min_samples_leaf(min_samples_leaf)
                        .bootstrap(bootstrap)
                        .oob_score(oob_score)
                        .seed(seed)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyRandomForestClassifier::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = RandomForestClassifier::<f64>::builder()
                        .n_estimators(n_estimators)
                        .max_depth(max_depth)
                        .n_bins(n_bins)
                        .max_features(max_features)
                        .min_samples_split(min_samples_split)
                        .min_samples_leaf(min_samples_leaf)
                        .bootstrap(bootstrap)
                        .oob_score(oob_score)
                        .seed(seed)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyRandomForestClassifier::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict = argmax(predict_proba)` mapped back through `classes_` (i32).
    fn predict_labels(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<i32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyRandomForestClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(
                        TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool),
                    )
                }
                AnyRandomForestClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(
                        TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool),
                    )
                }
                _ => Err(not_fitted("random_forest_classifier", "predict")),
            }
        })
    }

    /// `predict_proba(x)` (f32 fitted path) → `n_query × n_classes` host
    /// floats, rows sum to 1.
    fn predict_proba_f32(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyRandomForestClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(
                        TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool),
                    )
                }
                AnyRandomForestClassifier::F64(_) => {
                    Err(dtype_mismatch("random_forest_classifier", "f32", "f64"))
                }
                _ => Err(not_fitted(
                    "random_forest_classifier",
                    "predict_proba (f32 path)",
                )),
            }
        })
    }
    /// `predict_proba(x)` (f64 fitted path).
    fn predict_proba_f64(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyRandomForestClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(
                        TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool),
                    )
                }
                AnyRandomForestClassifier::F32(_) => {
                    Err(dtype_mismatch("random_forest_classifier", "f64", "f32"))
                }
                _ => Err(not_fitted(
                    "random_forest_classifier",
                    "predict_proba (f64 path)",
                )),
            }
        })
    }

    /// The DISTINCT sorted training labels (sklearn's `classes_` contract).
    /// Empty before `fit`.
    fn classes_(&self) -> Vec<i32> {
        match &self.inner {
            AnyRandomForestClassifier::F32(e) => e.classes().to_vec(),
            AnyRandomForestClassifier::F64(e) => e.classes().to_vec(),
            AnyRandomForestClassifier::Unfit { .. } => Vec::new(),
        }
    }

    /// RF-IMP-02: `feature_importances_` (f32 fitted path) — length
    /// `n_features`, sums to 1 (RF-IMP-01).
    fn feature_importances_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyRandomForestClassifier::F32(est) => Ok(est.feature_importances().to_vec()),
            AnyRandomForestClassifier::F64(_) => {
                Err(dtype_mismatch("random_forest_classifier", "f32", "f64"))
            }
            AnyRandomForestClassifier::Unfit { .. } => Err(not_fitted(
                "random_forest_classifier",
                "feature_importances_ (f32 path)",
            )),
        }
    }
    /// RF-IMP-02: `feature_importances_` (f64 fitted path).
    fn feature_importances_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyRandomForestClassifier::F64(est) => Ok(est.feature_importances().to_vec()),
            AnyRandomForestClassifier::F32(_) => {
                Err(dtype_mismatch("random_forest_classifier", "f64", "f32"))
            }
            AnyRandomForestClassifier::Unfit { .. } => Err(not_fitted(
                "random_forest_classifier",
                "feature_importances_ (f64 path)",
            )),
        }
    }

    /// RF-OOB-02: `oob_score_` (f32 fitted path) — `Some(score)` iff the
    /// estimator was constructed with `oob_score=True`; `None` otherwise
    /// (RF-OOB-01). `Err` (not-fitted) only if called before `fit`.
    fn oob_score_f32(&self) -> PyResult<Option<f32>> {
        match &self.inner {
            AnyRandomForestClassifier::F32(est) => Ok(est.oob_score()),
            AnyRandomForestClassifier::F64(_) => {
                Err(dtype_mismatch("random_forest_classifier", "f32", "f64"))
            }
            AnyRandomForestClassifier::Unfit { .. } => Err(not_fitted(
                "random_forest_classifier",
                "oob_score_ (f32 path)",
            )),
        }
    }
    /// RF-OOB-02: `oob_score_` (f64 fitted path).
    fn oob_score_f64(&self) -> PyResult<Option<f64>> {
        match &self.inner {
            AnyRandomForestClassifier::F64(est) => Ok(est.oob_score()),
            AnyRandomForestClassifier::F32(_) => {
                Err(dtype_mismatch("random_forest_classifier", "f64", "f32"))
            }
            AnyRandomForestClassifier::Unfit { .. } => Err(not_fitted(
                "random_forest_classifier",
                "oob_score_ (f64 path)",
            )),
        }
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyRandomForestClassifier::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyRandomForestClassifier::Unfit { .. } => None,
            AnyRandomForestClassifier::F32(_) => Some("f32"),
            AnyRandomForestClassifier::F64(_) => Some("f64"),
        }
    }
}

// ===========================================================================
// RandomForestRegressor (TASK-09) — Fit + Predict + feature_importances_
// (RF-IMP-02) + oob_score_ (RF-OOB-02). Same composition as
// `PyRandomForestClassifier` above, minus `classes_`/`predict_proba`, plus a
// float `predict_f32`/`_f64` (mirrors
// [`crate::estimators::linear::PyLinearRegression::predict_f32`]'s
// `Predict`-trait shape). The regressor's sklearn default `max_features` is
// `1.0` ("all features"), NOT the classifier's `"sqrt"` — an omitted argument
// AND an explicit `max_features=None` both resolve to `All`, which is exactly
// sklearn's regressor semantics for both cases (no parity subtlety, unlike the
// classifier whose omitted default is `"sqrt"` but whose explicit `None` means
// "all" — handled by the shim).
// ===========================================================================

crate::any_estimator_typestate! {
    any:   AnyRandomForestRegressor,
    algo:  mlrs_algos::ensemble::random_forest_regressor::RandomForestRegressor,
    unfit: {
        n_estimators: usize, max_depth: usize, n_bins: usize,
        max_features: MaxFeaturesArg, min_samples_split: f64,
        min_samples_leaf: f64, bootstrap: bool, oob_score: bool, seed: u64,
    },
}

/// sklearn-compatible `RandomForestRegressor`. Defaults mirror
/// `RandomForestRegressorBuilder`'s own defaults verbatim (D-08 single
/// source): `n_estimators=100, max_depth=10, n_bins=32, max_features=1.0
/// ("all"), min_samples_split=2.0, min_samples_leaf=1.0, bootstrap=True,
/// oob_score=False, seed=42`.
#[pyclass(name = "RandomForestRegressor")]
pub struct PyRandomForestRegressor {
    inner: AnyRandomForestRegressor,
}

impl PyRandomForestRegressor {
    /// Rust-callable default constructor (smoke-test seam — see
    /// [`PyRandomForestClassifier::unfit_default`]).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyRandomForestRegressor::Unfit {
                n_estimators: 100,
                max_depth: 10,
                n_bins: 32,
                max_features: MaxFeaturesArg::All,
                min_samples_split: 2.0,
                min_samples_leaf: 1.0,
                bootstrap: true,
                oob_score: false,
                seed: 42,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyRandomForestRegressor::Unfit { .. })
    }
}

#[pymethods]
impl PyRandomForestRegressor {
    #[new]
    #[pyo3(signature = (
        n_estimators = 100,
        max_depth = 10,
        n_bins = 32,
        max_features = None,
        min_samples_split = 2.0,
        min_samples_leaf = 1.0,
        bootstrap = true,
        oob_score = false,
        seed = 42,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        max_depth: usize,
        n_bins: usize,
        max_features: Option<&Bound<'_, PyAny>>,
        min_samples_split: f64,
        min_samples_leaf: f64,
        bootstrap: bool,
        oob_score: bool,
        seed: u64,
    ) -> PyResult<Self> {
        // For the regressor there is no `max_features` parity subtlety:
        // sklearn's own regressor default IS "all features" (`max_features=1.0`)
        // and its explicit `None` ALSO means "all features", so an omitted arg
        // and an explicit `None` both correctly resolve to `All` here (the
        // shim forwards its `1.0` default / an explicit `None` accordingly).
        // `"all"` / `1.0` are equivalent explicit spellings.
        let max_features = match max_features {
            Some(v) => parse_max_features(v)?,
            None => MaxFeaturesArg::All,
        };
        Ok(Self {
            inner: AnyRandomForestRegressor::Unfit {
                n_estimators,
                max_depth,
                n_bins,
                max_features,
                min_samples_split,
                min_samples_leaf,
                bootstrap,
                oob_score,
                seed,
            },
        })
    }

    /// Fit on `x` (`rows × cols`, row-major) + continuous target `y`. Same
    /// contract as [`PyRandomForestClassifier::fit`] (builder validation,
    /// GIL release, f64 guard); `max_features` resolved against `cols` before
    /// `py.detach`.
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
        let (
            n_estimators,
            max_depth,
            n_bins,
            max_features,
            min_samples_split,
            min_samples_leaf,
            bootstrap,
            oob_score,
            seed,
        ) = match &self.inner {
            AnyRandomForestRegressor::Unfit {
                n_estimators,
                max_depth,
                n_bins,
                max_features,
                min_samples_split,
                min_samples_leaf,
                bootstrap,
                oob_score,
                seed,
            } => (
                *n_estimators,
                *max_depth,
                *n_bins,
                max_features.resolve(cols),
                *min_samples_split,
                *min_samples_leaf,
                *bootstrap,
                *oob_score,
                *seed,
            ),
            _ => return Err(not_fitted("random_forest_regressor", "re-fit")),
        };
        let fitted = py.detach(|| -> PyResult<AnyRandomForestRegressor> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = RandomForestRegressor::<f32>::builder()
                        .n_estimators(n_estimators)
                        .max_depth(max_depth)
                        .n_bins(n_bins)
                        .max_features(max_features)
                        .min_samples_split(min_samples_split)
                        .min_samples_leaf(min_samples_leaf)
                        .bootstrap(bootstrap)
                        .oob_score(oob_score)
                        .seed(seed)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyRandomForestRegressor::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = RandomForestRegressor::<f64>::builder()
                        .n_estimators(n_estimators)
                        .max_depth(max_depth)
                        .n_bins(n_bins)
                        .max_features(max_features)
                        .min_samples_split(min_samples_split)
                        .min_samples_leaf(min_samples_leaf)
                        .bootstrap(bootstrap)
                        .oob_score(oob_score)
                        .seed(seed)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyRandomForestRegressor::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → length-`rows` host `Vec<f32>` (f32 fitted path).
    fn predict_f32(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| -> PyResult<Vec<f32>> {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyRandomForestRegressor::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let out = TypestatePredict::predict(est, &mut pool, &xd, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(out.to_host_metered(&mut pool))
                }
                AnyRandomForestRegressor::F64(_) => {
                    Err(dtype_mismatch("random_forest_regressor", "f32", "f64"))
                }
                _ => Err(not_fitted("random_forest_regressor", "predict (f32 path)")),
            }
        })
    }
    /// `predict(x)` → length-`rows` host `Vec<f64>` (f64 fitted path).
    fn predict_f64(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| -> PyResult<Vec<f64>> {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyRandomForestRegressor::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let out = TypestatePredict::predict(est, &mut pool, &xd, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(out.to_host_metered(&mut pool))
                }
                AnyRandomForestRegressor::F32(_) => {
                    Err(dtype_mismatch("random_forest_regressor", "f64", "f32"))
                }
                _ => Err(not_fitted("random_forest_regressor", "predict (f64 path)")),
            }
        })
    }

    /// RF-IMP-02: `feature_importances_` (f32 fitted path) — length
    /// `n_features`, sums to 1 (RF-IMP-01).
    fn feature_importances_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyRandomForestRegressor::F32(est) => Ok(est.feature_importances().to_vec()),
            AnyRandomForestRegressor::F64(_) => {
                Err(dtype_mismatch("random_forest_regressor", "f32", "f64"))
            }
            AnyRandomForestRegressor::Unfit { .. } => Err(not_fitted(
                "random_forest_regressor",
                "feature_importances_ (f32 path)",
            )),
        }
    }
    /// RF-IMP-02: `feature_importances_` (f64 fitted path).
    fn feature_importances_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyRandomForestRegressor::F64(est) => Ok(est.feature_importances().to_vec()),
            AnyRandomForestRegressor::F32(_) => {
                Err(dtype_mismatch("random_forest_regressor", "f64", "f32"))
            }
            AnyRandomForestRegressor::Unfit { .. } => Err(not_fitted(
                "random_forest_regressor",
                "feature_importances_ (f64 path)",
            )),
        }
    }

    /// RF-OOB-02: `oob_score_` (f32 fitted path) — `Some(score)` iff the
    /// estimator was constructed with `oob_score=True`; `None` otherwise
    /// (RF-OOB-01). `Err` (not-fitted) only if called before `fit`.
    fn oob_score_f32(&self) -> PyResult<Option<f32>> {
        match &self.inner {
            AnyRandomForestRegressor::F32(est) => Ok(est.oob_score()),
            AnyRandomForestRegressor::F64(_) => {
                Err(dtype_mismatch("random_forest_regressor", "f32", "f64"))
            }
            AnyRandomForestRegressor::Unfit { .. } => Err(not_fitted(
                "random_forest_regressor",
                "oob_score_ (f32 path)",
            )),
        }
    }
    /// RF-OOB-02: `oob_score_` (f64 fitted path).
    fn oob_score_f64(&self) -> PyResult<Option<f64>> {
        match &self.inner {
            AnyRandomForestRegressor::F64(est) => Ok(est.oob_score()),
            AnyRandomForestRegressor::F32(_) => {
                Err(dtype_mismatch("random_forest_regressor", "f64", "f32"))
            }
            AnyRandomForestRegressor::Unfit { .. } => Err(not_fitted(
                "random_forest_regressor",
                "oob_score_ (f64 path)",
            )),
        }
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyRandomForestRegressor::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyRandomForestRegressor::Unfit { .. } => None,
            AnyRandomForestRegressor::F32(_) => Some("f32"),
            AnyRandomForestRegressor::F64(_) => Some("f64"),
        }
    }
}

// ===========================================================================
// HistGradientBoostingClassifier (TASK-18, PY-ENS-03, structural) — Fit +
// PredictLabels + PredictProba + classes_. Mechanically identical to
// `PyRandomForestClassifier` above (same typestate-fit-with-y template, same
// dtype-dispatch/error-mapping shape) MINUS `max_features`/`bootstrap`/
// `oob_score` (HGB has none of these) and MINUS `feature_importances_`/
// `oob_score_` entirely — sklearn's own `HistGradientBoostingClassifier` does
// not expose either attribute (boosting is not a bagging/OOB scheme), so
// this class deliberately carries no `feature_importances_f32/_f64`/
// `oob_score_f32/_f64` methods (SPEC §2 non-goal; verified absent by
// `crates/mlrs-py/tests/test_random_forest.py`).
//
// `n_bins` defaults to `64` (the Rust builder default), NOT `255` — the
// `n_bins=255` deterministic-tier oracle override (TASK-24) is a TEST-TIME
// construction argument, not a changed Python-visible default.
// ===========================================================================

crate::any_estimator_typestate! {
    any:   AnyHistGradientBoostingClassifier,
    algo:  mlrs_algos::ensemble::hist_gradient_boosting_classifier::HistGradientBoostingClassifier,
    unfit: {
        max_iter: usize, learning_rate: f64, max_depth: usize, n_bins: usize,
        l2_regularization: f64, min_samples_leaf: usize,
    },
}

/// sklearn-compatible `HistGradientBoostingClassifier`. Defaults mirror
/// `HistGradientBoostingClassifierBuilder`'s own defaults verbatim (D-08
/// single source): `max_iter=100, learning_rate=0.1, max_depth=6, n_bins=64,
/// l2_regularization=0.0, min_samples_leaf=20`.
#[pyclass(name = "HistGradientBoostingClassifier")]
pub struct PyHistGradientBoostingClassifier {
    inner: AnyHistGradientBoostingClassifier,
}

impl PyHistGradientBoostingClassifier {
    /// Rust-callable default constructor (smoke-test seam — see
    /// [`PyRandomForestClassifier::unfit_default`]).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyHistGradientBoostingClassifier::Unfit {
                max_iter: 100,
                learning_rate: 0.1,
                max_depth: 6,
                n_bins: 64,
                l2_regularization: 0.0,
                min_samples_leaf: 20,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyHistGradientBoostingClassifier::Unfit { .. })
    }
}

#[pymethods]
impl PyHistGradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (
        max_iter = 100,
        learning_rate = 0.1,
        max_depth = 6,
        n_bins = 64,
        l2_regularization = 0.0,
        min_samples_leaf = 20,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_iter: usize,
        learning_rate: f64,
        max_depth: usize,
        n_bins: usize,
        l2_regularization: f64,
        min_samples_leaf: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: AnyHistGradientBoostingClassifier::Unfit {
                max_iter,
                learning_rate,
                max_depth,
                n_bins,
                l2_regularization,
                min_samples_leaf,
            },
        })
    }

    /// Fit on `x` (`rows × cols`, row-major) + integer-valued class label
    /// vector `y`. Same contract as [`PyRandomForestClassifier::fit`] (builder
    /// validation, GIL release PY-03, f64 guard D-04) — HGB's builder has no
    /// data-dependent argument (unlike RF's `max_features`), so nothing needs
    /// resolving before `py.detach`.
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
        let (max_iter, learning_rate, max_depth, n_bins, l2_regularization, min_samples_leaf) =
            match &self.inner {
                AnyHistGradientBoostingClassifier::Unfit {
                    max_iter,
                    learning_rate,
                    max_depth,
                    n_bins,
                    l2_regularization,
                    min_samples_leaf,
                } => (
                    *max_iter,
                    *learning_rate,
                    *max_depth,
                    *n_bins,
                    *l2_regularization,
                    *min_samples_leaf,
                ),
                _ => return Err(not_fitted("hist_gradient_boosting_classifier", "re-fit")),
            };
        let fitted = py.detach(|| -> PyResult<AnyHistGradientBoostingClassifier> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = HistGradientBoostingClassifier::<f32>::builder()
                        .max_iter(max_iter)
                        .learning_rate(learning_rate)
                        .max_depth(max_depth)
                        .n_bins(n_bins)
                        .l2_regularization(l2_regularization)
                        .min_samples_leaf(min_samples_leaf)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyHistGradientBoostingClassifier::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = HistGradientBoostingClassifier::<f64>::builder()
                        .max_iter(max_iter)
                        .learning_rate(learning_rate)
                        .max_depth(max_depth)
                        .n_bins(n_bins)
                        .l2_regularization(l2_regularization)
                        .min_samples_leaf(min_samples_leaf)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyHistGradientBoostingClassifier::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict = argmax(predict_proba)` mapped back through `classes_` (i32).
    fn predict_labels(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<i32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyHistGradientBoostingClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(
                        TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool),
                    )
                }
                AnyHistGradientBoostingClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(
                        TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool),
                    )
                }
                _ => Err(not_fitted("hist_gradient_boosting_classifier", "predict")),
            }
        })
    }

    /// `predict_proba(x)` (f32 fitted path) → `n_query × n_classes` host
    /// floats, rows sum to 1.
    fn predict_proba_f32(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyHistGradientBoostingClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(
                        TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool),
                    )
                }
                AnyHistGradientBoostingClassifier::F64(_) => Err(dtype_mismatch(
                    "hist_gradient_boosting_classifier",
                    "f32",
                    "f64",
                )),
                _ => Err(not_fitted(
                    "hist_gradient_boosting_classifier",
                    "predict_proba (f32 path)",
                )),
            }
        })
    }
    /// `predict_proba(x)` (f64 fitted path).
    fn predict_proba_f64(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyHistGradientBoostingClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(
                        TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?
                            .to_host_metered(&mut pool),
                    )
                }
                AnyHistGradientBoostingClassifier::F32(_) => Err(dtype_mismatch(
                    "hist_gradient_boosting_classifier",
                    "f64",
                    "f32",
                )),
                _ => Err(not_fitted(
                    "hist_gradient_boosting_classifier",
                    "predict_proba (f64 path)",
                )),
            }
        })
    }

    /// The DISTINCT sorted training labels (sklearn's `classes_` contract).
    /// Empty before `fit`.
    fn classes_(&self) -> Vec<i32> {
        match &self.inner {
            AnyHistGradientBoostingClassifier::F32(e) => e.classes().to_vec(),
            AnyHistGradientBoostingClassifier::F64(e) => e.classes().to_vec(),
            AnyHistGradientBoostingClassifier::Unfit { .. } => Vec::new(),
        }
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyHistGradientBoostingClassifier::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyHistGradientBoostingClassifier::Unfit { .. } => None,
            AnyHistGradientBoostingClassifier::F32(_) => Some("f32"),
            AnyHistGradientBoostingClassifier::F64(_) => Some("f64"),
        }
    }
}

// ===========================================================================
// HistGradientBoostingRegressor (TASK-19, PY-ENS-04, structural) — Fit +
// Predict. Mechanically identical to `PyHistGradientBoostingClassifier` above
// (same builder-setter names/defaults, same dtype-dispatch/error-mapping
// shape) MINUS `classes_`/`predict_labels`/`predict_proba_f32/_f64`, plus a
// float `predict_f32`/`_f64` (mirrors
// [`PyRandomForestRegressor::predict_f32`]'s `Predict`-trait shape). Like the
// classifier, this class deliberately carries no `feature_importances_f32/_f64`/
// `oob_score_f32/_f64` methods (SPEC §2 non-goal — sklearn's own HGB
// estimators expose neither attribute).
// ===========================================================================

crate::any_estimator_typestate! {
    any:   AnyHistGradientBoostingRegressor,
    algo:  mlrs_algos::ensemble::hist_gradient_boosting_regressor::HistGradientBoostingRegressor,
    unfit: {
        max_iter: usize, learning_rate: f64, max_depth: usize, n_bins: usize,
        l2_regularization: f64, min_samples_leaf: usize,
    },
}

/// sklearn-compatible `HistGradientBoostingRegressor`. Defaults mirror
/// `HistGradientBoostingRegressorBuilder`'s own defaults verbatim (D-08
/// single source): `max_iter=100, learning_rate=0.1, max_depth=6, n_bins=64,
/// l2_regularization=0.0, min_samples_leaf=20`.
#[pyclass(name = "HistGradientBoostingRegressor")]
pub struct PyHistGradientBoostingRegressor {
    inner: AnyHistGradientBoostingRegressor,
}

impl PyHistGradientBoostingRegressor {
    /// Rust-callable default constructor (smoke-test seam — see
    /// [`PyRandomForestClassifier::unfit_default`]).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyHistGradientBoostingRegressor::Unfit {
                max_iter: 100,
                learning_rate: 0.1,
                max_depth: 6,
                n_bins: 64,
                l2_regularization: 0.0,
                min_samples_leaf: 20,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyHistGradientBoostingRegressor::Unfit { .. })
    }
}

#[pymethods]
impl PyHistGradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (
        max_iter = 100,
        learning_rate = 0.1,
        max_depth = 6,
        n_bins = 64,
        l2_regularization = 0.0,
        min_samples_leaf = 20,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_iter: usize,
        learning_rate: f64,
        max_depth: usize,
        n_bins: usize,
        l2_regularization: f64,
        min_samples_leaf: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: AnyHistGradientBoostingRegressor::Unfit {
                max_iter,
                learning_rate,
                max_depth,
                n_bins,
                l2_regularization,
                min_samples_leaf,
            },
        })
    }

    /// Fit on `x` (`rows × cols`, row-major) + continuous target `y`. Same
    /// contract as [`PyHistGradientBoostingClassifier::fit`] (builder
    /// validation, GIL release PY-03, f64 guard D-04) — HGB's builder has no
    /// data-dependent argument, so nothing needs resolving before
    /// `py.detach`.
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
        let (max_iter, learning_rate, max_depth, n_bins, l2_regularization, min_samples_leaf) =
            match &self.inner {
                AnyHistGradientBoostingRegressor::Unfit {
                    max_iter,
                    learning_rate,
                    max_depth,
                    n_bins,
                    l2_regularization,
                    min_samples_leaf,
                } => (
                    *max_iter,
                    *learning_rate,
                    *max_depth,
                    *n_bins,
                    *l2_regularization,
                    *min_samples_leaf,
                ),
                _ => return Err(not_fitted("hist_gradient_boosting_regressor", "re-fit")),
            };
        let fitted = py.detach(|| -> PyResult<AnyHistGradientBoostingRegressor> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = HistGradientBoostingRegressor::<f32>::builder()
                        .max_iter(max_iter)
                        .learning_rate(learning_rate)
                        .max_depth(max_depth)
                        .n_bins(n_bins)
                        .l2_regularization(l2_regularization)
                        .min_samples_leaf(min_samples_leaf)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyHistGradientBoostingRegressor::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = HistGradientBoostingRegressor::<f64>::builder()
                        .max_iter(max_iter)
                        .learning_rate(learning_rate)
                        .max_depth(max_depth)
                        .n_bins(n_bins)
                        .l2_regularization(l2_regularization)
                        .min_samples_leaf(min_samples_leaf)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyHistGradientBoostingRegressor::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → length-`rows` host `Vec<f32>` (f32 fitted path).
    fn predict_f32(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| -> PyResult<Vec<f32>> {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyHistGradientBoostingRegressor::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let out = TypestatePredict::predict(est, &mut pool, &xd, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(out.to_host_metered(&mut pool))
                }
                AnyHistGradientBoostingRegressor::F64(_) => Err(dtype_mismatch(
                    "hist_gradient_boosting_regressor",
                    "f32",
                    "f64",
                )),
                _ => Err(not_fitted(
                    "hist_gradient_boosting_regressor",
                    "predict (f32 path)",
                )),
            }
        })
    }
    /// `predict(x)` → length-`rows` host `Vec<f64>` (f64 fitted path).
    fn predict_f64(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| -> PyResult<Vec<f64>> {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyHistGradientBoostingRegressor::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let out = TypestatePredict::predict(est, &mut pool, &xd, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(out.to_host_metered(&mut pool))
                }
                AnyHistGradientBoostingRegressor::F32(_) => Err(dtype_mismatch(
                    "hist_gradient_boosting_regressor",
                    "f64",
                    "f32",
                )),
                _ => Err(not_fitted(
                    "hist_gradient_boosting_regressor",
                    "predict (f64 path)",
                )),
            }
        })
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyHistGradientBoostingRegressor::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyHistGradientBoostingRegressor::Unfit { .. } => None,
            AnyHistGradientBoostingRegressor::F32(_) => Some("f32"),
            AnyHistGradientBoostingRegressor::F64(_) => Some("f64"),
        }
    }
}
