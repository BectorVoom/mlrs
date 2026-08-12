//! `GaussianMixture` `#[pyclass]` wrapper (MIX-01) — `PyGaussianMixture`.
//!
//! A `Fit` (unsupervised — `y = None`) surface over
//! [`mlrs_algos::mixture::gaussian_mixture::GaussianMixture`], dtype-dispatched
//! (D-06) through the macro-emitted `AnyGaussianMixture` enum.
//!
//! ## Why every method here takes the HOST slice
//! `host_fit_applicable` is unconditionally `true`
//! (`GaussianMixture::host_fit_applicable` docs), so this wrapper always calls
//! [`GaussianMixture::fit_from_host_slice`] rather than uploading through
//! `Fit::fit`. On SMALL/MEDIUM fits, or on any backend without a genuine
//! device EM engine (`mlrs_backend::prims::gmm_device` module docs), that ALSO
//! means no upload at all: the Arrow buffer the caller handed in is borrowed
//! straight through, with no `from_host` memcpy — which for a `1M × 16` design
//! is a saving of the same order as an entire sklearn `fit`
//! ([[mlrs-linear-predict-cpu]]). On a large `n` fit on cuda/rocm,
//! `fit_from_host_slice` uploads internally (once) and runs the EM loop's
//! bulk passes device-resident — the `pool` parameter this wrapper now threads
//! through via [`crate::lock_pool`] is what that internal upload uses; it is a
//! no-op borrow of the process-global pool on every OTHER shape/backend.
//!
//! `guard_f64()?` still runs on the F64 arm before anything else (D-04) so an
//! f64 request on an f64-incapable backend raises rather than silently
//! downcasting — even though this estimator's arithmetic is `f64` internally
//! regardless, its ACCESSORS return `f64` buffers and the guard is what keeps
//! the reported dtype honest.
//!
//! ## `warm_start` across the boundary
//! The Rust `fit` CONSUMES the estimator (typestate), while the wrapper rebuilds
//! from `self.params` at every `fit` (WR-02). So a `warm_start` refit cannot
//! move the fitted value in; instead the previous fit's parameter block is
//! snapshotted into `self.warm` and handed to the builder's `warm_params`
//! setter, which is exactly what `into_warm_start` does internally.

use pyo3::prelude::*;

use crate::estimators::linear::parse_device;

use mlrs_algos::mixture::bayesian_gaussian_mixture::{
    BayesianGaussianMixture, BayesianMixtureParams,
};
use mlrs_algos::mixture::gaussian_mixture::{GaussianMixture, MixtureParams};

use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::ingress::{
    as_f32, as_f64, capsule_to_array, float_dtype, host_slice_f32, host_slice_f64, FloatDtype,
};

crate::any_estimator_typestate! {
    any:   AnyGaussianMixture,
    algo:  mlrs_algos::mixture::gaussian_mixture::GaussianMixture,
    unfit: { n_components: usize },
}

/// The verbatim sklearn-named ctor hyperparameters, persisted in the wrapper so
/// a SECOND `fit` of the same object rebuilds correctly (WR-02).
#[derive(Clone)]
struct GmmParams {
    /// DEVICE-PARAM-01, a STRING until `fit` (D-09).
    device: String,
    n_components: usize,
    covariance_type: String,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    n_init: usize,
    init_params: String,
    weights_init: Option<Vec<f64>>,
    means_init: Option<Vec<f64>>,
    precisions_init: Option<Vec<f64>>,
    random_state: Option<u64>,
    warm_start: bool,
    verbose: usize,
    verbose_interval: usize,
}

/// sklearn-compatible `GaussianMixture` (MIX-01).
#[pyclass(name = "GaussianMixture")]
pub struct PyGaussianMixture {
    inner: AnyGaussianMixture,
    params: GmmParams,
    /// Snapshot of the previous fit's parameters, for a `warm_start` refit.
    warm: Option<MixtureParams>,
}

impl PyGaussianMixture {
    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyGaussianMixture::Unfit { .. })
    }
}

/// Build an unfit `GaussianMixture<F>` from the stored ctor hyperparameters.
/// Monomorphized per float width by the caller's `$float`.
macro_rules! gmm_build {
    ($float:ty, $p:expr, $warm:expr) => {
        GaussianMixture::<$float>::builder()
            .device(parse_device($p.device.as_str())?)
            .n_components($p.n_components)
            .covariance_type($p.covariance_type.clone())
            .tol($p.tol)
            .reg_covar($p.reg_covar)
            .max_iter($p.max_iter)
            .n_init($p.n_init)
            .init_params($p.init_params.clone())
            .weights_init($p.weights_init.clone())
            .means_init($p.means_init.clone())
            .precisions_init($p.precisions_init.clone())
            .random_state($p.random_state)
            .warm_start($p.warm_start)
            .verbose($p.verbose)
            .verbose_interval($p.verbose_interval)
            .warm_params($warm)
            .build::<$float>()
            .map_err(build_err_to_py)?
    };
}

/// Dispatch a `&self` scoring call that needs the fitted estimator of EITHER
/// float width, returning the same Rust type from both arms.
macro_rules! gmm_scored {
    ($self:expr, $what:expr, $x:expr, $rows:expr, $cols:expr, |$e:ident, $xh:ident| $body:expr) => {{
        let xa = capsule_to_array($x)?;
        match (&$self.inner, float_dtype(&xa)?) {
            (AnyGaussianMixture::F32($e), FloatDtype::F32) => {
                let $xh = host_slice_f32(as_f32(&xa)?)?;
                $body
            }
            (AnyGaussianMixture::F64($e), FloatDtype::F64) => {
                crate::capability::guard_f64()?;
                let $xh = host_slice_f64(as_f64(&xa)?)?;
                $body
            }
            _ => Err(not_fitted("gaussian_mixture", $what)),
        }
    }};
}

#[pymethods]
impl PyGaussianMixture {
    /// `GaussianMixture(n_components=1, covariance_type='full', tol=1e-3,
    /// reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
    /// weights_init=None, means_init=None, precisions_init=None,
    /// random_state=None, warm_start=False, verbose=0, verbose_interval=10)` —
    /// sklearn's signature one-for-one.
    #[new]
    #[pyo3(signature = (
        n_components = 1,
        covariance_type = "full".to_string(),
        tol = 1e-3,
        reg_covar = 1e-6,
        max_iter = 100,
        n_init = 1,
        init_params = "kmeans".to_string(),
        weights_init = None,
        means_init = None,
        precisions_init = None,
        random_state = None,
        warm_start = false,
        verbose = 0,
        verbose_interval = 10,
        device = "auto".to_string(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_components: usize,
        covariance_type: String,
        tol: f64,
        reg_covar: f64,
        max_iter: usize,
        n_init: usize,
        init_params: String,
        weights_init: Option<Vec<f64>>,
        means_init: Option<Vec<f64>>,
        precisions_init: Option<Vec<f64>>,
        random_state: Option<u64>,
        warm_start: bool,
        verbose: usize,
        verbose_interval: usize,
        device: String,
    ) -> Self {
        Self {
            inner: AnyGaussianMixture::Unfit { n_components },
            params: GmmParams {
                device,
                n_components,
                covariance_type,
                tol,
                reg_covar,
                max_iter,
                n_init,
                init_params,
                weights_init,
                means_init,
                precisions_init,
                random_state,
                warm_start,
                verbose,
                verbose_interval,
            },
            warm: None,
        }
    }

    /// Fit on `x` (`rows × cols`). Unsupervised — no `y`. GIL released (PY-03);
    /// f64 guarded on an f64-incapable backend (D-04). Takes the caller's Arrow
    /// buffer by reference; `pool` (the process-global `lock_pool`) is threaded
    /// through for the rare shape/backend where `fit_from_host_slice` takes the
    /// device EM engine and uploads once (module docs) — every other call never
    /// touches it.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let p = self.params.clone();
        let warm = if p.warm_start { self.warm.clone() } else { None };
        let fitted = py.detach(|| -> PyResult<AnyGaussianMixture> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let est = gmm_build!(f32, p, warm);
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    Ok(AnyGaussianMixture::F32(
                        est.fit_from_host_slice(&mut pool, xh, (rows, cols))
                            .map_err(algo_err_to_py)?,
                    ))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let est = gmm_build!(f64, p, warm);
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    Ok(AnyGaussianMixture::F64(
                        est.fit_from_host_slice(&mut pool, xh, (rows, cols))
                            .map_err(algo_err_to_py)?,
                    ))
                }
            }
        })?;
        self.warm = Some(match &fitted {
            AnyGaussianMixture::F32(e) => e.params_f64().clone(),
            AnyGaussianMixture::F64(e) => e.params_f64().clone(),
            AnyGaussianMixture::Unfit { .. } => unreachable!("fit returns a fitted arm"),
        });
        self.inner = fitted;
        Ok(())
    }

    // -- fitted attributes ------------------------------------------------- #

    fn weights_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.weights()),
            _ => Err(not_fitted("gaussian_mixture", "weights_ (f32)")),
        }
    }
    fn weights_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyGaussianMixture::F64(e) => Ok(e.weights()),
            _ => Err(not_fitted("gaussian_mixture", "weights_ (f64)")),
        }
    }
    fn means_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.means()),
            _ => Err(not_fitted("gaussian_mixture", "means_ (f32)")),
        }
    }
    fn means_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyGaussianMixture::F64(e) => Ok(e.means()),
            _ => Err(not_fitted("gaussian_mixture", "means_ (f64)")),
        }
    }
    fn covariances_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.covariances()),
            _ => Err(not_fitted("gaussian_mixture", "covariances_ (f32)")),
        }
    }
    fn covariances_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyGaussianMixture::F64(e) => Ok(e.covariances()),
            _ => Err(not_fitted("gaussian_mixture", "covariances_ (f64)")),
        }
    }
    fn precisions_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.precisions()),
            _ => Err(not_fitted("gaussian_mixture", "precisions_ (f32)")),
        }
    }
    fn precisions_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyGaussianMixture::F64(e) => Ok(e.precisions()),
            _ => Err(not_fitted("gaussian_mixture", "precisions_ (f64)")),
        }
    }
    fn precisions_cholesky_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.precisions_cholesky()),
            _ => Err(not_fitted("gaussian_mixture", "precisions_cholesky_ (f32)")),
        }
    }
    fn precisions_cholesky_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyGaussianMixture::F64(e) => Ok(e.precisions_cholesky()),
            _ => Err(not_fitted("gaussian_mixture", "precisions_cholesky_ (f64)")),
        }
    }

    /// The training-set assignment from `fit`'s terminal E-step — sklearn's
    /// `fit_predict(X)` return value, already computed.
    fn labels_(&self) -> PyResult<Vec<i32>> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.labels().to_vec()),
            AnyGaussianMixture::F64(e) => Ok(e.labels().to_vec()),
            _ => Err(not_fitted("gaussian_mixture", "labels_")),
        }
    }

    /// The sklearn SHAPE of `covariances_` / `precisions_` for the configured
    /// `covariance_type`, so the shim can reshape the flat buffer without
    /// re-deriving the rule.
    fn covariance_shape(&self) -> PyResult<Vec<usize>> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.covariance_shape()),
            AnyGaussianMixture::F64(e) => Ok(e.covariance_shape()),
            _ => Err(not_fitted("gaussian_mixture", "covariance_shape")),
        }
    }

    /// `device_` — the EM engine that actually ran (`"cpu"` / `"gpu"`), read
    /// off the fitted estimator. `None` before `fit`.
    fn device_used(&self) -> Option<&'static str> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => e.device_arm(),
            AnyGaussianMixture::F64(e) => e.device_arm(),
            _ => None,
        }
    }

    fn converged(&self) -> PyResult<bool> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.converged()),
            AnyGaussianMixture::F64(e) => Ok(e.converged()),
            _ => Err(not_fitted("gaussian_mixture", "converged_")),
        }
    }
    fn n_iter(&self) -> PyResult<usize> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.n_iter()),
            AnyGaussianMixture::F64(e) => Ok(e.n_iter()),
            _ => Err(not_fitted("gaussian_mixture", "n_iter_")),
        }
    }
    fn lower_bound(&self) -> PyResult<f64> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.lower_bound()),
            AnyGaussianMixture::F64(e) => Ok(e.lower_bound()),
            _ => Err(not_fitted("gaussian_mixture", "lower_bound_")),
        }
    }
    /// `lower_bounds_` — the winning restart's per-iteration bound trace.
    /// Always `f64` (a log-likelihood), like `lower_bound`.
    fn lower_bounds(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.lower_bounds().to_vec()),
            AnyGaussianMixture::F64(e) => Ok(e.lower_bounds().to_vec()),
            _ => Err(not_fitted("gaussian_mixture", "lower_bounds_")),
        }
    }

    fn n_features_in(&self) -> PyResult<usize> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.n_features_in()),
            AnyGaussianMixture::F64(e) => Ok(e.n_features_in()),
            _ => Err(not_fitted("gaussian_mixture", "n_features_in_")),
        }
    }
    fn n_parameters(&self) -> PyResult<usize> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => Ok(e.n_parameters()),
            AnyGaussianMixture::F64(e) => Ok(e.n_parameters()),
            _ => Err(not_fitted("gaussian_mixture", "_n_parameters")),
        }
    }

    // -- scoring ----------------------------------------------------------- #

    /// `predict(X)` — the argmax component per row.
    fn predict_labels(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<i32>> {
        gmm_scored!(self, "predict", x, rows, cols, |e, xh| py
            .detach(|| e.predict_labels_host(xh, (rows, cols)))
            .map_err(algo_err_to_py))
    }

    fn predict_proba_f32(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        match &self.inner {
            AnyGaussianMixture::F32(e) => {
                let xh = host_slice_f32(as_f32(&xa)?)?;
                py.detach(|| e.predict_proba_host(xh, (rows, cols)))
                    .map_err(algo_err_to_py)
            }
            _ => Err(not_fitted("gaussian_mixture", "predict_proba (f32)")),
        }
    }
    fn predict_proba_f64(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        crate::capability::guard_f64()?;
        let xa = capsule_to_array(x)?;
        match &self.inner {
            AnyGaussianMixture::F64(e) => {
                let xh = host_slice_f64(as_f64(&xa)?)?;
                py.detach(|| e.predict_proba_host(xh, (rows, cols)))
                    .map_err(algo_err_to_py)
            }
            _ => Err(not_fitted("gaussian_mixture", "predict_proba (f64)")),
        }
    }
    fn predict_log_proba_f32(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        match &self.inner {
            AnyGaussianMixture::F32(e) => {
                let xh = host_slice_f32(as_f32(&xa)?)?;
                py.detach(|| e.predict_log_proba_host(xh, (rows, cols)))
                    .map_err(algo_err_to_py)
            }
            _ => Err(not_fitted("gaussian_mixture", "predict_log_proba (f32)")),
        }
    }
    fn predict_log_proba_f64(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        crate::capability::guard_f64()?;
        let xa = capsule_to_array(x)?;
        match &self.inner {
            AnyGaussianMixture::F64(e) => {
                let xh = host_slice_f64(as_f64(&xa)?)?;
                py.detach(|| e.predict_log_proba_host(xh, (rows, cols)))
                    .map_err(algo_err_to_py)
            }
            _ => Err(not_fitted("gaussian_mixture", "predict_log_proba (f64)")),
        }
    }

    /// `score_samples(X)` — the per-row log-density. Always `f64`: it is a
    /// log-likelihood, which sklearn also returns at full precision.
    fn score_samples(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        gmm_scored!(self, "score_samples", x, rows, cols, |e, xh| py
            .detach(|| e.score_samples_host(xh, (rows, cols)))
            .map_err(algo_err_to_py))
    }

    /// `score(X)` — the mean per-sample log-density.
    fn score(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<f64> {
        gmm_scored!(self, "score", x, rows, cols, |e, xh| py
            .detach(|| e.score_host(xh, (rows, cols)))
            .map_err(algo_err_to_py))
    }

    fn bic(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<f64> {
        gmm_scored!(self, "bic", x, rows, cols, |e, xh| py
            .detach(|| e.bic(xh, (rows, cols)))
            .map_err(algo_err_to_py))
    }

    fn aic(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<f64> {
        gmm_scored!(self, "aic", x, rows, cols, |e, xh| py
            .detach(|| e.aic(xh, (rows, cols)))
            .map_err(algo_err_to_py))
    }

    /// `sample(n_samples)` — returns `(flat X, y)`. `seed` replaces sklearn's
    /// `random_state`, whose numpy stream is not reproducible from Rust.
    fn sample_f32(&self, n_samples: usize, seed: u64) -> PyResult<(Vec<f32>, Vec<i32>)> {
        match &self.inner {
            AnyGaussianMixture::F32(e) => e.sample(n_samples, seed).map_err(algo_err_to_py),
            _ => Err(not_fitted("gaussian_mixture", "sample (f32)")),
        }
    }
    fn sample_f64(&self, n_samples: usize, seed: u64) -> PyResult<(Vec<f64>, Vec<i32>)> {
        match &self.inner {
            AnyGaussianMixture::F64(e) => e.sample(n_samples, seed).map_err(algo_err_to_py),
            _ => Err(not_fitted("gaussian_mixture", "sample (f64)")),
        }
    }

    /// The fitted arm's dtype (`"f32"` / `"f64"`, `None` while unfit) — what the
    /// shim's `MlrsBase._suffix` reads to pick the dtype-suffixed accessor.
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyGaussianMixture::Unfit { .. } => None,
            AnyGaussianMixture::F32(_) => Some("f32"),
            AnyGaussianMixture::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// BayesianGaussianMixture (MIX-02)
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyBayesianGaussianMixture,
    algo:  mlrs_algos::mixture::bayesian_gaussian_mixture::BayesianGaussianMixture,
    unfit: { n_components: usize },
}

/// The verbatim sklearn-named ctor hyperparameters, persisted so a SECOND `fit`
/// of the same object rebuilds correctly (WR-02).
#[derive(Clone)]
struct BgmParams {
    /// DEVICE-PARAM-01, a STRING until `fit` (D-09).
    device: String,
    n_components: usize,
    covariance_type: String,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    n_init: usize,
    init_params: String,
    weight_concentration_prior_type: String,
    weight_concentration_prior: Option<f64>,
    mean_precision_prior: Option<f64>,
    mean_prior: Option<Vec<f64>>,
    degrees_of_freedom_prior: Option<f64>,
    covariance_prior: Option<Vec<f64>>,
    random_state: Option<u64>,
    warm_start: bool,
    verbose: usize,
    verbose_interval: usize,
}

/// sklearn-compatible `BayesianGaussianMixture` (MIX-02).
///
/// Structurally identical to [`PyGaussianMixture`] — same host-only ingress,
/// same dtype dispatch, same `warm_start` snapshot dance — with the four
/// variational posteriors and five resolved priors added to the accessor
/// surface, and `bic` / `aic` / the three `*_init` injections removed because
/// sklearn does not define them here.
#[pyclass(name = "BayesianGaussianMixture")]
pub struct PyBayesianGaussianMixture {
    inner: AnyBayesianGaussianMixture,
    params: BgmParams,
    /// Snapshot of the previous fit's posterior, for a `warm_start` refit.
    warm: Option<BayesianMixtureParams>,
}

impl PyBayesianGaussianMixture {
    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyBayesianGaussianMixture::Unfit { .. })
    }
}

/// Build an unfit `BayesianGaussianMixture<F>` from the stored ctor
/// hyperparameters. Monomorphized per float width by the caller's `$float`.
macro_rules! bgm_build {
    ($float:ty, $p:expr, $warm:expr) => {
        BayesianGaussianMixture::<$float>::builder()
            .device(parse_device($p.device.as_str())?)
            .n_components($p.n_components)
            .covariance_type($p.covariance_type.clone())
            .tol($p.tol)
            .reg_covar($p.reg_covar)
            .max_iter($p.max_iter)
            .n_init($p.n_init)
            .init_params($p.init_params.clone())
            .weight_concentration_prior_type($p.weight_concentration_prior_type.clone())
            .weight_concentration_prior($p.weight_concentration_prior)
            .mean_precision_prior($p.mean_precision_prior)
            .mean_prior($p.mean_prior.clone())
            .degrees_of_freedom_prior($p.degrees_of_freedom_prior)
            .covariance_prior($p.covariance_prior.clone())
            .random_state($p.random_state)
            .warm_start($p.warm_start)
            .verbose($p.verbose)
            .verbose_interval($p.verbose_interval)
            .warm_params($warm)
            .build::<$float>()
            .map_err(build_err_to_py)?
    };
}

/// Dispatch a `&self` scoring call that needs the fitted estimator of EITHER
/// float width, returning the same Rust type from both arms.
macro_rules! bgm_scored {
    ($self:expr, $what:expr, $x:expr, $rows:expr, $cols:expr, |$e:ident, $xh:ident| $body:expr) => {{
        let xa = capsule_to_array($x)?;
        match (&$self.inner, float_dtype(&xa)?) {
            (AnyBayesianGaussianMixture::F32($e), FloatDtype::F32) => {
                let $xh = host_slice_f32(as_f32(&xa)?)?;
                $body
            }
            (AnyBayesianGaussianMixture::F64($e), FloatDtype::F64) => {
                crate::capability::guard_f64()?;
                let $xh = host_slice_f64(as_f64(&xa)?)?;
                $body
            }
            _ => Err(not_fitted("bayesian_gaussian_mixture", $what)),
        }
    }};
}

/// Read one `f64` buffer off whichever fitted arm is live.
macro_rules! bgm_attr_f64 {
    ($self:expr, $what:expr, |$e:ident| $body:expr) => {
        match &$self.inner {
            AnyBayesianGaussianMixture::F32($e) => Ok($body),
            AnyBayesianGaussianMixture::F64($e) => Ok($body),
            _ => Err(not_fitted("bayesian_gaussian_mixture", $what)),
        }
    };
}

#[pymethods]
impl PyBayesianGaussianMixture {
    /// sklearn's signature one-for-one. Note `n_components` is KEYWORD-ONLY in
    /// sklearn's `BayesianGaussianMixture` (unlike `GaussianMixture`, where it
    /// is positional); the Python shim reproduces that, and this wrapper takes
    /// it positionally because the shim always passes every argument.
    #[new]
    #[pyo3(signature = (
        n_components = 1,
        covariance_type = "full".to_string(),
        tol = 1e-3,
        reg_covar = 1e-6,
        max_iter = 100,
        n_init = 1,
        init_params = "kmeans".to_string(),
        weight_concentration_prior_type = "dirichlet_process".to_string(),
        weight_concentration_prior = None,
        mean_precision_prior = None,
        mean_prior = None,
        degrees_of_freedom_prior = None,
        covariance_prior = None,
        random_state = None,
        warm_start = false,
        verbose = 0,
        verbose_interval = 10,
        device = "auto".to_string(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_components: usize,
        covariance_type: String,
        tol: f64,
        reg_covar: f64,
        max_iter: usize,
        n_init: usize,
        init_params: String,
        weight_concentration_prior_type: String,
        weight_concentration_prior: Option<f64>,
        mean_precision_prior: Option<f64>,
        mean_prior: Option<Vec<f64>>,
        degrees_of_freedom_prior: Option<f64>,
        covariance_prior: Option<Vec<f64>>,
        random_state: Option<u64>,
        warm_start: bool,
        verbose: usize,
        verbose_interval: usize,
        device: String,
    ) -> Self {
        Self {
            inner: AnyBayesianGaussianMixture::Unfit { n_components },
            params: BgmParams {
                device,
                n_components,
                covariance_type,
                tol,
                reg_covar,
                max_iter,
                n_init,
                init_params,
                weight_concentration_prior_type,
                weight_concentration_prior,
                mean_precision_prior,
                mean_prior,
                degrees_of_freedom_prior,
                covariance_prior,
                random_state,
                warm_start,
                verbose,
                verbose_interval,
            },
            warm: None,
        }
    }

    /// Fit on `x` (`rows × cols`). Unsupervised — no `y`. GIL released (PY-03);
    /// f64 guarded on an f64-incapable backend (D-04). The design is borrowed
    /// straight from the caller's Arrow buffer and never uploaded: this
    /// estimator's engine is host-resident on every backend.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let p = self.params.clone();
        let warm = if p.warm_start { self.warm.clone() } else { None };
        let fitted = py.detach(|| -> PyResult<AnyBayesianGaussianMixture> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let est = bgm_build!(f32, p, warm);
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    Ok(AnyBayesianGaussianMixture::F32(
                        est.fit_from_host_slice(&mut pool, xh, (rows, cols))
                            .map_err(algo_err_to_py)?,
                    ))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let est = bgm_build!(f64, p, warm);
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    Ok(AnyBayesianGaussianMixture::F64(
                        est.fit_from_host_slice(&mut pool, xh, (rows, cols))
                            .map_err(algo_err_to_py)?,
                    ))
                }
            }
        })?;
        self.warm = Some(match &fitted {
            AnyBayesianGaussianMixture::F32(e) => e.params_f64().clone(),
            AnyBayesianGaussianMixture::F64(e) => e.params_f64().clone(),
            AnyBayesianGaussianMixture::Unfit { .. } => unreachable!("fit returns a fitted arm"),
        });
        self.inner = fitted;
        Ok(())
    }

    // -- fitted attributes ------------------------------------------------- #

    fn weights_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyBayesianGaussianMixture::F32(e) => Ok(e.weights()),
            _ => Err(not_fitted("bayesian_gaussian_mixture", "weights_ (f32)")),
        }
    }
    fn weights_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyBayesianGaussianMixture::F64(e) => Ok(e.weights()),
            _ => Err(not_fitted("bayesian_gaussian_mixture", "weights_ (f64)")),
        }
    }
    fn means_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyBayesianGaussianMixture::F32(e) => Ok(e.means()),
            _ => Err(not_fitted("bayesian_gaussian_mixture", "means_ (f32)")),
        }
    }
    fn means_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyBayesianGaussianMixture::F64(e) => Ok(e.means()),
            _ => Err(not_fitted("bayesian_gaussian_mixture", "means_ (f64)")),
        }
    }
    fn covariances_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyBayesianGaussianMixture::F32(e) => Ok(e.covariances()),
            _ => Err(not_fitted("bayesian_gaussian_mixture", "covariances_ (f32)")),
        }
    }
    fn covariances_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyBayesianGaussianMixture::F64(e) => Ok(e.covariances()),
            _ => Err(not_fitted("bayesian_gaussian_mixture", "covariances_ (f64)")),
        }
    }
    fn precisions_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyBayesianGaussianMixture::F32(e) => Ok(e.precisions()),
            _ => Err(not_fitted("bayesian_gaussian_mixture", "precisions_ (f32)")),
        }
    }
    fn precisions_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyBayesianGaussianMixture::F64(e) => Ok(e.precisions()),
            _ => Err(not_fitted("bayesian_gaussian_mixture", "precisions_ (f64)")),
        }
    }
    fn precisions_cholesky_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyBayesianGaussianMixture::F32(e) => Ok(e.precisions_cholesky()),
            _ => Err(not_fitted(
                "bayesian_gaussian_mixture",
                "precisions_cholesky_ (f32)",
            )),
        }
    }
    fn precisions_cholesky_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyBayesianGaussianMixture::F64(e) => Ok(e.precisions_cholesky()),
            _ => Err(not_fitted(
                "bayesian_gaussian_mixture",
                "precisions_cholesky_ (f64)",
            )),
        }
    }

    /// `weight_concentration_` as `(a, b)`. `b` is EMPTY under
    /// `dirichlet_distribution`, which the shim turns back into sklearn's
    /// single-array form.
    ///
    /// Always `f64`: these are concentration parameters, not design-dtype
    /// quantities, and sklearn keeps them at full precision too.
    fn weight_concentration(&self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        bgm_attr_f64!(self, "weight_concentration_", |e| {
            let (a, b) = e.weight_concentration();
            (a.to_vec(), b.to_vec())
        })
    }
    fn mean_precision(&self) -> PyResult<Vec<f64>> {
        bgm_attr_f64!(self, "mean_precision_", |e| e.mean_precision().to_vec())
    }
    fn degrees_of_freedom(&self) -> PyResult<Vec<f64>> {
        bgm_attr_f64!(self, "degrees_of_freedom_", |e| e
            .degrees_of_freedom()
            .to_vec())
    }
    /// sklearn's SHAPE for `degrees_of_freedom_`: empty (a scalar) under
    /// `covariance_type='tied'`, `[n_components]` otherwise.
    fn degrees_of_freedom_shape(&self) -> PyResult<Vec<usize>> {
        bgm_attr_f64!(self, "degrees_of_freedom_", |e| e.degrees_of_freedom_shape())
    }

    /// The five resolved `*_prior_` attributes, in sklearn's ctor order:
    /// `(weight_concentration_prior_, mean_precision_prior_, mean_prior_,
    /// degrees_of_freedom_prior_, covariance_prior_)`.
    #[allow(clippy::type_complexity)]
    fn priors(&self) -> PyResult<(f64, f64, Vec<f64>, f64, Vec<f64>)> {
        bgm_attr_f64!(self, "*_prior_", |e| {
            let p = e.priors();
            (
                p.weight_concentration,
                p.mean_precision,
                p.mean.clone(),
                p.degrees_of_freedom,
                p.covariance.clone(),
            )
        })
    }

    /// The training-set assignment from `fit`'s terminal E-step.
    fn labels_(&self) -> PyResult<Vec<i32>> {
        bgm_attr_f64!(self, "labels_", |e| e.labels().to_vec())
    }

    /// The sklearn SHAPE of `covariances_` / `precisions_`.
    fn covariance_shape(&self) -> PyResult<Vec<usize>> {
        bgm_attr_f64!(self, "covariance_shape", |e| e.covariance_shape())
    }

    /// `device_` — the EM engine that actually ran (`"cpu"` / `"gpu"`), read
    /// off the fitted estimator. `None` before `fit`.
    fn device_used(&self) -> Option<&'static str> {
        match &self.inner {
            AnyBayesianGaussianMixture::F32(e) => e.device_arm(),
            AnyBayesianGaussianMixture::F64(e) => e.device_arm(),
            _ => None,
        }
    }

    fn converged(&self) -> PyResult<bool> {
        bgm_attr_f64!(self, "converged_", |e| e.converged())
    }
    fn n_iter(&self) -> PyResult<usize> {
        bgm_attr_f64!(self, "n_iter_", |e| e.n_iter())
    }
    fn lower_bound(&self) -> PyResult<f64> {
        bgm_attr_f64!(self, "lower_bound_", |e| e.lower_bound())
    }
    fn lower_bounds(&self) -> PyResult<Vec<f64>> {
        bgm_attr_f64!(self, "lower_bounds_", |e| e.lower_bounds().to_vec())
    }
    fn n_features_in(&self) -> PyResult<usize> {
        bgm_attr_f64!(self, "n_features_in_", |e| e.n_features_in())
    }

    // -- scoring ----------------------------------------------------------- #

    /// `predict(X)` — the argmax component per row.
    fn predict_labels(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<i32>> {
        bgm_scored!(self, "predict", x, rows, cols, |e, xh| py
            .detach(|| e.predict_labels_host(xh, (rows, cols)))
            .map_err(algo_err_to_py))
    }

    fn predict_proba_f32(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        match &self.inner {
            AnyBayesianGaussianMixture::F32(e) => {
                let xh = host_slice_f32(as_f32(&xa)?)?;
                py.detach(|| e.predict_proba_host(xh, (rows, cols)))
                    .map_err(algo_err_to_py)
            }
            _ => Err(not_fitted(
                "bayesian_gaussian_mixture",
                "predict_proba (f32)",
            )),
        }
    }
    fn predict_proba_f64(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        crate::capability::guard_f64()?;
        let xa = capsule_to_array(x)?;
        match &self.inner {
            AnyBayesianGaussianMixture::F64(e) => {
                let xh = host_slice_f64(as_f64(&xa)?)?;
                py.detach(|| e.predict_proba_host(xh, (rows, cols)))
                    .map_err(algo_err_to_py)
            }
            _ => Err(not_fitted(
                "bayesian_gaussian_mixture",
                "predict_proba (f64)",
            )),
        }
    }
    fn predict_log_proba_f32(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        match &self.inner {
            AnyBayesianGaussianMixture::F32(e) => {
                let xh = host_slice_f32(as_f32(&xa)?)?;
                py.detach(|| e.predict_log_proba_host(xh, (rows, cols)))
                    .map_err(algo_err_to_py)
            }
            _ => Err(not_fitted(
                "bayesian_gaussian_mixture",
                "predict_log_proba (f32)",
            )),
        }
    }
    fn predict_log_proba_f64(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        crate::capability::guard_f64()?;
        let xa = capsule_to_array(x)?;
        match &self.inner {
            AnyBayesianGaussianMixture::F64(e) => {
                let xh = host_slice_f64(as_f64(&xa)?)?;
                py.detach(|| e.predict_log_proba_host(xh, (rows, cols)))
                    .map_err(algo_err_to_py)
            }
            _ => Err(not_fitted(
                "bayesian_gaussian_mixture",
                "predict_log_proba (f64)",
            )),
        }
    }

    /// `score_samples(X)` — the per-row log-density. Always `f64`.
    fn score_samples(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        bgm_scored!(self, "score_samples", x, rows, cols, |e, xh| py
            .detach(|| e.score_samples_host(xh, (rows, cols)))
            .map_err(algo_err_to_py))
    }

    /// `score(X)` — the mean per-sample log-density.
    fn score(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<f64> {
        bgm_scored!(self, "score", x, rows, cols, |e, xh| py
            .detach(|| e.score_host(xh, (rows, cols)))
            .map_err(algo_err_to_py))
    }

    /// `sample(n_samples)` — returns `(flat X, y)`. `seed` replaces sklearn's
    /// `random_state`, whose numpy stream is not reproducible from Rust.
    fn sample_f32(&self, n_samples: usize, seed: u64) -> PyResult<(Vec<f32>, Vec<i32>)> {
        match &self.inner {
            AnyBayesianGaussianMixture::F32(e) => {
                e.sample(n_samples, seed).map_err(algo_err_to_py)
            }
            _ => Err(not_fitted("bayesian_gaussian_mixture", "sample (f32)")),
        }
    }
    fn sample_f64(&self, n_samples: usize, seed: u64) -> PyResult<(Vec<f64>, Vec<i32>)> {
        match &self.inner {
            AnyBayesianGaussianMixture::F64(e) => {
                e.sample(n_samples, seed).map_err(algo_err_to_py)
            }
            _ => Err(not_fitted("bayesian_gaussian_mixture", "sample (f64)")),
        }
    }

    /// The fitted arm's dtype (`"f32"` / `"f64"`, `None` while unfit).
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyBayesianGaussianMixture::Unfit { .. } => None,
            AnyBayesianGaussianMixture::F32(_) => Some("f32"),
            AnyBayesianGaussianMixture::F64(_) => Some("f64"),
        }
    }
}
