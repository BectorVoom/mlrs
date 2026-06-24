//! Kernel-family `#[pyclass]` wrappers (KERNEL-01/KERNEL-02 — PY-06 incremental
//! share): `PyKernelRidge` (fit/predict) and `PyKernelDensity`
//! (fit/`score_samples`).
//!
//! Both reuse the shipped [`any_estimator!`](crate::any_estimator) Unfit/F32/F64
//! dtype-dispatch machinery (D-06) — v2 adds ZERO new binding infrastructure.
//! Each device-compute body honors the two load-bearing contracts documented on
//! [`crate::dispatch`]:
//!
//! 1. **GIL release (PY-03).** The `mlrs_algos` trait call runs inside
//!    `py.detach(|| { … })` around a lock of the process-global pool
//!    ([`crate::global_pool`]).
//! 2. **f64 guard (D-04).** On the `FloatDtype::F64` dispatch arm,
//!    [`crate::capability::guard_f64`]`()?` runs BEFORE any upload, so f64 on an
//!    f64-incapable backend raises a clear `PyValueError` and never allocates a
//!    device buffer.
//!
//! ## Unfit stores the kernel NAME + raw hyperparameters (Open Q3)
//! A `#[pyclass]` cannot be generic over `F`, and the precision-typed
//! `Kernel<F>` / resolved bandwidth are only known once `n_features` is available
//! (the `gamma=None → 1/n_features` and `scott`/`silverman` rules, D-05/D-09). So
//! each `Unfit` arm stores the kernel NAME (a `u8` tag) + the raw scalar
//! hyperparameters (`alpha`/`gamma`/`degree`/`coef0` for KernelRidge; the
//! bandwidth spec for KernelDensity); the typed `KernelKind` / `KdKernel` /
//! `BandwidthSpec` is built at `fit`, where the algos estimator then resolves the
//! gamma / bandwidth from `n_features` (Open Q3 / D-05 gamma=None path).
//!
//! Fitted-attribute accessors are dtype-suffixed (`dual_coef_f32`/`_f64`,
//! `log_density_f32`/`_f64`) per the v2 incremental-wrap precedent (STATE.md
//! [07-07]); `bandwidth_` is a single `f64` scalar so it is single-typed.
//!
//! Tests live in `crates/mlrs-py/tests/` (AGENTS.md §2 — never an in-source
//! `#[cfg(test)] mod tests`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mlrs_algos::density::kernel_density::{BandwidthSpec, KdKernel, KernelDensity};
use mlrs_algos::kernel_ridge::kernel_ridge::{KernelKind, KernelRidge};
// Phase 16 (D-01): KernelDensity is migrated to the typestate surface — its
// consuming-self `Fit` and the `Fitted`-gated `ScoreSamples` accessor are
// imported under disambiguating `Typestate*` aliases and called via UFCS.
// KernelRidge (same file) uses INHERENT `fit`/`predict` methods (no trait glob),
// so this file references no other estimator-trait surface.
use mlrs_algos::typestate::{Fit as TypestateFit, ScoreSamples as TypestateScoreSamples};

use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::ingress::{
    as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype,
};

/// Parse a sklearn kernel name into the typed [`KernelKind`] (KernelRidge family).
/// An unknown name is a `PyValueError` (the FFI validate-before-dispatch guard,
/// T-08-05-02 sibling — the algos estimator also re-validates at `fit`).
fn parse_kernel_kind(name: &str) -> PyResult<KernelKind> {
    match name {
        "linear" => Ok(KernelKind::Linear),
        "rbf" => Ok(KernelKind::Rbf),
        "poly" | "polynomial" => Ok(KernelKind::Poly),
        "sigmoid" => Ok(KernelKind::Sigmoid),
        other => Err(PyValueError::new_err(format!(
            "KernelRidge: unknown kernel '{other}' (expected one of \
             linear/rbf/poly/sigmoid)"
        ))),
    }
}

/// Parse a sklearn KD kernel name into the typed [`KdKernel`] (KernelDensity
/// family). An unknown name is a `PyValueError`.
fn parse_kd_kernel(name: &str) -> PyResult<KdKernel> {
    match name {
        "gaussian" => Ok(KdKernel::Gaussian),
        "tophat" => Ok(KdKernel::Tophat),
        "epanechnikov" => Ok(KdKernel::Epanechnikov),
        "exponential" => Ok(KdKernel::Exponential),
        "linear" => Ok(KdKernel::Linear),
        "cosine" => Ok(KdKernel::Cosine),
        other => Err(PyValueError::new_err(format!(
            "KernelDensity: unknown kernel '{other}' (expected one of \
             gaussian/tophat/epanechnikov/exponential/linear/cosine)"
        ))),
    }
}

/// Resolve the KernelDensity `bandwidth` argument (numeric value or the
/// `'scott'`/`'silverman'` string rule) into a typed [`BandwidthSpec`]. The
/// resolved numeric bandwidth is computed at `fit` from `n_samples`/`n_features`
/// (D-09); this only selects the spec.
fn parse_bandwidth(value: &str, numeric: f64) -> PyResult<BandwidthSpec> {
    match value {
        "" | "numeric" => Ok(BandwidthSpec::Numeric(numeric)),
        "scott" => Ok(BandwidthSpec::Scott),
        "silverman" => Ok(BandwidthSpec::Silverman),
        other => Err(PyValueError::new_err(format!(
            "KernelDensity: unknown bandwidth rule '{other}' (expected a numeric \
             value, 'scott', or 'silverman')"
        ))),
    }
}

// ---------------------------------------------------------------------------
// KernelRidge — fit (X, y) + predict (X_test); dual_coef_
// ---------------------------------------------------------------------------

crate::any_estimator! {
    any:   AnyKernelRidge,
    algo:  mlrs_algos::kernel_ridge::kernel_ridge::KernelRidge,
    unfit: { kernel: String, alpha: f64, gamma: Option<f64>, degree: f64, coef0: f64 },
}

/// sklearn-compatible `KernelRidge` (kernel ridge regression, KERNEL-01).
///
/// The kernel NAME + raw `alpha`/`gamma`/`degree`/`coef0` are stored in the
/// `Unfit` arm; the typed `Kernel<F>` (with `gamma=None → 1/n_features`, D-05) is
/// built by the algos estimator at `fit` once `n_features` is known (Open Q3).
#[pyclass(name = "KernelRidge")]
pub struct PyKernelRidge {
    inner: AnyKernelRidge,
}

impl PyKernelRidge {
    /// Rust-callable default constructor for the smoke test (sklearn defaults:
    /// `kernel="linear"`, `alpha=1.0`, `gamma=None`, `degree=3`, `coef0=1`).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyKernelRidge::Unfit {
                kernel: "linear".to_string(),
                alpha: 1.0,
                gamma: None,
                degree: 3.0,
                coef0: 1.0,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyKernelRidge::Unfit { .. })
    }
}

#[pymethods]
impl PyKernelRidge {
    /// `KernelRidge(kernel="linear", alpha=1.0, gamma=None, degree=3.0, coef0=1.0)`.
    #[new]
    #[pyo3(signature = (kernel = "linear".to_string(), alpha = 1.0, gamma = None, degree = 3.0, coef0 = 1.0))]
    fn new(kernel: String, alpha: f64, gamma: Option<f64>, degree: f64, coef0: f64) -> Self {
        Self {
            inner: AnyKernelRidge::Unfit {
                kernel,
                alpha,
                gamma,
                degree,
                coef0,
            },
        }
    }

    /// Fit `(K + αI)·dual_coef_ = y` over the kernel matrix (D-06). `x` is
    /// `rows × cols` row-major; `y` is `rows × n_targets` row-major (a single
    /// target is `n_targets = 1`). GIL released (PY-03); f64 guarded on an
    /// f64-incapable backend (D-04).
    #[pyo3(signature = (x, y, rows, cols, n_targets = 1))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
        n_targets: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        let (kernel, alpha, gamma, degree, coef0) = match &self.inner {
            AnyKernelRidge::Unfit {
                kernel,
                alpha,
                gamma,
                degree,
                coef0,
            } => (kernel.clone(), *alpha, *gamma, *degree, *coef0),
            _ => ("linear".to_string(), 1.0, None, 3.0, 1.0),
        };
        let kind = parse_kernel_kind(&kernel)?;
        let fitted = py.detach(|| -> PyResult<AnyKernelRidge> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = KernelRidge::<f32>::new(
                        kind,
                        alpha as f32,
                        gamma.map(|g| g as f32),
                        degree as f32,
                        coef0 as f32,
                    );
                    est.fit(&mut pool, &xd, &yd, (rows, cols), n_targets)
                        .map_err(algo_err_to_py)?;
                    Ok(AnyKernelRidge::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est =
                        KernelRidge::<f64>::new(kind, alpha, gamma, degree, coef0);
                    est.fit(&mut pool, &xd, &yd, (rows, cols), n_targets)
                        .map_err(algo_err_to_py)?;
                    Ok(AnyKernelRidge::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x_test)` → row-major `(rows × n_targets)` host `Vec<f32>` (D-03).
    /// GIL released; `NotFitted` if not in the f32 arm.
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
                AnyKernelRidge::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let out = est
                        .predict(&mut pool, &xd, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(out.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kernel_ridge", "predict (f32 path)")),
            }
        })
    }

    /// `predict(x_test)` → row-major `(rows × n_targets)` host `Vec<f64>` (D-03).
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
                AnyKernelRidge::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let out = est
                        .predict(&mut pool, &xd, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(out.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kernel_ridge", "predict (f64 path)")),
            }
        })
    }

    /// Host copy of the fitted `dual_coef_` (row-major `n_samples × n_targets`),
    /// f32 arm. `NotFitted` if not in the f32 arm.
    fn dual_coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyKernelRidge::F32(e) => e.dual_coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("kernel_ridge", "dual_coef_ (f32)")),
        }
    }
    /// Host copy of the fitted `dual_coef_`, f64 arm.
    fn dual_coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyKernelRidge::F64(e) => e.dual_coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("kernel_ridge", "dual_coef_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyKernelRidge::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyKernelRidge::Unfit { .. } => None,
            AnyKernelRidge::F32(_) => Some("f32"),
            AnyKernelRidge::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// KernelDensity — fit (X) + score_samples (Q); log_density
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyKernelDensity,
    algo:  mlrs_algos::density::kernel_density::KernelDensity,
    unfit: { kernel: String, bandwidth_rule: String, bandwidth: f64 },
}

/// sklearn-compatible `KernelDensity` (kernel density estimation, KERNEL-02).
///
/// The kernel NAME + the bandwidth spec (numeric value or `'scott'`/`'silverman'`
/// rule) are stored in the `Unfit` arm; the resolved numeric `bandwidth_` is
/// computed by the algos estimator at `fit` from `n_samples`/`n_features` (D-09 /
/// Open Q3). `score_samples` is the one new exposed method.
#[pyclass(name = "KernelDensity")]
pub struct PyKernelDensity {
    inner: AnyKernelDensity,
}

impl PyKernelDensity {
    /// Rust-callable default constructor for the smoke test (sklearn defaults:
    /// `kernel="gaussian"`, `bandwidth=1.0`).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyKernelDensity::Unfit {
                kernel: "gaussian".to_string(),
                bandwidth_rule: "numeric".to_string(),
                bandwidth: 1.0,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyKernelDensity::Unfit { .. })
    }
}

#[pymethods]
impl PyKernelDensity {
    /// `KernelDensity(kernel="gaussian", bandwidth=1.0, bandwidth_rule="numeric")`.
    ///
    /// `bandwidth_rule` selects the spec: `"numeric"` uses `bandwidth` as-is;
    /// `"scott"` / `"silverman"` ignore `bandwidth` and resolve the host closed
    /// form at `fit` (D-09).
    #[new]
    #[pyo3(signature = (kernel = "gaussian".to_string(), bandwidth = 1.0, bandwidth_rule = "numeric".to_string()))]
    fn new(kernel: String, bandwidth: f64, bandwidth_rule: String) -> Self {
        Self {
            inner: AnyKernelDensity::Unfit {
                kernel,
                bandwidth_rule,
                bandwidth,
            },
        }
    }

    /// Fit on `x` (`rows × cols`). Stores `X_fit_` and resolves `bandwidth_`
    /// (D-09). GIL released (PY-03); f64 guarded on an f64-incapable backend
    /// (D-04).
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (kernel, bandwidth_rule, bandwidth) = match &self.inner {
            AnyKernelDensity::Unfit {
                kernel,
                bandwidth_rule,
                bandwidth,
            } => (kernel.clone(), bandwidth_rule.clone(), *bandwidth),
            _ => ("gaussian".to_string(), "numeric".to_string(), 1.0),
        };
        let kd_kernel = parse_kd_kernel(&kernel)?;
        let spec = parse_bandwidth(&bandwidth_rule, bandwidth)?;
        let fitted = py.detach(|| -> PyResult<AnyKernelDensity> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = KernelDensity::<f32>::builder()
                        .kernel(kd_kernel)
                        .bandwidth(spec)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyKernelDensity::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = KernelDensity::<f64>::builder()
                        .kernel(kd_kernel)
                        .bandwidth(spec)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyKernelDensity::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `score_samples(q)` → length-`rows` host `Vec<f32>` of log-densities (the
    /// ONE new exposed method, D-12). GIL released; `NotFitted` if not in the f32
    /// arm.
    fn score_samples_f32(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f32>> {
        let qa = capsule_to_array(q)?;
        py.detach(|| -> PyResult<Vec<f32>> {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKernelDensity::F32(est) => {
                    let qd = validated_f32(as_f32(&qa)?, &mut pool)?;
                    let out = TypestateScoreSamples::score_samples(est, &mut pool, &qd, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(out.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kernel_density", "score_samples (f32 path)")),
            }
        })
    }

    /// `score_samples(q)` → length-`rows` host `Vec<f64>` of log-densities (D-12).
    fn score_samples_f64(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        let qa = capsule_to_array(q)?;
        py.detach(|| -> PyResult<Vec<f64>> {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyKernelDensity::F64(est) => {
                    let qd = validated_f64(as_f64(&qa)?, &mut pool)?;
                    let out = TypestateScoreSamples::score_samples(est, &mut pool, &qd, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(out.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kernel_density", "score_samples (f64 path)")),
            }
        })
    }

    /// The fitted f32-arm `log_density` for a query set is materialized by
    /// [`score_samples_f32`](Self::score_samples_f32); this dtype-suffixed
    /// accessor name documents the f32 log-density egress contract (the value is
    /// produced per-call from `score_samples`, KernelDensity having no stored
    /// fitted log-density array). Kept for accessor-name symmetry with the v2
    /// dtype-suffixed precedent.
    fn log_density_f32(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f32>> {
        self.score_samples_f32(py, q, rows, cols)
    }

    /// f64-arm `log_density` accessor (see [`log_density_f32`](Self::log_density_f32)).
    fn log_density_f64(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Vec<f64>> {
        self.score_samples_f64(py, q, rows, cols)
    }

    /// The resolved numeric `bandwidth_` (`> 0`) after `fit` — a single `f64`
    /// scalar (single-typed, no dtype suffix; the algos estimator keeps it in
    /// `f64` regardless of `F`).
    fn bandwidth_(&self) -> PyResult<f64> {
        match &self.inner {
            AnyKernelDensity::F32(e) => Ok(e.bandwidth()),
            AnyKernelDensity::F64(e) => Ok(e.bandwidth()),
            _ => Err(not_fitted("kernel_density", "bandwidth_")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyKernelDensity::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyKernelDensity::Unfit { .. } => None,
            AnyKernelDensity::F32(_) => Some("f32"),
            AnyKernelDensity::F64(_) => Some("f64"),
        }
    }
}
