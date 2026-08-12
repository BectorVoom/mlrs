//! Manifold-learning `#[pyclass]` wrapper (PY-01/PY-02 — BLDR-04): `PyUMAP`.
//!
//! `PyUMAP` is the FIRST PyO3 shell over a v3 TYPESTATE estimator
//! ([`mlrs_algos::manifold::umap::Umap<F, S = Unfit>`], Plan 02). It proves the
//! Phase-12 builder + typestate convention is INVISIBLE to Python (BLDR-04): the
//! same `Unfit/F32/F64` dtype-dispatch enum the 30 legacy estimators use (here
//! emitted by [`any_estimator_typestate!`](crate::any_estimator_typestate) whose
//! fitted arms spell `<F, Fitted>` explicitly — D-04), the same error mappers
//! ([`algo_err_to_py`]/[`build_err_to_py`]/[`not_fitted`], reused UNCHANGED —
//! D-13), and the same `py.detach` + `lock_pool` + `guard_f64`-on-F64 fit
//! contract as [`crate::estimators::linear::PyLinearRegression`].
//!
//! The one structural difference from the legacy shells: the fitted arm stores a
//! `Umap<F, Fitted>` (the consuming [`Fit::fit`] returns the `Fitted`-tagged
//! sibling), and the `fit` body is STRICTLY simpler than the legacy `&mut self`
//! form — it builds the `Unfit` estimator, consumes it through `fit`, and stores
//! the returned `Fitted` value. The compile-time typestate makes a
//! transform-before-fit a COMPILE error on the Rust side (Plan 03's gate); at the
//! Python boundary (which has no compile guarantee) the `Unfit`-arm accessor
//! returns the runtime [`not_fitted`] analog → `PyValueError` (D-13).
//!
//! UMAP is unsupervised — `fit` takes no `y`. The fitted surface is the
//! `embedding_` accessor (the trivial Phase-12 shell emits an all-zeros
//! `(n, n_components)` buffer; real UMAP lands in Phase 14).
//!
//! Tests live in `crates/mlrs-py/tests/manifold_test.rs` (AGENTS.md §2).

use pyo3::prelude::*;

use crate::estimators::linear::parse_device;

use mlrs_algos::manifold::umap::{Init, Metric, Umap};
use mlrs_algos::typestate::{Fit, Transform};

use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

crate::any_estimator_typestate! {
    any:   AnyUmap,
    algo:  mlrs_algos::manifold::umap::Umap,
    unfit: {
        n_neighbors: usize, n_components: usize, min_dist: f64, spread: f64,
        metric: String, n_epochs: Option<usize>, init: String,
        random_state: Option<u64>, learning_rate: f64, set_op_mix_ratio: f64,
        local_connectivity: f64, repulsion_strength: f64,
        negative_sample_rate: usize, a: Option<f64>, b: Option<f64>,
        device: String,
    },
}

/// Parse the sklearn-named `metric` string into the algos [`Metric`] enum
/// (data-INDEPENDENT — surfaces as a `PyValueError` via [`build_err_to_py`]-style
/// mapping). Only `"euclidean"` carries meaning in the Phase-12 shell.
fn parse_metric(s: &str) -> PyResult<Metric> {
    match s {
        "euclidean" => Ok(Metric::Euclidean),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "umap: unsupported metric {other:?}; expected \"euclidean\""
        ))),
    }
}

/// Parse the sklearn-named `init` string into the algos [`Init`] enum.
fn parse_init(s: &str) -> PyResult<Init> {
    match s {
        "spectral" => Ok(Init::Spectral),
        "random" => Ok(Init::Random),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "umap: unsupported init {other:?}; expected \"spectral\" or \"random\""
        ))),
    }
}

/// sklearn/umap-learn-compatible `UMAP` (manifold learning). The v3 typestate
/// estimator collapses behind the same `Unfit/F32/F64` enum the legacy shells use
/// (BLDR-04) — the convention is invisible to Python.
#[pyclass(name = "UMAP")]
pub struct PyUMAP {
    inner: AnyUmap,
}

impl PyUMAP {
    /// Rust-callable default constructor (cross-crate smoke seam — proves the
    /// macro-expanded wrapper instantiates in the `Unfit` arm WITHOUT a Python
    /// interpreter). Mirrors the `#[new]` defaults (umap-learn defaults). See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyUmap::Unfit {
                device: "auto".to_string(),
                n_neighbors: 15,
                n_components: 2,
                min_dist: 0.1,
                spread: 1.0,
                metric: "euclidean".to_string(),
                n_epochs: None,
                init: "spectral".to_string(),
                random_state: None,
                learning_rate: 1.0,
                set_op_mix_ratio: 1.0,
                local_connectivity: 1.0,
                repulsion_strength: 1.0,
                negative_sample_rate: 5,
                a: None,
                b: None,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyUmap::Unfit { .. })
    }

    /// Rust-callable `embedding_` (f32 arm) accessor for the cross-crate
    /// not-fitted test (the live PyO3 boundary path runs in UAT, MEMORY). Returns
    /// the [`not_fitted`] analog on the `Unfit` arm.
    pub fn embedding_f32_for_test(&self) -> PyResult<Vec<f32>> {
        self.embedding_f32_inner()
    }

    /// Extract the dtype-INDEPENDENT hyperparameter tuple from the `Unfit` arm for
    /// the `fit_transform_*` builder path. Returns the [`not_fitted`] analog on an
    /// already-fitted arm (sklearn `fit_transform` re-fits from the constructed
    /// estimator; re-fit on a fitted wrap is the same boundary error as `fit`).
    #[allow(clippy::type_complexity)]
    fn unfit_hyperparams(
        &self,
    ) -> PyResult<(
        usize, usize, f64, f64, String, Option<usize>, String, Option<u64>, f64, f64, f64, f64, usize, Option<f64>, Option<f64>, String,
    )> {
        match &self.inner {
            AnyUmap::Unfit {
                n_neighbors, n_components, min_dist, spread, metric, n_epochs,
                init, random_state, learning_rate, set_op_mix_ratio,
                local_connectivity, repulsion_strength, negative_sample_rate, a, b,
                device,
            } => Ok((
                *n_neighbors, *n_components, *min_dist, *spread, metric.clone(),
                *n_epochs, init.clone(), *random_state, *learning_rate,
                *set_op_mix_ratio, *local_connectivity, *repulsion_strength,
                *negative_sample_rate, *a, *b, device.clone(),
            )),
            _ => Err(not_fitted("umap", "fit_transform (re-fit)")),
        }
    }

    fn embedding_f32_inner(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyUmap::F32(e) => Ok(e.embedding(&pool)),
            _ => Err(not_fitted("umap", "embedding_ (f32)")),
        }
    }

    fn embedding_f64_inner(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyUmap::F64(e) => Ok(e.embedding(&pool)),
            _ => Err(not_fitted("umap", "embedding_ (f64)")),
        }
    }
}

#[pymethods]
impl PyUMAP {
    /// `UMAP(n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0,
    /// metric="euclidean", n_epochs=None, init="spectral", random_state=None,
    /// learning_rate=1.0, set_op_mix_ratio=1.0, local_connectivity=1.0,
    /// repulsion_strength=1.0, negative_sample_rate=5, a=None, b=None)`.
    #[new]
    #[pyo3(signature = (
        n_neighbors = 15, n_components = 2, min_dist = 0.1, spread = 1.0,
        metric = "euclidean".to_string(), n_epochs = None,
        init = "spectral".to_string(), random_state = None, learning_rate = 1.0,
        set_op_mix_ratio = 1.0, local_connectivity = 1.0, repulsion_strength = 1.0,
        negative_sample_rate = 5, a = None, b = None,
        device = "auto".to_string(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_neighbors: usize,
        n_components: usize,
        min_dist: f64,
        spread: f64,
        metric: String,
        n_epochs: Option<usize>,
        init: String,
        random_state: Option<u64>,
        learning_rate: f64,
        set_op_mix_ratio: f64,
        local_connectivity: f64,
        repulsion_strength: f64,
        negative_sample_rate: usize,
        a: Option<f64>,
        b: Option<f64>,
        device: String,
    ) -> Self {
        Self {
            inner: AnyUmap::Unfit {
                device,
                n_neighbors,
                n_components,
                min_dist,
                spread,
                metric,
                n_epochs,
                init,
                random_state,
                learning_rate,
                set_op_mix_ratio,
                local_connectivity,
                repulsion_strength,
                negative_sample_rate,
                a,
                b,
            },
        }
    }

    /// Fit on `x` (`rows × cols`, row-major). Unsupervised — no `y`. The
    /// data-INDEPENDENT hyperparameters are validated at `build()` BEFORE the
    /// device upload (`build_err_to_py` → `ValueError`, T-12-02); GIL released
    /// (PY-03); f64 guarded on an f64-incapable backend (D-04 / T-12-07). The
    /// consuming `typestate::Fit::fit` returns the `Fitted`-tagged sibling, stored
    /// in the matching dtype arm.
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (
            n_neighbors, n_components, min_dist, spread, metric_s, n_epochs,
            init_s, random_state, learning_rate, set_op_mix_ratio,
            local_connectivity, repulsion_strength, negative_sample_rate, a, b,
            device_s,
        ) = match &self.inner {
            AnyUmap::Unfit {
                n_neighbors, n_components, min_dist, spread, metric, n_epochs,
                init, random_state, learning_rate, set_op_mix_ratio,
                local_connectivity, repulsion_strength, negative_sample_rate, a, b,
                device,
            } => (
                *n_neighbors, *n_components, *min_dist, *spread, metric.clone(),
                *n_epochs, init.clone(), *random_state, *learning_rate,
                *set_op_mix_ratio, *local_connectivity, *repulsion_strength,
                *negative_sample_rate, *a, *b, device.clone(),
            ),
            _ => return Err(not_fitted("umap", "re-fit")),
        };
        // Construction-time enum-string validation (→ ValueError).
        let metric = parse_metric(&metric_s)?;
        let init = parse_init(&init_s)?;
        let fitted = py.detach(|| -> PyResult<AnyUmap> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = Umap::<f32>::builder()
                        .device(parse_device(&device_s)?)
                        .n_neighbors(n_neighbors)
                        .n_components(n_components)
                        .min_dist(min_dist)
                        .spread(spread)
                        .metric(metric)
                        .n_epochs(n_epochs)
                        .init(init)
                        .random_state(random_state)
                        .learning_rate(learning_rate)
                        .set_op_mix_ratio(set_op_mix_ratio)
                        .local_connectivity(local_connectivity)
                        .repulsion_strength(repulsion_strength)
                        .negative_sample_rate(negative_sample_rate)
                        .a(a)
                        .b(b)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyUmap::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = Umap::<f64>::builder()
                        .device(parse_device(&device_s)?)
                        .n_neighbors(n_neighbors)
                        .n_components(n_components)
                        .min_dist(min_dist)
                        .spread(spread)
                        .metric(metric)
                        .n_epochs(n_epochs)
                        .init(init)
                        .random_state(random_state)
                        .learning_rate(learning_rate)
                        .set_op_mix_ratio(set_op_mix_ratio)
                        .local_connectivity(local_connectivity)
                        .repulsion_strength(repulsion_strength)
                        .negative_sample_rate(negative_sample_rate)
                        .a(a)
                        .b(b)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyUmap::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `transform(X_new)` (f32 arm) — embed `rows × cols` NEW points against the
    /// fitted fuzzy graph and return the host `(rows × n_components)` embedding
    /// (row-major). Forwards to the Rust `Umap<f32, Fitted>` [`Transform::transform`]
    /// (umap.rs:568). GIL released (PY-03); the `Unfit`/wrong-dtype arm returns the
    /// runtime [`not_fitted`] analog (D-13). The fit-time `n_features_in_` is
    /// enforced inside the Rust method (T-16-V5 / WR-02).
    fn transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyUmap::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(Transform::transform(est, &mut pool, &xd, (rows, cols))
                        .map_err(algo_err_to_py)?
                        .to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("umap", "transform (f32 path)")),
            }
        })
    }

    /// `transform(X_new)` (f64 arm) — see [`Self::transform_f32`]. `guard_f64()` is
    /// applied BEFORE the device upload on the f64-incapable backend (T-16-GUARDF64).
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyUmap::F64(est) => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(Transform::transform(est, &mut pool, &xd, (rows, cols))
                        .map_err(algo_err_to_py)?
                        .to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("umap", "transform (f64 path)")),
            }
        })
    }

    /// `fit_transform(X)` (f32 path) — fit to `x` (`rows × cols`, row-major) and
    /// return the fitted embedding host buffer in one call (umap-learn's
    /// `UMAP.fit_transform`). Unsupervised — no `y`. Builds via the umap builder
    /// (data-INDEPENDENT validation → `ValueError`, T-12-02), releases the GIL
    /// (PY-03), and consumes the `Unfit` estimator through
    /// `Umap::<f32, Unfit>::fit_transform` (umap.rs:215). Does NOT mutate `self`
    /// (the fitted estimator is dropped — sklearn `fit_transform` semantics return
    /// the embedding; call `fit` then `embedding_f32` to retain the estimator).
    fn fit_transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        let (
            n_neighbors, n_components, min_dist, spread, metric_s, n_epochs,
            init_s, random_state, learning_rate, set_op_mix_ratio,
            local_connectivity, repulsion_strength, negative_sample_rate, a, b,
            device_s,
        ) = self.unfit_hyperparams()?;
        let metric = parse_metric(&metric_s)?;
        let init = parse_init(&init_s)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
            let est = Umap::<f32>::builder()
                        .device(parse_device(&device_s)?)
                .n_neighbors(n_neighbors)
                .n_components(n_components)
                .min_dist(min_dist)
                .spread(spread)
                .metric(metric)
                .n_epochs(n_epochs)
                .init(init)
                .random_state(random_state)
                .learning_rate(learning_rate)
                .set_op_mix_ratio(set_op_mix_ratio)
                .local_connectivity(local_connectivity)
                .repulsion_strength(repulsion_strength)
                .negative_sample_rate(negative_sample_rate)
                .a(a)
                .b(b)
                .build::<f32>()
                .map_err(build_err_to_py)?;
            est.fit_transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)
        })
    }

    /// `fit_transform(X)` (f64 path) — see [`Self::fit_transform_f32`].
    /// `guard_f64()` is applied BEFORE the device upload (T-16-GUARDF64).
    fn fit_transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        let (
            n_neighbors, n_components, min_dist, spread, metric_s, n_epochs,
            init_s, random_state, learning_rate, set_op_mix_ratio,
            local_connectivity, repulsion_strength, negative_sample_rate, a, b,
            device_s,
        ) = self.unfit_hyperparams()?;
        let metric = parse_metric(&metric_s)?;
        let init = parse_init(&init_s)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            crate::capability::guard_f64()?;
            let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
            let est = Umap::<f64>::builder()
                        .device(parse_device(&device_s)?)
                .n_neighbors(n_neighbors)
                .n_components(n_components)
                .min_dist(min_dist)
                .spread(spread)
                .metric(metric)
                .n_epochs(n_epochs)
                .init(init)
                .random_state(random_state)
                .learning_rate(learning_rate)
                .set_op_mix_ratio(set_op_mix_ratio)
                .local_connectivity(local_connectivity)
                .repulsion_strength(repulsion_strength)
                .negative_sample_rate(negative_sample_rate)
                .a(a)
                .b(b)
                .build::<f64>()
                .map_err(build_err_to_py)?;
            est.fit_transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)
        })
    }

    /// Host `embedding_` (f32 arm, length `rows × n_components`, row-major) or the
    /// runtime [`not_fitted`] analog on the `Unfit`/wrong-dtype arm (D-13).
    fn embedding_f32(&self) -> PyResult<Vec<f32>> {
        self.embedding_f32_inner()
    }
    /// Host `embedding_` (f64 arm) or the [`not_fitted`] analog.
    fn embedding_f64(&self) -> PyResult<Vec<f64>> {
        self.embedding_f64_inner()
    }

    /// `True` once `fit` has run (either dtype arm), for the shim's fitted-check.
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyUmap::Unfit { .. })
    }
    /// `"f32"`/`"f64"` of the fitted arm, or `None` before `fit`.
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyUmap::Unfit { .. } => None,
            AnyUmap::F32(_) => Some("f32"),
            AnyUmap::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// TSNE (TSNE-01 / TSNE-PARAMS) — sklearn's FULL constructor surface.
// Fit + embedding_/kl_divergence_/n_iter_ (fit/fit_transform only, sklearn
// parity: TSNE is transductive and has no out-of-sample transform).
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyTsne,
    algo:  mlrs_algos::manifold::tsne::Tsne,
    unfit: {},
}

/// The hyperparameters, persisted on the WRAPPER rather than read back out of
/// the `Unfit` arm (WR-02). Keeping them here is what makes a SECOND `fit` of
/// the same Python object work: `fit` consumes the typestate estimator, so the
/// `Unfit` payload is gone once the first fit lands.
struct TsneParams {
    n_components: usize,
    perplexity: f64,
    early_exaggeration: f64,
    /// `None` is sklearn's `'auto'`.
    learning_rate: Option<f64>,
    max_iter: usize,
    n_iter_without_progress: usize,
    min_grad_norm: f64,
    metric: String,
    metric_p: Option<f64>,
    metric_v: Option<Vec<f64>>,
    metric_vi: Option<Vec<f64>>,
    init: String,
    /// sklearn's `init=<ndarray>`, flattened row-major. When present, `init`
    /// carries the sentinel `"array"`.
    init_array: Option<Vec<f64>>,
    verbose: usize,
    random_state: Option<u64>,
    method: String,
    angle: f64,
    n_jobs: Option<i32>,
    /// DEVICE-PARAM-01, a STRING until `build` (D-09).
    device: String,
}

/// Parse the sklearn-named `init` string. The `"array"` sentinel is emitted by
/// the Python layer when the user passed an ndarray, and is paired with
/// `init_array`.
fn parse_tsne_init(
    s: &str,
    array: Option<&Vec<f64>>,
) -> PyResult<mlrs_algos::manifold::tsne::TsneInit> {
    use mlrs_algos::manifold::tsne::TsneInit;
    match s {
        "pca" => Ok(TsneInit::Pca),
        "random" => Ok(TsneInit::Random),
        "array" => match array {
            Some(v) => Ok(TsneInit::Array(v.clone())),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "tsne: init=\"array\" requires the flattened init array",
            )),
        },
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "tsne: unsupported init {other:?}; expected \"pca\", \"random\", or an ndarray"
        ))),
    }
}

/// Parse the sklearn-named `method` string.
fn parse_tsne_method(s: &str) -> PyResult<mlrs_algos::manifold::tsne::TsneMethod> {
    mlrs_algos::manifold::tsne::TsneMethod::from_sklearn_name(s).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "tsne: unsupported method {s:?}; expected \"barnes_hut\" or \"exact\""
        ))
    })
}

/// Parse the sklearn-named `metric` string (the full 22-value `StrOptions`
/// set, aliases collapsed).
fn parse_tsne_metric(s: &str) -> PyResult<mlrs_algos::manifold::tsne_metric::TsneMetric> {
    mlrs_algos::manifold::tsne_metric::TsneMetric::from_sklearn_name(s).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "tsne: unsupported metric {s:?}"
        ))
    })
}

/// sklearn-compatible `TSNE` with the full 1.9.0 parameter surface:
/// `method` in `{'barnes_hut', 'exact'}`, the 22 `metric` strings plus
/// `metric_params`, `init` in `{'pca', 'random', <ndarray>}`, and the
/// `angle` / `n_iter_without_progress` / `min_grad_norm` / `verbose` /
/// `n_jobs` knobs. `fit` + `embedding_`, NO out-of-sample `transform`.
#[pyclass(name = "TSNE")]
pub struct PyTSNE {
    inner: AnyTsne,
    params: TsneParams,
}

impl PyTSNE {
    /// Rust-callable default constructor for the smoke test.
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyTsne::Unfit {},
            params: TsneParams {
                device: "auto".to_string(),
                n_components: 2,
                perplexity: 30.0,
                early_exaggeration: 12.0,
                learning_rate: None,
                max_iter: 1000,
                n_iter_without_progress: 300,
                min_grad_norm: 1e-7,
                metric: "euclidean".to_string(),
                metric_p: None,
                metric_v: None,
                metric_vi: None,
                init: "pca".to_string(),
                init_array: None,
                verbose: 0,
                random_state: None,
                method: "barnes_hut".to_string(),
                angle: 0.5,
                n_jobs: None,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyTsne::Unfit { .. })
    }
}

#[pymethods]
impl PyTSNE {
    /// `TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,
    /// learning_rate=None ('auto'), max_iter=1000, n_iter_without_progress=300,
    /// min_grad_norm=1e-7, metric="euclidean", metric_p=None, metric_v=None,
    /// metric_vi=None, init="pca", init_array=None, verbose=0,
    /// random_state=None, method="barnes_hut", angle=0.5, n_jobs=None)`.
    ///
    /// `metric_params` is unpacked by the Python layer into the three scipy
    /// keywords the Rust side understands, so the boundary stays typed rather
    /// than carrying a `PyDict`.
    #[new]
    #[pyo3(signature = (
        n_components = 2, perplexity = 30.0, early_exaggeration = 12.0,
        learning_rate = None, max_iter = 1000, n_iter_without_progress = 300,
        min_grad_norm = 1e-7, metric = String::from("euclidean"),
        metric_p = None, metric_v = None, metric_vi = None,
        init = String::from("pca"), init_array = None, verbose = 0,
        random_state = None, method = String::from("barnes_hut"),
        angle = 0.5, n_jobs = None, device = "auto".to_string(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_components: usize,
        perplexity: f64,
        early_exaggeration: f64,
        learning_rate: Option<f64>,
        max_iter: usize,
        n_iter_without_progress: usize,
        min_grad_norm: f64,
        metric: String,
        metric_p: Option<f64>,
        metric_v: Option<Vec<f64>>,
        metric_vi: Option<Vec<f64>>,
        init: String,
        init_array: Option<Vec<f64>>,
        verbose: usize,
        random_state: Option<u64>,
        method: String,
        angle: f64,
        n_jobs: Option<i32>,
        device: String,
    ) -> PyResult<Self> {
        // Data-independent string surface validated AT CONSTRUCTION, so a typo
        // raises where the user wrote it rather than at `fit`.
        parse_tsne_init(&init, init_array.as_ref())?;
        parse_tsne_method(&method)?;
        parse_tsne_metric(&metric)?;
        Ok(Self {
            inner: AnyTsne::Unfit {},
            params: TsneParams {
                device,
                n_components,
                perplexity,
                early_exaggeration,
                learning_rate,
                max_iter,
                n_iter_without_progress,
                min_grad_norm,
                metric,
                metric_p,
                metric_v,
                metric_vi,
                init,
                init_array,
                verbose,
                random_state,
                method,
                angle,
                n_jobs,
            },
        })
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        use mlrs_algos::manifold::tsne::{LearningRate, Tsne};
        use mlrs_algos::manifold::tsne_metric::MetricParams;

        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let p = &self.params;
        let init = parse_tsne_init(&p.init, p.init_array.as_ref())?;
        let method = parse_tsne_method(&p.method)?;
        let metric = parse_tsne_metric(&p.metric)?;
        let metric_params = MetricParams {
            p: p.metric_p,
            v: p.metric_v.clone(),
            vi: p.metric_vi.clone(),
            w: None,
        };
        let lr = match p.learning_rate {
            None => LearningRate::Auto,
            Some(v) => LearningRate::Value(v),
        };
        let seed = p.random_state.unwrap_or(0);
        let (
            n_components,
            perplexity,
            early_exaggeration,
            max_iter,
            n_iter_without_progress,
            min_grad_norm,
            verbose,
            angle,
            n_jobs,
        ) = (
            p.n_components,
            p.perplexity,
            p.early_exaggeration,
            p.max_iter,
            p.n_iter_without_progress,
            p.min_grad_norm,
            p.verbose,
            p.angle,
            p.n_jobs,
        );

        let fitted = py.detach(|| -> PyResult<AnyTsne> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = Tsne::<f32>::builder()
                        .device(parse_device(&p.device)?)
                        .n_components(n_components)
                        .perplexity(perplexity)
                        .early_exaggeration(early_exaggeration)
                        .learning_rate(lr)
                        .max_iter(max_iter)
                        .n_iter_without_progress(n_iter_without_progress)
                        .min_grad_norm(min_grad_norm)
                        .metric(metric)
                        .metric_params(metric_params)
                        .init(init)
                        .verbose(verbose)
                        .seed(seed)
                        .method(method)
                        .angle(angle)
                        .n_jobs(n_jobs)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = est
                        .fit(&mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyTsne::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = Tsne::<f64>::builder()
                        .device(parse_device(&p.device)?)
                        .n_components(n_components)
                        .perplexity(perplexity)
                        .early_exaggeration(early_exaggeration)
                        .learning_rate(lr)
                        .max_iter(max_iter)
                        .n_iter_without_progress(n_iter_without_progress)
                        .min_grad_norm(min_grad_norm)
                        .metric(metric)
                        .metric_params(metric_params)
                        .init(init)
                        .verbose(verbose)
                        .seed(seed)
                        .method(method)
                        .angle(angle)
                        .n_jobs(n_jobs)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = est
                        .fit(&mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyTsne::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// Fitted `embedding_` (f32 arm), flattened row-major.
    fn embedding_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyTsne::F32(e) => Ok(e.embedding(&pool)),
            _ => Err(not_fitted("tsne", "embedding_ (f32)")),
        }
    }
    /// Fitted `embedding_` (f64 arm), flattened row-major.
    fn embedding_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyTsne::F64(e) => Ok(e.embedding(&pool)),
            _ => Err(not_fitted("tsne", "embedding_ (f64)")),
        }
    }
    /// Final `kl_divergence_`, either dtype arm.
    /// `device_` — the engine that actually ran (`"cpu"` / `"gpu"`), read off
    /// the fitted estimator. `None` before `fit`.
    fn device_used(&self) -> Option<&'static str> {
        match &self.inner {
            AnyTsne::F32(e) => e.device_arm(),
            AnyTsne::F64(e) => e.device_arm(),
            _ => None,
        }
    }

    fn kl_divergence_(&self) -> PyResult<f64> {
        match &self.inner {
            AnyTsne::F32(e) => Ok(e.kl_divergence()),
            AnyTsne::F64(e) => Ok(e.kl_divergence()),
            _ => Err(not_fitted("tsne", "kl_divergence_")),
        }
    }
    /// Iterations run (`n_iter_`), either dtype arm.
    fn n_iter_(&self) -> PyResult<usize> {
        match &self.inner {
            AnyTsne::F32(e) => Ok(e.n_iter()),
            AnyTsne::F64(e) => Ok(e.n_iter()),
            _ => Err(not_fitted("tsne", "n_iter_")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyTsne::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyTsne::Unfit { .. } => None,
            AnyTsne::F32(_) => Some("f32"),
            AnyTsne::F64(_) => Some("f64"),
        }
    }
}
