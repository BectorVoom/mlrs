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

use mlrs_algos::manifold::umap::{Init, Metric, Umap};
use mlrs_algos::typestate::Fit;

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
    ) -> Self {
        Self {
            inner: AnyUmap::Unfit {
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
        ) = match &self.inner {
            AnyUmap::Unfit {
                n_neighbors, n_components, min_dist, spread, metric, n_epochs,
                init, random_state, learning_rate, set_op_mix_ratio,
                local_connectivity, repulsion_strength, negative_sample_rate, a, b,
            } => (
                *n_neighbors, *n_components, *min_dist, *spread, metric.clone(),
                *n_epochs, init.clone(), *random_state, *learning_rate,
                *set_op_mix_ratio, *local_connectivity, *repulsion_strength,
                *negative_sample_rate, *a, *b,
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
