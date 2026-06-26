# Phase 16: Builder Retrofit Sweep + Shim Coverage - Pattern Map

**Mapped:** 2026-06-24
**Files analyzed:** 7 representative work-items (covering 29 estimators + 8 PyO3 wraps + ~15 shim classes + 3 test files)
**Analogs found:** 7 / 7 (all exact — every target shape has a born-with-convention or shipped exemplar in-tree)

> This is a broad mechanical sweep. PATTERNS are organized by **shape template**, not by per-file rows.
> The per-estimator green-suite gate (CONTEXT D-03) is the safety mechanism; the analog excerpts below are
> what the executor copies. Authoritative shape spec: `crates/mlrs-algos/src/manifold/umap.rs`.

## File Classification

| Work-item | Role | Data Flow | Closest Analog | Match Quality |
|-----------|------|-----------|----------------|---------------|
| 1. Shape-A retrofit (no-builder, arg-`new`) — `linear/ridge.rs` (pilot A) + 17 more | model/estimator | request-response (fit→predict) | `manifold/umap.rs` (target shape) + `linear/ridge.rs` (current) | exact |
| 2. Shape-B retrofit (has builder) — `linear/mbsgd_regressor.rs` (pilot B) + 8 more | model/estimator | request-response | `manifold/umap.rs` (Fit/Fitted) + own builder (mbsgd_regressor.rs:89-288) | exact |
| 3. typestate.rs trait additions (5 new traits + `inverse_transform` default) | trait-surface (model) | request-response | existing 4 traits in `typestate.rs` + 9 old shapes in `traits.rs` | exact |
| 4. PyO3 call-site migration (8 files in `mlrs-py/src/estimators/`) | provider (FFI binding) | request-response | `estimators/manifold.rs` (PyUMAP) / `cluster.rs` (PyHDBSCAN) consuming-self | exact |
| 5. PyO3 method-gap fill (SHIM-02) — PyUMAP `transform`/`fit_transform`; PyHDBSCAN `fit_predict`/`probabilities_`/`outlier_scores_` | provider (FFI binding) | request-response / transform | PyUMAP `fit` arm-match forwarder (manifold.rs:275-281); `cluster.py` labels forwarder | role-match |
| 6. New pure-Python shim classes (~15, incl. `UMAP`/`HDBSCAN`) | provider (sklearn shim) | request-response | `python/mlrs/linear.py::Ridge` (linear.py:57-88) + `cluster.py::DBSCAN` | exact |
| 7. Static shim test extension (AST-purity + matrix expand) | test | batch/transform (static analysis) | `test_params.py` (runtime purity) + `test_shims.py` `ALL_12` | role-match (AST check is NEW) |

## Pattern Assignments

### 1. Shape-A retrofit — `linear/ridge.rs` and the 17 other no-builder estimators (model, request-response)

**Target shape analog:** `crates/mlrs-algos/src/manifold/umap.rs` (born-with-convention exemplar).
**Current (pre-retrofit) analog:** `crates/mlrs-algos/src/linear/ridge.rs`.

The retrofit replaces Ridge's current 4 parts with umap.rs's 5 parts. **Config fields + fit-body math stay byte-identical (D-03).**

**(a) Struct: add `S = Unfit` + `PhantomData<S>`** — replace `ridge.rs:68` `pub struct Ridge<F> {` and the field block. Template from `umap.rs:124-170`:
```rust
// umap.rs:124-170 (the shape) — apply to Ridge:
pub struct Ridge<F, S = Unfit> {
    alpha: F,                 // hyperparam field UNCHANGED (ridge.rs:71)
    fit_intercept: bool,      // UNCHANGED (ridge.rs:73)
    coef_: Option<DeviceArray<ActiveRuntime, F>>,      // fitted state UNCHANGED
    intercept_: Option<DeviceArray<ActiveRuntime, F>>, // UNCHANGED
    _state: PhantomData<S>,   // NEW — the only added field
}
```

**(b) Zero-arg `new()` + `builder()` + `into_builder()` + `hyperparams_eq()` on `impl<F> Ridge<F, Unfit>`** — replace the arg-taking `Ridge::new(alpha, fit_intercept)` at `ridge.rs:90-97`. Template `umap.rs:172-279`:
```rust
// umap.rs:181-208 (shape). new() is the SINGLE source of sklearn defaults, zero-arg:
impl<F> Ridge<F, Unfit> where F: Float + CubeElement + Pod {
    pub fn new() -> Self {
        Self { alpha: F::from_int(1), /* sklearn Ridge alpha=1.0 */ fit_intercept: true,
                coef_: None, intercept_: None, _state: PhantomData }
    }
    pub fn builder() -> RidgeBuilder { RidgeBuilder::default() }
    pub fn into_builder(self) -> RidgeBuilder { /* copy hyperparams — umap.rs:251-269 */ }
    pub fn hyperparams_eq(&self, other: &Self) -> bool { /* umap.rs:229-245 — BLDR-01 test */ }
}
impl<F> Default for Ridge<F, Unfit> { fn default() -> Self { Self::new() } } // umap.rs:272-279
```

**(c) Builder with `Default` RE-DERIVED from `new()` (Pitfall 1 — never re-list literals).** Template `umap.rs:285-446`:
```rust
#[derive(Debug, Clone, Copy)]
pub struct RidgeBuilder { alpha: f64, fit_intercept: bool }  // setters take f64 (A5 convention)
impl Default for RidgeBuilder {
    fn default() -> Self { Ridge::<f64, Unfit>::new().into_builder() }  // umap.rs:309 — SINGLE source
}
impl RidgeBuilder {
    pub fn alpha(mut self, v: f64) -> Self { self.alpha = v; self }
    pub fn fit_intercept(mut self, v: bool) -> Self { self.fit_intercept = v; self }
    // data-INDEPENDENT validation here (alpha >= 0 → BuildError::InvalidAlpha), umap.rs:400-423.
    pub fn build<F>(self) -> Result<Ridge<F, Unfit>, BuildError> where F: Float + CubeElement + Pod { … }
}
```
NOTE: Ridge's current `alpha < 0` check lives in `fit` (`ridge.rs:140-146` `AlgoError::InvalidAlpha`). Per D-04/Pitfall 7 the data-INDEPENDENT half moves to `build()` → `BuildError` (mbsgd_regressor.rs:219-223 shows the `BuildError::InvalidAlpha` shape). Builder setters are `f64`; `build::<F>()` casts (A5).

**(d) Imports swap.** `ridge.rs:58-59`:
```rust
use crate::error::AlgoError;
use crate::traits::{Fit, Predict};          // OLD — remove
// → add BuildError + the typestate surface (umap.rs:43-45):
use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};
```

**(e) `Fit::fit` consumes self → returns `Ridge<F, Fitted>`; fit-body math byte-identical (Pitfall 2).** Replace `ridge.rs:124-311` signature only. Template `umap.rs:448-526`:
```rust
impl<F> Fit<F> for Ridge<F, Unfit> where F: Float + CubeElement + Pod {
    type Fitted = Ridge<F, Fitted>;
    fn fit(self, pool, x, y, shape) -> Result<Ridge<F, Fitted>, AlgoError> {
        validate_geometry(x, shape)?;        // replaces the inline guard at ridge.rs:147-154
        /* …EVERY compute line from ridge.rs:168-305 copied VERBATIM… */
        Ok(Ridge { alpha: self.alpha, fit_intercept: self.fit_intercept,
                   coef_: Some(coef), intercept_: Some(intercept_dev), _state: PhantomData })
    }                                        // was: self.coef_ = Some(..); Ok(self) at ridge.rs:307-309
}
```
Use the shared `validate_geometry` (typestate.rs:59-76) for the data-DEPENDENT guard; the `alpha < 0` host-side check moves out to `build()`.

**(f) `Predict` + accessors move onto `impl<F> Ridge<F, Fitted>`, dropping `NotFitted`.** Template `umap.rs:528-592`:
```rust
impl<F> Ridge<F, Fitted> {
    pub fn coef(&self, pool) -> Vec<F> {            // was Result<Vec<F>, AlgoError> at ridge.rs:101-109
        self.coef_.as_ref().expect("Some by construction on Ridge<F, Fitted>").to_host(pool)
    }                                               // NO ok_or(NotFitted) — umap.rs:536-541
    pub fn intercept(&self, pool) -> F { … }
}
impl<F> Predict<F> for Ridge<F, Fitted> { fn predict(&self, …) { /* ridge.rs:317-373 body verbatim */ } }
```

**Per-estimator new() signatures** (RESEARCH §Inventory — what each `new()` becomes zero-arg, defaults move to builder): LinearRegression `new(fit_intercept)`, Lasso/ElasticNet/LogisticRegression also have `with_opts(...)`, PCA/TruncatedSvd `new(n_components)`, DBSCAN `new(eps, min_samples)`, covariance/projection/neighbors per the inventory table. KMeans (`new`/`with_init`/`with_opts`) + SpectralClustering (6-arg) are the multi-constructor stress cases → **handle KMeans last** (D-06).

**Shape A' subset (KernelRidge, SpectralEmbedding):** same build-out PLUS *adopting* `typestate::Fit`/`Predict`/`Transform` (they currently have inherent non-trait `fit`/`predict`, no `crate::traits` import). Adopt the trait for the single-surface end-state (RESEARCH Open Q3).

---

### 2. Shape-B retrofit — `linear/mbsgd_regressor.rs` (pilot B) + the 8 other has-builder estimators (model, request-response)

**Analog (own file, already has the builder):** `crates/mlrs-algos/src/linear/mbsgd_regressor.rs:89-288`.
**Trait-swap target:** `crates/mlrs-algos/src/manifold/umap.rs:448-547` (Fit/Fitted shape).

Builder already exists and is correct — this is **typestate-param + trait-swap ONLY**. The minimal delta:

**(a) Struct gets `<F, S = Unfit>` + `PhantomData<S>`** — `mbsgd_regressor.rs:36`:
```rust
pub struct MBSGDRegressor<F, S = Unfit> {
    config: SgdConfig,                              // UNCHANGED (mbsgd_regressor.rs:38)
    coef_: Option<DeviceArray<ActiveRuntime, F>>,  // UNCHANGED
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
    _state: PhantomData<S>,                         // NEW
}
```

**(b) The ONLY builder edit: `build<F>()` return type.** `mbsgd_regressor.rs:213`:
```rust
pub fn build<F>(self) -> Result<MBSGDRegressor<F, Unfit>, BuildError>   // was MBSGDRegressor<F>
```
And the returned struct literal (mbsgd_regressor.rs:282-286) gains `_state: PhantomData`. The entire builder body (validation at :217-265, `SgdConfig` lowering at :266-281) stays untouched.

**(c) Imports swap** — `mbsgd_regressor.rs:30`: `use crate::traits::{Fit, Predict};` → `use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};`

**(d) `Fit::fit` consuming-self + accessors/`Predict` → `impl …<F, Fitted>`**, same mechanical move as Shape-A (e)/(f). The `fit` body (`mbsgd_regressor.rs:294` onward — the `sgd_solve` drive) is byte-identical; only the signature + final return change. `coef`/`intercept` accessors (`mbsgd_regressor.rs:62-82`) lose `ok_or(NotFitted)` → `.expect(...)`.

**The 9 Shape-B estimators:** LinearSVC, LinearSVR, MBSGDClassifier, MBSGDRegressor + 5 NB (GaussianNB/MultinomialNB/BernoulliNB/ComplementNB/CategoricalNB). All already have hand-written builders → fast trait-swap (RESEARCH §Inventory). The 5 NB implement the 4-trait set (Fit + PredictLabels + PredictProba + PredictLogProba) — those accessor traits must already exist in typestate.rs (work-item 3) first.

---

### 3. typestate.rs trait additions — 5 new traits + `Transform::inverse_transform` default (trait-surface, Wave 0 BLOCKING)

**Analog (existing 4 traits, the shape to mirror):** `crates/mlrs-algos/src/typestate.rs:127-215` (`Fit`/`Predict`/`Transform`/`PartialFit`).
**Source signatures to port (old surface):** `crates/mlrs-algos/src/traits.rs:158-303` (`PredictLabels`/`KNeighbors`/`ScoreSamples`/`PredictProba`/`PredictLogProba`).

The 5 missing traits are all `&self` accessor traits — port them with the SAME signature as `traits.rs`, intended to be `impl`'d ONLY on the `Fitted`-tagged estimator (exactly how `Transform` at typestate.rs:169-181 is impl'd only on `Umap<F, Fitted>`). They have no associated `type Fitted` (they don't transition state — they read fitted state).

**Template — copy the `Predict` shape (typestate.rs:152-164) and the `traits.rs` body:**
```rust
// typestate.rs new trait — PredictLabels (copy signature from traits.rs:168-181):
pub trait PredictLabels<F> where F: Float + CubeElement + Pod {
    fn predict_labels(&self, pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>, shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError>;
}
```
Add identically: `KNeighbors` (traits.rs:191-213 — returns `(F, i32)` tuple, takes `k: usize`), `ScoreSamples` (traits.rs:231-245), `PredictProba` (traits.rs:256-270), `PredictLogProba` (traits.rs:288-303).

**`Transform::inverse_transform` default** — typestate.rs's `Transform` (line 169-181) lacks the `inverse_transform` default that `traits.rs:145-155` carries (PCA's reconstruction path). Port the defaulted method into typestate's `Transform` verbatim:
```rust
// traits.rs:145-155 — add to typestate.rs Transform trait:
fn inverse_transform(&self, _pool, _z, _shape) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
    Err(AlgoError::Unsupported { estimator: "transform", operation: "inverse_transform" })
}
```

**`PartialFit` is already present** (typestate.rs:195-215) — IncrementalPCA impls it on BOTH `Unfit` and `Fitted` (Pitfall 5). The `State`/`Unfit`/`Fitted`/`sealed::Sealed` machinery (typestate.rs:82-115) and `validate_geometry` (typestate.rs:59-76) are done.

**Gate:** `cargo build -p mlrs-algos` + a `typestate_test.rs` case. The module doc header (typestate.rs:9-18) currently says "30 estimators continue to compile against [traits.rs]... FROZEN" — that comment is superseded by D-01 (traits.rs is DELETED at phase end); update the doc when the surface converges.

---

### 4. PyO3 call-site migration — the 8 files in `mlrs-py/src/estimators/` (provider/FFI, the hidden cross-crate surface)

**Analog (already-migrated consuming-self target):** `crates/mlrs-py/src/estimators/manifold.rs:194-271` (PyUMAP fit) and `cluster.rs:408-466` (PyHDBSCAN fit).
**Current (old-trait shape to replace):** `crates/mlrs-py/src/estimators/linear.rs:225-262` (PyRidge fit) and the 7 sibling files.

The 8 files that `use mlrs_algos::traits` (linear.rs, decomposition.rs, cluster.rs, covariance.rs, projection.rs, density/kernel.rs, neighbors.rs, naive_bayes.rs) currently do `let mut est = T::<f32>::new(args); est.fit(&mut pool, …)?`. They migrate to the builder + consuming-self shape.

**Current shape (linear.rs:243-256 — PyRidge f32/f64 arms):**
```rust
FloatDtype::F32 => {
    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
    let mut est = Ridge::<f32>::new(alpha as f32, fit_intercept);   // arg-new + `as f32`
    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;  // &mut self
    Ok(AnyRidge::F32(est))
}
```

**Target shape (manifold.rs:220-242 — the consuming-self builder form):**
```rust
FloatDtype::F32 => {
    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
    let est = Ridge::<f32>::builder()
        .alpha(alpha)                  // f64 setter — drop the `as f32` (Pitfall 7 / A5)
        .fit_intercept(fit_intercept)
        .build::<f32>()
        .map_err(build_err_to_py)?;    // NEW — data-independent validation → ValueError
    let fitted = est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
    Ok(AnyRidge::F32(fitted))          // arm type is now Ridge<f32, Fitted>
}
```

**The `any_estimator!` → `any_estimator_typestate!` macro swap (Pitfall 4).** The `AnyRidge` enum at `linear.rs:189-193` uses `crate::any_estimator!` whose `F32` arm spells `$algo<f32>` (dispatch.rs:108). After retrofit the arm must be `$algo<f32, Fitted>`, so switch to `crate::any_estimator_typestate!` (dispatch.rs:158-173) exactly as PyUMAP (`manifold.rs:36-46`) and PyHDBSCAN (`AnyHdbscan`) already do. The `Unfit { .. }` hyperparam arm + matcher are identical between the two macros (dispatch.rs:170 vs 106).

**Import swap per file:** e.g. `linear.rs:23` `use mlrs_algos::traits::{Fit, Predict, PredictLabels, PredictProba};` → `use mlrs_algos::typestate::{Fit, Predict, PredictLabels, PredictProba};`. NOTE the HDBSCAN wrap already imports the typestate `Fit` aliased as `TypestateFit` and calls `TypestateFit::fit(est, …)` (cluster.rs:443) to disambiguate — use that aliasing where a file mixes surfaces mid-migration.

**Keep:** `py.detach` GIL release, `crate::lock_pool()` (poison-recovering, NOT `.lock().expect()` — Security V5), `guard_f64()` before the F64 arm (manifold.rs:244), `validated_f32/f64`, `algo_err_to_py`/`build_err_to_py`/`not_fitted` mappers — all UNCHANGED (D-13).

**Distribution (RESEARCH §Call-Site Migration):** ~42 sites across these 8 PyO3 files, ~40 in `mlrs-algos/tests/`. **Migrate each estimator's PyO3 arm in the SAME commit as the estimator** (Pitfall 3) so `cargo build -p mlrs-py` never breaks. **Final commit:** delete `traits.rs` + its `pub mod traits;` once `grep -rn 'mlrs_algos::traits\|crate::traits' crates/` is empty across BOTH crates.

---

### 5. PyO3 method-gap fill (SHIM-02) — PyUMAP `transform`/`fit_transform`; PyHDBSCAN `fit_predict`/`probabilities_`/`outlier_scores_` (provider/FFI)

The wraps EXIST and are registered (manifold.rs:74 PyUMAP, cluster.rs:324 PyHDBSCAN). Only specific `#[pymethods]` are missing (RESEARCH verified). These are forwarders onto the fitted arm.

**Analog — the dtype-arm-match forwarder pattern (manifold.rs:118-132 / 275-281):**
```rust
// PyUMAP already does this for embedding_ — copy the arm-match shape for transform:
fn embedding_f32_inner(&self) -> PyResult<Vec<f32>> {
    let pool = crate::lock_pool();
    match &self.inner {
        AnyUmap::F32(e) => Ok(e.embedding(&pool)),
        _ => Err(not_fitted("umap", "embedding_ (f32)")),
    }
}
```

**PyUMAP `transform`/`fit_transform`** — the Rust `Umap<F, Fitted>` already has `Transform::transform` (umap.rs:568) and `Umap<F, Unfit>::fit_transform` (umap.rs:215). Add `#[pymethods]` that `py.detach` + `guard_f64` (for F64) + match the `AnyUmap::{F32,F64}` arm + call the Rust method, returning host `Vec<f32>`/`Vec<f64>` (mirror PyRidge's `predict_f32`/`predict_f64` split at linear.rs:264-289 — a `#[pyclass]` method can't be generic over F, so split by dtype suffix). `transform` needs `import` of `typestate::Transform` in manifold.rs.

**PyHDBSCAN `fit_predict`/`probabilities_`/`outlier_scores_`** — the Rust `Hdbscan<F, Fitted>` has `fit_predict` (hdbscan.rs:284), GLOSH `outlier_scores_`, and `probabilities_`. Add forwarders mirroring `labels_inner` (cluster.rs:359-366): a dtype-agnostic arm-match (`AnyHdbscan::F32(e) | F64(e)` for i32 labels; dtype-suffixed for `F` outputs). `n_features_in_` surfacing is handled Python-side by `_post_fit` once the shim class exists (work-item 6).

**Registration:** new `#[pymethods]` on an existing `#[pyclass]` need no `lib.rs` change (already registered at lib.rs:265-266). Keep `not_fitted` on the wrong arm.

---

### 6. New pure-Python shim classes — ~15 incl. `mlrs.UMAP` / `mlrs.HDBSCAN` (provider/sklearn shim)

**Analog (regressor + transform):** `crates/mlrs-py/python/mlrs/linear.py::Ridge` (linear.py:57-88).
**Analog (cluster, labels-only, no predict):** `crates/mlrs-py/python/mlrs/cluster.py::DBSCAN` (cluster.py:72-98) and `KMeans` (cluster.py:16-69).
**Analog (transformer):** `python/mlrs/decomposition.py::PCA` `transform` forwarder (decomposition.py:32-34).

Each shim is a faithful `MlrsBase` subclass. `get_params`/`set_params`/`clone` come FREE from `BaseEstimator` given a pure `__init__` (base.py:28). The machinery (`_normalize`, `_ext`, `_suffix`/`_suffixed` at base.py:119-143, `_post_fit` at base.py:147-158, `_check_fitted`, `__sklearn_tags__` at base.py:171-182) is complete.

**Faithful `__init__` (purity rule — store each arg verbatim under the same name, no compute):** linear.py:60-63 (Ridge):
```python
def __init__(self, alpha=1.0, fit_intercept=True, output_type="input"):
    self.alpha = alpha             # verbatim, same name — NO validation/computation
    self.fit_intercept = fit_intercept
    self.output_type = output_type # base param every shim adds
```

**`fit` → `_ext()` → store handle → `_post_fit` → return self:** linear.py:65-72 (Ridge):
```python
def fit(self, X, y):
    xa, rows, cols = self._normalize(X)
    ya = self._normalize_y(y, dtype=LinearRegression._x_float(xa))
    obj = self._ext().Ridge(self.alpha, self.fit_intercept)   # PyO3 wrapper ctor
    obj.fit(xa, ya, rows, cols)
    self._mlrs_obj = obj
    self._post_fit(cols)
    return self
```

**Dtype-suffixed fitted accessor:** linear.py:79-83 (Ridge `coef_`): `self._suffixed("coef")()` reads `coef_f32`/`coef_f64` via base.py:136-143.

**Param-name boundary mappings (RESEARCH §Python Shim):** sklearn-named params even when the Rust field differs — LogisticRegression exposes `C` not `c` (linear.py:202 `self.C = C`); KMeans exposes `random_state`→Rust `seed` (cluster.py:36). For the NEW classes:
- `mlrs.UMAP`: `n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0, metric="euclidean", n_epochs=None, init="spectral", random_state=None, learning_rate=1.0, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5, a=None, b=None` (matches PyUMAP `#[new]` signature at manifold.rs:142-148). `TransformerMixin` (`fit` + `transform`/`fit_transform`).
- `mlrs.HDBSCAN`: `min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, cluster_selection_method="eom", metric="euclidean", alpha=1.0, max_cluster_size=0` (matches PyHDBSCAN `#[new]` at cluster.rs:375-379). `ClusterMixin` (gives `fit_predict`); `labels_` property like cluster.py:62-64 / 89-91.

**Other missing classes (RESEARCH §Python Shim, 14 total):** LinearSVC, LinearSVR, MBSGDClassifier, MBSGDRegressor (RegressorMixin/ClassifierMixin, linear.py family), the 5 NB (ClassifierMixin), KernelRidge (RegressorMixin), KernelDensity, SpectralClustering, SpectralEmbedding. Each PyO3 wrap already exists (32 wraps) — the shim is mechanical.

**Pitfall 6 (zero-arg constructibility):** every new `__init__` must default ALL args (sklearn-style) so `check_parameters_default_constructible` passes. PCA is the existing exception (required `n_components`, special-cased in test `_construct` at test_params.py:86-91) — new classes must NOT require args.

---

### 7. Static shim test extension — AST `__init__`-purity + matrix expansion (test, Wave 0)

**Analog (runtime purity, the matrix to expand):** `crates/mlrs-py/python/tests/test_params.py` (`EXPECTED_PARAMS`/`SET_PARAM`/`ALL_12` at :25-83; runtime purity at :117-127).
**Analog (importability + shim surface matrix):** `test_shims.py` (`ALL_12` at :26, parametrized tests).

**(a) Expand the matrix** — replace the hard-coded `ALL_12` (= `list(EXPECTED_PARAMS)`, test_params.py:83) with the full shim set; add `EXPECTED_PARAMS` + `SET_PARAM` rows for every new class incl. UMAP/HDBSCAN. Template row (test_params.py:50-57, KMeans — multi-param + a sklearn-renamed `random_state`):
```python
"KMeans": {"n_clusters": 8, "init": "k-means++", "max_iter": 300, "tol": 1e-4,
           "random_state": None, "output_type": "input"},
```
Add e.g. `"UMAP": {"n_neighbors": 15, "n_components": 2, "min_dist": 0.1, … , "output_type": "input"}` and `"HDBSCAN": {"min_cluster_size": 5, "min_samples": None, … , "output_type": "input"}` matching the shim `__init__` defaults (work-item 6). Same expansion in `test_shims.py:26` and `test_estimator_checks.py`.

**(b) ADD the AST-based purity test — NEW (no `import ast` exists today).** Strengthens the runtime check at test_params.py:117-127. Parse `inspect.getsource(cls.__init__)`, assert the body is ONLY `self.<arg> = <arg>` assignments (each ctor arg stored verbatim, same name) — reject any `ast.Call`/`ast.BinOp`/comparison node (= validation/computation). Sketch:
```python
import ast, inspect
def test_init_purity_ast(name):
    cls = getattr(mlrs, name)
    tree = ast.parse(inspect.getsource(cls.__init__).strip())
    fn = tree.body[0]
    for stmt in fn.body:
        assert isinstance(stmt, ast.Assign)            # only assignments
        tgt = stmt.targets[0]
        assert isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self"
        assert isinstance(stmt.value, ast.Name)        # self.x = x  (bare Name, no Call/BinOp)
        assert tgt.attr == stmt.value.id               # SAME name (purity)
    # allow the `output_type` assignment from a literal default in the same shape
```
This is the strongest SHIM-01 guarantee without FFI (D-07 step 3).

**(c) Confirm the fit-free `estimator_checks` subset runs green** (D-07 step 4) — `check_no_attributes_set_in_init`, `check_parameters_default_constructible`, `check_get_params_invariance` are yielded by `parametrize_with_checks` (test_estimator_checks.py). Verify they are NOT in the `_expected_failed_checks` xfail map for the new classes (A7).

**(d) Rust-side defaults-equality unit test** per estimator: `T::new().hyperparams_eq(&T::builder().build()?)` (BLDR-01). Template: `hyperparams_eq` at umap.rs:229-245; the umap_test.rs case is the reference.

## Shared Patterns

### The byte-identical-math contract (D-03 — applies to ALL Shape-A/B retrofits)
**Source:** umap.rs:454-525 (fit reconstructs into the Fitted arm, math unchanged).
**Apply to:** every estimator `fit`. ONLY the signature (`&mut self → self`), the return (`Ok(self) → Ok(T{ …, _state: PhantomData })`), and the guard call (inline shape check → `validate_geometry`) change. ZERO compute-line deltas — the code-review gate diffs the fit body (Pitfall 2).

### Single-source defaults (D-08 / BLDR-01 — applies to ALL builders)
**Source:** umap.rs:309 `Umap::<f64, Unfit>::new().into_builder()`.
**Apply to:** every `Builder::default()`. NEVER re-list literal defaults — derive from `new()` (Pitfall 1, default-drift breaks the oracle gate silently).

### Builder validation = data-INDEPENDENT only; geometry stays in fit (Security V5)
**Source:** mbsgd_regressor.rs:217-265 (`build()` → `BuildError`) + validate_geometry (typestate.rs:59-76, called at fit top).
**Apply to:** every retrofit. The `alpha>=0`-type checks move to `build()` → `BuildError`; the `(rows,cols)`/`x.len()` geometry guard stays at the TOP of `fit` BEFORE any device launch. Do not drop any existing validation in the move.

### PyO3 fit contract (D-04/D-13 — applies to all 8 migrated wrap files)
**Source:** manifold.rs:217-268 (PyUMAP fit).
**Apply to:** every PyO3 `fit`: `py.detach` GIL release, `crate::lock_pool()` (poison-recovering), `guard_f64()` before the F64 arm, `build_err_to_py`/`algo_err_to_py`/`not_fitted` mappers, `validated_f32/f64`. The `Any*` enum switches to `any_estimator_typestate!` (Fitted arms spelled explicitly, dispatch.rs:158-173).

### Pure-Python shim faithfulness (SHIM-01 — applies to all ~15 new classes)
**Source:** linear.py:57-88 (Ridge: pure `__init__`, `fit`→`_ext()`→`_post_fit`→self, `_suffixed` accessors).
**Apply to:** every new shim. `get_params`/`set_params`/`clone`/`n_features_in_`/`__sklearn_tags__` come free from `MlrsBase` (base.py) given a faithful `__init__`.

## No Analog Found

None. Every target shape has a born-with-convention exemplar (umap.rs / hdbscan.rs), a shipped sibling (mbsgd_regressor.rs builder, PyUMAP/PyHDBSCAN wraps, linear.py shims), or a direct old→new signature port (traits.rs → typestate.rs). The only genuinely NEW artifact is the AST-purity test (work-item 7b) — and even that extends the existing runtime purity test at test_params.py:117-127.

## Metadata

**Analog search scope:** `crates/mlrs-algos/src/{manifold,linear,cluster,typestate.rs,traits.rs}`, `crates/mlrs-py/src/{dispatch.rs,estimators/}`, `crates/mlrs-py/python/{mlrs,tests}/`.
**Files scanned:** ~16 read in full or targeted; estimator inventory (29) + call-site distribution taken from the verified RESEARCH.md tables rather than re-grepped.
**Pattern extraction date:** 2026-06-24
**Key cross-references:** RESEARCH §Exemplar Shape, §Call-Site Migration, §Common Pitfalls (1-7); CONTEXT D-01…D-08.
