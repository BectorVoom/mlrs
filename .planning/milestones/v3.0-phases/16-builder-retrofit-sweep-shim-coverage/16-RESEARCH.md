# Phase 16: Builder Retrofit Sweep + Shim Coverage - Research

**Researched:** 2026-06-24
**Domain:** Rust typestate/builder convention retrofit (compile-time lifecycle) + PyO3 enum-collapse + pure-Python sklearn shim
**Confidence:** HIGH (every count and shape below was verified by direct file inspection, not taken from CONTEXT.md numbers)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01: Full convergence — delete old `traits.rs`.** Port all 9 old `crate::traits` traits (`Fit`, `PartialFit`, `Predict`, `Transform`, `PredictLabels`, `KNeighbors`, `ScoreSamples`, `PredictProba`, `PredictLogProba`) to typestate-aware versions in `mlrs_algos::typestate`, migrate every existing estimator to the consuming-self surface, then **hard-delete `crate::traits`** at phase end. One trait surface, zero permanent debt.
- **D-02: The typestate convention is layered onto ALL ~21 estimators**, including the ones that already have a `builder()` — those still `use crate::traits` and need the `<F,S>` state param + trait migration even though their builder exists. UMAP/HDBSCAN are already on the new surface (born with it, Phase 12/14/15).
- **D-03: Config fields + fit numerics are byte-identical across the retrofit.** The retrofit wraps construction and lifecycle around each algorithm; it never touches the struct's field set or the fit body math. This preserves every shipped 1e-5 / exact-label gate. Each estimator is migrated **under its own green suite** (migrate → run suite → green → commit).
- **D-04: `new()` → zero-arg sklearn-defaults on every estimator; all args move to the builder.** Existing arg-taking constructors are **removed**; their `::new(`/`with_*` call sites migrate to `T::builder().param(..).…build()?`. Single-source invariant `T::new() == T::builder().build()?` == sklearn default holds uniformly.
- **D-05: Pilot Ridge + MBSGDRegressor — the two structurally-distinct retrofit shapes.** Ridge = no-builder / arg-taking-`new` (full build-out). MBSGDRegressor = already-has-builder / old-trait (typestate-param + trait-swap-only). Both green under their suites before the bulk sweep.
- **D-06: Sweep the rest module-by-module**, each estimator gated by its own suite (linear → decomposition → cluster → covariance → projection → density → neighbors → kernel_ridge → naive_bayes). **KMeans handled late** as the multi-constructor (`new`/`with_init`/`with_opts`) stress case.
- **D-07: Full static shim gate (maximum verifiable without FFI).** Per pure-Python class: (1) import without `_mlrs`; (2) `get_params`/`set_params` round-trip + `clone()` equivalence; (3) **AST-based `__init__`-purity assertion**; (4) the **fit-free subset of sklearn `estimator_checks`** (`check_no_attributes_set_in_init`, `check_parameters_default_constructible`, `check_get_params_invariance`). Plus Rust-side unit tests. Live `check_estimator` FFI run stays **deferred** → UAT.
- **D-08: UMAP/HDBSCAN PyO3 wraps (SHIM-02)** follow the shipped pattern: `#[pyclass]` on `any_estimator!`, GIL release, `guard_f64` before F64, sklearn-named params, trailing-underscore fitted attrs, `n_features_in_` set/enforced, `fit` returns `self`; UMAP `transform`/`fit_transform`; HDBSCAN `fit_predict`/`labels_`. Reuse `build_err_to_py` / `algo_err_to_py`.

### Claude's Discretion
- **Boilerplate generation (researcher to evaluate):** a `derive`/declarative macro to emit the per-estimator state param + builder + impl blocks. Researcher MAY evaluate; hand-written retrofit is fully acceptable if the macro cost/benefit doesn't pay off. Either way the per-estimator green-suite gate (D-03) is non-negotiable. **→ See §Derive-Macro Evaluation (recommendation: hand-written, NO macro).**
- Exact module/file ordering within the sweep, naming of ported typestate traits, and whether call-site migration is one commit per estimator or per module — planner's call.

### Deferred Ideas (OUT OF SCOPE)
- Live FFI `estimator_checks` / `check_estimator` run — deferred (no maturin+pyarrow host).
- Any change to fit algorithm bodies / numerics, or to config-struct field sets.
- New estimators, new algorithms, device-kernel work.
- Builder/typestate boilerplate `derive` macro (evaluated below; recommended NOT adopted — stays deferred).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BLDR-03 | Builder + typestate convention retrofitted across all existing estimators, piloted on 1–2, preserving every gate | §Inventory & Classification (exact 27-estimator list + 2 shapes), §Exemplar Shape (umap.rs/hdbscan.rs template), §Pilot Recipe (Ridge + MBSGDRegressor step sequences), §Call-Site Migration (verified 85 sites incl. the hidden PyO3 surface), §Pitfalls |
| SHIM-01 | Pure-Python class per estimator stores ctor args verbatim; `get_params`/`set_params`/`clone` round-trip; extend v1 → v2 + the two new | §Python Shim (existing 18 classes verified, 14 missing classes enumerated, MlrsBase supplies the machinery for free) |
| SHIM-02 | UMAP + HDBSCAN PyO3-wrapped with the shipped pattern | §PyO3 Wraps — wraps **already shipped** (`PyUMAP` manifold.rs:74, `PyHDBSCAN` cluster.rs:324, registered lib.rs:265-266) but with **VERIFIED method gaps**: PyUMAP lacks `transform`/`fit_transform`; PyHDBSCAN lacks `fit_predict`/`probabilities_`/`outlier_scores_`. SHIM-02 = fill those `#[pymethods]` + add the pure-Python UMAP/HDBSCAN shim classes. NOT "two new wraps". |
| SHIM-03 | Shim verified by Rust unit tests + a static Python check; live run deferred | §Static Shim Gate (existing `test_shims.py`/`test_params.py`/`test_estimator_checks.py` infra; the AST-purity check from D-07 step 3 is NEW — no `import ast` exists today) |
</phase_requirements>

## Summary

This is a **construction-and-lifecycle plumbing phase**, not an algorithm phase. Every estimator's hyperparameter fields and fit-body math stay byte-identical (D-03); what changes is (a) the type signature gains an `S = Unfit` state slot + `PhantomData`, (b) construction routes through a zero-arg `new()` + owned builder, and (c) the trait surface moves from the legacy `&mut self`-returning `crate::traits` to the consuming-self, `Fitted`-gated `mlrs_algos::typestate`. After every estimator migrates, `traits.rs` is hard-deleted (D-01).

**The single hardest, most under-specified fact this research surfaced:** the migration is NOT confined to `mlrs-algos`. **8 PyO3 files in `crates/mlrs-py/src/estimators/` import `mlrs_algos::traits` and call the old `let mut est = T::new(..); est.fit(..)` shape.** Deleting `traits.rs` breaks all 8. Each `AnyEstimator::{F32,F64}` enum arm currently holds `T<f32>` / `T<f64>` (no state param) and must become `T<f32, Fitted>` / `T<f64, Fitted>`, and each `fit` body must change from mutate-in-place to `let fitted = T::builder()…build()?.fit(…)?`. **CONTEXT.md described the call-site migration as "~137 sites in `crates/mlrs-algos/tests/`"; reality is ~85 arg-taking constructor sites repo-wide, of which ~42 live in `mlrs-py/src/estimators/` (the PyO3 wraps) and only ~40 in `mlrs-algos/tests/`.** The PyO3 surface is the load-bearing part the plan must not miss.

**Second material divergence:** SHIM-02's "two new PyO3 wraps" for UMAP/HDBSCAN are **already shipped** (born-with-convention in Phases 14/15). `PyUMAP`/`PyHDBSCAN` exist, use the `AnyUmap`/`AnyHdbscan` `{Unfit,F32,F64}` collapse, call `guard_f64`, expose sklearn-named ctors, and are registered. SHIM-02's real remaining work is the **pure-Python** `mlrs.UMAP` / `mlrs.HDBSCAN` shim classes (which do NOT exist) plus verifying the Rust wrap surface against D-08.

**Primary recommendation:** Hand-write the retrofit (no derive macro — see §Derive-Macro Evaluation). Drive the sweep estimator-by-estimator under each green suite, migrating the matching `mlrs-py/src/estimators/*.rs` PyO3 call sites in the SAME commit as the estimator (so the workspace never has a half-migrated estimator whose PyO3 wrap still calls the deleted trait). Delete `traits.rs` only in the final commit, after the last estimator and the last PyO3 importer are off it.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Typestate state param + builder + zero-arg `new()` | mlrs-algos (estimator source) | — | The convention lives on the Rust struct; this is where `<F,S>` + `PhantomData` + builder are added |
| Consuming-self `Fit`/`Predict`/… traits | mlrs-algos `typestate.rs` | — | Trait definitions must grow to mirror all 9 old traits before any estimator can migrate |
| `traits.rs` deletion | mlrs-algos | mlrs-py (importers) | Orphan-rule/coherence: deletion only succeeds once zero call sites reference it — crosses into mlrs-py |
| Estimator construction call sites (tests) | mlrs-algos `tests/` | — | `T::new(args)` → `T::builder()…build()?` mechanical rewrite |
| Estimator construction + fit call sites (PyO3) | **mlrs-py `src/estimators/`** | mlrs-algos (trait surface) | **The hidden surface**: 8 files call old-trait `est.fit(&mut)`; must become consuming-self into `Any*::{F32,F64}` arms |
| PyO3 enum collapse (`Any*::{Unfit,F32,F64}`) | mlrs-py `dispatch.rs` / `estimators/` | — | Fitted arms change `T<f32>` → `T<f32, Fitted>`; the runtime `NotFittedError` analog stays (BLDR-04) |
| Pure-Python sklearn shim classes | mlrs-py `python/mlrs/` | — | `MlrsBase` supplies `get_params`/`set_params`/`clone`; work = faithful `__init__`s for the missing 14 classes |
| Static shim verification | mlrs-py `python/tests/` | — | Extend `ALL_12` matrix → all classes; ADD the AST-purity check (new) |

## Standard Stack

No new packages. This phase uses only what is already in the workspace.

### Core (already present, verified in-tree)
| Component | Where | Purpose | Why Standard |
|-----------|-------|---------|--------------|
| `mlrs_algos::typestate` | `crates/mlrs-algos/src/typestate.rs` | Target trait surface (`Unfit`/`Fitted`, sealed `State`, consuming-self `Fit`/`Predict`/`Transform`/`PartialFit`) | Already the Phase-12 canonical surface; UMAP/HDBSCAN already implement against it |
| `PhantomData<S>` (std) | per estimator struct | Zero-sized lifecycle marker | The exact mechanism in umap.rs/hdbscan.rs |
| `sklearn.base.BaseEstimator` | `crates/mlrs-py/python/mlrs/base.py` | Supplies `get_params`/`set_params`/`clone` for free | `MlrsBase` already subclasses it |
| `sklearn.utils.estimator_checks.parametrize_with_checks` | `test_estimator_checks.py` | The fit-free check subset for D-07 | Already wired; extend the estimator list |
| `pyo3` `#[pyclass]` + `any_estimator!` | `crates/mlrs-py/src/dispatch.rs` | Float-dtype enum collapse | BLDR-04 contract, already shipped for all 32 wraps |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-written retrofit | `derive`/declarative macro | Rejected — see §Derive-Macro Evaluation. Per-estimator green-suite gate + heterogeneous shapes kill the payoff |
| Per-estimator commits | Per-module commits | Either acceptable (D-06 discretion). Recommend **per-estimator** (one estimator + its PyO3 sites + its suite per commit) for clean bisect/revert under the gate |

**Installation:** none. (No `npm view` / `pip index` / `cargo search` applicable — zero new dependencies, consistent with REQUIREMENTS "Zero new compute dependencies".)

## Package Legitimacy Audit

Not applicable — this phase installs **no external packages**. All work is against in-workspace crates (`mlrs-algos`, `mlrs-py`) and already-pinned dev-dependencies (`sklearn` for the shim tests, already in the Python test env). No registry verification required.

## Inventory & Classification (verified — supersedes CONTEXT counts)

**Estimators currently `use crate::traits` (verified `grep -rln 'use crate::traits' crates/mlrs-algos/src`): 27, not "~21".** Listed by D-06 sweep module order, with builder presence and the implemented trait set (from `use crate::traits::{…}`):

### linear/ (10)
| Estimator | File | Shape | Builder? | Traits implemented | new() signature |
|-----------|------|-------|----------|--------------------|------------------|
| LinearRegression | linear/linear_regression.rs | A (arg-`new`) | no | Fit, Predict | `new(fit_intercept: bool)` |
| **Ridge** (PILOT A) | linear/ridge.rs | A | no | Fit, Predict | `new(alpha: F, fit_intercept: bool)` |
| Lasso | linear/lasso.rs | A | no | Fit, Predict | `new(alpha, fit_intercept)` + `with_opts(alpha, fit_intercept, max_iter, tol)` |
| ElasticNet | linear/elastic_net.rs | A | no | Fit, Predict | `new(alpha, l1_ratio, fit_intercept)` + `with_opts(…)` |
| LogisticRegression | linear/logistic.rs | A | no | Fit, PredictLabels, PredictProba | `new(c, fit_intercept)` + `with_opts(c, fit_intercept, max_iter, tol)` |
| LinearSVC | linear/linear_svc.rs | B (has builder) | **yes** | Fit, PredictLabels | builder-only |
| LinearSVR | linear/linear_svr.rs | B | **yes** | Fit, Predict | builder-only |
| MBSGDClassifier | linear/mbsgd_classifier.rs | B | **yes** | Fit, PredictLabels, PredictProba | builder-only |
| **MBSGDRegressor** (PILOT B) | linear/mbsgd_regressor.rs | B | **yes** | Fit, Predict | builder-only |
| *(sgd_config.rs is a shared config, not an estimator)* | — | — | — | — | — |

### decomposition/ (3)
| PCA | decomposition/pca.rs | A | no | Fit, Transform | `new(n_components: usize)` |
| TruncatedSvd | decomposition/truncated_svd.rs | A | no | Fit, Transform | `new(n_components: usize)` |
| IncrementalPCA | decomposition/incremental_pca.rs | A | no | **Fit, PartialFit, Transform** | `new(n_components, whiten, batch_size: Option<usize>)` — only `PartialFit` consumer |

### cluster/ (3 on old traits; HDBSCAN already migrated)
| KMeans (LATE) | cluster/kmeans.rs | A-multi | no | Fit, PredictLabels | `new(n_clusters, seed)` + `with_init(n_clusters, init: Vec<F>)` + `with_opts(n_clusters, seed, max_iter, tol)` |
| DBSCAN | cluster/dbscan.rs | A | no | Fit, PredictLabels | `new(eps: f64, min_samples: usize)` |
| SpectralClustering | cluster/spectral_clustering.rs | A-multi-arg | no | Fit | `new(n_clusters, n_components: Option, affinity: String, gamma: F, n_neighbors, seed)` |

*(VERIFIED: `spectral_embedding.rs` is `struct SpectralEmbedding<F>` with arg-taking `new` and NO `crate::traits` import (same shape A' as KernelRidge — inherent `fit`, no trait). It has a PyO3 wrap (spectral.rs:65) and a test. **It IS in scope** for the state-param + builder + zero-arg-new retrofit and should ADOPT `typestate::Fit`/`Transform`. This makes the retrofit set 27 (`crate::traits`) + KernelRidge + SpectralEmbedding = **29 estimators**.)*

### covariance/ (2)
| EmpiricalCovariance | covariance/empirical_covariance.rs | A | no | Fit | `new(assume_centered: bool, store_precision: bool)` |
| LedoitWolf | covariance/ledoit_wolf.rs | A | no | Fit | `new(assume_centered: bool)` |

### projection/ (2)
| GaussianRandomProjection | projection/gaussian.rs | A | no | Fit, Transform | `new(n_components: NComponents, seed: u64, eps: f64)` |
| SparseRandomProjection | projection/sparse.rs | A | no | Fit, Transform | `new(…)` (multi-arg) |

### density/ (1)
| KernelDensity | density/kernel_density.rs | A | no | **ScoreSamples** (no Fit in the `use` line — verify; likely Fit + ScoreSamples) | `new(kernel: KdKernel, bandwidth: BandwidthSpec)` |

### neighbors/ (3)
| NearestNeighbors | neighbors/nearest.rs | A | no | Fit, KNeighbors | `new(n_neighbors: usize)` |
| KNeighborsClassifier | neighbors/classifier.rs | A | no | Fit, PredictLabels, PredictProba | `new(n_neighbors: usize)` |
| KNeighborsRegressor | neighbors/regressor.rs | A | no | Fit, Predict | `new(n_neighbors: usize)` |

### kernel_ridge/ (1) — VERIFIED: shape A', no trait
| KernelRidge | kernel_ridge/kernel_ridge.rs | **A' (arg-new, NO trait)** | no | **NONE** — does not `use crate::traits`; `fit`/`predict` are inherent methods on `KernelRidge<F>` | `new(kernel: KernelKind, alpha: F, gamma: Option<F>, degree: F, coef0: F)` |

*(VERIFIED: `kernel_ridge.rs` imports only `crate::error::AlgoError`, has `struct KernelRidge<F>` with no state param, arg-taking `new`, and NO `crate::traits` import — its `fit`/`predict` are inherent. Retrofit = add `<F, S=Unfit>` + builder + zero-arg new + ADOPT the `typestate::Fit`/`Predict` traits (it's the one estimator that was never on the trait surface). In scope.)*

### naive_bayes/ (5)
| GaussianNB | naive_bayes/gaussian_nb.rs | B | **yes** | Fit, PredictLabels, PredictLogProba, PredictProba | builder-only |
| MultinomialNB | naive_bayes/multinomial_nb.rs | B | **yes** | (same 4) | builder-only |
| BernoulliNB | naive_bayes/bernoulli_nb.rs | B | **yes** | (same 4) | builder-only |
| ComplementNB | naive_bayes/complement_nb.rs | B | **yes** | (same 4) | builder-only |
| CategoricalNB | naive_bayes/categorical_nb.rs | B | **yes** | (same 4) | builder-only |

**Reconciliation with CONTEXT.md:**
- CONTEXT says "~21 not born with it" and "11 already have a builder()". **Verified reality: 27 estimators on `crate::traits`; 9 of them already have `builder()`** (LinearSVC, LinearSVR, MBSGDClassifier, MBSGDRegressor, + 5 NB). The "11" in CONTEXT counted UMAP+HDBSCAN, which are NOT on `crate::traits` (already migrated) — so among the retrofit set, **9 are shape B, 18 are shape A**. KernelRidge + SpectralEmbedding are unverified and may add to the count.
- **Shape A (no builder, arg-taking new, ON old traits): 18** — full build-out (add state param + builder + zero-arg new + trait migration + call-site sweep).
- **Shape B (has builder, no state param): 9** — typestate-param + trait-swap-only (builder already exists; just add `<F,S>` + migrate `crate::traits` → `typestate`).
- **Shape A' (arg-new, NEVER on a trait — inherent fit/predict): 2** — KernelRidge + SpectralEmbedding. Full build-out PLUS *adopting* `typestate::Fit`/`Predict`/`Transform` (they had no trait before).
- **TOTAL retrofit set: 29 estimators** (27 on `crate::traits` + 2 shape-A'). This is the number the plan should drive, not "~21".

## Current Trait Surface (the 9 old traits → typestate ports)

`crates/mlrs-algos/src/traits.rs` defines **9** traits, all generic `<F: Float + CubeElement + Pod>`, all method signatures take `&mut self` (Fit/PartialFit) or `&self` (the rest) and key fitted state on `Option<DeviceArray>` with a runtime `AlgoError::NotFitted` guard:

| Old trait (traits.rs) | Signature shape | typestate.rs status |
|----------------------|-----------------|---------------------|
| `Fit<F>` | `fn fit(&mut self, …) -> Result<&mut Self>` | **EXISTS** — consuming-self, `type Fitted`, `fn fit(self,…) -> Result<Self::Fitted>` |
| `PartialFit<F>` | `fn partial_fit(&mut self,…) -> Result<&mut Self>` | **EXISTS** — consuming-self, `type Fitted` (defined-but-unused; IncrementalPCA is the consumer) |
| `Predict<F>` | `fn predict(&self,…) -> Result<DeviceArray<F>>` | **EXISTS** — `&self`, gated on `Fitted` impl |
| `Transform<F>` (+ default `inverse_transform`) | `fn transform(&self,…)` | **EXISTS** — `&self`, but the default `inverse_transform` is NOT yet ported (PCA needs it; see Pitfall) |
| `PredictLabels<F>` | `fn predict_labels(&self,…) -> DeviceArray<i32>` | **MISSING** — must be ADDED to typestate.rs |
| `KNeighbors<F>` | `fn kneighbors(&self,…) -> (DeviceArray<F>, DeviceArray<i32>)` | **MISSING** — add |
| `ScoreSamples<F>` | `fn score_samples(&self,…) -> DeviceArray<F>` | **MISSING** — add |
| `PredictProba<F>` | `fn predict_proba(&self,…) -> DeviceArray<F>` | **MISSING** — add |
| `PredictLogProba<F>` | `fn predict_log_proba(&self,…) -> DeviceArray<F>` | **MISSING** — add |

**typestate.rs currently has 4 of 9** (`Fit`, `PartialFit`, `Predict`, `Transform`). **Wave 0 of this phase must ADD the 5 missing accessor traits** (`PredictLabels`, `KNeighbors`, `ScoreSamples`, `PredictProba`, `PredictLogProba`) to `typestate.rs`, each as a `&self` trait intended to be implemented ONLY on the `Fitted`-tagged estimator (mirroring how `Transform` is impl'd only on `Umap<F, Fitted>`). It must also port `Transform::inverse_transform`'s default (PCA-only) so PCA doesn't regress.

**Resolution of the "Phase 12 Not started yet UMAP/HDBSCAN born-with-convention" question:** typestate.rs DOES exist and is populated (sealed `State`, `Unfit`/`Fitted`, 4 lifecycle traits, `validate_geometry` helper). UMAP (`manifold/umap.rs`) and HDBSCAN (`cluster/hdbscan.rs`) DO implement `typestate::Fit`/`Transform` already. So the convention landed in the source tree (Phase 12 foundation + Phase 14/15 born-with) even if the ROADMAP status text lags. **The migration target surface is real and partially built; the gap is the 5 discrete-output traits.**

## Exemplar Shape (the template to replicate — from umap.rs / hdbscan.rs)

The born-with-convention pattern, verified in `manifold/umap.rs` and `cluster/hdbscan.rs`, has exactly these parts. Every retrofit reproduces them:

```rust
// Source: crates/mlrs-algos/src/manifold/umap.rs (verbatim structure)

// 1. Struct gains S = Unfit + a PhantomData<S> slot (and PhantomData<F> if F is
//    otherwise unused in the Fitted-only fields — hdbscan.rs carries _float too).
pub struct Ridge<F, S = Unfit> {
    alpha: F,              // hyperparameter fields — UNCHANGED (D-03)
    fit_intercept: bool,   // UNCHANGED
    coef_: Option<DeviceArray<ActiveRuntime, F>>,      // fitted state UNCHANGED
    intercept_: Option<DeviceArray<ActiveRuntime, F>>, // (Some by construction on Fitted)
    _state: PhantomData<S>,                             // NEW: zero-sized marker
}

// 2. new() is the SINGLE source of sklearn defaults, in the Unfit state, zero-arg.
impl<F> Ridge<F, Unfit> where F: Float + CubeElement + Pod {
    pub fn new() -> Self { /* alpha = 1.0 (sklearn Ridge default), fit_intercept = true, … */ }
    pub fn builder() -> RidgeBuilder { RidgeBuilder::default() }
    pub fn into_builder(self) -> RidgeBuilder { /* copy hyperparams back */ }
    pub fn hyperparams_eq(&self, other: &Self) -> bool { /* BLDR-01 defaults-eq test */ }
}
impl<F> Default for Ridge<F, Unfit> { fn default() -> Self { Self::new() } }

// 3. Builder: owned chained setters; Default RE-DERIVES from new() (NOT literal copies).
#[derive(Debug, Clone, Copy)]
pub struct RidgeBuilder { alpha: f64, fit_intercept: bool }
impl Default for RidgeBuilder { fn default() -> Self { Ridge::<f64, Unfit>::new().into_builder() } }
impl RidgeBuilder {
    pub fn alpha(mut self, v: f64) -> Self { self.alpha = v; self }
    pub fn fit_intercept(mut self, v: bool) -> Self { self.fit_intercept = v; self }
    // data-INDEPENDENT validation here (alpha >= 0) → BuildError; data-DEPENDENT stays in fit()
    pub fn build<F>(self) -> Result<Ridge<F, Unfit>, BuildError> where F: Float + CubeElement + Pod { … }
}

// 4. typestate::Fit CONSUMES self, returns the Fitted-tagged sibling.
impl<F> typestate::Fit<F> for Ridge<F, Unfit> where F: Float + CubeElement + Pod {
    type Fitted = Ridge<F, Fitted>;
    fn fit(self, pool, x, y, shape) -> Result<Ridge<F, Fitted>, AlgoError> {
        validate_geometry(x, shape)?;          // shared helper (typestate.rs)
        /* …EXISTING fit-body math, byte-identical (D-03)… */
        Ok(Ridge { /* hyperparams moved from self */, coef_: Some(coef), …, _state: PhantomData })
    }
}

// 5. Predict (and the discrete-output accessors / fitted-attr getters) exist ONLY on Fitted.
impl<F> typestate::Predict<F> for Ridge<F, Fitted> { fn predict(&self, …) { … } }
impl<F> Ridge<F, Fitted> {
    pub fn coef(&self, pool) -> Vec<F> { self.coef_.as_ref().expect("Some on Fitted").to_host(pool) }
    pub fn intercept(&self, pool) -> F { … }   // NO NotFitted branch — typestate replaces it
}
```

**Key shape facts** (each a gate the planner should encode as a verification step):
- `new()` lives on `impl<F> T<F, Unfit>` and is **zero-arg** (sklearn defaults inline).
- `Builder::default()` calls `T::<f64, Unfit>::new().into_builder()` — **NEVER re-list literal defaults** (Pitfall: default drift).
- `build<F>()` is generic over `F` on the (non-generic) builder; validates data-independent params → `BuildError`.
- `Fit::fit` consumes `self`, runs the **unchanged** math, and **reconstructs** the struct into the `Fitted` arm (field-by-field move; the only new field is `_state: PhantomData`).
- Fitted accessors drop the `ok_or(NotFitted)` and use `.expect("… by construction on T<F, Fitted>")`.
- `fit_transform` / `fit_predict` convenience wrappers live on `T<F, Unfit>` and internally call `self.fit(…)?` then the Fitted accessor (see umap.rs:215, hdbscan.rs:284).

## Derive-Macro Evaluation (CONTEXT defers this to the researcher)

**Recommendation: hand-write the retrofit. Do NOT introduce a derive/declarative macro.** Confidence: HIGH.

Rationale, grounded in the verified inventory:

1. **The shapes are heterogeneous, not uniform.** A macro pays off when N near-identical units differ only in a parameter list. Here the 27 estimators span: 2 fit shapes (`Fit` vs `Fit`+`PartialFit` for IncrementalPCA), 6 distinct accessor-trait combinations (Fit+Predict; Fit+Transform; Fit+PredictLabels; Fit+PredictLabels+PredictProba; Fit+PredictLabels+PredictProba+PredictLogProba; Fit+KNeighbors; ScoreSamples), `Option`-vs-`String`-vs-`enum` hyperparameter types, multi-constructor cases (KMeans `with_init`/`with_opts`, SpectralClustering 6-arg), and 9 that already have a hand-tuned builder with bespoke `BuildError` variants and validation. A macro general enough to cover all of this becomes a second DSL to maintain.

2. **The per-estimator green-suite gate (D-03) is the dominant cost, and a macro doesn't reduce it.** The expensive, risk-bearing step is "migrate one estimator → run its oracle suite → confirm 1e-5/exact gate holds → commit." That is irreducible regardless of how the boilerplate is emitted. The boilerplate itself is a ~30-minute mechanical edit per estimator following the umap.rs template; a macro would front-load equal-or-greater design time and add a debugging-opacity tax on the gate (a macro-expansion error is harder to localize than a hand-written impl when a suite goes red).

3. **9 estimators already have hand-written builders** that a macro would have to either leave alone (defeating uniformity) or rewrite (re-introducing default-drift risk against shipped gates — exactly what D-03 forbids).

4. **Debuggability + reviewability under a broad parallel-unsafe sweep.** This phase is explicitly the one broad-edit phase; reviewers and the code-review gate read diffs. Hand-written impls produce reviewable, greppable, bisectable diffs. A macro hides the migration in expansion, making the "did the fit body actually stay byte-identical?" review (the core safety question) materially harder.

5. **LOC math doesn't favor it at this count.** ~27 estimators × ~40 lines of mechanical builder/typestate scaffolding ≈ 1,100 LOC, but most of it is copy-adapt from umap.rs, not novel. A `#[derive(Typestate)]` proc-macro that handles the field-move reconstruction, the `Fitted`-gated accessor routing, and the 9-trait surface would itself be 400–800 LOC of proc-macro + a new crate dependency in the workspace, plus its own tests — net negative at N=27, and it would still need per-estimator escape hatches.

**Counter-acknowledgment:** if the project later grows to ~60+ estimators with uniform shapes, revisit. For Phase 16, hand-written is correct. (This keeps the Deferred-Ideas macro genuinely deferred — no scope change either way, per CONTEXT.)

## Call-Site Migration (verified — the real surface)

**Verified count:** `grep -rohE '\bEstimator(::<…>)?::(new|with_init|with_opts)\(' crates/` → **85 arg-taking constructor calls repo-wide** (not 137, not tests-only). Distribution:

| Location | Sites | Migration shape |
|----------|-------|------------------|
| `crates/mlrs-py/src/estimators/*.rs` (8 files) | ~42 | **The hidden surface.** `let mut est = T::<f32>::new(args); est.fit(&mut pool, …)?` → `let fitted = T::<f32>::builder().…build().map_err(build_err_to_py)?.fit(&mut pool, …).map_err(algo_err_to_py)?;` then store `Any*::F32(fitted)`. Arm types change `T<f32>` → `T<f32, Fitted>`. |
| `crates/mlrs-algos/tests/*.rs` (~22 files) | ~40 | `T::<F>::new(args)` → `T::<F>::builder().param(args)…build()?`; fit changes from `let mut est = …; est.fit(…)?` to `let fitted = est.fit(…)?` (consuming) |
| `crates/mlrs-algos/src/cluster/spectral_clustering.rs` | 1 | internal self-construction |
| `crates/mlrs-py/tests/*.rs` | 1 | smoke test |

**Per-estimator breakdown** (arg-taking, incl. turbofish): DBSCAN 8, IncrementalPCA 7, GaussianRandomProjection 7, SpectralClustering 5, SparseRandomProjection 5, Ridge 4, NearestNeighbors 4, LinearRegression 4, KernelDensity 4, KMeans 4 (+3 `with_init` +2 `with_opts`), TruncatedSvd 3, Pca 3, LogisticRegression 3 (`with_opts`)+1, LedoitWolf 3, KNeighborsRegressor 3, KNeighborsClassifier 3, EmpiricalCovariance 3, Lasso 2 (`with_opts`)+1, ElasticNet 2 (`with_opts`)+1.

**Mechanical recipe** (per call site):
1. `Ridge::<F>::new(alpha, fit_intercept)` → `Ridge::<F>::builder().alpha(alpha).fit_intercept(fit_intercept).build()?` (note: builder setters take `f64`/native; `new` took `F` for some — watch the `alpha as f32` casts in PyO3, e.g. linear.rs:246).
2. `let mut est = …; est.fit(p, x, y, shape)?; /* use est */` → `let est = ….fit(p, x, y, shape)?; /* use est (now Fitted) */` — drop the `mut`, the binding is now the Fitted value.
3. In PyO3: the fitted value goes into `Any*::F32(est)` / `Any*::F64(est)`; **the enum arm type must be updated to `T<f32, Fitted>`**.

**Call sites that resist:**
- **KMeans multi-constructor** (D-06 late case): `with_init(n_clusters, init: Vec<F>)` injects a caller-supplied init matrix; `with_opts(n_clusters, seed, max_iter, tol)`. The builder must gain an `init(Option<Vec<F>>)` setter (the fixture tests use injected init — kmeans_test.rs:104 `KMeans::<F>::with_init(KM_K, init_host)`), so the builder surface is wider than the others. Keep KMeans last (recipe proven).
- **SpectralClustering 6-arg `new`** including `affinity: String` and `Option<usize>` — straightforward but the most fields; the builder needs an `Option`/`String` setter pattern.
- **GaussianRandomProjection `new(n_components: NComponents, …)`** — `NComponents` is an enum; the builder setter must accept it.
- **PyO3 `alpha as f32` narrowing** (linear.rs:246, 392, 548, 699): builder setters are `f64`-typed (see MBSGDRegressorBuilder), so the f32 arm passes `f64` to the builder then `build::<f32>()` monomorphizes — confirm the builder lowers `f64 → F` internally (mbsgd does this via `SgdConfig`); for the new shape-A builders, decide whether setters take `f64` (uniform with mbsgd/umap) or `F` (matches old `new`). **Recommend `f64` setters** for uniformity with the shipped builders; `build::<F>()` casts.

## Pilot Recipe

### Pilot A — Ridge (shape A, full build-out)
Suite: `cargo test --features cpu --test ridge_test` (oracle fixture `Ridge(solver='cholesky', fit_intercept=True)`, 1e-5 abs+rel; f64 gate + f32). Call sites: ridge_test.rs (2, incl. `Ridge::<f32>::new(1.0,true)` at :276) + linear.rs PyO3 (2, at :246/:254).

Steps:
1. **Prereq (Wave 0):** ensure `typestate.rs` has `Predict` (it does). No new trait needed for Ridge.
2. Add `S = Unfit` + `_state: PhantomData<S>` to `struct Ridge<F, S = Unfit>`.
3. Move `new(alpha, fit_intercept)` → zero-arg `new()` on `impl Ridge<F, Unfit>` with sklearn defaults (`alpha = 1.0`, `fit_intercept = true`); add `RidgeBuilder` (`.alpha(f64)`, `.fit_intercept(bool)`, `build::<F>()` validating `alpha >= 0` → `BuildError::InvalidAlpha`), `Default`, `into_builder`, `hyperparams_eq`.
4. Swap `use crate::traits::{Fit, Predict}` → `use crate::typestate::{Fit, Predict, Fitted, Unfit, validate_geometry}`. Convert `Fit::fit(&mut self) -> &mut Self` → `fit(self) -> Ridge<F, Fitted>` (reconstruct struct, byte-identical math). Move `coef`/`intercept` accessors + `impl Predict` onto `impl Ridge<F, Fitted>`, dropping `NotFitted`.
5. Migrate the 4 call sites (ridge_test.rs ×2, linear.rs PyO3 ×2); update `AnyRidge::{F32,F64}` arms to `Ridge<f32,Fitted>`/`Ridge<f64,Fitted>`.
6. `cargo test --features cpu --test ridge_test` green (and `cargo build -p mlrs-py --features …` compiles). Commit.

### Pilot B — MBSGDRegressor (shape B, trait-swap-only)
Suite: `cargo test --features cpu --test mbsgd_regressor_test`. **Builder already exists** (verified mbsgd_regressor.rs:51-288 — full `MBSGDRegressorBuilder` with `Default`, all setters, `build<F>() -> Result<MBSGDRegressor<F>, BuildError>`).

Steps:
1. Add `S = Unfit` + `PhantomData<S>` to `struct MBSGDRegressor<F, S = Unfit>` (note: it has no float-only field forcing `PhantomData<F>` — `config: SgdConfig` is F-independent, `coef_/intercept_` carry `F`, so `<F, S>` is fine).
2. Change `build<F>()` return `MBSGDRegressor<F>` → `MBSGDRegressor<F, Unfit>` (the ONLY builder edit).
3. Swap `use crate::traits::{Fit, Predict}` → typestate; convert `fit(&mut self)->&mut Self` → `fit(self)->MBSGDRegressor<F,Fitted>`; move accessors + `Predict` to `impl …<F, Fitted>`. **Also gate `PartialFit`** if mbsgd implements streaming (verify — `incremental_pca` is the named `PartialFit` consumer; mbsgd may not).
4. Migrate call sites: mbsgd_regressor_test.rs + linear.rs PyO3 (linear.rs:1271 uses `MBSGDRegressor::<f32>::builder()…` — already builder-shaped, so only the `est.fit(&mut)` → consuming-self + `AnyMBSGDRegressor` arm type change).
5. Suite green. Commit.

## Python Shim (SHIM-01) — verified state

**Existing pure-Python shim classes (verified `grep '^class' mlrs/*.py`): 18 classes** (one is `MlrsBase`), i.e. **17 estimator shims**, NOT "v1 12":
DBSCAN, KMeans, EmpiricalCovariance, LedoitWolf, PCA, IncrementalPCA, TruncatedSVD, LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, GaussianRandomProjection, SparseRandomProjection.

**PyO3 wraps exist for 32 estimators** (verified `#[pyclass(name=…)]`). **Missing pure-Python shim classes (14)** — these have a Rust `#[pyclass]` but NO `mlrs.<Name>` Python class:
LinearSVC, LinearSVR, MBSGDClassifier, MBSGDRegressor, GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB, KernelRidge, KernelDensity, SpectralClustering, SpectralEmbedding, **UMAP, HDBSCAN**.

*(That is 15 names; CONTEXT's "v2 18 + UMAP/HDBSCAN" target implies the in-scope shim set. The planner should reconcile the exact target list against EXPECTED_PARAMS, but the safe reading: add a faithful `MlrsBase` subclass for every PyO3-wrapped estimator that lacks one, prioritizing the SHIM-02 pair UMAP/HDBSCAN and the 5 NB + 4 SVM/MBSGD families.)*

**Each new shim class is mechanical** (template from `mlrs/linear.py::Ridge`): subclass the right mixin + `MlrsBase`, `__init__(self, <sklearn-named args with sklearn defaults>, output_type="input")` storing each verbatim, a `fit` that normalizes input, calls `self._ext().<Name>(...)`, stores `self._mlrs_obj`, calls `self._post_fit(cols)`, returns `self`, plus the dtype-suffixed fitted-attr accessors via `self._suffixed(...)`. `get_params`/`set_params`/`clone` come free from `BaseEstimator` given a faithful `__init__`. The `MlrsBase` machinery (`_normalize`, `_check_predict_X`, `_suffix`, `_post_fit`, `__sklearn_tags__`) is already complete.

**Param-name boundary mappings to honor** (sklearn-named, verified pattern): LogisticRegression exposes capital `C` (not Rust `c`); KMeans exposes `random_state` (mapped to Rust `seed`). Replicate this for new classes: UMAP `random_state`/`n_neighbors`/`min_dist`/etc.; HDBSCAN `min_cluster_size`/`min_samples`/`cluster_selection_method`/etc.

## Static Shim Gate (SHIM-03 / D-07)

Existing infra (verified):
- `test_shims.py` (144 lines): parametrizes over `ALL_12`; tests importability without ext, mixin composition, `get_params`/`set_params` round-trip, `clone` preserves unfitted params, `output_type` present, family-specific surface (DBSCAN no predict, NN has kneighbors), fitted-attr raises before fit, `fit` returns self.
- `test_params.py` (139 lines): `EXPECTED_PARAMS`/`SET_PARAM` tables over `ALL_12`; default-param-names-match-sklearn, set_params round-trip, **runtime** init-purity (`test_init_purity_stores_kwargs_verbatim` — runtime, not AST), capital-C, random_state.
- `test_estimator_checks.py` (184 lines): uses `parametrize_with_checks` with a `_expected_failed_checks` xfail map for by-design failures (sparse/pickle/dtype-object/etc.).

**Work for D-07:**
1. **Extend `ALL_12` → the full shim set** in all three test files (replace the hard-coded `ALL_12` lists; add `EXPECTED_PARAMS`/`SET_PARAM` entries for every new class incl. UMAP/HDBSCAN).
2. **ADD the AST-based `__init__`-purity assertion (D-07 step 3) — this is NEW.** Verified: `grep 'import ast'` returns NOTHING in the test tree. Today's purity check is runtime-only. Add a test that `ast.parse(inspect.getsource(cls.__init__))` and asserts the body is only `self.<arg> = <arg>` assignments (each ctor arg stored verbatim under the same name, no `Call`/`BinOp`/validation nodes), plus the `output_type` assignment. This is the strongest SHIM-01 guarantee without FFI.
3. **Confirm the fit-free `estimator_checks` subset** named in D-07 (`check_no_attributes_set_in_init`, `check_parameters_default_constructible`, `check_get_params_invariance`) is reached. `parametrize_with_checks` yields these automatically; verify they are NOT in the xfail map (so they actually run green). They were not found explicitly in the file — confirm they pass for every new class (esp. `check_parameters_default_constructible`, which requires zero-arg constructibility — UMAP/HDBSCAN must construct with no required args).
4. Rust-side unit tests: the `*_test.rs` oracle suites already exercise the Rust surface; add/confirm the BLDR-01 `T::new().hyperparams_eq(&T::builder().build()?)` defaults-equality test per estimator (umap_test.rs has the template).

## PyO3 Wraps (SHIM-02) — already shipped; verify, don't rebuild

**Finding (HIGH confidence): `PyUMAP` and `PyHDBSCAN` already exist, are complete, and are registered.** This materially shrinks SHIM-02.
- `crates/mlrs-py/src/estimators/manifold.rs:74` `#[pyclass(name="UMAP")]` `PyUMAP` over `AnyUmap::{Unfit, F32, F64}`; `#[new]` with umap-learn-named params + sklearn defaults; `fit` with `py.detach` GIL release + `guard_f64()` before the F64 arm; `embedding_` accessors (f32/f64); registered `lib.rs:265`.
- `crates/mlrs-py/src/estimators/cluster.rs:324` `#[pyclass(name="HDBSCAN")]` `PyHDBSCAN` over `AnyHdbscan::{Unfit,F32,F64}`; `#[new]` with sklearn-named params; `fit` + `guard_f64`; `labels_()` getter (i32, noise=-1), `is_fitted`, `dtype`; registered `lib.rs:266`.

**Remaining SHIM-02 tasks — VERIFIED real gaps** (D-08 surface completion on the existing wraps):
- **PyUMAP is MISSING `transform` and `fit_transform`** (VERIFIED: `grep 'fn transform\|fn fit_transform' manifold.rs` → none; only `embedding_f32`/`embedding_f64` getters + `fit` exist). The Rust `Umap<F,Fitted>` HAS `Transform` + `fit_transform` (umap.rs:215, 549) — so this is forwarding work: add `#[pymethods] fn transform(...)` and `fn fit_transform(...)` on PyUMAP that match on the `AnyUmap::{F32,F64}` arm and call the Rust `Transform::transform` / `fit_transform`, with `guard_f64` + GIL release.
- **PyHDBSCAN is MISSING `fit_predict`, `probabilities_`, and `outlier_scores_`** (VERIFIED: only `labels_`/`fit`/`is_fitted`/`dtype` present). The Rust `Hdbscan<F,Fitted>` has `fit_predict` (hdbscan.rs:284) + GLOSH `outlier_scores_` (HDBS-03) + `probabilities_` (HDBS-01). Add the forwarding `#[pymethods]`. HDBS-04 `centroids_`/`medoids_` may also need surfacing.
- `n_features_in_` surfacing on both wraps (Python `MlrsBase._post_fit` handles the shim side once the pure-Python class exists).
- These are surface-completion edits on EXISTING wraps, NOT new wraps. The "two new PyO3 wraps" framing in CONTEXT is wrong — the wraps exist; only specific methods are missing.

## Sweep Ordering & Commit Granularity

**Recommended order** (D-06 module order, with dependencies):
1. **Wave 0 (blocking):** Add the 5 missing accessor traits to `typestate.rs` (`PredictLabels`, `KNeighbors`, `ScoreSamples`, `PredictProba`, `PredictLogProba`) + `Transform::inverse_transform` default. Nothing else can migrate until the target traits exist. Gate: `cargo build -p mlrs-algos` + `typestate_test.rs`.
2. **Pilots:** Ridge (A) → MBSGDRegressor (B), each its own commit + suite.
3. **linear/** remainder: LinearRegression, Lasso, ElasticNet, LogisticRegression, LinearSVC, LinearSVR, MBSGDClassifier.
4. **decomposition/**: PCA, TruncatedSvd, IncrementalPCA (the `PartialFit` case — exercises the multi-transition typestate; do after simpler linear cases).
5. **cluster/**: DBSCAN, SpectralClustering, then **KMeans last** (multi-constructor).
6. **covariance/**: EmpiricalCovariance, LedoitWolf.
7. **projection/**: Gaussian, Sparse.
8. **density/**: KernelDensity.
9. **neighbors/**: NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor.
10. **kernel_ridge/**: KernelRidge (verify its current trait state first).
11. **naive_bayes/**: 5 NB (shape B, builder exists — fast).
12. **Final commit:** delete `crates/mlrs-algos/src/traits.rs` + its `pub mod traits;` in lib.rs, once zero `use crate::traits` / `use mlrs_algos::traits` remain (grep must return empty across BOTH crates).

**Commit granularity: ONE estimator per commit** (estimator source + its `mlrs-algos/tests` call sites + its `mlrs-py/src/estimators` PyO3 call sites + arm-type change, all together), gated by that estimator's suite. Rationale: keeps the workspace compiling at every commit (a half-migrated estimator whose PyO3 wrap still calls the old trait won't build), gives clean bisect/revert under D-03, and matches `use_worktrees: false` sequential execution (per project memory: worktree isolation is broken; this phase is parallel-unsafe anyway).

## Common Pitfalls

### Pitfall 1: Default-value drift when args move to the builder
**What goes wrong:** Re-listing default literals in `Builder::default()` instead of deriving from `new()`; a typo (`alpha = 1.0` vs `0.1`) silently changes the oracle baseline and the 1e-5 gate fails — or worse, passes against a wrong fixture.
**Avoid:** `Builder::default()` MUST be `T::<f64, Unfit>::new().into_builder()` (umap.rs:309 pattern). `new()` is the single source. Add the `hyperparams_eq` defaults-equality unit test per estimator (BLDR-01).
**Warning sign:** the estimator's oracle suite goes red on a "pure construction" commit.

### Pitfall 2: Accidental fit-body edit during the move-to-consuming-self
**What goes wrong:** Rewriting `fit(&mut self){ self.coef_ = Some(x) }` → `fit(self) -> Fitted` tempts a "cleanup" that alters numerics (reorders a centering pass, changes an `F` cast). D-03 forbids ANY math change.
**Avoid:** Mechanically reconstruct: keep every compute line identical; only change the signature, the final `self.field = …` → struct-literal in the returned `Fitted` value, and `&mut Self`→`Self::Fitted`. Diff-review the fit body for ZERO compute deltas (the code-review gate is on).
**Warning sign:** any non-signature line inside `fn fit` changed in the diff.

### Pitfall 3: Deleting traits.rs while PyO3 still imports it (the hidden cross-crate break)
**What goes wrong:** `traits.rs` deletion compiles `mlrs-algos` but breaks `mlrs-py` (8 `use mlrs_algos::traits::…` files). The plan that only greps `mlrs-algos/tests` misses this.
**Avoid:** Before the deletion commit, `grep -rn 'mlrs_algos::traits\|crate::traits' crates/` must return EMPTY across both crates. Migrate each estimator's PyO3 wrap in the same commit as the estimator.
**Warning sign:** `cargo build -p mlrs-py` fails after a `mlrs-algos` commit.

### Pitfall 4: `AnyEstimator` arm type mismatch after typestate
**What goes wrong:** The `Any*::F32(T<f32>)` arms hold the no-state type; after retrofit the fitted value is `T<f32, Fitted>`. Forgetting to update the enum arm type → type error, or (if `S` defaults) a subtle `T<f32, Unfit>` stored as fitted.
**Avoid:** Change each `Any*::{F32,F64}` arm to `T<f32, Fitted>` / `T<f64, Fitted>` explicitly; the `Unfit` arm stays the stored-hyperparams struct (BLDR-04 runtime collapse unchanged).
**Warning sign:** PyO3 `fit` stores a value whose type rustc still prints as `T<f32>` (elided `Unfit`).

### Pitfall 5: IncrementalPCA's `PartialFit` multi-transition
**What goes wrong:** `PartialFit` must be impl'd on BOTH `Unfit` (first batch) and `Fitted` (subsequent batches) for `Fitted → Fitted` streaming (typestate.rs:183 doc). A naive port only impls it on `Unfit` and breaks multi-batch `partial_fit`.
**Avoid:** Impl `typestate::PartialFit for IncrementalPCA<F, Unfit>` (`type Fitted = IncrementalPCA<F, Fitted>`) AND `for IncrementalPCA<F, Fitted>` (`type Fitted = Self`). It's the only `PartialFit` consumer — do it after simpler cases.

### Pitfall 6: `check_parameters_default_constructible` requires zero-arg construction
**What goes wrong:** sklearn's estimator_checks (D-07 subset) needs every shim class constructible with no required args. The current `test_params._construct` special-cases `PCA(n_components=2)` — meaning PCA's shim has a REQUIRED arg. New classes must default ALL args (sklearn-style) or the check fails.
**Avoid:** Give every new shim `__init__` sklearn defaults for all params (UMAP `n_neighbors=15` etc.; HDBSCAN `min_cluster_size=5` etc.). Reconcile PCA's required `n_components` if it's pulled into the full matrix.

### Pitfall 7: Builder setter type (`f64` vs `F`) vs old `new` (`F`)
**What goes wrong:** Old shape-A `new` took `alpha: F`; the shipped builders (mbsgd/umap) take `f64` setters and cast in `build::<F>()`. Mixing conventions means PyO3 `alpha as f32` casts become inconsistent.
**Avoid:** Standardize on `f64` setters (uniform with shipped builders); `build::<F>()` does the `host_to_f64`/cast. Update PyO3 to pass `f64` to the builder (drop the `as f32` at the call, let `build::<f32>()` narrow).

## State of the Art

| Old Approach (legacy traits.rs) | Current Approach (typestate.rs) | When Changed | Impact |
|--------------------------------|----------------------------------|--------------|--------|
| `fit(&mut self) -> Result<&mut Self>` | `fit(self) -> Result<Self::Fitted>` (consuming, retags) | Phase 12 | predict-before-fit is a compile error, not runtime `NotFitted` |
| `Option<DeviceArray>` + runtime `NotFitted` guard | `Fitted`-gated accessors, `.expect("Some by construction")` | Phase 12 | Fitted accessors have no error branch |
| arg-taking `new(args)` + `with_*` | zero-arg `new()` + owned builder, single-source defaults | Phase 12/16 | `T::new() == T::builder().build()?` == sklearn default |
| `T<F>` PyO3 enum arms | `T<F, Fitted>` arms behind the same `Unfit/F32/F64` collapse | Phase 16 | BLDR-04 surface unchanged at the Python boundary |

**Deprecated/outdated (removed at phase end):**
- `crates/mlrs-algos/src/traits.rs` — all 9 traits hard-deleted (D-01) once every estimator + every PyO3 importer is migrated.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A5 | Builder setters should be `f64`-typed (uniform with mbsgd/umap), not `F` | Call-Site Migration / Pitfall 7 | If `F`-typed chosen, PyO3 casts differ; either works but must be consistent — a planner decision, not a fact gap |
| A6 | The "v2 18" shim target = a faithful `MlrsBase` subclass for every PyO3-wrapped estimator lacking one (14 missing, incl. UMAP/HDBSCAN) | Python Shim | The exact 18-vs-more target should be reconciled against EXPECTED_PARAMS; under/over-building the shim set |
| A7 | `check_no_attributes_set_in_init` / `check_parameters_default_constructible` / `check_get_params_invariance` run green via `parametrize_with_checks` and are not in the xfail map | Static Shim Gate | If xfailed, D-07 step 4 isn't actually exercised |

**RESOLVED during research (formerly assumptions, now VERIFIED facts):**
- ~~A1~~ KernelRidge: VERIFIED `struct KernelRidge<F>`, arg-`new`, NO `crate::traits` import → shape A' (inherent fit, must adopt traits). In scope. Count → 29.
- ~~A2~~ SpectralEmbedding: VERIFIED same shape A' (arg-`new`, no trait). In scope. Count → 29.
- ~~A3~~ KernelDensity: VERIFIED imports/impls ONLY `ScoreSamples` (no `Fit`) — score-only lifecycle; its `fit` is inherent. Retrofit must gate `ScoreSamples` on `Fitted` and decide its fit-trait adoption.
- ~~A4~~ PyUMAP/PyHDBSCAN surface gaps: VERIFIED real — PyUMAP lacks `transform`/`fit_transform`; PyHDBSCAN lacks `fit_predict`/`probabilities_`/`outlier_scores_`. These ARE real SHIM-02 tasks (not pure verification). See §PyO3 Wraps.

## Open Questions

1. **Exact shim target list ("v2 18 + UMAP/HDBSCAN").**
   - Known: 18 Python classes today (17 estimators + MlrsBase); 32 PyO3 wraps; 14 missing shims.
   - Unclear: whether all 14 are in-scope or a curated 18+2.
   - Recommendation: planner reconciles against `EXPECTED_PARAMS`; default to "shim every wrapped estimator," prioritizing UMAP/HDBSCAN + the NB/SVM families.

2. **Builder setter type convention (`f64` vs `F`) — A5, a decision not a gap.**
   - Recommendation: `f64` setters uniform with shipped mbsgd/umap builders; `build::<F>()` narrows. Lock this in the plan's Wave 0 so all 29 retrofits follow one convention.

3. **KernelDensity / KernelRidge / SpectralEmbedding fit-trait adoption.**
   - These three have inherent (non-trait) `fit`. The retrofit must decide: do they adopt `typestate::Fit` (uniform surface, lets `traits.rs`-deletion grep stay clean) or keep an inherent consuming-`self` fit?
   - Recommendation: adopt `typestate::Fit` for uniformity — it's the single-surface end-state D-01 wants.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust `--features cpu` | All Rust gate suites (f64 + f32) | ✓ | workspace | — |
| Rust `--features rocm` | f32 GPU gate | ✓ (per project memory, gfx1100) | ROCm 7.1.1 | f64-on-rocm SKIPs-with-log |
| sklearn (Python) | static shim gate (`estimator_checks`, `BaseEstimator`) | ✓ (in Python test env) | >=1.6 | — |
| maturin + pyarrow (compiled `_mlrs`) | LIVE `check_estimator` FFI run | ✗ | — | **Static path only (D-07/SHIM-03); live run → UAT** (per project memory "Python wheel untestable in env") |
| numpy | shim egress + AST test | ✓ | — | — |

**Missing dependencies with no fallback:** none that block this phase (the FFI gap is by-design deferred).
**Missing dependencies with fallback:** maturin+pyarrow → static shim gate is the maximum verifiable; live `check_estimator` routes to UAT.

## Validation Architecture

`nyquist_validation: true` (config.json) → section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework (Rust) | `cargo test` integration tests in `crates/*/tests/` (AGENTS.md §2 — NO in-source `#[cfg(test)]`; verified zero in src) |
| Framework (Python) | `pytest` in `crates/mlrs-py/python/tests/` |
| Config | workspace `Cargo.toml`; per-test oracle fixtures (committed `.npz` blobs) |
| Quick run command | `cargo test --features cpu --test <estimator>_test` (per-estimator gate) |
| Full suite command | `cargo test --features cpu` (WARNING: per memory, full run is slow ~6min+ and can exhaust disk; prefer targeted) |
| Compile-fail gate | `cargo test --features cpu --test compile_fail` (trybuild — predict-before-fit proof) |
| Python shim gate | `pytest crates/mlrs-py/python/tests/test_shims.py test_params.py test_estimator_checks.py` (importable without `_mlrs`) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BLDR-03 | Each estimator's oracle gate holds post-retrofit | unit/oracle | `cargo test --features cpu --test ridge_test` (×27 per-estimator) | ✅ (all `*_test.rs` exist) |
| BLDR-03 | Defaults equality `new()==builder().build()?` | unit | `cargo test --features cpu --test <est>_test` (add `hyperparams_eq` case) | ⚠️ Wave 0 (per-estimator; umap_test has template) |
| BLDR-03 | predict-before-fit is a compile error | compile-fail | `cargo test --features cpu --test compile_fail` (extend `tests/ui/` per estimator family) | ✅ harness exists; ⚠️ add fixtures |
| BLDR-03 | `traits.rs` fully removed | grep gate | `! grep -rq 'mlrs_algos::traits\|crate::traits' crates/` | ✅ (grep) |
| SHIM-01 | get/set_params + clone round-trip per class | python | `pytest …/test_params.py` | ✅ extend `ALL_12` |
| SHIM-01 | AST `__init__`-purity | python | `pytest …/test_params.py::test_init_purity_ast` | ❌ Wave 0 — NEW (no `import ast` today) |
| SHIM-02 | UMAP/HDBSCAN Py wrap surface (D-08) | rust/python | existing `mlrs-py` tests + `test_shims.py` | ✅ wraps shipped; ❌ `transform`/`fit_transform` (UMAP) + `fit_predict`/`probabilities_`/`outlier_scores_` (HDBSCAN) VERIFIED missing — add |
| SHIM-03 | fit-free `estimator_checks` subset | python | `pytest …/test_estimator_checks.py` | ✅ extend estimator list |

### Sampling Rate
- **Per task commit:** the migrated estimator's `cargo test --features cpu --test <est>_test` (+ `cargo build -p mlrs-py` to catch the PyO3 break).
- **Per wave merge:** the module's suites + `cargo build` both crates.
- **Phase gate:** `cargo test --features cpu` (targeted set) green + `compile_fail` green + the full `pytest` shim suite green + the `traits.rs`-gone grep returns empty. Live `check_estimator` → UAT.

### Wave 0 Gaps
- [ ] `typestate.rs`: add `PredictLabels`, `KNeighbors`, `ScoreSamples`, `PredictProba`, `PredictLogProba` + `Transform::inverse_transform` default — blocks all estimators implementing those traits.
- [ ] `test_params.py`: ADD AST-based `__init__`-purity test (`import ast`, `inspect.getsource`) — D-07 step 3, does not exist.
- [ ] `test_shims.py`/`test_params.py`/`test_estimator_checks.py`: replace `ALL_12` with the full shim set; add `EXPECTED_PARAMS`/`SET_PARAM` rows for every new class incl. UMAP/HDBSCAN.
- [ ] New pure-Python shim modules/classes for the 14 missing estimators (mechanical, MlrsBase template).
- [ ] `tests/ui/`: per-estimator-family compile-fail fixtures (or accept the UMAP fixture as the representative typestate proof).
- [ ] PyUMAP: add `transform`/`fit_transform` `#[pymethods]`; PyHDBSCAN: add `fit_predict`/`probabilities_`/`outlier_scores_` `#[pymethods]` (VERIFIED missing).
- [ ] Lock the builder-setter type convention (`f64`) before the sweep.

## Security Domain

`security_enforcement: true`, ASVS level 1. This is a construction/lifecycle refactor with no new data-handling surface; the security-relevant property is that **the retrofit must not weaken existing input validation**.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | **yes** | The data-DEPENDENT geometry guard (`validate_geometry`, typestate.rs:59) MUST stay at the top of every ported `fit`, BEFORE any device launch. Data-INDEPENDENT hyperparameter validation moves to `Builder::build()` → `BuildError` (e.g. `alpha >= 0`), and must not be dropped in the move. The PyO3 `guard_f64()` (D-04) before any F64 upload must remain. |
| V6 Cryptography | no | — |

### Known Threat Patterns for {Rust typestate retrofit + PyO3 boundary}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Untrusted host geometry reaching a device read | Tampering / DoS | `validate_geometry` before launch — preserved verbatim in every ported `fit` (ASVS V5) |
| Dropped hyperparameter validation when args move to builder | Tampering | Validation relocated to `build()` → typed `BuildError`, not silently removed (e.g. Ridge `alpha < 0`, umap `min_dist <= spread`) |
| f64 on f64-incapable backend → device fault | DoS | `guard_f64()` BEFORE upload on the F64 arm — unchanged (D-04) |
| Rogue `impl State` introducing a third lifecycle state | Tampering | `State` is sealed (typestate.rs sealed::Sealed) — closed set; no action needed, just don't break the seal |
| Mutex-poison brick on a panicked `fit` | DoS | PyO3 uses `lock_pool()` (recovers from poison) not `.lock().expect()` — keep this in migrated PyO3 `fit` bodies |

## Sources

### Primary (HIGH confidence) — direct file inspection this session
- `crates/mlrs-algos/src/traits.rs` — the 9 old traits + exact signatures
- `crates/mlrs-algos/src/typestate.rs` — target surface (4 of 9 traits present; sealed State; Unfit/Fitted; validate_geometry)
- `crates/mlrs-algos/src/manifold/umap.rs` + `cluster/hdbscan.rs` — born-with-convention exemplars (struct/new/builder/Default/Fit/Fitted-gated accessors)
- `crates/mlrs-algos/src/linear/ridge.rs` (pilot A) + `linear/mbsgd_regressor.rs` (pilot B) + `cluster/kmeans.rs` (multi-ctor)
- All 27 `use crate::traits` source files (inventory + trait sets + new() signatures, grep-verified)
- `crates/mlrs-py/src/dispatch.rs` (any_estimator! contract) + `estimators/*.rs` (8 `mlrs_algos::traits` importers, PyO3 fit shape, AnyUmap/AnyHdbscan collapse, 32 pyclass wraps) + `lib.rs` (registration)
- `crates/mlrs-py/python/mlrs/base.py` (MlrsBase machinery) + `mlrs/*.py` (18 existing shim classes) + `python/tests/test_shims.py`/`test_params.py`/`test_estimator_checks.py` (static gate infra)
- `.planning/config.json` (workflow toggles, security_enforcement, nyquist)
- `grep` counts: 85 arg-taking constructor call sites repo-wide; 8 PyO3 trait-importers; 0 in-source `#[cfg(test)]`

### Secondary (project knowledge)
- Project memory: worktree isolation broken → sequential execution; full `cargo test` slow + disk-exhausting → targeted gates; Python wheel untestable in env → static gate + UAT; rocm is the runnable f32 GPU gate, f64-on-rocm SKIPs.
- `Skill(spike-findings-mlrs)` — confirmed NOT central (kernel-authoring landmines; this phase touches no kernels).

## Metadata

**Confidence breakdown:**
- Inventory & classification: HIGH — every file opened, counts grep-verified (and explicitly flagged where CONTEXT diverges: 27 not ~21, 85 sites not 137, sites in mlrs-py not just tests).
- Trait surface & target: HIGH — both traits.rs and typestate.rs read in full; the 5-missing-traits gap is concrete.
- Exemplar/pilot recipes: HIGH — umap.rs/ridge.rs/mbsgd_regressor.rs read directly.
- PyO3 SHIM-02: HIGH — pyclass + registration + collapse enum verified shipped; the `transform`/`fit_transform` (UMAP) and `fit_predict`/`probabilities_`/`outlier_scores_` (HDBSCAN) gaps are VERIFIED-missing (not assumed), so SHIM-02 scope is now concrete.
- Shim target list: MEDIUM — 14 missing classes verified; exact in-scope count (A6) needs planner reconciliation.
- Derive-macro recommendation: HIGH — grounded in the verified heterogeneity + the gate cost.

**Research date:** 2026-06-24
**Valid until:** 2026-07-24 (stable; internal codebase, no external version churn). All file-read open questions from the first pass (KernelRidge/SpectralEmbedding trait state, UMAP/HDBSCAN Py surface) were RESOLVED in-session — see the resolved block in §Assumptions Log. Remaining open items are planner decisions (shim target list, builder-setter type), not unread facts.
