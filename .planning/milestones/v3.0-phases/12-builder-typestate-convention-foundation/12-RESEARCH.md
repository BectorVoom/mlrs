# Phase 12: Builder + Typestate Convention Foundation - Research

**Researched:** 2026-06-23
**Domain:** Rust API design — owned builder + compile-time fit/unfit typestate + sealed marker traits; two new-estimator shells (UMAP/HDBSCAN) demonstrating the convention; PyO3 collapse. NO algorithm, NO device kernels, NO retrofit.
**Confidence:** HIGH (codebase machinery read at path:line; typestate/trybuild idioms cited; sklearn/umap param surfaces cited + already pinned in REQUIREMENTS.md)

## Summary

Phase 12 is a pure Rust API-foundation phase. Every decision is already locked in `12-CONTEXT.md` (D-01…D-13); this research supplies the **HOW**, grounded in the actual codebase, so the planner can write file-accurate tasks. The phase adds: (1) a new `mlrs_algos::typestate` module holding the sealed `State` marker trait + `Unfit`/`Fitted` ZST markers + the four typestate-aware traits (`Fit`/`Predict`/`Transform`/`PartialFit` — same names as the OLD `traits.rs`, new path, **coexisting**); (2) two new-estimator shells — `Umap<F, S=Unfit>` in a NEW `manifold/` module and `Hdbscan<F, S=Unfit>` in the existing `cluster/` — each with the full sklearn param surface, an owned chained-setter builder mirroring `mbsgd_regressor.rs`, `new()` as the single defaults source, a NON-algorithmic real `fit` body (zeros embedding / all-`-1` labels), and fitted-attr accessors gated on `impl T<F, Fitted>`; (3) a `trybuild` compile-fail gate proving `predict`-before-`fit` won't compile; (4) PyO3 shells via the existing `any_estimator!` macro with fitted arms now spelling `<f32, Fitted>`/`<f64, Fitted>` (D-04).

The two load-bearing facts the planner must respect: **the old `crates/mlrs-algos/src/traits.rs` is FROZEN this phase** (Success Criterion 3 — all 30 existing estimators keep compiling on the old `&mut self` surface), so the new traits must live in a separate module and the shells impl ONLY the new surface; and **`any_estimator!` currently emits only the enum skeleton** (`dispatch.rs:91`), so the planner adds the `#[pymethods]` `fit`/accessor bodies per shell exactly as the existing hand-written `linear.rs` wrappers do — the macro change for D-04 is additive (fitted arms already spell `<f32>`/`<f64>`; they become `<f32, Fitted>`/`<f64, Fitted>` once `S` exists, but `S=Unfit` default means the existing 35 call sites' fitted arms ALSO need the explicit `Fitted` spelling only after Phase 16 retrofit — in Phase 12 the macro stays backward-compatible because the old estimators have no `S` param).

**Primary recommendation:** Hand-write both shells (no derive-macro generator — that is deferred to Phase 16), put the new traits + markers in one new `typestate.rs`, re-export from `lib.rs`, add `trybuild` as a dev-dependency, and structure the two shell test files + one `tests/ui/` compile-fail file per AGENTS.md §2 (tests separated from source). The state param `S` is purely additive to the existing `<F: Float + CubeElement + Pod>` bound and does not disturb any existing monomorphization.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01: Per-estimator state type-param.** `Umap<F, S = Unfit>` carrying `PhantomData<S>`. Textbook typestate; chosen over a shared `Unfit<E>/Fitted<E>` wrapper and over two distinct config/fitted types, accepting the signature-touching Phase-16 retrofit cost.
- **D-02: `fit` consumes `self`.** `fit(self, ..) -> Result<T<F, Fitted>, AlgoError>` — takes `self` by value to re-tag the marker. `predict`/`transform`/fitted-attr accessors implemented **only** on `impl T<F, Fitted>`; `T<F, Unfit>` has no such impl → predict-before-fit is a compile error. Chaining works: `est.fit(x)?.predict(x)`.
- **D-03: Marker types + sealed `State`.** `Unfit`/`Fitted` ZST markers + sealed `State` trait bound. Default param `S = Unfit`, so a bare `T<F>` means unfit.
- **D-04: `any_estimator!` fitted arms spell the state explicitly.** Because `S` defaults to `Unfit`, the macro's fitted arms become `F32(T<f32, Fitted>)` / `F64(T<f64, Fitted>)`. Confirm against `crates/mlrs-py/src/dispatch.rs:91`.
- **D-05: Redefine canonical traits as typestate-aware.** `Fit` consumes `self` with associated `type Fitted`; `Predict`/`Transform` bound to the fitted type (no impl on `Unfit` → compile error).
- **D-06: `PartialFit` is consuming and multi-transition.** `partial_fit(self) -> Result<Self::Fitted>` impl'd on BOTH `T<F, Unfit>` (first batch) and `T<F, Fitted>` (subsequent), modeling `Unfit → Fitted → Fitted` so a caller can predict between batches. Designed in now even though UMAP/HDBSCAN don't use it (Phase-16 target for the streaming estimators).
- **D-07: Introduce the new surface by COEXISTENCE, same names, new module.** `traits.rs` stays UNTOUCHED. New traits go in a new module (e.g. `mlrs_algos::typestate`). UMAP/HDBSCAN impl ONLY the new surface. Old traits deleted at the END of Phase 16. Names collide only by path, never at a call site.
- **D-08: `new()` is the canonical defaults source.** `T::new()` constructs the struct literal with sklearn defaults, returns `T<F, Unfit>`, sets `_state: PhantomData`, trusts defaults valid (bypasses `build()` validation), so `T::new() == T::builder().build()?`. Builder's `impl Default` re-derives from `new()` (e.g. `Umap::new().into_builder()`); `T::builder()` returns `Builder::default()`.
- **D-09: Reuse the shipped builder shape.** Owned chained setters (`fn param(mut self, ..) -> Self`), `build<F>(self) -> Result<T<F, Unfit>, BuildError>` (builder itself non-generic), data-independent validation only — the v2 pattern, e.g. `crates/mlrs-algos/src/linear/mbsgd_regressor.rs`.
- **D-10: Full shape + trivial fit.** Real sklearn param surfaces + real fitted-attr accessors (`embedding_` on fitted UMAP, `labels_` on fitted HDBSCAN), plus a NON-algorithmic real fit body (set `n_features_in_`, return zeros embedding / all-noise `-1` labels) → a runtime round-trip test runs.
- **D-11: Compile-fail gate is mandatory.** A `trybuild`-style test proving predict/transform-before-fit fails to compile is the structural proof of BLDR-02.
- **D-12: Module homes.** UMAP → new `manifold/` module under `crates/mlrs-algos/src/`; HDBSCAN → existing `cluster/`. PyO3 shells via existing `any_estimator!` in `crates/mlrs-py/src/estimators/` (new `manifold.rs`; HDBSCAN extends `cluster.rs` or its own file — planner's call).
- **D-13: Typestate collapses behind `any_estimator!`.** Python sees `Unfit/F32/F64` only; runtime `NotFittedError` analog at the boundary; reuse single-site `build_err_to_py`/`algo_err_to_py` mappers.

### Claude's Discretion

- Exact naming/bounds of the `State` sealed trait and marker types; whether the new trait module is `typestate.rs` or another name; whether HDBSCAN's PyO3 shell extends `cluster.rs` or gets its own file — all follow existing structure, planner's call.

### Deferred Ideas (OUT OF SCOPE)

- **Builder/typestate boilerplate generator** (a derive/declarative macro to emit the per-estimator builder + state-param + impl blocks across all 30 estimators) — raised, NOT decided. Belongs with Phase-16 retrofit-sweep planning. Phase 12 hand-writes the two shells to fix the convention first.
- **Old-trait deletion / final single-surface convergence** — END of Phase 16, not here. Phase 12 leaves both surfaces live (D-07).

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BLDR-01 | Construct any estimator via idiomatic Rust builder — `T::builder().param(..)…build() -> Result<T<Unfit>, BuildError>`, owned chained setters, sklearn-equal defaults, typed `thiserror` validation, single-source defaults (`builder().build()? == new()` == sklearn default). | The v2 builder shape is shipped verbatim in `mbsgd_regressor.rs:51-287` (owned setters + `build<F>() -> Result<_, BuildError>`); D-08 `new()` single-source-defaults is the one new wrinkle (see Pattern 2). `BuildError` (`error.rs:384-547`) is the existing typed validation surface. |
| BLDR-02 | fit/unfit as compile-time typestate (`T<Unfit>` → `T<Fitted>`); `predict`/`transform`/fitted-attr accessors only on `T<Fitted>` → predict-before-fit fails to compile. | Sealed-marker typestate (Pattern 1) + `trybuild` compile-fail gate (Pattern 4). The OLD traits return `&mut self`/`Option` runtime `NotFitted` (`traits.rs:53-67`, `error.rs:65-73`); the new surface replaces this with a compile-time guarantee on the new module only. |
| BLDR-04 | PyO3 surface unchanged — Rust typestate collapses behind `any_estimator!` `Unfit/F32/F64`, runtime `NotFittedError` analog at the boundary. | `any_estimator!` (`dispatch.rs:91-115`) is the collapse target; D-04 fitted-arm change. `not_fitted()` / `algo_err_to_py()` / `build_err_to_py()` mappers already exist (`errors.rs:56-88`). |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Sealed `State` trait + `Unfit`/`Fitted` markers | `mlrs-algos` (new `typestate` module) | — | Pure type-level machinery; no device, no Python. Lives next to the estimators that consume it. |
| Typestate-aware `Fit`/`Predict`/`Transform`/`PartialFit` traits | `mlrs-algos` (new `typestate` module) | — | Same crate as the old `traits.rs` (coexistence, D-07); re-exported from `lib.rs`. |
| Owned builder + `new()` defaults + `build()` validation | `mlrs-algos` (per-shell file) | `error.rs` (`BuildError`) | Data-independent hyperparameter validation is a library concern; `thiserror` in libs. |
| Trivial non-algorithmic `fit` body (zeros / `-1`) | `mlrs-algos` (per-shell file) | `mlrs-backend` (`DeviceArray`/`BufferPool` to allocate the output buffer) | The fit must run on-device enough to round-trip a `DeviceArray`, but compute NOTHING (no kernel). |
| Compile-fail proof (predict-before-fit) | `mlrs-algos/tests/ui/` (`trybuild`) | — | Test-tier structural proof; tests separated from source (AGENTS.md §2). |
| PyO3 `#[pyclass]` shell + dtype dispatch + GIL release + f64 guard | `mlrs-py` (`estimators/manifold.rs`, `cluster.rs`) | `any_estimator!` macro, `errors.rs` mappers | The Python boundary; `anyhow`/`PyErr` here, never `thiserror`. |
| Runtime `NotFittedError` analog | `mlrs-py` (`errors.rs` — existing `not_fitted()`) | — | The compile-time guarantee is Rust-side; Python sees a runtime error. |

## Standard Stack

### Core (already in-tree — Phase 12 adds NO new compute dependency)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | `0.10.0` (workspace pin) | `Float`/`CubeElement` element-trait bound the estimators are generic over | Already the project compute substrate; the `S` state param is additive to `<F: Float + CubeElement + Pod>`. `[VERIFIED: Cargo.toml:16]` |
| `bytemuck` | `1` (workspace pin, `features=["derive"]`) | `Pod` bound for host↔device materialize-at-accessor | Already the bound on every estimator. `[VERIFIED: Cargo.toml:31]` |
| `thiserror` | workspace pin | typed `BuildError`/`AlgoError` in the library | Project convention (CLAUDE.md: thiserror in libs). `[VERIFIED: mlrs-algos/Cargo.toml]` |
| `pyo3` | (mlrs-py dep) | `#[pyclass]`/`#[pymethods]` shells | Existing PyO3 surface. `[VERIFIED: estimators/linear.rs:11]` |

### Supporting (NEW dev-dependency this phase)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `trybuild` | `1.0.117` | compile-fail UI test proving `predict` on `T<Unfit>` won't compile (D-11) | dev-dependency of `mlrs-algos` ONLY; runs in `tests/`. `[VERIFIED: npm/crates equivalent — cargo search trybuild = 1.0.117; by dtolnay]` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `trybuild` | hand-rolled `compile_fail` doctest | Doctests can assert a snippet fails to compile, but trybuild gives a stable `.stderr` golden file, glob support, and is the de-facto standard. D-11 says "trybuild-style or equivalent"; trybuild is the lowest-friction equivalent. `[ASSUMED]` doctest alternative viability |
| Per-estimator `S` param (D-01) | shared `Unfit<E>/Fitted<E>` wrapper | Locked AGAINST by D-01 — do not relitigate. |
| sealed `State` via private-module supertrait | non-sealed marker trait | Sealing prevents downstream `impl State for MyType`, keeping the state set closed (the canonical idiom). D-03 asks for sealed. |

**Installation:** Add to `crates/mlrs-algos/Cargo.toml` under `[dev-dependencies]`:
```toml
trybuild = "1.0.117"
```

**Version verification:** `cargo search trybuild` → `trybuild = "1.0.117"` (latest). `[VERIFIED: crates.io via cargo search]`

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `trybuild` | crates.io | ~7 yrs (1.0 since 2019) | 50M+ total | github.com/dtolnay/trybuild | OK | Approved (dev-dependency only) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

`trybuild` is authored by David Tolnay (`dtolnay`), the same author as `thiserror`/`syn`/`quote` already in the workspace; it is the standard Rust compile-fail test harness. Confirmed via Context7 (`/dtolnay/trybuild`, High reputation, benchmark 92) AND `cargo search`. No `build.rs`/postinstall risk in the Rust ecosystem model. `[VERIFIED: crates.io + Context7]`

## Architecture Patterns

### System Architecture Diagram

```text
                       crates/mlrs-algos/src/
                       ┌─────────────────────────────────────────────┐
   (FROZEN this phase) │  traits.rs   ── OLD &mut-self Fit/Predict/   │
                       │              Transform/PartialFit (30 impls) │
                       │                                              │
   (NEW this phase)    │  typestate.rs ── sealed State + Unfit/Fitted │
                       │      │           markers + NEW consuming     │
                       │      │           Fit/Predict/Transform/      │
                       │      │           PartialFit (same names)     │
                       │      ▼                                       │
                       │  manifold/umap.rs   cluster/hdbscan.rs       │
                       │  Umap<F,S=Unfit>     Hdbscan<F,S=Unfit>      │
                       │   ├ new() ─┐          ├ new() ─┐             │
                       │   ├ builder()│        ├ builder()│           │
                       │   │  UmapBuilder      │  HdbscanBuilder      │
                       │   │  .param(..)─►Self  │  .param(..)─►Self    │
                       │   │  .build::<F>()─►Result<T<F,Unfit>,Build…> │
                       │   ├ impl Fit  for T<F,Unfit>  ─fit(self)─┐   │
                       │   │   type Fitted = T<F,Fitted>          ▼   │
                       │   └ impl Predict/Transform for T<F,Fitted>   │
                       │        embedding_()/labels_() accessors HERE │
                       └─────────────────────────────────────────────┘
                              │ lib.rs re-exports typestate::*
                              ▼
   tests/ui/predict_before_fit.rs  (trybuild compile_fail) ── proves
        T::<f32,Unfit>.predict(..)  ►  E0599 no method `predict`
                              │
   ───────────────────────────────────────────────────────────────────
                       crates/mlrs-py/src/estimators/
                       ┌─────────────────────────────────────────────┐
                       │ manifold.rs (UMAP)   cluster.rs (+HDBSCAN)   │
                       │  any_estimator!{ any: AnyUmap, algo: …,      │
                       │     unfit:{ n_neighbors, min_dist, … } }     │
                       │   enum AnyUmap {                             │
                       │     Unfit{ params },                        │
                       │     F32(Umap<f32, Fitted>),  ◄── D-04        │
                       │     F64(Umap<f64, Fitted>) }                │
                       │   #[pymethods] fit: py.detach → lock_pool →  │
                       │     float_dtype → guard_f64 (F64) →          │
                       │     Umap::<F>::new()…fit(self)→ store Fitted │
                       │   accessor before fit → not_fitted()→PyValue │
                       └─────────────────────────────────────────────┘
```

### Recommended Project Structure

```
crates/mlrs-algos/src/
├── typestate.rs              # NEW: sealed State + Unfit/Fitted + 4 new traits
├── lib.rs                    # EDIT: `pub mod typestate;` + re-export typestate::*
├── manifold/                 # NEW module (D-12)
│   ├── mod.rs                # `pub mod umap; pub use umap::Umap;`
│   └── umap.rs               # Umap<F,S> + UmapBuilder + impls
└── cluster/
    ├── mod.rs                # EDIT: add `pub mod hdbscan; pub use hdbscan::Hdbscan;`
    └── hdbscan.rs            # NEW: Hdbscan<F,S> + HdbscanBuilder + impls

crates/mlrs-algos/tests/
├── umap_test.rs              # build==new defaults eq, round-trip on trivial fit, PoolStats gate
├── hdbscan_test.rs           # same
├── typestate_test.rs         # (optional) runtime assertions the markers compose
├── compile_fail.rs           # `#[test] fn ui() { trybuild::TestCases::new().compile_fail("tests/ui/*.rs"); }`
└── ui/
    ├── predict_before_fit.rs        # Umap::<f32>::new().predict(..)  → must NOT compile
    ├── predict_before_fit.stderr    # golden expected error
    ├── transform_before_fit.rs      # same for transform
    └── transform_before_fit.stderr

crates/mlrs-py/src/estimators/
├── manifold.rs               # NEW: PyUmap via any_estimator!
├── cluster.rs                # EDIT: add PyHdbscan via any_estimator! (or new hdbscan.rs)
└── mod.rs                    # EDIT: `pub mod manifold;`
```

### Pattern 1: Sealed-trait typestate with ZST markers (D-01/D-02/D-03/D-05)

**What:** Encode fit/unfit at the type level so methods only exist on the right state.
**When to use:** The whole new estimator surface.
**Example:**
```rust
// crates/mlrs-algos/src/typestate.rs
use std::marker::PhantomData;
use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use crate::error::AlgoError;

mod sealed { pub trait Sealed {} }

/// Sealed marker: the two estimator lifecycle states (D-03). Downstream crates
/// cannot add states.
pub trait State: sealed::Sealed {}

/// Zero-sized marker — a freshly-built, not-yet-fitted estimator.
pub struct Unfit;
/// Zero-sized marker — a fitted estimator; only here do predict/transform exist.
pub struct Fitted;

impl sealed::Sealed for Unfit {}
impl sealed::Sealed for Fitted {}
impl State for Unfit {}
impl State for Fitted {}

/// Consuming fit (D-05): takes `self` by value to RE-TAG the state marker.
pub trait Fit<F>
where F: Float + CubeElement + Pod
{
    /// The fitted monomorphization (`Self` with `S = Fitted`).
    type Fitted;
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError>;
}

/// Predict — bound to the fitted type only (no impl on Unfit → compile error).
pub trait Predict<F>
where F: Float + CubeElement + Pod
{
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;
}

/// Transform — same shape; UMAP implements this (project new data into embedding).
pub trait Transform<F>
where F: Float + CubeElement + Pod
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;
}

/// Multi-transition consuming partial_fit (D-06): impl'd on BOTH Unfit and Fitted.
/// UMAP/HDBSCAN do NOT use it; defined now so Phase-16 IncrementalPCA has a target.
pub trait PartialFit<F>
where F: Float + CubeElement + Pod
{
    type Fitted;
    fn partial_fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError>;
}
```
*Sources:* [CITED: cliffle.com/blog/rust-typestate/] (the canonical Rust typestate write-up — ZST markers + sealed trait + state-gated methods), [CITED: doc.rust-lang.org/std/marker/struct.PhantomData.html] (PhantomData semantics + variance). The `State`-bound and sealed module mirror these idioms.

**Estimator struct + state-gated impls (D-02 — the locked concrete shape from CONTEXT §Specific Ideas):**
```rust
// crates/mlrs-algos/src/manifold/umap.rs
pub struct Umap<F, S = Unfit> {
    // hyperparameters (validated copies)
    n_neighbors: usize, n_components: usize, min_dist: f64, spread: f64,
    metric: Metric, /* … */
    // fitted fields — None until fit (Option even on Fitted is fine; the
    // compile-time guarantee is that accessors only EXIST on Fitted)
    embedding_: Option<DeviceArray<ActiveRuntime, F>>,
    n_features_in_: usize,
    _state: PhantomData<S>,
}

impl<F: Float + CubeElement + Pod> Umap<F, Unfit> {
    pub fn new() -> Self { /* sklearn defaults, _state: PhantomData */ }
    pub fn builder() -> UmapBuilder { UmapBuilder::default() }
}

impl<F: Float + CubeElement + Pod> Fit<F> for Umap<F, Unfit> {
    type Fitted = Umap<F, Fitted>;
    fn fit(self, pool, x, _y, shape) -> Result<Umap<F, Fitted>, AlgoError> {
        // NON-ALGORITHMIC trivial fit (D-10): set n_features_in_, allocate a
        // zeros embedding (n × n_components). NO kernel, NO compute.
        let (n, n_features) = shape;
        let zeros = vec![F::from_int(0); n * self.n_components];
        let embedding = DeviceArray::from_host(pool, &zeros);
        Ok(Umap { /* move every hyperparam field */, embedding_: Some(embedding),
                  n_features_in_: n_features, _state: PhantomData })
    }
}

impl<F: Float + CubeElement + Pod> Umap<F, Fitted> {
    // Fitted-attr accessor — exists ONLY here (D-02)
    pub fn embedding(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> { /* … */ }
    pub fn n_features_in(&self) -> usize { self.n_features_in_ }
}
impl<F: Float + CubeElement + Pod> Transform<F> for Umap<F, Fitted> { /* … */ }
```
*Sources:* shape locked in `12-CONTEXT.md` §Specific Ideas (lines 99-105); fitted-field/`None`-until-fit precedent `mbsgd_regressor.rs:36-43`; `DeviceArray::from_host` usage `gaussian_nb_test.rs:140`.

### Pattern 2: `new()` as single defaults source + builder `Default` round-trip (D-08)

**What:** `new()` writes the struct literal directly (trusting defaults valid); the builder's `Default` re-derives from `new()` so `new() == builder().build()?` holds by construction.
**When to use:** Both shells.
**Example:**
```rust
impl<F: Float + CubeElement + Pod> Umap<F, Unfit> {
    pub fn new() -> Self {
        Self { n_neighbors: 15, n_components: 2, min_dist: 0.1, spread: 1.0,
               metric: Metric::Euclidean, /* … sklearn defaults … */,
               embedding_: None, n_features_in_: 0, _state: PhantomData }
    }
    pub fn into_builder(self) -> UmapBuilder { /* copy hyperparams into builder */ }
}
impl Default for UmapBuilder {
    fn default() -> Self {
        // re-derive from new() so the two defaults sources can never drift.
        // Note: new() is generic over F but the builder is NOT (D-09); pick a
        // concrete F (e.g. f64) purely to read the hyperparameter defaults, which
        // are F-independent scalars/enums. See Open Question 1.
        Umap::<f64, Unfit>::new().into_builder()
    }
}
```
**Caveat (Open Q1):** `new()` is generic over `F` but the builder is non-generic (D-09). The defaults are F-independent scalars, so `Default for UmapBuilder` can call `Umap::<f64,_>::new().into_builder()` and discard the `F`. The equality test `new()==builder().build()?` (Success Criterion 1) compares the *validated hyperparameter struct*, not the full estimator — derive `PartialEq` on the hyperparameter subset, or compare field-by-field in the test. The fitted `Option` fields are `None` in both, so they compare equal.
*Sources:* D-08 (`12-CONTEXT.md:38`); builder shape `mbsgd_regressor.rs:51,107-126,213`.

### Pattern 3: Owned chained-setter builder + `build<F>()` (D-09 — mirror `mbsgd_regressor.rs`)

**What:** Non-generic builder struct, `fn param(mut self, ..) -> Self` setters, generic-at-build `build<F>(self) -> Result<T<F, Unfit>, BuildError>` doing data-INDEPENDENT validation only.
**When to use:** Both shells — copy the structure verbatim from `mbsgd_regressor.rs`.
**Example:** see `mbsgd_regressor.rs:128-287` — `MBSGDRegressorBuilder` is the exact template: `#[derive(Debug, Clone, Copy)]`, `impl Default` (the defaults source today; D-08 changes this to re-derive from `new()`), one `fn name(mut self, v) -> Self` per field, then `pub fn build<F>(self) -> Result<MBSGDRegressor<F>, BuildError> where F: Float + CubeElement + Pod` that validates and returns the struct. UMAP's documented validation is `min_dist ≤ spread` (UMAP-01 / `REQUIREMENTS.md:23`) → a NEW `BuildError` variant (see Pitfall 3). HDBSCAN's is `min_cluster_size ≥ 2`, `min_samples ≥ 1` → new variants or reuse `InvalidMinSamples` (note: that one lives on `AlgoError`, not `BuildError` — see Pitfall 3).
*Sources:* `mbsgd_regressor.rs:89-287`; `BuildError` `error.rs:384-547`.

### Pattern 4: trybuild compile-fail gate (D-11 — BLDR-02 structural proof)

**What:** A `tests/ui/*.rs` file that calls `predict` on `T<Unfit>`; trybuild asserts it does NOT compile and matches a golden `.stderr`.
**When to use:** Once per "before-fit" method (predict, transform).
**Example:**
```rust
// crates/mlrs-algos/tests/compile_fail.rs
#[test]
fn ui_predict_before_fit() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
```
```rust
// crates/mlrs-algos/tests/ui/predict_before_fit.rs
use mlrs_algos::manifold::umap::Umap;
use mlrs_algos::typestate::Transform; // brings the trait into scope
fn main() {
    let est = Umap::<f32>::new();            // T<f32, Unfit>
    let _ = est.transform(/* … */);          // ERROR: no method `transform` on Unfit
}
```
The golden `.stderr` records the `E0599` "no method named `transform` found for struct `Umap<f32, Unfit>`" message. Generate it the first time with `TRYBUILD=overwrite cargo test --features cpu ui_predict_before_fit` then commit the `.stderr`.

**Caveats for THIS project:**
- **Feature gates.** trybuild compiles the `ui/*.rs` files as standalone crates depending on `mlrs-algos`. They need a backend feature so `ActiveRuntime` resolves. Pass it through the test invocation (`cargo test --features cpu`); the ui file itself does not enable features. Because the failure (`no method predict on Unfit`) is in type resolution, it is **backend-independent** — the same `.stderr` holds under cpu/rocm. Prefer a ui example whose body does NOT actually need a live `BufferPool`/`DeviceArray` argument to reach the error: the method-not-found error fires at name resolution before argument type-checking, so `est.transform()` with no/placeholder args still produces the diagnostic. Keep the ui file MINIMAL to keep `.stderr` stable across rustc versions. `[CITED: dtolnay/trybuild README — compile_fail + .stderr]`
- **`.stderr` brittleness across rustc.** Golden stderr can drift between toolchain versions. Mitigate by keeping the ui file tiny (one method call) and by documenting the pinned toolchain. This is a known trybuild trade-off. `[ASSUMED]` exact drift behavior on this repo's toolchain — verify on first run.
- **Tests separated from source (AGENTS.md §2).** trybuild's model IS tests-in-`tests/`; the `ui/*.rs` files are fixtures, not in-source `#[cfg(test)] mod`. Fully compliant.
*Sources:* [CITED: github.com/dtolnay/trybuild README]; Context7 `/dtolnay/trybuild`.

### Pattern 5: PyO3 collapse via `any_estimator!` (D-04/D-13/BLDR-04)

**What:** Reuse the existing macro skeleton (`dispatch.rs:91`) for the enum, then hand-write the `#[pymethods]` exactly like `linear.rs`.
**When to use:** Both PyO3 shells.
**Critical facts the planner must know:**
- `any_estimator!` TODAY emits ONLY the three-state enum (`Unfit { fields } | F32(Estimator<f32>) | F64(Estimator<f64>)`) — the `#[pymethods]` are hand-written per estimator (`dispatch.rs:108-113` NOTE; `linear.rs:32-108` is the worked example). So the UMAP/HDBSCAN PyO3 shells follow `PyLinearRegression` line-for-line: `#[pyclass]` holding `inner: AnyUmap`, `#[new]` storing `Unfit { params }`, `fn fit` doing `py.detach(|| { lock_pool(); float_dtype; guard_f64() on F64; Umap::<F>::new()…fit(self); store F32/F64 arm })`, accessors matching the fitted arm.
- **D-04 macro change:** the macro's fitted arms are `F32($algo<f32>)` / `F64($algo<f64>)` today (`dispatch.rs:104-106`). For the typestate estimators they must become `F32($algo<f32, Fitted>)` / `F64($algo<f64, Fitted>)`. Because the OLD 30 estimators have NO `S` param, a naive edit to the shared macro would break their 35 existing call sites. **Recommendation:** add an OPTIONAL macro arm (e.g. a `fitted_marker:` parameter, or a second macro `any_estimator_typestate!`) so the new shells get the `Fitted` spelling while the existing call sites stay on the no-marker arm. The existing macro is `#[macro_export]`; a new variant is additive and file-disjoint. (Planner's call which form; either keeps Success Criterion 3 green.)
- **fit consumes self (D-02) maps cleanly:** in the PyO3 `fit`, the current pattern is `let mut est = Estimator::<F>::new(..); est.fit(&mut pool, …)?; Ok(Any::F32(est))` (`linear.rs:92-94`). The new consuming form is `let fitted = Umap::<F>::new().fit(&mut pool, …)?; Ok(AnyUmap::F32(fitted))` — strictly simpler (no `mut`, no `&mut self`).
- **Runtime NotFittedError (D-13):** an accessor called on the `Unfit` arm returns `not_fitted("umap", "embedding_")` → `PyValueError` the shim re-raises as `sklearn.exceptions.NotFittedError`. `not_fitted()` already exists (`errors.rs:84-88`). Reuse `algo_err_to_py` / `build_err_to_py` (`errors.rs:56-73`) — do NOT add new mappers.
*Sources:* `dispatch.rs:62-115`, `linear.rs:32-108`, `errors.rs:56-88`, `estimators/mod.rs:30-39`.

### Anti-Patterns to Avoid

- **Editing `traits.rs`.** It is FROZEN this phase (D-07 / Success Criterion 3). The new traits go in a NEW module. Touching `traits.rs` risks the 30-estimator green suite.
- **A trivial fit that reads as a placeholder algorithm.** The fit body must set `n_features_in_` and produce a *real, addressable* `DeviceArray` (zeros embedding / all-`-1` labels) so a round-trip test runs — but it must compute NOTHING (no KNN, no kernel, no `todo!()`). A `todo!()`/`unimplemented!()` body would fail the round-trip test and read as "algorithm deferred" rather than "convention demonstrated." Document the body explicitly as "non-algorithmic shell — real UMAP lands in Phase 14."
- **Generic builder.** D-09: the builder struct is NON-generic; `F` appears only on `build<F>()`. Do not parameterize `UmapBuilder<F>`.
- **Sharing the `S` marker across estimators via a wrapper.** D-01 locked the per-estimator param; don't introduce `Unfit<E>`.
- **Re-deriving defaults in two places.** D-08: `new()` is the ONLY source; the builder's `Default` calls `new().into_builder()`. Hand-writing the builder `Default` with literal values would re-introduce drift.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Proving predict-before-fit won't compile | a custom `Command::new("rustc")` harness or string-matching script | `trybuild` (dev-dep) | Golden `.stderr`, glob, stable, standard. |
| Typed construction errors | a new error enum | existing `BuildError` (`error.rs:384`) + add variants | Single-site `build_err_to_py` already maps the whole enum (D-13). |
| Runtime not-fitted at Python boundary | a new exception type | existing `not_fitted()` / `algo_err_to_py()` (`errors.rs:56,84`) | D-13 says reuse the single-site mappers. |
| Dtype dispatch enum | hand-written `enum AnyUmap` | `any_estimator!` (`dispatch.rs:91`) | The macro is the BLDR-04 collapse target; emit the enum, hand-write `#[pymethods]`. |
| Sealed marker trait | re-inventing variance/marker plumbing | `PhantomData<S>` + private `sealed::Sealed` supertrait | The textbook idiom (Pattern 1). |

**Key insight:** This phase is almost entirely *assembly* of shipped machinery (`BuildError`, `any_estimator!`, the error mappers, the `mbsgd_regressor` builder shape) plus ONE genuinely new idiom (the sealed typestate) and ONE new dev-dep (`trybuild`). Resist building anything custom for construction/validation/dispatch — it all exists.

## Runtime State Inventory

> Phase 12 is greenfield (NEW module + NEW shells), not a rename/refactor. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a renamed string. **None — verified: this phase ADDS files (`typestate.rs`, `manifold/`, `cluster/hdbscan.rs`, PyO3 shells, test files) and ADDITIVELY edits `lib.rs`/`cluster/mod.rs`/`estimators/mod.rs`/`Cargo.toml`; it renames nothing and migrates no data.** The one existing-file behavioral edit is the OPTIONAL `any_estimator!` macro arm (additive — existing call sites unchanged). Section otherwise omitted as non-applicable.

## Common Pitfalls

### Pitfall 1: Touching the frozen `traits.rs` and breaking the 30-estimator suite
**What goes wrong:** Adding the consuming `Fit`/`Predict` to `traits.rs` (or changing the old `Fit` signature) breaks every existing estimator's `impl Fit`/`impl Predict` and every PyO3 call site.
**Why it happens:** The new traits share NAMES with the old ones (D-07); it's tempting to "just update them."
**How to avoid:** New module `typestate.rs`; re-export under a path. At call sites the two never collide because each consumer imports exactly one (`use mlrs_algos::traits::Fit` vs `use mlrs_algos::typestate::Fit`). The shells import ONLY `typestate::*`.
**Warning signs:** Any diff to `traits.rs`; any existing test file failing to compile.

### Pitfall 2: `S = Unfit` default vs the `any_estimator!` macro spelling
**What goes wrong:** The macro's existing fitted arms `F32($algo<f32>)` would, for a typestate estimator, mean `F32(Umap<f32, Unfit>)` (default `S`) — the WRONG arm (it should be `Fitted`). D-04 exists precisely for this.
**Why it happens:** The `S = Unfit` default silently fills in `Unfit` when the marker is omitted.
**How to avoid:** New shells use a macro arm that spells `<f32, Fitted>` / `<f64, Fitted>` explicitly (Pattern 5). Do NOT retro-edit the shared macro arm used by the 30 existing (no-`S`) estimators — make the change additive.
**Warning signs:** A PyO3 `fit` that compiles but stores an `Unfit`-typed value in the `F32` arm; type-mismatch errors mentioning `Unfit` where `Fitted` is expected.

### Pitfall 3: `BuildError` vs `AlgoError` for the new validations (the D-08 split)
**What goes wrong:** Putting `min_dist ≤ spread` (UMAP) or `min_cluster_size ≥ 2` (HDBSCAN) validation in `fit` instead of `build`, or reusing an `AlgoError` variant for a build-time check.
**Why it happens:** Some existing validations (e.g. `AlgoError::InvalidMinSamples`, `error.rs:126`) live on `AlgoError` because the OLD estimators validate at `fit`. The NEW convention is data-INDEPENDENT validation at `build()` → `BuildError` (D-08/D-09).
**How to avoid:** Add NEW `BuildError` variants for the shell hyperparameters (`InvalidMinDistSpread`, `InvalidMinClusterSize`, etc.) following the `error.rs:384-547` pattern (`#[error("…")]` + `estimator: &'static str` + the offending value). The trivial fit body still does the data-DEPENDENT geometry guard (`n_samples==0 || x.len()!=n*p`) → `AlgoError::Prim(PrimError::ShapeMismatch)` exactly like `mbsgd_regressor.rs:305`.
**Warning signs:** A hyperparameter error surfacing only after data is uploaded; a build-time check returning `AlgoError`.

### Pitfall 4: A `Default` impl re-deriving defaults (drift)
**What goes wrong:** Writing `impl Default for UmapBuilder` with literal `15`/`0.1`/… duplicates the defaults that `new()` already holds → the two can silently diverge, breaking Success Criterion 1.
**Why it happens:** The existing `mbsgd_regressor.rs:107-126` `Default` DOES hold the literals (that's the v2 shape, pre-D-08). D-08 changes this.
**How to avoid:** `impl Default for UmapBuilder { fn default() -> Self { Umap::<f64,_>::new().into_builder() } }` — one source. Add a test asserting `Umap::<F>::new()`'s hyperparameter struct equals `Umap::builder().build::<F>()?`'s (Success Criterion 1; derive `PartialEq` on the hyperparameter subset).
**Warning signs:** Two lists of default literals in the same file.

### Pitfall 5: trybuild `.stderr` drift / feature resolution
**What goes wrong:** The golden `.stderr` mismatches under a different rustc, or the ui crate fails to resolve `ActiveRuntime` because no backend feature is on.
**Why it happens:** stderr is toolchain-sensitive; the ui file is compiled as a dependent crate.
**How to avoid:** Keep the ui file to ONE method call (minimal stderr surface); run the trybuild test under `--features cpu` (the primary gate); regenerate stderr with `TRYBUILD=overwrite` on the pinned toolchain and commit it. Document the toolchain in the test file header.
**Warning signs:** The ui test failing on CI but passing locally (toolchain mismatch); `E0432`/`E0463` unresolved-crate errors in the ui output (missing feature).

### Pitfall 6: The trivial fit allocating but never round-tripping (false "compiles" gate)
**What goes wrong:** A fit that returns `Umap<F, Fitted>` with `embedding_: None` compiles but the round-trip test (`fit(x)?.embedding()`) panics/errors — undermining D-10's "runtime end-to-end" requirement.
**Why it happens:** Forgetting to actually allocate the zeros buffer.
**How to avoid:** The fit MUST `DeviceArray::from_host(pool, &vec![F::from_int(0); n*n_components])` (UMAP) / all-`-1` `i32` labels (HDBSCAN) and store `Some(..)`. Test: `fit` → accessor returns a buffer of the right shape full of the sentinel value.
**Warning signs:** `embedding_` left `None` after fit; round-trip test returning `NotFitted`.

## Code Examples

See Patterns 1–5 above — all examples are grounded in `mbsgd_regressor.rs`, `linear.rs`, `dispatch.rs`, `errors.rs`, and the cited typestate/trybuild sources. The single most-copied template is `crates/mlrs-algos/src/linear/mbsgd_regressor.rs:89-287` (builder) and `crates/mlrs-py/src/estimators/linear.rs:32-108` (PyO3 shell).

## State of the Art

| Old Approach (this codebase, v1/v2) | New Approach (Phase 12 convention) | When Changed | Impact |
|--------------------------------------|-------------------------------------|--------------|--------|
| `fit(&mut self) -> Result<&mut Self>` + `Option<..>` fitted fields + runtime `AlgoError::NotFitted` (`traits.rs:53-67`, `error.rs:65`) | consuming `fit(self) -> Result<T<F, Fitted>>`; accessors only on `T<F, Fitted>`; predict-before-fit is a COMPILE error | Phase 12 (new module + new estimators ONLY; old estimators migrate in Phase 16) | Compile-time safety on the new surface; both surfaces coexist until end of Phase 16. |
| `new(args)` / `with_opts()` construction (v1) → builder `Default` holding literal defaults (v2, `mbsgd_regressor.rs:107`) | `new()` single-source defaults; builder `Default` re-derives via `new().into_builder()` (D-08) | Phase 12 | One defaults source; `new()==builder().build()?` by construction. |

**Deprecated/outdated within this phase's scope:**
- Nothing is deleted in Phase 12. The old `traits.rs` surface is deprecated-in-intent (slated for deletion at end of Phase 16) but stays fully live now.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A doctest could substitute for trybuild as the compile-fail mechanism. | Stack / Alternatives | LOW — trybuild is recommended regardless; D-11 permits "or equivalent." |
| A2 | trybuild `.stderr` drift on this repo's pinned toolchain is manageable by keeping the ui file minimal. | Pitfall 5 | LOW–MED — verify on first `TRYBUILD=overwrite` run; if stderr proves unstable, switch to trybuild's `compile_fail` without exact stderr matching (it still asserts non-compilation). |
| A3 | The umap-learn full constructor has ~36 kwargs; the Phase-12 shell mirrors ONLY the v3-oracle subset named in UMAP-01 (`n_neighbors`, `n_components`, `metric`, `min_dist`, `spread`, `n_epochs`, `init`, `random_state`, `learning_rate`, `set_op_mix_ratio`, `local_connectivity`, `repulsion_strength`, `negative_sample_rate`, `a`, `b`), not every kwarg. | Standard Stack / sklearn surfaces | LOW — REQUIREMENTS.md UMAP-01 is the authoritative pinned list; the full readthedocs list is informational. Confirm with planner whether `densmap`/`target_*`/`output_metric` are in-scope (they are explicitly OUT per the v3 Out-of-Scope table). |
| A4 | HDBSCAN shell mirrors the HDBS-01 subset (`min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, `cluster_selection_method`, `metric`, `alpha`, `max_cluster_size`) — NOT `algorithm`/`leaf_size`/`n_jobs`/`copy` (host/perf-only, no v3 oracle value). | Standard Stack / sklearn surfaces | LOW — REQUIREMENTS.md HDBS-01 is authoritative. |
| A5 | `Default for UmapBuilder` can pin a concrete `F` (e.g. `f64`) to read F-independent scalar defaults from `new()`. | Pattern 2 / Open Q1 | LOW — defaults are scalars/enums independent of `F`; if any default ever became F-typed, the builder would need a different source. None do today. |
| A6 | Only `IncrementalPCA` impls `PartialFit` today (`MBSGDClassifier`/`Regressor` impl only `Fit`). The D-06 multi-transition design targets the future Phase-16 retrofit of those streaming estimators, but the only CURRENT `PartialFit` consumer is `IncrementalPCA`. | Pattern 1 (PartialFit) | LOW — verified by grep (`impl.*PartialFit` → `incremental_pca.rs` only). The new `PartialFit` trait is defined but UNUSED in Phase 12 (no shell uses it); this is intentional per D-06. |

## Open Questions (RESOLVED)

1. **Builder `Default` and the generic `new()`.**
   - **RESOLVED in 12-02:** builder `Default` re-derives via `Umap::<f64,_>::new().into_builder()` (pins `f64` for the const-defaults path).
   - What we know: `new()` is `impl Umap<F, Unfit>` (generic); the builder is non-generic (D-09); defaults are F-independent scalars.
   - What's unclear: whether to express `Default for UmapBuilder` as `Umap::<f64,_>::new().into_builder()` (pin an arbitrary `F`) or to factor the default scalars into a free `const`/fn that both `new()` and the builder read.
   - Recommendation: pin `f64` in `Default` (simplest, one source); the planner may instead extract a `fn defaults() -> UmapHyperparams` free function if it reads cleaner. Either satisfies D-08.

2. **One shared macro variant vs a second macro for the typestate fitted-arm spelling.**
   - **RESOLVED in 12-04:** add a SECOND `any_estimator_typestate!` macro (additive; the shared `any_estimator!` arm is NOT edited).
   - What we know: the existing `any_estimator!` is shared by 35 call sites with no `S` param; the new shells need `<f32, Fitted>`/`<f64, Fitted>`.
   - What's unclear: add an optional `fitted_marker:` token to the existing macro, or ship `any_estimator_typestate!` alongside.
   - Recommendation: a second additive macro (`any_estimator_typestate!`) is the lowest-risk to Success Criterion 3 (the existing macro is byte-for-byte untouched). Planner's call.

3. **HDBSCAN PyO3 shell home (D-12 explicitly leaves this open).**
   - **RESOLVED in 12-04:** extend `crates/mlrs-py/src/estimators/cluster.rs` (no new file).
   - Recommendation: extend `estimators/cluster.rs` (it already hosts the cluster family) unless it grows unwieldy; a new `estimators/hdbscan.rs` is equally valid. Follow whichever keeps `estimators/mod.rs` edits minimal.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain + cargo | all | ✓ | (workspace) | — |
| `--features cpu` backend | tests (round-trip, PoolStats, trybuild) | ✓ | cpu MLIR | — |
| `--features rocm` backend | f32 round-trip gate | ✓ (per MEMORY) | ROCm 7.1.1 / gfx1100 | f64 SKIPS-with-log on rocm |
| `trybuild` crate | compile-fail gate (D-11) | ✓ (crates.io) | 1.0.117 | doctest `compile_fail` (A1) |
| maturin + pyarrow (live PyO3 pytest) | NOT this phase | ✗ | — | Rust-side gates compensate; live FFI deferred (per MEMORY + SHIM-03). |

**Missing dependencies with no fallback:** none — Phase 12 is fully buildable/testable under `--features cpu` (and f32 under rocm).
**Missing dependencies with fallback:** live PyO3 pytest (maturin/pyarrow absent) — but Phase 12's PyO3 work is the Rust-side `#[pyclass]` shell + cross-crate smoke test (`unfit_default()` pattern, `linear.rs:48`), which compiles and runs without a Python interpreter. The runtime `NotFittedError` analog is verified by a Rust unit test on the `Unfit`-arm accessor path, not a live pytest.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (integration tests in `crates/*/tests/`) + `trybuild` for compile-fail |
| Config file | none — cargo convention; `tests/` dir per AGENTS.md §2 |
| Quick run command | `cargo test -p mlrs-algos --features cpu umap_test` (per-shell, targeted) |
| Full suite command | `cargo test -p mlrs-algos -p mlrs-py --features cpu` (per MEMORY: full algos suite ~6min — background the full run, gate targeted) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BLDR-01 | `Umap::builder().build::<F>()? == Umap::<F>::new()` (defaults equality, single source) | unit | `cargo test -p mlrs-algos --features cpu umap::defaults_equal` | ❌ Wave 0 (`tests/umap_test.rs`) |
| BLDR-01 | invalid hyperparam (`min_dist > spread`) → `BuildError::InvalidMinDistSpread` | unit | `cargo test -p mlrs-algos --features cpu umap::build_rejects_bad_min_dist` | ❌ Wave 0 |
| BLDR-01 | HDBSCAN `min_cluster_size < 2` → `BuildError` | unit | `cargo test -p mlrs-algos --features cpu hdbscan::build_rejects` | ❌ Wave 0 (`tests/hdbscan_test.rs`) |
| BLDR-02 | `predict`/`transform` on `T<Unfit>` fails to compile | compile-fail | `cargo test -p mlrs-algos --features cpu ui_predict_before_fit` | ❌ Wave 0 (`tests/compile_fail.rs` + `tests/ui/*.rs`+`.stderr`) |
| BLDR-02 | trivial fit round-trip: `fit(x)?` then `embedding()` returns `(n, n_components)` zeros / `labels_` all `-1` | unit (runtime) | `cargo test -p mlrs-algos --features cpu umap::fit_roundtrip` | ❌ Wave 0 |
| BLDR-02 | PoolStats no-leak across re-construct/fit (memory gate — see below) | unit | `cargo test -p mlrs-algos --features cpu umap::fit_no_leak` | ❌ Wave 0 |
| BLDR-04 | every EXISTING `any_estimator!` call site still compiles + its suite green (Success Criterion 3) | regression | `cargo test -p mlrs-py --features cpu` (existing suites) | ✅ exists (35 call sites) |
| BLDR-04 | PyO3 UMAP/HDBSCAN shell instantiates in `Unfit` arm without an interpreter (cross-crate smoke) | unit | `cargo test -p mlrs-py --features cpu manifold::unfit_default` | ❌ Wave 0 (`mlrs-py/tests/`) |
| BLDR-04 | accessor on `Unfit` arm → `not_fitted()` `PyValueError` (runtime NotFitted analog) | unit | `cargo test -p mlrs-py --features cpu manifold::not_fitted_before_fit` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** the targeted per-shell test (`cargo test -p mlrs-algos --features cpu <shell>_test`) + the trybuild ui test.
- **Per wave merge:** `cargo test -p mlrs-algos --features cpu` (algos) + `cargo test -p mlrs-py --features cpu` (py) — confirms Success Criterion 3 (existing suites green) on every merge.
- **Phase gate:** full `cargo test --features cpu` green (background per MEMORY — full algos suite ~6min; reduce_test/svd_test are the slow ones, unrelated to this phase), plus f32 round-trip under `--features rocm` for both shells (f64-on-rocm SKIPS-with-log).

### Wave 0 Gaps
- [ ] `crates/mlrs-algos/tests/umap_test.rs` — covers BLDR-01/02 (defaults equality, build rejects, fit round-trip, no-leak)
- [ ] `crates/mlrs-algos/tests/hdbscan_test.rs` — same for HDBSCAN
- [ ] `crates/mlrs-algos/tests/compile_fail.rs` + `tests/ui/{predict,transform}_before_fit.rs` + `.stderr` — BLDR-02 structural proof
- [ ] `crates/mlrs-py/tests/manifold_test.rs` (or extend an existing py test) — `unfit_default` smoke + `not_fitted` runtime analog
- [ ] `trybuild = "1.0.117"` added to `crates/mlrs-algos/[dev-dependencies]`
- [ ] (no new framework install — cargo + trybuild only)

## Security Domain

> `security_enforcement: true`, `security_asvs_level: 1` (config.json:42-43). This is a pure Rust API phase with NO new untrusted input surface (no new device kernel, no new Python ingress beyond the standard `any_estimator!` path that already guards dtype + f64). The relevant control is input validation of hyperparameters at the trust boundary.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a (library, no auth) |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a |
| V5 Input Validation | yes | Hyperparameter validation at `build()` → typed `BuildError` (data-independent) + geometry guard at `fit()` → `AlgoError::Prim(ShapeMismatch)` (data-dependent). This is the SAME validate-before-launch contract every shipped estimator follows (`mbsgd_regressor.rs:217-265,303-312`). The trivial fit allocates a fixed-size buffer from validated `shape` — no untrusted size reaches a kernel. |
| V6 Cryptography | no | n/a (no crypto, no RNG seed security concern — `seed`/`random_state` are determinism knobs, not secrets) |

### Known Threat Patterns for the Rust estimator surface

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-range hyperparameter (e.g. `n_components` huge, `min_dist` NaN) reaching a buffer alloc | Tampering / DoS | Validate at `build()` BEFORE data; reject with typed `BuildError` (the D-08 split). For the trivial fit, the only allocation is `n * n_components` zeros — both bounded by validated/`shape` values; add a finite/`>0` check on `min_dist`/`spread` (UMAP) and `min_cluster_size>=2` (HDBSCAN). |
| Malformed geometry (`x.len() != n*p`) at fit | Tampering | Geometry guard at fit → `PrimError::ShapeMismatch` (mirror `mbsgd_regressor.rs:305`). |
| f64 on an f64-incapable backend (rocm) | DoS (panic) | `guard_f64()?` BEFORE upload on the F64 PyO3 arm (existing contract, `linear.rs:97`); algos-side f64 round-trip SKIPS-with-log on rocm. |
| Panic in `fit` poisoning the global pool mutex | DoS | Use `crate::lock_pool()` (poison-recovering) in the PyO3 shell, NOT `global_pool().lock().expect()` (`dispatch.rs:30-33`). The trivial fit cannot panic (no compute), but follow the sanctioned path regardless. |

**Memory gate (milestone requirement — per-phase build-failing PoolStats gate):** For this no-device-compute phase the meaningful memory gate is a **no-leak gate on the trivial fit's one allocation**: construct+fit (or re-fit) the shell several times at the same shape and assert `pool.stats().live_bytes` does not climb — exactly the `refit_releases_buffers` pattern in `gaussian_nb_test.rs:294-340`. This proves the zeros-embedding / `-1`-labels `DeviceArray` is released into the free-list across re-fit and that the `Option<DeviceArray>` fitted field doesn't leak when overwritten. This is the established no-device-heavy gate shape (the NB family — reductions-only — uses the same `live_bytes`-monotone assertion). Include one such test per shell; it is build-failing in the sense that a leak (live_bytes growth) fails the assertion. `[VERIFIED: gaussian_nb_test.rs:294-340]`

## Sources

### Primary (HIGH confidence — read in-session at path:line)
- `crates/mlrs-algos/src/traits.rs:1-304` — OLD `Fit`/`PartialFit`/`Predict`/`Transform`/`PredictLabels`/`KNeighbors`/`ScoreSamples`/`PredictProba`/`PredictLogProba` (`&mut self`), FROZEN this phase.
- `crates/mlrs-algos/src/error.rs:30-363` (`AlgoError`, incl. `NotFitted:65-73`), `:384-547` (`BuildError`, all variants).
- `crates/mlrs-algos/src/linear/mbsgd_regressor.rs:36-363` — canonical v2 builder (owned setters + `build<F>() -> Result<_, BuildError>`) + `Fit`/`Predict`.
- `crates/mlrs-algos/src/decomposition/incremental_pca.rs:67-243` — the ONLY current `PartialFit` consumer (`partial_fit:224-242`, `fit:249-`); D-06 multi-transition target.
- `crates/mlrs-py/src/dispatch.rs:62-115` — `any_estimator!` macro (enum-only skeleton; `#[pymethods]` hand-written per estimator).
- `crates/mlrs-py/src/estimators/linear.rs:32-120` — worked PyO3 shell (`#[pyclass]` + `Unfit` arm + `fit` with `py.detach`/`guard_f64`/dtype dispatch + accessor).
- `crates/mlrs-py/src/errors.rs:39-141` — `algo_err_to_py`/`build_err_to_py`/`not_fitted` single-site mappers (reuse, D-13).
- `crates/mlrs-algos/tests/gaussian_nb_test.rs:294-340` — PoolStats `live_bytes` no-leak gate pattern; pool setup + f64-skip-with-log idiom.
- `crates/mlrs-algos/src/lib.rs`, `cluster/mod.rs`, `decomposition/mod.rs`, `estimators/mod.rs` — module-registration + re-export pattern.
- `crates/mlrs-algos/Cargo.toml`, root `Cargo.toml:13-31` — workspace deps; no `trybuild` present (must add).
- `.planning/REQUIREMENTS.md:23,30,37-40` — UMAP-01 / HDBS-01 authoritative param surfaces + BLDR-01/02/04.
- `AGENTS.md §2` — tests separated from source.
- `.planning/config.json:20,42-43` — `nyquist_validation:true`, `security_enforcement:true`, ASVS level 1.

### Secondary (MEDIUM confidence — official docs, cited)
- [CITED: github.com/dtolnay/trybuild README + Context7 `/dtolnay/trybuild`] — `compile_fail`, `.stderr`, dev-dependency, glob. trybuild 1.0.117 (cargo search).
- [CITED: cliffle.com/blog/rust-typestate/] — canonical Rust typestate: ZST markers + sealed trait + state-gated methods + consuming transitions.
- [CITED: doc.rust-lang.org/std/marker/struct.PhantomData.html] — `PhantomData<S>` semantics + variance.
- [CITED: scikit-learn.org sklearn.cluster.HDBSCAN] — full constructor + fitted attrs (`labels_`/`probabilities_`/`n_features_in_`/`centroids_`/`medoids_`).
- [CITED: umap-learn.readthedocs.io/en/latest/api.html] — full UMAP constructor kwargs + defaults.

### Tertiary (LOW confidence — flagged for validation)
- A1 (doctest as trybuild substitute), A2 (`.stderr` drift on this toolchain), A5 (`f64`-pin in builder `Default`) — see Assumptions Log; verify on first run.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every reused library is already in-tree (read at path:line); the one new dep (`trybuild`) is version-verified + legitimacy-audited.
- Architecture (typestate + builder + PyO3 collapse): HIGH — locked by D-01…D-13 and mirrored from shipped code (`mbsgd_regressor.rs`, `linear.rs`, `dispatch.rs`); typestate idiom cited.
- Pitfalls: HIGH — derived from the frozen-`traits.rs` constraint, the `S=Unfit` default/macro interaction, and the D-08 BuildError/AlgoError split, all grounded in read code.
- sklearn/umap param surfaces: HIGH for the in-scope subset (pinned in REQUIREMENTS.md UMAP-01/HDBS-01); full upstream kwarg lists cited as MEDIUM (informational).
- trybuild `.stderr` stability on this repo's toolchain: MEDIUM — verify on first `TRYBUILD=overwrite` run.

**Research date:** 2026-06-23
**Valid until:** 2026-07-23 (stable — internal API conventions + a dtolnay dev-dep; the only external-doc dependency is the sklearn/umap param lists, already frozen into REQUIREMENTS.md).
