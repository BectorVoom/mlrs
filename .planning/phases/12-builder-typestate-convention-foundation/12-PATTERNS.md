# Phase 12: Builder + Typestate Convention Foundation - Pattern Map

**Mapped:** 2026-06-23
**Files analyzed:** 16 new/modified
**Analogs found:** 14 / 16 (2 genuinely new: the sealed-typestate idiom + the trybuild gate — both have a partial analog plus a cited external idiom)

All analogs below were read at path:line in-session and verified to exist. The CONTEXT (D-01…D-13) and RESEARCH already cited these; this map confirms them and pins the exact excerpt the planner copies per file.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-algos/src/typestate.rs` (NEW) | new trait module (sealed `State` + `Unfit`/`Fitted` ZSTs + consuming `Fit`/`Predict`/`Transform`/`PartialFit`) | request-response / transform / event-driven (partial_fit) | `crates/mlrs-algos/src/traits.rs` (the FROZEN old surface, same trait names) | role-match (signatures change: `&mut self` → consuming `self`; add `type Fitted`) |
| `crates/mlrs-algos/src/manifold/mod.rs` (NEW) | module wiring | — | `crates/mlrs-algos/src/cluster/mod.rs` | exact |
| `crates/mlrs-algos/src/manifold/umap.rs` (NEW) | estimator shell (struct + builder + typestate impls + trivial fit + `embedding_`) | CRUD/transform (fit→transform) | `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` | role-match (builder shape exact; typestate `S` param + consuming fit are new) |
| `crates/mlrs-algos/src/cluster/hdbscan.rs` (NEW) | estimator shell (struct + builder + typestate impls + trivial fit + `labels_`) | CRUD (fit→labels) | `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` + `cluster/dbscan.rs` (labels-only, no standalone predict) | role-match |
| `crates/mlrs-algos/src/error.rs` (MODIFY) | error surface (new `BuildError` variants) | — | `crates/mlrs-algos/src/error.rs:384-547` (`BuildError` enum, in-file) | exact (extend in place) |
| `crates/mlrs-algos/src/lib.rs` (MODIFY) | module wiring / re-export | — | `crates/mlrs-algos/src/lib.rs:47-67` (existing `pub mod` + `pub use`) | exact |
| `crates/mlrs-algos/src/cluster/mod.rs` (MODIFY) | module wiring / re-export | — | `crates/mlrs-algos/src/cluster/mod.rs:21-43` | exact |
| `crates/mlrs-algos/Cargo.toml` (MODIFY) | config (add `trybuild` dev-dep) | — | `crates/mlrs-algos/Cargo.toml [dev-dependencies]` | exact |
| `crates/mlrs-py/src/dispatch.rs` (MODIFY/ADD) | macro (second `any_estimator_typestate!` spelling `<f32, Fitted>`) | — | `crates/mlrs-py/src/dispatch.rs:90-115` (`any_estimator!`) | exact (additive clone, marker swap) |
| `crates/mlrs-py/src/estimators/manifold.rs` (NEW) | PyO3 binding shell | request-response | `crates/mlrs-py/src/estimators/linear.rs:32-183` (`PyLinearRegression`) | exact |
| `crates/mlrs-py/src/estimators/cluster.rs` (MODIFY) — HDBSCAN PyO3 (planner's call: here or new `hdbscan.rs`) | PyO3 binding shell | request-response | `crates/mlrs-py/src/estimators/linear.rs` + `cluster.rs` (`PyDBSCAN` labels-only) | exact |
| `crates/mlrs-py/src/estimators/mod.rs` (MODIFY) | module wiring | — | `crates/mlrs-py/src/estimators/mod.rs:31-39` | exact |
| `crates/mlrs-py/src/lib.rs` (MODIFY) | pyclass registration | — | `crates/mlrs-py/src/lib.rs:217-258` (`m.add_class::<…>()?`) | exact |
| `crates/mlrs-algos/tests/umap_test.rs` (NEW) | test (defaults-eq, build-rejects, fit round-trip, no-leak) | — | `crates/mlrs-algos/tests/gaussian_nb_test.rs:280-337` | role-match |
| `crates/mlrs-algos/tests/hdbscan_test.rs` (NEW) | test | — | `crates/mlrs-algos/tests/gaussian_nb_test.rs:280-337` | role-match |
| `crates/mlrs-algos/tests/compile_fail.rs` + `tests/ui/*.rs`+`.stderr` (NEW) | test (trybuild compile-fail gate) | — | no in-repo analog — external dtolnay/trybuild idiom (RESEARCH Pattern 4) | no-analog (see below) |
| `crates/mlrs-py/tests/manifold_test.rs` (NEW) | test (cross-crate `unfit_default` smoke + `not_fitted` runtime analog) | — | `PyLinearRegression::unfit_default`/`is_unfit` seam (`estimators/linear.rs:44-56`) | role-match |

---

## Pattern Assignments

### `crates/mlrs-algos/src/typestate.rs` (NEW — sealed `State` + markers + 4 consuming traits)

**Analog:** `crates/mlrs-algos/src/traits.rs` (FROZEN — same trait NAMES, new path; the new module changes signatures `&mut self → self` and adds `type Fitted`). The sealed-ZST idiom itself is the one genuinely new piece (RESEARCH Pattern 1, cited cliffle.com typestate + std PhantomData).

**Imports pattern** — mirror `traits.rs:36-43` exactly (same bounds, same backend types), plus `PhantomData`:
```rust
use std::marker::PhantomData;            // NEW (markers)
use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use crate::error::AlgoError;
```

**Old trait signature being replaced** (`traits.rs:53-67` — the FROZEN form; DO NOT edit, copy the *shape* into the new module with `self`/`type Fitted`):
```rust
pub trait Fit<F> where F: Float + CubeElement + Pod {
    fn fit(&mut self, pool: &mut BufferPool<ActiveRuntime>,
           x: &DeviceArray<ActiveRuntime, F>, y: Option<&DeviceArray<ActiveRuntime, F>>,
           shape: (usize, usize)) -> Result<&mut Self, AlgoError>;   // ← old: &mut self
}
```

**New consuming surface to author** (D-05/D-06; RESEARCH Pattern 1 is the exact target — sealed module + ZSTs + `type Fitted`; `PartialFit` impl'd on BOTH `Unfit` and `Fitted`):
```rust
mod sealed { pub trait Sealed {} }
pub trait State: sealed::Sealed {}
pub struct Unfit;
pub struct Fitted;
impl sealed::Sealed for Unfit {}  impl State for Unfit {}
impl sealed::Sealed for Fitted {} impl State for Fitted {}

pub trait Fit<F> where F: Float + CubeElement + Pod {
    type Fitted;
    fn fit(self, pool: &mut BufferPool<ActiveRuntime>,
           x: &DeviceArray<ActiveRuntime, F>, y: Option<&DeviceArray<ActiveRuntime, F>>,
           shape: (usize, usize)) -> Result<Self::Fitted, AlgoError>;   // ← consumes self
}
// Predict<F> / Transform<F>: same body shape as traits.rs:109-138 (&self → DeviceArray).
// PartialFit<F>: `type Fitted; fn partial_fit(self, ..) -> Result<Self::Fitted, AlgoError>`.
```

**Re-export** (after writing the module, add to `lib.rs` — see lib.rs assignment): `pub mod typestate;` and a re-export. Because the names collide with `traits::*`, the planner must NOT glob both into the same `pub use`; re-export under the module path (consumers write `use mlrs_algos::typestate::Fit`).

---

### `crates/mlrs-algos/src/manifold/umap.rs` (NEW — estimator shell)

**Analog:** `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` (the canonical v2 builder, read in full). Copy the builder structure verbatim; layer the `S` typestate param + consuming fit on top.

**Struct — fitted fields are `Option`, `None` until fit** (`mbsgd_regressor.rs:36-43`); add the state marker (RESEARCH Pattern 1 / CONTEXT §Specific Ideas):
```rust
pub struct Umap<F, S = Unfit> {
    n_neighbors: usize, n_components: usize, min_dist: f64, spread: f64, /* … UMAP-01 subset … */
    embedding_: Option<DeviceArray<ActiveRuntime, F>>,   // None until fit (mirrors coef_)
    n_features_in_: usize,
    _state: PhantomData<S>,
}
```

**Builder — owned chained setters + non-generic struct + `build<F>()`** (`mbsgd_regressor.rs:89-126,128-198,213-287`). The setter/`#[derive(Debug, Clone, Copy)]` shape is copied 1:1:
```rust
#[derive(Debug, Clone, Copy)]
pub struct UmapBuilder { n_neighbors: usize, min_dist: f64, spread: f64, /* … */ }

impl UmapBuilder {
    pub fn min_dist(mut self, v: f64) -> Self { self.min_dist = v; self }   // ← one per field (mbsgd:130-198)
    pub fn build<F>(self) -> Result<Umap<F, Unfit>, BuildError>
    where F: Float + CubeElement + Pod {
        // data-INDEPENDENT validation BEFORE any data (mbsgd:217-265). UMAP: min_dist <= spread,
        // finite min_dist/spread → NEW BuildError variant (see error.rs assignment).
        if !(self.min_dist <= self.spread) {
            return Err(BuildError::InvalidMinDist { estimator: "umap", min_dist: self.min_dist });
        }
        Ok(Umap { /* move every hyperparam */, embedding_: None, n_features_in_: 0, _state: PhantomData })
    }
}
```

**`new()` as the single defaults source + `Default` round-trip** (D-08 — this is the ONE deviation from `mbsgd_regressor.rs`, whose `Default` at `:107-126` holds the literals). New shape (RESEARCH Pattern 2):
```rust
impl<F: Float + CubeElement + Pod> Umap<F, Unfit> {
    pub fn new() -> Self { Self { n_neighbors: 15, n_components: 2, min_dist: 0.1, spread: 1.0,
        /* sklearn defaults */, embedding_: None, n_features_in_: 0, _state: PhantomData } }
    pub fn builder() -> UmapBuilder { UmapBuilder::default() }
    pub fn into_builder(self) -> UmapBuilder { /* copy hyperparams */ }
}
impl Default for UmapBuilder { fn default() -> Self { Umap::<f64, Unfit>::new().into_builder() } }
```

**Consuming `Fit` + trivial NON-algorithmic fit body** (D-10; geometry guard mirrors `mbsgd_regressor.rs:303-312` `PrimError::ShapeMismatch`; allocation mirrors `DeviceArray::from_host` in the tests):
```rust
impl<F: Float + CubeElement + Pod> Fit<F> for Umap<F, Unfit> {
    type Fitted = Umap<F, Fitted>;
    fn fit(self, pool, x, _y, shape) -> Result<Umap<F, Fitted>, AlgoError> {
        let (n, p) = shape;
        if n == 0 || p == 0 || x.len() != n * p {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch { operand: "x", rows: n, cols: p, len: x.len() }));
        }
        let zeros = vec![F::from_int(0); n * self.n_components];     // NO kernel, NO compute
        let embedding = DeviceArray::from_host(pool, &zeros);
        Ok(Umap { /* move hyperparams */, embedding_: Some(embedding), n_features_in_: p, _state: PhantomData })
    }
}
```

**Fitted-only accessors + `Transform`** (D-02 — exist ONLY on `impl Umap<F, Fitted>`; this is what makes predict-before-fit a compile error). Accessor body mirrors `mbsgd_regressor.rs:62-70` `coef()` (`to_host` on the `Option`, but here the `Option` is always `Some` on `Fitted`):
```rust
impl<F: Float + CubeElement + Pod> Umap<F, Fitted> {
    pub fn embedding(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.embedding_.as_ref().unwrap().to_host(pool)   // Some by construction on Fitted
    }
    pub fn n_features_in(&self) -> usize { self.n_features_in_ }
}
impl<F: Float + CubeElement + Pod> Transform<F> for Umap<F, Fitted> { /* … */ }
```

---

### `crates/mlrs-algos/src/cluster/hdbscan.rs` (NEW — estimator shell)

**Analog:** same `mbsgd_regressor.rs` builder template as UMAP, plus `cluster/dbscan.rs` for the labels-only surface (HDBSCAN, like DBSCAN, exposes `labels_` and no standalone `predict`). Differences from UMAP:
- Param subset HDBS-01: `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, `cluster_selection_method`, `metric`, `alpha`, `max_cluster_size`.
- `build<F>()` validation: `min_cluster_size >= 2`, `min_samples >= 1` → NEW `BuildError::InvalidMinClusterSize` (see error.rs assignment).
- Trivial fit allocates **all-`-1` `i32`** labels (DBSCAN noise sentinel — `cluster/mod.rs:12` documents `-1`), stored device-resident as `labels_: Option<DeviceArray<ActiveRuntime, i32>>`. Fit body otherwise identical to UMAP's (geometry guard + `from_host` of `vec![-1_i32; n]`).
- Fitted accessor `labels(&self, pool) -> Vec<i32>` on `impl Hdbscan<F, Fitted>` only.

---

### `crates/mlrs-algos/src/error.rs` (MODIFY — extend `BuildError`)

**Analog:** the `BuildError` enum itself, `error.rs:383-547` (read in full). Add new variants at the end of the enum following the existing variant shape EXACTLY (`#[error("…")]` + `estimator: &'static str` + the offending value). Worked precedent — `InvalidVarSmoothing` (`error.rs:507-520`):
```rust
#[error("estimator '{estimator}': var_smoothing = {var_smoothing} is invalid (must be >= 0)")]
InvalidVarSmoothing { estimator: &'static str, var_smoothing: f64 },
```
New variants to add (names are planner's discretion; CONTEXT suggests `InvalidMinDist` / `InvalidMinClusterSize`):
```rust
#[error("estimator '{estimator}': min_dist = {min_dist} is invalid (must be <= spread and finite)")]
InvalidMinDist { estimator: &'static str, min_dist: f64 },
#[error("estimator '{estimator}': min_cluster_size = {min_cluster_size} is invalid (must be >= 2)")]
InvalidMinClusterSize { estimator: &'static str, min_cluster_size: usize },
```
**Critical (Pitfall 3):** these are data-INDEPENDENT → `BuildError` at `build()`, NOT `AlgoError` at `fit()`. The single-site `build_err_to_py` (`errors.rs:71-73`) already maps the whole enum — no PyO3 mapper change needed.

---

### `crates/mlrs-py/src/dispatch.rs` (MODIFY/ADD — `any_estimator_typestate!`)

**Analog:** the existing `any_estimator!` macro, `dispatch.rs:90-115` (read in full). The current fitted arms (`dispatch.rs:104-106`):
```rust
F32($algo $( :: $algo_rest )* <f32>),
F64($algo $( :: $algo_rest )* <f64>),
```
For typestate estimators these must spell `<f32, Fitted>` / `<f64, Fitted>` (D-04 — because `S = Unfit` default would otherwise resolve `<f32>` to the WRONG `Unfit` arm, Pitfall 2). **Add a SECOND macro** `any_estimator_typestate!` (additive — leaves the shared 35-call-site macro byte-for-byte untouched, protecting Success Criterion 3). It is the existing macro cloned with the two fitted arms changed to:
```rust
F32($algo $( :: $algo_rest )* <f32, mlrs_algos::typestate::Fitted>),
F64($algo $( :: $algo_rest )* <f64, mlrs_algos::typestate::Fitted>),
```
Both the enum skeleton and the `Unfit { $field : $ty }` arm are copied unchanged. `#[macro_export]` like the original.

---

### `crates/mlrs-py/src/estimators/manifold.rs` (NEW — PyO3 UMAP shell)

**Analog:** `crates/mlrs-py/src/estimators/linear.rs:32-183` (`PyLinearRegression`, read in full) — the worked `any_estimator!` + `#[pymethods]` shell. Copy line-for-line, swapping the consuming-fit call.

**Imports + macro invocation + pyclass + smoke seam** (`linear.rs:11-56`):
```rust
use pyo3::prelude::*;
use mlrs_algos::manifold::umap::Umap;
use mlrs_algos::typestate::{Fit, Transform};          // NEW surface (not traits::)
use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

crate::any_estimator_typestate! {                      // ← the SECOND macro (D-04)
    any: AnyUmap, algo: mlrs_algos::manifold::umap::Umap,
    unfit: { n_neighbors: usize, min_dist: f64, spread: f64, /* … */ },
}

#[pyclass(name = "UMAP")]
pub struct PyUmap { inner: AnyUmap }

impl PyUmap {
    pub fn unfit_default() -> Self { Self { inner: AnyUmap::Unfit { /* sklearn defaults */ } } }   // smoke seam (linear.rs:48)
    pub fn is_unfit(&self) -> bool { matches!(self.inner, AnyUmap::Unfit { .. }) }
}
```

**`fit` body — consuming form is STRICTLY simpler than `linear.rs:86-105`** (no `mut`, no `&mut self`; D-02 maps cleanly). Keep `py.detach` + `lock_pool` + `guard_f64()?` on the F64 arm + `build_err_to_py`/`algo_err_to_py` (RESEARCH Pattern 5):
```rust
let fitted = py.detach(|| -> PyResult<AnyUmap> {
    let mut pool = crate::lock_pool();
    match dt {
        FloatDtype::F32 => {
            let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
            let est = Umap::<f32>::builder()./* setters */.build::<f32>().map_err(build_err_to_py)?;
            let fitted = est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;  // consumes, returns Fitted
            Ok(AnyUmap::F32(fitted))
        }
        FloatDtype::F64 => {
            crate::capability::guard_f64()?;            // BEFORE upload (linear.rs:97)
            let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
            let fitted = Umap::<f64>::builder()./* setters */.build::<f64>().map_err(build_err_to_py)?
                .fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
            Ok(AnyUmap::F64(fitted))
        }
    }
})?;
self.inner = fitted;
```

**`embedding_` accessor + runtime NotFitted (D-13)** — mirror `linear.rs:142-148` `coef_f32` exactly: match the fitted arm, else `not_fitted("umap", "embedding_")` (`errors.rs:84-88` — reuse, do not add a mapper):
```rust
fn embedding_f32(&self, py: Python<'_>, ...) -> PyResult<Vec<f32>> {
    py.detach(|| { let mut pool = crate::lock_pool(); match &self.inner {
        AnyUmap::F32(est) => Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool)),
        _ => Err(not_fitted("umap", "embedding_ (f32 path)")),
    }})
}
```
Add `is_fitted` / `dtype` exactly as `linear.rs:172-182`.

**HDBSCAN PyO3** (planner's call — extend `estimators/cluster.rs` or new `hdbscan.rs`): same template; the `labels_` accessor returns `Vec<i32>` (mirror `PyKMeans`/`PyDBSCAN` label accessors in `cluster.rs`, which return `to_host_metered` i32), and HDBSCAN has no standalone predict (labels-only, like `PyDBSCAN`).

---

### Module wiring (MODIFY — `lib.rs`, `cluster/mod.rs`, `estimators/mod.rs`, `mlrs-py/src/lib.rs`)

**`mlrs-algos/src/lib.rs`** (`:47-66`): add `pub mod manifold;` and `pub mod typestate;` to the `pub mod` block. Do NOT glob-re-export `typestate::*` alongside `traits::*` (name collision); re-export under the path only.

**`mlrs-algos/src/cluster/mod.rs`** (`:21-43` — existing `pub mod` + `pub use`): add `pub mod hdbscan;` and `pub use hdbscan::Hdbscan;` following the `spectral_*` precedent at `:39-43`.

**`manifold/mod.rs`** (NEW): mirror `cluster/mod.rs:21-43` shape — `pub mod umap; pub use umap::Umap;`.

**`mlrs-py/src/estimators/mod.rs`** (`:31-39`): add `pub mod manifold;` (one line, alphabetical with the existing `pub mod cluster;` … list).

**`mlrs-py/src/lib.rs`** (`:217-258`): add `m.add_class::<PyUmap>()?;` (and `PyHdbscan`) in the registration block following the per-phase grouping comment pattern (`:246` etc.), plus the matching `use estimators::manifold::PyUmap;` import alongside `:208-216`.

---

### `crates/mlrs-algos/Cargo.toml` (MODIFY — add dev-dep)

**Analog:** the existing `[dev-dependencies]` block (verified present: `mlrs-core`, `env_logger`). Add:
```toml
trybuild = "1.0.117"
```
No other dep changes — `trybuild` is dev-only, by dtolnay (same author as the workspace `thiserror`/`syn`), legitimacy-audited in RESEARCH.

---

### Tests (NEW — `umap_test.rs`, `hdbscan_test.rs`, `compile_fail.rs`+`ui/`, `manifold_test.rs`)

**`tests/umap_test.rs` / `tests/hdbscan_test.rs`** — analog `crates/mlrs-algos/tests/gaussian_nb_test.rs` (read `:280-337`). Three reusable excerpts:

1. **build-rejects** (`gaussian_nb_test.rs:280-292`):
```rust
let bad = GaussianNB::<f64>::builder().var_smoothing(-1.0).build::<f64>().err();
assert!(matches!(bad, Some(BuildError::InvalidVarSmoothing { var_smoothing, .. }) if var_smoothing == -1.0));
```
→ adapt to `Umap::<f64>::builder().min_dist(2.0).spread(1.0).build::<f64>()` → `BuildError::InvalidMinDist`.

2. **PoolStats no-leak gate** (`gaussian_nb_test.rs:294-337` — the milestone per-phase memory gate). Copy the pool setup (`runtime::active_client()` → `BufferPool::new`), the f64-skip-with-log guard (`:299-304`), and the monotone assertion (`:323-335`):
```rust
let live_after_first = pool.stats().live_bytes;
for k in 0..REFITS { /* re-fit at same shape */ let live = pool.stats().live_bytes;
    assert!(live <= live_after_first, "live_bytes grew across re-fit {k} …"); }
```
**Adaptation for the consuming fit:** because `fit(self)` consumes, the no-leak loop re-CONSTRUCTS (`Umap::new()` / `builder().build()`) and re-fits each iteration, asserting `live_bytes` does not climb across the zeros-embedding (UMAP) / `-1`-labels (HDBSCAN) allocation + drop.

3. **fit round-trip** (new — D-10 runtime proof): `Umap::<F>::new().fit(&mut pool, &xd, None, (n, p))?` then `.embedding(&pool)` returns `n * n_components` zeros (HDBSCAN: `.labels()` all `-1`). Pool setup copied from `gaussian_nb_test.rs:308-313`.

4. **defaults-eq** (BLDR-01): assert the hyperparameter subset of `Umap::<F>::new()` equals `Umap::builder().build::<F>()?` (derive `PartialEq` on the hyperparam fields, or compare field-by-field — RESEARCH Open Q1 / Pitfall 4).

**`tests/compile_fail.rs` + `tests/ui/{predict,transform}_before_fit.rs`+`.stderr`** — NO in-repo analog (first trybuild use). External idiom (RESEARCH Pattern 4, cited dtolnay/trybuild):
```rust
// tests/compile_fail.rs
#[test] fn ui() { trybuild::TestCases::new().compile_fail("tests/ui/*.rs"); }
```
```rust
// tests/ui/transform_before_fit.rs
use mlrs_algos::manifold::umap::Umap;
use mlrs_algos::typestate::Transform;
fn main() { let est = Umap::<f32>::new(); let _ = est.transform(/* … */); }  // E0599 no method on Unfit
```
Generate `.stderr` with `TRYBUILD=overwrite cargo test -p mlrs-algos --features cpu ui` on the pinned toolchain, then commit. Keep the ui file to ONE method call (Pitfall 5 — stderr stability). AGENTS.md §2 compliant (ui files are `tests/` fixtures, not in-source `#[cfg(test)] mod`).

**`crates/mlrs-py/tests/manifold_test.rs`** — analog the `unfit_default`/`is_unfit` cross-crate seam (`estimators/linear.rs:44-56`). Runs without a Python interpreter: assert `PyUmap::unfit_default().is_unfit()` (BLDR-04 smoke) and that an accessor on the `Unfit` arm returns the `not_fitted` `PyValueError` (D-13 runtime analog).

---

## Shared Patterns

### Builder shape (owned chained setters + non-generic builder + `build<F>()`)
**Source:** `crates/mlrs-algos/src/linear/mbsgd_regressor.rs:89-287`
**Apply to:** `manifold/umap.rs`, `cluster/hdbscan.rs`
Copy `#[derive(Debug, Clone, Copy)]` builder struct, `fn param(mut self, v) -> Self` setters (`:130-198`), and `build<F>(self) -> Result<T<F, Unfit>, BuildError>` with data-INDEPENDENT validation up front (`:217-265`). The ONLY change from the template: `Default` re-derives via `new().into_builder()` (D-08), not literal defaults.

### `BuildError` validation (data-independent, at `build()`)
**Source:** `crates/mlrs-algos/src/error.rs:383-547` (variant shape) + `mbsgd_regressor.rs:217-265` (call site)
**Apply to:** both shell builders. Validate at `build()` → typed `BuildError`; the data-DEPENDENT geometry guard stays at `fit` → `AlgoError::Prim(PrimError::ShapeMismatch)` (`mbsgd_regressor.rs:303-312`). Pitfall 3: never put a hyperparameter check in `fit` or reuse an `AlgoError` variant for it.

### Geometry guard at fit
**Source:** `crates/mlrs-algos/src/linear/mbsgd_regressor.rs:303-312`
**Apply to:** both trivial fit bodies — `if n == 0 || p == 0 || x.len() != n*p { return Err(AlgoError::Prim(PrimError::ShapeMismatch { … })) }`.

### Fitted-field accessor (`Option` → `to_host`, NotFitted)
**Source:** `crates/mlrs-algos/src/linear/mbsgd_regressor.rs:62-82`
**Apply to:** the `Fitted`-gated accessors — but on the new surface the accessor lives on `impl T<F, Fitted>` (no `NotFitted` branch needed Rust-side; the `Option` is `Some` by construction). The `NotFitted` runtime path moves to the PyO3 `Unfit` arm.

### PyO3 dispatch contracts (GIL release, f64 guard, poison-recovering lock)
**Source:** `crates/mlrs-py/src/estimators/linear.rs:86-108` + `dispatch.rs:35-57`
**Apply to:** both PyO3 shells. `py.detach(|| { let mut pool = crate::lock_pool(); … })`; `guard_f64()?` BEFORE the F64 upload; `lock_pool()` (poison-recovering), never `global_pool().lock().expect()`.

### Single-site error mappers (reuse, never duplicate — D-13)
**Source:** `crates/mlrs-py/src/errors.rs:56-88`
**Apply to:** both PyO3 shells. `build_err_to_py` (build), `algo_err_to_py` (fit/predict), `not_fitted("umap"|"hdbscan", op)` (Unfit-arm accessor). No new mapper for the new `BuildError` variants — `build_err_to_py` already maps the whole enum.

### Module registration / re-export
**Source:** `crates/mlrs-algos/src/cluster/mod.rs:21-43`, `lib.rs:47-66`, `crates/mlrs-py/src/estimators/mod.rs:31-39`, `crates/mlrs-py/src/lib.rs:217-258`
**Apply to:** all wiring edits. One `pub mod` + `pub use` line per new module; one `m.add_class::<…>()?` + `use` per new pyclass. **Exception (Pitfall 1):** do NOT glob `typestate::*` into the same `pub use` as `traits::*` (name collision) — re-export under the path.

### PoolStats no-leak memory gate
**Source:** `crates/mlrs-algos/tests/gaussian_nb_test.rs:294-337`
**Apply to:** both shell test files. `live_bytes` monotone-non-increasing across re-construct+fit, plus the `skip_f64_with_log()` guard for the rocm f64 gate.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/mlrs-algos/tests/compile_fail.rs` + `tests/ui/*.rs`+`.stderr` | test (compile-fail gate) | — | First `trybuild` use in the repo — no in-tree analog. Use the external dtolnay/trybuild idiom (RESEARCH Pattern 4) verbatim. Planner uses RESEARCH §Pattern 4 + Pitfall 5, not a codebase file. |
| `crates/mlrs-algos/src/typestate.rs` (the sealed-ZST + consuming-trait IDIOM specifically) | new trait module | — | Trait *names/signatures* have a partial analog in `traits.rs`, but the sealed `State`/`Unfit`/`Fitted` ZST + `type Fitted` consuming pattern is genuinely new. Use RESEARCH §Pattern 1 (cited cliffle.com typestate + std PhantomData) for the idiom; `traits.rs:36-138` for the per-trait body shape. |

---

## Metadata

**Analog search scope:** `crates/mlrs-algos/src/{linear,cluster,decomposition}/`, `crates/mlrs-algos/src/{traits,error,lib}.rs`, `crates/mlrs-algos/tests/`, `crates/mlrs-py/src/{dispatch,errors,lib}.rs`, `crates/mlrs-py/src/estimators/`, `crates/mlrs-algos/Cargo.toml`.
**Files scanned (read in-session at path:line):** `mbsgd_regressor.rs` (full), `traits.rs` (full), `error.rs:360-547`, `incremental_pca.rs:200-320`, `dispatch.rs` (full), `estimators/linear.rs:1-1225` (PyLinearRegression worked example), `errors.rs` (full), `gaussian_nb_test.rs:280-337`, `lib.rs` (algos, full), `cluster/mod.rs` (full), `estimators/mod.rs` (full), `estimators/cluster.rs:1-40`, `mlrs-py/src/lib.rs:208-260`, plus directory/Cargo structural checks.
**Pattern extraction date:** 2026-06-23
