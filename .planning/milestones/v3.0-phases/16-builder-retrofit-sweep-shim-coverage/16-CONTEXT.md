# Phase 16: Builder Retrofit Sweep + Shim Coverage - Context

**Gathered:** 2026-06-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Retrofit the **Phase-12 builder + typestate convention across every existing estimator** (~21 not born with it) and **complete the pure-Python sklearn shim** (`get_params`/`set_params`/`clone` coverage from the v1 12 → v2 18 + UMAP/HDBSCAN, plus PyO3-wrapping the two new estimators). This is the one broad-edit, parallel-unsafe phase — isolated last to protect file-disjoint discipline and the shipped 1e-5 / exact-label gates.

**"Additive" is about the algorithm internals, not the surface.** The convention being retrofitted *is* the typestate, so each estimator gains the `<F, S = Unfit>` state param, an owned builder, a zero-arg `new()`, and migrates to the consuming-self `typestate` trait surface. What stays byte-identical is each estimator's **config-struct fields and fit numerics** — that is what preserves the gates. Requirements in scope: **BLDR-03**, **SHIM-01**, **SHIM-02**, **SHIM-03**.

**In scope:**
- Per-estimator typestate retrofit: add `S = Unfit` state param + `PhantomData`, owned chained-setter builder, zero-arg `new()` (sklearn defaults), migrate fit→consuming-self + Fitted-gated accessors on the new `mlrs_algos::typestate` traits.
- **Full convergence:** port all 9 old `crate::traits` trait shapes to typestate-aware versions, migrate every estimator, then **delete `crate::traits` (`traits.rs`)** at phase end → single trait surface.
- Convert all arg-taking `new(args)` / `with_*()` constructors to the zero-arg-defaults convention; all parameterization moves to the builder; migrate the ~137 `::new(` call sites.
- Python shim: extend pure-Python classes (v1 12 → v2 18 + UMAP/HDBSCAN); PyO3-wrap UMAP/HDBSCAN; full static shim gate.
- Pilot 1–2 estimators under the green suite before the full sweep.

**Out of scope:**
- Any change to fit algorithm bodies / numerics, or to config-struct field sets (additive front-door only; gates must not move).
- The live `estimator_checks` / `check_estimator` FFI run — stays deferred (no maturin+pyarrow host in this environment; SHIM-03 covers the static path).
- New estimators, new algorithms, device-kernel work.

</domain>

<decisions>
## Implementation Decisions

### Retrofit depth & old-trait fate (BLDR-03)
- **D-01: Full convergence — delete old `traits.rs`.** Port all 9 old `crate::traits` traits (`Fit`, `PartialFit`, `Predict`, `Transform`, `PredictLabels`, `KNeighbors`, `ScoreSamples`, `PredictProba`, `PredictLogProba`) to typestate-aware versions in `mlrs_algos::typestate`, migrate every existing estimator to the consuming-self surface, then **hard-delete `crate::traits`** at phase end. One trait surface, zero permanent debt — realizes Phase-12 D-07's stated end-state. Not chosen: additive coexistence (keep `traits.rs` live) — rejected as standing two-surface debt; deferred-deletion shim — rejected, the user wants the full removal in this phase.
- **D-02: The typestate convention is layered onto ALL ~21 estimators**, including the 11 that already have a `builder()` (MBSGD/SVC/SVR + 5×NB) — those still `use crate::traits` and need the `<F,S>` state param + trait migration even though their builder exists. UMAP/HDBSCAN are already on the new surface (born with it, Phase 12).
- **D-03: Config fields + fit numerics are byte-identical across the retrofit.** The retrofit wraps construction and lifecycle around each algorithm; it never touches the struct's field set or the fit body math. This is the literal meaning of BLDR-03 "additive / fit path untouched" and the mechanism that preserves every shipped 1e-5 / exact-label gate. Each estimator is migrated **under its own green suite** (migrate → run suite → green → commit).

### new() reconciliation (BLDR-01 / D-08 convention)
- **D-04: `new()` → zero-arg sklearn-defaults on every estimator; all args move to the builder.** Existing arg-taking constructors (`Ridge::new(alpha, fit_intercept)`, `KMeans::new(n_clusters, seed)` + `with_init` + `with_opts`, etc.) are **removed**; their ~137 `::new(` call sites migrate to `T::builder().param(..).…build()?`. This makes the single-source invariant `T::new() == T::builder().build()?` == sklearn default hold uniformly. Not chosen: keep `new(args)` + add builder (rejected — two construction idioms, invariant only partial); convert-simple-keep-multi (rejected — leaves inconsistency exactly at the complex KMeans case).

### Pilot selection & sweep order (BLDR-03 SC1)
- **D-05: Pilot Ridge + MBSGDRegressor — the two structurally-distinct retrofit shapes.**
  - **Ridge** = the *no-builder / arg-taking-`new`* shape → proves the full build-out: add state param + builder + `new()`-conversion + trait migration + call-site sweep.
  - **MBSGDRegressor** = the *already-has-builder / old-trait* shape → proves the typestate-param + trait-swap-only path (no builder to invent).
  Both pilots green under their suites before the bulk sweep.
- **D-06: Sweep the rest module-by-module**, each estimator gated by its own suite (linear → decomposition → cluster → covariance → projection → density → neighbors → kernel_ridge → naive_bayes). **KMeans handled late** as the multi-constructor (`new`/`with_init`/`with_opts`) stress case, after the recipe is proven.

### Python shim verification (SHIM-01 / SHIM-02 / SHIM-03)
- **D-07: Full static shim gate (maximum verifiable without FFI).** Per pure-Python class (all 18 + UMAP/HDBSCAN):
  1. import without the compiled `_mlrs` extension;
  2. assert `get_params(deep=True)` / `set_params(**kw)` round-trip exactly + `clone()` equivalence;
  3. **AST-based `__init__`-purity assertion** — each ctor arg stored verbatim, same name, no validation/computation;
  4. the **fit-free subset of sklearn `estimator_checks`** (`check_no_attributes_set_in_init`, `check_parameters_default_constructible`, `check_get_params_invariance`).
  Plus Rust-side unit tests. The live `check_estimator` FFI run stays **deferred** (no maturin+pyarrow host) → route to UAT. Builds on the existing `MlrsBase` (`crates/mlrs-py/python/mlrs/base.py`) + `test_shims.py` / `test_params.py` / `test_estimator_checks.py` infra.
- **D-08: UMAP/HDBSCAN PyO3 wraps (SHIM-02)** follow the shipped pattern: `#[pyclass]` on the existing `any_estimator!` machinery, GIL release, `guard_f64` before F64, sklearn-named params, trailing-underscore fitted attrs, `n_features_in_` set/enforced, `fit` returns `self`; correct surface (UMAP `transform`/`fit_transform`; HDBSCAN `fit_predict`/`labels_`). Reuse the single-site `build_err_to_py` / `algo_err_to_py` mappers.

### Claude's Discretion
- **Boilerplate generation (researcher to evaluate):** with ~21 estimators receiving the same mechanical retrofit, a `derive`/declarative macro to emit the per-estimator state param + builder + impl blocks is a candidate (raised but not decided in Phase 12). The researcher MAY evaluate it; hand-written retrofit is fully acceptable if the macro cost/benefit doesn't pay off at this count. Either way the per-estimator green-suite gate (D-03) is non-negotiable.
- Exact module/file ordering within the sweep, naming of ported typestate traits, and whether the call-site migration is one commit per estimator or per module — planner's call, following existing structure.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — BLDR-03, SHIM-01, SHIM-02, SHIM-03 (in scope); BLDR-01/02/04 (Phase-12 convention this retrofits toward); the "Variant A" note (additive retrofit isolated to protect gates).
- `.planning/ROADMAP.md` § "Phase 16" — goal, depends-on (Phase 12 convention, Phase 14 UMAP, Phase 15 HDBSCAN), four Success Criteria.

### The convention being retrofitted (the single target shape)
- `.planning/phases/12-builder-typestate-convention-foundation/12-CONTEXT.md` — D-01…D-13 (typestate encoding, consuming-self fit, single-source `new()` defaults, coexistence-then-converge D-07, PyO3 collapse). **The authoritative spec for the shape every estimator converges to.**
- `crates/mlrs-algos/src/typestate.rs` — the new typestate trait surface + `Unfit`/`Fitted` markers; **target** of the migration (must grow to mirror all 9 old traits).
- `crates/mlrs-algos/src/traits.rs` — the OLD 9-trait `&mut self` surface; **DELETE at phase end** (D-01).

### Reference implementations (born-with-convention exemplars)
- `crates/mlrs-algos/src/manifold/umap.rs` — full convention exemplar: `Umap<F, S = Unfit>` + `PhantomData`, zero-arg `new()`, `UmapBuilder` + `Default`, consuming-self `Fit` on `typestate`. The template to retrofit toward.
- `crates/mlrs-algos/src/cluster/hdbscan.rs` — second exemplar (cluster surface).

### Retrofit targets (representative shapes)
- `crates/mlrs-algos/src/linear/ridge.rs` — pilot A: no-builder, arg-taking `new(alpha, fit_intercept)`, `use crate::traits`.
- `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` — pilot B: already has `builder()` but `MBSGDRegressor<F>` (no state param) + `use crate::traits`.
- `crates/mlrs-algos/src/cluster/kmeans.rs` — late case: `new(n_clusters, seed)` + `with_init` + `with_opts` multi-constructor.

### PyO3 + shim
- `crates/mlrs-py/src/dispatch.rs` — `any_estimator!` machinery + the `Unfit/F32/F64` enum (PyO3 collapse target; D-13/D-04 from Phase 12).
- `crates/mlrs-py/src/estimators/` — per-family PyO3 wraps (`manifold.rs`, `cluster.rs` for UMAP/HDBSCAN).
- `crates/mlrs-py/python/mlrs/base.py` — `MlrsBase` (subclasses sklearn `BaseEstimator`; `get_params`/`set_params`/`clone` come free from faithful `__init__`).
- `crates/mlrs-py/python/tests/test_shims.py`, `test_params.py`, `test_estimator_checks.py` — existing static shim test infra to extend (D-07).

### Project conventions
- `AGENTS.md` §2 — tests separated from source (tests live in `crates/*/tests/`).
- `.planning/codebase/CONVENTIONS.md` — coding conventions.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **UMAP/HDBSCAN born-with-convention shells** — exact target shape (`<F, S=Unfit>` + `PhantomData`, zero-arg `new()`, builder + `Default`, consuming-self `typestate::Fit`). Copy the pattern; don't reinvent.
- **`MlrsBase` + `_io` ingress/egress** (`mlrs-py/python/mlrs/base.py`) — already supplies `get_params`/`set_params`/`clone`/`n_features_in_`/`__sklearn_tags__` for free; shim work = adding faithful per-class `__init__`s + extending the static test matrix.
- **Single-site PyO3 error mappers** (`build_err_to_py` / `algo_err_to_py`) and `any_estimator!` macro — reused for the UMAP/HDBSCAN wraps, not duplicated.

### Established Patterns
- All ~21 retrofit targets currently `use crate::traits` (`&mut self` fit, `Option<...>` fitted fields, runtime `NotFitted`). The retrofit REPLACES this with the consuming-self `typestate` surface + compile-time Fitted-gating — config fields and fit numerics held byte-identical.
- Estimators are generic `<F: Float + CubeElement + Pod>`; the state param `S` is additive to that (`<F, S = Unfit>`).
- ~137 `::new(` call sites in `crates/mlrs-algos/tests/` migrate to `builder()` (D-04).

### Integration Points
- `mlrs_algos::typestate` grows to mirror all 9 old traits; every estimator + every test call site converges onto it; `traits.rs` deleted last.
- PyO3 wraps for UMAP/HDBSCAN land in `mlrs-py/src/estimators/` and register via `any_estimator!`.
- Pure-Python shim classes for the v2 18 + UMAP/HDBSCAN added under `mlrs-py/python/mlrs/`.

</code_context>

<specifics>
## Specific Ideas

- **Per-estimator green-suite gate is the safety mechanism** (D-03): migrate one estimator → run its suite → green → commit, never a big-bang rewrite. This is how a broad, parallel-unsafe sweep preserves every shipped gate.
- Pilot order is fixed: **Ridge first** (full build-out), **MBSGDRegressor second** (trait-swap-only), before any bulk module sweep; **KMeans last** within the sweep.
- The live FFI `check_estimator` cannot run here (no maturin+pyarrow) — the **static subset is the maximum verifiable gate**; the live run is explicitly UAT/deferred (consistent with the standing "Python wheel untestable in env" constraint).

</specifics>

<deferred>
## Deferred Ideas

- **Builder/typestate boilerplate `derive` macro** — moved from "deferred" to a *candidate for this phase*: researcher to evaluate whether a generator pays off at the ~21-estimator count (D-Claude's-Discretion). If it doesn't, hand-written retrofit stands. Not a scope expansion either way — same surface, same gates.

### Reviewed Todos (not folded)
None — no pending todos matched this phase.

</deferred>

---

*Phase: 16-Builder Retrofit Sweep + Shim Coverage*
*Context gathered: 2026-06-24*
