# Phase 12: Builder + Typestate Convention Foundation - Context

**Gathered:** 2026-06-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish the **canonical Rust-native estimator-construction convention** — an owned chained-setter builder + a compile-time fit/unfit typestate + the existing typed `BuildError` validation surface — and demonstrate it end-to-end on two new-estimator shells (UMAP, HDBSCAN) so Phases 14–15 are born builder-fronted and Phase 16 has a single target shape to retrofit toward.

**Pure API foundation: no algorithm, no device kernels, no retrofit.** The broad 30-estimator retrofit sweep is BLDR-03 / Phase 16. Requirements in scope: **BLDR-01** (builder), **BLDR-02** (typestate), **BLDR-04** (PyO3 collapse).

**In scope:** the typestate marker types + the new typestate-aware trait surface (new module, coexisting with the untouched old traits); the builder/`new()` defaults convention; UMAP/HDBSCAN shells (full param + fitted-attr surface, trivial non-algorithmic fit); a PyO3 shell for each via the existing `any_estimator!` machinery; a compile-fail gate proving predict-before-fit won't compile.

**Out of scope:** any real UMAP/HDBSCAN algorithm or device work (Phases 14–15); retrofitting the 30 existing estimators to the new surface (Phase 16); deleting the old `traits.rs` surface (end of Phase 16); the Python sklearn shim coverage (SHIM-*, Phase 16).
</domain>

<decisions>
## Implementation Decisions

### Typestate encoding (BLDR-02)
- **D-01: Per-estimator state type-param.** Each estimator struct gains a marker param: `Umap<F, S = Unfit>` carrying `PhantomData<S>`. This is the textbook typestate; chosen knowingly over a shared `Unfit<E>/Fitted<E>` wrapper and over two distinct config/fitted types, accepting the signature-touching Phase-16 retrofit cost.
- **D-02: `fit` consumes `self`.** Transition is `fit(self, ..) -> Result<T<F, Fitted>, AlgoError>` — it must take `self` by value to re-tag the marker. `predict` / `transform` / fitted-attr accessors are implemented **only** on `impl T<F, Fitted>`; `T<F, Unfit>` has no such impl, so predict-before-fit is a compile error. Chaining still works: `est.fit(x)?.predict(x)`.
- **D-03: Marker types + sealed `State`.** Introduce `Unfit` / `Fitted` zero-sized marker types and a sealed `State` trait bound (exact bound/naming is a planner detail). Default param `S = Unfit`, so a bare `T<F>` means the unfit state.
- **D-04: `any_estimator!` fitted arms must spell the state explicitly.** Because `S` defaults to `Unfit`, the macro's fitted arms become `F32(T<f32, Fitted>)` / `F64(T<f64, Fitted>)` (not `T<f32>`). The PyO3 `fit` path already transitions `Unfit { params }` → builds `T<f32, Unfit>` → `fit` → stores the `Fitted` monomorphization, which maps cleanly onto the enum. The macro may need a small change to thread the `Fitted` marker; confirm against `crates/mlrs-py/src/dispatch.rs:91`.

### Trait surface & additive-retrofit shape (BLDR-02, sets up BLDR-03)
- **D-05: Redefine the canonical traits as typestate-aware** — the eventual *single* surface. `Fit` consumes `self` and exposes an associated `type Fitted`; `Predict` / `Transform` are bound to the fitted type:
  ```rust
  trait Fit<F> { type Fitted; fn fit(self, ..) -> Result<Self::Fitted, AlgoError>; }
  impl<F> Fit<F> for Umap<F, Unfit> { type Fitted = Umap<F, Fitted>; .. }
  impl<F> Predict<F> for Umap<F, Fitted> { .. }   // no impl on Unfit → compile error
  ```
- **D-06: `PartialFit` is consuming and multi-transition** — designed in now even though UMAP/HDBSCAN don't use it, so the Phase-16 retrofit of `IncrementalPCA` / `MBSGDClassifier` / `MBSGDRegressor` has a defined target. `partial_fit(self) -> Result<Self::Fitted>` is implemented on **both** `T<F, Unfit>` (first batch) **and** `T<F, Fitted>` (subsequent batches), so a caller can `predict` between batches and keep streaming. The convention is `Unfit → Fitted → Fitted`, not a binary one-shot.
- **D-07: Introduce the new surface by COEXISTENCE, same names, new module.** Success Criterion 3 requires all 30 existing estimators (still on the old `&mut self` `Fit`/`Predict`/`Transform` in `crates/mlrs-algos/src/traits.rs`) to keep compiling and passing their suites in Phase 12. So redefining in place is forbidden here. Put the new typestate-aware traits in a **new module** (e.g. `mlrs_algos::typestate`), leave `traits.rs` untouched. UMAP/HDBSCAN impl **only** the new surface. Phase 16 migrates each existing estimator old→new under its green suite, then deletes the old `traits.rs` traits at the end — converging to the single redefined surface. Names collide only by path, never at a call site.

### Single-source defaults (BLDR-01)
- **D-08: `new()` is the canonical defaults source.** `T::new()` constructs the struct literal with the sklearn defaults directly, returns `T<F, Unfit>`, and sets `_state: PhantomData`. It trusts defaults as valid (bypasses `build()` validation), so `T::new() == T::builder().build()?` holds by construction. The builder's `impl Default` re-derives from `new()` (e.g. `Umap::new().into_builder()`); `T::builder()` returns `Builder::default()`. `new()` is retained as the zero-arg sklearn-default shortcut per the requirement.
- **D-09: Reuse the shipped builder shape.** Owned chained setters (`fn param(mut self, ..) -> Self`), `build<F>(self) -> Result<T<F, Unfit>, BuildError>` generic-at-build (builder itself non-generic), data-independent hyperparameter validation only — exactly the v2 (Phase 7–11) pattern, e.g. `crates/mlrs-algos/src/linear/mbsgd_regressor.rs`. Phase 12 canonicalizes this shape; it does not invent it.

### UMAP/HDBSCAN shell scope (Success Criterion 4)
- **D-10: Full shape + trivial fit.** Real sklearn param surfaces (UMAP: `n_neighbors`, `min_dist`, `n_components`, `metric`, …; HDBSCAN: `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, …) and real fitted-attr accessors (`embedding_` on fitted UMAP, `labels_` on fitted HDBSCAN), plus a **non-algorithmic real fit body** (set `n_features_in_`, return a zeros embedding / all-noise `-1` labels). This gives a runtime end-to-end round-trip test, not just compilation.
- **D-11: Compile-fail gate is mandatory.** A `trybuild`-style test (or equivalent) proving predict/transform-before-fit fails to compile is the structural proof of BLDR-02.
- **D-12: Module homes.** UMAP introduces a new `manifold/` module under `crates/mlrs-algos/src/` (none exists today); HDBSCAN lands in the existing `cluster/`. PyO3 shells wrap each via the existing `any_estimator!` machinery in `crates/mlrs-py/src/estimators/` (new `manifold.rs`; HDBSCAN can extend `cluster.rs` — planner's call following the existing file layout).

### PyO3 collapse (BLDR-04)
- **D-13: Typestate collapses behind `any_estimator!`.** The Rust `T<F, Unfit>`/`T<F, Fitted>` distinction is invisible to Python: the `Unfit/F32/F64` enum holds raw params in `Unfit`, the `Fitted` monomorphizations in `F32`/`F64`. A runtime `NotFittedError` analog at the Python boundary covers predict-before-fit (the compile-time guarantee is Rust-side only). Reuse the shipped single-site error mappers (`build_err_to_py` / `algo_err_to_py`, D-09 from prior phases).

### Claude's Discretion
- Exact naming/bounds of the `State` sealed trait and marker types; whether the new trait module is `typestate.rs` or another name; whether HDBSCAN's PyO3 shell extends `cluster.rs` or gets its own file — all follow existing structure, planner's call.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — BLDR-01 / BLDR-02 / BLDR-04 (in scope), plus BLDR-03 / SHIM-* (Phase 16 context the convention must remain retrofit-able to)
- `.planning/ROADMAP.md` § "Phase 12" — goal, depends-on (Phase 11), four Success Criteria

### Existing machinery this phase builds on / must not break
- `crates/mlrs-algos/src/traits.rs` — the OLD `Fit`/`PartialFit`/`Predict`/`Transform` (`&mut self`) surface; **stays untouched in Phase 12** (D-07), deleted only at end of Phase 16
- `crates/mlrs-algos/src/error.rs` — `BuildError` (data-independent, line ~384) and `AlgoError::NotFitted`; the D-08 build/fit validation split (`thiserror` in libs)
- `crates/mlrs-py/src/dispatch.rs:91` — `any_estimator!` macro (`Unfit/F32/F64` enum); D-04/D-13 fitted-arm change lands here
- `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` — canonical v2 builder example (owned setters + `build<F>() -> Result<_, BuildError>`); template for D-09
- `crates/mlrs-algos/src/decomposition/incremental_pca.rs` — `PartialFit` consumer; the multi-transition target for D-06 (with `MBSGDClassifier`/`MBSGDRegressor`)

### Project conventions
- `AGENTS.md` §2 — tests separated from source (never in-source `#[cfg(test)] mod tests`; tests live in `crates/*/tests/`)
- `.planning/codebase/CONVENTIONS.md` — coding conventions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **v2 builder pattern** (`mbsgd_regressor.rs` et al.): owned chained setters + `build<F>() -> Result<T<F>, BuildError>` already shipped on Phase 7–11 estimators — Phase 12 canonicalizes this, does not reinvent it.
- **`BuildError` / `AlgoError` split** (`error.rs`): data-independent (build) vs data-dependent (fit) validation already exists (D-08 prior). `AlgoError::NotFitted` is the runtime analog reused at the PyO3 boundary.
- **`any_estimator!` macro** (`dispatch.rs:91`): the `Unfit/F32/F64` dtype-dispatch enum — the PyO3 collapse target (BLDR-04); fitted arms need the explicit `Fitted` marker (D-04).
- **Single-site PyO3 error mappers** (`build_err_to_py` / `algo_err_to_py`): reused, not duplicated.

### Established Patterns
- Today fit is `fit(&mut self) -> Result<&mut Self, AlgoError>` with `Option<...>` fitted fields + runtime `NotFitted` (e.g. `incremental_pca.rs`, `truncated_svd.rs`, `ridge.rs`). The new convention REPLACES this shape — but only on the new module/new estimators in Phase 12; the old shape stays live on the 30 existing estimators until Phase 16.
- Estimators are generic `<F: Float + CubeElement + Pod>`; the state param `S` is additive to that.

### Integration Points
- New `mlrs_algos::typestate` module (markers + new traits) — consumed by the UMAP/HDBSCAN shells now, by all estimators in Phase 16.
- New `manifold/` module in `mlrs-algos`; `cluster/` extended for HDBSCAN.
- `any_estimator!` invocation per shell in `mlrs-py`.

</code_context>

<specifics>
## Specific Ideas

- Concrete shape locked in discussion:
  ```rust
  pub struct Umap<F, S = Unfit> { /* hyperparams + Option fitted fields */ _state: PhantomData<S> }
  impl<F> Umap<F, Unfit>  { pub fn fit(self, ..) -> Result<Umap<F, Fitted>, AlgoError> }
  impl<F> Umap<F, Fitted> { pub fn predict(&self, ..) -> .. }   // transform for UMAP, predict-style for HDBSCAN
  impl Umap<F> { pub fn new() -> Self { /* sklearn defaults */ } pub fn builder() -> UmapBuilder { UmapBuilder::default() } }
  impl Default for UmapBuilder { fn default() -> Self { Umap::new().into_builder() } }
  ```
- Fitted-attr accessors gated at compile time on `T<F, Fitted>` (D-02).

</specifics>

<deferred>
## Deferred Ideas

- **Builder/typestate boilerplate generator** (a `derive`/declarative macro to emit the per-estimator builder + state-param + impl blocks across all 30 estimators) — raised as a consideration, NOT decided. Belongs with the Phase-16 retrofit-sweep planning where the boilerplate cost is realized; the researcher may evaluate it. Phase 12 hand-writes the two shells to fix the convention first.
- **Old-trait deletion / final single-surface convergence** — happens at the END of Phase 16, not here. Phase 12 deliberately leaves both surfaces live (D-07).

### Reviewed Todos (not folded)
None — no pending todos matched this phase.

</deferred>

---

*Phase: 12-Builder + Typestate Convention Foundation*
*Context gathered: 2026-06-23*
