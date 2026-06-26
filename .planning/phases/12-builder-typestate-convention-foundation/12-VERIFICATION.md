---
phase: 12-builder-typestate-convention-foundation
verified: 2026-06-23T02:10:44Z
status: passed
score: 4/4
overrides_applied: 0
human_verification_resolved:
  resolved: 2026-06-26
  test: "Live PyO3 estimator checks for PyUMAP and PyHDBSCAN"
  how: |
    The "no maturin/pyarrow host" assumption no longer held in this run — PyPI was reachable,
    so a venv was provisioned (maturin 1.14, pyarrow 24, numpy 2.5, sklearn 1.9), the cpu wheel
    was built (`maturin develop -m crates/mlrs-py/Cargo.toml --features cpu,extension-module`,
    exit 0, mlrs-cpu-0.1.0 installed editable), and the live UAT script
    (scratchpad/uat_12_live_ffi.py) exercised the real interpreter + pyarrow capsule path.
  result: |
    ALL 22 assertions PASS for BOTH dtype arms (f32 + f64). UMAP fit → embedding_ (75, 2) finite,
    fit_transform, transform(X_new) (10, 2), same-random_state reproducibility; HDBSCAN fit →
    labels_ (75,) int32 in {-1..k}, probabilities_ ∈ [0,1]; unfit accessors raise NotFittedError;
    BuildError::InvalidMinDist (UMAP min_dist>spread) and InvalidMinClusterSize (HDBSCAN
    min_cluster_size<2) surface as Python ValueError with the expected messages. The concrete
    PyValueError class and real capsule ingress are now verified end-to-end. See 12-UAT.md (passed).
    NOTE: the original "zeros shell / all -1" expectation reflected the pre-algorithm Phase-12
    shells; UMAP (Phase 14) and HDBSCAN (Phase 15) now do real work, so the live test asserts
    real embeddings/labels — strictly stronger than the shell gate.
---

# Phase 12: Builder + Typestate Convention Foundation — Verification Report

**Phase Goal:** Establish the idiomatic Rust-native estimator-construction convention — a shared owned-builder + fit/unfit typestate + typed validation error surface — so the v3 estimators (UMAP/HDBSCAN) are born builder-fronted and the later retrofit has a single target shape. Pure API foundation; no algorithm, no device work.
**Verified:** 2026-06-23T02:10:44Z (live-FFI item resolved 2026-06-26)
**Status:** passed
**Re-verification:** No — initial verification; the single `human_needed` item (live PyO3 FFI) was resolved 2026-06-26 via a maturin+pyarrow wheel build + live UAT (all 22 assertions pass, both dtype arms — see frontmatter `human_verification_resolved` and 12-UAT.md)

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Developer can construct via `T::builder().param(..).build() -> Result<T<Unfit>, BuildError>` with owned chained setters, typed thiserror variants, and `T::builder().build()? == T::new()` (single-source defaults) | VERIFIED | `UmapBuilder` and `HdbscanBuilder` exist with owned `fn name(mut self, v) -> Self` setters; `build::<F>()` validates data-independent params and returns `Result<T<Unfit>, BuildError>`. `defaults_equal` tests pass for both shells: `T::new().hyperparams_eq(&T::builder().build::<F>()?)`. `cargo test --test umap_test --test hdbscan_test --features cpu` — 8/8 pass. |
| 2 | Fit/unfit distinction modeled at compile time (`T<Unfit>` → `T<Fitted>`); predict/transform/fitted-attr accessors exist only on fitted type; predict-before-fit fails to compile | VERIFIED | `mlrs_algos::typestate` sealed `State` trait + `Unfit`/`Fitted` ZST markers exist (194 lines, substantive). `impl Fit<F> for Umap<F, Unfit>` with `type Fitted = Umap<F, Fitted>` confirmed. `embedding`/`n_features_in`/`Transform` exist only on `impl Umap<F, Fitted>`. `labels`/`n_features_in` only on `impl Hdbscan<F, Fitted>`. Compile-fail gate passes: `cargo test --test compile_fail --features cpu` — both ui fixtures (E0277 transform-on-Unfit, E0308 Fitted-value-requires-Fitted) fail to compile; goldens explicitly reference `Unfit`. |
| 3 | PyO3 surface unchanged: Rust typestate collapses behind existing `any_estimator!` `Unfit/F32/F64` enum; runtime NotFittedError analog at Python boundary; every existing `any_estimator!` call site still compiles and passes its suite | VERIFIED | Second additive macro `any_estimator_typestate!` added; `dispatch.rs` now defines BOTH macros (count=2); original `any_estimator!` body byte-for-byte unchanged (commit 547b146 is `+63/-0`). 30 legacy `any_estimator!` invocations unchanged. `cargo test -p mlrs-py --features cpu` — all test binaries pass: allocator_test (3), ingress_test (7), probe_test (4), pyclass_smoke_test (4), sgd_smoke_test (3), spectral_smoke_test (2), manifold_test (2). `not_fitted("umap", "embedding_")` runtime analog verified by `not_fitted_before_fit` test. PyUMAP and PyHDBSCAN registered in `lib.rs`. |
| 4 | Convention demonstrated end-to-end on UMAP/HDBSCAN shells so Phases 14–15 inherit it from birth | VERIFIED | `Umap<F, S=Unfit>` in new `manifold/` module (461 lines); `Hdbscan<F, S=Unfit>` in existing `cluster/` module (346 lines). Both: owned-setter builder, single-source defaults, consuming `Fit::fit`, trivial non-algorithmic fit (zeros embedding / all-`-1` labels), Fitted-only accessors, BuildError validation at build(). PyO3 shells `PyUMAP` (295 lines) and `PyHDBSCAN` wrap each via `any_estimator_typestate!`. All 8 algos tests pass and both manifold_test cases pass. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-algos/src/typestate.rs` | Sealed State + Unfit/Fitted markers + consuming Fit/Predict/Transform/PartialFit traits | VERIFIED | 194 lines; contains `pub trait State`, `pub struct Unfit`, `pub struct Fitted`, `pub trait Fit` with `type Fitted`, `pub trait Predict`, `pub trait Transform`, `pub trait PartialFit` with `type Fitted` |
| `crates/mlrs-algos/src/lib.rs` | `pub mod typestate;` registered, no glob re-export | VERIFIED | Line 64: `pub mod typestate;`; deliberate non-glob comment present (D-07); `pub mod manifold;` at line 54 |
| `crates/mlrs-algos/Cargo.toml` | `trybuild = "1.0.117"` in `[dev-dependencies]` | VERIFIED | Line 41: `trybuild = "1.0.117"`; `cargo tree -e dev | grep trybuild` confirms `trybuild v1.0.117` |
| `crates/mlrs-algos/tests/typestate_test.rs` | Runtime smoke: ZST markers compose, module importable | VERIFIED | 3/3 tests pass: `markers_are_zero_sized`, `markers_satisfy_sealed_state_bound`, `typestate_module_is_importable` |
| `crates/mlrs-algos/src/manifold/umap.rs` | `Umap<F,S>` + `UmapBuilder` + new/builder/into_builder + Fit + Transform + fitted accessors | VERIFIED | 461 lines; `pub struct Umap`; `impl Fit<F> for Umap<F, Unfit>` with `type Fitted = Umap<F, Fitted>`; `embedding`/`n_features_in`/`Transform` only on `Umap<F, Fitted>` |
| `crates/mlrs-algos/src/cluster/hdbscan.rs` | `Hdbscan<F,S>` + `HdbscanBuilder` + new/builder/into_builder + Fit + labels accessor | VERIFIED | 346 lines; `pub struct Hdbscan`; `impl Fit<F> for Hdbscan<F, Unfit>` with `type Fitted = Hdbscan<F, Fitted>`; `labels`/`n_features_in` only on `Hdbscan<F, Fitted>`; no Predict/Transform (labels-only, correct) |
| `crates/mlrs-algos/src/error.rs` | `BuildError::InvalidMinDist` + `BuildError::InvalidMinClusterSize` | VERIFIED | Lines 557 and 572; grep count==2; both used in build() validation and exercised by build_rejects tests |
| `crates/mlrs-algos/tests/umap_test.rs` | defaults_equal, build_rejects, fit_roundtrip, fit_no_leak | VERIFIED | 4/4 pass: `defaults_equal`, `build_rejects_bad_min_dist`, `fit_roundtrip`, `fit_no_leak` |
| `crates/mlrs-algos/tests/hdbscan_test.rs` | defaults_equal, build_rejects, fit_roundtrip, fit_no_leak | VERIFIED | 4/4 pass: `defaults_equal`, `build_rejects_bad_min_cluster_size`, `fit_roundtrip`, `fit_no_leak` |
| `crates/mlrs-algos/tests/compile_fail.rs` | trybuild harness over tests/ui/*.rs | VERIFIED | Contains `#[test] fn ui()` → `trybuild::TestCases::new().compile_fail("tests/ui/*.rs")` |
| `crates/mlrs-algos/tests/ui/transform_before_fit.rs` | Fixture: transform on Umap<f32, Unfit> must NOT compile | VERIFIED | Asserts `Transform<f32>` bound on `Umap<f32, Unfit>`; E0277 diagnostic |
| `crates/mlrs-algos/tests/ui/transform_before_fit.stderr` | Golden mentioning `Unfit` | VERIFIED | Contains "Umap<f32, Unfit>", "found `Unfit`", "expected `Fitted`" |
| `crates/mlrs-algos/tests/ui/predict_before_fit.rs` | Fixture: fitted-accessor state on Unfit must NOT compile | VERIFIED | Passes `Umap<f32, Unfit>` where `Umap<f32, Fitted>` required; E0308 diagnostic |
| `crates/mlrs-algos/tests/ui/predict_before_fit.stderr` | Golden mentioning `Unfit` | VERIFIED | Contains "found struct `Umap<f32, Unfit>`", "expected `Umap<f32, Fitted>`" |
| `crates/mlrs-py/src/dispatch.rs` | Second additive macro `any_estimator_typestate!` | VERIFIED | Both macros defined (count=2); fitted arms spell `<f32, mlrs_algos::typestate::Fitted>` / `<f64, ...::Fitted>`; original `any_estimator!` body byte-for-byte unchanged |
| `crates/mlrs-py/src/estimators/manifold.rs` | `PyUMAP` via `any_estimator_typestate!` — consuming fit, embedding_ accessor, not_fitted analog, unfit_default/is_unfit | VERIFIED | 295 lines; invokes `crate::any_estimator_typestate!`; `unfit_default()`/`is_unfit()`; consuming `fit` with `py.detach` + `lock_pool` + `guard_f64`; `embedding_f32`/`embedding_f64`; `not_fitted` on Unfit arm |
| `crates/mlrs-py/src/estimators/cluster.rs` | `PyHDBSCAN` via `any_estimator_typestate!` — labels_ accessor, not_fitted analog | VERIFIED | `any_estimator_typestate!` invoked for `AnyHdbscan`; `PyHDBSCAN` struct with `unfit_default()`/`is_unfit()`/`labels_for_test()`; `labels_` accessor with `not_fitted` on Unfit arm; TypestateFit alias resolves Fit name collision |
| `crates/mlrs-py/tests/manifold_test.rs` | unfit_default smoke + not_fitted runtime analog | VERIFIED | 2/2 pass: `typestate_shells_construct_unfit`, `not_fitted_before_fit` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/mlrs-algos/src/lib.rs` | `crates/mlrs-algos/src/typestate.rs` | `pub mod typestate` | WIRED | Line 64; module compiles and is importable |
| `crates/mlrs-algos/src/manifold/umap.rs` | `mlrs_algos::typestate::Fit` | `impl<F> Fit<F> for Umap<F, Unfit> { type Fitted = Umap<F, Fitted>; }` | WIRED | Line 355 of umap.rs; `use crate::typestate::{Fit, ...}` at line 39 |
| `crates/mlrs-algos/src/cluster/hdbscan.rs` | `mlrs_algos::typestate::Fit` | `impl<F> Fit<F> for Hdbscan<F, Unfit> { type Fitted = Hdbscan<F, Fitted>; }` | WIRED | Line 277 of hdbscan.rs; `use crate::typestate::{Fit, ...}` at line 44 |
| `crates/mlrs-algos/src/manifold/umap.rs` | `crate::error::BuildError::InvalidMinDist` | `build()` validation | WIRED | `if !self.min_dist.is_finite() || self.min_dist > self.spread { return Err(BuildError::InvalidMinDist {...}) }`; test `build_rejects_bad_min_dist` PASS |
| `crates/mlrs-algos/tests/compile_fail.rs` | `crates/mlrs-algos/tests/ui/*.rs` | trybuild glob `compile_fail("tests/ui/*.rs")` | WIRED | `cargo test --test compile_fail --features cpu` passes; both ui fixtures confirmed non-compiling |
| `crates/mlrs-py/src/estimators/manifold.rs` | `any_estimator_typestate!` | macro invocation producing AnyUmap with F32(Umap<f32, Fitted>) | WIRED | `crate::any_estimator_typestate! { any: AnyUmap, algo: mlrs_algos::manifold::umap::Umap, ... }` |
| `crates/mlrs-py/src/estimators/manifold.rs` | `crate::errors::not_fitted` | Unfit-arm accessor → `not_fitted("umap", "embedding_ (f32)")` | WIRED | `_ => Err(not_fitted("umap", "embedding_ (f32)"))` in `embedding_f32_inner`; test `not_fitted_before_fit` PASS |
| `crates/mlrs-py/src/lib.rs` | `PyUMAP, PyHDBSCAN` | `m.add_class` | WIRED | Lines 265-266: `m.add_class::<PyUMAP>()?` and `m.add_class::<PyHDBSCAN>()?` |

### Data-Flow Trace (Level 4)

Not applicable for this phase. The estimator shells produce intentional non-algorithmic (zeros/all-`-1`) data flows by design; the trivial fit bodies are the complete implementation of Phase 12 SC4. Real data flows land in Phases 14 (UMAP) and 15 (HDBSCAN). The `fit_roundtrip` tests verify the non-algorithmic shells produce the expected trivial output (zeros embedding, all-`-1` labels) — this is the correct behavior for Phase 12, not a stub to flag.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| typestate_test: ZST markers + sealed bound | `cargo test -p mlrs-algos --features cpu --test typestate_test` | 3 passed | PASS |
| umap_test: defaults_equal, build_rejects, fit_roundtrip, fit_no_leak | `cargo test -p mlrs-algos --features cpu --test umap_test` | 4 passed | PASS |
| hdbscan_test: defaults_equal, build_rejects, fit_roundtrip, fit_no_leak | `cargo test -p mlrs-algos --features cpu --test hdbscan_test` | 4 passed | PASS |
| compile_fail: predict/transform-before-fit fails to compile, goldens match | `cargo test -p mlrs-algos --features cpu --test compile_fail` | 1 passed (2 ui fixtures ok) | PASS |
| mlrs-py full suite: new shells + all existing estimators green | `cargo test -p mlrs-py --features cpu` | All test binaries pass; manifold_test (2), allocator_test (3), ingress_test (7), probe_test (4), pyclass_smoke_test (4), sgd_smoke_test (3), spectral_smoke_test (2) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| BLDR-01 | Plans 12-01, 12-02 | Builder convention with owned chained setters, typed BuildError, single-source defaults | SATISFIED (convention demonstrated on UMAP/HDBSCAN shells; full 30-estimator retrofit deferred to Phase 16 / BLDR-03 per roadmap design) | `T::builder().build()? == T::new()` confirmed by defaults_equal tests; owned setters verified in umap.rs and hdbscan.rs; typed thiserror BuildError variants verified |
| BLDR-02 | Plans 12-01, 12-03 | Compile-time typestate — predict-before-fit fails to compile | SATISFIED | Sealed State + Unfit/Fitted markers + consuming Fit trait + compile-fail gate passing |
| BLDR-04 | Plan 12-04 | PyO3 surface unchanged; typestate collapses behind any_estimator!; runtime NotFittedError | SATISFIED | Second additive macro; original any_estimator! unchanged; 35 legacy call sites green; not_fitted test passes |

Note: REQUIREMENTS.md marks BLDR-01 as "Pending" in the traceability table (not a checkbox checked). This reflects that the broad "any estimator" retrofit is still pending (BLDR-03, Phase 16). Phase 12's contract is the *convention* for BLDR-01, not the full sweep — the ROADMAP note explicitly states: "the builder convention (BLDR-01/02/04) leads in Phase 12 so UMAP/HDBSCAN are born builder-fronted; the broad, parallel-unsafe 30-estimator retrofit sweep (BLDR-03) is isolated to Phase 16." SC-1 is demonstrably satisfied within Phase 12's scope.

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `crates/mlrs-algos/src/manifold/umap.rs` | Non-algorithmic fit body ("NON-algorithmic trivial fit... real UMAP lands in Phase 14") | INFO | Intentional shell — documented in-source; Phase 14 fills the body. Not a blocking stub. |
| `crates/mlrs-algos/src/cluster/hdbscan.rs` | Non-algorithmic fit body ("NON-algorithmic trivial fit... real HDBSCAN lands in Phase 15") | INFO | Intentional shell — documented in-source; Phase 15 fills the body. Not a blocking stub. |

No TBD, FIXME, or XXX markers found in any Phase 12 files. No unresolved debt markers. The intentional "non-algorithmic shell" comments reference specific future phases (14/15) and are the design intent of Phase 12 SC4, not completion gaps.

`traits.rs` frozen invariant: `git diff HEAD -- crates/mlrs-algos/src/traits.rs` produces no output — the file is byte-for-byte unchanged as required.

### Human Verification Required

#### 1. Live PyO3 Estimator Integration (UMAP + HDBSCAN)

**Test:** In an environment with maturin + pyarrow installed, build the mlrs wheel (`maturin develop --features cpu`) and run:
```python
import numpy as np
import mlrs._mlrs as _m

X = np.random.rand(10, 4).astype(np.float32)
arr = _make_arrow_capsule(X)  # using the project's capsule helper

umap = _m.UMAP()
umap.fit(arr, 10, 4)
embedding = umap.embedding_f32()
assert len(embedding) == 10 * 2  # n * n_components zeros shell

hdbscan = _m.HDBSCAN()
hdbscan.fit(arr, 10, 4)
labels = hdbscan.labels_()
assert labels == [-1] * 10  # all-noise shell

# NotFittedError analog
umap2 = _m.UMAP()
try:
    umap2.embedding_f32()
    assert False, "should have raised"
except Exception as e:
    assert "ValueError" in type(e).__name__ or "NotFitted" in str(e)

# BuildError validation
umap3 = _m.UMAP(min_dist=2.0, spread=1.0)
try:
    umap3.fit(arr, 10, 4)
    assert False
except Exception as e:
    assert "min_dist" in str(e).lower()
```
**Expected:** All assertions pass; the concrete Python exception is `PyValueError` (the `not_fitted`/`build_err_to_py` mapper produces a Python ValueError type the shim re-raises as NotFittedError).
**Why human (original):** No maturin or pyarrow in this environment (MEMORY "Python wheel untestable in env" / SHIM-03). The Rust-side gates — consuming fit, build()/guard_f64() chain, not_fitted runtime analog — are all verified by cargo tests without an interpreter. The live capsule FFI path and concrete Python exception class assertion require UAT.

**RESOLVED 2026-06-26:** The host assumption no longer held — PyPI was reachable, so maturin+pyarrow were installed in a venv, the cpu wheel was built (`maturin develop`, exit 0), and the live UAT script ran the real interpreter + pyarrow capsule path: all 22 assertions pass for both dtype arms (UMAP embedding_/fit_transform/transform/reproducibility; HDBSCAN labels_/probabilities_; unfit→NotFittedError; BuildError→ValueError with expected messages). See frontmatter `human_verification_resolved` and 12-UAT.md (status: complete, result: passed).

---

## Deferred Items

None. All four phase success criteria are verified. The algorithmic fit bodies for UMAP (Phase 14) and HDBSCAN (Phase 15) and the full builder retrofit sweep (BLDR-03, Phase 16) are later-phase work explicitly scoped out of Phase 12.

---

_Verified: 2026-06-23T02:10:44Z_
_Verifier: Claude (gsd-verifier)_
