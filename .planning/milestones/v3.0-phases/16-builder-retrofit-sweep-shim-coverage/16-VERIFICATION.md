---
phase: 16-builder-retrofit-sweep-shim-coverage
verified: 2026-06-25T04:15:00Z
status: verified
human_verification_resolved: 2026-06-26
score: 4/4
behavior_unverified: 0
overrides_applied: 0
human_verification_outcome: "RESOLVED 2026-06-26 via /gsd-verify-work 16 (16-UAT.md test 1, PASS). The full umap_test oracle suite ran to completion: `test result: ok. 35 passed; 0 failed; 0 ignored; finished in 14237.04s` (exit 0). All 5 transform_property_* out-of-sample gates, all 5 layout_property_*, all 5 spectral_init_*, reproducible_f64, fit_roundtrip, and fit_no_leak passed. Truth 1 is now fully VERIFIED (no longer PRESENT_BEHAVIOR_UNVERIFIED) — the typestate convergence did not perturb any UMAP numeric path."
human_verification:
  - test: "Run the full umap_test oracle suite to confirm UMAP's fit/transform/fit_transform gates still pass after the typestate convergence"
    expected: "All umap_test tests pass (property gates: trustworthiness, kNN-overlap >= umap-learn - margin, same random_state reproducibility, transform sub-gate on new points)"
    status: RESOLVED 2026-06-26 — 35/35 passed (see human_verification_outcome)
    why_human: "umap_test was launched but not completed during phase execution due to exceptionally slow runtime under CPU contention (>25 min). The compile_fail trybuild gate proves the UMAP typestate surface is correct (Unfit arm does not satisfy Transform<f32>; Fitted arm does), but the 1e-5/property-gate oracle run is a behavior-dependent truth that requires the full test execution, not just symbol presence."
behavior_unverified_items:
  - truth: "Every shipped 1e-5 / exact-label gate still passes after the retrofit — specifically for UMAP (stochastic property gate + transform sub-gate)"
    test: "cargo test --features cpu --test umap_test"
    expected: "All umap_test tests pass including property gates (trustworthiness, kNN-overlap, reproducibility) and transform gate on out-of-sample points"
    status: RESOLVED 2026-06-26 — 35/35 passed, 0 failed (finished in 14237.04s, exit 0) via /gsd-verify-work 16
    why_human: "umap_test was launched during phase execution but is exceptionally slow (~25+ min under CPU contention per the 'backend test suite slow' memory entry). The compile_fail trybuild gate proves the UMAP typestate surface is structurally correct, but does not exercise the numeric property gates. All other 12 oracle suites ran and passed; this is the only unsettled numeric gate."
---

# Phase 16: Builder Retrofit Sweep + Shim Coverage — Verification Report

**Phase Goal:** Retrofit the Phase-12 builder + typestate convention ADDITIVELY across all existing estimators (builder constructs the existing config struct; fit path untouched), piloted on 1-2 estimators under the green suite before the full sweep, preserving every shipped 1e-5 / exact-label gate; and complete the pure-Python sklearn shim (get_params/set_params/clone round-trip; coverage extended to all estimators + UMAP/HDBSCAN; static Python check). Plus the two new PyO3 wrap method gaps (SHIM-02).

**Verified:** 2026-06-25T04:15:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Builder + typestate convention retrofitted additively across ALL existing estimators; new() kept as thin zero-arg wrapper; fit path untouched; piloted first; every shipped gate still passes | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED (UMAP oracle) | 32 estimators confirmed with `<F, S = Unfit>` state param; `crate::traits` references empty across both crates' src; `traits.rs` deleted; Ridge, PCA, KMeans, NearestNeighbors, GaussianNB, KernelRidge, HDBSCAN oracle suites all pass; UMAP compile_fail gate green; UMAP full oracle timed out (see behavior_unverified_items) |
| 2 | Every estimator's pure-Python class stores each constructor arg unchanged in __init__ (no validation/computation) and exposes get_params/set_params that round-trip exactly and are clone()-compatible (coverage extended to all 18 + UMAP/HDBSCAN) | ✓ VERIFIED | test_shims.py + test_params.py: 255 passed in shimvenv (no compiled _mlrs); 32-entry EXPECTED_PARAMS confirmed covering UMAP+HDBSCAN; AST-based __init__-purity test (`test_init_purity_ast`) exists and is green over all 32 shims; `import ast` present in test_params.py |
| 3 | UMAP and HDBSCAN PyO3-wrapped with sklearn-named params, trailing-underscore fitted attrs, n_features_in_ enforced, fit returns self, correct surface (UMAP transform/fit_transform; HDBSCAN fit_predict/labels_) | ✓ VERIFIED | `PyUMAP` (#[pyclass]) in manifold.rs with `transform_f32/f64`, `fit_transform_f32/f64`, `embedding_f32/f64` methods; `PyHDBSCAN` (#[pyclass]) in cluster.rs with `fit`, `fit_predict`, `labels_()`, `probabilities_()`, `outlier_scores_()` methods; both registered via `m.add_class` in lib.rs (lines 265-266); `_post_fit(cols)` called by Python UMAP.fit and HDBSCAN.fit (sets `n_features_in_`); GIL release via `py.detach()`; `guard_f64()` before F64 uploads; `any_estimator_typestate!` macro used; `cargo build -p mlrs-py --features cpu` clean |
| 4 | The shim is verified by Rust-side unit tests plus a static Python check; the live estimator_checks/check_estimator run stays DEFERRED by design (needs maturin+pyarrow host this environment lacks) — do NOT count this deferral as a gap; verify it is honestly documented | ✓ VERIFIED | typestate_test: 6 passed (including `new_accessor_traits_resolve_on_fitted_marker`, `transform_inverse_transform_default_returns_unsupported`); compile_fail trybuild: `ui ok (1 passed)` — predict_before_fit.rs and transform_before_fit.rs both confirmed compile-fail; Python static gate: 255 passed (test_shims.py + test_params.py) + 82 passed (fit-free estimator_checks: check_no_attributes_set_in_init, check_parameters_default_constructible, check_get_params_invariance) + 1278 deselected (by-design deferred fit-based checks); VALIDATION.md §UAT row explicitly documents the live check_estimator as deferred; 16-12-SUMMARY.md records the deferred UAT note |

**Score:** 4/4 truths verified (1 present + behavior-unverified on UMAP full oracle)

*Note: Truth 1 is classified as PRESENT_BEHAVIOR_UNVERIFIED rather than VERIFIED because the UMAP numeric property gate (a behavior-dependent truth requiring the full oracle run) was not completed during phase execution. All other oracle suites (12/13 targeted suites) ran and passed. The UMAP typestate surface is proven correct by the compile_fail trybuild gate.*

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-algos/src/typestate.rs` | 9 traits (4 pre-existing + 5 added: PredictLabels, KNeighbors, ScoreSamples, PredictProba, PredictLogProba) + Transform::inverse_transform default | ✓ VERIFIED | `grep -c 'pub trait PredictLabels\|...'` returns 5; `grep -n 'fn inverse_transform'` returns line 206; `cargo build -p mlrs-algos --features cpu` clean |
| `crates/mlrs-algos/src/traits.rs` | HARD-DELETED (D-01) | ✓ VERIFIED | `test ! -f crates/mlrs-algos/src/traits.rs` passes; file does not exist |
| `crates/mlrs-algos/src/lib.rs` | `pub mod traits;` removed; `pub mod typestate;` retained; `pub use error::AlgoError` retained | ✓ VERIFIED | `grep 'pub mod typestate'` → line 64; no `pub mod traits` found; `pub use error::AlgoError` at line 70 |
| `crates/mlrs-algos/tests/compile_fail.rs` | trybuild harness asserting predict-before-fit does not compile | ✓ VERIFIED | File exists; `cargo test --features cpu --test compile_fail` → `ui ok (1 passed)` |
| `crates/mlrs-algos/tests/ui/predict_before_fit.rs` + `.stderr` | Fixture proving Umap<f32, Unfit> does not satisfy Transform<f32> (E0277) | ✓ VERIFIED | Both files exist; stderr names `Unfit` and `Fitted` explicitly; test green |
| `crates/mlrs-algos/tests/typestate_test.rs` | Compile/build proof for 5 new traits + inverse_transform default | ✓ VERIFIED | `cargo test --features cpu --test typestate_test` → 6 passed |
| `crates/mlrs-py/src/estimators/manifold.rs` | PyUMAP #[pyclass] with transform/fit_transform surface | ✓ VERIFIED | `transform_f32/f64`, `fit_transform_f32/f64`, `embedding_f32/f64` methods present with real compute (calls `Transform::transform` and returns device buffer to host) |
| `crates/mlrs-py/src/estimators/cluster.rs` | PyHDBSCAN #[pyclass] with fit/fit_predict/labels_/probabilities_/outlier_scores_ | ✓ VERIFIED | All methods present; `fit_predict` calls `self.fit()` then returns `labels_`; `labels_inner()` shared across `labels_()` and `fit_predict` |
| `crates/mlrs-py/python/mlrs/manifold.py` | UMAP shim class with faithful __init__, fit, transform, fit_transform | ✓ VERIFIED | `class UMAP(TransformerMixin, MlrsBase)` with 16 pure verbatim assignments in `__init__`, `fit` returns `self`, `transform` calls `_suffixed("transform")`, `fit_transform` calls `self.fit` then returns `self.embedding_` |
| `crates/mlrs-py/python/mlrs/cluster.py` | HDBSCAN shim class with faithful __init__, fit, labels_ property | ✓ VERIFIED | `class HDBSCAN(ClusterMixin, MlrsBase)` with 8 pure verbatim assignments in `__init__`, `fit` returns `self` via `_post_fit` + `return self`, `labels_` property via `_mlrs_obj.labels_()` |
| `crates/mlrs-py/python/tests/test_params.py` | AST-based __init__-purity test; UMAP/HDBSCAN in matrix | ✓ VERIFIED | `import ast` at line 18; `test_init_purity_ast` at line 377; EXPECTED_PARAMS has 32 entries including UMAP and HDBSCAN |
| `crates/mlrs-py/python/tests/test_shims.py` | 32-shim matrix; UMAP/HDBSCAN in ALL_SHIMS | ✓ VERIFIED | `ALL_SHIMS = _exported_shim_names()` auto-discovers; UMAP + HDBSCAN tested for mixin membership, method surface, and `labels_`/`embedding_` fitted attrs |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| All estimator `*.rs` in `crates/mlrs-algos/src/` | `crates/mlrs-algos/src/typestate.rs` | `use crate::typestate::{Fit, Fitted, Predict, Unfit, ...}` | ✓ WIRED | Survey across all 8 module directories: every estimator file with `<F, S = Unfit>` uses `crate::typestate` imports; zero `use crate::traits` references remain in any source file |
| `crates/mlrs-algos/src/lib.rs` | `crates/mlrs-algos/src/typestate.rs` | `pub mod typestate;` (line 64) | ✓ WIRED | typestate is the single re-exported trait module |
| `crates/mlrs-py/src/lib.rs` | `crates/mlrs-py/src/estimators/manifold.rs` + `cluster.rs` | `use estimators::cluster::{..., PyHDBSCAN}` + `use estimators::manifold::PyUMAP` + `m.add_class::<PyUMAP>()?` (line 265) + `m.add_class::<PyHDBSCAN>()?` (line 266) | ✓ WIRED | Both classes imported and registered in the Python module |
| `crates/mlrs-py/python/mlrs/manifold.py` | `crates/mlrs-py/src/estimators/manifold.rs` | `self._ext().UMAP(...)` + `self._suffixed("transform")` | ✓ WIRED | Python shim calls PyO3-exposed UMAP constructor and transform methods |
| `crates/mlrs-py/python/mlrs/cluster.py` | `crates/mlrs-py/src/estimators/cluster.rs` | `self._ext().HDBSCAN(...)` + `self._mlrs_obj.labels_()` | ✓ WIRED | Python shim calls PyO3-exposed HDBSCAN constructor and labels_ accessor |
| `crates/mlrs-py/python/mlrs/base.py` | Python shim `fit` methods | `_post_fit(cols)` sets `n_features_in_` (line 158) | ✓ WIRED | Every fit() method in UMAP and HDBSCAN shims calls `_post_fit(cols)` |
| `crates/mlrs-algos/tests/compile_fail.rs` | `crates/mlrs-algos/tests/ui/*.rs` | `t.compile_fail("tests/ui/*.rs")` (trybuild) | ✓ WIRED | predict_before_fit.rs and transform_before_fit.rs both exercised; test result: `ui ok (1 passed)` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `manifold.py UMAP.fit` | `self._mlrs_obj` | `self._ext().UMAP(...)` → `obj.fit(xa, rows, cols)` → `PyUMAP::fit` → `TypestateFit::fit(est, ...)` → actual UMAP compute | Yes — device fit returns `Umap<F, Fitted>` with real embedding | ✓ FLOWING |
| `manifold.py UMAP.transform` | `out` from `_suffixed("transform")(xa, rows, cols)` | `PyUMAP::transform_f32/f64` → `Transform::transform(est, ...)` → device kernel | Yes — calls device Transform trait on fitted estimator | ✓ FLOWING |
| `cluster.py HDBSCAN.labels_` | return from `self._mlrs_obj.labels_()` | `PyHDBSCAN::labels_()` → `labels_inner()` → device fitted arm | Yes — returns actual cluster labels from fitted HDBSCAN | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Typestate surface is coherent; inverse_transform default works | `cargo test --features cpu --test typestate_test` | 6 passed | ✓ PASS |
| Ridge oracle (pilot A; full build-out) | `cargo test --features cpu --test ridge_test` | 6 passed | ✓ PASS |
| PCA oracle (decomposition module; Transform + inverse) | `cargo test --features cpu --test pca_test` | 11 passed | ✓ PASS |
| HDBSCAN oracle (born-with-convention; 40 tests) | `cargo test --features cpu --test hdbscan_test` | 40 passed | ✓ PASS |
| GaussianNB oracle (naive_bayes module; shape B) | `cargo test --features cpu --test gaussian_nb_test` | 7 passed | ✓ PASS |
| KernelRidge oracle (shape A'; adopted Fit+Predict) | `cargo test --features cpu --test kernel_ridge_test` | 5 passed | ✓ PASS |
| NearestNeighbors oracle (neighbors module; KNeighbors trait) | `cargo test --features cpu --test nearest_neighbors_test` | 5 passed | ✓ PASS |
| Predict-before-fit is a compile error (BLDR-02 regression guard) | `cargo test --features cpu --test compile_fail` | ui ok (1 passed) — predict_before_fit.rs and transform_before_fit.rs do NOT compile; error names Unfit/Fitted | ✓ PASS |
| Python static shim gate (255 checks) | `PYTHONPATH=... pytest tests/test_shims.py tests/test_params.py -q` (shimvenv) | 255 passed in 0.95s | ✓ PASS |
| Fit-free sklearn estimator_checks (82 checks) | `PYTHONPATH=... pytest tests/test_estimator_checks.py -k "check_no_attributes... or fit_free" -q` | 82 passed, 1278 deselected | ✓ PASS |
| UMAP full oracle (property/numeric gates) | umap_test timed out in phase execution | NOT RUN (>25 min under CPU contention) | ⚠️ SKIP — routes to Human Verification |
| Both crates build clean | `cargo build -p mlrs-algos --features cpu` + `cargo build -p mlrs-py --features cpu` | Finished clean (2 pre-existing dead-code warnings in mlrs-py, unrelated to phase 16) | ✓ PASS |

### Probe Execution

No explicit `scripts/*/tests/probe-*.sh` probes declared for this phase. Phase-end gate ran in-plan (Task 2 of Plan 16-12) covering compile_fail + targeted oracle suites + Python static gate.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| BLDR-03 | Plans 16-00 through 16-12 | Builder + typestate convention retrofitted additively across all existing estimators; piloted; gates preserved | ✓ SATISFIED | 32 estimators confirmed with `<F, S = Unfit>`; `traits.rs` hard-deleted; zero `crate::traits` live references; 12/13 targeted oracle suites pass; UMAP compile_fail gate green; UMAP numeric oracle pending human verification |
| SHIM-01 | Plans 16-00, 16-10, 16-11 | Every estimator's pure-Python class stores constructor args verbatim; get_params/set_params round-trip; clone-compatible; extended to all 18 + UMAP/HDBSCAN | ✓ SATISFIED | 32-entry EXPECTED_PARAMS; AST purity test in test_params.py; 255 passed (test_shims.py + test_params.py); UMAP and HDBSCAN both in matrix |
| SHIM-02 | Plan 16-10 | UMAP and HDBSCAN PyO3-wrapped with sklearn-named params, trailing-underscore fitted attrs, n_features_in_ enforced, fit returns self, correct surface | ✓ SATISFIED | PyUMAP + PyHDBSCAN #[pyclass] confirmed with full method surfaces; n_features_in_ set via _post_fit; GIL release; guard_f64 before F64; registered in lib.rs |
| SHIM-03 | Plans 16-00, 16-12 | Shim verified by Rust-side unit tests + static Python check; live check_estimator DEFERRED by design (no maturin+pyarrow host) | ✓ SATISFIED | typestate_test 6 passed; compile_fail 1 passed; Python static 255 + 82 passed; VALIDATION.md §UAT documents deferral; 16-12-SUMMARY.md records deferred-UAT note; test_estimator_checks.py header documents maturin dependency |

**Orphaned requirements check:** REQUIREMENTS.md Traceability table maps BLDR-03, SHIM-01, SHIM-02, SHIM-03 to Phase 16. All four are accounted for.

**REQUIREMENTS.md documentation inconsistency (non-blocking):** The traceability table shows `BLDR-03 | Phase 16 | In Progress` (line 93) while the checkbox on line 39 is `[x]` (checked). This is a documentation-only gap — the code evidence proves BLDR-03 is complete (traits.rs deleted, single trait surface, all targeted suites green). No code gap.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/mlrs-algos/tests/random_projection_test.rs` | 31 | Comment contains string `crate::traits` | ℹ️ Info | Comment-only; not live code; the empty-grep convergence gate excludes doc-comments; no functional issue |

No TBD/FIXME/XXX/HACK/PLACEHOLDER markers found in any Phase 16 modified source files. No stub implementations found. The two pre-existing dead-code warnings in `cargo build -p mlrs-py` are unrelated to Phase 16 (they pre-date the phase per the 16-09 and 16-12 SUMMARY notes).

### Human Verification Required

#### 1. UMAP Full Oracle Suite

**Test:** Run `cargo test --features cpu --test umap_test` from the project root.

**Expected:** All umap_test tests pass:
- `fit_no_leak` (PoolStats memory gate — fit does not leak device buffers)
- Property gates: trustworthiness and kNN-overlap >= umap-learn - margin
- Stochastic reproducibility: same `random_state` → same embedding within mlrs
- `fit_transform` equivalence to `fit` + `embedding_`
- `transform` (out-of-sample) property sub-gate on held-out points
- Any additional tests in the suite

**Why human:** `umap_test` was launched during phase execution but is exceptionally slow (>25 min under CPU contention per the "backend test suite slow" project memory entry). The compile_fail trybuild gate proves the UMAP typestate surface is structurally correct — `Umap<f32, Unfit>` does not satisfy `Transform<f32>`; `Umap<f32, Fitted>` does — but structural correctness does not exercise the numeric property gates (trustworthiness, kNN-overlap, reproducibility). UMAP never used `traits.rs` (born-with-convention), so the deletion cannot have perturbed its numeric path; the verification conservatively requires the full oracle run to confirm no regression.

**Note:** This is the only unsettled gate. 12 of the 13 targeted oracle suites ran and passed during phase execution. The live `check_estimator` FFI run (SHIM-03) is separately and intentionally deferred to UAT — that is by-design and does NOT require human verification as an unknown; it is documented.

---

## Summary

Phase 16 delivered:

1. **Single trait surface (BLDR-03):** `crates/mlrs-algos/src/traits.rs` is hard-deleted. `pub mod traits;` and the crate-root `pub use traits::{...}` re-export are removed from `lib.rs`. The convergence grep across both crates' src (excluding traits.rs) returns EMPTY. `mlrs_algos::typestate` is the only trait surface. All 32 estimators (30 retrofitted this phase, 2 born-with-convention in Phases 12/15) carry `<F, S = Unfit>` + `PhantomData<S>`, consuming-self `Fit<F>`, and Fitted-gated accessor impls. `new()` is zero-arg (sklearn defaults) on all estimators. The pilot sequence was honored (Ridge first, MBSGDRegressor second; KMeans last within the sweep). D-03 additive safety confirmed by spot-checking Ridge, KMeans, and PCA fit bodies — all numeric paths are byte-identical to pre-retrofit.

2. **Python shim completeness (SHIM-01):** 32 pure-Python estimator shims cover all v2/v3 estimators including UMAP and HDBSCAN. Every `__init__` stores constructor args verbatim (enforced by the AST-purity test added in Plan 16-00/16-11). `get_params`/`set_params`/`clone` round-trip is free from `MlrsBase(BaseEstimator)`. 255 static shim tests pass; 82 fit-free sklearn estimator_checks pass.

3. **PyO3 UMAP/HDBSCAN wraps (SHIM-02):** `PyUMAP` and `PyHDBSCAN` are `#[pyclass]`-wrapped via `any_estimator_typestate!`, with GIL release, `guard_f64` before F64, sklearn-named params, trailing-underscore fitted attrs, `n_features_in_` enforced via Python `_post_fit`, `fit` returns `self`, and correct surfaces (UMAP: `transform`/`fit_transform`; HDBSCAN: `fit_predict`/`labels_`). Both are registered in `mlrs-py/src/lib.rs`.

4. **Verification gates (SHIM-03):** Rust-side: typestate_test (6 passed), compile_fail trybuild (predict-before-fit and transform-before-fit do NOT compile; error names Unfit/Fitted). Python-side: 255 static shim tests + 82 fit-free sklearn checks pass. Live `check_estimator` FFI run is honestly documented as DEFERRED to UAT (no maturin+pyarrow host; clearly noted in VALIDATION.md, test_estimator_checks.py header, and 16-12-SUMMARY.md).

**Single unsettled item:** The UMAP full numeric oracle (`umap_test`) did not complete during phase execution due to exceptional runtime. All other 12 targeted oracle suites passed. The UMAP typestate surface is proven structurally correct by the compile_fail gate. A complete run is needed to formally certify the numeric property gates are unperturbed.

---

_Verified: 2026-06-25T04:15:00Z_
_Verifier: Claude (gsd-verifier)_
