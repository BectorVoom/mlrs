---
phase: 12-builder-typestate-convention-foundation
reviewed: 2026-06-23T00:00:00Z
depth: standard
files_reviewed: 22
files_reviewed_list:
  - crates/mlrs-algos/Cargo.toml
  - crates/mlrs-algos/src/cluster/hdbscan.rs
  - crates/mlrs-algos/src/cluster/mod.rs
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/src/lib.rs
  - crates/mlrs-algos/src/manifold/mod.rs
  - crates/mlrs-algos/src/manifold/umap.rs
  - crates/mlrs-algos/src/typestate.rs
  - crates/mlrs-algos/tests/compile_fail.rs
  - crates/mlrs-algos/tests/hdbscan_test.rs
  - crates/mlrs-algos/tests/typestate_test.rs
  - crates/mlrs-algos/tests/ui/predict_before_fit.rs
  - crates/mlrs-algos/tests/ui/predict_before_fit.stderr
  - crates/mlrs-algos/tests/ui/transform_before_fit.rs
  - crates/mlrs-algos/tests/ui/transform_before_fit.stderr
  - crates/mlrs-algos/tests/umap_test.rs
  - crates/mlrs-py/src/dispatch.rs
  - crates/mlrs-py/src/estimators/cluster.rs
  - crates/mlrs-py/src/estimators/manifold.rs
  - crates/mlrs-py/src/estimators/mod.rs
  - crates/mlrs-py/src/lib.rs
  - crates/mlrs-py/tests/manifold_test.rs
findings:
  critical: 0
  warning: 5
  info: 5
  total: 10
status: issues_found
---

# Phase 12: Code Review Report

**Reviewed:** 2026-06-23T00:00:00Z
**Depth:** standard
**Files Reviewed:** 22
**Status:** issues_found

## Summary

This phase delivers the v3 builder + typestate convention foundation: a sealed
`State` marker trait with `Unfit`/`Fitted` markers, a consuming-`self` typestate
`Fit`/`Predict`/`Transform`/`PartialFit` surface, two non-algorithmic estimator
SHELLS (`Umap`, `Hdbscan`) that demonstrate the convention end-to-end, a
`trybuild` compile-fail gate, and the two PyO3 wrappers (`PyUMAP`, `PyHDBSCAN`)
plus a second `any_estimator_typestate!` dispatch macro.

The typestate design is sound and the compile-fail gate is well-reasoned. No
BLOCKER-class correctness or security defects were found: the shells are
deliberately non-algorithmic, the device-side data-dependent geometry guards are
present, the f64 capability guard runs before the f64 upload on both Py wrappers,
and a `rows`/`cols` mismatch from Python is caught as a typed `ShapeMismatch`
rather than panicking in `DeviceArray::from_host`.

The findings below are robustness and consistency gaps that will become live
bugs (not just shell quirks) when Phases 14/15 fill in the real algorithms, plus
an inconsistency in the mutex-lock convention in a file this phase edited.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: `n_components == 0` is accepted, producing a silently-empty embedding

**File:** `crates/mlrs-algos/src/manifold/umap.rs:322-352` (build), `:366-411` (fit), `:443-460` (transform)
**Issue:** `UmapBuilder::build` validates only `min_dist` (finite and `<= spread`).
It never rejects `n_components == 0`. With `n_components = 0`, `fit` allocates
`vec![F::from_int(0i64); n * 0]` = an empty buffer, `embedding()` returns an
empty `Vec`, and `transform` likewise returns an empty buffer — all without
error. sklearn/umap-learn require `n_components >= 1`. This is data-independent
hyperparameter validation that the D-08 split places at `build()`, so it should
be rejected there rather than silently producing a degenerate empty result that
the Phase-14 algorithm must special-case. The same gap exists for
`n_neighbors == 0`, also undefined in umap-learn.
**Fix:** Add a guard in `UmapBuilder::build` before the `Ok(...)`:
```rust
if self.n_components == 0 {
    return Err(BuildError::InvalidNComponents {  // or a new BuildError variant
        estimator: "umap",
        n_components: self.n_components,
    });
}
```
At minimum enforce `n_components >= 1`; consider `n_neighbors >= 1` too.

### WR-02: `Transform::transform` for `Umap<F, Fitted>` ignores fitted `n_features_in_`

**File:** `crates/mlrs-algos/src/manifold/umap.rs:443-460`; trait doc `crates/mlrs-algos/src/typestate.rs:126-127,143-144`
**Issue:** The `Transform` trait doc explicitly states transform "Errors if the
geometry disagrees with the fitted `n_features`." The shell impl validates only
`n == 0 || p == 0 || x.len() != n * p` — it never compares the incoming `p`
against the fitted `self.n_features_in_`. A caller that fitted on `p=3` and
transforms a `p=5` matrix gets a silently-wrong all-zeros `(rows, n_components)`
buffer instead of the documented error. The `n_features_in_` field is stored
specifically to enable this check but is unused in `transform`. When Phase 14
fills in the real projection, this missing guard becomes an out-of-contract
device read against the fitted components.
**Fix:** Add the feature-count check the trait documents:
```rust
let (n, p) = shape;
if p != self.n_features_in_ {
    return Err(AlgoError::Prim(PrimError::ShapeMismatch {
        operand: "x", rows: n, cols: p, len: x.len(),
    }));
    // or a dedicated AlgoError naming the fitted-vs-supplied feature mismatch
}
```

### WR-03: New `PyHDBSCAN` uses the sanctioned `lock_pool`, but `PyKMeans`/`PyDBSCAN` in the same edited file still use the panicking lock form

**File:** `crates/mlrs-py/src/estimators/cluster.rs:87,112,128,135,143,223,246,255` (panicking) vs `:360,429` (sanctioned)
**Issue:** `crates/mlrs-py/src/lib.rs:96-157` documents `lock_pool()` as the
SANCTIONED, poison-recovering lock path and states explicitly that "one
surviving `global_pool().lock().expect(...)` re-panics on a poisoned mutex and
re-bricks the interpreter, making the brick-prevention only partial." This phase
added `PyHDBSCAN` to `cluster.rs` using `lock_pool()` correctly, but left every
`KMeans`/`DBSCAN` lock site in the same file on the panicking
`global_pool().lock().expect("pool mutex")` form. The result is an inconsistent
failure mode within one file: a panic in `PyKMeans::fit` poisons the global
mutex; `PyHDBSCAN::labels_` then recovers via `lock_pool`, but `PyKMeans`'s own
later calls re-panic. `lib.rs` calls the legacy `cluster` wrappers a
"pre-existing, tracked migration," but the recovery is only sound if every site
participates, and this phase touched the file without resolving the mix.
**Fix:** Convert the `cluster.rs` lock sites to `crate::lock_pool()` so the file
this phase edited does not mix the two forms; or record an explicit tracking
issue and reference it at each remaining panicking site.

### WR-04: HDBSCAN leaves `min_samples`/`max_cluster_size` unvalidated with no tracked follow-up

**File:** `crates/mlrs-algos/src/cluster/hdbscan.rs:250-274` (build), `:288-325` (fit)
**Issue:** `HdbscanBuilder::build` guards only `min_cluster_size >= 2`.
`min_samples` is "stored verbatim and is NOT validated," and `max_cluster_size`
is unbounded. For a Phase-12 shell this is acceptable, but `min_samples = Some(0)`
and a `max_cluster_size` smaller than `min_cluster_size` are both geometrically
meaningless and will flow silently into Phase 15, which reaches a real device
kernel. `AlgoError::InvalidMinSamples` already exists in `error.rs:119-131` as
the precedent for the `>= 1` guard, so the validation pattern is established but
not applied here. The risk is that the deferred validation is forgotten when the
real compute lands.
**Fix:** When Phase 15 lands, extend `build()` with `min_samples >= 1` (when
`Some`) and a `max_cluster_size == 0 || max_cluster_size >= min_cluster_size`
check. Track explicitly (e.g. a `// TODO(phase-15): validate min_samples/
max_cluster_size` at the build guard) so it is not lost.

### WR-05: `predict_before_fit.stderr` golden is brittle and elides `Unfit` on its primary diagnostic line

**File:** `crates/mlrs-algos/tests/ui/predict_before_fit.stderr:5,10`; harness `crates/mlrs-algos/tests/compile_fail.rs:26-34`
**Issue:** The golden renders the found type as `Umap<f32>` on line 5 (the
defaulted `S = Unfit` elided) and only the "found struct" note on line 10 shows
`Umap<f32, Unfit>`. The stated VALUE gate is "non-compilation that references the
`Unfit` state." Two problems: (1) the golden was generated against rustc 1.96.0
while the toolchain pins `channel = "stable"` (a moving target), so a stable
bump can fail the exact-match golden on wording rather than on a real regression;
(2) because the primary line already elides `Unfit`, a future rustc that also
elided it from the note would let the test pass while no longer naming `Unfit`
at all, silently weakening the gate. `transform_before_fit.rs` (the `E0277`
trait-bound form) is more robust because it prints `Unfit` verbatim.
**Fix:** Prefer the `E0277` trait-bound assertion style (as in
`transform_before_fit.rs`) for both fixtures so the diagnostic always names the
`Unfit` argument, or pin an exact toolchain for the `compile_fail` job so the
golden cannot be silently invalidated by stable drift.

## Info

### IN-01: UMAP and HDBSCAN shells diverge in method order and doc text

**File:** `crates/mlrs-algos/src/cluster/hdbscan.rs:145-168` vs `crates/mlrs-algos/src/manifold/umap.rs:150-194`
**Issue:** UMAP orders `hyperparams_eq` before `into_builder`; HDBSCAN reverses
it, and HDBSCAN's `into_builder` doc drops UMAP's "available to callers who want
to tweak a constructed estimator before fitting" sentence. The two shells are
the copyable template for Phase 14/15, so the drift invites inconsistency.
**Fix:** Align method order and doc wording between the two shells.

### IN-02: Single-variant `Metric` enum with a write-only stored field

**File:** `crates/mlrs-algos/src/cluster/hdbscan.rs:49-53`, `crates/mlrs-algos/src/manifold/umap.rs:44-48`; parse sites `crates/mlrs-py/src/estimators/cluster.rs:299-306`, `manifold.rs:51-58`
**Issue:** `Metric` has exactly one variant (`Euclidean`); the stored `metric`
field is verbatim-stored but ignored by the trivial fit. Intentional shell
scaffolding, but the field is currently write-only and a linter may flag it.
**Fix:** No action for the shell; confirm Phase 14/15 consumes `metric`.

### IN-03: Duplicated geometry-guard block across `fit` and `transform`

**File:** `crates/mlrs-algos/src/manifold/umap.rs:378-385,450-457`, `crates/mlrs-algos/src/cluster/hdbscan.rs:299-306`
**Issue:** The `n == 0 || p == 0 || x.len() != n * p` guard plus the
`AlgoError::Prim(PrimError::ShapeMismatch { ... })` construction is copy-pasted
three times verbatim. A shared helper would remove the duplication and keep the
sites in sync — relevant given WR-02, where one of the three needs an extra check.
**Fix:** Extract a crate-private `validate_geometry(x, shape)` helper and call it
from each `fit`/`transform`.

### IN-04: `_state_phantom` helper is exported (doc-hidden) but unused

**File:** `crates/mlrs-algos/src/typestate.rs:191-194`
**Issue:** `_state_phantom<S: State>()` is a `pub fn` described as a downstream
helper, but no code in the phase calls it — the shells build `PhantomData`
directly. Harmless dead surface area today.
**Fix:** Drop it until a consumer exists, or add a one-line test that invokes it
so it stays compiled-exercised.

### IN-05: Two near-identical dispatch macros kept in lockstep only by prose

**File:** `crates/mlrs-py/src/dispatch.rs:90-115,150-178`
**Issue:** `any_estimator!` and `any_estimator_typestate!` are identical except
for the two fitted-arm type spellings. The doc explains why a second macro
exists rather than parameterizing, but the bodies must be kept in sync by hand
for any future field/derive change. Low risk while both are skeleton-only.
**Fix:** Consider a single macro with an optional `state:` token, or add a
cross-link comment at both sites.

---

_Reviewed: 2026-06-23T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
