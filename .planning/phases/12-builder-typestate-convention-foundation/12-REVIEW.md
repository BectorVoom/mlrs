---
phase: 12-builder-typestate-convention-foundation
reviewed: 2026-06-23T00:00:00Z
depth: standard
files_reviewed: 20
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
  - crates/mlrs-algos/tests/ui/transform_before_fit.rs
  - crates/mlrs-algos/tests/umap_test.rs
  - crates/mlrs-py/src/dispatch.rs
  - crates/mlrs-py/src/estimators/cluster.rs
  - crates/mlrs-py/src/estimators/manifold.rs
  - crates/mlrs-py/src/estimators/mod.rs
  - crates/mlrs-py/src/lib.rs
  - crates/mlrs-py/tests/manifold_test.rs
findings:
  critical: 0
  warning: 4
  info: 5
  total: 9
status: issues_found
---

# Phase 12: Code Review Report

**Reviewed:** 2026-06-23
**Depth:** standard
**Files Reviewed:** 20
**Status:** issues_found

## Summary

Phase 12 introduces the v3 typestate/builder convention foundation: a sealed `State`
marker trait with `Unfit`/`Fitted` markers, four consuming-`self` lifecycle traits,
two non-algorithmic estimator shells (UMAP, HDBSCAN), a second additive dispatch
macro (`any_estimator_typestate!`), and the first two PyO3 shells over typestate
estimators. The phase invariants were respected during review: the trivial fits
(zeros embedding / all `-1` labels) are NOT flagged as missing algorithm; the frozen
`traits::*` surface and the additive `any_estimator_typestate!` clone are accepted as
intentional; cfg-gated no-feature diagnostics are treated as false positives.

The typestate machinery itself is sound — the sealing is correct, the markers are
zero-sized, and the compile-fail gate proves the `Unfit` value/trait gate. No
Critical defects were found. The findings below are robustness and consistency
defects: a hyperparameter-validation gap in `UmapBuilder::build` (`spread` is never
validated, so NaN/negative `spread` slips through and disables the `min_dist`
check), a misleading not-fitted error on the wrong-dtype embedding accessor (a
purpose-built `dtype_mismatch` helper exists but is bypassed), and the new Phase-12
HDBSCAN PyO3 fit path sharing a file with legacy KMeans/DBSCAN wrappers that use the
non-poison-recovering lock path the module doc explicitly deprecates.

## Warnings

### WR-01: `UmapBuilder::build` never validates `spread`; NaN/negative `spread` defeats the `min_dist` guard

**File:** `crates/mlrs-algos/src/manifold/umap.rs:326-331`
**Issue:** The only build-time guard is
`if !self.min_dist.is_finite() || self.min_dist > self.spread`. `spread` is the
right-hand operand of the comparison but is itself never validated:
- If `spread` is `NaN`, `self.min_dist > self.spread` is `false` (all comparisons
  with NaN are false), so any finite `min_dist` passes and a `NaN` `spread` is
  stored. The doc comment for `InvalidMinDist` ("must be finite and `<= spread`")
  is silently violated — a NaN `spread` makes the relation meaningless.
- If `spread` is negative (e.g. `-1.0`) with a negative `min_dist <= spread`, the
  pair is accepted, even though `spread` is "the effective scale of embedded
  points" and must be positive (umap-learn requires `spread > 0`).

In Phase 14 the real curve-fit derives `a`/`b` from `min_dist`/`spread`; a NaN or
non-positive `spread` admitted here will surface much later as a NaN embedding far
from its origin. The validate-before-fit contract (D-08) is meant to reject
untrusted hyperparameters at the boundary, and `spread` is one.
**Fix:**
```rust
if !self.spread.is_finite() || self.spread <= 0.0 {
    return Err(BuildError::InvalidSpread {
        estimator: "umap",
        spread: self.spread,
    });
}
if !self.min_dist.is_finite() || self.min_dist > self.spread {
    return Err(BuildError::InvalidMinDist {
        estimator: "umap",
        min_dist: self.min_dist,
    });
}
```
(Add an `InvalidSpread` variant to `BuildError`, or fold it into `InvalidMinDist`'s
message; the blanket `build_err_to_py` mapper already covers any new variant.) At
minimum, guard `self.spread.is_finite()` so the NaN-defeats-the-check path is closed.

### WR-02: Wrong-dtype embedding accessor reports "not fitted" instead of a dtype mismatch

**File:** `crates/mlrs-py/src/estimators/manifold.rs:118-132` (and `:275-281`)
**Issue:** `embedding_f32_inner` returns `not_fitted("umap", "embedding_ (f32)")`
for ANY non-`F32` arm — including the `F64` arm, i.e. an estimator that IS fitted
but in the other dtype. The codebase already documents this exact hazard and ships
a purpose-built helper for it: `crate::errors::dtype_mismatch` (errors.rs:90-99,
"surfacing a `not_fitted` 'called before fit' error would mislead a Python user who
fitted in `fitted_dtype` and called the `requested_dtype` accessor", WR-04). The new
UMAP shell regresses on that decision: a user who fits f64 and reads `embedding_f32`
gets "not fitted yet: call fit before embedding_ (f32)", which is false — the
estimator is fitted. Same defect on `embedding_f64_inner` for the `F32` arm.
**Fix:** Distinguish the fitted-but-wrong-dtype case from the genuinely-unfit case,
mirroring the WR-04 helper:
```rust
fn embedding_f32_inner(&self) -> PyResult<Vec<f32>> {
    let pool = crate::lock_pool();
    match &self.inner {
        AnyUmap::F32(e) => Ok(e.embedding(&pool)),
        AnyUmap::F64(_) => Err(dtype_mismatch("umap", "f32", "f64")),
        AnyUmap::Unfit { .. } => Err(not_fitted("umap", "embedding_ (f32)")),
    }
}
```
(import `dtype_mismatch` from `crate::errors`). Apply symmetrically to the f64 path.

### WR-03: Phase-12 HDBSCAN wrapper shares a file with KMeans/DBSCAN bodies still on the deprecated panicking lock path

**File:** `crates/mlrs-py/src/estimators/cluster.rs:87, 112, 128, 135, 143, 223, 246, 255`
**Issue:** `lib.rs:108-118` declares `lock_pool` the SANCTIONED lock path and warns
that "one surviving `global_pool().lock().expect(\"pool mutex\")` re-panics on a
poisoned mutex and re-bricks the interpreter, making the brick-prevention only
partial." The new `PyHDBSCAN` body (added this phase) correctly uses `lock_pool()`,
but it was added into `cluster.rs` alongside `PyKMeans`/`PyDBSCAN`, both of which
still call `crate::global_pool().lock().expect("pool mutex")` directly. The result
is a single source file where one estimator is poison-safe and two are not. The
module doc frames the legacy form as "a pre-existing, tracked migration," so this is
acknowledged rather than newly introduced — but Phase 12 touched this file and is
the natural point to migrate the two siblings, especially given HDBSCAN's poison
recovery is only effective if every lock site in the process participates.
**Fix:** Convert the eight `global_pool().lock().expect("pool mutex")` sites in
`PyKMeans`/`PyDBSCAN` to `crate::lock_pool()`, matching the `PyHDBSCAN`/`PyUMAP`
bodies, or file an explicit follow-up so the partial-recovery gap is not lost.

### WR-04: `UmapBuilder` / `HdbscanBuilder` derive `Copy` but expose owned-`self` chained setters — silent setter-result drops compile

**File:** `crates/mlrs-algos/src/manifold/umap.rs:210` and `crates/mlrs-algos/src/cluster/hdbscan.rs:183`
**Issue:** Both builders are `#[derive(Debug, Clone, Copy)]` and use the
owned-`self` "consuming" setter style (`pub fn n_neighbors(mut self, v) -> Self`).
Because the builder is `Copy`, a misuse like `let b = Umap::builder(); b.min_dist(0.5); b.build::<f32>()` compiles WITHOUT a moved-value error (the setter receives a
copy, mutates it, and the result is dropped), silently discarding `min_dist(0.5)`.
The consuming-setter idiom relies on `!Copy` to make "ignored builder result" a
move-after-use error and force `let b = b.min_dist(0.5)`. With `Copy` derived that
safety net is gone. (The fluent one-liner form used in the PyO3 wrappers is correct;
the risk is for downstream Rust callers using the builder across statements.)
**Fix:** Drop `Copy` from both builder derives (keep `Clone`):
```rust
#[derive(Debug, Clone)]
pub struct UmapBuilder { /* ... */ }
```
This makes a dropped-setter-result a compile error, restoring the guarantee the
consuming-setter convention is supposed to provide.

## Info

### IN-01: `_state_phantom` is dead code with no caller in-tree

**File:** `crates/mlrs-algos/src/typestate.rs:191-194`
**Issue:** The `#[doc(hidden)] pub fn _state_phantom<S: State>()` helper has no
caller anywhere in the phase (the shells construct `PhantomData` inline). It is
documented as "a zero-cost helper for downstream estimator authors (Plan 02)," but
Plan 02 (UMAP/HDBSCAN) does not use it. It is harmless (zero codegen) but is
unexercised public API surface.
**Fix:** Remove it, or add a test that invokes it so its `State`-composability claim
is actually verified, or downgrade it to a doctest example.

### IN-02: `PartialFit` trait is defined-but-unused with no compile-level exercise

**File:** `crates/mlrs-algos/src/typestate.rs:165-185`
**Issue:** `PartialFit` is intentionally unimplemented in Phase 12 (Phase-16
retrofit target, documented). However nothing in the test suite even names it, so a
future signature drift (e.g. a wrong associated-type bound) would not surface until
Phase 16. `typestate_test.rs` exercises `State`/`Unfit`/`Fitted` but not the four
lifecycle traits.
**Fix:** Add a trivial `assert_impl`-style compile probe, or a doc example, that
references `PartialFit`'s signature so the frozen contract is regression-guarded
before Phase 16 consumes it.

### IN-03: `Predict`/`Transform` doc comments promise an `n_features` geometry check the shells do not perform

**File:** `crates/mlrs-algos/src/typestate.rs:126-127, 143-144` and `crates/mlrs-algos/src/manifold/umap.rs:443-460`
**Issue:** The trait docs state `predict`/`transform` "Errors if the geometry
disagrees with the fitted `n_features`." The UMAP `Transform` shell validates only
`n == 0 || p == 0 || x.len() != n*p` and ignores the fitted `n_features_in_`, so a
caller can `transform` a matrix whose `p != n_features_in_` and get zeros back with
no error. This is acceptable for the non-algorithmic shell, but the trait-level doc
asserts a guarantee the only implementation does not honor, which will mislead
Phase-14 implementers.
**Fix:** Either soften the trait doc to "implementations SHOULD validate against the
fitted `n_features`," or add the `p != self.n_features_in_` check to the shell so the
documented contract holds from the start.

### IN-04: Repeated geometry-guard block duplicated across three fit/transform bodies

**File:** `crates/mlrs-algos/src/cluster/hdbscan.rs:299-306`, `crates/mlrs-algos/src/manifold/umap.rs:378-385`, `crates/mlrs-algos/src/manifold/umap.rs:450-457`
**Issue:** The identical `if n == 0 || p == 0 || x.len() != n * p { return Err(AlgoError::Prim(PrimError::ShapeMismatch { operand: "x", rows: n, cols: p, len: x.len() })); }` block appears verbatim three times. As more typestate
estimators land (the convention is meant to be copied), this stanza will proliferate.
**Fix:** Extract a small crate-internal helper, e.g.
`fn check_xy_geometry(x_len: usize, shape: (usize, usize)) -> Result<(), AlgoError>`,
and call it from each body so the geometry contract has one definition.

### IN-05: `Metric` enum is duplicated between `manifold::umap` and `cluster::hdbscan` with single identical variant

**File:** `crates/mlrs-algos/src/manifold/umap.rs:44-48` and `crates/mlrs-algos/src/cluster/hdbscan.rs:49-53`
**Issue:** Both modules define `pub enum Metric { Euclidean }` independently, and the
two PyO3 parsers (`parse_metric`, `parse_hdbscan_metric`) are near-identical. This is
acceptable per-estimator decoupling for now, but with only one variant each it is
pure duplication that the two string parsers must each track. Noting for the Phase
14/15 expansion where the metric sets diverge — at that point the divergence
justifies the split; today it does not.
**Fix:** No action required for Phase 12; revisit when the metric sets actually
diverge. If they are expected to stay aligned, consider a shared `metric` module.

---

_Reviewed: 2026-06-23_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
