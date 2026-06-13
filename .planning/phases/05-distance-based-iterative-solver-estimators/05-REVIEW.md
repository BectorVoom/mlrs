---
phase: 05-distance-based-iterative-solver-estimators
reviewed: 2026-06-13T07:05:38Z
depth: standard
files_reviewed: 41
files_reviewed_list:
  - crates/mlrs-algos/src/cluster/dbscan.rs
  - crates/mlrs-algos/src/cluster/kmeans.rs
  - crates/mlrs-algos/src/cluster/mod.rs
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/src/lib.rs
  - crates/mlrs-algos/src/linear/coordinate_descent.rs
  - crates/mlrs-algos/src/linear/lasso.rs
  - crates/mlrs-algos/src/linear/logistic.rs
  - crates/mlrs-algos/src/linear/mod.rs
  - crates/mlrs-algos/src/neighbors/classifier.rs
  - crates/mlrs-algos/src/neighbors/mod.rs
  - crates/mlrs-algos/src/neighbors/nearest.rs
  - crates/mlrs-algos/src/neighbors/regressor.rs
  - crates/mlrs-algos/src/traits.rs
  - crates/mlrs-algos/tests/dbscan_test.rs
  - crates/mlrs-algos/tests/elastic_net_test.rs
  - crates/mlrs-algos/tests/kmeans_test.rs
  - crates/mlrs-algos/tests/knn_classifier_test.rs
  - crates/mlrs-algos/tests/knn_regressor_test.rs
  - crates/mlrs-algos/tests/lasso_test.rs
  - crates/mlrs-algos/tests/logistic_test.rs
  - crates/mlrs-algos/tests/nearest_neighbors_test.rs
  - crates/mlrs-backend/src/prims/coordinate_descent.rs
  - crates/mlrs-backend/src/prims/dbscan.rs
  - crates/mlrs-backend/src/prims/kmeans.rs
  - crates/mlrs-backend/src/prims/lbfgs.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/src/prims/topk.rs
  - crates/mlrs-backend/tests/cd_test.rs
  - crates/mlrs-backend/tests/dbscan_mask_test.rs
  - crates/mlrs-backend/tests/kmeanspp_test.rs
  - crates/mlrs-backend/tests/lbfgs_test.rs
  - crates/mlrs-backend/tests/lloyd_test.rs
  - crates/mlrs-backend/tests/memory_gate_test.rs
  - crates/mlrs-backend/tests/topk_test.rs
  - crates/mlrs-kernels/src/coordinate.rs
  - crates/mlrs-kernels/src/dbscan.rs
  - crates/mlrs-kernels/src/kmeans.rs
  - crates/mlrs-kernels/src/lbfgs.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-kernels/src/topk.rs
findings:
  critical: 3
  warning: 7
  info: 4
  total: 14
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-06-13T07:05:38Z
**Depth:** standard
**Files Reviewed:** 41
**Status:** issues_found

## Summary

Phase 5 adds the distance-based and iterative-solver estimators (KMeans, DBSCAN,
KNN family, Lasso/ElasticNet, LogisticRegression) plus their supporting prims and
kernels. The code is heavily annotated, the validate-before-launch discipline is
applied consistently, and prior review tags (CR-01, CR-02, WR-01..WR-04) are
visibly addressed. The hyperparameter guards (ASVS V5), u32-overflow guards
(WR-03), and the index/class bounds checks in the KNN gather (WR-02) are genuine
and well placed.

The adversarial pass surfaces three correctness BLOCKERS, each on a code path the
committed oracle fixtures never exercise (every fixture uses contiguous labels and
well-separated blobs that never empty a cluster), plus a set of robustness/quality
WARNINGS. The most serious is the KMeans empty-cluster relocation, which can drive
a cluster count NEGATIVE and silently place a centroid at the origin — a wrong
result, not the NaN the code claims to prevent.

This is an adversarial review: the goal was to find defects on untested paths, not
to confirm the tested oracle paths pass. No `<structural_findings>` block was
provided, so all findings below are narrative.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: KMeans empty-cluster relocation can drive a donor count negative and silently zero a centroid

**File:** `crates/mlrs-backend/src/prims/kmeans.rs:184-209`
**Issue:** The relocation loop ranks ALL `n` samples by distance-to-assigned-center
descending and hands the top `n_empty` to the empty clusters, decrementing each
donor:

```rust
for (rank, &c) in empties.iter().enumerate() {
    let i = order[rank];
    let donor = labels[i] as usize;
    ...
    counts_i64[c] += 1;
    counts_i64[donor] -= 1;
}
```

Nothing prevents two of the chosen farthest points (`order[0]`, `order[1]`, …)
from sharing the SAME donor cluster, nor from draining a donor that has only one
member. sklearn's `_relocate_empty_clusters_dense` guards against exactly this: it
never relocates a point whose donor would be emptied. Here a donor with
`count == 1` that loses its single point lands at `counts_i64[donor] == 0`, and a
donor that is the donor for two relocations lands at `-1`. The finalize loop then
does `if counts_i64[c] > 0 { ... }` and otherwise leaves `centers[c]` at the
origin. So the documented "guard anyway: a 0 count leaves the center at the origin
rather than a NaN" produces a WRONG centroid (the origin) instead of the correct
mean — a silent correctness failure, not the NaN-avoidance it claims. A `-1` count
on a cluster that later regains points makes `inv = 1.0 / counts_i64[c]` negative.
The KMeans fixture is 3 well-separated blobs that never empties a cluster, so this
entire path is untested.
**Fix:** Mirror sklearn: when selecting farthest points to relocate, skip any
candidate whose donor cluster currently has `count <= 1` (advance to the next
farthest), track relocated points so none is reused, and surface a typed
`PrimError` if no valid donor remains — never leave a center at the origin.

### CR-02: LogisticRegression infers `n_classes` from `max(label)+1`, mislabeling non-contiguous targets and risking an OOB device read

**File:** `crates/mlrs-algos/src/linear/logistic.rs:217-232`
**Issue:** Class count is `n_classes = ((max_label + 1) as usize).max(2)`. This is
wrong in two provable ways:

1. **Non-contiguous labels.** Training labels `{0, 2}` (class 1 absent — a legal
   sklearn input that remaps to `classes_ = [0, 2]`, K=2) yield `max_label = 2 →
   n_classes = 3`. A phantom never-trained class 1 is fit, and `predict_labels`
   can return `1` (a class that does not exist). sklearn returns the original id
   `2`.
2. **Single-class input.** All labels `= 0` gives `max_label = 0`, then `.max(2)`
   forces `n_classes = 2`, fitting a binary model on a degenerate one-class
   problem instead of rejecting it (sklearn raises "needs at least 2 classes").

Worse, the softmax kernel (`crates/mlrs-kernels/src/lbfgs.rs:138`,
`yi = u32::cast_from(y[i])`) trusts `yi < k_classes` and indexes `w[yi*d ..]`. If
the K computed here is ever smaller than `max_label + 1` (e.g. a future change, or
a label set whose distinct count differs from `max+1`), that is an out-of-bounds
device read of the weight buffer — the validate-before-launch contract is
defeated because K is derived from the wrong quantity. The fixture only uses
contiguous `0..K`, so this is untested.
**Fix:** Collect distinct sorted labels, remap to a dense `[0, n_classes)` index
(store the `classes_` map for the `predict_labels` inverse), reject `n_distinct <
2` with a typed error, and set `n_classes = n_distinct`. The kernel must only see
remapped indices.

### CR-03: KNeighborsClassifier `n_classes` uses train `max+1`; a label gap returns a non-existent class id

**File:** `crates/mlrs-algos/src/neighbors/classifier.rs:142-164` (guard at 219-228)
**Issue:** `n_classes_ = max(y_class) + 1` over the TRAIN labels. With
non-contiguous labels (e.g. `{0, 2}`), `n_classes_ = 3` allocates a 3-wide proba
row whose column 1 is structurally always zero; `argmax_rows` over `[0, 3)` can
return class id `1`, which never existed in training. sklearn maps neighbor votes
through `classes_ = [0, 2]` and returns `2`. The WR-02 guard at line 221
(`class as usize >= n_classes`) only catches ids `>= max+1`, not the GAP at id 1,
so the wrong label passes silently. The committed KNN fixture draws `y_class` from
`rng.integers(0, KNN_N_CLASSES)`, contiguous by luck, so the gap case is untested.
**Fix:** Build `classes_` as the distinct sorted training labels, index proba
columns by dense class position, and map the final argmax column back through
`classes_` so a non-contiguous set returns the correct original id. Do not infer
width from `max+1`.

## Warnings

### WR-01: `lbfgs_minimize` reports a line-search breakdown as a successful (non-`NotConverged`) result

**File:** `crates/mlrs-backend/src/prims/lbfgs.rs:205-213, 248-253`; consumer at `crates/mlrs-algos/src/linear/logistic.rs:331-337`
**Issue:** When `line_search_wolfe` returns `None`, the solver `break`s with the
current `converged` flag (usually `false`) and `iters < maxiter`. The LogReg
estimator only surfaces `NotConverged` when `result.iters >= maxiter &&
!result.converged`, and its own comment explicitly ACCEPTS "early stop `iters <
maxiter`" as converged (Pitfall 5). So a genuine line-search breakdown at a
NON-stationary point (e.g. a NaN/degenerate gradient) is reported to the caller as
success, yielding a non-minimizer with no error. This is a different stop reason
than the legitimate ftol stall but is indistinguishable in `LbfgsResult`.
**Fix:** Add a `stop_reason` (Converged / FtolStall / LineSearchFailed / MaxIter)
to `LbfgsResult`; the estimator must surface `NotConverged` on `LineSearchFailed`
regardless of `iters`.

### WR-02: LogReg objective closure leaks pool buffers on a panic between acquire and release

**File:** `crates/mlrs-algos/src/linear/logistic.rs:276-301`
**Issue:** The closure acquires `w_d`/`b_d` from `pool` (lines 276-277) and
releases them at 282-283, but the release is NOT panic-safe: it sits after the
fallible `softmax_loss_grad` call and is only reached on the normal path. A true
panic (a kernel-launch assertion, or an `unreachable!` in a bit-cast for a
non-f32/f64 `F`) between acquire and release strands both device handles for the
process lifetime. The WR-01-tagged error capture only handles `Result::Err`, not
unwinding.
**Fix:** Release via an RAII guard whose `Drop` returns the handles, or scope the
acquisitions so a panic cannot strand them.

### WR-03: KMeans never surfaces non-convergence and silences it via a `tol_scaled` that can be exactly zero

**File:** `crates/mlrs-algos/src/cluster/kmeans.rs:256-348`
**Issue:** For a constant-feature design (every column identical) the mean feature
variance is 0, so `tol_scaled = tol · 0 = 0`. The Lloyd loop can then only stop on
the strict label-equality break or `max_iter`; under f32 centroid jitter the
strict break may never fire, so the loop silently exhausts `max_iter` and returns
a non-converged fit with NO `NotConverged` (KMeans has no convergence error path
at all, unlike Lasso/LogReg). Also note `tol_scaled` is computed in an O(n·d) host
double-pass over the full materialized `x_host`, contradicting the "heavy work
stays on-device" docstring.
**Fix:** Decide and document KMeans's non-convergence contract (sklearn warns), and
add a constant-feature regression test; if matching sklearn's `tol == 0`
semantics, keep the zero but cover it.

### WR-04: DBSCAN adjacency is materialized as `n × n` `u32` then duplicated as `n × n` `bool` — 5× the bitmask footprint, both held live

**File:** `crates/mlrs-backend/src/prims/dbscan.rs:115-168`; `crates/mlrs-kernels/src/dbscan.rs:78-83`
**Issue:** The kernel writes the adjacency as `u32` (`0`/`1`), the host reads it
into `adj_u32` (line 155), then maps it to a second full `Vec<bool>` `adjacency`
(line 168) while `adj_u32` is still alive. For the documented DoS surface
(T-05-04-02, "large `n` drives the n² allocation") this is 4× the memory a bitmask
needs on-device plus a redundant host copy, halving the `n` at which the prim OOMs
versus what the module's own "bounded, accepted" framing implies.
**Fix:** Keep one representation (e.g. test `adj_u32[..] != 0` inside
`neighbors()`), or pack into a bitset; do not hold both `adj_u32` and `adjacency`.

### WR-05: `kmeanspp_sample` degenerate fallback `expect`s rather than returning a typed error

**File:** `crates/mlrs-backend/src/prims/kmeans.rs:359-365`
**Issue:** When `total <= 0.0` the fallback is
`(0..n).find(|i| !chosen.contains(i)).expect("k <= n guarantees an unused index")`.
This panics if the `chosen.len() < k <= n` invariant is ever violated by a future
caller (e.g. `k == n` on all-duplicate data where `chosen` already holds all `n`),
violating the project's "typed error, never a panic across the boundary"
convention echoed elsewhere in this phase. It is also O(k·n) via repeated
`chosen.contains`.
**Fix:** Track membership in a `vec![bool; n]` and return a `PrimError` rather than
`expect` when no unused index remains.

### WR-06: `inertia` / `inertia_rows_host` omit the `k >= 1` guard their sibling kmeans entry points enforce

**File:** `crates/mlrs-backend/src/prims/kmeans.rs:222-268, 421-466`
**Issue:** Both derive `k = centers.len() / d` and check each label `< k`, but
neither rejects `k == 0`: an empty `centers` buffer passes `centers.len() % d != 0`
(0 % d == 0) and the function proceeds with `k == 0`. `lloyd_update` /
`kmeanspp_sample` go through `validate_geometry` which enforces `1 <= k`; this
inconsistent guard surface invites a future caller to hit the gap.
**Fix:** Add `if k == 0 { return Err(ShapeMismatch{operand:"centers", ..}) }` to
both.

### WR-07: `cd_solve` reads `y` back to host twice, contradicting its "acquired ONCE, reused" claim

**File:** `crates/mlrs-backend/src/prims/coordinate_descent.rs:91-110`
**Issue:** Line 92 materializes `y` to host (`y_host`); line 110 reads `y` to host
AGAIN to seed `r_dev` (`DeviceArray::from_host(pool, &y.to_host(pool))`). The
second copy is a duplicate device→host transfer of a buffer already in `y_host`.
Not a correctness bug, but it contradicts the module docstring's bounded-allocation
"acquired ONCE, reused" guarantee and doubles the `y` materialization cost.
**Fix:** Seed `r_dev` from the existing `y_host` (mapped back to `F`) or by a
device-side clone.

## Info

### IN-01: `fit_predict` round-trips labels host→device→host→device unnecessarily

**File:** `crates/mlrs-algos/src/cluster/dbscan.rs:131-140`; `crates/mlrs-algos/src/cluster/kmeans.rs:415-424`
**Issue:** `fit_predict` calls `self.labels(pool)` (device→host) then
`DeviceArray::from_host(pool, &labels)` (host→device) although `self.labels_` is
already device-resident. A device-side clone would avoid the round-trip. Minor —
labels are small.
**Fix:** Clone the existing `labels_` device buffer instead of host-bouncing.

### IN-02: `host_to_f64` / `f64_to_host` bit-cast helpers duplicated across 6+ modules

**File:** `crates/mlrs-algos/src/cluster/kmeans.rs:429`; `linear/coordinate_descent.rs:232`; `linear/lasso.rs:179`; `linear/logistic.rs:485`; `neighbors/classifier.rs:267`; `neighbors/regressor.rs:182`
**Issue:** The identical `match size_of::<F>() { 4 => .., 8 => .., _ =>
unreachable!() }` cast pair is copy-pasted into at least six estimator modules
(plus the prims and tests), with per-file `unreachable!` messages that drift.
**Fix:** Hoist a shared `mlrs_core` helper (or a small sealed trait) and remove the
~12 copies.

### IN-03: `KMeans` exposes no `max_iter` / `tol` override, unlike the other iterative estimators

**File:** `crates/mlrs-algos/src/cluster/kmeans.rs:112-142`
**Issue:** `new` and `with_init` hardcode `max_iter = 300`, `tol = 1e-4`.
`Lasso::with_opts` and `LogisticRegression::with_opts` both expose the override;
KMeans does not, an inconsistent surface the Phase-6 PyO3 layer will need.
**Fix:** Add a `with_opts`/builder exposing `max_iter`/`tol` for parity.

### IN-04: Dead `SEED` constant kept alive only by a `let _ = SEED;` lint suppression

**File:** `crates/mlrs-algos/tests/kmeans_test.rs:38, 260`
**Issue:** `const SEED: u64 = 42;` is never used (tests inject a fixed init); the
test ends with `let _ = SEED;` purely to silence dead-code.
**Fix:** Remove the unused constant and the suppression line.

---

_Reviewed: 2026-06-13T07:05:38Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
