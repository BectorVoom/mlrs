---
phase: 05-distance-based-iterative-solver-estimators
reviewed: 2026-06-13T04:46:32Z
depth: standard
files_reviewed: 23
files_reviewed_list:
  - crates/mlrs-algos/src/traits.rs
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/src/cluster/mod.rs
  - crates/mlrs-algos/src/cluster/kmeans.rs
  - crates/mlrs-algos/src/cluster/dbscan.rs
  - crates/mlrs-algos/src/neighbors/mod.rs
  - crates/mlrs-algos/src/neighbors/nearest.rs
  - crates/mlrs-algos/src/neighbors/classifier.rs
  - crates/mlrs-algos/src/neighbors/regressor.rs
  - crates/mlrs-algos/src/linear/coordinate_descent.rs
  - crates/mlrs-algos/src/linear/lasso.rs
  - crates/mlrs-algos/src/linear/elastic_net.rs
  - crates/mlrs-algos/src/linear/logistic.rs
  - crates/mlrs-kernels/src/topk.rs
  - crates/mlrs-kernels/src/kmeans.rs
  - crates/mlrs-kernels/src/dbscan.rs
  - crates/mlrs-kernels/src/coordinate.rs
  - crates/mlrs-kernels/src/lbfgs.rs
  - crates/mlrs-backend/src/prims/topk.rs
  - crates/mlrs-backend/src/prims/kmeans.rs
  - crates/mlrs-backend/src/prims/dbscan.rs
  - crates/mlrs-backend/src/prims/coordinate_descent.rs
  - crates/mlrs-backend/src/prims/lbfgs.rs
findings:
  critical: 2
  warning: 7
  info: 5
  total: 14
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-06-13T04:46:32Z
**Depth:** standard
**Files Reviewed:** 23
**Status:** issues_found

## Summary

Reviewed the Phase-5 distance-based & iterative-solver estimator stack: the trait
surface, the two clustering estimators (KMeans, DBSCAN), the three KNN estimators,
the coordinate-descent linear models (Lasso/ElasticNet), LogisticRegression, and
all five backing kernels + prim launch wrappers.

The code is carefully written and unusually well-documented. The DBSCAN host DFS
was verified line-by-line against `_dbscan_inner.pyx` (labels-on-pop, ascending
neighbor push, LIFO) and matches sklearn exactly. The topk `select_k`
selection-by-rank kernel was traced through duplicate-value and out-of-order cases
and is correct. The CubeCL GATHER patterns are a deliberate cubecl-cpu constraint
and were not treated as findings.

Two BLOCKERs surfaced, both correctness-affecting on edge inputs that fall inside
the documented "matches sklearn within 1e-5" contract:

1. The k-means++ first-center draw and the inner weighted draw use modulo-reduced
   PRNG output, which (a) silently breaks the documented cross-backend
   seed-reproducibility invariant on the boundary and (b) the weighted-draw cumulative
   scan can index a wrong sample under FP rounding, both of which can shift the final
   partition past the oracle's label-permutation tolerance.
2. `lloyd_update`'s empty-cluster relocation does not reproduce sklearn's relocation
   target, so any fit that hits an empty cluster diverges from the oracle — and the
   code self-documents this as an approximation.

The remaining findings are robustness gaps (panics on edge inputs, an `.expect()`
inside the L-BFGS inner loop, an unvalidated label space) and doc/quality defects.

## Critical Issues

### CR-01: `lloyd_update` empty-cluster relocation diverges from sklearn

**File:** `crates/mlrs-backend/src/prims/kmeans.rs:150-198`
**Issue:** The relocation comment itself admits the implementation does NOT match
sklearn: sklearn's `_relocate_empty_clusters` moves an empty cluster to the point
with the largest squared distance **to its own currently-assigned center**, and it
selects those points from the global per-sample distance-to-assigned-center array
(the same array used for inertia), removing the relocated point's contribution and
decrementing the donor cluster. This code instead relocates to "the sample with the
max squared distance to the nearest **non-empty new** center," with a per-empty
`used[]` exclusion that has no sklearn analogue. For any dataset/init that produces
an empty cluster during Lloyd (common with k-means++ on clustered data or small n),
the resulting `cluster_centers_` / `labels_` / `inertia_` will not match the sklearn
oracle even up to a label permutation — violating the core 1e-5 contract. The
estimator (`kmeans.rs`) runs `with_init` for the deterministic oracle precisely so
both sides run identical Lloyd; this relocation rule breaks that equivalence the
moment a cluster empties.

**Fix:** Reproduce sklearn's rule exactly. The relocation needs the per-sample
squared distance to the *assigned* center (already computed by `inertia_rows`) and
the old assignment, so the relocation must be lifted into the estimator's Lloyd loop
(where labels + the previous centers are available) rather than living inside the
stateless prim. Concretely, mirror `_relocate_empty_clusters_dense`:

```text
# after computing new sums/counts, before dividing:
empty   = [c for c in 0..k if counts[c] == 0]
far_idx = argsort(dist_to_assigned_center)[::-1][:len(empty)]   # global ranking
for c, i in zip(empty, far_idx):
    old = labels[i]
    centers[c] = X[i]
    counts[c]  += 1
    counts[old] -= 1     # donor loses the point
    # (sklearn also fixes the running sums; recompute affected rows)
```
Until the prim is given the assigned-center distances + old labels, it cannot
reproduce sklearn and should not silently approximate.

### CR-02: k-means++ host draw has modulo bias and an FP-rounding mis-pick that break seed-reproducibility and can shift the partition

**File:** `crates/mlrs-backend/src/prims/kmeans.rs:321, 347-366`
**Issue:** Two distinct defects in the documented-seeded k-means++ sampler:

1. Line 321: `let first = (rng.next_u64() % n as u64) as usize;` — modulo reduction
   of a 64-bit value is biased for `n` not a power of two and, more importantly, is
   NOT the reduction sklearn uses (`sample_int = rng.randint(n)` over its own MT19937
   stream). The module docstring promises "the same `seed` yields the SAME indices
   across runs and backends" as a correctness invariant, but this draw is neither
   sklearn-equivalent nor even unbiased. Because the *first* center seeds the entire
   greedy D²-weighted chain, a different first index produces a different final
   partition — which can exceed the oracle's label-permutation tolerance.

2. Lines 347-366: the weighted draw does `target = rng.next_f64() * total` then a
   forward cumulative scan picking the first `i` with `acc >= target`. Under f64
   rounding the accumulated `acc` can fall a few ULP short of `total` on the last
   element, so when `target` rounds to essentially `total` the loop never triggers
   `acc >= target` and falls through to the initializer `pick = n - 1` — silently
   selecting the last sample regardless of its weight (and `n-1` may even be an
   already-chosen, zero-weight index, then "repaired" to an arbitrary unused index).
   This is a wrong-sample selection, not just a tie-break nicety.

**Fix:** For (1), if the goal is sklearn agreement the only robust path is to INJECT
the init for the oracle (which `KMeans::with_init` already supports) and treat the
internal sampler as best-effort; but then the "reproducible across backends" claim
must be downgraded in the docs, OR replace the draw with an unbiased
rejection-sampled `randint` and document that it is mlrs-specific, not
sklearn-bit-identical. For (2), clamp the scan so it cannot fall through:

```rust
let target = rng.next_f64() * total;
let mut acc = 0.0_f64;
let mut pick = n - 1;            // only used if EVERY weight is 0, already guarded by total<=0
for (i, &w) in min_d2.iter().enumerate() {
    acc += w;
    if acc >= target { pick = i; break; }
}
// guarantee `pick` has positive weight (rounding can land past the last positive bin):
if min_d2[pick] <= 0.0 {
    pick = min_d2.iter().rposition(|&w| w > 0.0).unwrap_or(pick);
}
```

## Warnings

### WR-01: `softmax_loss_grad` failure panics inside the L-BFGS inner loop

**File:** `crates/mlrs-algos/src/linear/logistic.rs:266`
**Issue:** The objective closure calls
`softmax_loss_grad(..).expect("softmax_loss_grad geometry validated before launch")`.
The closure is invoked on every L-BFGS iteration AND multiple times per line-search
step. Any `PrimError` from the prim (e.g. a future allocator/launch failure, or a
geometry assumption that does not hold for a degenerate `k`) becomes a hard panic
that unwinds through the solver, instead of the typed `AlgoError` the rest of the
estimator surface returns. A panicking library call across the PyO3 boundary (Phase
6) is a crash, not a recoverable error.

**Fix:** Make the closure fallible and thread the error out. Either have the closure
capture a `Result` slot and return a sentinel large loss + zero grad on error (then
check the slot after `lbfgs_minimize`), or change `lbfgs_minimize`'s closure bound to
`FnMut(&[f64]) -> Result<(f64, Vec<f64>), PrimError>` and propagate with `?`.

### WR-02: KNN `predict_proba` / regressor `predict` panic on empty training labels and index gather is unchecked at runtime

**File:** `crates/mlrs-algos/src/neighbors/classifier.rs:205-209`, `crates/mlrs-algos/src/neighbors/regressor.rs:160-161`
**Issue:** `y_class[train_idx]` / `y_reg[train_idx]` index host vectors with
`idx_host[q*k+j] as usize`, and the bounds check is only a `debug_assert!`
(classifier) or absent (regressor). In release builds a corrupted/oversized index
from `top_k` (or a `k`/`n_train` mismatch slipping past validation) is an unchecked
panic or, worse, a silent wrong read. The classifier's `class as usize` slot write
into `proba` is likewise only `debug_assert`-guarded, so an out-of-range class id
(possible if test labels exceed train `max+1`) writes out of bounds in debug and is
UB-adjacent in release if it ever fires.

**Fix:** Promote the `debug_assert!`s to real bounds checks that return
`AlgoError::Prim(PrimError::ShapeMismatch{..})`, or document and enforce the
invariant that `top_k` indices are always `< n_train` and class ids `< n_classes`
with a runtime guard at the gather site.

### WR-03: `KMeans::assign` casts `usize` n/k/d into kernel `u32` with no overflow guard, and `predict_labels` recomputes instead of reusing fitted labels

**File:** `crates/mlrs-algos/src/cluster/kmeans.rs:178-192, 382`
**Issue:** Two issues. (a) The prim layer casts `n as u32`, `k as u32`, `d as u32`
throughout (kmeans/dbscan/cd/lbfgs prims) with no check that the value fits in
`u32`. For the n²-DBSCAN and n×k distance paths a large-but-legal `usize` silently
truncates to a wrong `u32` launch geometry → out-of-bounds device reads. This is the
exact class of "untrusted geometry becomes OOB device read" the validation comments
claim to prevent, but the cast is unguarded. (b) `predict_labels` re-runs the full
distance+argmin against the fitted centers even when called with the training matrix;
not a bug, but note it can produce labels that differ from the stored `labels_` if
the final post-Lloyd assignment pass and a fresh assign tie-break differently — they
should be identical, so any divergence is a latent bug worth a test.

**Fix:** Add a `usize → u32` guard in each prim's validate step
(`if n > u32::MAX as usize { return Err(ShapeMismatch...) }`) for every dimension
passed to a launch. Confirm with a test that `predict_labels(X_train)` equals the
stored `labels_`.

### WR-04: ElasticNet/Lasso recompute column centering and norms with two full host readbacks of X

**File:** `crates/mlrs-algos/src/linear/coordinate_descent.rs:126-158` + `crates/mlrs-backend/src/prims/coordinate_descent.rs:91-101`
**Issue:** `cd_fit` reads X to host, centers it, re-uploads `x_centered`, and then
`cd_solve` immediately reads the centered X back to host again to compute
`norm2_cols`. Beyond the redundant round-trip (perf, out of scope), the correctness
concern is numerical: centering is done in f64 then truncated back to `F` (f32) at
line 149 before `cd_solve` re-reads and re-promotes to f64 for the norms. For f32
this double f64→f32→f64 narrowing of the centered design injects extra rounding into
the very quantity (`norm2_cols`, the soft-threshold denominator) that drives the
exact-zero sparsity pattern the docs promise to match within 1e-5. sklearn centers
in the working dtype once.

**Fix:** Center in `F` directly (matching sklearn's dtype) or pass the f64 centered
design through to `cd_solve` without the intermediate `F` narrowing, so the norms and
the residual are computed from one consistent representation.

### WR-05: LogisticRegression accepts a non-contiguous / gapped label space silently

**File:** `crates/mlrs-algos/src/linear/logistic.rs:216-232`
**Issue:** `n_classes` is derived as `max(round(y)) + 1`. If the integer labels are
non-contiguous (e.g. classes `{0, 2}` with no `1`), `k = 3` weight rows are trained,
class 1 gets an all-zero target and a meaningless probability column, and
`predict_labels` can emit class `1` which never appeared in training. sklearn
re-labels to a contiguous `classes_` index space and would never do this. The
fixture happens to be contiguous (documented), so the oracle passes, but the public
API silently mis-behaves on a realistic input.

**Fix:** Build the sorted unique label set, map to contiguous `[0, n_classes)`
indices for the solve, and map predictions back through `classes_` (store the
original labels for `predict_labels` output), exactly like sklearn.

### WR-06: DBSCAN `min_samples` validated as `< 1` but the field is `usize` — dead branch, and `eps` finiteness rejected via the wrong typed error path

**File:** `crates/mlrs-algos/src/cluster/dbscan.rs:160-171`
**Issue:** `min_samples` is `usize`; `self.min_samples < 1` is only true for `0`,
which is correct, but the comment in `error.rs:119` says a core point needs ≥1 and
the prim (`prims/dbscan.rs:209`) re-checks `min_samples < 1`. More substantively,
the estimator validates `!(self.eps >= 0.0) || !self.eps.is_finite()` and maps BOTH
to `InvalidEps`, but a non-finite (NaN/inf) `eps` is a different failure class than a
negative one; mapping `NaN` to "must be >= 0" is a misleading diagnostic. Minor, but
the duplicated validation in estimator + prim can drift.

**Fix:** Keep a single source of truth for the eps/min_samples guard (estimator),
and give NaN/inf its own message, or fold the finiteness check into the message text.

### WR-07: `enet_gap` and the softmax kernel recompute logits/dot products O(k) and O(rows·cols) times per call with no early exit, risking divergence under f32 reassociation

**File:** `crates/mlrs-kernels/src/lbfgs.rs:101-172`, `crates/mlrs-kernels/src/coordinate.rs:166-192`
**Issue:** The softmax kernel computes `raw[i,k]` THREE separate times per `(i,k)`
(row-max pass, sum-exp pass, gradient pass), each an independent `d`-length dot
product. In f32 these three dot products are computed in the same order so they
should agree bit-for-bit, but any future refactor that reorders one pass will
silently desynchronize `lse[i]` from the `p[i,k]` used in the gradient, producing a
subtly wrong gradient that L-BFGS will still "converge" on — landing off the oracle
with no error. This is a latent correctness fragility in the project's
highest-risk estimator.

**Fix:** Compute `raw[i,*]` once into a small per-row scratch (k is tiny) and reuse
it for the max, the sum-exp, the loss term, and the gradient, so the three passes
cannot drift. If the cubecl-cpu lowering forbids the scratch array, add a test that
pins `lse[i]` consistency across the passes.

## Info

### IN-01: LogisticRegression doc comments contradict the actual default constants

**File:** `crates/mlrs-algos/src/linear/logistic.rs:95-97, 115-117, 122-123`
**Issue:** The struct field docs and `new`/`with_opts` docs say defaults are
`max_iter = 100` and `tol = 1e-4`, but the constants are `LOG_DEFAULT_MAX_ITER = 300`
(line 69) and `LOG_DEFAULT_TOL = 1e-5` (line 77), and `new` passes those. The docs
are stale relative to the code.
**Fix:** Update the three doc comments to state `max_iter = 300` and `tol = 1e-5`.

### IN-02: `error.rs` `InvalidEps` message says "must be >= 0" but the variant is documented "non-positive" / the comment says `eps >= 0`

**File:** `crates/mlrs-algos/src/error.rs:108-117`
**Issue:** The doc line 108 says "non-positive neighborhood radius `eps`" while the
guard and message accept `eps == 0`. "Non-positive" implies `<= 0` is rejected, but
`eps == 0` is actually allowed (only `< 0`/NaN rejected). Wording is contradictory.
**Fix:** Change "non-positive" to "negative".

### IN-03: `kmeans.rs` stores `labels_` then `fit_predict` reads them back and re-uploads

**File:** `crates/mlrs-algos/src/cluster/kmeans.rs:401-403`, `crates/mlrs-algos/src/cluster/dbscan.rs:137-139`
**Issue:** `fit_predict` calls `self.labels(pool)` (device→host) then
`DeviceArray::from_host` (host→device), a pointless round-trip when `labels_` is
already a device buffer.
**Fix:** Clone the device handle (or return a device-resident view) instead of the
host bounce. Out of v1 perf scope but trivially avoidable.

### IN-04: Magic constant `1e-6` integer-label tolerance is unexplained

**File:** `crates/mlrs-algos/src/linear/logistic.rs:221`
**Issue:** `(li - lf).abs() > 1e-6` rejects non-integer labels with a hardcoded
tolerance; the same idea in the classifier uses `.round()` with no tolerance
(classifier.rs:144). Inconsistent label-integrality policy across the two estimators.
**Fix:** Extract a shared `nearest_integer_label` helper with one documented
tolerance.

### IN-05: `host_to_f64` / `f64_to_host` bit-cast helper duplicated verbatim across six files

**File:** `crates/mlrs-algos/src/cluster/kmeans.rs:409`, `neighbors/classifier.rs:246`, `neighbors/regressor.rs:171`, `linear/coordinate_descent.rs:221`, `linear/elastic_net.rs:272`, `linear/logistic.rs:457` (+ backend prims)
**Issue:** The identical `match size_of::<F>()` f32/f64 reinterpret pair is copied
into at least six estimator files and three prim files. A single off-by-one in one
copy would be a silent numeric corruption.
**Fix:** Hoist to one shared `mlrs-core` (or `mlrs-algos` internal) helper module and
import it everywhere.

---

_Reviewed: 2026-06-13T04:46:32Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
