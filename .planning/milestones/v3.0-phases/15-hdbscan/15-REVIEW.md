---
phase: 15-hdbscan
reviewed: 2026-06-24T07:23:37Z
depth: standard
files_reviewed: 19
files_reviewed_list:
  - crates/mlrs-algos/src/cluster/hdbscan.rs
  - crates/mlrs-algos/src/cluster/hdbscan/centers.rs
  - crates/mlrs-algos/src/cluster/hdbscan/condense.rs
  - crates/mlrs-algos/src/cluster/hdbscan/glosh.rs
  - crates/mlrs-algos/src/cluster/hdbscan/mst.rs
  - crates/mlrs-algos/src/cluster/hdbscan/select.rs
  - crates/mlrs-algos/src/cluster/hdbscan/single_linkage.rs
  - crates/mlrs-algos/src/cluster/hdbscan/stability.rs
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-algos/tests/hdbscan_test.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/src/prims/mutual_reachability.rs
  - crates/mlrs-backend/tests/mutual_reachability_test.rs
  - crates/mlrs-core/src/label_perm.rs
  - crates/mlrs-core/src/lib.rs
  - crates/mlrs-core/tests/helpers_test.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-kernels/src/mutual_reachability.rs
  - scripts/gen_oracle.py
findings:
  critical: 0
  warning: 6
  info: 5
  total: 11
status: issues_found
---

# Phase 15: Code Review Report

**Reviewed:** 2026-06-24T07:23:37Z
**Depth:** standard
**Files Reviewed:** 19
**Status:** issues_found

## Summary

This phase implements the full HDBSCAN host back-end (core distances →
mutual-reachability → MST → single-linkage → condensed tree → EoM/leaf selection
→ labelling/probabilities/GLOSH/centers) plus a single new device GATHER kernel
(`mutual_reachability`) and its backend wrapper. The code is dense, heavily
documented, and faithfully ports sklearn `_hdbscan/_tree.pyx` and the `hdbscan`
0.8.44 GLOSH pipeline. The oracle-gate test suite is broad and asserts real
values (not just non-panic), including the duplicate-point R-9 case.

The algorithm logic is largely sound: I traced the MST variants, the condense
runt-pruning, stability accumulation, EoM/leaf selection, the union-find label
mapping, and the GLOSH parallel tree, and found them to match the documented
sklearn/hdbscan references in the cases the test fixtures exercise. The device
kernel is cpu-MLIR-safe by construction and the backend wrapper validates
geometry before launch.

No BLOCKER-class defects (memory corruption, security, data loss, definite
wrong-answer-on-tested-path) were proven. The findings below are robustness gaps
on **public** primitive functions that the validated `fit` path happens to
shield, several documented-behavior/actual-behavior mismatches, and degenerate
edge cases that can panic or diverge from sklearn outside the fixture coverage.
Because the project's core value is "correct ML that matches sklearn within
1e-5" and these primitives are re-exported (`pub`) and unit-tested directly, the
robustness gaps are classified WARNING rather than INFO.

## Warnings

### WR-01: `clamp(1, n)` panics when `n == 0` in public core-distance functions

**File:** `crates/mlrs-algos/src/cluster/hdbscan/mst.rs:209` and `crates/mlrs-algos/src/cluster/hdbscan.rs:699`
**Issue:** `min_samples.clamp(1, n)` panics with `min > max` when `n == 0`
(verified: `usize::clamp(1, 0)` → `panic!("min > max. min = 1, max = 0")`).
`core_distances_dense` is a `pub fn` re-exported via
`mlrs_algos::cluster::hdbscan::mst::*` and called directly in tests, so it is
part of the public surface. The `Hdbscan::fit` path is shielded because
`validate_geometry` rejects `n == 0` first (`typestate.rs:67`), but any direct
caller (or a future internal caller that skips the guard) gets a panic instead
of a typed error. The same `clamp(1, n)` appears on the feature path at
`hdbscan.rs:699`.
**Fix:** Guard `n == 0` (and `core.len()`/`dist.len()`) before clamping, or
clamp the upper bound up to `1`:
```rust
// core_distances_dense
if n == 0 {
    return Vec::new();
}
let k = min_samples.clamp(1, n.max(1)) - 1;
```
For `feature_metric_single_linkage`, `n >= 1` is guaranteed by the prior
`validate_geometry`, so a `debug_assert!(n >= 1)` documenting the invariant is
sufficient there.

### WR-02: `get_clusters` computes `n_samples == 0` when the condensed tree has no singleton rows

**File:** `crates/mlrs-algos/src/cluster/hdbscan/select.rs:251-257`
**Issue:** `n_samples` is derived as `max(child)+1` over rows with
`cluster_size == 1`, falling back to `unwrap_or(0)` when there are no singleton
rows. A condensed tree whose every child is a sub-cluster (no point ever falls
out as a singleton — a degenerate but structurally reachable tree) yields
`n_samples = 0`, which then flows into `do_labelling` as a sizing parameter and
into `max_cluster_size` sentinel arithmetic (`n_samples + 1`). The resulting
`labels` vector would be length `root_cluster` (≠ true point count), masked only
by the defensive `labels_i32.resize(n, -1)` in `tree_to_labels`. The probability
vector would then be silently mis-sized relative to the real point count.
**Fix:** Derive `n_samples` from the true point count instead of inferring it
from singleton rows, e.g. thread the known `n` from `tree_to_labels` into
`get_clusters` (sklearn passes `num_points` explicitly rather than reconstructing
it from the tree).

### WR-03: `centroids_`/`medoids_` doc claims `None` when no cluster forms, but code returns `Some(empty)`

**File:** `crates/mlrs-algos/src/cluster/hdbscan.rs:177-184` and `565-585`
**Issue:** The field docs state `centroids_`/`medoids_` are `Some` "only when
`store_centers` requests centroids AND the fit produced at least one cluster;
`None` otherwise". But in `fit` (lines 565-585) the code always wraps the result
of `weighted_cluster_center` in `Some(DeviceArray::from_host(...))` whenever
`store_centers.is_some()` — even when the fit is all-noise (`n_clusters == 0`),
in which case `weighted_cluster_center` returns `Some(vec![])` (empty), so
`centroids_` becomes `Some(empty_device_array)`, not `None`. A consumer relying
on the documented `None`-means-no-cluster contract will instead receive an empty
`Some`. The `centers_match` test only fits a fixture with real clusters, so this
divergence is untested.
**Fix:** Either map empty results to `None`:
```rust
let cent_dev = cent.filter(|c| !c.is_empty()).map(|c| { ... });
```
or correct the field docs to "`Some(possibly-empty)` whenever `store_centers`
requests them".

### WR-04: Asymmetric precomputed matrix silently produces wrong MST (no validation)

**File:** `crates/mlrs-algos/src/cluster/hdbscan.rs:614-662` (`precomputed_single_linkage`)
**Issue:** The doc (lines 621-626) acknowledges sklearn *requires* the
precomputed matrix to be symmetric (`np.allclose(X, X.T)`) and that the dense
Variant-A MST reads `mr[current_node][..]` rows, so an asymmetric input
"would silently use the upper-triangle reading". The code only validates
squareness (`n != p`), not symmetry. An asymmetric `Metric::Precomputed` input
therefore produces silently-wrong clustering rather than a typed error — exactly
the untrusted-input-becomes-wrong-answer class the rest of this codebase guards
against (T-15-03-V5a names the squareness guard; symmetry is left undefended).
**Fix:** Add a host-side symmetry check (within tolerance) before computing core
distances, returning a typed error on violation, or document the precondition as
a hard caller contract with a `debug_assert`. At minimum a `debug_assert!` of
near-symmetry would catch fixture regressions in test builds:
```rust
debug_assert!(
    (0..n).all(|i| (0..n).all(|j|
        (dist_raw[i*n+j] - dist_raw[j*n+i]).abs() <= 1e-9)),
    "precomputed distance matrix must be symmetric",
);
```

### WR-05: Recursive `UnionFind::find` / `recurse_leaf_dfs` / `traverse_upwards` can stack-overflow on large/degenerate inputs

**File:** `crates/mlrs-algos/src/cluster/hdbscan/select.rs:54-60` (`find`), `103-118` (`recurse_leaf_dfs`), `137-167` (`traverse_upwards`)
**Issue:** `TreeUnionFind::find` recurses for path compression; `recurse_leaf_dfs`
recurses per cluster-tree level; `traverse_upwards` recurses per ancestor.
Path-compression `find` on a freshly-built union-find (before compression) can
recurse to depth O(chain length), and a deeply nested cluster tree drives
`recurse_leaf_dfs`/`traverse_upwards` to depth O(tree height). On adversarial or
large inputs this risks a stack overflow (an uncatchable abort) rather than a
graceful failure. sklearn's `find` is iterative for exactly this reason.
**Fix:** Convert `find` to an iterative two-pass loop (walk to root, then
re-point), and convert `recurse_leaf_dfs`/`traverse_upwards` to explicit-stack
iteration. The iterative `find` is a direct mechanical translation:
```rust
fn find(&mut self, mut x: usize) -> usize {
    let mut root = x;
    while self.parent[root] != root { root = self.parent[root]; }
    while self.parent[x] != root { let n = self.parent[x]; self.parent[x] = root; x = n; }
    root
}
```

### WR-06: `n_clusters` in `weighted_cluster_center` assumes a dense `0..k` label range; a sparse range over-allocates and emits zero rows

**File:** `crates/mlrs-algos/src/cluster/hdbscan/centers.rs:89-101`
**Issue:** `n_clusters` is computed as `max(label)+1`, and the loop emits a row
per `c in 0..n_clusters`, skipping clusters with no members (leaving that row
all-zeros). This is correct *only* if the label space is guaranteed dense
(`0..k` with no gaps). `select::get_clusters` does produce a dense `cluster_map`
(`0..k`), so the current pipeline is safe — but the function is `pub`-adjacent
(called with whatever labels `fit` produces) and the invariant is implicit. If
labels ever arrive sparse (e.g. a future selection change or a direct caller),
the output silently contains all-zero "phantom" center rows that the
`centers_match` test's permutation mapping could misattribute.
**Fix:** Either assert the dense-range invariant
(`debug_assert!(distinct_nonneg_labels == n_clusters)`) or densify labels to a
contiguous `0..k` inside `weighted_cluster_center` so the output rows correspond
exactly to the distinct present clusters.

## Info

### IN-01: `usize::MAX` used as a fabricated `len` sentinel in overflow-guard errors

**File:** `crates/mlrs-algos/src/cluster/hdbscan.rs:744` and `crates/mlrs-backend/src/prims/mutual_reachability.rs:76`
**Issue:** The `checked_mul` overflow guards construct `PrimError::ShapeMismatch`
with `len: usize::MAX` as a placeholder because no numeric-overflow variant
exists. The error message will report a nonsensical "len = 18446744073709551615",
which is misleading for diagnosis. This mirrors a pre-existing knn_graph
precedent, so it is consistent — but a dedicated `Overflow` variant (or a clearer
sentinel) would diagnose better.
**Fix:** Add a `PrimError::Overflow { operand, rows, cols }` variant, or document
the sentinel in the error path.

### IN-02: `nn` computed only for a `debug_assert_eq!` in the cosine path

**File:** `crates/mlrs-algos/src/cluster/hdbscan.rs:739-748`
**Issue:** The `n.checked_mul(n)` result `nn` is bound for the overflow guard but
then only consumed by `debug_assert_eq!(dist_dense.len(), nn)` (line 748), which
is compiled out in release. The overflow guard is still load-bearing (the `?`
short-circuits on overflow), so this is not dead code, but the `nn` binding's
only *use* vanishes in release — readers may flag it. Consider a comment
clarifying the guard is the point, not the assert.
**Fix:** None required for correctness; optionally `let _ = nn;` documentation or
fold the assert into a `#[cfg(debug_assertions)]` block.

### IN-03: `host_pairwise` is duplicated verbatim across two modules

**File:** `crates/mlrs-algos/src/cluster/hdbscan.rs:1026-1066` and `crates/mlrs-algos/src/cluster/hdbscan/centers.rs:157-216`
**Issue:** The FAST-metric pairwise-distance closure is implemented twice
(hdbscan.rs's version `unreachable!`s on Cosine; centers.rs's version handles
Cosine). The euclidean/manhattan/chebyshev/minkowski arms are byte-identical
duplication. A future metric fix must be applied in two places or they drift.
**Fix:** Extract a single shared `host_pairwise` into a `hdbscan/distance.rs`
submodule with a `Cosine` arm, and have both callers use it.

### IN-04: `gen_oracle.py` is build-time-only and out of the runtime path

**File:** `scripts/gen_oracle.py` (entire HDBSCAN section ~924-1230)
**Issue:** The script regenerates committed `.npz` fixtures and is never run in
CI/tests (per the module memory note "fixtures are committed blobs, not
test-time"). The HDBSCAN generation logic correctly forces
`algorithm='generic'` for the hdbscan cross-check and asserts each non-default
knob diverges from default before writing. No defects found; noting that bugs
here cannot affect shipped behavior (only future fixture regeneration), so
review weight is low.
**Fix:** None.

### IN-05: `do_labelling` result over-sizes then truncates with a defensive `max`

**File:** `crates/mlrs-algos/src/cluster/hdbscan/select.rs:426,463`
**Issue:** `result` is allocated `root_cluster.max(n_samples)` then truncated to
`root_cluster`. Since `root_cluster == n_samples` by the condense relabel
convention (root relabels to `n_samples`), the `.max()` and the over-size are
dead defensive code in every reachable case. Harmless, but the defensive sizing
papers over WR-02 (where `n_samples` can be `0`) rather than asserting the
invariant.
**Fix:** Replace with an explicit invariant check
(`debug_assert_eq!(root_cluster, n_samples)`) and size to `n_samples` directly.

---

_Reviewed: 2026-06-24T07:23:37Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
