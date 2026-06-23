---
phase: 13-knn-graph-primitive-feasibility-keystone
reviewed: 2026-06-23T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - crates/mlrs-backend/src/prims/knn_graph.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/tests/knn_graph_test.rs
  - crates/mlrs-backend/tests/self_drop_gather_test.rs
  - crates/mlrs-kernels/src/distance.rs
  - crates/mlrs-kernels/src/lib.rs
  - scripts/gen_oracle.py
findings:
  critical: 2
  warning: 5
  info: 4
  total: 11
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-06-23
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

The KNN-graph primitive (PRIM-11) is a host orchestrator composing the launch-proven
`distance`/`top_k` prims with three new cpu-MLIR-safe direct distance kernels
(manhattan/chebyshev/minkowski) and a self-drop GATHER kernel. The kernel authoring
respects the documented cpu-MLIR contract (no SharedMemory / Atomic / infinity / mutable-bool
scan; statement-form running max; static `F::powf`; per-row GATHER launch shape). The
Cosine `1−cos` math, the Euclidean deferred-sqrt boundary, the index-identity self-drop, and
the host-side geometry validation are all correct as written.

Two BLOCKERs dominate the assessment, both touching the integrity of the oracle gate this
phase is supposed to be the keystone of:

1. **The chebyshev fixtures were hand-regenerated outside the canonical generator**
   (`scripts/gen_oracle.py` was NOT updated). Re-running the documented regeneration tool
   silently REVERTS the committed fixture, reintroducing the tie-break divergence the GREEN
   commit fixed. The committed blob and the generator that is supposed to produce it now disagree.

2. **The chebyshev fixture was patched to match the prim's tie-break rather than an independent
   oracle.** Combined with the set-based index assertion, this weakens the chebyshev gate's
   ability to catch a real boundary miscompile.

The remaining findings are robustness / honesty-of-comment / dead-code items.

## Critical Issues

### CR-01: Chebyshev oracle fixtures are unreproducible — `gen_oracle.py` does not produce the committed blobs

**File:** `scripts/gen_oracle.py:814-897` (`gen_knn_metric`); committed blobs `tests/fixtures/knn_chebyshev_f{32,64}_seed42.npz`
**Issue:**
Commit `7f73d4e` regenerated `knn_chebyshev_f32_seed42.npz` and `knn_chebyshev_f64_seed42.npz`
"with a stable (lowest-index) argsort so the oracle matches the PRIM-11 convention", but it
did **not** change `gen_oracle.py`. The generator still does:

```python
nn = NearestNeighbors(n_neighbors=k_query, algorithm="brute", metric=metric, p=p_arg).fit(x)
distances, indices = nn.kneighbors(x)   # sklearn's arbitrary boundary-tie order
...
np.savez(out_path, ..., distances=c(distances), indices=c(indices), ...)
```

`sklearn.NearestNeighbors.kneighbors` does NOT guarantee a lowest-index tie-break at a
distance boundary (the commit message itself documents row 25 of chebyshev getting idx 4 from
sklearn vs idx 0 under lowest-index). The module docstring declares this script the
"*canonical* regeneration tool" whose output is "checked in so CI never runs this script."
Running `python3 scripts/gen_oracle.py` today will OVERWRITE the hand-patched chebyshev blobs
with sklearn-tie-order blobs, reverting the fix and turning the chebyshev oracle test red (or,
worse, leaving a divergent blob that someone re-commits). The committed artifact is no longer
derivable from the committed generator — a reproducibility/data-integrity defect on the phase
keystone.

**Fix:** Encode the lowest-index tie-break in the generator so the committed blob is
reproducible. After `kneighbors`, re-sort each row by `(distance, index)` lexicographically
for ALL metrics (a stable secondary key on the neighbour index), e.g.:

```python
distances, indices = nn.kneighbors(x)
# Lowest-index tie-break: stable lexsort by (distance, neighbour index) per row so the
# committed oracle matches the PRIM-11 top_k convention and is reproducible by this script.
for r in range(distances.shape[0]):
    order = np.lexsort((indices[r], distances[r]))  # primary=distance, secondary=index
    distances[r] = distances[r][order]
    indices[r] = indices[r][order]
```

Then re-run the generator and confirm the committed chebyshev blobs are byte-identical to the
hand-patched ones; commit the generator change alongside. (This also future-proofs the other
four metrics against the same latent divergence the commit message says it "audited" manually.)

### CR-02: Chebyshev gate compares against a fixture edited to match the implementation, not an independent oracle

**File:** `crates/mlrs-backend/tests/knn_graph_test.rs:144-168` (set comparison); fixture provenance per commit `7f73d4e`
**Issue:**
`check_knn_metric` asserts per-row index **set** equality (`BTreeSet`) and sorted-distance
equality. At a `k+1` distance boundary a tie means one tied index is inside the window and one
is outside; changing the tie-break therefore changes the **set** membership, not merely the
ordering. The chebyshev fixture was regenerated specifically so its boundary index matches the
prim's lowest-index pick. Because the fixture was conformed to the implementation's tie-break
(and per CR-01 that re-sort lives only as a one-off edit, not in the generator), the chebyshev
oracle no longer independently verifies the prim's boundary selection: if the prim's tie-break
were itself miscompiled to (say) "lowest index" by accident vs. by design, the hand-edited
fixture would still agree. The phase's stated purpose is an INDEPENDENT sklearn oracle; for
chebyshev that independence has been partially traded away to make the gate green.

The distance-value assertion still provides SOME protection (the sorted distance vectors must
match within 1e-5, and a wrong-index pick at the boundary generally also perturbs the distance
multiset). But the index gate specifically is now circular for the tied-boundary case.

**Fix:** Land CR-01 (put the lowest-index lexsort in the generator) so the fixture is derived
by an INDEPENDENT rule (lexicographic `(distance, index)`) rather than copied from the prim's
output, then document in the test that chebyshev's tied-boundary index is pinned by the
generator's lexsort, not by the implementation. Optionally add an explicit assertion that the
chebyshev row-25 boundary tie resolves to the lowest index, so the convention is gated by a
named, human-readable expectation independent of the blob.

## Warnings

### WR-01: `compute_tile_distance` doc-comment describes the OPPOSITE of the Cosine code path

**File:** `crates/mlrs-backend/src/prims/knn_graph.rs:256-260`
**Issue:**
The comment for the `Euclidean | Cosine` arm states: *"the boundary sqrt recovers √(2·(1−cos))
and the oracle compares in that space ≤1e-5."* But the actual code sets `needs_sqrt =
matches!(metric, Metric::Euclidean)` (line 141) — Cosine does NOT get the boundary sqrt; it
selects on the squared value `2(1−cos)` and is then HALVED host-side to `1−cos` (lines 144,
202-206). The oracle is `metric='cosine'` = `1−cos`, NOT `√(2(1−cos))`. The comment contradicts
both the code and the actual (correct) behaviour, and will mislead the next maintainer into
thinking Cosine sqrt-s. The code is right; the comment is wrong.

**Fix:** Replace the Cosine portion of the comment, e.g.: *"Cosine's normalised-row GEMM gives
2·(1−cos) (squared Euclidean of unit vectors); top_k selects on that order-preserving value
with NO boundary sqrt, and the returned k values are halved host-side (line 202) to the true
cosine distance 1−cos."*

### WR-02: `validate_geometry` u32-overflow loop omits `n` (rows), the largest launched dimension

**File:** `crates/mlrs-backend/src/prims/knn_graph.rs:430-441`
**Issue:**
The overflow guard iterates `[("x", n), ("x", d), ("k", k + 1)]`, but the kernels are launched
with `n as u32` in MANY more places than the train-row dimension: `compute_tile_distance` casts
`n as u32` (line 289/293/297) for `rows_y`, and `self_drop_full` casts `n as u32` (line 354)
for `rows`. The guard DOES include `n`, so the `n` dimension is covered. However `k + 1` can
itself overflow `usize` addition if `k == usize::MAX` BEFORE the loop computes `k + 1` — `k` is
already bounded by `k <= max_k <= n` at lines 409-417, and `n` is bounded later, so in practice
`k` is small. The real gap: `tile` (≤ `QUERY_TILE` = 8) is fine, but the **product**
`out_len = tile * n` (line 272) and `n * k_internal` (line 168) host allocations are not
overflow-checked; on a 32-bit host `n * k_internal` could wrap. This is a robustness gap, not a
live bug on 64-bit targets.

**Fix:** Add `n.checked_mul(k + 1)` and `tile.checked_mul(n)` guards (or assert the host is
64-bit), and reject overflow with `ShapeMismatch` as the existing precedent does. At minimum
add a comment that the host-buffer products assume a 64-bit `usize`.

### WR-03: `self_drop_full` fallback (self absent) silently drops the FARTHEST neighbour with no diagnostic

**File:** `crates/mlrs-kernels/src/distance.rs:191-223`; `crates/mlrs-backend/src/prims/knn_graph.rs:230`
**Issue:**
The self-drop kernel's R-3 fallback: if the query row's own index is absent from the
top-`(k+1)` (cannot happen for X-vs-X, but the prim does not assert X-vs-X), `bump` stays 0 for
every `s`, so `src = s` and the kernel silently returns the first `k` of the `k+1` neighbours —
dropping column `k` (the farthest). For a genuine X-vs-X graph self is always present, so this
is benign today. But `knn_graph` is a public prim with no runtime guarantee that the caller
passes X-vs-X (the signature only takes one matrix, so it IS always X-vs-X — acceptable), yet
the kernel is independently launchable (see `self_drop_gather_test.rs`) where the invariant is
NOT enforced. If a future caller feeds a `top_k` result where self legitimately fell outside
`k+1`, the kernel drops a real neighbour with no error. This is a latent correctness trap
guarded only by an undocumented precondition.

**Fix:** Acceptable for this phase given the single-matrix X-vs-X signature, but document the
precondition explicitly at the `knn_graph` API (self MUST be in the top-`(k+1)`, which holds
for X-vs-X) and consider a debug-only host assertion in `self_drop_full` that each row's
`tk_idx_full` window contains `row` before launch.

### WR-04: Per-call host round-trips (`to_host`) on every tile defeat the "device-resident" contract and stress the memory gate's assumptions

**File:** `crates/mlrs-backend/src/prims/knn_graph.rs:150, 195-196, 226-227, 327-328`
**Issue:**
The orchestrator reads `x` to host (line 150), reads every tile's `top_k` result back to host
(lines 195-196), then re-uploads the assembled buffers (lines 226-227 / 327-328). This is the
documented host-segment composition pattern and is functionally correct, but it means the
graph result makes a full device→host→device round-trip, and the per-tile `to_host` calls use
the UNMETERED `to_host` (not `to_host_metered`). The memory gate test
(`knn_memory_gate_query_axis_tiled`) asserts `live_bytes` exactly equals a baseline across
iterations and `reuses > 0`; whether those hold depends on the pool accounting for these
unmetered reads/uploads consistently. The correctness risk is low, but the "NO host round-trip"
spirit of the sibling `top_k`/`distance` prims (which keep results device-resident) is broken
here, and the project's memory-efficiency-is-first-class constraint is only met in the
sub-quadratic-residency sense, not the zero-copy sense.

**Fix:** Out of v1 scope as a performance item, but flag for the consumer phases (UMAP/HDBSCAN):
document that `knn_graph` currently returns via a host round-trip, and consider a device-side
gather/assemble for the tiled top_k results in a follow-up so the result never leaves the device.

### WR-05: `include_self=true` with `k == n` selects all rows but the duplicate-row test path can mask an off-by-one in self placement

**File:** `crates/mlrs-backend/src/prims/knn_graph.rs:409`; `crates/mlrs-backend/tests/knn_graph_test.rs:409-434`
**Issue:**
For `include_self=true`, `max_k = n`, so `k` may equal `n` and `k_internal = k = n`, passing
`top_k(n)` over an `n`-column block (allowed: `k <= cols == n`). The `knn_include_self_returns_self_at_col0`
test asserts self@col0 only for NON-duplicate rows and skips both duplicate rows entirely
(lines 415-423), accepting "either self or its duplicate" at col 0. That skip is justified by
the genuine distance-0 tie, but it means the duplicate rows contribute ZERO verification of
self placement under `include_self=true`. Since the duplicate rows are exactly where the
lowest-index tie-break matters most, the test's strongest case is excused. Not a code bug, but
a coverage gap that lets a self-placement regression at a distance-0 tie pass.

**Fix:** Strengthen the duplicate-row branch to assert col-0 index is EITHER `row` OR the
partner duplicate index (a concrete two-valued check), not merely that the distance is ~0 —
that pins the lowest-index tie-break instead of waiving it.

## Info

### IN-01: Stale scaffold comments in `mod.rs` / `lib.rs` describe modules as "empty compiling shell until then"

**File:** `crates/mlrs-backend/src/prims/mod.rs:25-31`; `crates/mlrs-kernels/src/lib.rs:9-14, 36-39`
**Issue:** The registration comments still say the KNN modules are "Empty compiling shell until
then" / "Empty compiling module until then", but plans 13-02/13-03 have landed the bodies. The
comments are now historical noise that misdescribe the current state.
**Fix:** Update the comments to past tense ("landed by plan 13-02/03") or remove the
scaffold-era prose.

### IN-02: `metric` argument is `Copy` but `compute_tile_distance` re-matches it twice

**File:** `crates/mlrs-backend/src/prims/knn_graph.rs:255-301`
**Issue:** The function matches `metric` to route, then inside the direct-kernel arm matches
`metric` AGAIN (line 286) with an `unreachable!()` default. This is correct but slightly
redundant; the inner match could destructure the outer arm. Minor readability.
**Fix:** Optional — collapse to a single match or pass an enum discriminant; low value.

### IN-03: `p` is passed both as enum field `Metric::Minkowski{p}` AND as a separate `p` argument, validated for agreement only loosely

**File:** `crates/mlrs-backend/src/prims/knn_graph.rs:420-429`; called at `tests/knn_graph_test.rs:124-133`
**Issue:** `validate_geometry` checks both `mp >= 1.0` AND `p >= 1.0` but never checks `mp == p`.
The kernel uses the SEPARATE `p` argument (line 298), not the enum's `mp`. A caller could pass
`Metric::Minkowski{p: 3.0}` with argument `p = 5.0` and get a graph under exponent 5 while the
type says 3. The test helper `metric_p` keeps them in sync, so this is latent. Dual-source-of-
truth for the same scalar is an API smell.
**Fix:** Either drop the standalone `p` argument and read it from the enum, or assert `mp == p`
in `validate_geometry` and reject the mismatch.

### IN-04: `f64_to_host::<F>(0.0)` initialiser allocates a full zero vector then overwrites every slot

**File:** `crates/mlrs-backend/src/prims/knn_graph.rs:168-169`
**Issue:** `tk_val_full` is zero-initialised via `vec![f64_to_host::<F>(0.0); n * k_internal]`
then fully written tile-by-tile, so the init value is dead. Harmless; the loop always covers all
slots because tiles partition `0..n` exactly. Noted only because if a tile gather were ever
short (it is not), the zero-init would silently mask it.
**Fix:** None required; optionally `Vec::with_capacity` + extend, but the current form is clear.

---

_Reviewed: 2026-06-23_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
