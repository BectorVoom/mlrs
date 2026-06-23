---
phase: 13-knn-graph-primitive-feasibility-keystone
verified: 2026-06-23T12:00:00Z
status: gaps_found
score: 3/4 must-haves verified
overrides_applied: 0
gaps:
  - truth: "For each metric, indices are set-equal to sklearn.neighbors.NearestNeighbors (with the matching metric) up to tie-ordering and distances match to <=1e-5 (f64), with the lowest-index tie-break documented as the mlrs convention — verifiable by re-running the canonical oracle generator"
    status: failed
    reason: "CR-01/CR-02 oracle integrity failure: the committed chebyshev fixtures were hand-regenerated in commit 7f73d4e to enforce the lowest-index tie-break, but scripts/gen_oracle.py (declared the canonical regeneration tool by its own module docstring) was NOT updated. Running 'python3 scripts/gen_oracle.py' today silently reverts the committed chebyshev blobs to sklearn-arbitrary-tie-order, breaking the gate. The committed artifact is not derivable from the committed generator. Additionally (CR-02), because the chebyshev fixture was conformed to the prim's own tie-break rather than an independent rule in the generator, the chebyshev index gate is partially circular at the tied boundary: a miscompile that happened to produce lowest-index selection would still pass."
    artifacts:
      - path: "scripts/gen_oracle.py"
        issue: "gen_knn_metric() calls nn.kneighbors(x) and saves indices without any lexsort/stable-argsort post-processing. Confirmed by reading lines 870-897: no re-sort after kneighbors. Running this script regenerates chebyshev fixtures with sklearn-arbitrary tie order (idx 4 at row 25), not the lowest-index (idx 0) that the committed blob carries."
      - path: "tests/fixtures/knn_chebyshev_f32_seed42.npz"
        issue: "Committed fixture was hand-patched in 7f73d4e (not via gen_oracle.py) to lowest-index tie-break at row 25. The generator does not reproduce it."
      - path: "tests/fixtures/knn_chebyshev_f64_seed42.npz"
        issue: "Same hand-patch; same non-reproducibility. Fixture sizes are 140 bytes smaller than the other metrics' fixtures, consistent with different array content not produced by the generator."
    missing:
      - "Add lowest-index lexsort to gen_knn_metric() in scripts/gen_oracle.py: after 'distances, indices = nn.kneighbors(x)', apply per-row 'order = np.lexsort((indices[r], distances[r]))' (primary: distance, secondary: neighbour index) for every metric. This makes the committed chebyshev blobs reproducible and makes the oracle independent of the prim's tie-break selection."
      - "Re-run gen_oracle.py and confirm the new chebyshev blobs are byte-identical to the hand-patched ones, then commit the generator change alongside."
      - "Optionally add an explicit row-25 assertion in knn_graph_test.rs that the chebyshev boundary tie resolves to the lowest index, so the convention is gated by a named human-readable expectation independent of the blob."
---

# Phase 13: KNN-Graph Primitive (PRIM-11) Verification Report

**Phase Goal:** Land the single shared KNN-graph primitive — ascending-ordered k-nearest-neighbor indices (n, k) + distances (n, k) over a multi-metric distance layer (Euclidean, Manhattan/L1, Cosine, Chebyshev/L∞, Minkowski-p), with a self-inclusion parameter — exposed as a new standalone `mlrs-backend` prim fn composed cpu-MLIR-safe from the launch-proven distance → top-k GATHER path (no SharedMemory/atomics/heap kernel), and standalone-validate it (per metric) BEFORE UMAP or HDBSCAN consume it. Emits the DIRECTED (indices, distances) graph only (symmetrization deferred to consumers). This is the milestone's feasibility keystone.
**Verified:** 2026-06-23T12:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| SC-1 | KNN-graph prim returns ascending-ordered (n,k) indices + distances with metric param + self-inclusion param, composed from distance→top-k GATHER, no new heap kernel, directed-only output | VERIFIED | `knn_graph<F>` at knn_graph.rs:114 returns `(DeviceArray<u32>, DeviceArray<F>)` (n,k). Metric enum (5 variants) at line 61. Composition: distance()→top_k()→self_drop_gather() wired at lines 49-50, 45. No new heap kernel. Symmetrization absent (directed only, per doc at line 9). `cargo build -p mlrs-backend --features cpu` exits 0. |
| SC-2 | Prim launches under --features cpu (not just compile) for every metric including new Manhattan/Chebyshev/Minkowski kernels; no Atomic/SharedMemory/F::INFINITY/mutable-bool; rocm f32 | VERIFIED | `cargo test -p mlrs-backend --features cpu --test knn_graph_test` 14/14 GREEN (2.59s). self_drop_gather_test 2/2 GREEN. distance.rs code lines contain no SharedMemory/Atomic/F::INFINITY (grep returns 0 matches). CUBE_POS_X/UNIT_POS_X==0 shape confirmed at distance.rs:201-203. SUMMARY-02 documents rocm f32 green / f64 skips-with-log. |
| SC-3 | For EACH metric, indices set-equal to sklearn NearestNeighbors up to tie-ordering; distances <=1e-5 f64; lowest-index tie-break documented AND reproducible via the canonical generator | FAILED (BLOCKER) | Test gate is GREEN (14/14) — but the chebyshev gate is partially circular. gen_oracle.py (declared canonical, module docstring: "canonical regeneration tool") does NOT apply lexsort after kneighbors (confirmed: lines 870-897 have no re-sort). Commit 7f73d4e regenerated knn_chebyshev_{f32,f64}_seed42.npz by hand without updating gen_oracle.py. Running the generator today reverts the fixtures. The oracle's independence from the prim's own tie-break is compromised for chebyshev's tied boundary. CR-01 and CR-02 from 13-REVIEW.md are confirmed. |
| SC-4 | Build-failing PoolStats memory gate passes (big distance operand query-axis tiled; never full n×n resident-and-leaking) | VERIFIED | knn_memory_gate_query_axis_tiled GREEN: peak_bytes=1464 << ITERS(4)×n×n(14400); live_bytes=0 after warmup (conserved); reuse_delta=333>0. QUERY_TILE=8 confirmed in knn_graph.rs:84. Hard assert!s at test lines 487-532 confirmed substantive. |

**Score:** 3/4 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-backend/src/prims/knn_graph.rs` | Metric enum + knn_graph<F> prim (validate-before-launch, metric routing, query-axis-tiled composition, self_drop_gather) | VERIFIED | 468 lines. pub enum Metric at line 61. pub fn knn_graph at line 114. validate_geometry at line 391. QUERY_TILE=8 at line 84. Uses distance(), top_k(), self_drop_gather(). |
| `crates/mlrs-kernels/src/distance.rs` | manhattan_dist/chebyshev_dist/minkowski_dist (2D feature-loop) + self_drop_gather (per-row GATHER) #[cube(launch)] kernels | VERIFIED | 223 lines. All four kernels defined. STATIC F::powf used (grep: 2 occurrences in code lines). CUBE_POS_X/UNIT_POS_X==0 shape in self_drop_gather. No forbidden constructs in code lines. |
| `crates/mlrs-backend/tests/knn_graph_test.rs` | Per-metric oracle harness + duplicate-point VALUE assert + geometry-rejection + query-axis memory gate | VERIFIED | 540 lines. All 6 required test functions present. BTreeSet set comparison at line 163. DIST_TOL=1e-5 at line 59. skip_f64_with_log at multiple call sites. |
| `scripts/gen_oracle.py` | gen_knn extended with metric/p params + duplicate-point design, producing reproducible fixtures | PARTIAL-STUB | gen_knn_metric() added with metric/p params (line 814). Duplicate-point design present (KNN_DUP_ROW_A/B). Fixtures committed. BUT: no lexsort post-processing — re-running the generator reverts chebyshev fixtures (CR-01). The generator does not produce the committed chebyshev blobs. |
| `tests/fixtures/knn_{metric}_{f32,f64}_seed42.npz` (10 files) | Per-metric oracle fixtures (5 metrics x f32+f64), duplicate-point bearing | VERIFIED (conditional) | All 10 files exist and have content. Tests load and use them correctly. Chebyshev fixtures exist in correct form for tests to pass — but they are not reproducible from gen_oracle.py. |
| `crates/mlrs-backend/tests/self_drop_gather_test.rs` | Launch smoke test proving non-zero readback + index-identity self-drop | VERIFIED | Exists (7900 bytes). 2/2 tests GREEN under --features cpu. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `knn_graph.rs` | `crates/mlrs-kernels/src/distance.rs` | `manhattan_dist`, `chebyshev_dist`, `minkowski_dist`, `self_drop_gather` | WIRED | knn_graph.rs:45 `use mlrs_kernels::{chebyshev_dist, manhattan_dist, minkowski_dist, self_drop_gather};`. All four launch in compute_tile_distance and self_drop_full. |
| `knn_graph.rs` | `crates/mlrs-backend/src/prims/topk.rs` | `top_k()` | WIRED | knn_graph.rs:50 `use crate::prims::topk::top_k;`. Called at knn_graph.rs:189 in tile loop. |
| `knn_graph.rs` | `crates/mlrs-backend/src/prims/distance.rs` | `distance()` GEMM path | WIRED | knn_graph.rs:49 `use crate::prims::distance::distance;`. Called in compute_tile_distance for Euclidean/Cosine arm. |
| `crates/mlrs-kernels/src/lib.rs` | `crates/mlrs-kernels/src/distance.rs` | `pub mod distance;` + `pub use distance::{...}` | WIRED | lib.rs:14 `pub mod distance;`. lib.rs:39 `pub use distance::{chebyshev_dist, manhattan_dist, minkowski_dist, self_drop_gather};`. |
| `crates/mlrs-backend/src/prims/mod.rs` | `knn_graph.rs` | `pub mod knn_graph;` | WIRED | prims/mod.rs:31 `pub mod knn_graph;`. |
| `knn_graph_test.rs` | `tests/fixtures/knn_*.npz` | `load_npz(fixture(name))` | WIRED | test:86-93 fixture() resolver. All 10 fixtures loaded via load_npz. |
| `scripts/gen_oracle.py` | `tests/fixtures/knn_chebyshev_*.npz` | gen_knn_metric() | BROKEN | Generator does not reproduce the committed chebyshev blobs (CR-01). The link exists in code but the output diverges on the tied boundary. |

---

## Data-Flow Trace (Level 4)

The prim is a compute function (not a UI component), but the data-flow from fixtures to assertion is the critical path:

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `knn_graph_test.rs` | `got_idx`, `got_dist` | `knn_graph<F>()` -> device arrays -> `.to_host()` | Yes — full pipeline from device computation | FLOWING |
| `knn_graph_test.rs` | `ref_dist`, `ref_idx` (chebyshev) | `knn_chebyshev_*.npz` via `load_npz` | Yes — but blob was hand-produced, not from gen_oracle.py | HOLLOW at generator level |
| `knn_graph<F>` | distance block | `compute_tile_distance()` -> device kernel launch | Yes — launches kernel, reads back real values | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 14 per-metric oracle + safety tests pass | `cargo test -p mlrs-backend --features cpu --test knn_graph_test` | 14 passed; 0 failed; finished in 2.59s | PASS |
| self_drop_gather launch smoke test (non-zero readback) | `cargo test -p mlrs-backend --features cpu --test self_drop_gather_test` | 2 passed; 0 failed; finished in 0.25s | PASS |
| mlrs-kernels bare build | `cargo build -p mlrs-kernels` | Finished dev profile in 0.06s | PASS |
| mlrs-backend cpu build | `cargo build -p mlrs-backend --features cpu` | Finished dev profile in 0.10s | PASS |
| gen_oracle.py parses | `python3 -c "import ast; ast.parse(open('scripts/gen_oracle.py').read())"` | parse-ok | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| PRIM-11 | 13-01, 13-02, 13-03 | Shared multi-metric KNN-graph prim with self-inclusion, standalone-validated per metric vs sklearn, build-failing memory gate, directed output only, cpu-MLIR-safe composition | PARTIALLY SATISFIED | All functional behaviors implemented and test-green. Oracle integrity gap (CR-01/CR-02) means the chebyshev validation is not independently reproducible. Core prim correctness is confirmed; validation pipeline reproducibility is broken for chebyshev. |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/gen_oracle.py` | 871 | `distances, indices = nn.kneighbors(x)` with no lexsort post-processing — documented as "canonical regeneration tool" but cannot reproduce the committed chebyshev fixtures | BLOCKER | Re-running the generator silently overwrites the hand-patched chebyshev blobs, reverting the tie-break fix and breaking the gate. This is an oracle reproducibility defect on the phase keystone. |

No TBD/FIXME/XXX debt markers found in any phase-13 modified file (knn_graph.rs, distance.rs, knn_graph_test.rs, gen_oracle.py).

---

## CR-01 and CR-02 Assessment (Critical Review Input)

The code review raised two BLOCKERs about oracle integrity. Both are independently verified here:

**CR-01 — Generator does not produce the committed chebyshev fixtures: CONFIRMED BLOCKER.**

Evidence:
- Commit `7f73d4e` modifies `tests/fixtures/knn_chebyshev_f32_seed42.npz` and `tests/fixtures/knn_chebyshev_f64_seed42.npz` but does NOT modify `scripts/gen_oracle.py` (confirmed: `git show 7f73d4e -- scripts/gen_oracle.py` returns empty).
- Reading `gen_knn_metric()` lines 870-897: after `distances, indices = nn.kneighbors(x)`, the function immediately wraps the arrays with `c()` and saves them. No lexsort, no argsort, no re-ordering of any kind.
- The generator's module docstring declares it the "canonical regeneration tool" whose output is "checked in so CI never runs this script." Running it today produces different chebyshev blobs than what is committed.
- The chebyshev fixtures are 140 bytes smaller than the other metrics' fixtures (3396 vs 3536 for f32), consistent with slightly different array content not produced by the same code path.

This constitutes a data-integrity gap at the oracle layer: the committed artifact (fixture blob) is not derivable from the committed generator, violating the reproducibility guarantee the generator explicitly claims.

**CR-02 — Chebyshev gate partial circularity: CONFIRMED, severity assessment is WARNING-level independent of CR-01.**

Evidence:
- The test `check_knn_metric` compares per-row index sets (`BTreeSet`) — at a k+1 distance boundary, changing the tie-break changes SET MEMBERSHIP, not just ordering. This is correctly identified by the review.
- The chebyshev fixture at row 25 was regenerated specifically to match the prim's own lowest-index pick (idx 0), as documented in the SUMMARY-03 key-decisions and the 7f73d4e commit message.
- If the prim's tie-break were misconfigured (e.g. accidentally yielding lowest-index by chance, not by design), the hand-edited fixture would still agree, so the index gate at the tied boundary is not fully independent.
- Mitigation: the distance value assertion still provides partial protection (sorted distance multisets must match within 1e-5). A wrong-index pick at the boundary generally produces a different distance value in the return set. However, for exact ties where both candidates have identical chebyshev distance, this protection is nullified.

The CR-02 issue is a weakening of the independence guarantee the phase's "standalone-validate against an INDEPENDENT sklearn oracle" claim requires. It does NOT make the gate completely toothless — the non-chebyshev oracle tests remain fully independent — but the chebyshev oracle's independence at the boundary is compromised.

**Combined verdict:** CR-01 alone is a BLOCKER because it makes the oracle generator and the committed fixture diverge, violating reproducibility. CR-02 is a compounding concern that weakens the chebyshev gate's detection power at the exact failure mode (boundary tie miscompile) the gate is supposed to catch. Together they mean "standalone-validate per metric against an INDEPENDENT sklearn oracle" is not fully achieved for Chebyshev.

The fix is straightforward and confined to one function in gen_oracle.py: add a per-row lexsort after `kneighbors` in `gen_knn_metric()`, re-run the generator, and verify the chebyshev blobs are byte-identical to the hand-patched ones.

---

## Human Verification Required

None — all required checks are programmatically verifiable.

---

## Gaps Summary

One gap blocks the phase goal:

**BLOCKER: Chebyshev oracle not reproducible from the canonical generator (CR-01 + CR-02)**

The phase goal requires standalone validation per metric against an INDEPENDENT sklearn oracle. For Chebyshev, the committed oracle fixture was manually patched to match the prim's tie-break without updating the canonical generator (`scripts/gen_oracle.py`). This creates two problems:

1. Re-running `python3 scripts/gen_oracle.py` silently reverts the chebyshev fixtures to sklearn-arbitrary tie order, breaking the gate (reproducibility failure — CR-01).
2. The chebyshev fixture was conformed to the prim's own tie-break rather than derived from an independent rule in the generator, making the chebyshev index gate partially circular at the tied boundary (independence failure — CR-02).

The four other metrics (Euclidean, Manhattan, Cosine, Minkowski) are unaffected: their oracle tests are independent, reproducible from gen_oracle.py, and green.

The functional implementation is complete and correct: knn_graph<F> compiles, launches, and passes all 14 test cases under --features cpu. The gap is entirely in the oracle pipeline for Chebyshev's tied boundary — specifically that gen_oracle.py does not encode the lowest-index tie-break that makes the committed fixture consistent and reproducible.

**Required fix:** Add `np.lexsort((indices[r], distances[r]))` post-processing for every row in `gen_knn_metric()` in `scripts/gen_oracle.py`, re-generate fixtures, and confirm chebyshev blobs are byte-identical to the hand-patched versions. No changes to the prim implementation or test assertions are required.

---

_Verified: 2026-06-23T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
