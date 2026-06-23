---
phase: 13-knn-graph-primitive-feasibility-keystone
plan: 01
subsystem: testing
tags: [knn, oracle-fixtures, cubecl, cpu-mlir, sklearn, prim-11, nyquist]

# Dependency graph
requires:
  - phase: 05-distance-topk-prims
    provides: distance.rs GEMM-expansion prim + topk.rs k-smallest select (the composition base)
  - phase: spike-13
    provides: spike-findings-mlrs skill (KNN-graph recipe + cpu-MLIR kernel-authoring landmines)
provides:
  - Per-metric KNN oracle fixtures (5 metrics x f32+f64, X-vs-X, k+1 self-inclusive, duplicate-point-bearing)
  - gen_knn_metric(seed, dtype, metric, p) generator in scripts/gen_oracle.py
  - mlrs-kernels::distance empty-but-registered kernel module (cpu-MLIR authoring-contract doc)
  - mlrs-backend::prims::knn_graph empty compiling prim shell + registration
  - knn_graph_test.rs Nyquist Wave-0 oracle harness (RED-by-design pending plan 13-03)
affects: [13-02-kernels, 13-03-prim, umap-phase-14, hdbscan-phase-15]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-metric X-vs-X KNN oracle with duplicate-point row for the R-9 silent-miscompile gate"
    - "Index SET-equality (BTreeSet, up to tie-ordering) + 1e-5 relative-tol distance compare"
    - "Wave-1 empty-but-registered module scaffold (kernel + prim) deferring bodies to later plans"

key-files:
  created:
    - crates/mlrs-kernels/src/distance.rs
    - crates/mlrs-backend/src/prims/knn_graph.rs
    - crates/mlrs-backend/tests/knn_graph_test.rs
    - tests/fixtures/knn_{euclidean,manhattan,cosine,chebyshev,minkowski}_{f32,f64}_seed42.npz
  modified:
    - scripts/gen_oracle.py
    - crates/mlrs-kernels/src/lib.rs
    - crates/mlrs-backend/src/prims/mod.rs

key-decisions:
  - "Fixtures routed to workspace-root tests/fixtures/ (the real _FIXTURE_DIR / fixture() location), not the crate-relative path the plan named"
  - "Metric tag carried in the FILENAME only, never an in-blob string array (load_npz decodes only 4/8-byte float arrays)"
  - "Minkowski test exponent fixed at p=3.0 (non-degenerate, exercises the general kernel not an L1/L2 fast path)"
  - "Duplicate-point design: train rows 0 and 4 made identical for the R-9 value gate"

patterns-established:
  - "metric_oracle_pair! macro emits the f64 (cpu-gate, skip_f64_with_log) + f32 (cpu+rocm) test pair per metric"
  - "Memory gate asserts sub-quadratic peak residency + live_bytes conservation + scratch reuse growth via PoolStats"

requirements-completed: [PRIM-11]

# Metrics
duration: 6min
completed: 2026-06-23
---

# Phase 13 Plan 01: KNN-Graph Nyquist Wave-0 Harness Summary

**Per-metric duplicate-point-bearing sklearn KNN oracle fixtures (5 metrics x f32/f64), the RED-by-design oracle/geometry/dup-point-value/memory-gate test harness, and the compiling distance-kernel + knn_graph-prim scaffolds — the Nyquist Wave-0 gate PRIM-11 plans 02/03 land GREEN against.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-06-23T04:05:42Z
- **Completed:** 2026-06-23T04:11:56Z
- **Tasks:** 3
- **Files modified/created:** 16 (3 source files, 10 fixtures, 3 modified)

## Accomplishments

- Extended `gen_oracle.py` with `gen_knn_metric(seed, dtype, metric, p)` over the full fixed metric set (Euclidean, Manhattan, Cosine, Chebyshev, Minkowski-p=3.0), X-vs-X, requesting `k+1` self-inclusive neighbours, with a deliberate duplicate-point train row (rows 0 and 4 identical) for the R-9 silent-miscompile gate. Committed 10 `.npz` fixtures (5 metrics x f32+f64) to `tests/fixtures/`. Existing Euclidean `knn_{f32,f64}_seed42.npz` left untouched (no regression to topk/neighbors suites).
- Scaffolded `crates/mlrs-kernels/src/distance.rs` as a registered empty module carrying the full cpu-MLIR authoring-contract doc (STATIC `F::powf`, STATEMENT-form running max, 2D `ABSOLUTE_POS_{X,Y}` pairwise launch, `CUBE_POS_X`/`UNIT_POS_X==0` self-drop GATHER, no cross-sibling-loop accumulator / SharedMemory / Atomic) — bodies deferred to plan 13-02. Created the empty `prims/knn_graph.rs` shell and registered both modules; `mlrs-kernels` builds bare and `mlrs-backend` builds `--features cpu`.
- Wrote `knn_graph_test.rs` (540 lines): per-metric index SET-equality + 1e-5 distance oracle (all 5 metrics x {f32,f64}), `knn_rejects_bad_geometry` (`k>n-1`/`p<1`/`n*d!=len` -> `ShapeMismatch{operand}`), the load-bearing R-9 `knn_self_drop_duplicate_point_value` VALUE gate (the only catch for FINDING 002-B), `knn_include_self_returns_self_at_col0` (HDBSCAN core-dist), and the `knn_memory_gate_query_axis_tiled` PoolStats gate. RED-by-design with an E0432 unresolved-symbol on `knn_graph`/`Metric`.

## Task Commits

Each task was committed atomically:

1. **Task 1: per-metric KNN oracle fixtures + duplicate-point design** - `625de38` (feat)
   - **Rule 1 fix (split commit):** `939c9f4` (fix) — dropped the in-blob string `metric` array
2. **Task 2: scaffold distance kernel module + knn_graph prim registration** - `091a28a` (feat)
3. **Task 3: knn_graph oracle harness (RED-by-design)** - `39ebb50` (test)

_Note: the Task-1 Rule-1 fix landed as a separate `fix(13-01)` commit because Task 2 was already committed on top of Task 1 (no in-place amend across an intervening commit)._

## Files Created/Modified

- `scripts/gen_oracle.py` - Added `gen_knn_metric()` + constants (`KNN_METRIC_P=3.0`, `KNN_DUP_ROW_A/B`); wired into `main()`. Existing `gen_knn` untouched.
- `tests/fixtures/knn_{metric}_{f32,f64}_seed42.npz` (10 files) - sklearn `NearestNeighbors` X-vs-X oracles; arrays `X, k, distances, indices, p, dup_row_a, dup_row_b` (all float — no string array).
- `crates/mlrs-kernels/src/distance.rs` - Registered empty module + cpu-MLIR authoring-contract doc; bodies owned by plan 13-02.
- `crates/mlrs-kernels/src/lib.rs` - `pub mod distance;` registration.
- `crates/mlrs-backend/src/prims/knn_graph.rs` - Empty compiling prim shell; body owned by plan 13-03.
- `crates/mlrs-backend/src/prims/mod.rs` - `pub mod knn_graph;` registration.
- `crates/mlrs-backend/tests/knn_graph_test.rs` - The full Wave-0 oracle/geometry/dup-point/memory-gate harness (RED until plan 13-03).

## Decisions Made

- **Fixture location:** routed to workspace-root `tests/fixtures/` (where `_FIXTURE_DIR` and the topk_test `fixture()` resolver actually point) rather than the `crates/mlrs-backend/tests/fixtures/` path the plan's prose named — the latter does not exist and would not be found by the test resolver.
- **No in-blob metric string:** the metric tag is encoded in the filename only; `mlrs_core::load_npz` decodes only 4/8-byte float arrays.
- **Minkowski p=3.0** so the oracle exercises the genuine general-exponent path, not an L1/L2 special-case.
- **Index SET-equality** (per-row `BTreeSet`) rather than exact-position compare, since cross-metric tie-ordering differs (PRIM-11).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed in-blob unicode `metric` array that breaks load_npz**
- **Found during:** Task 3 (writing the harness, while confirming fixture loading)
- **Issue:** Task 1's `gen_knn_metric` stored the metric name as a numpy `<U9` unicode array. `mlrs_core::load_npz` (oracle.rs:127) decodes ONLY 4/8-byte float arrays and returns `InvalidData` on any other dtype — so every new fixture would have failed to load in the consuming Rust test, silently defeating the whole harness.
- **Fix:** Dropped the `metric=` field from the generator (the tag is already in the filename; the float `p` carries the only metric-dependent scalar the test needs). Regenerated all 10 fixtures and verified each loads via `mlrs_core::load_npz` from Rust (indices shape `[30,6]`, `p`/`dup_row_a` readable).
- **Files modified:** `scripts/gen_oracle.py`, all 10 `knn_{metric}_*.npz`
- **Verification:** A throwaway Rust binary linking `mlrs-core` loaded all 10 fixtures and read every numeric array (ALL FIXTURES LOAD).
- **Committed in:** `939c9f4`

**2. [Rule 3 - Blocking] Build-verify command corrected (mlrs-kernels has no `cpu` feature)**
- **Found during:** Task 2 (build verification)
- **Issue:** The plan's verify ran `cargo build -p mlrs-kernels --features cpu`, but `mlrs-kernels` is intentionally backend-feature-free (it has no `cpu` feature — that lives on `mlrs-backend`/`mlrs-algos`/`mlrs-py`). The command errors with "the package 'mlrs-kernels' does not contain this feature: cpu".
- **Fix:** Built `mlrs-kernels` bare (it is generic-over-runtime) and `mlrs-backend --features cpu`. Both compile clean.
- **Files modified:** none (verification-command adjustment only)
- **Verification:** `cargo build -p mlrs-kernels` and `cargo build -p mlrs-backend --features cpu` both finish 0.
- **Committed in:** N/A (no code change)

---

**Total deviations:** 2 (1 Rule-1 bug fix, 1 Rule-3 verify-command correction)
**Impact on plan:** The Rule-1 fix was essential — without it the entire RED harness would fail at fixture-load, not at the intended missing-symbol gate. No scope creep; both adjustments keep the plan's gate honest.

## Issues Encountered

- The plan referenced `crates/mlrs-backend/tests/fixtures/` for fixture output and verification greps; the real fixture directory is the workspace-root `tests/fixtures/` (per `_FIXTURE_DIR` and the `fixture()` resolver in `topk_test.rs`). Routed to the correct location; the acceptance count (>=10 metric fixtures committed) holds there.

## Expected-RED State (for plans 02/03)

`cargo test -p mlrs-backend --features cpu --test knn_graph_test --no-run` fails with:

```
error[E0432]: unresolved imports `mlrs_backend::prims::knn_graph::knn_graph`,
              `mlrs_backend::prims::knn_graph::Metric`
```

This is the REAL gate, not a stubbed pass — the cascading E0308 mismatches are downstream of `Metric` failing to resolve. Plan 13-02 lands the direct distance + `self_drop_gather` kernels in `mlrs-kernels::distance`; plan 13-03 lands `Metric` + `knn_graph` in `prims/knn_graph.rs`, at which point this harness turns GREEN. The harness's contract for them:

- `knn_graph::<F>(pool, &x_dev, (n, d), k, metric, include_self, p) -> Result<(DeviceArray<_, u32> indices, DeviceArray<_, F> distances), PrimError>` — `(n, k)` outputs, indices first.
- `Metric::{Euclidean, Manhattan, Cosine, Chebyshev, Minkowski { p: f64 }}`.
- Geometry guards return `PrimError::ShapeMismatch { operand }` with operands `"k"` / `"p"` / `"x"` BEFORE launch.
- include_self=false drops self by INDEX IDENTITY (D-02); include_self=true returns self at col 0.

## Threat Flags

None — no new security-relevant surface. Fixtures are committed offline-generated trusted blobs; no runtime/network ingress (matches the plan's threat register: T-13-01 mitigated by the R-9 value gate, T-13-02/T-13-SC accepted).

## Next Phase Readiness

- Wave-1 scaffolds compile; the oracle gate is in place and RED-by-design. Plan 13-02 (kernels) and 13-03 (prim) can land RED->GREEN against a real value-asserting gate.
- No blockers.

## Self-Check: PASSED

All 5 created source files + 10 fixtures present on disk; all 4 task/fix commits (`625de38`, `939c9f4`, `091a28a`, `39ebb50`) exist in git history.

---
*Phase: 13-knn-graph-primitive-feasibility-keystone*
*Completed: 2026-06-23*
