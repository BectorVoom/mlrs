# Phase 13: KNN-Graph Primitive (feasibility keystone) - Pattern Map

**Mapped:** 2026-06-23
**Files analyzed:** 5 to create + 2 to modify = 7
**Analogs found:** 7 / 7 (all have strong in-repo analogs; kernels also have VALIDATED verbatim spike shapes)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-backend/src/prims/knn_graph.rs` (NEW) | prim (host orchestrator) | transform (compose + tile) | `crates/mlrs-backend/src/prims/topk.rs` + `distance.rs` | exact (same prim shape; composes both) |
| `crates/mlrs-kernels/src/distance.rs` (NEW) — `manhattan_dist` / `chebyshev_dist` / `minkowski_dist` | kernel (device) | transform (2D per-element feature loop) | `mlrs-kernels/src/elementwise.rs::dist_combine_clamp` + spike 001 verbatim | exact (2D launch idiom) + VALIDATED spike shape |
| `self_drop_gather` kernel (in NEW `distance.rs` or `elementwise.rs`) | kernel (device) | transform (per-row GATHER) | `mlrs-kernels/src/topk.rs::select_k` + spike 002 verbatim | exact (`CUBE_POS_X`/`UNIT_POS_X==0` shape) + VALIDATED spike shape |
| `crates/mlrs-backend/src/prims/mod.rs` (MODIFY) | config (module registration) | — | existing `pub mod distance;` lines | exact |
| `crates/mlrs-kernels/src/lib.rs` (MODIFY) | config (module + re-export) | — | existing `pub mod` + `pub use` lines | exact |
| `crates/mlrs-backend/tests/knn_graph_test.rs` (NEW) | test (oracle + memory gate) | request-response (load → compose → assert) | `crates/mlrs-backend/tests/topk_test.rs` + `memory_gate_test.rs` | exact |
| `scripts/gen_oracle.py` (MODIFY) — per-metric `gen_knn` | utility (fixture generator) | batch (sklearn → `.npz`) | existing `gen_knn` (Euclidean only) | role-match (extend with `metric`/`p`) |

**No `Metric` enum file is listed separately** — per CONTEXT D-03/Claude's discretion, the `Metric` enum lives in `knn_graph.rs` alongside the prim (mirroring how `prims/kernel_matrix.rs` owns its `Kernel<F>` enum — see `prims/mod.rs:21-24`).

## Pattern Assignments

### `crates/mlrs-backend/src/prims/knn_graph.rs` (prim host orchestrator, transform)

**Analog:** `crates/mlrs-backend/src/prims/topk.rs` (signature/validation/pool shape) + `prims/distance.rs` (multi-stage composition + scratch release for tiling).

**Public signature pattern** — copy the `topk.rs` shape (`pool` first, geometry-tuples, `Option<DeviceArray>` reuse outputs, `Result<_, PrimError>`, `where F: Float + CubeElement + Pod`, `#[allow(clippy::too_many_arguments)]`). topk.rs lines 60-72:
```rust
#[allow(clippy::too_many_arguments)]
pub fn top_k<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    dist: &DeviceArray<ActiveRuntime, F>,
    rows: usize, cols: usize, k: usize, sqrt: bool,
    out_val: Option<DeviceArray<ActiveRuntime, F>>,
    out_idx: Option<DeviceArray<ActiveRuntime, u32>>,
) -> Result<(DeviceArray<ActiveRuntime, F>, DeviceArray<ActiveRuntime, u32>), PrimError>
where F: Float + CubeElement + Pod {
```
For `knn_graph` the new params are `x` + `(n, d)`, `k`, `metric: Metric`, `include_self: bool`, and (per CONTEXT Claude's-discretion / RESEARCH Open Q3) `p: f64` carried inside `Metric::Minkowski { p }` or as a separate arg. Return `(indices: DeviceArray<_, u32>, distances: DeviceArray<_, F>)` `(n, k)`.

**Validate-before-launch pattern** — copy `topk.rs` `validate_geometry` (lines 148-210) verbatim in spirit: a private `fn validate_geometry(...) -> Result<(), PrimError>` called BEFORE any `unsafe` launch. Reuse `PrimError::ShapeMismatch` for ALL geometry/`k` violations (there is no `InvalidK`/`InvalidArg` variant — confirmed in `mlrs-core/src/error.rs:66-155`; variants are `ShapeMismatch`, `DimMismatch`, `NotSquare`, `NotConverged`, `NotPositiveDefinite`). New guards this prim adds (RESEARCH Pitfalls 4 & 6):
- `n * d == x.len()` (operand `"x"`).
- `1 <= k` and, when `include_self == false`, `k + 1 <= n` i.e. `k <= n - 1` (operand `"k"`).
- when `Metric::Minkowski { p }`: `p >= 1.0` host-side (operand `"p"` via `ShapeMismatch`, since `PrimError` has no numeric-range variant — same synthetic-operand idiom topk.rs uses for `"k"` at lines 167-174).
- u32-overflow guard on `n`/`d`/`k` (copy topk.rs lines 178-187, WR-03).

**Composition pattern (the heart of the prim)** — model on `distance.rs` lines 94-181 (multi-stage device-resident pipeline with `release_into(pool)` on transient scratch). Concrete flow per RESEARCH diagram + spike 002:
1. `K = if include_self { k } else { k + 1 }`.
2. Metric routing:
   - `Euclidean` → `distance::<F>(pool, x, (n,d), x, (n,d), false, ...)` (the GEMM fast path; `sqrt=false` — defer to top_k).
   - `Cosine` → L2-normalize rows (host glue reusing `row_reduce` SumSq + `scale`, see `distance.rs:115-118` precedent) then `distance()`; result `1 − x̂·ŷ`.
   - `Manhattan`/`Chebyshev`/`Minkowski` → launch the matching NEW direct kernel (launch idiom below).
3. `top_k::<F>(pool, &dist, n, K_cols, K, sqrt = matches!(metric, Euclidean|Cosine), ...)`.
4. If `include_self == false`: launch `self_drop_gather` → `(n, k)` directed output.
   If `include_self == true`: the `top_k(k)` result IS the output.
5. Release transient scratch (`dist`, the `(n, k+1)` top_k intermediate when self-dropping) via `release_into(pool)` — copy `distance.rs:164-166`.

**Memory gate / query-axis tiling pattern (R-6 / Pitfall 5)** — there is NO existing tiling loop to copy; the pattern to follow is `distance.rs`'s `release_into(pool)` scratch-release (lines 156-166) applied inside a host loop over query-row blocks. Process a block of query rows, top_k that block, `release_into` its distance scratch, advance. Tile size is Claude's discretion (RESEARCH Open Q2). The asserting gate lives in the test file (below).

**Host-launch idiom for the new 2D distance kernels** (cubecl 0.10) — copy `distance.rs` lines 131-154 exactly: `let client = pool.client().clone();` → `launch_dims_2d` (lines 234-243, 16×16 ceiling-div) → `unsafe { ArrayArg::from_raw_parts(handle, len) }` per operand → `kernel::launch::<F, ActiveRuntime>(&client, count, dim, args…, scalars_by_value)`. **Scalars pass BY VALUE in cubecl 0.10 — no `ScalarArg` wrapper** (distance.rs comment line 151; topk.rs line 118).

---

### `crates/mlrs-kernels/src/distance.rs` — `manhattan_dist` / `chebyshev_dist` / `minkowski_dist` (device kernels, 2D feature-loop)

**Analog:** `mlrs-kernels/src/elementwise.rs::dist_combine_clamp` (lines 437-457) for the 2D launch + bounds-check + STATEMENT-form idiom; **VALIDATED verbatim shape** in `Skill("spike-findings-mlrs")` `sources/001-.../kernels_and_harness.rs` lines 27-114.

**Structural pattern (from `dist_combine_clamp`):** `#[cube(launch)] pub fn name<F: Float + CubeElement>(... rows: u32, cols: u32)`, `let i = ABSOLUTE_POS_X; let j = ABSOLUTE_POS_Y; if i < rows && j < cols { ... }`, scalar dims as `u32` by value, no SharedMemory/atomics/`F::INFINITY`.

**Minkowski (the named cpu-MLIR unknown — VALIDATED, spike 001 lines 85-114) — copy verbatim:**
```rust
#[cube(launch)]
pub fn minkowski_dist<F: Float + CubeElement>(
    x: &Array<F>, y: &Array<F>, out: &mut Array<F>,
    rows_x: u32, rows_y: u32, cols: u32, p: F,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x { if j < rows_y {
        let xb = i * cols; let yb = j * cols;
        let mut acc = F::from_int(0i64);
        let mut kk = 0u32;
        while kk < cols {
            let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
            acc += F::powf(diff, p);          // STATIC F::powf — lowers under cpu-MLIR (NOT x.powf())
            kk += 1u32;
        }
        let inv_p = F::new(1.0) / p;
        out[(i * rows_y + j) as usize] = F::powf(acc, inv_p);
    }}
}
```
**Manhattan** (spike 001 lines 27-52): identical loop body but `acc += diff;` and `out[..] = acc;` (no root).
**Chebyshev** (spike 001 lines 54-83): running max via STATEMENT form `if diff > acc { acc = acc_or_diff }` — diffs are `≥0` so seed `acc = F::from_int(0i64)` is correct:
```rust
let diff = (x[(xb + kk) as usize] - y[(yb + kk) as usize]).abs();
if diff > acc { acc = diff; }
```
**`p: F` is by value** (cubecl 0.10 — same as `scale`'s `factor: F`, elementwise.rs:73; `poly_map`'s `degree: F`, elementwise.rs:131). Host casts the validated `p: f64` to `F` before launch (RESEARCH Open Q3).

**`.abs()` is the instance form** (jacobi-proven, per spike 001 comment line 23) — the ONE place instance form is allowed; `F::powf` MUST be the static form.

---

### `self_drop_gather` kernel (device kernel, per-row GATHER)

**Analog:** `mlrs-kernels/src/topk.rs::select_k` (lines 53-180) for the `CUBE_POS_X`/`UNIT_POS_X==0` per-row launch shape + `u32`/`F`-only accumulators + `if`-guards; **VALIDATED verbatim shape** in `Skill("spike-findings-mlrs")` `sources/002-.../composition_and_self_drop.rs` lines 28-76.

**Copy verbatim (spike 002 lines 28-76):**
```rust
#[cube(launch)]
pub fn self_drop_gather<F: Float + CubeElement>(
    in_val: &Array<F>, in_idx: &Array<u32>,
    out_val: &mut Array<F>, out_idx: &mut Array<u32>,
    rows: u32, k: u32, k1: u32, // k1 = k + 1
) {
    let row = CUBE_POS_X;                       // native u32 (NOT ABSOLUTE_POS usize)
    if row < rows { if UNIT_POS_X == 0u32 {
        let ibase = row * k1;
        let obase = row * k;
        let mut s = 0u32;
        while s < k {
            let mut bump = 0u32;                // init INSIDE the consuming loop
            let mut c = 0u32;
            while c < s + 1u32 {                // nested count, SAME outer iteration
                if in_idx[(ibase + c) as usize] == row { bump += 1u32; }
                c += 1u32;
            }
            let src = s + bump;                 // src = s + (#self-cols at cols ≤ s)
            out_val[(obase + s) as usize] = in_val[(ibase + src) as usize];
            out_idx[(obase + s) as usize] = in_idx[(ibase + src) as usize];
            s += 1u32;
        }
    }}
}
```
**Launch shape** (spike 002 lines 80-85; identical to `topk.rs::launch_dims_rows` lines 215-220): `CubeCount::Static(n.max(1), 1, 1)`, `CubeDim { x:1, y:1, z:1 }`.

**Two cpu-MLIR landmines this kernel must NOT trip (both VALIDATED failures — RESEARCH Anti-Patterns):**
- **002-A (loud):** do NOT use bare 1D `ABSOLUTE_POS` launch → MLIR pass failure, kernel never runs (reads back zeros). Use `CUBE_POS_X`/`UNIT_POS_X==0`.
- **002-B (silent miscompile):** do NOT use a cross-sibling-loop accumulator (flag written in one `while`, read in a separate sibling `while`). Recompute the shift per-slot with the self-contained nested count above.

**Fallback (R-3):** when self is absent from top-(k+1) (shouldn't happen for X-vs-X), `bump` stays 0 for all `s` → `src = s` → drops the last column `k`. This is the documented correct fallback.

---

### `crates/mlrs-backend/src/prims/mod.rs` (MODIFY — module registration)

**Analog:** the existing `pub mod distance;` / `pub mod topk;` lines (mod.rs:36, 50). Add one line:
```rust
pub mod knn_graph;
```

### `crates/mlrs-kernels/src/lib.rs` (MODIFY — module + re-export)

**Analog:** existing `pub mod topk;` (lib.rs:27) + the `pub use elementwise::{ ... }` block (lib.rs:30-34). Add:
```rust
pub mod distance;                              // NEW (if kernels land in a new file)
pub use distance::{chebyshev_dist, manhattan_dist, minkowski_dist, self_drop_gather};
```
(If the kernels instead extend `elementwise.rs` per RESEARCH A1, add the four symbols to the existing `pub use elementwise::{...}` block — both are file-disjoint and compile.)

---

### `crates/mlrs-backend/tests/knn_graph_test.rs` (NEW — per-metric oracle + memory gate)

**Analog:** `crates/mlrs-backend/tests/topk_test.rs` (oracle structure, fixture loading, f32/f64 split, capability gate) + `tests/memory_gate_test.rs` (PoolStats query-axis gate).

**Fixture + dtype helpers — copy from topk_test.rs lines 45-67, 143-160:**
- `host_to_f64` / `from_f64` / `fixture_vec::<F>` byte-cast helpers (topk_test.rs:45-67).
- `fixture(name)` workspace-root resolver (topk_test.rs:143-151).
- `load_npz(fixture(name))` + `case.expect_f64("X")` / `"distances"` / `"indices"` (topk_test.rs:77-81; `OracleCase` from `mlrs_core`).

**Per-metric oracle body — model on `check_topk` (topk_test.rs:73-141)**, generalized to take a `Metric`:
```rust
fn check_knn_metric<F>(fixture_name: &str, metric: Metric, include_self: bool)
where F: Float + CubeElement + Pod {
    // f64 path: capability::skip_f64_with_log() early-return (topk_test.rs:192-195)
    // load fixture; knn_graph(pool, x, (n,d), k, metric, include_self, p);
    // assert indices SET-EQUAL up to tie-ordering; distances ≤1e-5 (DIST_TOL).
}
```
**Distance tolerance** — copy topk_test.rs `DIST_TOL = 1e-5` and the relative-tol assert (lines 43, 126-131). **Indices: set-equal up to tie-ordering** (CONTEXT/PRIM-11) — NOT the exact-index assert topk_test.rs uses, because cross-metric tie-ordering may differ; assert the returned index SET equals the sklearn index set per row (a per-row `HashSet`/sorted-vec compare), with the lowest-index tie-break still documented.

**Capability gate (f64-on-rocm skip) — copy verbatim from topk_test.rs:187-197:**
```rust
if capability::skip_f64_with_log() {
    println!("knn f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
    return;
}
```
Plus the `f32` companion test that runs on cpu AND rocm (topk_test.rs:177-183). Use `capability::active_backend_name()` + `capability::log_oracle_dtype(...)` (topk_test.rs:180-181).

**Geometry-rejection test — copy topk_test.rs `topk_rejects_bad_geometry` (lines 248-279):** assert `k <= n-1` (self-excl), `p >= 1`, and `n*d != len` are each `Err(PrimError::ShapeMismatch { operand, .. })` BEFORE launch (operands `"k"`, `"p"`, `"x"`).

**ADVERSARIAL duplicate-point VALUE assert (R-9 — the load-bearing gate):** model on spike 002's `host_knn` brute-force reference (composition_and_self_drop.rs:90-119) and the dup-row assertions (lines 191-201). The fixture MUST include a duplicate-point row; with `include_self=false`, neighbor 0 of the duplicate's query row MUST be the GENUINE duplicate index (not self), proving index-identity self-drop. Assert VALUES, not just non-panic — a silent miscompile (002-B) passes a happy-path check.

**`include_self=true` returns self at col 0** (HDBSCAN core dist) — spike 002 lines 203-219 pattern: assert self is present in every row.

**Memory gate (R-6, query-axis tiled) — copy the PoolStats assertion idiom from `memory_gate_test.rs`** (e.g. `memory_gate_reuse_bounded` lines 89-201): thread ONE `BufferPool`, run `knn_graph` and assert `peak_bytes`/`live_bytes` are SUB-QUADRATIC in `n` (never full `n×n` resident), `live_bytes` conserves after warmup, and scratch `reuses > 0`. These are HARD build-failing `assert!`s, backend-agnostic (drive f32). Threshold tuning is Claude's discretion (RESEARCH Open Q2).

---

### `scripts/gen_oracle.py` — per-metric `gen_knn` (MODIFY)

**Analog:** the existing `gen_knn` (gen_oracle.py:728-789, Euclidean only) + the `gen_dbscan` `metric=` precedent (line 707).

**Extend pattern** — add `metric` (and `p` for Minkowski) params to `NearestNeighbors`:
```python
nn = NearestNeighbors(n_neighbors=KNN_K, algorithm="brute",
                      metric=metric,            # "euclidean"|"manhattan"|"cosine"|"chebyshev"|"minkowski"
                      p=p).fit(x)               # p only used for minkowski
distances, indices = nn.kneighbors(xq)          # ascending
```
**Keep the existing structure:** the `c(arr)` dtype-cast closure (lines 781-782), `dtype_tag` + `f"knn_{metric}_{dtype_tag}_seed{seed}.npz"` naming (extend line 786 with the metric), `np.savez(out_path, X=, Xq=, distances=, indices=, ...)` (lines 787+). **ADD a duplicate-point design** (two identical train rows) for the R-9 adversarial gate — model on spike 001's dup-row fixture (kernels_and_harness.rs:143-149, rows 0 and 4 equal).
**Regen requires a `/tmp` venv with numpy** (PEP 668; MEMORY.md `oracle-fixture-regen-needs-venv`); fixtures are committed `.npz` blobs (f32 + f64 per metric).

## Shared Patterns

### Validate-before-launch (ASVS V5)
**Source:** `crates/mlrs-backend/src/prims/topk.rs::validate_geometry` (lines 148-210) and `distance.rs::validate_geometry` (lines 186-228).
**Apply to:** `knn_graph.rs` (host validation of `n*d`, `k<=n-1`, `p>=1`, u32-overflow) BEFORE any `unsafe` kernel launch. Return `PrimError::ShapeMismatch` (the project's all-purpose geometry error — no dedicated `InvalidK`/`InvalidArg` variant exists; `mlrs-core/src/error.rs:66-155`).

### cpu-MLIR-safe kernel authoring
**Source:** `mlrs-kernels/src/elementwise.rs` module doc (lines 18-25, STATEMENT-form clamp) + `topk.rs` module doc (lines 19-29, no SharedMemory/`F::INFINITY`/mutable-bool) + `Skill("spike-findings-mlrs")` `cpu-mlir-kernel-authoring.md`.
**Apply to:** all three new distance kernels AND `self_drop_gather`. Rules: STATIC `F::powf`/`F::exp` (never instance `x.powf()`); STATEMENT-form `if` for running-max/clamp (never if-expression in value position); `F`/`u32` accumulators only (no mutable bool); `CUBE_POS_X`/`UNIT_POS_X==0` for per-row selecting kernels (never bare `ABSOLUTE_POS` 1D); no cross-sibling-loop accumulator (recompute per-slot via nested count).

### Device residency + pool/out buffer reuse (D-05 / D-11)
**Source:** `distance.rs` lines 124-181 (acquire from pool or reuse caller `out`; `release_into(pool)` on consumed scratch; NO mid-pipeline host read-back) + `topk.rs` lines 83-141.
**Apply to:** `knn_graph.rs` — every intermediate (distance block, top_k(k+1) result) stays a `DeviceArray`; transient scratch released at true byte size; outputs are caller-owned (never released by the prim).

### Capability f64-on-rocm skip-with-log
**Source:** `crates/mlrs-backend/src/capability.rs::skip_f64_with_log` (lines 146-154); usage in `topk_test.rs:192-195`.
**Apply to:** every f64 test in `knn_graph_test.rs` — early-`return` when it reports `true`, after logging `active_backend_name()` + `log_oracle_dtype`.

### Oracle fixture loading
**Source:** `mlrs_core::{load_npz, OracleCase}` + `case.expect_f64(name)` (topk_test.rs:31, 77-81); workspace-root `fixture()` resolver (topk_test.rs:143-151).
**Apply to:** `knn_graph_test.rs` fixture loading for every per-metric `.npz`.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | — | — | Every new file has a strong in-repo analog. The query-axis TILING LOOP inside `knn_graph.rs` is the only sub-component with no direct copy-from — but its building block (`release_into(pool)` scratch release) is established in `distance.rs:156-166`, and the asserting gate copies `memory_gate_test.rs`. RESEARCH flags this as MEDIUM-risk (A3) — planner should confirm the tile threshold in the memory gate. |

## Metadata

**Analog search scope:** `crates/mlrs-backend/src/prims/`, `crates/mlrs-backend/src/`, `crates/mlrs-kernels/src/`, `crates/mlrs-backend/tests/`, `crates/mlrs-core/src/`, `scripts/`, `.claude/skills/spike-findings-mlrs/` (refs + sources).
**Files scanned:** distance.rs, topk.rs, elementwise.rs, topk.rs (kernel), lib.rs, prims/mod.rs, topk_test.rs, memory_gate_test.rs, capability.rs, error.rs, gen_oracle.py, both spike source files, knn-graph-primitive.md, cpu-mlir-kernel-authoring.md (via SKILL index).
**Pattern extraction date:** 2026-06-23
