# Phase 13: KNN-Graph Primitive (feasibility keystone) - Research

**Researched:** 2026-06-23
**Domain:** CubeCL device-kernel authoring (cpu-MLIR f64 gate), multi-metric pairwise distance, top-k composition, sklearn oracle validation, PoolStats memory gating
**Confidence:** HIGH (spikes 001+002 VALIDATED both feasibility unknowns; all reusable machinery read in-source this session)

## Summary

Phase 13 lands one new standalone `mlrs-backend` prim — `knn_graph<F>` — that composes the
already-launch-proven `distance.rs` (GEMM-expansion Euclidean) and `topk.rs` (k-smallest +
lowest-index tie-break) prims into a **directed** `(indices, distances)` `(n, k)` KNN graph over a
fixed multi-metric set (Euclidean, Manhattan, Cosine, Chebyshev, Minkowski-p) with an
`include_self: bool` parameter. The two genuine feasibility unknowns for this phase — (a) do new
direct feature-loop distance kernels, **including in-kernel `F::powf` for Minkowski-p**, lower
under cpu-MLIR; and (b) does the directed `distance → top_k → self-drop` composition with
index-identity self-drop launch under cpu-MLIR — were **both VALIDATED in spikes 001 and 002**
(`Skill("spike-findings-mlrs")`). This research does NOT re-derive those; it builds the plan on top
of them and fills the gaps the spikes did not cover: GEMM-expansion reuse for Euclidean/Cosine, the
PoolStats query-axis-tiled memory gate, per-metric sklearn oracle fixture generation, and the
concrete integration with `prims/mod.rs`, `mlrs-kernels`, and the test harness conventions.

The build is almost entirely **assembly of validated parts**. The only genuinely new kernels are
three direct pairwise distance kernels (Manhattan/Chebyshev/Minkowski-p — verbatim shapes proven in
spike 001) plus one `self_drop_gather` kernel (verbatim shape proven in spike 002, with two
documented cpu-MLIR landmines to avoid). Everything else (top-k select, GEMM-expansion, sqrt
boundary, pool/out buffer reuse, capability f64-skip, oracle-fixture loading) already ships and was
read in-source.

**Primary recommendation:** Build `crates/mlrs-backend/src/prims/knn_graph.rs` as a thin host
orchestrator: validate geometry + `metric`/`k`/`p` host-side, route Euclidean/Cosine through the
existing `distance()` GEMM fast path and Manhattan/Chebyshev/Minkowski-p through three new direct
kernels in `mlrs-kernels`, then `top_k(k or k+1, sqrt=…)`, then (for `include_self=false`) one
`self_drop_gather` kernel — copying the EXACT proven kernel shapes from
`Skill("spike-findings-mlrs")` and the host-launch idiom from `distance.rs`/`topk.rs`. Gate with a
per-metric sklearn oracle (incl. a duplicate-point row asserting VALUES) and a query-axis-tiled
PoolStats gate.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Pairwise distance (Euclidean/Cosine) | mlrs-backend prim (`distance.rs` GEMM) | mlrs-kernels (gemm/reduce/combine kernels) | GEMM-expansion is Euclidean-specific; already launch-proven, device-resident |
| Pairwise distance (Manhattan/Chebyshev/Minkowski-p) | mlrs-kernels (new `#[cube(launch)]` kernels) | mlrs-backend prim (launch wrapper) | Direct feature-loop kernels; GEMM cannot back L1/L∞/Lp |
| Top-k select (k-smallest + idx) | mlrs-backend prim (`topk.rs`) | mlrs-kernels (`topk::select_k`) | Already ships; lowest-index tie-break documented |
| Self-drop (index identity) | mlrs-kernels (new `self_drop_gather` kernel) | mlrs-backend prim (`knn_graph.rs` launch) | Per-output-slot GATHER, cpu-MLIR-safe shape from spike 002 |
| Metric routing + k/p validation | mlrs-backend prim (`knn_graph.rs` host) | — | Host-side `Result` validation before any unsafe launch (project D-04 lineage) |
| L2-normalization (Cosine) | mlrs-backend prim (host glue, reuses reduce/scale) | mlrs-kernels (existing `scale`/reduce) | Cosine = GEMM on unit rows; normalize is a pre-pass |
| Memory bounding (no full n×n) | mlrs-backend prim (host tiling loop) | pool (`PoolStats`) | Query-axis tile; PoolStats gate asserts no n×n resident |
| f64-on-rocm skip | mlrs-backend (`capability::skip_f64_with_log`) | test harness | Existing project gate; rocm has no f64 |

## Standard Stack

This is a closed Rust workspace with a **zero-new-compute-dependency** mandate (ROADMAP §v3.0,
CLAUDE.md). No packages are added this phase. The "stack" is the existing internal crates +
in-test sklearn oracle.

### Core (existing, reused — no install)
| Component | Location | Purpose | Why Standard |
|-----------|----------|---------|--------------|
| `cubecl` 0.10.0 | workspace dep | Generic-over-float/runtime device kernels | Project compute substrate (CLAUDE.md) `[VERIFIED: Cargo.toml]` |
| `distance<F>` prim | `crates/mlrs-backend/src/prims/distance.rs` | GEMM-expansion squared-Euclidean, sqrt boundary, pool/out reuse | Launch-proven; backs Euclidean+Cosine `[VERIFIED: read in-source]` |
| `top_k<F>` prim | `crates/mlrs-backend/src/prims/topk.rs` | k-smallest + idx per row, lowest-index tie-break, sqrt boundary | Launch-proven; the second half of every metric path `[VERIFIED: read in-source]` |
| `mlrs-kernels` | `crates/mlrs-kernels/src/` | Feature-free `#[cube(launch)]` kernels (new distance kernels land here) | Backend-feature-free kernel home (D-13) `[VERIFIED: read in-source]` |
| `BufferPool`/`PoolStats` | `crates/mlrs-backend/src/pool.rs` | Byte-keyed buffer reuse + `allocations`/`reuses`/`peak_bytes`/`live_bytes` counters | Memory-gate substrate `[VERIFIED: read in-source]` |
| `capability::skip_f64_with_log` | `crates/mlrs-backend/src/capability.rs` | f64-on-rocm skip-with-log gate | Existing per-prim backend gate `[VERIFIED: grep]` |
| `mlrs_core::{load_npz, OracleCase}` | `crates/mlrs-core` | Load committed `.npz` oracle fixtures | All prim oracle tests use it `[VERIFIED: read topk_test.rs]` |

### Supporting (existing patterns to follow)
| Pattern | Source | When to Use |
|---------|--------|-------------|
| `gen_oracle.py` generator + committed `.npz` | `scripts/gen_oracle.py` (`gen_knn`) | New per-metric KNN fixtures |
| `dist_combine_clamp` STATEMENT-form clamp | `mlrs-kernels/src/elementwise.rs` | Model for new 2D distance kernels' guard/seed idiom |
| `memory_gate_test.rs` PoolStats assertions | `crates/mlrs-backend/tests/memory_gate_test.rs` | New build-failing query-axis memory gate |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New `knn_graph.rs` prim (D-03) | Extend `mlrs-algos` `NearestNeighbors` core | Wrong altitude for a shared prim; UMAP/HDBSCAN call from algos — REJECTED in CONTEXT |
| Index-identity self-drop (D-02) | First-zero-distance drop | Drops a genuine neighbor on duplicate points — REJECTED, proven wrong in spike 002 |
| Special-case Euclidean=Minkowski(2) for correctness | One general direct kernel | General kernel reproduces L1/L2 to ≤1e-9 (spike 001); special-case ONLY as a perf optimization (GEMM is worth it for Euclidean/Cosine) |
| Full n×n distance matrix | Query-axis tiling | n×n is the memory-gate failure mode (R-6); tile over query (i) axis |

**Installation:** None. `npm`/`pip`/`cargo` add: **no new dependency** (ROADMAP zero-new-compute-dependency mandate).

**Version verification:**
- `cubecl = "0.10.0"` `[VERIFIED: /home/user/Documents/workspace/mlrs/Cargo.toml line 16]`
- cpu-MLIR backend is `cubecl-cpu` 0.10; f64 supported on cpu, NOT on rocm `[VERIFIED: MEMORY.md rocm-is-runnable-gpu-gate + cpu-mlir-kernel-authoring.md]`

## Package Legitimacy Audit

> Not applicable — this phase installs **zero external packages**. All work composes existing
> internal workspace crates (`mlrs-backend`, `mlrs-kernels`, `mlrs-core`) and the already-pinned
> `cubecl` 0.10.0 dependency. No registry verification needed.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```text
                  knn_graph<F>(pool, x, (n,d), k, metric, include_self, p, out) -> (indices(n,k), distances(n,k))
                                              │
                       ┌──────────────────────┴───────────────────────┐
                       │  HOST validation (before ANY unsafe launch)   │
                       │  - geometry: n*d == x.len()                   │
                       │  - 1 <= k <= n-1 (k+1 <= n if !include_self)  │
                       │  - metric ∈ fixed set; if Minkowski: p >= 1   │
                       └──────────────────────┬───────────────────────┘
                                              │ metric routing
            ┌─────────────────────────────────┼──────────────────────────────────┐
            ▼ Euclidean / Cosine              ▼ Manhattan / Chebyshev / Minkowski-p
   ┌───────────────────────────┐      ┌──────────────────────────────────────────┐
   │ (Cosine: L2-normalize rows)│      │ NEW direct pairwise kernel (mlrs-kernels) │
   │ distance() GEMM-expansion  │      │ 2D launch, one unit per (i,j),            │
   │ ‖x‖²+‖y‖²−2XYᵀ, clamp≥0    │      │ while kk<cols { acc op= f(Δ) }            │
   │ sqrt deferred to top_k     │      │ (Minkowski: F::powf per term + ^(1/p))    │
   └─────────────┬─────────────┘      └────────────────────┬─────────────────────┘
                 └──────────── n×rows distance block ───────┘   (query-axis TILED; never full n×n leaking)
                                              │
                                              ▼
                          top_k(dist, n, rows, K, sqrt=euclidean-only)
                          K = k        if include_self=true   ── (HDBSCAN: self at col 0)
                          K = k+1      if include_self=false  ── (UMAP)
                                              │  (val(n,K), idx(n,K)) ascending, lowest-idx tie-break
                            ┌─────────────────┴──────────────────┐
              include_self=true                       include_self=false
                    │                                          │
                    ▼                                          ▼
            return (idx, val) (n,k)            self_drop_gather kernel (NEW, mlrs-kernels)
                                               drop col whose idx == query-row index
                                               (index identity; fallback drop last col)
                                                          │
                                                          ▼
                                               return (idx, val) (n,k) directed
```
*File-to-implementation mapping is in Component Responsibilities; the diagram is data flow only.*

### Recommended Project Structure
```
crates/mlrs-backend/src/prims/
├── knn_graph.rs        # NEW prim: host orchestration, metric routing, validation, composition
├── distance.rs         # REUSE: Euclidean GEMM-expansion (+ Cosine on normalized rows)
├── topk.rs             # REUSE: k-smallest + lowest-index tie-break
└── mod.rs              # ADD: `pub mod knn_graph;`

crates/mlrs-kernels/src/
├── distance.rs (NEW) or extend elementwise.rs  # manhattan_dist/chebyshev_dist/minkowski_dist + self_drop_gather kernels
└── lib.rs              # ADD: module + `pub use` of new kernel symbols

crates/mlrs-backend/tests/
└── knn_graph_test.rs   # NEW: per-metric sklearn oracle (incl. duplicate-point VALUE assert) + memory gate

scripts/gen_oracle.py   # ADD: per-metric KNN fixtures via NearestNeighbors(metric=…)
```

### Pattern 1: Prim host shape (validate-before-launch, pool/out reuse)
**What:** Every mlrs-backend prim is `fn prim<F>(pool, operands…, out: Option<…>) -> Result<…, PrimError>`; geometry validated BEFORE any unsafe launch; outputs device-resident; optional caller-supplied buffer reuse (D-11).
**When to use:** `knn_graph<F>` follows this exactly.
**Example:**
```rust
// Source: crates/mlrs-backend/src/prims/topk.rs (read in-source 2026-06-23)
pub fn top_k<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    dist: &DeviceArray<ActiveRuntime, F>,
    rows: usize, cols: usize, k: usize, sqrt: bool,
    out_val: Option<DeviceArray<ActiveRuntime, F>>,
    out_idx: Option<DeviceArray<ActiveRuntime, u32>>,
) -> Result<(DeviceArray<ActiveRuntime, F>, DeviceArray<ActiveRuntime, u32>), PrimError>
where F: Float + CubeElement + Pod {
    validate_geometry(dist.len(), (rows, cols), k, /*…*/)?;   // BEFORE launch
    // … acquire from pool or reuse out, launch select_k, optional sqrt boundary …
}
```

### Pattern 2: Direct pairwise distance kernel (cpu-MLIR-safe) — PROVEN spike 001
**What:** One `#[cube(launch)]` per metric, one unit per output element `(i,j)`, runtime `while kk < cols` feature loop, `F`/`u32` accumulators with `if`-guarded updates.
**When to use:** Manhattan/Chebyshev/Minkowski-p (GEMM cannot back these).
**Example:**
```rust
// Source: Skill("spike-findings-mlrs") sources/001-.../kernels_and_harness.rs (VALIDATED under --features cpu)
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
            acc += F::powf(diff, p);          // STATIC F::powf — lowers under cpu-MLIR (NOT instance x.powf())
            kk += 1u32;
        }
        let inv_p = F::new(1.0) / p;
        out[(i * rows_y + j) as usize] = F::powf(acc, inv_p);
    }}
}
// Launch: CubeDim {x:16,y:16}, ceiling-div counts over (rows_x, rows_y).
```
Manhattan: `acc += diff;`. Chebyshev: `if diff > acc { acc = diff; }` (statement form; diffs ≥0 so seed 0 correct). `[VERIFIED: spike 001 VALIDATED, depth-probed ≤1e-9 vs L1/L2]`

### Pattern 3: Self-drop by index identity (cpu-MLIR-safe) — PROVEN spike 002
**What:** Per output slot, compute the self-shift with a SELF-CONTAINED nested count inside the consuming loop (no cross-sibling-loop accumulator).
**When to use:** `include_self=false` only.
**Example:**
```rust
// Source: Skill("spike-findings-mlrs") cpu-mlir-kernel-authoring.md FINDING 002-B avoidance (VALIDATED)
let row = CUBE_POS_X;                  // per-row, one-selecting-unit shape (NOT bare ABSOLUTE_POS 1D)
if row < rows { if UNIT_POS_X == 0u32 {
    let mut s = 0u32;
    while s < k {
        let mut bump = 0u32;          // init INSIDE the consuming loop (not a sibling loop)
        let mut c = 0u32;
        while c < s + 1u32 {          // nested, read in the SAME outer iteration
            if in_idx[(ibase + c) as usize] == row { bump += 1u32; }
            c += 1u32;
        }
        let src = s + bump;           // no carry across sibling loops
        // copy in_idx[ibase+src] / in_val[ibase+src] -> out[obase+s]
        s += 1u32;
    }
}}
// Launch: CubeCount::Static(n,1,1), CubeDim {x:1,y:1,z:1}.
```
**Fallback:** if self isn't in the top-(k+1) (shouldn't happen for X-vs-X), drop the last column. `[VERIFIED: spike 002 VALIDATED on duplicate-point adversarial case]`

### Anti-Patterns to Avoid
- **Bare-`ABSOLUTE_POS` 1D launch for a per-row loop kernel** → MLIR pass failure `"operation with block successors must terminate its parent block"`, kernel never runs (output reads back zeros). Use the `CUBE_POS_X`/`UNIT_POS_X==0` shape. `[VERIFIED: spike 002 FINDING 002-A]`
- **Cross-sibling-loop mutable accumulator** (flag written in one `while`, read in a separate sibling `while`) → **SILENT MISCOMPILE** (value never updates; compiles, launches, returns plausible wrong data). Recompute per-row positional values with a self-contained nested accumulate. `[VERIFIED: spike 002 FINDING 002-B]`
- **Instance `x.powf()` form** → can mis-lower in the `#[cube]` IR. Use static `F::powf(x, p)`. `[VERIFIED: cpu-mlir-kernel-authoring.md]`
- **`if`-expression in value position** for running max / clamp → mis-lowers; use STATEMENT form (`let mut v=…; if cond { v=… }`). `[VERIFIED: elementwise.rs dist_combine_clamp + cpu-mlir authoring doc]`
- **`SharedMemory` / `Atomic` / `F::INFINITY` / mutable-bool scans / descending-shift loops** → panic at launch on cpu-MLIR. `[VERIFIED: MEMORY.md cubecl-cpu-no-shared-memory + cpu-mlir authoring doc]`
- **Materializing the full n×n distance block** → memory-gate failure (R-6); tile over query axis.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Euclidean pairwise distance | A direct Euclidean feature-loop kernel | `distance()` GEMM-expansion (`distance.rs`) | Launch-proven, faster (GEMM), clamp+sqrt boundary handled |
| Cosine distance | A bespoke cosine kernel | L2-normalize rows + `distance()` GEMM | `1 − x̂·ŷ`; reuses the same validated GEMM path |
| k-smallest + indices | A new selection/sort kernel | `top_k()` (`topk.rs`) | Lowest-index tie-break already documented; sqrt boundary built in |
| sqrt-of-distance boundary | sqrt over the whole n×rows matrix | `sqrt=true` on `top_k` (sqrt only the k returned) | Cheaper; squared distance is order-preserving for selection |
| Buffer reuse / memory accounting | Manual alloc tracking | `BufferPool` + `PoolStats` | Byte-keyed reuse, `reuses`/`peak_bytes`/`live_bytes` counters ship |
| f64-on-rocm handling | Per-test cfg gymnastics | `capability::skip_f64_with_log()` | Existing project gate, used by every f64 prim test |
| Oracle fixtures | Hand-rolled expected values | `gen_oracle.py` + committed `.npz` via `load_npz` | sklearn is the source of truth; fixtures are committed blobs |

**Key insight:** The only kernels that genuinely must be written are the three direct distance
kernels (verbatim from spike 001) and one self-drop kernel (verbatim from spike 002). Everything
else is composition of validated parts — and the spikes proved the composition itself launches.

## Runtime State Inventory

> Not a rename/refactor/migration phase — this is greenfield prim addition. Section omitted per
> output spec (no stored data / live config / OS-registered state to migrate). The only "state"
> consideration is the committed oracle fixtures, covered under Environment Availability.

## Common Pitfalls

### Pitfall 1: Silent miscompile in self-drop passes a happy-path test
**What goes wrong:** A cross-sibling-loop accumulator compiles, launches, and returns plausible-but-wrong neighbor indices. A non-duplicate fixture never exercises the divergence.
**Why it happens:** cpu-MLIR silently miscompiles the cross-loop pattern (FINDING 002-B); without a distance-0 duplicate row the wrong column is never selected.
**How to avoid:** Self-contained nested accumulate (Pattern 3). The oracle gate MUST include a duplicate-point row (two samples at distance 0) and assert VALUES, not just non-panic (R-9).
**Warning signs:** Test green on well-spread data, indices diverge only when two train points coincide.

### Pitfall 2: First-zero-distance self-drop removes a genuine neighbor
**What goes wrong:** With `include_self=false`, dropping "the first distance-0 entry" removes a real duplicate neighbor and keeps self, diverging from sklearn on tie-heavy data.
**Why it happens:** Duplicate points sit at distance 0 alongside self.
**How to avoid:** Drop by INDEX IDENTITY (idx == query row), fallback drop last column (D-02 / R-3).
**Warning signs:** Index set-mismatch vs oracle only on datasets with coincident points.

### Pitfall 3: f32 catastrophic cancellation → negative squared distance (Euclidean GEMM path)
**What goes wrong:** `‖x‖²+‖y‖²−2XYᵀ` for near-identical rows lands slightly negative; sqrt → NaN.
**Why it happens:** Floating-point cancellation in the GEMM expansion.
**How to avoid:** Already handled — `distance.rs` applies `max(d²,0)` UNCONDITIONALLY (statement-form clamp) before any sqrt. Reuse `distance()`; do NOT bypass the clamp. The direct kernels (L1/L∞/Lp) are sums of non-negatives so they cannot go negative.
**Warning signs:** NaN distances only on near-duplicate rows.

### Pitfall 4: Minkowski-p `p` not validated host-side
**What goes wrong:** `p < 1` (or `p == 0`) yields a non-metric / division issues; the kernel does NOT guard `p`.
**Why it happens:** The kernel computes `^(1/p)` blindly.
**How to avoid:** Validate `p ≥ 1` at the prim boundary (host) before launch; return a typed `PrimError` (CONTEXT Claude's-discretion: choose `F` vs `f64` for `p` and the `p≥1` check). `[CITED: knn-graph-primitive.md "Validate p ≥ 1 host-side"]`

### Pitfall 5: Full n×n distance matrix resident → memory gate red
**What goes wrong:** Computing the entire `n×n` block at once (and not releasing it) makes `peak_bytes`/`live_bytes` scale as n², failing the PoolStats gate.
**Why it happens:** Naive "compute all distances, then top-k all rows."
**How to avoid:** Tile over the query (i) axis — process a block of query rows, top-k that block, release its distance scratch (the `distance.rs` `release_into(pool)` precedent), advance. Big distance operand kept global; query-axis tiled (D-CONTEXT criterion 4). Tile size is Claude's discretion.
**Warning signs:** `peak_bytes` grows quadratically with n in the gate.

### Pitfall 6: k+1 exceeds n when include_self=false on small fixtures
**What goes wrong:** Internal `k+1` query with `k = n-1` reads past the train set.
**Why it happens:** Self-exclusion needs `k+1` neighbors available.
**How to avoid:** Validate `k+1 <= n` (i.e. `k <= n-1`) host-side when `include_self=false`; `top_k` already validates `1 <= K <= cols`, but surface a clear prim-level error.

## Code Examples

### Host-launch idiom (cubecl 0.10) for a new 2D distance kernel
```rust
// Source: crates/mlrs-backend/src/prims/distance.rs (dist_combine_clamp launch) + cpu-mlir-kernel-authoring.md
let client = pool.client().clone();
let (count, dim) = launch_dims_2d(rows_x, rows_y);   // CubeDim{16,16,1}, ceiling-div
// SAFETY: lengths are validated element counts; kernel bounds-checks i<rows_x && j<rows_y.
let x_arg   = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
let y_arg   = unsafe { ArrayArg::from_raw_parts(y.handle().clone(), y.len()) };
let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
minkowski_dist::launch::<F, ActiveRuntime>(
    &client, count, dim, x_arg, y_arg, out_arg,
    rows_x as u32, rows_y as u32, cols as u32, p,   // scalars BY VALUE (no ScalarArg in 0.10)
);
```

### Per-metric oracle test shape
```rust
// Source: crates/mlrs-backend/tests/topk_test.rs (read in-source) — extend per metric
fn check_knn_metric<F>(fixture_name: &str, metric: Metric, include_self: bool)
where F: Float + CubeElement + Pod {
    if std::mem::size_of::<F>() == 8 && capability::skip_f64_with_log() { return; }  // rocm f64 skip
    let case = load_npz(fixture(fixture_name)).expect("load knn fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let ref_idx: Vec<f64> = case.expect_f64("indices").to_vec();
    let ref_dist: Vec<f64> = case.expect_f64("distances").to_vec();
    // … knn_graph(pool, x, (n,d), k, metric, include_self, p) …
    // assert: indices set-equal up to tie-ordering; distances ≤1e-5 (f64); lowest-index tie-break.
    // MUST include a fixture with a duplicate-point row asserting VALUES (R-9).
}
```

### New per-metric fixture generation
```python
# Source: scripts/gen_oracle.py gen_knn (extend with a metric param)
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=KNN_K, algorithm="brute",
                      metric=metric,                 # "euclidean"|"manhattan"|"cosine"|"chebyshev"|"minkowski"
                      p=p).fit(x)                     # p only for minkowski
distances, indices = nn.kneighbors(xq)               # ascending
# Add a DUPLICATE-POINT design (two identical rows) for the R-9 adversarial gate.
# Regen needs a /tmp venv with numpy (PEP 668); fixtures are committed blobs (MEMORY.md).
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| KNN over v1 distance prim (Euclidean only) | Full fixed multi-metric set incl. Minkowski-p | This session (D-05 user scope expansion) | New direct L1/L∞/Lp kernels required |
| Named "symmetrize-map" step in Phase-13 spike | Directed graph only; symmetrization deferred to consumers | This session (D-04) | Spike scope re-scoped (D-06); no symmetrize here |
| Minkowski-p `pow` feasibility unknown | `F::powf` proven to lower under cpu-MLIR | Spike 001 (VALIDATED) | No longer a blocker |
| Directed-compose + self-drop feasibility unknown | Index-identity self-drop proven, 2 landmines documented | Spike 002 (VALIDATED) | No longer a blocker |

**Deprecated/outdated:** none specific to this phase. (Project-wide: `cubecl-matmul`/`cubecl-linalg` abandoned in favor of `cubek-*`, but this phase composes existing prims and adds no matmul dep — MEMORY.md.)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | New direct distance kernels should live in a new `mlrs-kernels/src/distance.rs` (or extend `elementwise.rs`) | Project Structure | Low — Claude's discretion per CONTEXT; both compile, file-disjoint |
| A2 | `top_k`'s existing `1 <= k <= cols` validation suffices for the `k+1` internal query (prim adds its own `k <= n-1` guard) | Pitfall 6 | Low — additive guard, verified by reading topk.rs |
| A3 | Query-axis tiling can reuse `distance.rs`'s `release_into(pool)` scratch-release precedent to bound `peak_bytes` | Pitfall 5 | Medium — exact tiling/release wiring is new; planner should spike the gate threshold |
| A4 | Cosine = L2-normalize rows then GEMM `1 − x̂·ŷ` matches sklearn `metric='cosine'` to ≤1e-5 | Stack / routing | Medium — normalization edge case (zero-norm rows) needs host handling; confirm vs oracle |
| A5 | A single duplicate-point design in one fixture satisfies R-9 across all metrics | Oracle | Low — could need per-metric duplicate rows; cheap to add |

**Note:** No `[ASSUMED]` package claims (zero new dependencies). A3/A4 are the only medium-risk
items and both are validated by the per-metric oracle + memory gate this phase ships.

## Open Questions

1. **Exact `Metric` enum shape and whether Euclidean=Minkowski(2)/Manhattan=Minkowski(1) route to fast paths**
   - What we know: General direct kernel reproduces L1/L2 to ≤1e-9 (spike 001); GEMM is worth special-casing for Euclidean/Cosine.
   - What's unclear: Whether to special-case Minkowski(1)/(2) to the direct-L1/GEMM paths for perf.
   - Recommendation: `Metric::{Euclidean, Manhattan, Cosine, Chebyshev, Minkowski{p}}`; route Euclidean→GEMM, Cosine→normalized GEMM, the rest→direct kernels. Special-case Minkowski(2)→GEMM only if profiling warrants (Claude's discretion per CONTEXT).

2. **Query-axis tile size for the PoolStats gate**
   - What we know: Criterion locks "query-axis tiled, big distance operand kept global, never full n×n resident."
   - What's unclear: Optimal tile size (Claude's discretion per CONTEXT).
   - Recommendation: Pick a fixed row-block tile, assert `peak_bytes` is sub-quadratic in n in the gate; tune if red.

3. **`p` type (`F` vs `f64`) and validation**
   - What we know: Kernel takes `p: F` (spike 001 shape); `p ≥ 1` must be host-validated.
   - Recommendation: Accept `p: f64` at the prim boundary, validate `p ≥ 1`, cast to `F` for the kernel (Claude's discretion per CONTEXT).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cargo` + `--features cpu` (cpu-MLIR) | f64 correctness gate (all metrics) | ✓ | cubecl-cpu 0.10 | — (primary gate) |
| `--features rocm` (gfx1100) | f32 GPU gate | ✓ | ROCm 7.1.1 | f64-on-rocm skips-with-log |
| `--features cuda` | opportunistic | compile-only | — | untestable in this env (CLAUDE.md) |
| numpy (oracle fixture regen) | new per-metric `.npz` fixtures | via /tmp venv | PEP 668 workaround | fixtures are committed blobs; regen only when adding fixtures (MEMORY.md) |
| sklearn `NearestNeighbors` | oracle source of truth | via /tmp venv | — | committed `.npz` |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:**
- Full `cargo test --features cpu` exhausts disk (ENOSPC) — run TARGETED tests (`knn_graph_test`), `cargo clean` to recover (MEMORY.md). Background the full run.
- Backend suite is slow (~6 min cpu) — run targeted post-merge gates, background full run (MEMORY.md).
- Python wheel path untestable here — N/A this phase (prim only, no estimator/PyO3 per D-03).

## Validation Architecture

> nyquist_validation is enabled (config.json `workflow.nyquist_validation: true`).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (cargo test) + committed `.npz` oracle via `mlrs_core::load_npz` |
| Config file | none (Cargo workspace); tests in `crates/*/tests/` per AGENTS.md §2 |
| Quick run command | `cargo test --features cpu -p mlrs-backend --test knn_graph_test` |
| Full suite command | `cargo test --features cpu -p mlrs-backend` (slow ~6min; background it) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRIM-11 | Directed `(idx,dist)` `(n,k)` per metric, ascending, lowest-idx tie-break | oracle (unit) | `cargo test --features cpu -p mlrs-backend --test knn_graph_test` | ❌ Wave 0 |
| PRIM-11 | Indices set-equal to sklearn (per metric) up to tie-ordering; distances ≤1e-5 f64 | oracle | same | ❌ Wave 0 |
| PRIM-11 | `include_self=false` self-drop by index identity (duplicate-point VALUE assert, R-9) | oracle (adversarial) | same | ❌ Wave 0 |
| PRIM-11 | `include_self=true` returns self at col 0 (HDBSCAN core dist) | oracle | same | ❌ Wave 0 |
| PRIM-11 | Every metric launches under `--features cpu` (launch, not just compile) incl. Minkowski-p `pow` | launch | same (cpu feature) | ❌ Wave 0 |
| PRIM-11 | rocm f32 launch; f64-on-rocm skips-with-log | launch | `cargo test --features rocm -p mlrs-backend --test knn_graph_test` | ❌ Wave 0 |
| PRIM-11 | Build-failing PoolStats memory gate (query-axis tiled; no full n×n resident) | memory gate | `cargo test --features cpu -p mlrs-backend --test knn_graph_test memory_gate` | ❌ Wave 0 |
| PRIM-11 | `p ≥ 1` / `k ≤ n-1` host validation returns typed `PrimError` (no launch) | unit | same | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --features cpu -p mlrs-backend --test knn_graph_test`
- **Per wave merge:** `cargo build --features cpu -p mlrs-backend && cargo build --features rocm -p mlrs-backend` + targeted knn test
- **Phase gate:** Full `mlrs-backend` cpu suite green (backgrounded) + rocm f32 launch before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `crates/mlrs-backend/tests/knn_graph_test.rs` — per-metric oracle + duplicate-point VALUE assert + memory gate (covers PRIM-11)
- [ ] `scripts/gen_oracle.py` — per-metric KNN fixtures (`gen_knn` extended with `metric`/`p`; add a duplicate-point design)
- [ ] Committed `.npz` fixtures (f32 + f64) per metric, regenerated via /tmp numpy venv
- [ ] `crates/mlrs-kernels` module + `pub use` for new distance + self-drop kernels
- [ ] `crates/mlrs-backend/src/prims/mod.rs` — `pub mod knn_graph;`

## Security Domain

> `security_enforcement` enabled (config.json), ASVS level 1. This is an internal compute prim
> (no auth, sessions, network, secrets, or user-facing input surface) — most ASVS categories are
> N/A. The relevant control is input/geometry validation before unsafe device launches.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (library prim) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Host-side geometry/`metric`/`k`/`p` validation returning typed `PrimError` BEFORE any `unsafe` kernel launch (the `distance.rs`/`topk.rs` precedent; mitigates out-of-bounds device reads) |
| V6 Cryptography | no | — |

### Known Threat Patterns for the prim
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device read from bad geometry/`k` | Tampering / DoS | Validate `n*d==len`, `1≤k≤n-1`, `k+1≤n` (self-excl), `p≥1` BEFORE launch; reject overflowing-`u32` dims (topk.rs WR-03 precedent) |
| `unsafe { ArrayArg::from_raw_parts }` length mismatch | Tampering | Pass validated element counts only; kernels bounds-check `i<rows && j<cols` (T-0203-01 precedent) |
| Silent miscompile returns wrong data | Repudiation (data integrity) | Per-metric oracle VALUE assertions + duplicate-point adversarial gate (R-9) |

## Sources

### Primary (HIGH confidence)
- `Skill("spike-findings-mlrs")` — SKILL.md, `references/knn-graph-primitive.md`, `references/cpu-mlir-kernel-authoring.md`, `sources/001-*`, `sources/002-*` (both spikes VALIDATED under `--features cpu`)
- `crates/mlrs-backend/src/prims/distance.rs` — read in-source (GEMM-expansion, clamp, sqrt boundary, pool/out reuse, validate-before-launch)
- `crates/mlrs-backend/src/prims/topk.rs` — read in-source (k-smallest, lowest-index tie-break, geometry/k validation, launch shape)
- `crates/mlrs-kernels/src/elementwise.rs` — read in-source (STATEMENT-form clamp, 2D launch idiom, scalar-by-value)
- `crates/mlrs-backend/tests/topk_test.rs`, `tests/memory_gate_test.rs` — read in-source (oracle + PoolStats gate patterns)
- `scripts/gen_oracle.py` (`gen_knn`) — read in-source (fixture generation + `NearestNeighbors`)
- `.planning/REQUIREMENTS.md` PRIM-11, `.planning/ROADMAP.md` §Phase 13, `13-CONTEXT.md` (D-01..D-06)

### Secondary (MEDIUM confidence)
- Project MEMORY.md landmines (cpu-MLIR no-SharedMemory; rocm f64 unsupported; oracle venv; disk/suite-slowness)

### Tertiary (LOW confidence)
- none — no web research needed; phase is fully grounded in spikes + in-source machinery.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all reused prims read in-source; zero new deps
- Architecture: HIGH — both feasibility unknowns VALIDATED by spikes 001/002 with verbatim kernel shapes
- Pitfalls: HIGH — landmines documented from actual spike failures (002-A loud, 002-B silent)
- Memory gate / tiling wiring: MEDIUM — pattern clear (release_into precedent) but exact tile/threshold is new work (A3)
- Cosine normalization edge cases: MEDIUM — confirm zero-norm handling vs oracle (A4)

**Research date:** 2026-06-23
**Valid until:** 2026-07-23 (stable — internal codebase + pinned cubecl 0.10; spikes already validated)
