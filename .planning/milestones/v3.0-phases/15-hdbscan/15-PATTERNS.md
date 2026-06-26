# Phase 15: HDBSCAN - Pattern Map

**Mapped:** 2026-06-24
**Files analyzed:** 11 (6 new, 5 modified)
**Analogs found:** 11 / 11 (all have a strong in-repo analog)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-algos/src/cluster/hdbscan.rs` (MODIFY) | estimator (builder+typestate, labels-only) | request-response + transform | `crates/mlrs-algos/src/cluster/dbscan.rs` (host-walk-over-device-mask labels-only estimator) + the file's own Phase-12 shell | exact (role+flow) |
| `crates/mlrs-algos/src/cluster/hdbscan/mst.rs` (NEW) | host back-end module | transform (pure scalar) | `crates/mlrs-algos/src/cluster/dbscan.rs` host DFS body (lines 194-227) + `cluster/spectral.rs` host recover helper | role-match (host scalar) |
| `crates/mlrs-algos/src/cluster/hdbscan/single_linkage.rs` (NEW) | host back-end module (UnionFind) | transform | `dbscan.rs` host LIFO stack (sequential merge precedent) | role-match |
| `crates/mlrs-algos/src/cluster/hdbscan/condense.rs` (NEW) | host back-end module (tree BFS) | transform | `dbscan.rs` host graph walk | role-match |
| `crates/mlrs-algos/src/cluster/hdbscan/stability.rs` (NEW) | host back-end module | transform | `dbscan.rs` host scalar pass | role-match |
| `crates/mlrs-algos/src/cluster/hdbscan/select.rs` (NEW) | host back-end module (eom/leaf/ε labelling+probs) | transform | `dbscan.rs` labelling + `label_perm.rs` HashMap bookkeeping | role-match |
| `crates/mlrs-algos/src/cluster/hdbscan/glosh.rs` (NEW) | host back-end module (outlier_scores) | transform | `stability.rs` sibling (same tree pass) | role-match |
| `crates/mlrs-algos/src/cluster/hdbscan/centers.rs` (NEW) | host back-end module (centroid/medoid) | batch reduce (host) | `spectral_embedding.rs` host `host_to_f64`/`f64_to_host` reduce (lines 241-284) | role-match |
| `crates/mlrs-kernels/src/mutual_reachability.rs` (NEW) | device kernel (`#[cube(launch)]`) | transform (per-element GATHER) | `crates/mlrs-kernels/src/distance.rs` `manhattan_dist`/`chebyshev_dist` (2D GATHER) | exact (role+flow) |
| `crates/mlrs-core/src/label_perm.rs` (MODIFY) | utility (test helper) | transform | `best_match_accuracy`/`best_mapping` in the same file (lines 51-117) | exact (extend in place) |
| `crates/mlrs-algos/tests/hdbscan_test.rs` (REPLACE) | test (oracle) | request-response | `crates/mlrs-algos/tests/kmeans_test.rs` (exact-label + value-gate precedent) | exact |
| `scripts/gen_oracle.py` (MODIFY — add `gen_hdbscan_*`) | config/fixture-gen | batch | `gen_dbscan` (lines 684-725) + `gen_knn_metric` (lines 814-960, per-metric loop) | exact |

The device front-end (core distances + the feature-metric pairwise distances) is NOT a new file — it CALLS `crates/mlrs-backend/src/prims/knn_graph.rs::knn_graph(..., include_self=true)` and the existing `distance`/direct kernels. The new device kernel is only the mutual-reachability GATHER (and only on the feature-metric/dense path; precomputed + the host MR computation are pure host).

---

## Pattern Assignments

### `crates/mlrs-algos/src/cluster/hdbscan.rs` (estimator — MODIFY)

**Analog:** `crates/mlrs-algos/src/cluster/dbscan.rs` (host-walk-over-device-result labels-only estimator) + the file's own Phase-12 shell scaffolding (keep builder/typestate verbatim).

**Keep from the existing shell (do NOT rewrite):** the builder (`HdbscanBuilder`, lines 184-280), `Hdbscan::new` as single-source-of-defaults (lines 118-134), `into_builder`/`hyperparams_eq`, the `Fit` consume-self → `Fitted` signature (lines 293-322), and the `Fitted`-only accessors (lines 325-343).

**Extend the `Metric` enum** (currently lines 48-52, `Euclidean`-only) to mirror `knn_graph.rs::Metric` (lines 60-75) PLUS `Precomputed`:
```rust
// hdbscan.rs current — extend to 6 variants matching knn_graph.rs::Metric + Precomputed
pub enum Metric {
    Euclidean,
    Manhattan,
    Cosine,
    Chebyshev,
    Minkowski { p: f64 },   // p validated >= 1 host-side (knn_graph precedent)
    Precomputed,            // NEW — X is the n×n distance matrix; bypass device front-end
}
```

**Resolve the deferred validation TODO** (currently lines 260-264 in `build`). Use the typed-error precedent — `AlgoError::InvalidMinSamples` already exists (`error.rs:126`); `max_cluster_size` needs a new `BuildError` variant mirroring `InvalidMinClusterSize` (`error.rs:605-610`):
```rust
// dbscan.rs:166-171 precedent for the min_samples >= 1 guard shape:
if self.min_samples < 1 {
    return Err(AlgoError::InvalidMinSamples { estimator: "dbscan", min_samples: self.min_samples });
}
// build()-side: min_samples >= 1 when Some; max_cluster_size 0 (unbounded) else >= min_cluster_size.
// Follow InvalidMinClusterSize (error.rs:605-610) as the template for a new InvalidMaxClusterSize.
```

**Add new fitted fields + `Fitted`-only accessors** following the shell's `labels_: Option<DeviceArray<..., i32>>` pattern (lines 98, 332-337). New: `probabilities_`/`outlier_scores_` (`Option<DeviceArray<..., F>>`), `centroids_`/`medoids_` (`Option<DeviceArray<..., F>>`), and `store_centers` hyperparameter. DBSCAN's two-fitted-field precedent (`labels_` + `core_sample_indices_`, dbscan.rs:75-78) is the multi-output template.

**Replace the trivial `fit` body** (lines 304-321) with the device-front-end → host-back-end pipeline. The orchestration shape is DBSCAN's `fit` (dbscan.rs:147-246): validate host-side BEFORE launch, run the device stage, `to_host` the device result, run the sequential host algorithm, then `from_host` the fitted state:
```rust
// dbscan.rs:181-244 — the exact "device stage → to_host → host sequential walk → from_host" shape:
//   1. validate geometry + hyperparams host-side (T-05-07-01 / ASVS V5)
//   2. let mask = eps_core_mask::<F>(pool, x, ...)?;          // device stage
//   3. host DFS over mask.is_core / mask.neighbors (lines 197-227)  // sequential host
//   4. let labels_dev = DeviceArray::from_host(pool, &labels);      // re-materialize device-resident
//   5. self.labels_ = Some(labels_dev);
// HDBSCAN substitutes: device stage = knn_graph(include_self=true) + mutual-reachability;
// host stage = mst → single_linkage → condense → stability → select → labelling/probs.
```

**Precomputed short-circuit (D-02):** mirror DBSCAN's host-side validation gate (dbscan.rs:160-179) — when `metric == Precomputed`, validate `x` is square `(n,n)` (and document symmetry), read it to host (`x.to_host(pool)`), and feed straight into core-distance + mutual-reachability, skipping `knn_graph`.

---

### `crates/mlrs-algos/src/cluster/hdbscan/*.rs` host back-end modules (NEW)

**Analog:** `crates/mlrs-algos/src/cluster/dbscan.rs` host DFS (lines 194-227) — the precedent for "the prim does the device n²-heavy work, the estimator owns the SEQUENTIAL host graph algorithm, ported to match the sklearn `.pyx` index-ordered traversal bit-for-bit."

**Host-scalar f64 bridging** (for MST distances / lambda / stability): use the shared `mlrs_core::{host_to_f64, f64_to_host}` pair (exported `lib.rs:23`), exactly as `spectral_embedding.rs` does (lines 241, 271-273):
```rust
// spectral_embedding.rs:241 — read a device buffer to host f64 for sequential scalar math:
let dd_host: Vec<f64> = dd.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
// ... pure host scalar algorithm ...
let out_dev = DeviceArray::from_host(pool, &out_host);   // re-materialize (line 284)
```

**HashMap bookkeeping** for stability/relabel/condensed-tree dicts: the `std::collections::HashMap` pattern in `label_perm.rs:23-36` (confusion/index maps) is the in-repo precedent — the sklearn `.pyx` uses `dict`/array bookkeeping that ports to `HashMap<i64,_>` / `Vec`.

**The D-04 unstable-argsort seam (mst.rs):** the RESEARCH (Pattern 4, Pitfall 1) is explicit that `np.argsort` (quicksort) is unstable and the gate fixtures should use DISTINCT MST edge weights so the sort is tie-free. The existing `label_perm.rs:64` shows the in-repo DETERMINISTIC tie-break idiom (`sort_by(|a,b| ... .then(...).then(...))`) — but note this is the mlrs lowest-index convention, which the CONTEXT canonical_refs WARNS must NOT be conflated with HDBSCAN's oracle tie-break. `mst.rs` must replicate the ORACLE rule (sklearn `mst_from_data_matrix` strict-`<` lowest-`j` for Variant B; `np.argmin` first-min for Variant A) — see RESEARCH Pattern 3.

---

### `crates/mlrs-kernels/src/mutual_reachability.rs` (device kernel — NEW)

**Analog:** `crates/mlrs-kernels/src/distance.rs::manhattan_dist` / `chebyshev_dist` (lines 66-126) — the proven cpu-MLIR-safe per-element 2D GATHER kernel shape. The mutual-reachability kernel is `out[i*n+j] = max(core[i], core[j], d_ij/alpha)` — structurally identical to `chebyshev_dist`'s running-max idiom.

**Imports/launch pattern** (copy verbatim from distance.rs:47, 66-91):
```rust
use cubecl::prelude::*;

#[cube(launch)]
pub fn mutual_reachability<F: Float + CubeElement>(
    d: &Array<F>,       // pairwise distance block (or X itself for precomputed-on-device)
    core: &Array<F>,    // per-row core distance, length n
    out: &mut Array<F>,
    rows_x: u32, rows_y: u32, /* alpha passed by value, scalar */ alpha: F,
) {
    let i = ABSOLUTE_POS_X;
    let j = ABSOLUTE_POS_Y;
    if i < rows_x {
        if j < rows_y {
            // STATEMENT-form running max (chebyshev_dist:117-123 precedent),
            // NEVER an if-expression in value position:
            let mut acc = d[(i * rows_y + j) as usize] / alpha;  // d_ij / alpha
            let ci = core[i as usize];
            let cj = core[j as usize];
            if ci > acc { acc = ci; }
            if cj > acc { acc = cj; }
            out[(i * rows_y + j) as usize] = acc;
        }
    }
}
```

**cpu-MLIR landmines (MANDATORY — from `.claude/skills/spike-findings-mlrs/references/cpu-mlir-kernel-authoring.md`):**
- 2D `ABSOLUTE_POS_X`/`ABSOLUTE_POS_Y`, `CubeDim {x:16, y:16}`, ceiling-div counts, guarded `if i<rows { if j<cols {...} }` — use `knn_graph.rs::launch_dims_2d` (lines 481-490) verbatim. NEVER a bare 1D `ABSOLUTE_POS` launch (FINDING 002-A, loud pass failure → reads back zeros).
- BANNED entirely (panic at launch): `SharedMemory`, `Atomic`, `F::INFINITY`, mutable-`bool` scans, descending-shift loops. The MR kernel is SharedMemory-free by construction (per-element, no cross-thread state).
- NO cross-sibling-loop accumulator (FINDING 002-B, SILENT miscompile). The MR kernel has no loop — inert here, but if a feature-loop variant is added, obey it.
- `.abs()` is the only allowed instance form; use static `F::powf` if any exponent is needed. Scalars (`alpha`) pass BY VALUE in cubecl 0.10 (no `ScalarArg`).
- Re-export line in `mlrs-kernels/src/lib.rs` follows the existing `pub use distance::{manhattan_dist, chebyshev_dist, minkowski_dist, self_drop_gather}` precedent.

**Host launch wrapper** (in the backend, calling this kernel): copy the `unsafe { ArrayArg::from_raw_parts(handle.clone(), len) }` 2-arg-by-value idiom and the SAFETY comment from `knn_graph.rs::compute_tile_distance` (lines 286-323).

---

### `crates/mlrs-core/src/label_perm.rs` (utility — MODIFY in place)

**Analog:** the existing `best_match_accuracy` (lines 93-111) and `best_mapping` (lines 51-78) in the SAME file. Add a `best_match_accuracy_pinned_noise` (RESEARCH Code Examples spec) that pins `-1 → -1`:
```rust
// Extend the existing module. Reuse best_mapping (line 51) over labels with -1
// FILTERED OUT of both vocabularies, then force-insert (-1, -1):
pub fn best_match_accuracy_pinned_noise(pred: &[i64], reference: &[i64]) -> f64 {
    // 1. build best_mapping (line 51) over pred/reference with -1 removed from both
    // 2. map.insert(-1, -1) unconditionally
    // 3. remap pred (a pred==-1 stays -1); accuracy = exact matches incl. -1==-1
}
```
Existing `confusion`/`sorted_unique` (lines 23-43) already densify arbitrary `i64` labels — filter `-1` before passing in. Add the `pub use` line to `mlrs-core/src/lib.rs:24` alongside the existing `best_match_accuracy` export.

---

### `crates/mlrs-algos/tests/hdbscan_test.rs` (test — REPLACE shell tests)

**Analog:** `crates/mlrs-algos/tests/kmeans_test.rs` — the exact-label-up-to-permutation + value-gate-up-to-same-permutation precedent (D-09). REMOVE the shell's `fit_roundtrip` all-`-1` test (lines 62-94) — fit no longer returns all-`-1`. KEEP `defaults_equal` / `build_rejects_bad_min_cluster_size` (extend the latter for min_samples/max_cluster_size validation).

**Fixture loader + f32/f64 cast helpers** (copy verbatim from kmeans_test.rs:39-62):
```rust
fn fixture(name: &str) -> PathBuf { /* workspace_root/tests/fixtures/<name> — kmeans_test.rs:39-46 */ }
fn f64_to<F: Pod>(v: f64) -> F { /* kmeans_test.rs:48-54 */ }
fn host_to_f64<F: Pod>(v: F) -> f64 { /* kmeans_test.rs:56-62 */ }
```

**Exact-label gate (HDBS-02)** — copy the `best_match_accuracy == 1.0` shape (kmeans_test.rs:140-144), but use the NEW pinned-noise matcher:
```rust
let acc = mlrs_core::best_match_accuracy_pinned_noise(&labels, &labels_ref);
assert!((acc - 1.0).abs() < f64::EPSILON, "{label}: labels not a -1-pinned permutation of sklearn");
```

**Value gate with SAME permutation (probabilities/centers — HDBS-01/03/04):** copy the per-cluster `best_mapping` then `assert_close` pattern (kmeans_test.rs:148-156). Probabilities/outlier_scores are PER-POINT (compare directly after the point already carries its label); centroids/medoids are PER-CLUSTER (map each fitted cluster id to its sklearn id via `best_mapping`, then compare rows — Pitfall 6). The `assert_close` allclose helper (abs-OR-rel ≤1e-5) is kmeans_test.rs:66-84.

**f64 capability gate (verbatim, every f64 test):** kmeans_test.rs:183-189:
```rust
if capability::skip_f64_with_log() { println!("hdbscan f64 backend={backend}: SKIPPED"); return; }
```

**Memory gate (PoolStats):** the `pool.stats().live_bytes` re-fit-no-growth pattern is in the shell test (`hdbscan_test.rs:99-133`, `fit_no_leak`). For the device-front-end n×n gate, assert `pool.stats().peak_bytes` stays sub-quadratic — `PoolStats { reuses, peak_bytes, live_bytes }` (pool.rs:36-47); planner sets the exact assertion (CONTEXT Claude's-discretion).

---

### `scripts/gen_oracle.py` (fixture-gen — MODIFY)

**Analog:** `gen_dbscan` (lines 684-725, the `c()` dtype-cast + `np.savez` labels-only clustering fixture) and `gen_knn_metric` (lines 814-960, the per-metric loop with metric-in-FILENAME-only and the duplicate-point R-9 design).

**Per-metric + f32/f64 dispatch:** mirror the `gen_knn_metric` dispatch in `main()` (the per-metric loop precedent) — one fixture per `{euclidean, manhattan, cosine, chebyshev, minkowski, precomputed}` × `{f32, f64}`. Metric tag rides the FILENAME (`hdbscan_{metric}_{dtype}_seed{seed}.npz`), NEVER an in-blob string (`load_npz` decodes only 4/8-byte float arrays — oracle.rs:115-135).

**The `c()` cast + `np.savez`** convention (gen_dbscan:711-724) — store `X`, `labels`, `probabilities`, `centroids`, `medoids` (sklearn), plus `hdb_labels`/`outlier_scores` (hdbscan 0.8.44, D-07). The RESEARCH Code-Examples skeleton (`gen_hdbscan`) gives the exact kwargs incl. `copy=True` (silence the sklearn 1.10 FutureWarning) and `metric_params={'p':3.0}` for minkowski.

**Distinct-MST-edge-weight design (Pitfall 1):** gate fixtures use distinct-weight blobs (like `gen_knn`'s "spread points so distances are distinct" + per-row offset, lines 757-760); a SEPARATE tie-heavy + duplicate-point fixture (like `gen_knn_metric`'s `x[DUP_B]=x[DUP_A]`, line 862) characterizes whether ties flip labels. Nested-density designs for the non-default eom/leaf/ε/max_cluster_size knobs (Pitfall 5).

**Regen environment:** `/tmp` venv with `numpy>=1.26 scikit-learn==1.9.0 hdbscan==0.8.44` (PEP-668; fixtures are committed blobs — project memory "oracle-fixture-regen-needs-venv").

---

## Shared Patterns

### Device-front-end → host-back-end orchestration
**Source:** `crates/mlrs-algos/src/cluster/dbscan.rs::fit` (lines 147-246)
**Apply to:** `hdbscan.rs::fit` (and all host back-end submodules)
The canonical mlrs "dodge the GPU-atomics wall" shape: device prim does the n²-heavy parallel work, `to_host` the result ONCE (the documented round-trip), run the inherently-sequential algorithm on the host ported to match the sklearn `.pyx` index-ordering bit-for-bit, then `from_host` the fitted state device-resident.
```rust
let mask = eps_core_mask::<F>(pool, x, n, p, eps, min_samples)?;   // device
// ... sequential host walk over mask.is_core / mask.neighbors ...   // host port
let labels_dev = DeviceArray::from_host(pool, &labels);             // re-materialize
self.labels_ = Some(labels_dev);
```

### Host-side validation BEFORE any unsafe launch (ASVS V5 / T-13-06)
**Source:** `knn_graph.rs::validate_geometry` (lines 411-476) + `dbscan.rs::fit` guards (lines 160-179)
**Apply to:** `hdbscan.rs::fit` (precomputed squareness, alpha>0, minkowski p>=1, n×n/n·k overflow) and the new kernel host wrapper
Typed `PrimError::ShapeMismatch` / `AlgoError::Invalid*` returned for bad geometry/params; `checked_mul` overflow guards (knn_graph.rs:420, 465-474) before any `pool.acquire`. The Minkowski `p >= 1` validation (knn_graph.rs:444-453) is the exact precedent.

### cpu-MLIR-safe kernel authoring
**Source:** `.claude/skills/spike-findings-mlrs/references/cpu-mlir-kernel-authoring.md` + `crates/mlrs-kernels/src/distance.rs` (the doc-comment contract, lines 12-45)
**Apply to:** `mutual_reachability.rs`
Per-element 2D GATHER, statement-form `if` running-max, `F`/`u32` accumulators only, no SharedMemory/Atomic/`F::INFINITY`/mutable-bool/shift-loop, `launch_dims_2d` shape, scalars by value. The per-metric oracle MUST assert VALUES incl. a duplicate-point row (R-9), not just non-panic.

### Oracle fixture loading (no Python at test time)
**Source:** `crates/mlrs-core/src/oracle.rs::load_npz` (lines 77-83) + `OracleCase::expect_f64` (lines 60-63)
**Apply to:** `hdbscan_test.rs`
`load_npz(fixture("hdbscan_<metric>_<dtype>_seed42.npz"))` → `case.expect_f64("labels")`. Decodes only 4/8-byte float arrays (so labels are stored as float-valued, cast `as i64` in the test — kmeans_test.rs:133 precedent).

### Exact-label + same-permutation value gates
**Source:** `crates/mlrs-algos/tests/kmeans_test.rs::run_centers_labels` (lines 128-157)
**Apply to:** all `hdbscan_test.rs` oracle tests
`best_match_accuracy(_pinned_noise) == 1.0` for labels; `best_mapping` then per-cluster `assert_close` (≤1e-5 abs-OR-rel) for centers; per-point direct `assert_close` for probabilities/outlier_scores.

### Single-source-of-defaults builder + typestate
**Source:** the existing `hdbscan.rs` shell (`Hdbscan::new` lines 118-134, builder lines 184-280, `Fit` consume-self lines 293-322)
**Apply to:** keep verbatim; only EXTEND (enum, fields, validation, fit body).

---

## No Analog Found

No file in this phase lacks a strong in-repo analog. Two SEAMS have only a partial structural analog (the algorithm itself has an authoritative EXTERNAL oracle — the sklearn `_hdbscan/*.pyx` read verbatim in RESEARCH — rather than an in-repo one):

| File / seam | Role | Data Flow | Note |
|-------------|------|-----------|------|
| `hdbscan/mst.rs` (the unstable-argsort tie-break, D-04) | host | transform | No in-repo unstable-sort precedent; `label_perm.rs:64` shows the DETERMINISTIC tie-break idiom but the ORACLE rule (sklearn `np.argsort` quicksort / `np.argmin` first-min) must be ported from RESEARCH Pattern 3-4. This is the pre-planning SPIKE's TRUE GATE (D-05). |
| `hdbscan/condense.rs` / `stability.rs` / `select.rs` / `glosh.rs` | host | transform | No in-repo tree-condensation/stability precedent; port the sklearn `_tree.pyx` (and hdbscan `_hdbscan_tree.pyx` for GLOSH) line-for-line per RESEARCH Patterns 5-7 + Code Examples. DBSCAN's host-walk is the structural analog (sequential host scalar over device-produced data), not the algorithm analog. |

## Metadata

**Analog search scope:** `crates/mlrs-algos/src/cluster/`, `crates/mlrs-kernels/src/`, `crates/mlrs-backend/src/prims/`, `crates/mlrs-core/src/`, `crates/mlrs-algos/tests/`, `scripts/gen_oracle.py`, `.claude/skills/spike-findings-mlrs/`
**Files scanned:** hdbscan.rs (shell), dbscan.rs, kmeans.rs, spectral_embedding.rs, cluster/mod.rs, knn_graph.rs, distance.rs (kernels), label_perm.rs, oracle.rs, error.rs, kmeans_test.rs, hdbscan_test.rs (shell), dbscan gen + knn_metric gen + main() dispatch in gen_oracle.py, pool.rs, both spike-findings references
**Pattern extraction date:** 2026-06-24
