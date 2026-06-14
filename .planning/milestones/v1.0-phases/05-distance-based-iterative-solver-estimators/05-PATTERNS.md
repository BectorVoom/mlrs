# Phase 5: Distance-Based & Iterative-Solver Estimators - Pattern Map

**Mapped:** 2026-06-12
**Files analyzed:** 27 new + 4 modified (file count is Claude's-discretion granularity from RESEARCH §"Recommended Project Structure"; planner may merge/split)
**Analogs found:** 31 / 31 (every new file has a concrete in-tree analog — this phase is "mostly assembly")

> **Primitive-first ordering is binding (D-01).** Within each estimator's plan
> set, every NEW kernel + its `mlrs-backend/prims` launch wrapper + its standalone
> oracle + its PoolStats memory-gate case MUST land and pass BEFORE the estimator
> consumes it — the Phase-4 Cholesky precedent (`cholesky.rs` kernel →
> `prims/cholesky.rs` wrapper → `cholesky_test.rs` → `memory_gate_test.rs` →
> `ridge.rs`). The classification table is ordered kernels → wrappers → estimators
> → tests deliberately.

---

## File Classification

### Kernels (feature-free `#[cube]`, `crates/mlrs-kernels/src/`)

| New File | Role | Data Flow | Closest Analog | Match |
|----------|------|-----------|----------------|-------|
| `topk.rs` | kernel | transform (partial-select) | `reduce.rs` `argmin_shared` (value+index pair carry, lowest-index tie-break) | role+flow |
| `kmeans.rs` (D² + centroid-sum-by-label + inertia) | kernel | reduce / transform | `reduce.rs` `reduce_sumsq_shared` + `elementwise.rs` `center_columns` | role+flow |
| `dbscan.rs` (eps-threshold + per-row core-count → mask) | kernel | transform (2D map + per-row count) | `elementwise.rs` `dist_combine_clamp` (2D `(i,j)` map over a `rows×cols` matrix) | role+flow |
| `coordinate.rs` (CD soft-threshold + axpy residual update) | kernel | transform (dot + axpy) | `elementwise.rs` `scale`/`center_columns` (per-element map with scalar `F` args) + `reduce.rs` (column dot) | role-match |
| `lbfgs.rs` (stable softmax loss + grad) | kernel | reduce + transform | `reduce.rs` `reduce_max_shared` (the logsumexp max) + `dist_combine_clamp` (2D logits map) | role-match |

### Launch wrappers + host orchestration (`crates/mlrs-backend/src/prims/`)

| New File | Role | Data Flow | Closest Analog | Match |
|----------|------|-----------|----------------|-------|
| `prims/topk.rs` | launch-wrapper | request-response | `prims/distance.rs` (composes gemm+reduce, validates-before-launch, device-resident out) | exact |
| `prims/kmeans.rs` (D²-sample host-RNG + Lloyd update) | launch-wrapper | request-response + host RNG | `prims/distance.rs` + `prims/reduce.rs::argmin_rows` (host loop over device segments) | role+flow |
| `prims/dbscan.rs` (n² dist + core mask → host readback) | launch-wrapper | request-response (terminal readback) | `prims/distance.rs` (n² matrix) + `prims/cholesky.rs` (tiny `info` readback idiom) | role+flow |
| `prims/coordinate_descent.rs` (CD host loop, 1 scalar/iter) | launch-wrapper | iterative / host-loop | `prims/cholesky.rs` (validate→launch→scalar-readback) + `prims/reduce.rs` (gap dots) | role-match |
| `prims/lbfgs.rs` (L-BFGS two-loop host + softmax launches) | launch-wrapper | iterative / host-loop | `prims/cholesky.rs` (the new-primitive wrapper shape) + `prims/gemm.rs` (Xw/grad) | role-match |

### Estimators + trait/error extensions (`crates/mlrs-algos/src/`)

| New/Modified File | Role | Data Flow | Closest Analog | Match |
|-------------------|------|-----------|----------------|-------|
| `traits.rs` (MODIFY: add label/KNeighbors/proba traits) | trait-extension | — | `traits.rs` existing `Fit`/`Predict`/`Transform` | exact |
| `error.rs` (MODIFY: add hyperparameter variants) | trait-extension | — | `error.rs` existing `AlgoError` variants | exact |
| `lib.rs` (MODIFY: add `cluster`/`neighbors` mods) | config | — | `lib.rs` existing `pub mod` index | exact |
| `cluster/kmeans.rs` | estimator | CRUD (fit/predict) | `linear/ridge.rs` (new-prim consumer, center-then-solve, device-resident state) | role-match |
| `cluster/dbscan.rs` | estimator | event-driven (host graph walk) | `linear/ridge.rs` (Fit shape) + host DFS (no estimator analog) | partial |
| `neighbors/nearest.rs` | estimator | request-response | `linear/ridge.rs` (Fit) + `prims/reduce.rs::argmin_rows` (host index readback) | role-match |
| `neighbors/classifier.rs` | estimator | CRUD (predict/proba) | `cluster/kmeans.rs` (sibling, same vote path) + `ridge.rs` | role-match |
| `neighbors/regressor.rs` | estimator | CRUD (predict) | `linear/ridge.rs` `Predict<F>` impl | role-match |
| `linear/coordinate_descent.rs` (shared CD host loop) | estimator (shared helper) | iterative | `linear/ridge.rs` (centering + center-then-solve intercept) | role-match |
| `linear/lasso.rs` (= ElasticNet `l1_ratio=1`) | estimator | iterative | `linear/ridge.rs` thin wrapper shape | role-match |
| `linear/elastic_net.rs` | estimator | iterative | `linear/ridge.rs` + `coordinate_descent.rs` helper | role-match |
| `linear/logistic.rs` (L-BFGS multinomial) | estimator | iterative | `linear/ridge.rs` (Fit) + `prims/lbfgs.rs` | role-match |
| `cluster/mod.rs`, `neighbors/mod.rs` (NEW) | config | — | `linear/mod.rs` module-index | exact |

### Oracle tests (`crates/*/tests/`) + fixture gen

| New/Modified File | Role | Data Flow | Closest Analog | Match |
|-------------------|------|-----------|----------------|-------|
| `mlrs-backend/tests/topk_test.rs` | oracle-test | — | `cholesky_test.rs` (standalone prim oracle) | exact |
| `mlrs-backend/tests/kmeans_prim_test.rs` | oracle-test | — | `cholesky_test.rs` | exact |
| `mlrs-backend/tests/dbscan_prim_test.rs` | oracle-test | — | `cholesky_test.rs` | exact |
| `mlrs-backend/tests/coordinate_descent_test.rs` | oracle-test | — | `cholesky_test.rs` (algebraic-invariant on convex objective) | exact |
| `mlrs-backend/tests/lbfgs_test.rs` (convex-quadratic standalone) | oracle-test | — | `cholesky_test.rs` (`‖A·x−b‖` invariant → `x*=A⁻¹b` invariant) | exact |
| `mlrs-algos/tests/kmeans_test.rs` | oracle-test | — | `ridge_test.rs` (load_npz→fit→assert_close) + `label_perm` (D-09) | exact |
| `mlrs-algos/tests/dbscan_test.rs` | oracle-test | — | `ridge_test.rs` + `label_perm` | exact |
| `mlrs-algos/tests/neighbors_test.rs` (×3) | oracle-test | — | `ridge_test.rs` | exact |
| `mlrs-algos/tests/lasso_test.rs` / `elastic_net_test.rs` / `logistic_test.rs` | oracle-test | — | `ridge_test.rs` (alpha-sweep harness) | exact |
| `scripts/gen_oracle.py` (MODIFY: 6 new generators) | fixture-gen | — | `gen_ridge` / `gen_cholesky` | exact |
| `mlrs-backend/tests/memory_gate_test.rs` (MODIFY: D-10 + DBSCAN cases) | memory-gate | — | existing `memory_gate_reuse_bounded` / `memory_gate_no_midpipeline_readback` gates | exact |

---

## Pattern Assignments

### Kernels — `crates/mlrs-kernels/src/*.rs`

**Shared kernel contract** (all new kernels, from `mlrs-kernels/src/lib.rs:1-7` + `reduce.rs:36-37`):
- Generic over `<F: Float + CubeElement>`, launched `::launch::<F, R>`, NO backend feature.
- `SharedMemory::<F>::new(N)` takes a **comptime** size; bound the active region by a runtime `n` arg (the `reduce.rs` 256-cap idiom and the `cholesky.rs` `MAX_DIM` cap).
- **No hardcoded plane width** — use `PLANE_DIM` if a plane path, else a shared-mem tree (D-03).
- `continue` is NOT supported in `#[cube]` — use `if`-wrapped branches (`cholesky.rs:155`).
- Scalar args passed by value, no `ScalarArg` wrapper (`dist_combine_clamp`'s `rows: u32, cols: u32`; `scale`'s `factor: F`).
- After writing, add the kernel to `lib.rs` `pub mod` + `pub use` re-export block (`lib.rs:8-23`).

#### `topk.rs` — partial-select-k (D-02)

**Analog:** `reduce.rs` `argmin_shared` (lines 344-390) — carries `(value, index)` through the reduction with a lowest-index tie-break. Top-k generalizes this from k=1 to k.

**Value+index pair carry with lowest-index tie-break** (`reduce.rs:366-389` — replicate the tie rule exactly):
```rust
// Strictly smaller value wins; on a tie the LOWER index wins (D-02 convention).
if ov < cv {
    sval[tid as usize] = ov;
    sidx[tid as usize] = oi;
} else if ov == cv {
    if oi < ci { sidx[tid as usize] = oi; }
}
```
Output two arrays — `out_val: &mut Array<F>` (k distances/row) and `out_idx: &mut Array<u32>` (k indices/row) — exactly as `argmin_shared` writes `out_val`/`out_idx`. The host re-uploads `u32` indices as `i32` (D-06, see Shared Patterns).

#### `kmeans.rs` — D² + centroid-sum-by-label + inertia

**Analog (squared-norm reduction):** `reduce.rs` `reduce_sumsq_shared` (lines 161-186) for the D² and inertia accumulation. **Analog (per-element map):** `elementwise.rs` `center_columns` (lines 81-93) for the centroid sum-by-label scatter shape (`c = tid % cols`).

#### `dbscan.rs` — eps-threshold + per-row core-count

**Analog:** `elementwise.rs` `dist_combine_clamp` (lines 107-127) — the 2D `(i, j)` map over a `rows×cols` matrix with `if i < rows && j < cols` bounds-check. DBSCAN thresholds `D[i,j] <= eps²` and counts per row:
```rust
// dist_combine_clamp's 2D map shape is the template (elementwise.rs:116-126):
let i = ABSOLUTE_POS_X;
let j = ABSOLUTE_POS_Y;
if i < rows && j < cols {
    let idx = (i * cols + j) as usize;
    // ... DBSCAN: bit = (d2[idx] <= eps2); accumulate per-row count ...
}
```

#### `coordinate.rs` — CD soft-threshold + residual axpy

**Analog:** `elementwise.rs` `scale` (lines 64-79, scalar-`F` per-element map) for the residual axpy; column-dot reuses `reduce.rs`. The per-coordinate math (RESEARCH §"Code Examples" — un-normalized form, `l1_reg = α·l1_ratio·n`): `w_j = sign(t)·max(|t|−l1_reg, 0)/(norm2_cols[j]+l2_reg)`, then `R += (w_j_old − w_j)·X[:,j]`.

#### `lbfgs.rs` — stable softmax loss/grad

**Analog (the max for logsumexp):** `reduce.rs` `reduce_max_shared` (lines 304-329). **Analog (2D logits map):** `dist_combine_clamp`. The two-loop recursion + line search are **host-side** (D-10), NOT in the kernel — the kernel only emits loss + grad. Stable form (RESEARCH Pitfall 4): `m = max_k raw_k; lse = m + log(Σ exp(raw_k − m))`.

---

### Launch wrappers — `crates/mlrs-backend/src/prims/*.rs`

**Shared wrapper contract** (the Phase-4 Cholesky precedent — `prims/cholesky.rs`, `prims/distance.rs`):

**1. Validate geometry BEFORE any `unsafe` launch (ASVS V5)** (`prims/cholesky.rs:195-238`, `prims/distance.rs:186-228`):
```rust
fn validate_geometry(a_len: usize, /* ... */) -> Result<(), PrimError> {
    if rows.checked_mul(cols).map(|v| v != a_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch { operand: "x", rows, cols, len: a_len });
    }
    // ... each rows*cols == len, dims agree, optional `out` matches ...
    Ok(())
}
```

**2. Thread an optional reused `out` buffer through (D-11)** — every wrapper signature ends with `out: Option<DeviceArray<ActiveRuntime, F>>` and acquires from `pool` only when `None` (`prims/distance.rs:124-129`, `prims/cholesky.rs:113-123`).

**3. Launch via `kernel::launch::<F, ActiveRuntime>` with `ArrayArg::from_raw_parts`** at the VALIDATED element counts (`prims/cholesky.rs:137-154`, `prims/distance.rs:142-154`):
```rust
let client = pool.client().clone();
let count = CubeCount::Static(1, 1, 1);
let dim = CubeDim { x: n as u32, y: 1, z: 1 };
// SAFETY: lengths are the carried/validated element counts, NEVER raw caller geometry.
let a_arg = unsafe { ArrayArg::from_raw_parts(a.handle().clone(), n * n) };
my_kernel::launch::<F, ActiveRuntime>(&client, count, dim, a_arg, /* ... */, n as u32);
```

**4. Release transient scratch at its TRUE byte size; return a device-resident `DeviceArray`** (`prims/distance.rs:164-180`). The caller owns the returned `out`; never release it.

**5. Standard launch-config helpers** (`prims/distance.rs:234-253`): `launch_dims_2d` (16×16 ceil-div cube) and `launch_dims_1d` (256 block ceil-div).

#### `prims/dbscan.rs` & `prims/cholesky.rs` — the tiny-scalar-readback idiom

DBSCAN reads the core mask + adjacency back to host (D-04 documented exception). The readback pattern is `prims/cholesky.rs:167-169`:
```rust
let info_dev = DeviceArray::<ActiveRuntime, F>::from_raw(info_handle, 3);
let info = info_dev.to_host(pool);
info_dev.release_into(pool);
```

#### `prims/coordinate_descent.rs` & `prims/lbfgs.rs` — host-driven loop, 1 scalar/iter (D-10)

**Structure** (RESEARCH §"Pattern 2"): acquire all solver buffers (`R`, `norm2_cols`, `w`; or grad + history `(s,y)×m=10`) ONCE before the loop, reuse every iteration, read back exactly ONE scalar per outer convergence check (duality gap / max-proj-grad). The scalar-readback uses the same `to_host` idiom as `prims/cholesky.rs:168`. Pin the convergence constants from RESEARCH §"Standard Stack" (CD: `tol·‖y‖²`, `max_iter=1000`; L-BFGS: `m=10`, `gtol=1e-4`, `ftol=64·eps`, `maxiter=100`).

---

### Estimators — `crates/mlrs-algos/src/`

#### `traits.rs` (MODIFY) — add D-05/D-07 traits

**Analog:** the existing `Fit`/`Predict`/`Transform` traits (`traits.rs:46-113`). Mirror the exact bound + signature shape. New traits return integer `DeviceArray<ActiveRuntime, i32>` (D-06):
```rust
// Mirror Predict<F> (traits.rs:66-79) but return i32 labels (D-05/D-06):
pub trait PredictLabels<F> where F: Float + CubeElement + Pod {
    fn predict_labels(&self, pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>, shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError>;
}
// KNeighbors returns BOTH distances (F) and indices (i32) (D-07):
//   -> Result<(DeviceArray<ActiveRuntime, F>, DeviceArray<ActiveRuntime, i32>), AlgoError>
// PredictProba returns per-class fractions (F).
```
Re-export new traits from `lib.rs:33` (`pub use traits::{...}`).

#### `error.rs` (MODIFY) — add hyperparameter-guard variants

**Analog:** the existing `AlgoError::InvalidAlpha` / `InvalidNComponents` variants (`error.rs:34-57`). Add `InvalidK`, `InvalidEps`, `InvalidMinSamples`, `InvalidL1Ratio`, `InvalidC`, `NotConverged` in the SAME `#[error("...")]` + struct-variant style:
```rust
#[error("estimator '{estimator}': alpha = {alpha} is invalid (must be >= 0)")]
InvalidAlpha { estimator: &'static str, alpha: f64 },  // ← copy this exact shape
```
`#[from] PrimError` (line 91) already lets estimator methods `?` a prim call.

#### `cluster/kmeans.rs`, `cluster/dbscan.rs`, `neighbors/*`, `linear/{lasso,elastic_net,logistic,coordinate_descent}.rs`

**Primary analog:** `linear/ridge.rs` — the Phase-4 estimator that consumes a NEW primitive (Cholesky) and does center-then-solve. Replicate:

**Struct + `new` + device-resident fitted state + host accessors** (`ridge.rs:68-122`):
```rust
pub struct Ridge<F> {
    alpha: F, fit_intercept: bool,
    coef_: Option<DeviceArray<ActiveRuntime, F>>,        // device-resident (D-03)
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}
// accessor materializes host-side ON DEMAND, NotFitted before fit (ridge.rs:101-109):
pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
    self.coef_.as_ref().map(|c| c.to_host(pool))
        .ok_or(AlgoError::NotFitted { estimator: "ridge", operation: "coef_" })
}
```
KMeans stores `cluster_centers_` (F) + `labels_`/`core_sample_indices_` (i32). The `labels_` host accessor materializes the i32 array.

**`Fit` impl: validate hyperparameter + geometry BEFORE any prim launch** (`ridge.rs:135-166`):
```rust
let alpha64 = host_to_f64(self.alpha);
if alpha64 < 0.0 { return Err(AlgoError::InvalidAlpha { estimator: "ridge", alpha: alpha64 }); }
if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
    return Err(AlgoError::Prim(PrimError::ShapeMismatch { operand: "x", rows: n_samples, cols: n_features, len: x.len() }));
}
```
KMeans validates `k`, DBSCAN validates `eps`/`min_samples`, CD validates `alpha`/`l1_ratio`, LogReg validates `C` — each with the matching new `AlgoError` variant.

**Center-then-solve intercept (D-13 for Lasso/EN/LogReg)** — REUSE the Ridge centering exactly (`ridge.rs:168-203` host two-pass means + `intercept_ = ȳ − x̄·coef_` at `ridge.rs:285-293`). The `host_to_f64`/`f64_to_host` helpers (`ridge.rs:378-393`) are copied verbatim into each estimator file.

**`Predict<F>` impl (regressors + KMeans)** (`ridge.rs:313-373`): GEMM `X·coef`, broadcast intercept, device-resident output. `KNeighborsRegressor` and the linear models use this directly.

**`linear/lasso.rs` = thin wrapper over ElasticNet (`l1_ratio=1`)** — mirror the `linear/mod.rs:1-18` "deliberately different solvers, do not unify" doc note; lasso delegates to the shared `coordinate_descent.rs` helper.

**Module index files** (`cluster/mod.rs`, `neighbors/mod.rs`): copy `linear/mod.rs:20-21` (`pub mod kmeans; pub mod dbscan;`). The estimator plans add their own `pub mod` line; `lib.rs` adds the two new top-level `pub mod cluster; pub mod neighbors;`.

---

### Oracle tests — `crates/*/tests/`

#### Prim oracles (`mlrs-backend/tests/*_test.rs`)

**Analog:** `cholesky_test.rs` (the Phase-4 standalone-primitive oracle). Replicate:
- `fixture(name)` workspace-root resolver (`cholesky_test.rs:46-53` / `ridge_test.rs:43-50`).
- `host_to_f64`/`from_f64` size-dispatch helpers (`cholesky_test.rs:64-78`).
- An **algebraic invariant** check where no committed sklearn value fits — `cholesky_test.rs:9-12` `‖A·x − b‖ ≤ 1e-5`. **L-BFGS standalone uses this**: a convex quadratic `½xᵀAx − bᵀx` whose minimizer is `x* = A⁻¹b`, asserted within 1e-5 (RESEARCH Pitfall 5 — isolates "is my L-BFGS correct" from "does it match sklearn's path").
- f64 capability gate verbatim (`cholesky_test.rs:20-22`; `ridge_test.rs:170-180`):
```rust
if capability::skip_f64_with_log() { println!("... SKIPPED ..."); return; }
```

#### Estimator oracles (`mlrs-algos/tests/*_test.rs`)

**Analog:** `ridge_test.rs`. Replicate the full harness:
- `load_npz(fixture(...))` → build `DeviceArray::from_host` → `estimator.fit(&mut pool, &x_dev, Some(&y_dev), shape)` → materialize → `assert_close` (`ridge_test.rs:93-126`).
- The alpha-sweep loop (`ridge_test.rs:131-158`) is the template for the CD/LogReg penalty sweeps.
- `assert_close` abs-OR-rel 1e-5 with the strict-absolute arm never loosened (`ridge_test.rs:71-89`); import `Tolerance, F32_TOL, F64_TOL` from `mlrs_core`.
- **Clustering tests (KMeans/DBSCAN)** additionally use `mlrs_core::label_perm` for up-to-permutation comparison (D-09): `best_match_accuracy(pred_i64, ref_i64)` (`label_perm.rs:93-111`) must be `1.0`. KMeans tests inject fixed init centers (D-09) — the fixture carries an `init` array.

#### `scripts/gen_oracle.py` (MODIFY)

**Analog:** `gen_ridge` (lines 467-509) and `gen_cholesky`. Add `gen_kmeans`, `gen_dbscan`, `gen_knn`, `gen_lasso`, `gen_elastic_net`, `gen_logistic`. Each:
```python
def gen_kmeans(seed=SEED, dtype=np.float32) -> str:
    from sklearn.cluster import KMeans          # import inside the fn (gen_ridge:479)
    rng = np.random.default_rng(seed)           # numpy authoritative RNG (Pitfall 7)
    # ... fit sklearn, store X + injected init + cluster_centers_/labels_/inertia_ ...
    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    out_path = os.path.join(_FIXTURE_DIR, f"kmeans_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, X=c(x), init=c(init), centers=c(km.cluster_centers_),
             labels=c(km.labels_), inertia=c([km.inertia_]))
    return out_path
```
Register both dtype calls in `main()` (lines after the Phase-4 block, ~tail): `for dtype in (np.float32, np.float64): print(f"wrote {gen_kmeans(dtype=dtype)}")`. **Regen needs the `/tmp/oracle-venv`** (PEP 668 — committed blobs, not test-time; see MEMORY).

#### `memory_gate_test.rs` (MODIFY)

**Analog:** existing gates `memory_gate_reuse_bounded` (lines 85-197) and `memory_gate_no_midpipeline_readback` (lines 214-298). Add:

- **D-10 iterative-solver exception gate** (per CONTEXT D-10 + RESEARCH §"Pattern 2"): assert across the CD/L-BFGS outer loop that `allocations` is FLAT after warmup (buffers reused) and `read_backs` grows by exactly 1 per OUTER convergence check (one scalar, never a per-iteration array). Reuse the snapshot-per-iteration + `assert_eq!(live_after[iter], live_baseline, ...)` structure (`memory_gate_test.rs:106-178`). **State the exception in a doc comment** so it doesn't read as a regression of the no-mid-pipeline-readback rule.
- **DBSCAN n² bound gate** (per CONTEXT D-04): assert the n² distance matrix is the dominant allocation and is reused (not re-allocated per call), and document that DBSCAN DELIBERATELY reads the mask back (so the `read_backs == 0` mid-pipeline assertion of gate 2 does NOT apply to DBSCAN — it gets the bounded-allocation form instead).

The hard-assert idiom to copy (`memory_gate_test.rs:140-149`):
```rust
assert_eq!(live_after[iter], live_baseline,
    "D-10 gate 1a (live_bytes conserved) FAILED on {backend}: iter {iter} ...");
```

---

## Shared Patterns

### Validate-before-launch (ASVS V5) — ALL launch wrappers + estimator `fit`/`predict`
**Source:** `prims/cholesky.rs:195-238`, `prims/distance.rs:186-228`, `ridge.rs:135-166`
Every untrusted hyperparameter (`k`, `eps`, `min_samples`, `alpha`, `l1_ratio`, `C`) and geometry is checked and returns a typed `AlgoError`/`PrimError` BEFORE any `unsafe` launch — never an out-of-bounds device read.

### Device-resident fitted state + lazy host-materialize (D-03)
**Source:** `ridge.rs:76-78` (state) + `ridge.rs:101-121` (accessors)
**Apply to:** every estimator. `coef_`/`cluster_centers_`/`intercept_` are `Option<DeviceArray<..., F>>`; `labels_`/`core_sample_indices_`/neighbor `indices` are `Option<DeviceArray<..., i32>>`. Host copy only at the accessor/oracle boundary.

### i32 labels/indices from u32 argmin readback (D-06)
**Source:** RESEARCH §"Pattern 3" / `reduce.rs::argmin_rows` returns `Vec<u32>` (`prims/reduce.rs:338-356`); `DeviceArray<R, F: Pod>` is generic (`device_array.rs:50`), `BufferPool` is byte-keyed (`pool.rs:60-65`) → i32 works with ZERO pool/bridge changes.
**Apply to:** KMeans labels, KNN votes/indices, DBSCAN labels.
```rust
let labels_u32 = argmin_rows::<F>(pool, &dist, rows, k)?;           // host Vec<u32>
let labels_i32: Vec<i32> = labels_u32.iter().map(|&l| l as i32).collect();
let labels_dev: DeviceArray<ActiveRuntime, i32> = DeviceArray::from_host(pool, &labels_i32);
// DBSCAN noise = -1 is directly representable in i32.
```

### f64-on-rocm skip-with-log (D-07) — ALL tests
**Source:** `capability::skip_f64_with_log()` (`capability.rs:147`), used at `ridge_test.rs:174-177`, `cholesky_test.rs:20-22`
**Apply to:** every f64 test function — guard with `if capability::skip_f64_with_log() { return; }`. f32 runs on rocm; f64 on cpu.

### `host_to_f64` / `f64_to_host` size-dispatch helpers
**Source:** `ridge.rs:378-393`, `cholesky_test.rs:64-78`
**Apply to:** every estimator and test file doing host-side f64 combine (centering, intercept, gap math). Copy verbatim (per-file local helper, not shared — matches the existing convention).

### Tolerance + assert_close (1e-5 abs-OR-rel, strict-absolute never loosened)
**Source:** `mlrs_core::{assert_close, Tolerance, F32_TOL, F64_TOL}` (`lib.rs:20-25`), used `ridge_test.rs:71-89`
**Apply to:** all oracle tests. LogReg is the documented escape-hatch case (RESEARCH Pitfall 5) — gate on `predict_proba`/`predict` (gauge-invariant) at 1e-5, `coef_` at a looser per-family bound.

---

## No Analog Found

No file is fully analog-less, but two have only **partial** structural precedent — flag for the planner:

| File | Role | Data Flow | Gap (planner uses RESEARCH instead) |
|------|------|-----------|-------------------------------------|
| `cluster/dbscan.rs` (host DFS/union-find) | estimator | event-driven | No existing host sequential-graph-walk in the codebase. Pin the exact LIFO index-ordered DFS from RESEARCH §"DBSCAN core mask + host DFS" / Pitfall 7 (`_dbscan_inner.pyx`). The `Fit` shell mirrors `ridge.rs`; the DFS body is new. |
| `prims/lbfgs.rs` + `linear/logistic.rs` (two-loop recursion + strong-Wolfe line search) | launch-wrapper / estimator | iterative | No existing host iterative-optimizer. The WRAPPER shell mirrors `prims/cholesky.rs`; the L-BFGS host loop math is pinned in RESEARCH §"LogReg objective + gradient" / Pitfall 5 (`m=10`, scipy L-BFGS-B constants). HIGHEST PROJECT RISK — validate standalone on the convex quadratic FIRST. |

---

## Metadata

**Analog search scope:** `crates/mlrs-kernels/src/`, `crates/mlrs-backend/src/{prims,}/`, `crates/mlrs-algos/src/{linear,decomposition,}/`, `crates/mlrs-core/src/`, `crates/*/tests/`, `scripts/gen_oracle.py`
**Files scanned:** kernels (reduce, cholesky, elementwise, lib), backend prims (cholesky, distance, reduce, gemm, mod), pool, device_array, capability, algos (traits, error, lib, ridge, linear/mod), core (oracle, label_perm, lib, error), tests (ridge_test, cholesky_test, memory_gate_test), gen_oracle.py
**Pattern extraction date:** 2026-06-12
