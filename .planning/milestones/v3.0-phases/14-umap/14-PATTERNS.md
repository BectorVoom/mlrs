# Phase 14: UMAP - Pattern Map

**Mapped:** 2026-06-23
**Files analyzed:** 4 (3 created/modified + 1 fixture-generator)
**Analogs found:** 4 / 4 (every artifact has a strong in-repo analog)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-algos/src/manifold/umap.rs` (MODIFY: real `fit`/`fit_transform`/`transform` + extend `Metric` to 5) | estimator (manifold) | request-response (host orchestration over prims) | `crates/mlrs-algos/src/cluster/spectral_embedding.rs` (Laplacian→eig→recover host orchestration) | role + flow exact |
| `crates/mlrs-kernels/src/umap_layout.rs` (NEW `umap_layout_step<F>` GATHER kernel) | kernel (device) | batch / transform (per-owner GATHER SGD step) | `crates/mlrs-kernels/src/topk.rs::select_k` (per-owner `CUBE_POS_X`/`UNIT_POS_X==0` GATHER) + `crates/mlrs-kernels/src/sgd.rs` (two-pass SGD math) | role + flow exact |
| `crates/mlrs-backend/src/prims/sgd.rs::sgd_solve` (PATTERN to mirror, NOT modified) | prim (host epoch driver) | batch (host loop → per-step launch → readback) | itself — `sgd_solve` IS the precedent the Spike flag names | exact precedent |
| Host stage fns (smooth-kNN ρ/σ, membership, t-conorm, init_graph_transform, LM a/b, neg-sample draws) — in `umap.rs` (or a private `umap_internals.rs` sibling) | utility (host numerics) | transform (host array math over `(n,k)` KNN) | `crates/mlrs-backend/src/prims/rng.rs` (host SplitMix64 + Box–Muller + Fisher–Yates host glue) | role-match |
| `crates/mlrs-algos/tests/umap_test.rs` (MODIFY: add value-gate + property-gate + reproducibility + transform) | test | request-response | itself (existing shell tests) + `*_test.rs` oracle-fixture precedent (`OracleCase`/`load_npz` + `skip_f64_with_log`) | role exact |

**Reuse prims (called as-is, NOT created):** `knn_graph` (`prims/knn_graph.rs`), `laplacian` (`prims/laplacian.rs`), `eig` (`prims/eig.rs`), `recover` (`cluster/spectral.rs`), `SplitMix64`/`permutation` (`prims/rng.rs`).

---

## Pattern Assignments

### `crates/mlrs-algos/src/manifold/umap.rs` — `fit`/`transform` bodies (estimator, host orchestration)

**Analog:** `crates/mlrs-algos/src/cluster/spectral_embedding.rs` (the Laplacian→eig→recover orchestration the spectral-init step reuses verbatim).

**Imports pattern** (spectral_embedding.rs:32-45) — the exact prim imports the new `fit` body needs (add `knn_graph` + `rng`):
```rust
use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::laplacian::laplacian;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};
use crate::cluster::spectral::recover;   // shared spectral host recovery
use crate::error::AlgoError;
```

**Spectral-init core pattern** (spectral_embedding.rs:139-143, 239-283) — the load-bearing sequence to copy for UMAP's `init='spectral'`:
```rust
// Size cap → fallback (UMAP FALLS BACK to random init above the cap, does NOT error like SE):
const MAX_DIM: usize = 64;            // == eig.rs MAX_DIM; the Jacobi ceiling (D-05)
if n_samples > MAX_DIM { /* UMAP: random uniform(-10,10) init instead of NSamplesExceedsMaxDim */ }

let (l, dd) = laplacian::<F>(pool, &a, n_samples)?;            // umap L = I − D^-1/2 G D^-1/2 (byte-identical)
let dd_host: Vec<f64> = dd.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
let (w_desc, v_desc) = eig::<F>(pool, &l, n_samples, Some(l_out))?;   // DESCENDING, Jacobi, n≤64
let v_host = v_desc.to_host(pool);
let init = recover::<F>(&v_host, &dd_host, n_samples, self.n_components, /*drop_first=*/ true);
// then noisy_scale_coords(max=10, noise=1e-4) via SplitMix64 normal draws (RESEARCH Pattern 4)
```
NOTE (Pitfall 3 / Q3): umap-learn's `spectral_layout` applies NO sign flip; `recover` DOES. Either compare the spectral value-gate up-to-sign per column, or thread a `recover(..., sign_flip=false)` flag. Resolve by dumping + diffing (Assumption A3).

**`fit` signature + geometry-guard pattern** (umap.rs:383-394, current shell — keep the typestate, replace the zeros body):
```rust
fn fit(self, pool: &mut BufferPool<ActiveRuntime>, x: &DeviceArray<ActiveRuntime, F>,
       _y: Option<&DeviceArray<ActiveRuntime, F>>, shape: (usize, usize),
) -> Result<Umap<F, Fitted>, AlgoError> {
    let (n, p) = shape;
    validate_geometry(x, shape)?;     // data-DEPENDENT guard BEFORE any launch (keep)
    // …real pipeline replaces `let zeros = vec![F::from_int(0i64); n*self.n_components];`…
}
```

**KNN call (reuse prim)** — `include_self=false` for UMAP (knn_graph.rs:127-135):
```rust
let (knn_idx, knn_dist) = knn_graph::<F>(pool, x, (n, d), self.n_neighbors,
    metric /* map Umap::Metric → knn_graph::Metric */, /*include_self=*/ false, p)?;
```

**`Metric` enum extension** (REPLACE umap.rs:44-48) — mirror `knn_graph::Metric` shape EXACTLY (knn_graph.rs:60-75, Pitfall 4):
```rust
pub enum Metric {
    Euclidean,
    Manhattan,
    Cosine,
    Chebyshev,
    Minkowski { p: f64 },   // carry `p` as f64 (matches the prim; avoids lossy conversion)
}
```

**transform frozen-subset pattern** (umap.rs:452-466, current shell — keep the `p != n_features_in_` `ShapeMismatch` guard; replace the zeros body with the D-03 path): KNN(new→train) → membership → `init_graph_transform` row-normalized weighted avg → drive the SAME `umap_layout_step` with `owners = new points only`, `move_other=false` (RESEARCH Pattern 7).

---

### `crates/mlrs-kernels/src/umap_layout.rs` — `umap_layout_step<F>` (NEW device kernel, GATHER)

**Analog (launch shape):** `crates/mlrs-kernels/src/topk.rs::select_k` (per-owner `CUBE_POS_X`/`UNIT_POS_X==0`).
**Analog (SGD math + host driver):** `crates/mlrs-kernels/src/sgd.rs` + `crates/mlrs-backend/src/prims/sgd.rs::sgd_solve`.

**Launch-shape pattern — COPY VERBATIM** (topk.rs:62-67; the 002-A landmine guard — NEVER bare `ABSOLUTE_POS` 1D):
```rust
#[cube(launch)]
pub fn umap_layout_step<F: Float + CubeElement>(/* … */) {
    let row = CUBE_POS_X;                 // u32, one OWNER row per cube
    if row < n_owners {
        if UNIT_POS_X == 0u32 {
            // …per-owner attract/repel work, all in ONE outer loop body…
        }
    }
}
```
Launch (per cpu-mlir-kernel-authoring.md): `CubeCount::Static(n_owners, 1, 1)`, `CubeDim {x:1, y:1, z:1}`.

**Anti-pattern guard — cross-sibling-loop accumulator (002-B SILENT miscompile).** Accumulate each owner's coordinate delta INSIDE the same loop that consumes it; do NOT write a delta in one `while` and read it in a sibling `while`. See topk.rs:119-177 (nested `r`/`c` self-contained accumulate) for the proven self-contained-nested shape.

**Gradient math — mirror sgd.rs's `F`/`u32`-only forward `while` GATHER** (sgd.rs:60-124). Gradients run in SQUARED distance (RESEARCH Pattern 5 / §Code Examples, VERIFIED umap-learn 0.5.12):
```text
attractive (dist² > 0): grad = (-2·a·b·pow(dist², b−1)) / (a·pow(dist²,b)+1)  else 0
repulsive  (dist² > 0): grad = (2·gamma·b) / ((0.001+dist²)·(a·pow(dist²,b)+1))  (skip when neg_k==j; 0 at dist²==0)
grad_d = clip(grad·(cur_d − other_d), −4, 4)   // grad_d = 4.0 in the dist²==0 repulsive branch
cur_d += grad_d · alpha;   if move_other: other_d −= grad_d · alpha
```

**`pow` form (cpu-MLIR-proven):** use the STATIC `F::powf(dist_squared, b−1)`, NEVER instance `x.powf()` (cpu-mlir-kernel-authoring.md "Math (allowed)"; Spike 001 Minkowski-p).

**Clip without `F::INFINITY`** (banned constant; cpu-mlir-kernel-authoring.md "Banned entirely") — statement-form `if` (the topk.rs running-best idiom):
```rust
let mut g = v;
if g > hi { g = hi; }
if g < lo { g = lo; }   // hi=4.0, lo=-4.0 finite literals; NO F::INFINITY, NO max/min intrinsic
```

**Frozen-subset mode (D-03):** kernel takes `n_owners` (contiguous owner count) and writes only `embedding[owner]`; non-owner neighbor coords are read-only GATHER targets. `fit`: owners = all `n`, `move_other=true`. `transform`: owners = `m` new points placed contiguously AFTER the `n` frozen training rows, `move_other=false`.

**Negative-sample indices are a HOST-drawn device BUFFER (D-05), never device RNG.** The kernel GATHERs a pre-packed `neg_idx` buffer; no in-kernel RNG (backend-divergent, banned in rng.rs).

**Re-export idiom** (lib.rs:31-33 + 39, the sgd/topk precedent): add `pub mod umap_layout;` to `mlrs-kernels/src/lib.rs` and a `pub use self::umap_layout_step as …` inside the new file (file-disjoint, parallel-safe).

---

### Host epoch driver — mirror `crates/mlrs-backend/src/prims/sgd.rs::sgd_solve` (the Spike-flag-named precedent)

**Analog:** `sgd_solve` (sgd.rs:148-427). The exact shape: `validate_geometry` → host epoch loop → per-step kernel launch → readback → cap.

**Host launch idiom — COPY** (sgd.rs:235-258; cubecl 0.10 — `ArrayArg::from_raw_parts` 2-arg by-value, scalars by value, no `ScalarArg`):
```rust
let client = pool.client().clone();
let arg = unsafe { ArrayArg::from_raw_parts(handle.clone(), len) };
let count = CubeCount::Static(n_owners as u32, 1, 1);     // per-owner (topk shape), NOT ceil-div
let dim = CubeDim { x: 1, y: 1, z: 1 };
umap_layout_step::launch::<F, ActiveRuntime>(&client, count, dim, /*…by-value scalars…*/);
```

**Epoch-loop + alpha decay + RNG plumbing** (sgd.rs:216-418 epoch shell; RESEARCH Pattern 6):
```rust
for epoch in 0..n_epochs {
    let alpha = self.learning_rate * (1.0 - epoch as f64 / n_epochs as f64);   // umap alpha decay
    // host draws neg-sample indices: SplitMix64 substream seeded f(random_state, epoch, edge_id),
    //   next_below(n) (unbiased), packed into a per-epoch neg_idx device buffer (D-05, order-deterministic)
    // launch umap_layout_step over owners …
}
```

**`NotConverged`/iteration-cap precedent:** sgd.rs:185-189 (max_iter cap) — apply the same finite-cap discipline to the host LM a/b fit (ASVS V5 DoS guard: cap iterations + typed error, the eig/sgd `MAX_SWEEPS` precedent).

---

### Host stage fns — mirror `crates/mlrs-backend/src/prims/rng.rs` (host-glue numerics)

**Analog:** `rng.rs` — pure host numeric routines (Box–Muller, Fisher–Yates) with one upload, NO device kernel.

**SplitMix64 — reuse VERBATIM** (rng.rs:58-108). `next_f64()` for uniforms/noise, `next_below(bound)` for UNBIASED neg-sample draws (rng.rs:87-107 — NEVER `% n`, the biased-modulo anti-pattern). `permutation(seed, n)` (rng.rs:252-265) for any epoch-order shuffle.

**Box–Muller normal draw pattern** (rng.rs:144-173) — reuse for spectral-init `noise=1e-4` and random-init normal draws (cached-pair stream determinism is load-bearing for D-05).

**LM a/b fit (D-06, NEW host numeric, NO device kernel):** self-contained Gauss-Newton/LM, capped iterations + typed non-convergence error (the `optimal_t0`/`schedule_eta` host-f64 helper precedent, sgd.rs:481-514). Target curve `1/(1 + a·x^(2b))` fit to `linspace(0, spread*3, 300)` (RESEARCH §Code Examples). Value-gate ≤1e-5 vs umap `find_ab_params`.

**smooth-kNN ρ/σ binary search (NEW host numeric):** per-row f64 binary search, `n_iter=64`, `SMOOTH_K_TOLERANCE=1e-5`, `MIN_K_DIST_SCALE=1e-3`, `target=log2(n_neighbors)*bandwidth`; rho-first then search on `d−rho` (RESEARCH Pattern 1 — ORDER is load-bearing). `hi = inf` is HOST f64 only (fine; the `F::INFINITY` ban is DEVICE-kernel-only).

---

### `crates/mlrs-algos/tests/umap_test.rs` (test) — MODIFY

**Analog:** itself (existing shell tests, umap_test.rs:31-141) + the `*_test.rs` oracle-fixture precedent.

**f64 capability-gate pattern — COPY VERBATIM** (umap_test.rs:64-71):
```rust
let backend = capability::active_backend_name();
capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
if capability::skip_f64_with_log() {
    println!("umap <case> f64 backend={backend}: SKIPPED (no f64 support)");
    return;            // cpu runs f64; rocm SKIPS f64-with-log (project memory)
}
```

**Pool/device setup pattern** (umap_test.rs:72-78): `runtime::active_client()` → `BufferPool::new(client)` → `DeviceArray::from_host`.

**Oracle-fixture value-gate (NEW):** load committed `.npz` via `mlrs_core::load_npz`/`OracleCase` (4/8-byte float arrays only — encode indices as float, metric tag in filename, per the `gen_knn_metric` precedent). Per-stage × per-metric matrix (D-02): `gen_umap_fuzzy_*`, `gen_umap_spectral_*`, `gen_umap_ab_*`, `gen_umap_layout_*`, `gen_umap_transform_*`. Generators added to `scripts/gen_oracle.py`, regen in `/tmp` numpy venv + `umap-learn==0.5.12` (PEP 668; fixtures are committed blobs — project memory landmines).

**Property-gate (NEW, NOT element-wise):** SplitMix64 ≠ umap's Tausworthe → coordinates can never match. Gate on trustworthiness / kNN-overlap / downstream-ARI relative to umap-learn (D-04). Reproducibility test: byte-identical across two runs per (backend, dtype) (D-05).

---

## Shared Patterns

### Geometry validation BEFORE any launch (ASVS V5)
**Source:** `sgd.rs:520-546 validate_geometry`, `knn_graph.rs:140`, `laplacian.rs:104`, `rng.rs:274-292`.
**Apply to:** the new `fit`/`transform` bodies (keep the existing `validate_geometry(x, shape)?` umap.rs:394), the host epoch driver, and every prim call. A malformed shape returns typed `PrimError::ShapeMismatch`/`AlgoError`, never an OOB device read.
```rust
validate_geometry(x.len(), y.len(), n, d)?;   // before any unsafe { ArrayArg::from_raw_parts(...) }
```

### cpu-MLIR kernel authoring (the primary correctness gate)
**Source:** `.claude/skills/spike-findings-mlrs/references/cpu-mlir-kernel-authoring.md`, `topk.rs`, `sgd.rs`.
**Apply to:** `umap_layout_step` ONLY (all host stages dodge this by being host-side).
- Launch: `CUBE_POS_X`/`UNIT_POS_X==0` per-owner shape, NEVER bare `ABSOLUTE_POS` 1D (002-A pass failure → reads back zeros).
- Accumulate deltas INSIDE the consuming loop, NEVER across sibling loops (002-B SILENT miscompile).
- Banned entirely: `SharedMemory`, `Atomic`, `F::INFINITY`, mutable-`bool` scans, descending-shift loops.
- Static `F::powf`/`F::exp`/`.sqrt()`/`.abs()` only; NEVER instance `x.powf()`.
- Generic `<F: Float + CubeElement>`, NO backend feature; validate f64 (the gate) AND f32.

### Host RNG reproducibility (ASVS V6, D-05)
**Source:** `rng.rs` (`SplitMix64`, `next_below`, `permutation`).
**Apply to:** init noise, random init, negative-sampling draws, any shuffle. NON-crypto, seeded from the caller's `random_state` u64, NEVER `OsRng`/`rand`/device RNG. Unbiased `next_below` (rejection), never `% n`. Every draw a pure function of `(random_state, epoch, edge)` so `fit`/`transform` are byte-identical run-to-run per (backend, dtype).

### Host f64 readback for value-gatable intermediates
**Source:** `sgd.rs:177` / `spectral_embedding.rs:241,271` (`.to_host(pool).iter().map(|&v| host_to_f64(v))`).
**Apply to:** every deterministic stage (smooth-kNN, membership, t-conorm, spectral coords, a/b). Host f64 matches umap-learn's own numpy/numba f64 without device-reduction-order drift — the reason the deterministic stages value-gate to ≤1e-5.

### Iteration-cap + typed non-convergence (ASVS DoS guard)
**Source:** `sgd.rs:185-189` (`max_iter` cap), `eig` `MAX_SWEEPS`.
**Apply to:** the host LM a/b fit (cap + typed error) and the epoch driver.

---

## No Analog Found

None. Every Phase-14 artifact has a strong in-repo analog:

| Artifact | Covered by |
|----------|-----------|
| `umap_layout_step` kernel | `topk.rs` (launch shape) + `sgd.rs` (SGD math) — composite, not a single file, but both shapes are proven |
| LM a/b fit + smooth-kNN search | `rng.rs` host-glue idiom + `sgd.rs` host-f64 helper precedent (the specific LM math is NEW host numerics — no exact analog, but the SHAPE is established; RESEARCH formulas are VERIFIED) |
| transform query-vs-train KNN | OPEN (Pitfall 5 / Q2): the Phase-13 prim is X-vs-X only. Compose `distance` + `top_k` in-estimator (no self-drop, new≠train) OR add a thin query-vs-train arg. Resolve in the spike (Assumption A2). |

## Metadata

**Analog search scope:** `crates/mlrs-algos/src/{manifold,cluster}/`, `crates/mlrs-backend/src/prims/`, `crates/mlrs-kernels/src/`, `crates/mlrs-algos/tests/`, `.claude/skills/spike-findings-mlrs/`.
**Files scanned:** umap.rs, spectral_embedding.rs, spectral.rs, knn_graph.rs, laplacian.rs, eig.rs, rng.rs, sgd.rs (prim), sgd.rs (kernel), topk.rs, lib.rs, umap_test.rs, both spike references.
**Pattern extraction date:** 2026-06-23
