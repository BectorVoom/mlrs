# Phase 9: Spectral Family - Pattern Map

**Mapped:** 2026-06-21
**Files analyzed:** 13 (8 new, 5 modified)
**Analogs found:** 13 / 13 (every file has a strong, validated in-repo analog)

This phase is ~90% wiring over already-validated v1 + Phase-8 code. Every new
file copies an existing, currently-passing analog. The ONLY genuinely new device
code is one SharedMemory-free elementwise map kernel in `mlrs-kernels` (the
`d_inv_sqrt` / `L` build), and one new host orchestration prim (`laplacian.rs`).

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-backend/src/prims/laplacian.rs` | prim (host orchestration) | transform | `crates/mlrs-backend/src/prims/kernel_matrix.rs` | exact (base-op → in-place map idiom) |
| `crates/mlrs-kernels/src/elementwise.rs` (add `laplacian_map`) | kernel (device map) | transform | `elementwise.rs::rbf_map` / `div_by_row` (`:114` / `:301`) | exact (per-element map, gather divisor) |
| `crates/mlrs-backend/src/prims/mod.rs` | config (module index) | n/a | same file `:24` (`pub mod kernel_matrix;`) | exact |
| `crates/mlrs-algos/src/cluster/spectral_embedding.rs` | estimator | transform (CRUD-ish) | `crates/mlrs-algos/src/cluster/kmeans.rs` + `kernel_ridge.rs` | role-match (Fit + host accessor) |
| `crates/mlrs-algos/src/cluster/spectral_clustering.rs` | estimator | request-response | `crates/mlrs-algos/src/cluster/kmeans.rs` | exact (Fit + PredictLabels / labels_) |
| `crates/mlrs-algos/src/cluster/mod.rs` | config (module index) | n/a | same file `:21-22` | exact |
| `crates/mlrs-algos/src/error.rs` (add variants) | model (error enum) | n/a | same file `InvalidK` `:99` / `InvalidGamma` `:260` | exact |
| `crates/mlrs-py/src/estimators/spectral.rs` | binding | request-response | `crates/mlrs-py/src/estimators/kernel.rs` | exact (`any_estimator!` ×2) |
| `crates/mlrs-py/src/estimators/mod.rs` | config | n/a | same file `:34` (`pub mod kernel;`) | exact |
| `crates/mlrs-py/src/lib.rs` (register classes) | config | n/a | same file `:171-172` (`add_class::<PyKernelRidge>`) | exact |
| `crates/mlrs-backend/tests/laplacian_test.rs` | test (oracle + gate) | n/a | `memory_gate_test.rs` + `kernel_matrix_test.rs` | exact |
| `crates/mlrs-algos/tests/spectral_*_test.rs` | test (oracle) | n/a | `crates/mlrs-algos/tests/kernel_ridge_test.rs` | exact |
| `scripts/gen_oracle.py` (add `gen_spectral_*`) | test fixture gen | n/a | `gen_oracle.py::gen_kernel_ridge` `:1384` | exact |

---

## Pattern Assignments

### `crates/mlrs-backend/src/prims/laplacian.rs` (prim, transform) — NEW PRIM-09

**Analog:** `crates/mlrs-backend/src/prims/kernel_matrix.rs` (the base-op →
in-place-map host orchestration). RESEARCH Open Q2 fixes the contract:
`laplacian.rs` RECEIVES a ready affinity `A (n×n)` and RETURNS `(L, dd)`; the
estimator builds the affinity. So the signature mirrors `kernel_matrix` but with
`n` (single dim) and a two-buffer return.

**Imports pattern** (copy `kernel_matrix.rs:42-52`):
```rust
use bytemuck::Pod;
use cubecl::prelude::*;
use mlrs_core::PrimError;
use mlrs_kernels::laplacian_map;              // NEW map kernel (add to mlrs-kernels)
use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::reduce::{row_reduce, ReducePath, ScalarOp};   // degree vector
use crate::runtime::ActiveRuntime;
```

**Validate-before-launch guard** (copy `kernel_matrix.rs:133-142` / `eig.rs:90-92`):
geometry guard (`a.len() == n*n`, `n != 0`) runs BEFORE any launch, returns a
typed `PrimError`. The `n ≤ 64` MAX_DIM cap is the ESTIMATOR's job (D-06,
`AlgoError`), NOT the prim — `laplacian.rs` stays cap-agnostic like
`kernel_matrix.rs`.

**Core pattern — the 4-step scipy `_laplacian_dense` host orchestration**
(RESEARCH Pattern 1, pinned to scipy):
```rust
// 1. zero the diagonal of A (in-place map: tid where row==col → 0) — RESEARCH
//    "Affinity diagonal handling" CONFIRMED: fill_diagonal(m,0) BEFORE degree.
// 2. w = row_reduce(A, n, n, ScalarOp::Sum, ReducePath::Shared)?  // degree, GATHER
//    (reduce.rs:180 — single-owner row reduction, no scatter/atomics)
// 3. dd[i] = if w[i]==0 {1} else {sqrt(w[i])}   // typed-zero guard, NO F::INFINITY
// 4. L[i,j] = -A[i,j]/(dd[i]*dd[j]); L[i,i] = if w[i]==0 {0} else {1}  // one map
```

**In-place / out-buffer reuse** (copy `kernel_matrix.rs:196-227`
`launch_map_in_place` + `launch_dims_1d` verbatim): the `L`/`d_inv_sqrt` map runs
over the affinity buffer; `dd` is a length-`n` side output. The dense `n×n` L
stays in GLOBAL memory (no SharedMemory tile — RESEARCH Anti-Pattern; gfx1100 LDS
≤ 65536 B).

**Anti-patterns (RESEARCH §Anti-Patterns):** no `F::INFINITY` for `1/sqrt(0)`
(the typed-zero guard replaces it); no SharedMemory in the new map kernel; no
edge-scatter/atomics for degree (use `row_reduce(Sum)`).

---

### `crates/mlrs-kernels/src/elementwise.rs` — add `laplacian_map` (kernel, transform)

**Analog:** `elementwise.rs::rbf_map` (`:114`) for the `#[cube(launch)]` shape and
`div_by_row` (`:301`) for the per-row gather-divisor pattern. Both are
SharedMemory-free, atomics-free, no-`F::INFINITY` — exactly the required profile.

**Map kernel shape** (copy `rbf_map:114-120` structure; gather `dd` by index like
`div_by_row:301-313`):
```rust
/// Normalized-Laplacian build map (PRIM-09): given affinity `a` (n×n row-major,
/// diagonal already zeroed) and the typed-zero guard vector `dd` (length n,
/// dd[i] = sqrt(degree_i) or 1 if isolated), write
///   out[i*n+j] = -a[i*n+j] / (dd[i]*dd[j])   for i != j
///   out[i*n+i] = if degree_i == 0 { 0 } else { 1 }   (= 1 - isolated)
/// STATEMENT-form guard (like clamp_nonneg:45 / kde_epanechnikov_map:194),
/// NEVER F::INFINITY. SharedMemory-free, atomics-free. `n` a scalar u32 by value.
#[cube(launch)]
pub fn laplacian_map<F: Float + CubeElement>(
    a: &Array<F>, dd: &Array<F>, output: &mut Array<F>, n: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < a.len() {
        let i = tid / n as usize;
        let j = tid % n as usize;
        // ... STATEMENT-form: diagonal forced to 0 or 1; off-diag = -a/(dd_i*dd_j)
    }
}
```
A separate tiny `d_inv_sqrt`-style guard (step 3, `dd = where(w==0,1,sqrt(w))`)
mirrors `sqrt_elem:61` + a statement guard. Register the new symbol via `pub use`
in `mlrs-kernels/src/lib.rs` (same place `rbf_map` is re-exported).

**Critical (RESEARCH Pattern 1):** step-4 diagonal is `1 - isolated` = **0** for a
zero-degree node, NOT 1 — this is the "no NaN/inf on zero-degree nodes" success
criterion.

---

### `crates/mlrs-backend/src/prims/mod.rs` (config) — register `laplacian`

**Analog:** the same file, `:24` (`pub mod kernel_matrix;`). Add `pub mod laplacian;`
next to the other prim registrations (`:12-38`). One line; file-disjoint Wave-0.

---

### `crates/mlrs-algos/src/cluster/spectral_clustering.rs` (estimator, request-response) — NEW SPECTRAL-02

**Analog:** `crates/mlrs-algos/src/cluster/kmeans.rs` (Fit + PredictLabels, i32
labels, validate-before-launch, device-resident fitted state).

**Imports + struct + `new`** (copy `kmeans.rs:52-124` shape): store
`n_clusters` / `n_components` (None→n_clusters, D-11) / `gamma` (default 1.0,
D-04) / `affinity` (default "rbf", D-01) / `seed`; fitted `labels_:
Option<DeviceArray<.., i32>>` (the `kmeans.rs:95` i32 idiom).

**Validate-before-launch** (copy `kmeans.rs:234-252` verbatim shape): reject
`n_samples > 64` → the NEW `AlgoError::NSamplesExceedsMaxDim` (D-06) BEFORE any
affinity/Laplacian/eig; reject `n_clusters` via existing `InvalidK` (`error.rs:99`);
reject non-finite gamma via existing `InvalidGamma` (`error.rs:260`).

**Core pipeline** (RESEARCH System Diagram + Pattern 2):
```rust
// affinity A = kernel_matrix(x, x, Kernel::Rbf{gamma:1.0})   (D-02, kernel_matrix.rs:164)
//   OR knn-connectivity (Pattern 3) for affinity="nearest_neighbors"
// (L, dd) = laplacian(pool, &A, n)?                          (NEW prim)
// (w_desc, v_desc) = eig(pool, &L, n, None)?                 (eig.rs:75, DESCENDING)
// reverse desc→asc; take n_components smallest cols (drop_first=FALSE for SC, D-11)
// maps = V_slice.T / dd  (recovery, D-07) → KMeans::new(n_clusters, seed).fit(maps)
// labels_ = kmeans.labels_   (KMeans::new, kmeans.rs:112 — NOT with_init, D-10)
```
Eig column extraction: copy RESEARCH "Eig column extraction" snippet (V col-major,
`v_host[col*n + i]`, smallest = descending col `n-1`).

**Trait surface** (RESEARCH §Recommended Structure): `Fit` + `PredictLabels`
(reuse the i32 surface) or `labels_(&pool)` accessor + `fit_predict` — mirror
KMeans. NO new trait (Discretion confirmed).

---

### `crates/mlrs-algos/src/cluster/spectral_embedding.rs` (estimator, transform) — NEW SPECTRAL-01

**Analog:** `kmeans.rs` for Fit/struct/validate; `kernel_ridge.rs` for the
`gamma=None → 1/n_features`-at-fit resolution (D-04). Differs from SC only in:
- default affinity `"nearest_neighbors"` (D-01); default `n_components = 2` (D-08).
- `gamma=None → 1/n_features` resolved at fit (copy KernelRidge's at-fit gamma
  resolution; RESEARCH D-04: `gamma.unwrap_or(1.0/n_features as F)`).
- `drop_first = TRUE` → compute `n_components+1` smallest, drop the trivial row 0
  (RESEARCH D-07/D-08 ordering: slice asc → `/dd` recovery → sign-flip → drop).
- store `embedding_: Option<DeviceArray<.., F>>` (n × n_components); host accessor
  `embedding_(&pool)` (NOT `Transform` — sklearn has no out-of-sample transform).

**Deterministic sign flip** (RESEARCH "Don't Hand-Roll" + D-07): reproduce
`_deterministic_vector_sign_flip` (5 lines) on the host `k × n` array BEFORE the
final `.T` — argmax(|row|) → sign of that element → multiply row.

**kNN-connectivity affinity builder** (D-03, RESEARCH Pattern 3): `distance(x, x,
sqrt=false)` (`distance.rs:79`) → `top_k(.., k=n_neighbors, sqrt=false)`
(`topk.rs:61`, lowest-index tie-break) → set `A[i,j]=1` for the k smallest cols
(includes self) → `A = 0.5·(A + Aᵀ)`. Small host math on the n×k indices.

---

### `crates/mlrs-algos/src/cluster/mod.rs` (config) — register the spectral modules

**Analog:** same file `:21-22` (`pub mod dbscan; pub mod kmeans;`). Add `pub mod
spectral_clustering;` + `pub mod spectral_embedding;` and a `pub use` of each
estimator (mirror `density/mod.rs:30-31` / `kernel_ridge/mod.rs:27-28`
`pub mod` + `pub use` pattern). Estimator plans edit this `mod.rs`, NOT `lib.rs`
(the cluster module is already registered in `lib.rs:42`).

---

### `crates/mlrs-algos/src/error.rs` (model) — add `NSamplesExceedsMaxDim` (+ optional `InvalidNNeighbors`)

**Analog:** existing `InvalidK` (`:99`) and `InvalidGamma` (`:260`) variants — same
`#[error(...)]` + struct-variant + `estimator: &'static str` shape.

**New variant** (D-06; copy the `InvalidK:95-106` shape):
```rust
/// A spectral estimator was given more samples than the dense eig MAX_DIM cap
/// (`n_samples > 64`). The normalized Laplacian is n×n and v1 `eig` caps n ≤ 64
/// (MAX_DIM); rejected at `fit` BEFORE any affinity/Laplacian/eig launch so the
/// message names the spectral cap, not eig's generic PrimError::NotSquare (D-06 /
/// ASVS V5).
#[error(
    "estimator '{estimator}': n_samples = {n_samples} exceeds the dense \
     eigensolver cap (must be <= {max} = MAX_DIM)"
)]
NSamplesExceedsMaxDim { estimator: &'static str, n_samples: usize, max: usize },
```
Reuse `InvalidK` for `n_clusters`/`n_neighbors` 1..=n_samples and `InvalidGamma`
for non-finite gamma (both already exist — no new variant needed for those).

---

### `crates/mlrs-py/src/estimators/spectral.rs` (binding, request-response) — NEW

**Analog:** `crates/mlrs-py/src/estimators/kernel.rs` (two `any_estimator!`
invocations, parse-name helper, Unfit/F32/F64, `py.detach`, `guard_f64`,
dtype-suffixed accessors).

**Imports** (copy `kernel.rs:35-45`): `pyo3` prelude/PyValueError;
`mlrs_algos::cluster::{SpectralEmbedding, SpectralClustering}`; `crate::errors::{algo_err_to_py, not_fitted}`; `crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype}`.

**`any_estimator!` ×2** (copy `kernel.rs:100-104` / `:297-301`):
```rust
crate::any_estimator! {
    any: AnySpectralEmbedding,
    algo: mlrs_algos::cluster::spectral_embedding::SpectralEmbedding,
    unfit: { n_components: usize, affinity: String, gamma: Option<f64>, n_neighbors: usize },
}
crate::any_estimator! {
    any: AnySpectralClustering,
    algo: mlrs_algos::cluster::spectral_clustering::SpectralClustering,
    unfit: { n_clusters: usize, n_components: Option<usize>, affinity: String, gamma: f64, n_neighbors: usize, seed: u64 },
}
```

**`fit` body** (copy `kernel.rs:159-213` verbatim shape): `capsule_to_array` →
`float_dtype` → `py.detach(|| { let mut pool = global_pool().lock()...; match dt {
F32 => validated_f32 + Estimator::<f32>::new(..).fit(..), F64 => guard_f64()? +
validated_f64 + ... } })`. The `guard_f64()?` on the F64 arm runs BEFORE upload
(D-04).

**Accessors** (copy `kernel.rs:266-280` dtype-suffixed pattern):
`embedding_f32`/`_f64` (SE, `Vec<F>` via `to_host_metered`),
`labels_` (SC, `Vec<i32>` — mirror the cluster.rs PyKMeans labels accessor),
`is_fitted` / `dtype`.

**`#[pyclass(name="SpectralEmbedding")]` / `#[pyclass(name="SpectralClustering")]`**
with `#[new]` + `#[pyo3(signature=(...))]` defaults (copy `kernel.rs:111-152`),
sklearn defaults: SE `n_components=2, affinity="nearest_neighbors", gamma=None,
n_neighbors=10`; SC `n_clusters=8, affinity="rbf", gamma=1.0, n_neighbors=10`.

---

### `crates/mlrs-py/src/estimators/mod.rs` + `crates/mlrs-py/src/lib.rs` (config) — register

- `estimators/mod.rs`: add `pub mod spectral;` (copy `:34` `pub mod kernel;`).
- `lib.rs`: `use estimators::spectral::{PySpectralEmbedding, PySpectralClustering};`
  (copy `:139`) + `m.add_class::<PySpectralEmbedding>()?;` /
  `m.add_class::<PySpectralClustering>()?;` (copy `:171-172`).

---

### Tests

#### `crates/mlrs-backend/tests/laplacian_test.rs` (test) — PRIM-09

**Analog:** `crates/mlrs-backend/tests/memory_gate_test.rs` (the PoolStats
build-failing gate shape: `BufferPool`, runtime line log, `live_bytes`/`peak_bytes`/
`reuses` assertions — `:88-90` `#[test]` + `env_logger`) AND the value-vs-host-reference
shape from `kernel_matrix_test.rs`.
Three cases (RESEARCH Test Map): (1) `L = I − D^-1/2 A D^-1/2` vs host reference
f32+f64; (2) `zero_degree` no NaN/inf on an isolated-node fixture; (3) `memory_gate`
PoolStats reuse-bounded / no-mid-pipeline-readback (mirror `memory_gate_test.rs`).

#### `crates/mlrs-algos/tests/spectral_embedding_test.rs` / `spectral_clustering_test.rs` (test) — SPECTRAL-01/02

**Analog:** `crates/mlrs-algos/tests/kernel_ridge_test.rs` (oracle-backed
estimator test: `load_npz` fixture loader `:35`, `fixture()` path helper `:49-56`,
`host_to_f64`/`f64_to` `:58-72`, `assert_close` abs-OR-rel with strict 1e-5
absolute floor `:78`, documented `*_F32_BAND` `:47`, `skip_f64_with_log` gate).
- SE: rbf + nearest_neighbors(default) value-match after sign align; degenerate
  subspace test (D-09, principal angles); `reject_oversize` (n>64 → typed
  `AlgoError::NSamplesExceedsMaxDim` BEFORE device).
- SC: `labels_` up to permutation via `best_match_accuracy`/`label_perm` on the
  well-separated fixture (D-10) — EXACT labels, no band (copy the label-permutation
  compare from `kmeans_test.rs` / `dbscan_test.rs`).

#### `scripts/gen_oracle.py` — `gen_spectral_embedding` / `gen_spectral_clustering`

**Analog:** `gen_oracle.py::gen_kernel_ridge` (`:1384-1454`): `np.random.default_rng(seed)`
fixture, fit the sklearn estimator with its OWN default constructor (D-01 — NO
override, unlike KernelRidge's explicit kwargs), `np.ascontiguousarray(...).astype(dtype)`
row-major, `np.savez` committed `.npz` (`tests/fixtures/`), registered in the
`__main__` dtype loop (`:1644-1648`). SE stores `X` + `embedding_`; SC stores `X` +
`labels_`. `n_samples ≤ 64` (D-05). Regen needs a /tmp venv with numpy+scipy+sklearn
(PEP 668, [[oracle-fixture-regen-needs-venv]]).

---

## Shared Patterns

### Validate-before-launch (ASVS V5)
**Source:** `crates/mlrs-algos/src/cluster/kmeans.rs:234-252` (the canonical
reject-then-launch block) + `error.rs` typed variants.
**Apply to:** both spectral estimators' `fit` (n_samples>64, n_clusters/n_neighbors,
gamma) BEFORE any affinity/Laplacian/eig/KMeans call.

### base-op → in-place map (host orchestration)
**Source:** `crates/mlrs-backend/src/prims/kernel_matrix.rs:144-227`
(`launch_map_in_place` + `launch_dims_1d` are copy-paste-ready).
**Apply to:** `laplacian.rs` (the `d_inv_sqrt`/`L` map over the affinity buffer).

### SharedMemory-free, no-`F::INFINITY` device map (cpu-MLIR-safe)
**Source:** `crates/mlrs-kernels/src/elementwise.rs` — `rbf_map:114`,
`div_by_row:301` (gather divisor), `kde_epanechnikov_map:194` (STATEMENT-form
guard). All atomics-free, SharedMemory-free, no infinity constant.
**Apply to:** the new `laplacian_map` kernel ([[cubecl-cpu-no-shared-memory]]).

### PyO3 `any_estimator!` dtype dispatch + GIL release + f64 guard
**Source:** `crates/mlrs-py/src/estimators/kernel.rs:100-213` (full fit body) +
`crates/mlrs-py/src/dispatch.rs:85` (macro). `py.detach` + `global_pool().lock()` +
`guard_f64()?`-before-upload.
**Apply to:** `estimators/spectral.rs` (both estimators).

### Oracle harness (committed `.npz`)
**Source:** `crates/mlrs-algos/tests/kernel_ridge_test.rs` (`load_npz`,
`assert_close` strict-1e-5-floor + documented f32 band) + `gen_oracle.py::gen_kernel_ridge`.
**Apply to:** all three spectral test files. f64 strict via `skip_f64_with_log`;
f32-on-rocm documented band (~1e-4, RESEARCH Pitfall 7); SC labels are an EXACT
hard gate (no band).

### eig reuse — descending→ascending + drop-trivial
**Source:** `crates/mlrs-backend/src/prims/eig.rs:75` (returns w DESCENDING,
V col-major `v_host[c*n+r]`, `MAX_DIM=64` at `:221`).
**Apply to:** both estimators' post-eig recovery (reverse, slice smallest, `/dd`,
sign-flip, drop-first per D-07/D-08). RESEARCH "Eig column extraction" snippet is
copy-ready.

## No Analog Found

None. Every Phase-9 file has a strong in-repo analog (the phase was deliberately
ordered after Phase 8 so the `kernel_matrix(Rbf)` affinity seam and the
`any_estimator!`/oracle harness already exist).

## Metadata

**Analog search scope:** `crates/mlrs-backend/src/prims/`,
`crates/mlrs-kernels/src/`, `crates/mlrs-algos/src/cluster/`,
`crates/mlrs-algos/src/{error.rs,lib.rs}`, `crates/mlrs-py/src/{dispatch.rs,lib.rs,estimators/}`,
`crates/*/tests/`, `scripts/gen_oracle.py`.
**Files scanned:** ~18 (read) across 5 crates + the oracle generator.
**Pattern extraction date:** 2026-06-21
