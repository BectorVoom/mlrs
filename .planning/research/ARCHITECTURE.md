# Architecture Research — v3.0 Manifold Algorithms & Rust-Native API

**Domain:** sklearn-compatible ML estimator library (Rust/CubeCL rewrite of cuML), v3.0 manifold + builder-API milestone
**Researched:** 2026-06-22
**Confidence:** HIGH for placement / trait deltas / dispatch / shim / build order / the device-host split of each new prim (every claim mirrors a shipped file read this pass). MEDIUM for the HDBSCAN on-device MST feasibility and the UMAP negative-sampling layout kernel under cpu-MLIR — the two genuine unknowns that each need a per-phase research spike before planning.

**Supersedes for v3:** the v2.0 `v2.0-research/ARCHITECTURE.md` is VALIDATED (shipped). This file is the v3 integration architecture: how the KNN-graph prim + UMAP + HDBSCAN + the builder-API retrofit slot into the shipped five-crate layering. The v1/v2 architecture is REUSED, not re-researched.

---

## Standard Architecture

### The fixed five-crate seam (REUSE — do not change)

```
┌──────────────────────────────────────────────────────────────────────┐
│ mlrs-kernels  #[cube] generic-float kernels, BACKEND-FEATURE-FREE      │
│   v3 NEW: knn_graph symmetrize map, umap_layout (neg-sample) update,   │
│           mutual_reach map, (mst/condense are HOST — see HDBSCAN)      │
├──────────────────────────────────────────────────────────────────────┤
│ mlrs-backend  prims/*  validate-geometry → unsafe launch → Result      │
│   owns ActiveRuntime, BufferPool, DeviceArray; the ONLY launch site    │
│   v3 NEW: prims/knn_graph.rs  (reuses distance + topk)                 │
│           prims/umap_layout.rs (new neg-sample SGD-layout solver)      │
│           prims/mutual_reach.rs (small, reuses distance/topk)          │
│           prims/mst.rs + prims/condense.rs are HOST-side (no kernel)   │
│   reuse: distance, topk, eig, laplacian, reduce, gemm, rng             │
├──────────────────────────────────────────────────────────────────────┤
│ mlrs-algos  estimator structs<F>; impl Fit/Transform/PredictLabels     │
│   COMPOSE prims, never launch kernels (D-13)                           │
│   v3 NEW: manifold/umap.rs, cluster/hdbscan.rs                         │
│   v3 NEW: a shared builder convention (trait/derive) + retrofit        │
├──────────────────────────────────────────────────────────────────────┤
│ mlrs-py  #[pyclass] via any_estimator! enum (Unfit/F32/F64);           │
│   dtype-suffixed accessors; py.detach + guard_f64                      │
│   v3 NEW: estimators/manifold.rs (UMAP), add hdbscan to cluster.rs     │
│   v3 REUSE: python/mlrs/*.py shim (MlrsBase + sklearn mixins)          │
├──────────────────────────────────────────────────────────────────────┤
│ scripts/gen_oracle.py + tests/fixtures/*.npz (committed blobs)         │
│   v3 NEW: umap-learn oracle (property gate), hdbscan oracle (labels)   │
└──────────────────────────────────────────────────────────────────────┘
```

The dependency arrows are **acyclic and unchanged**. Every algorithm addition is a *new file + a `pub mod`/`pub use` line* in the relevant crate root. The single shared-edit points stay `mlrs-py/src/lib.rs` (pyclass registration) and the family-module `mod.rs` files. The builder retrofit is the ONE v3 work item that touches existing estimator files broadly — see §Pattern 3 and the build-order sequencing.

### Component Responsibilities (v3 deltas only)

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| `prims/knn_graph.rs` | Build the shared (indices, distances) KNN graph + symmetrize; the single feasibility-critical prim | distance → topk → host/device symmetrize; feature-free, GATHER idiom |
| `prims/umap_layout.rs` | The new neg-sampling SGD layout solver (the one genuinely new UMAP kernel) | host epoch/edge-sampling loop + per-edge attract/repel device update (GATHER, single-owner) |
| `prims/mutual_reach.rs` | Mutual-reachability distance for HDBSCAN | core-dist (= k-th col of KNN dist) + pairwise max; reuses topk/distance |
| `prims/mst.rs` (HOST) | Minimum spanning tree over mutual-reach (Prim's) | pure host CPU loop — pointer-chasing, NOT on device |
| `prims/condense.rs` (HOST) | Condensed tree + stability extraction | pure host CPU — recursive tree walk, NOT on device |
| `manifold/umap.rs` | UMAP estimator: KNN→fuzzy set→init→optimize | composes knn_graph + (eig\|rng) init + umap_layout |
| `cluster/hdbscan.rs` | HDBSCAN estimator: mutual-reach→MST→condense→extract | composes mutual_reach (device) + mst/condense (host) |
| builder convention | Idiomatic Rust `Estimator::builder().setter().build()?` across all 30+ estimators | a `derive`-or-macro standard generalizing the Phase-10 `*Builder` precedent |

---

## Integration Points — new vs modified vs reused, per feature

### Feature 1 — KNN-graph primitive (the shared, feasibility-critical prim)

**This is the keystone. Land it standalone with its own gate test BEFORE UMAP/HDBSCAN consume it** (primitive-first discipline; the v1/v2 pattern where the prim is gated and the estimator is "mostly assembly").

| Aspect | Disposition | Detail |
|---|---|---|
| Pairwise distance | **REUSE** | `prims/distance.rs::distance` (squared, order-preserving) — already validated. |
| k-NN selection | **REUSE** | `prims/topk.rs::top_k(dist, rows, cols, k, sqrt, ...)` returns `(distances F, indices u32)` per query row, lowest-index tie-break, optional boundary-sqrt. **This IS the NearestNeighbors prim.** |
| Self-exclusion / k+1 | **NEW (host glue)** | When the graph is over the training set itself, request `k+1` and drop the self-column (the precedent already lives in the spectral kNN-connectivity builder). |
| Symmetrization | **NEW** | A small symmetrize step producing the union/intersection graph that BOTH consumers need. **Device path:** a `knn_graph_symmetrize` elementwise/scatter map over a dense `n×n` working buffer at v3 oracle sizes (GATHER-safe: single-owner write per output cell, no atomics). **Host path fallback:** build the COO/CSR adjacency on host from the read-back `(indices, distances)` when `n` is large enough that dense `n×n` is wasteful — flag the size cutoff in PITFALLS. |
| Output contract | **NEW (the shared seam)** | Return BOTH `indices (n×k, i32)` AND `distances (n×k, F)` in the **un-symmetrized neighbor-list form** plus a **symmetrized weighted-graph form** (COO triplets `(row, col, weight)` or dense `n×n`). UMAP needs the symmetrized *fuzzy union* (probabilistic t-conorm `a + b − a·b`); HDBSCAN needs the raw `(indices, distances)` to compute core-distance + mutual reachability. **Contract: emit the neighbor-list `(idx, dist)` as the primitive output; let each estimator apply its own edge-weighting** (UMAP fuzzy-set vs HDBSCAN mutual-reach). The symmetrize helper is a shared utility both call. |

**Device/host split (the cpu-MLIR answer):**
- **Device:** distance (reuse), top-k select (reuse — already proven on cpu-MLIR), and the symmetrize map (new, single-owner GATHER write per cell — no SharedMemory, no cross-unit atomics).
- **Host:** the k+1/self-drop bookkeeping and the choice of dense-vs-COO representation. No pointer-chasing here — the KNN graph is *embarrassingly parallel per query row*, so it is fully GATHER-idiom-compatible. **This prim is feature-free and standalone-gateable**, exactly like the v2 prims.

**Is it feature-free?** YES. It is `distance` + `topk` + a symmetrize map, all generic over `<F: Float + CubeElement + Pod>` and over runtime, no backend feature. Its gate test is a `tests/knn_graph_test.rs` with PoolStats memory gate + a sklearn `kneighbors_graph` oracle (the connectivity/distances are 1e-5-comparable since selection is deterministic).

### Feature 2 — UMAP

UMAP = KNN graph → fuzzy simplicial set → low-dim init → SGD layout optimization.

| Stage | Disposition | Detail |
|---|---|---|
| KNN graph | **REUSE (Feature 1)** | `knn_graph` prim provides `(indices, distances)`. |
| Fuzzy simplicial set | **NEW (mostly host + small device map)** | Per-point: find ρ (distance to nearest neighbor) + binary-search σ for the local connectivity target (host loop, like the LR-schedule host arithmetic in `sgd.rs`); membership strength `exp(−(d−ρ)/σ)` is a small elementwise device map (new `umap_fuzzy_map` kernel, GATHER-safe). Symmetrize via the probabilistic t-conorm (reuse the Feature-1 symmetrize helper with the fuzzy-union weight). |
| Low-dim init | **REUSE — with a hard caveat** | Spectral init = smallest nontrivial eigenvectors of the graph Laplacian → **reuse `prims/laplacian.rs` + `prims/eig.rs` (the v2 spectral path verbatim)**. **CAVEAT:** the Jacobi `eig` prim has `MAX_DIM` (= 64, per `jacobi_eig`); the spectral estimators already cap `n_samples > MAX_DIM`. Dense Jacobi eig does NOT scale to UMAP-sized graphs. **Recommendation: ship `init="random"` (reuse `prims/rng.rs`) as the DEFAULT correctness path, and offer `init="spectral"` only under the eig `MAX_DIM` cap** — flag the size limit loudly in PITFALLS (Lanczos/sparse-eig is deferred, same call as v2). This keeps UMAP runnable at realistic sizes without a new eigensolver. |
| SGD layout optimization | **NEW kernel — does NOT reuse the two-pass SGD solver** | The v2 `prims/sgd.rs` is a *supervised linear-model* solver (per-sample margin → weight update over a fixed coef vector). UMAP layout is a *fundamentally different* update: **edge-sampled attractive forces + negative-sampled repulsive forces over the embedding coordinate matrix**, with per-epoch learning-rate decay. The host loop shape (epoch loop + LR schedule + per-batch device launch) is the SAME orchestration pattern as `sgd.rs`/`lbfgs.rs`, but the device kernel is new: `umap_layout_step` applying attract/repel gradient to embedding rows. **Build it as `prims/umap_layout.rs` with a new `umap_layout_step` kernel.** Negative sampling = host-drawn negative indices (reuse `prims/rng.rs` SplitMix64) fed to the GATHER kernel (single-owner update per embedding row — no atomics; this is the cpu-MLIR feasibility question to spike). |

**Data flow:** `X → knn_graph(X,k) → (idx, dist) → fuzzy_set (host σ/ρ + device map + symmetrize) → init (rng default | eig under cap) → umap_layout_step × epochs → embedding_`.

**Device/host split:** device = distance, topk, fuzzy map, symmetrize, layout step; host = σ/ρ binary search, edge/negative sampling schedule, LR decay, epoch loop. **The layout step's per-row single-owner GATHER update under cpu-MLIR is the MEDIUM-confidence item — spike it before planning the UMAP phase** (precedent: the v2 SGD solver launched on cpu-MLIR first try, which is encouraging).

**Correctness gate:** stochastic layout → NOT element-wise 1e-5. Property/structural gate à la RandomProjection (D-12): trustworthiness/continuity metric vs `umap-learn`, k-NN preservation in the embedding, seed-reproducibility. Oracle broadens to `umap-learn`.

### Feature 3 — HDBSCAN

HDBSCAN = mutual-reachability → MST → condensed cluster tree → stability extraction.

| Stage | Disposition | Detail |
|---|---|---|
| KNN graph / core distance | **REUSE (Feature 1)** | Core distance of point i = distance to its k-th neighbor = the k-th column of `knn_graph` distances. Pure reuse. |
| Mutual-reachability distance | **NEW (small device map)** | `mreach(a,b) = max(core_a, core_b, d(a,b))`. A small elementwise map over the (dense or neighbor-list) distance graph — `prims/mutual_reach.rs` with a `mutual_reach_map` kernel (GATHER-safe, no atomics). |
| **MST construction** | **NEW — HOST-SIDE (Prim's / Boruvka)** | This is the inherently sequential / pointer-chasing stage. **Prim's algorithm over the mutual-reach graph runs on the HOST** (read the mutual-reach graph back once, build the MST in a CPU loop). GPU Boruvka needs atomics for the parallel edge-contraction and fights the cpu-MLIR no-atomics constraint head-on — **do NOT attempt on-device MST in v3.** cuML uses a GPU MST (raft); mlrs deliberately stays host here. Feasible because at v3 oracle sizes the MST is cheap and the device→host read-back is a one-time cost. |
| Condensed tree + stability | **NEW — HOST-SIDE** | Building the single-linkage hierarchy from the MST, condensing it (min_cluster_size), and computing per-cluster stability is a recursive tree walk — pure host CPU. NOT on device (pointer-chasing, no parallelism win, no atomics available). |
| Label extraction (EOM) | **NEW — HOST-SIDE** | Excess-of-mass cluster selection from the condensed tree → final labels (`-1` noise sentinel, already representable in the `i32` `PredictLabels` contract). Host. |

**Data flow:** `X → knn_graph(X,k) → core_dist (k-th col) → mutual_reach (device map) → [READ BACK] → MST (host Prim's) → single-linkage tree (host) → condense (host) → stability/EOM (host) → labels_`.

**Device/host split (the cpu-MLIR answer):** **device = only the embarrassingly-parallel front half** (distance, topk, mutual-reach map); **host = the entire tree half** (MST, condensation, stability, extraction). This is the correct split *regardless* of backend — the tree algorithms have no data-parallel structure to exploit and the no-atomics constraint makes a GPU MST infeasible in v3. HDBSCAN is therefore a "device front-end, host tree back-end" estimator. The host code is plain Rust (`Vec`/union-find), no CubeCL.

**Correctness gate:** exact labels up to permutation (the hard gate), oracle = `hdbscan` / `sklearn.cluster.HDBSCAN`. Reuse the v1 `label_perm` helper.

### Feature 4 — Rust-native builder-pattern API

**Critical finding: the builder pattern is ALREADY PARTIALLY SHIPPED.** Phase 10 (`linear/mod.rs`) and Phase 11 (`naive_bayes/mod.rs`) INTRODUCED `Estimator::builder().setter(..).build() -> Result<_, BuildError>` for high-arity estimators, with `BuildError` already defined in `mlrs-algos/src/error.rs`. The v1 low-arity estimators (LogisticRegression, LinearRegression, Ridge, PCA, KMeans, NN…) still use `new(...)` / `with_opts(...)` and were **deliberately NOT retrofitted (D-02)**. So v3 work is: (a) elevate the existing ad-hoc per-estimator `*Builder` structs into ONE shared convention, (b) add typestate, (c) retrofit the ~24 estimators that don't have a builder yet.

| Aspect | Disposition | Detail |
|---|---|---|
| Where it lives | **mlrs-algos** (NOT mlrs-core) | The builder is an estimator-construction concern; `mlrs-core` holds only float/oracle/error infra and must stay estimator-agnostic. Put the convention next to `traits.rs` in `mlrs-algos` — a `builder` module (a `derive` macro or a small declarative `builder!` macro + a shared `BuildError`, which already exists). It does NOT belong in `mlrs-backend` (no estimators there) or `mlrs-core`. |
| Typestate (Unfit→Fitted) | **NEW** | Two designs are viable. **(A) Type-level typestate:** `Estimator<F, Unfit>` → `fit` returns `Estimator<F, Fitted>`; predict/transform only impl'd on `Fitted`. Clean but multiplies the generic surface and complicates the PyO3 enum. **(B) Runtime fitted-flag (recommended for v3):** keep the single `Estimator<F>` struct, `build()` produces it Unfit, fitted attributes are `Option<DeviceArray>` (already the shipped pattern — see `gaussian_nb.rs` `theta_: None`), accessors/predict return `AlgoError::NotFitted` if `None`. **Recommendation: runtime fitted-flag** — it matches the shipped device-resident-`Option` state pattern, keeps the `any_estimator!` enum unchanged, and the *typestate guarantee surfaces in the Python shim* via `check_is_fitted`/`NotFittedError` (already wired in `MlrsBase`). Type-level typestate is gold-plating that fights the PyO3 `Unfit/F32/F64` enum. |
| Coexistence with `any_estimator!` | **REUSE — no conflict** | The PyO3 `any_estimator!` enum already has an `Unfit { hyperparameters }` arm. The Rust builder produces the *Rust* `Estimator<F>` that lands in the `F32(..)`/`F64(..)` arms after fit; the PyO3 `#[new]` keeps storing sklearn-named hyperparameters into the enum's `Unfit` arm. **The builder is the Rust-native front door; the PyO3 enum is the Python front door; both construct the same `Estimator<F>`.** No machinery change to dispatch.rs. |
| Coexistence with sklearn-mirror ctors | **MODIFIED (keep both)** | Keep `new`/`with_opts` as thin wrappers over `builder().…build().unwrap()` for backward compat, OR deprecate them. **Recommendation: builder is canonical; `new` becomes `builder().build().expect("defaults valid")` for the zero-required-arg estimators** so existing call sites and tests keep compiling. |
| Sequencing | **CONVENTION FIRST, retrofit as a sweep** | Establish the shared builder convention + typestate decision in an EARLY v3 phase so the NEW v3 estimators (UMAP/HDBSCAN) are *born with it*. Retrofit the existing 30 as a dedicated sweep phase. See build order. |

### Feature 5 — Pure-Python sklearn shim

**Critical finding: the shim ALREADY EXISTS and is shipped** (`crates/mlrs-py/python/mlrs/{base,linear,cluster,decomposition,neighbors,covariance,random_projection}.py`). `MlrsBase` subclasses sklearn `BaseEstimator` directly, giving `get_params`/`set_params`/`clone`/`__repr__` for free, plus `_normalize`/`_to_output` IO routing, `_check_fitted` (→ `NotFittedError`), and a `__sklearn_tags__` override. The PROJECT.md "shim not built / carried-forward" note refers specifically to **`check_estimator` live-FFI re-triage** (needs a maturin+pyarrow host this env lacks — deferred), NOT the shim package.

| Aspect | Disposition | Detail |
|---|---|---|
| Where it sits | **REUSE** | Above the PyO3 `_mlrs` extension (`from . import _io`; estimators call `_ext().PyX(...)`). It sits in `mlrs-py/python/mlrs/`, packaged INTO each of the four per-backend wheels (the shim is pure-Python and backend-agnostic; the `_mlrs` cdylib differs per wheel). |
| v3 additions | **NEW (assembly only)** | `python/mlrs/manifold.py` (UMAP — `TransformerMixin`, `fit_transform` primary, `embedding_`) and add HDBSCAN to `cluster.py` (`ClusterMixin`, `labels_`, `-1` noise). UMAP and HDBSCAN get PyO3 `#[pyclass]` wrappers (`estimators/manifold.rs`, extend `cluster.rs`) registered in `lib.rs`. No new shim machinery — `MlrsBase` covers dtype-suffix routing + output mirroring already. |
| `check_estimator` / get/set_params | **REUSE + extend coverage** | `get_params`/`set_params` come free from `BaseEstimator` given the `__init__`-purity rule (already enforced). Extend the existing `test_params.py`/`test_estimator_checks.py` to the new estimators. Live `check_estimator` triage stays deferred (no maturin+pyarrow host). |

---

## Recommended Project Structure (v3 deltas)

```
crates/
├── mlrs-kernels/src/
│   ├── knn_graph.rs        # NEW: symmetrize map kernel
│   ├── umap_layout.rs      # NEW: umap_layout_step (attract/repel GATHER)
│   ├── mutual_reach.rs     # NEW: mutual_reach_map (small elementwise)
│   └── elementwise.rs      # MODIFY: + umap_fuzzy_map
├── mlrs-backend/src/prims/
│   ├── knn_graph.rs        # NEW: distance→topk→symmetrize (the shared prim)
│   ├── umap_layout.rs      # NEW: host epoch/neg-sample loop + layout kernel
│   ├── mutual_reach.rs     # NEW: core-dist + mutual-reach (device front-end)
│   ├── mst.rs              # NEW: HOST Prim's (no kernel — pointer-chasing)
│   └── condense.rs         # NEW: HOST condensed-tree + stability (no kernel)
├── mlrs-algos/src/
│   ├── builder.rs          # NEW: shared builder convention + typestate
│   ├── manifold/umap.rs    # NEW: UMAP estimator
│   ├── cluster/hdbscan.rs  # NEW: HDBSCAN estimator
│   └── {all 30 estimators} # MODIFY (retrofit sweep): builder + fitted-flag
└── mlrs-py/
    ├── src/estimators/manifold.rs  # NEW: PyUMAP
    ├── src/estimators/cluster.rs   # MODIFY: + PyHDBSCAN
    ├── src/lib.rs                  # MODIFY: register PyUMAP/PyHDBSCAN
    └── python/mlrs/
        ├── manifold.py             # NEW: UMAP shim (TransformerMixin)
        └── cluster.py              # MODIFY: + HDBSCAN (ClusterMixin)
```

### Structure Rationale

- **`mst.rs`/`condense.rs` live in `mlrs-backend/src/prims/` but contain NO CubeCL** — they are host-side algorithm primitives. Placing them under `prims/` keeps the "estimators compose prims, never implement algorithm internals" discipline (D-13) intact: HDBSCAN's `cluster/hdbscan.rs` composes `mutual_reach` (device) + `mst` + `condense` (host) prims, it does not inline a Prim's loop.
- **`builder.rs` in mlrs-algos, not mlrs-core** — keeps `mlrs-core` estimator-agnostic; the builder is an estimator-construction concern co-located with `traits.rs`.

---

## Architectural Patterns

### Pattern 1: Device front-end, host tree back-end (HDBSCAN)

**What:** Run the embarrassingly-parallel, GATHER-friendly stages (distance, topk, mutual-reach) on device; read back once; run the inherently-sequential tree stages (MST, condense, stability) on the host in plain Rust.
**When to use:** Any algorithm whose back half is pointer-chasing / recursive with no data-parallel structure, under the cpu-MLIR no-atomics constraint.
**Trade-offs:** One device→host read-back (cheap at v3 sizes); avoids the infeasible GPU-MST atomics problem entirely; the host code is simple, testable Rust. The cost is it won't scale to millions of points without a GPU MST (deferred, same posture as cuML's raft MST which mlrs deliberately does not port in v3).

### Pattern 2: Reuse the topk/distance prims as the KNN graph (no new neighbor prim)

**What:** The KNN graph IS `distance → top_k` (already shipped + cpu-MLIR-proven) plus a symmetrize map. Do not build a new neighbor search.
**When to use:** Both UMAP and HDBSCAN front-ends.
**Trade-offs:** Dense `n×n` distance at v3 oracle sizes is fine (same as the kernel-matrix prim); approximate-NN (NN-Descent, like umap-learn's default) is deferred — flag the exact-vs-approximate divergence in PITFALLS since umap-learn uses approximate KNN.

### Pattern 3: Builder convention generalizing the Phase-10 precedent

**What:** Promote the per-estimator `*Builder` structs (shipped for the 9 Phase-10/11 estimators) into ONE shared convention — a `derive`/macro that emits `builder()` + setters + `build() -> Result<_, BuildError>`, reusing the existing `BuildError` enum.
**When to use:** All estimators; new v3 estimators adopt it from birth.
**Trade-offs:** A retrofit sweep touches ~24 existing files (the one broad-edit work item in v3 — schedule it as its own phase, not interleaved with algorithm phases, to keep waves feature-disjoint). Runtime fitted-flag (not type-level typestate) keeps the PyO3 enum unchanged.

**Example (the shipped builder shape to generalize):**
```rust
// Already shipped in gaussian_nb.rs — generalize this into builder.rs:
GaussianNB::builder().priors(p).var_smoothing(1e-9).build::<f32>()?  // -> GaussianNB<f32>, Unfit
```

### Pattern 4: Host orchestration + GATHER device step (UMAP layout, reusing the sgd.rs shape)

**What:** The host owns the epoch loop, LR schedule, and edge/negative-index sampling (SplitMix64); the device runs a single-owner per-row update kernel. Same *orchestration shape* as `prims/sgd.rs`/`prims/lbfgs.rs`, but a NEW layout kernel (not the SGD weight-update kernel).
**When to use:** UMAP layout optimization.
**Trade-offs:** The per-embedding-row single-owner update is the cpu-MLIR feasibility unknown to spike. Precedent is favorable (the v2 SGD two-pass GATHER solver launched on cpu-MLIR first try).

## Data Flow

### KNN-graph → UMAP
```
X →[device]→ distance → topk(k+1) →[host]→ self-drop, σ/ρ binary-search
  →[device]→ fuzzy_map + symmetrize(t-conorm) → init(rng default | eig≤MAX_DIM)
  →[host loop + device step]→ umap_layout_step × epochs → embedding_
```

### KNN-graph → HDBSCAN
```
X →[device]→ distance → topk → core_dist(k-th col) → mutual_reach_map
  →[READ BACK]→[host]→ MST(Prim's) → single-linkage → condense → stability/EOM → labels_
```

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| v3 oracle sizes (≤ few thousand pts) | Dense `n×n` distance + host MST + dense symmetrize all fine; eig-spectral-init only under `MAX_DIM`=64 cap. |
| Larger (10k+) | Switch KNN symmetrize to COO/CSR (host); UMAP must use `init="random"` (eig caps out); host MST still OK to ~100k. |
| Much larger (deferred) | Approximate-NN (NN-Descent), sparse-eig (Lanczos), GPU-MST (atomics — blocked by cpu-MLIR) — all deferred past v3, same posture as v2's Nyström/Lanczos deferrals. |

### Scaling Priorities
1. **First bottleneck:** dense `n×n` distance/symmetrize memory → COO/CSR on host.
2. **Second bottleneck:** spectral-init eig `MAX_DIM` cap → default to random init.

## Anti-Patterns

### Anti-Pattern 1: Attempting on-device MST / GPU tree construction in v3
**What people do:** Try a GPU Boruvka MST or device-side condensed-tree build for HDBSCAN.
**Why it's wrong:** Both need cross-unit atomics / dynamic parallelism that the cpu-MLIR backend cannot lower (the exact reason RandomForest/tree work is deferred past v3 per PROJECT.md Out-of-Scope). It will panic on the cpu gate.
**Do this instead:** Read the mutual-reach graph back once; run Prim's + condensation in plain host Rust (Pattern 1).

### Anti-Pattern 2: Reusing the v2 SGD weight-update kernel for UMAP layout
**What people do:** Try to force UMAP's attract/repel embedding optimization through `prims/sgd.rs`.
**Why it's wrong:** `sgd.rs` updates a fixed-length coef vector from per-sample margins (supervised linear model); UMAP updates an `n×d_embed` coordinate matrix from sampled edges + negative samples — a different gradient and data layout.
**Do this instead:** New `prims/umap_layout.rs` + `umap_layout_step` kernel; reuse only the *host orchestration shape* and `rng.rs` for sampling.

### Anti-Pattern 3: Interleaving the builder retrofit with algorithm phases
**What people do:** Retrofit builders to existing estimators inside the UMAP/HDBSCAN phases.
**Why it's wrong:** Breaks the file-disjoint, parallel-safe wave discipline — the retrofit touches ~24 existing estimator files; mixing it with new-estimator phases creates merge contention.
**Do this instead:** Establish the convention in an early phase; do the retrofit as its own dedicated sweep phase.

### Anti-Pattern 4: Building a new KNN/neighbor primitive
**What people do:** Write a fresh k-NN search for UMAP/HDBSCAN.
**Why it's wrong:** `distance` + `topk` already do this, are cpu-MLIR-proven, and are the shipped NearestNeighbors prim.
**Do this instead:** `prims/knn_graph.rs` composes them + a symmetrize map.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| knn_graph prim ↔ UMAP / HDBSCAN | `(indices i32, distances F)` neighbor-list + shared symmetrize helper | Each estimator applies its own edge weighting (fuzzy-union vs mutual-reach). |
| mutual_reach (device) ↔ mst/condense (host) | one device→host read-back of the mutual-reach graph | The HDBSCAN device/host seam. |
| Rust builder ↔ `any_estimator!` Unfit arm | both construct the same `Estimator<F>` | No dispatch.rs change. |
| `mlrs-py` shim (Python) ↔ `_mlrs` cdylib | `_ext().PyX(...)`, dtype-suffixed accessors | Shim is pure-Python, packaged into all four wheels. |

## Suggested phase / build order (continuing from Phase 12)

Primitive-first, dependencies respected (KNN-graph before UMAP/HDBSCAN; builder convention before retrofit), feature-disjoint waves.

```
Phase 12  Builder convention + typestate  [DO FIRST — born-with-it for new estimators]
   NEW: mlrs-algos/src/builder.rs (shared builder; runtime fitted-flag; reuse BuildError)
   establishes the convention so UMAP/HDBSCAN (P14/P15) adopt it from birth
   NO algorithm work — pure API foundation; lowest risk; unblocks everything
   (does NOT retrofit existing estimators yet — that is P16)

Phase 13  KNN-graph primitive  [the shared, feasibility-critical prim]
   NEW: prims/knn_graph.rs (distance + topk + symmetrize map) + knn_graph kernel
   reuse: distance, topk (cpu-MLIR-proven)
   gate: tests/knn_graph_test.rs (sklearn kneighbors_graph oracle + PoolStats)
   feature-free, standalone-validated — NOTHING consumes it yet (primitive-first)

Phase 14  UMAP   [HARD DEP on P12 builder + P13 knn_graph]
   NEW: prims/umap_layout.rs + umap_layout_step kernel (the one new solver)
   NEW: umap_fuzzy_map kernel; reuse rng (init+neg-sample), eig/laplacian (spectral init ≤MAX_DIM)
   NEW: manifold/umap.rs estimator (builder-fronted), estimators/manifold.rs PyUMAP, manifold.py shim
   SPIKE FIRST: layout-step single-owner GATHER on cpu-MLIR (MEDIUM-confidence unknown)
   gate: property/structural vs umap-learn (trustworthiness, kNN-preservation, seed-repro)

Phase 15  HDBSCAN   [HARD DEP on P12 builder + P13 knn_graph; feature-disjoint from P14]
   NEW: prims/mutual_reach.rs + mutual_reach_map kernel (device front-end)
   NEW: prims/mst.rs (HOST Prim's), prims/condense.rs (HOST condensed-tree + stability)
   NEW: cluster/hdbscan.rs estimator (builder-fronted), PyHDBSCAN in cluster.rs, cluster.py shim
   SPIKE FIRST: confirm host MST/condense matches hdbscan condensation exactly
   gate: exact labels up to permutation vs hdbscan/sklearn.cluster.HDBSCAN (label_perm helper)

Phase 16  Builder retrofit sweep + shim coverage  [DO LAST — broad-edit, parallel-unsafe]
   MODIFY: retrofit the ~24 pre-Phase-10 estimators to the P12 builder convention
   MODIFY: keep new()/with_opts as thin builder wrappers for back-compat
   EXTEND: test_params.py / test_estimator_checks.py to UMAP/HDBSCAN + retrofitted set
   isolate as its own phase so it never contends with new-estimator files
```

**Critical ordering facts:**
- **P12 (builder convention) FIRST** so UMAP/HDBSCAN are born builder-fronted — avoids re-touching them in the retrofit. P12 is pure API, no algorithm, lowest risk.
- **P13 (knn_graph) before P14/P15** — both consume it; land + gate it standalone (primitive-first; the v1/v2 "prim gated, estimator assembly" pattern).
- **P14 (UMAP) and P15 (HDBSCAN) are feature-disjoint waves** — different new files (manifold/ vs cluster/, umap_layout/fuzzy vs mutual_reach/mst/condense). They share ONLY the P13 knn_graph prim (already landed) → can be planned/built in parallel after P13.
- **P16 (retrofit) LAST** — it is the one broad, parallel-unsafe edit; isolating it preserves the file-disjoint wave discipline for P12–P15.
- **Two research spikes gate planning:** UMAP layout-step on cpu-MLIR (P14) and host MST/condense exactness (P15). Both MEDIUM-confidence; budget a spike before each phase's planning, exactly as v2 did for the SGD solver.

## Sources

- Shipped code (HIGH): `crates/mlrs-algos/src/{traits.rs, lib.rs, error.rs}`,
  `crates/mlrs-algos/src/{naive_bayes/{mod,gaussian_nb}.rs, linear/{mod,logistic}.rs, cluster/spectral_embedding.rs}`,
  `crates/mlrs-backend/src/{lib.rs, prims/{topk,sgd,laplacian,eig,distance}.rs}`,
  `crates/mlrs-kernels/src/lib.rs` (incl. `jacobi_eig::MAX_DIM`),
  `crates/mlrs-py/src/{dispatch.rs, lib.rs}`, `crates/mlrs-py/python/mlrs/base.py`,
  the `crates/mlrs-py/python/mlrs/*.py` shim package (confirms shim already shipped).
- Planning (HIGH): `.planning/PROJECT.md`, `.planning/notes/v3-hard-algorithm-backlog.md`,
  `.planning/codebase/{ARCHITECTURE,STRUCTURE}.md`, `.planning/milestones/v2.0-research/ARCHITECTURE.md`.
- cuML reference (read-only, behavior only): `cuml-main/cpp/src/{umap,hdbscan}/`,
  `cuml-main/python/cuml/cuml/{manifold,cluster}/`.
- MEDIUM-confidence unknowns to spike before P14/P15: UMAP `umap_layout_step` single-owner
  GATHER on cpu-MLIR; host MST/condense exactness vs `hdbscan`; KNN approximate-vs-exact
  divergence from umap-learn's NN-Descent default.

---
*Architecture research for: mlrs v3.0 manifold algorithms & Rust-native builder API*
*Researched: 2026-06-22*
