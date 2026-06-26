# Architecture Research — mlrs v4.0 (Tree Ensembles, Time-Series & Full-Surface Completion)

**Domain:** Integration of the final cuML algorithm surface (RandomForest→FIL→TreeSHAP, ARIMA, Kernel/Permutation SHAP, cuml.accel, sklearn-utility surface, genetic/symbolic regression) into the existing 5-crate CubeCL/PyO3 workspace.
**Researched:** 2026-06-26
**Confidence:** HIGH for crate/layer placement & build order (grounded in the shipped v1–v3 architecture read in-source); MEDIUM-HIGH for the GPU-tree feasibility framing (analogous to the proven Phase-13 spike pattern, but the histogram/split kernel is genuinely unproven under cpu-MLIR — that is exactly what the spike must answer).

> **Scope note.** This document answers *how the v4.0 features attach to the architecture that already exists* and *what new primitives/components each needs*. It does NOT re-derive the existing workspace (mlrs-core / -kernels / -backend / -algos / -py), the builder/typestate convention, the Arrow bridge, the oracle harness, the four per-backend wheels, or the cpu-MLIR GATHER discipline — all shipped and validated in v1–v3. It integrates **with** them.

---

## Standard Architecture

### The existing 5-crate spine (unchanged; everything v4.0 attaches here)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  mlrs-py        PyO3 #[pyclass] estimators (src/estimators/*.rs)            │
│                 + pure-Python sklearn shim (python/mlrs/*.py)               │
│                 + NEW: python/mlrs/accel/  (pure-Python import-hook proxy)  │
├──────────────────────────────────────────────────────────────────────────┤
│  mlrs-algos     sklearn-compatible estimators assembled from prims         │
│                 (linear/ cluster/ decomposition/ … + NEW ensemble/ tree/   │
│                  fil/ tsa/ explainer/ genetic/)  — builder/typestate        │
├──────────────────────────────────────────────────────────────────────────┤
│  mlrs-backend   prim host-orchestrators over BufferPool/PoolStats          │
│                 (distance/ topk/ svd/ … + NEW tree_hist/ best_split/        │
│                  node_partition/ tree_traverse/ batched_kalman/ program_eval)│
├──────────────────────────────────────────────────────────────────────────┤
│  mlrs-kernels   feature-free #[cube(launch)] kernels, generic over F+runtime│
│                 (distance/ topk/ reduce/ … + NEW tree.rs/ kalman.rs/        │
│                  program.rs)  — cpu-MLIR-safe GATHER idiom ONLY             │
├──────────────────────────────────────────────────────────────────────────┤
│  mlrs-core      device arrays, BufferPool, Arrow bridge, ORACLE HARNESS     │
│                 (compare/ label_perm/ sign_flip/ tolerance + NEW tree/SHAP  │
│                  comparison helpers)                                        │
└──────────────────────────────────────────────────────────────────────────┘
        ▲ ActiveRuntime feature-selected once: cuda(compile) / rocm(f32) / wgpu / cpu(f64)
```

The architectural rule that dominates v4.0: **a compute primitive is validated standalone (feature-free, GATHER idiom, cpu-MLIR-safe, oracle-gated) in mlrs-kernels+mlrs-backend BEFORE any mlrs-algos estimator consumes it.** This is the primitive-first discipline that made v1/v2/v3 estimators "mostly assembly." v4.0's hard new primitives all live in the tree family and ARIMA.

### Component Responsibilities (v4.0 additions)

| Component | Responsibility | Crate / lands-in | NEW or MODIFIED |
|-----------|----------------|------------------|-----------------|
| Tree histogram prim | Per-node × per-feature × per-bin gradient/count histograms via single-owner GATHER (no atomics, no SharedMemory) | mlrs-kernels `tree.rs` + mlrs-backend `prims/tree_hist.rs` | **NEW (spike-gated)** |
| Best-split reduction prim | Per-node argmax over (feature, bin) of the split-gain objective (gini/entropy/mse) | mlrs-kernels `tree.rs` + mlrs-backend `prims/best_split.rs` | **NEW (spike-gated)** |
| Node-partition prim | Stable partition of a node's sample-index range into left/right child ranges under the chosen split | mlrs-kernels `tree.rs` + mlrs-backend `prims/node_partition.rs` | **NEW (spike-gated)** |
| Quantile/bin-edge prim | Per-feature bin boundaries (quantile sketch) computed once before tree build | mlrs-backend `prims/quantiles.rs` (reuses `topk`/`reduce`/sort) | **NEW** |
| Tree node store + traversal (FIL) | Flat `SparseTreeNode{colid, threshold, left_child, value}` array; batched root→leaf traversal kernel | mlrs-kernels `tree.rs::traverse` + mlrs-backend `prims/tree_traverse.rs` | **NEW (depends on tree store)** |
| DecisionTree core | Level-wise (breadth-first) builder host loop driving the three tree prims | mlrs-algos `tree/` | **NEW (spike-gated)** |
| RandomForest estimator | Bagging/feature-subsampling over N decision trees; majority/mean aggregate | mlrs-algos `ensemble/` | **NEW (spike-gated)** |
| Batched Kalman prim | Sequential state-space recursion (one unit per series, sequential time loop), returns log-likelihood + residuals | mlrs-kernels `kalman.rs` + mlrs-backend `prims/batched_kalman.rs` | **NEW** |
| Batched L-BFGS | Many independent small optimizations (one per series/order) sharing the v1 L-BFGS prim | mlrs-backend `prims/lbfgs.rs` (MODIFY: batched wrapper) | **MODIFIED** |
| ARIMA / AutoARIMA estimator | Kalman-likelihood objective under batched L-BFGS + (Auto) order search host loop | mlrs-algos `tsa/` | **NEW** |
| Kernel/Permutation SHAP | Coalition sampling (host) → many `predict` calls on an existing fitted estimator → weighted linear solve (reuse) | mlrs-algos `explainer/` (+ thin Python) | **NEW (model-agnostic; no new device prim)** |
| TreeSHAP | Path-dependent expected-value traversal over the tree node store | mlrs-algos `explainer/tree_shap` | **NEW (depends on tree store/FIL)** |
| Program-eval prim (genetic) | Batch-evaluate a population of stack-programs over the dataset (device fitness) | mlrs-kernels `program.rs` + mlrs-backend `prims/program_eval.rs` | **NEW** |
| Symbolic regression estimator | Host-driven evolutionary loop (selection/crossover/mutation on expression trees), device-evaluated fitness | mlrs-algos `genetic/` | **NEW** |
| sklearn-utility surface | metrics / preprocessing / feature_extraction / model_selection | mostly **Python** `python/mlrs/`; light-device pieces reuse `reduce`/`distance` prims | **NEW (mostly host)** |
| cuml.accel drop-in | sys.modules import-hook proxy mapping sklearn/umap/hdbscan classes → existing mlrs estimators with CPU fallback | **pure-Python** `python/mlrs/accel/` | **NEW (zero Rust/PyO3)** |
| Oracle harness extensions | Tree-structure / forest-prediction comparison; SHAP-value tolerance; gplearn/shap/statsmodels oracles | mlrs-core `oracle.rs` + `scripts/gen_oracle.py` | **MODIFIED** |

---

## The spike's central question — GPU tree construction under cpu-MLIR

This is the make-or-break feasibility question and the gating first phase (model it on Phase 13's KNN-graph keystone). The spike must produce **VALIDATED verbatim kernel shapes** (à la spike 001/002) captured into a `spike-findings-mlrs`-style skill, and either GREEN-light the tree family or trigger scope adjustment.

### Why cuML's approach does not port

cuML's `batched-levelalgo` builder (`cuml-main/cpp/src/decisiontree/batched-levelalgo/`) is the canonical GPU decision tree. Its histogram kernel (`builder_kernels_impl.cuh`) uses **`__shared__` memory + `atomicAdd`**: each thread reads one sample, computes its bin, and **scatter-adds** into a shared per-(node,feature,bin) histogram. cpu-MLIR (`cubecl-cpu` 0.10) **bans both `SharedMemory` and cross-unit `Atomic`** (project memory: panics at launch). The entire histogram strategy must be inverted.

### The GATHER inversion the spike must prove

The proven mlrs idiom (Phase 13 `self_drop_gather`, all v2 prims) is **single-owner GATHER**: assign exactly one unit to each *output* slot; that unit *loops over inputs and accumulates locally*, so there is never a contended write. Applied to tree histograms:

- **Scatter (cuML, banned):** one unit per *sample* → `atomicAdd(hist[bin(sample)])`. Contended writes → needs atomics.
- **Gather (mlrs, the hypothesis to prove):** one unit per *(node, feature, bin)* output cell → loop over the node's sample-index range, count/sum the samples whose feature value falls in this bin. Each output cell is written by exactly one owner — **no atomics, no shared memory**. This is the `CUBE_POS_X`/`UNIT_POS_X==0`-style per-output-row shape that already lowers under cpu-MLIR.

The spike must concretely answer, each with a launched-under-`--features cpu` VALUE-asserting probe (not compile-only):

1. **Histogram (the crux).** Does the single-owner gather histogram — one unit per (node,feature,bin), inner `while` over the node's sample range, `if value < edge[b] && value >= edge[b-1] { acc += 1 / acc_grad += g }` — lower under cpu-MLIR and match a host reference on a duplicate-/tie-heavy fixture? Confirm both classification (count + per-class counts) and regression (sum + sum-of-squares / sum-of-gradients) accumulators. **Open risk:** the per-cell scan is O(bins × samples) work where cuML's scatter is O(samples); the spike must also confirm this is *tractable* (tiled over node-batch), not just correct.
2. **Best-split reduction.** Does a per-node argmax over (feature × bin) of the split-gain (gini/entropy for classification, MSE/variance-reduction for regression) — statement-form running-max with a `u32` best-index accumulator, read within the same outer iteration (the 002-B landmine: NO cross-sibling-loop accumulator) — lower and select the lowest-index tie correctly?
3. **Node partition.** Does a stable left/right partition of a node's sample-index slice (single-owner: one unit per node computes the two contiguous child ranges by a counted two-pass — count-left, then place — never a cross-loop carry) lower under cpu-MLIR? This is the "scan/compaction" shape; the spike must find the GATHER-safe form (likely per-node sequential placement, like `self_drop_gather`).
4. **Level-wise host loop.** Does the breadth-first driver (process a *batch* of frontier nodes per level, launch hist→split→partition, append children, recurse) compose without ever materializing a per-sample×per-node structure that blows the PoolStats memory gate? (Quantile bin-edges are precomputed once; histograms are node-batch × bins × features × outputs, tiled.)
5. **Node storage format.** Confirm the flat node array works: `SparseTreeNode { colid: u32, threshold: F, left_child: i32 (-1 = leaf), value: F or class-logits }`, right child = `left_child + 1` (cuML `flatnode.h` convention). One contiguous `DeviceArray` per tree; the forest is a concatenation + per-tree offset table. This format is the **contract between RF (writer), FIL (batched reader), and TreeSHAP (path reader)** — fixing it is part of the spike.

**If the spike fails** (histogram/split cannot be made both correct AND tractable under cpu-MLIR): the documented fallbacks, in preference order, are (a) **CPU-side host tree build** (build trees on the host in plain Rust, keep only FIL inference on device — sacrifices "single generic codebase" for trees but preserves the inference path and the surface); (b) restrict to small-data exact trees; (c) drop the tree family from v4.0 and re-scope (PROJECT.md explicitly allows "scope adjusts before committing"). The spike's deliverable is the GREEN/RED decision plus, if GREEN, the verbatim kernel shapes.

---

## Recommended Project Structure (v4.0 additions only)

```
crates/mlrs-kernels/src/
├── tree.rs              # NEW: gather-histogram, split-gain-reduce, node-partition,
│                        #      batched root→leaf traverse (FIL)  — cpu-MLIR-safe
├── kalman.rs            # NEW: per-series sequential state-space recursion (one unit/series)
├── program.rs           # NEW: stack-program batch evaluator (genetic fitness)
└── lib.rs               # MODIFY: pub mod + pub use of new kernel symbols

crates/mlrs-backend/src/prims/
├── quantiles.rs         # NEW: per-feature bin edges (sort/reduce reuse) — pre-pass for trees
├── tree_hist.rs         # NEW: histogram prim (host launch wrapper, node-batch tiled)
├── best_split.rs        # NEW: best-split reduction prim
├── node_partition.rs    # NEW: partition prim
├── tree_traverse.rs     # NEW: FIL batched-inference prim over the node store
├── batched_kalman.rs    # NEW: ARIMA likelihood/residual prim
├── program_eval.rs      # NEW: genetic fitness prim
├── lbfgs.rs             # MODIFY: add a batched wrapper (many small independent solves)
└── mod.rs               # MODIFY: pub mod the new prims

crates/mlrs-algos/src/
├── tree/                # NEW: DecisionTree core (level-wise builder host loop) + node store type
├── ensemble/            # NEW: RandomForestClassifier + RandomForestRegressor (bagging)
├── fil/                 # NEW: forest inference (predict/predict_proba over node store)
├── tsa/                 # NEW: ARIMA + AutoARIMA (Kalman objective + batched L-BFGS + order search)
├── explainer/           # NEW: kernel_shap, permutation_shap, tree_shap
├── genetic/             # NEW: SymbolicRegressor (host evolutionary loop, device fitness)
└── lib.rs               # MODIFY: register new modules

crates/mlrs-py/src/estimators/
├── ensemble.rs          # NEW: #[pyclass] RandomForest*  (f32/f64 dispatch, GIL release)
├── tsa.rs               # NEW: #[pyclass] ARIMA / AutoARIMA
├── explainer.rs         # NEW: #[pyclass] KernelExplainer / PermutationExplainer / TreeExplainer
├── genetic.rs           # NEW: #[pyclass] SymbolicRegressor
└── mod.rs               # MODIFY: register new estimator arms in any_estimator!/dispatch

crates/mlrs-py/python/mlrs/
├── ensemble.py          # NEW shim          ├── tsa.py            # NEW shim
├── fil.py               # NEW shim          ├── explainer.py      # NEW shim
├── genetic.py           # NEW shim
├── metrics.py           # NEW (mostly host) ├── preprocessing.py  # NEW (light-device)
├── feature_extraction.py# NEW (mostly host) ├── model_selection.py # NEW (pure host)
├── accel/               # NEW pure-Python subpackage (import-hook proxy)
│   ├── __init__.py      #   install()/uninstall() — sys.modules swap
│   ├── _hook.py         #   MetaPathFinder / module proxy machinery
│   └── _overrides.py    #   { "sklearn.ensemble.RandomForestClassifier": mlrs.RandomForestClassifier, … }
└── __init__.py          # MODIFY: re-export the new estimators + utility namespaces

crates/mlrs-core/src/
└── oracle.rs            # MODIFY: tree/forest-prediction & SHAP-value comparison helpers
scripts/gen_oracle.py    # MODIFY: sklearn RF, statsmodels/sklearn ARIMA, shap, gplearn fixtures
```

### Structure Rationale

- **All hard compute lands in mlrs-kernels + mlrs-backend first** (tree.rs, kalman.rs, program.rs and their prim wrappers) so it is standalone-validated before any estimator imports it. The three tree prims (hist/split/partition) are the spike's deliverables promoted to production prims.
- **mlrs-algos mirrors cuML's module names** (`ensemble/`, `fil/`, `tsa/`, `explainer/`, `genetic/`) — consistent with the existing `cluster/`, `manifold/`, etc., and with the codebase map's "where to add new code."
- **The tree node store is a first-class type in `mlrs-algos/src/tree/`**, owned by the tree builder, *read* by `fil/` and `explainer/tree_shap`. This is the single integration contract for the RF→FIL→TreeSHAP chain.
- **cuml.accel is pure Python, parallel to the family shims** — it imports the *already-sklearn-subclassing* mlrs estimators (every shim extends `MlrsBase(BaseEstimator)`) and swaps them into `sys.modules`. It needs **zero** Rust/PyO3/kernel work; it is a Python-only subsystem layered over the surface that already exists. CPU fallback = leave the real sklearn/umap/hdbscan class in place when mlrs has no equivalent or the config is unsupported.
- **The utility surface is mostly host Python.** metrics (accuracy/r2/pairwise) and model_selection (train_test_split/cross_val_score/KFold) are numpy/host; preprocessing scalers (StandardScaler/MinMax) are the only ones wanting a light-device pass and they reuse the existing `reduce` prim — almost no new kernels.

---

## Architectural Patterns

### Pattern 1: Single-owner GATHER histogram (the v4.0 keystone kernel)

**What:** Invert cuML's atomic scatter into a per-output-cell gather. One unit owns one (node, feature, bin) cell and loops over the node's samples, accumulating locally. No shared memory, no atomics — the only cpu-MLIR-safe way to build histograms.
**When to use:** All tree histogram construction (classification counts + per-class, regression sum/sum-sq or gradient/hessian).
**Trade-offs:** Correctness & cpu-MLIR-safety vs O(bins × samples) redundant reads (cuML's scatter is O(samples)). Mitigate by node-batch tiling and precomputed integer bin indices. **This is the spike's tractability risk.**

```rust
// SHAPE TO PROVE in the spike (cpu-MLIR-safe; NOT yet validated):
//   one unit per (node, feature, bin) output cell; inner while over node's samples.
#[cube(launch)]
pub fn gather_hist<F: Float + CubeElement>(
    binned: &Array<u32>,        // per-(sample,feature) precomputed bin index
    sample_idx: &Array<u32>,    // node's sample-index slice (gather, not contiguous)
    node_start: &Array<u32>, node_len: &Array<u32>,
    out_count: &mut Array<u32>, // [node, feature, bin]
    n_feat: u32, n_bins: u32, /* … */
) {
    let cell = ABSOLUTE_POS_X;                 // one owner per output cell
    // decode (node, feat, bin) from cell; bounds-guard
    let mut acc = 0u32;
    let mut s = 0u32;
    while s < len {                            // loop over THIS node's samples
        let smp = sample_idx[(start + s) as usize];
        if binned[(smp * n_feat + feat) as usize] == bin { acc += 1u32; }
        s += 1u32;
    }
    out_count[cell as usize] = acc;            // single writer — no atomic
}
```

### Pattern 2: Per-series sequential recursion (ARIMA Kalman)

**What:** State-space / Kalman recursion is inherently sequential in time but embarrassingly parallel across series. Assign one unit per series; loop the time axis sequentially inside that unit (the `self_drop_gather` per-row shape). Returns log-likelihood + standardized residuals to the host.
**When to use:** ARIMA likelihood evaluation under the optimizer.
**Trade-offs:** Parallelism = number of series (fine for AutoARIMA's many-candidate search; a single long series is under-parallel — acceptable, matches cuML's batched design). Host owns the L-BFGS outer loop; device owns the likelihood eval.

```rust
// one unit per series; sequential time loop — cpu-MLIR-safe (no atomics/shared mem)
let series = CUBE_POS_X;
if series < n_series { if UNIT_POS_X == 0u32 {
    // init state; for t in 0..T { predict; update; accumulate loglik } — F/u32 accumulators only
}}
```

### Pattern 3: Host-orchestrated, device-evaluated (SHAP & genetic)

**What:** The *search/sampling* logic stays on the host (Rust); the device is called only to **batch-evaluate** a model or population. Kernel/Permutation SHAP: host samples coalitions → calls the wrapped estimator's existing `predict` → host weighted-least-squares (reuse the v1 solver). Symbolic regression: host runs selection/crossover/mutation on expression trees → device batch-evaluates the population's fitness over the data (`program_eval` prim).
**When to use:** Any algorithm whose parallelism is "evaluate many candidates," not "one big kernel."
**Trade-offs:** Simple, reuses existing predict/solve paths, no exotic kernels — but host↔device round-trips per generation/coalition-batch. Keep batches large; never round-trip per individual.

### Pattern 4: Batched-small-solve over the existing L-BFGS prim

**What:** AutoARIMA fits many (p,d,q) candidates; each is a small independent optimization. Rather than a new batched-L-BFGS kernel, wrap the existing `lbfgs` prim in a host loop / batched dispatch — many small problems, shared prim. (cuML uses a true `batched_lbfgs`; mlrs can start host-batched and only build a device-batched variant if profiling demands it.)
**When to use:** ARIMA/AutoARIMA, any "many small optimizations" estimator.
**Trade-offs:** Host-batched is simplest and reuses a proven prim; device-batched is a later optimization, not a v4.0 requirement.

### Pattern 5: cuml.accel sys.modules import-hook (pure Python)

**What:** `mlrs.accel.install()` registers a `MetaPathFinder` / swaps `sys.modules` entries so `from sklearn.ensemble import RandomForestClassifier` resolves to the mlrs estimator. An `_overrides` table maps qualified sklearn/umap/hdbscan names → mlrs classes; unmapped names pass through to the real library (CPU fallback). Because every mlrs shim already subclasses `sklearn.base.BaseEstimator` (`MlrsBase`), the proxied objects satisfy `get_params`/`set_params`/`clone`/pipeline usage transparently.
**When to use:** Drop-in acceleration of existing sklearn scripts without code change.
**Trade-offs:** Zero compute work — it is *pure plumbing over the estimator surface that already exists*. The risk is behavioral parity (params mlrs doesn't support must fall back, not silently differ). Belongs LAST so it can proxy the full v4.0 surface.

---

## Data Flow

### RF → FIL → TreeSHAP (the gated dependency chain)

```
fit:  X,y ─▶ quantiles prim (bin edges, once)
              ▼
        level-wise host loop  ──▶ gather_hist prim ──▶ best_split prim ──▶ node_partition prim
              ▲  (append children, recurse per frontier-node batch)            │
              └───────────────────────────────────────────────────────────────┘
              ▼
        flat SparseTreeNode store  (per-tree DeviceArray + forest offset table)
              │  ── the single contract ──────────────────────────────────────┐
predict:      ▼                                                                ▼
        FIL tree_traverse prim (batched root→leaf, one unit per (row,tree))   TreeSHAP
              ▼                                                          (path-dependent
        aggregate (vote / mean) ─▶ predict / predict_proba               expected-value walk
                                                                          over the SAME store)
```

### ARIMA

```
y (n series) ─▶ host L-BFGS (per series / per candidate order)
                   │  proposes params θ
                   ▼
             batched_kalman prim (one unit/series, sequential time) ─▶ loglik, residuals
                   ▲────────────────────── gradient/value back to optimizer ──┘
AutoARIMA: stationarity test (host) → enumerate (p,d,q)(P,D,Q) → batched fit → AIC/BIC select
```

### Kernel/Permutation SHAP (model-agnostic)

```
fitted estimator + background data ─▶ host: sample coalitions/permutations
        ─▶ build masked design matrix ─▶ estimator.predict (existing device path, batched)
        ─▶ host weighted least squares (reuse v1 OLS/solver) ─▶ shap values
```

### Symbolic regression

```
host: init population (expression trees) ─▶ program_eval prim (batch fitness over X) 
   ─▶ host: tournament select / crossover / mutate ─▶ next generation ─▶ … ─▶ best program
```

---

## Build Order (honors primitive-first + spike gate + RF→FIL→TreeSHAP)

> Phase numbering continues from v3.0 (last = Phase 16); v4.0 starts at **Phase 17**. The order puts the RF feasibility spike FIRST as a gate, runs the independent (non-tree) tracks in parallel where the workspace allows, and respects the tree dependency chain.

| # | Phase | Depends on | Crate work | Gate role |
|---|-------|-----------|------------|-----------|
| **17** | **RandomForest GPU histogram/split FEASIBILITY SPIKE** (cpu-MLIR, GATHER hist + split + partition + node format) | — | mlrs-kernels probes | **GATING — make-or-break. GREEN/RED before any tree commit.** Models Phase 13. Emits verbatim kernel shapes + node-format contract, captured to a spike-findings skill. |
| 18 | Tree primitives (quantiles, gather-hist, best-split, node-partition) standalone-validated + DecisionTree core | 17 GREEN | mlrs-kernels `tree.rs`, mlrs-backend prims, mlrs-algos `tree/` | Primitive-first: prims oracle-gated vs sklearn `DecisionTree` BEFORE RF. |
| 19 | RandomForestClassifier + RandomForestRegressor (+ PyO3 + shim) | 18 | mlrs-algos `ensemble/`, mlrs-py | sklearn RF oracle (exact-label / prediction gate). |
| 20 | FIL — batched tree traversal over the node store | 18 (node format), 19 (a forest to infer) | mlrs-kernels traverse, mlrs-backend `tree_traverse.rs`, mlrs-algos `fil/` | predict/predict_proba parity. |
| 21 | TreeSHAP | 20 (FIL/tree store) | mlrs-algos `explainer/tree_shap` | `shap` library oracle. |
| 22 | ARIMA / AutoARIMA (batched Kalman prim + batched L-BFGS + order search) | independent of trees | mlrs-kernels `kalman.rs`, mlrs-backend `batched_kalman.rs`+`lbfgs` batched, mlrs-algos `tsa/` | Primitive-first: Kalman prim validated before ARIMA. **Can run parallel to 18–21.** |
| 23 | Kernel + Permutation SHAP (model-agnostic) | needs ≥1 fitted estimator surface (have it) | mlrs-algos `explainer/`, thin Python | `shap` oracle. No new device prim. **Parallel-eligible.** |
| 24 | sklearn-utility surface (metrics / preprocessing / feature_extraction / model_selection) | mostly independent | mostly Python; light `reduce` reuse for scalers | sklearn parity per-function. **Parallel-eligible.** |
| 25 | genetic / symbolic regression (program_eval prim + host evolutionary loop) | independent | mlrs-kernels `program.rs`, mlrs-backend `program_eval.rs`, mlrs-algos `genetic/` | `gplearn` oracle. Primitive-first: fitness prim before the loop. **Parallel-eligible.** |
| 26 | cuml.accel drop-in (pure-Python import-hook) | **the broadest possible estimator surface** | Python-only `python/mlrs/accel/` | LAST so it proxies the full v4.0 + v1–v3 surface; CPU fallback for the rest. |

**Ordering rationale.**
- **17 gates 18–21.** Nothing in the tree chain is committed until the spike answers the cpu-MLIR histogram/split question. This mirrors how Phase 13 gated UMAP/HDBSCAN.
- **18 before 19** (primitive-first): hist/split/partition prims are oracle-validated as standalone prims (vs a single sklearn DecisionTree) before RF assembles many of them.
- **19 before 20 before 21**: FIL needs a forest to traverse and the node-format contract; TreeSHAP needs FIL's tree store. Hard chain.
- **22 (ARIMA), 23 (model-agnostic SHAP), 24 (utility), 25 (genetic) are independent of the tree spike** and of each other — they can be sequenced in parallel waves or interleaved while the tree chain proceeds. Each still obeys primitive-first internally (Kalman prim → ARIMA; program_eval prim → symbolic regression).
- **26 (cuml.accel) is last** because its value is proportional to how much surface exists to proxy; building it after every estimator lands maximizes its override table and lets CPU-fallback handle only the genuinely-missing.

---

## Anti-Patterns

### Anti-Pattern 1: Porting cuML's atomic/shared-memory histogram
**What people do:** Translate `builder_kernels_impl.cuh`'s `atomicAdd` shared-memory scatter histogram directly into CubeCL.
**Why it's wrong:** cpu-MLIR bans `SharedMemory` and cross-unit `Atomic` (panics at launch). It would compile for cuda/wgpu but break the f64 cpu correctness gate.
**Do this instead:** Single-owner GATHER histogram (Pattern 1) — one unit per output cell loops over inputs. This is the spike's whole job.

### Anti-Pattern 2: Cross-sibling-loop accumulator in the split/partition kernels
**What people do:** Write a best-feature flag/index in one `while` loop and read it in a separate sibling loop (e.g. "find best gain, then in a second pass mark the split").
**Why it's wrong:** cpu-MLIR **silently miscompiles** this (FINDING 002-B) — compiles, launches, returns plausible wrong splits. A happy-path test passes; tie-heavy data diverges.
**Do this instead:** Recompute positional values with a self-contained nested accumulate inside the consuming loop (the `self_drop_gather` shape). Gate every tree prim with a duplicate-/tie-heavy fixture asserting VALUES (R-9 discipline).

### Anti-Pattern 3: Bypassing primitive-first for the tree family
**What people do:** Build RandomForest end-to-end and only then test it against sklearn.
**Why it's wrong:** The hist/split/partition prims are the risky parts; burying them inside a forest makes failures un-localizable and violates the discipline that made v1–v3 estimators "mostly assembly."
**Do this instead:** Validate quantiles/hist/split/partition as standalone oracle-gated prims (Phase 18) against a single sklearn DecisionTree before RF (Phase 19) consumes them.

### Anti-Pattern 4: Putting cuml.accel logic in Rust/PyO3
**What people do:** Try to implement the import-hook or class-proxying in the compiled extension.
**Why it's wrong:** It is purely a Python `sys.modules`/`MetaPathFinder` concern over classes that already subclass `BaseEstimator`. Rust adds nothing and can't touch `sys.modules`.
**Do this instead:** A pure-Python `python/mlrs/accel/` subpackage; zero changes to mlrs-py Rust. (Mirrors cuML's own `cuml/accel/` being pure Python.)

### Anti-Pattern 5: A bespoke device-batched L-BFGS before proving the need
**What people do:** Build a new batched-L-BFGS kernel for ARIMA up front.
**Why it's wrong:** The existing `lbfgs` prim is proven; ARIMA's many-small-solves can be host-batched first. A device-batched variant is a perf optimization, not a correctness requirement.
**Do this instead:** Host-batch over the existing prim (Pattern 4); build device-batched only if profiling demands it.

### Anti-Pattern 6: A new device kernel for model-agnostic SHAP or the utility surface
**What people do:** Write SHAP coalition kernels or reimplement metrics on-device.
**Why it's wrong:** Kernel/Permutation SHAP is host sampling + existing `predict` + existing solver; metrics/model_selection are host numpy. New kernels add cpu-MLIR risk for no benefit.
**Do this instead:** Host-orchestrate, device-evaluate (Pattern 3); reuse existing predict/reduce/solve paths.

---

## Integration Points

### New external oracle dependencies (test-only, host)

| Oracle | Used by | Integration pattern | Notes |
|--------|---------|---------------------|-------|
| `scikit-learn` (RandomForest, DecisionTree, metrics, preprocessing, model_selection) | Phases 18,19,24 | committed `.npz` via `scripts/gen_oracle.py` + `mlrs_core::load_npz` | already the project oracle; extend `gen_oracle.py`. Tree gate = prediction/label agreement, not bit-exact internal structure. |
| `shap` library | Phases 21, 23 | `.npz` SHAP-value fixtures; tolerance gate (not 1e-5 — sampling-based) | new test dep; SHAP values are approximate → property/tolerance gate like UMAP. |
| `statsmodels` / `sklearn` ARIMA reference | Phase 22 | `.npz` series + fitted params/forecasts | confirm which reference cuML matches; likelihood/forecast tolerance gate. |
| `gplearn` | Phase 25 | `.npz` fitness/best-program fixtures | symbolic regression is stochastic → property/structural gate (à la RandomProjection D-12). |
| `umap-learn` / `hdbscan` | Phase 26 (accel) | already present (v3) | accel proxies to mlrs UMAP/HDBSCAN; CPU fallback to these. |

Regen requires a `/tmp` venv with numpy/sklearn/shap/gplearn (PEP 668; MEMORY.md `oracle-fixture-regen-needs-venv`). Fixtures are committed blobs.

### Internal boundaries (the load-bearing contracts)

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Tree builder (writer) ↔ FIL & TreeSHAP (readers) | flat `SparseTreeNode{colid,threshold,left_child,value}` `DeviceArray` + forest offset table | **The single most important new contract.** Fix its layout in the Phase-17 spike; all three consumers depend on it. |
| mlrs-algos estimators ↔ mlrs-backend prims | `fn prim<F>(pool, operands…, out: Option<…>) -> Result<_, PrimError>`, validate-before-launch, device-resident | unchanged v1–v3 prim shape; every new prim follows it. |
| mlrs-py shims ↔ existing `MlrsBase(BaseEstimator)` | new shims subclass `MlrsBase`; dtype-suffix (`_f32`/`_f64`) delegation; `n_features_in_`, `_post_fit` | reuse the v1 binding machinery (`any_estimator!`, ingress/egress, capability) — v2 added zero binding infra across 18 estimators; expect the same. |
| cuml.accel ↔ mlrs estimator surface | `sys.modules` swap + `_overrides` name→class table; pass-through = CPU fallback | pure Python; depends only on shims being importable without the extension (already true). |
| Batched L-BFGS ↔ existing `lbfgs` prim | host loop / batched wrapper over the proven prim | MODIFY, not rewrite. |
| Backend gate | `capability::skip_f64_with_log()` per f64 test; cpu(f64)+rocm(f32) | unchanged; every new prim test uses it; f64-on-rocm skips-with-log. |
| PoolStats memory gate | per-phase build-failing `peak_bytes`/`live_bytes`/`reuses` asserts | tree histograms (node-batch×bins×feat×outputs) and ARIMA batches must be tiled, never quadratic-resident — same discipline as Phase 13's query-axis tiling. |

---

## Sources

- `.planning/PROJECT.md` — v4.0 milestone scope, constraints, key decisions (read in-source, HIGH)
- `.planning/notes/v3-hard-algorithm-backlog.md`, `notes/cuml-mlrs-gap-inventory.md` — dependency ordering, tree feasibility flag (HIGH)
- `.planning/milestones/v3.0-phases/13-knn-graph-primitive-feasibility-keystone/` — `13-CONTEXT.md`, `13-RESEARCH.md`, `13-PATTERNS.md` — the keystone-spike-first model to mirror for Phase 17 (HIGH)
- `Skill("spike-findings-mlrs")` + `references/cpu-mlir-kernel-authoring.md` — proven cpu-MLIR GATHER op-set, the 002-A (loud) / 002-B (silent) landmines, single-owner per-output-cell idiom (HIGH)
- On-disk crate layout: `crates/mlrs-{core,kernels,backend,algos,py}/` (prims/, kernels, estimators, shims) read in-source (HIGH)
- `crates/mlrs-py/python/mlrs/{__init__.py,base.py}` — `MlrsBase(BaseEstimator)` shim contract that cuml.accel proxies (HIGH)
- cuML reference (behavior, NOT to port): `cuml-main/cpp/src/decisiontree/batched-levelalgo/` (atomic+shared-memory histogram → confirms the inversion need), `cpp/include/cuml/tree/flatnode.h` (`SparseTreeNode` format, right=left+1), `cpp/src/arima/batched_kalman.cu` (sequential Kalman), `cpp/src/explainer/{kernel,permutation,tree}_shap.cu`, `cpp/src/genetic/{program,node,reg_stack}` (host-evolve/device-evaluate), `cuml/accel/` (pure-Python proxy) (HIGH for structure)
- Project MEMORY.md — cpu-MLIR no-SharedMemory/atomics, rocm f64-unsupported gate, oracle-venv, disk/suite-slowness (MEDIUM)

---
*Architecture research for: mlrs v4.0 integration (tree ensembles, time-series, explainers, genetic, utility surface, cuml.accel)*
*Researched: 2026-06-26*
