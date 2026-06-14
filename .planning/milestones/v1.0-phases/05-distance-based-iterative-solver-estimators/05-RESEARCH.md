# Phase 5: Distance-Based & Iterative-Solver Estimators - Research

**Researched:** 2026-06-12
**Domain:** ML estimator assembly on validated CubeCL primitives — clustering (KMeans, DBSCAN), neighbors (KNN×3), iterative-solver linear models (Lasso, ElasticNet, LogisticRegression); numerical agreement with scikit-learn ≤ 1e-5 (up to label permutation for clustering)
**Confidence:** HIGH (sklearn 1.9.0 source is installed locally and was read line-by-line as the source of truth; CubeCL i32 support confirmed in the crate cache; all reused primitives inspected in-tree)

## Summary

Phase 5 is "mostly assembly" over Phase-2/3/4 primitives, with a small set of NEW device primitives (D-01) that must each pass standalone oracle + memory-gate validation before any estimator consumes them. The single load-bearing research deliverable is reproducing scikit-learn's exact solver objectives, penalty scalings, and stopping criteria — these are now pinned **verbatim from the installed sklearn 1.9.0 / scipy 1.17.1 source**, which is the oracle CI will generate fixtures against.

The two highest-value findings: (1) sklearn's coordinate descent (`_cd_fast.pyx`) minimizes the **un-normalized** primal `½‖y−Xw‖² + l1_reg·‖w‖₁ + (l2_reg/2)·‖w‖₂²` with `l1_reg = α·l1_ratio·n_samples` and `l2_reg = α·(1−l1_ratio)·n_samples`, soft-thresholding `w_j = sign(t)·max(|t|−l1_reg,0)/(‖X_j‖²+l2_reg)`, and stopping on **duality gap ≤ tol·‖y‖²** (`tol=1e-4`, `max_iter=1000`); and (2) sklearn's LogisticRegression `lbfgs` minimizes `(1/n)·Σ loss + ½·l2_reg·‖coef‖²` with `l2_reg = 1/(C·n_samples)`, **intercept unpenalized**, via `scipy.optimize.minimize(method="L-BFGS-B", jac=True)` with `maxcor=10`, `maxls=50`, `gtol=tol` (=1e-4), `ftol=64·eps`, `maxiter=100`. The multinomial loss is the **symmetric, over-parameterized** softmax (n_classes full weight vectors, not n_classes−1), so binary is genuinely the 2-class case of the same path (D-12).

**Primary recommendation:** Plan the NEW primitives first (top-k select, k-means++ D²-sample, Lloyd assign+update, DBSCAN eps-region+core-mask, CD step, L-BFGS direction), each as a feature-free `#[cube]` kernel in `mlrs-kernels` + a launch/orchestration wrapper in `mlrs-backend/prims/` with its own oracle and the build-failing PoolStats gate — exactly the Phase-4 Cholesky precedent. Validate the L-BFGS primitive standalone on a convex quadratic with a known minimizer BEFORE LogReg consumes it. Adopt the **host-driven iteration loop with a single scalar readback per iteration** (D-10) for both CD and L-BFGS — this is structurally what sklearn/scipy do and is the only reliable way to reproduce their exact stopping behavior.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Pairwise distance (KMeans/DBSCAN/KNN) | mlrs-backend prim (PRIM-03, exists) | — | Already validated; one prim serves all three families |
| Top-k selection per query row | mlrs-kernels `#[cube]` + mlrs-backend wrapper (NEW) | reduce.argmin_rows precedent | Partial-select-k over distance rows; lowest-index tie-break |
| k-means++ D²-weighted sampling | mlrs-backend host orchestration + device D² (NEW) | distance prim + host RNG | D-09c: host-side seeded RNG reads device D² weights per center (init only) |
| Lloyd assign (argmin) | reduce.argmin_rows (exists) | distance prim | Per-sample label = argmin of squared distance to centers |
| Lloyd centroid update + inertia | mlrs-kernels (NEW) or column_reduce composition | reduce.mean | Sum-by-label / count; inertia = Σ d²(x_i, c_{label_i}) |
| DBSCAN eps-region + core mask | mlrs-kernels `#[cube]` + wrapper (NEW) | distance prim | Device: threshold n² distance, row-count ≥ min_samples → core bit |
| DBSCAN cluster expansion (DFS) | **host** (D-04) | — | Inherently sequential pointer-chasing; reproduces sklearn `dbscan_inner` |
| Coordinate-descent step | mlrs-kernels (NEW) + host loop | gemm, reduce | Host loop, device residual/dot update, single-scalar gap readback (D-10) |
| L-BFGS direction + softmax loss/grad | mlrs-kernels (NEW) + host loop | gemm | Host loop owns history (s,y) pairs + line search; device computes loss/grad |
| Fitted-state storage (centers/labels/coef) | mlrs-algos estimator (device-resident, D-03) | DeviceArray | Lazy host-materialize at accessor/oracle boundary |
| Hyperparameter validation | mlrs-algos estimator (host, pre-launch) | AlgoError | ASVS V5 — reject bad α/C/k/eps/min_samples before any kernel launch |

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 (New-primitive boundary):** Aggressive promotion — every device-compute kernel is its own validated standalone primitive in `mlrs-backend/src/prims/` (feature-free `#[cube]` kernel in `mlrs-kernels`), validated standalone f32+f64 cpu+rocm against a numpy/sklearn reference and/or algebraic invariant, WITH the build-failing PoolStats memory gate, BEFORE any estimator consumes it. **Exception:** DBSCAN cluster expansion is sequential graph traversal — host-side, NOT a device kernel (D-04).
- **D-02 (Top-k primitive):** New partial-select-k kernel over pairwise-distance rows → k indices + k distances per query row with **lowest-index tie-break** (Phase-2 `argmin_rows` convention). Shared by `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor`. Built on PRIM-03 squared-Euclidean distance. **NEIGH-01 brute-force only — no spatial index in v1.**
- **D-03 (Two separate solvers):** A SHARED coordinate-descent kernel serves Lasso + ElasticNet (Lasso = `l1_ratio==1`). An INDEPENDENT L-BFGS serves LogisticRegression. Do NOT unify them.
- **D-04 (DBSCAN device-computes, host-expands):** Device computes pairwise distance matrix + eps-threshold + core-point mask; HOST runs BFS/union-find cluster expansion. Labels follow sklearn: noise = `-1`, plus `core_sample_indices_`. The n² distance matrix is the dominant allocation; the gate BOUNDS it (and confirms buffer reuse) rather than expecting a no-readback pipeline — DBSCAN deliberately reads the mask/distances back to host.
- **D-05 (Label-returning trait):** Integer labels (`KMeans.labels_`, `DBSCAN.labels_`, `KNeighborsClassifier.predict`) cannot use F-typed `Predict<F>`. Add a clustering/classify trait returning an integer `DeviceArray`; keep `Predict<F>` for regressors (`KNeighborsRegressor`, linear models). (Exact trait names/signatures = Claude's discretion.)
- **D-06 (i32 labels/indices everywhere):** DBSCAN noise = `-1` forces signed; use `i32` uniformly for ALL cluster ids, class predictions, and neighbor indices. `DeviceArray<ActiveRuntime, i32>` — confirm pool/bridge support during planning.
- **D-07 (KNeighbors + PredictProba traits):** `NearestNeighbors.kneighbors` returns BOTH `(DeviceArray<F>` distances, `DeviceArray<i32>` indices`)`; `KNeighborsClassifier` needs `predict_proba`. Formalize a `KNeighbors` trait and a `PredictProba` trait.
- **D-08 (sklearn API shape per family):** Clustering estimators implement `Fit` (storing `labels_`/`inertia_`/`cluster_centers_`/`core_sample_indices_` device-resident) + a `fit_predict`. **`KMeans` also implements `Predict`** (assign new points to fitted centers via distance + argmin). **`DBSCAN` does NOT implement standalone `predict`** (no transductive predict — like sklearn). Fitted attributes device-resident, lazy host-materialize.
- **D-09 (KMeans init-injected oracle):** k-means++ RNG can't be reproduced bit-for-bit; oracle SUPPLIES initial centers; both mlrs and sklearn run Lloyd from identical init, compare `cluster_centers_`/`labels_`/`inertia_` up to label permutation within 1e-5 (reuse Phase-1 `label_perm`).
  - **D-09a:** Still implement k-means++ (CLUSTER-01 names it as sklearn default) — build the D²-weighted sampling primitive; drive the deterministic oracle from injected init; separately sanity-check k-means++ validity/seed-reproducibility.
  - **D-09b:** `n_init = 1` (sklearn 'auto' default for k-means++). `n_init=10` deferred.
  - **D-09c:** Host-side seeded RNG draws the next center from device-computed D² weights (read back per center — INIT ONLY, not the hot Lloyd loop).
- **D-10 (Host-driven iteration loop):** Host runs the convergence loop; each iteration launches device kernels and reads back EXACTLY ONE scalar convergence metric (duality gap for CD; gradient-norm/pgtol for L-BFGS). **MEMORY-GATE RECONCILIATION (planner MUST encode):** the per-iteration scalar readback is an explicit documented EXCEPTION to the no-mid-pipeline-readback rule. For iterative solvers the gate asserts BOUNDED ALLOCATION instead — solver buffers (residuals, gradients, L-BFGS history `(s,y)` pairs, coordinate state) are pool-managed and REUSED across iterations (allocation count flat after warmup), and there is no per-iteration ARRAY readback beyond the single scalar. State this exception in the gate test.
- **D-11 (Match sklearn convergence criteria):** Reproduce per-solver stopping — Lasso/EN duality-gap < tol (`tol=1e-4`, `max_iter=1000`); LogReg L-BFGS gradient-norm/pgtol (`max_iter=100`). Exact constants pinned by research (below).
- **D-12 (LogReg = multinomial softmax):** Numerically-stable softmax + cross-entropy matching sklearn `lbfgs` default (lbfgs → multinomial). Binary is the 2-class case of the same code path — ONE formulation. OvR rejected.
- **D-13 (Match sklearn objectives/penalty scaling):**
  - Lasso/EN: `(1/2n)·‖y−Xw‖² + α·penalty`, `l1_ratio` mixing L1/L2, intercept via centering (unpenalized), reuse Phase-2 column-mean + Phase-4 D-05 center-then-solve.
  - LogReg: L2 penalty default, strength via `C` (inverse regularization), intercept unpenalized.
  - Exact scalings pinned by research (below).

### Claude's Discretion
- Exact set/granularity of the new primitives under D-01 (e.g. is "Lloyd update" one primitive or assignment + centroid-recompute split; is the CD step one primitive) — subject to the memory gate, tolerance policy, no-hardcoded-plane-width rule. **Researcher-flagged below with a recommended set.**
- Module/file layout within `mlrs-algos` (e.g. `cluster/`, `neighbors/`, `linear/`); exact trait names/method signatures for D-05/D-07.
- L-BFGS history size `m` and line-search details that reproduce scipy's L-BFGS-B behavior — pick what holds tolerance. **Researcher recommendation below: m=10, strong-Wolfe line search.**
- Exact random shapes/seeds for oracle sweeps; which cases get committed sklearn fixtures vs algebraic-invariant-only checks.
- Naming of new estimator/primitive error variants (extend `thiserror` enums).
- Distance-metric scope for KNN/DBSCAN (v1 Euclidean/squared via PRIM-03); `weights='uniform'` vs `'distance'` (default `'uniform'`).

### Deferred Ideas (OUT OF SCOPE)
- KMeans `n_init=10` / multi-restart-keep-best (v1 = `n_init=1`).
- KNN `weights='distance'`, non-Euclidean metrics, spatial indices (kd-tree/ball-tree). v1 is brute-force Euclidean, `weights='uniform'`.
- DBSCAN device-side label propagation (v1 = host expansion).
- Additional sklearn constructor knobs: `algorithm`/`leaf_size`/`p` (neighbors); `selection='random'`/`positive`/`warm_start` (CD); `penalty='l1'/'elasticnet'`/`solver` choices/`class_weight`/`multi_class='ovr'` (LogReg); `metric` variants (DBSCAN).
- Reusing the new optimizer primitives elsewhere (top-k for ANN, L-BFGS for other GLMs, CD for other sparse models) — built for v1 consumers only.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| LINEAR-03 | Fit `Lasso` (coordinate-descent) with `alpha`, obtain sparse `coef_` matching sklearn within tolerance | CD algorithm pinned from `_cd_fast.pyx`: soft-threshold update + duality-gap stop; Lasso = `l1_ratio=1` → `l2_reg=0`. Sparsity pattern reproduced by exact `max(|t|−l1_reg,0)` zeroing. |
| LINEAR-04 | Fit `ElasticNet` (`alpha`, `l1_ratio`, shared CD with Lasso) matching sklearn | Same kernel; `l1_reg=α·l1_ratio·n`, `l2_reg=α·(1−l1_ratio)·n` mapping pinned from `_coordinate_descent.py:781-782`. |
| LINEAR-05 | Fit `LogisticRegression` (L-BFGS) binary+multiclass, stable softmax, `predict`/`predict_proba` matching sklearn `lbfgs` | L-BFGS-B params + objective `(1/n)Σloss + ½·l2_reg·‖coef‖²`, `l2_reg=1/(C·n)`, intercept unpenalized, symmetric multinomial softmax — all pinned from `_logistic.py` + `_linear_loss.py` + scipy `_lbfgsb_py.py`. **Highest risk — see L-BFGS section.** |
| CLUSTER-01 | Fit `KMeans` (k-means++), read `cluster_centers_`/`labels_`/`inertia_`, `predict` new points, match sklearn up to label permutation | Lloyd update + `_tolerance = mean(var(X,axis=0))·tol`, `max_iter=300`, empty-cluster relocation, inertia = Σ d² pinned from `_kmeans.py`. Injected-init oracle (D-09). |
| CLUSTER-02 | Fit `DBSCAN` (`eps`/`min_samples`), read `labels_` (noise=-1) + `core_sample_indices_`, match sklearn up to permutation | Core = (eps-neighbor-count incl. self) ≥ min_samples; DFS expansion pinned from `_dbscan_inner.pyx`; `eps` is `<= eps` on Euclidean distance. |
| NEIGH-01 | Fit `NearestNeighbors` (brute-force), `kneighbors` → k distances + indices within 1e-5 | Top-k prim (D-02); sklearn = `argpartition` then `argsort` of the k; returns sqrt(d²) Euclidean. |
| NEIGH-02 | `KNeighborsClassifier` (`fit`/`predict`/`predict_proba`) matching sklearn | predict_proba = per-class neighbor fraction (uniform weights); predict = argmax (lowest class index on tie). Pinned from `_classification.py`. |
| NEIGH-03 | `KNeighborsRegressor` (`fit`/`predict`) matching sklearn within tolerance | predict = mean of k neighbor targets (uniform weights). Pinned from `_regression.py`. |
</phase_requirements>

## Standard Stack

This phase adds **no new external crates**. Everything is in-workspace plus the already-pinned CubeCL/cubek stack. The "stack" here is the set of reused primitives and the sklearn/scipy source that is the numerical contract.

### Core (reused, already validated — DO NOT re-implement)
| Component | Location | Purpose | Phase |
|-----------|----------|---------|-------|
| `distance` (squared-Euclidean, GEMM-expansion, `max(d²,0)` clamp, optional sqrt) | `mlrs-backend/src/prims/distance.rs` | KMeans assignment, DBSCAN eps-query, KNN top-k input | PRIM-03 / P2 |
| `argmin_rows` / `argmax_rows` (per-row, **lowest-index tie-break**, returns `Vec<u32>`) | `mlrs-backend/src/prims/reduce.rs` | KMeans label assignment, KNN voting | PRIM-02 / P2 |
| `column_reduce` / `mean` / `row_reduce` (`ScalarOp::{Sum,Mean,Min,Max,SumSq,L2Norm}`) | `mlrs-backend/src/prims/reduce.rs` | Centering, centroid recompute, inertia, CD column norms | PRIM-02 / P2 |
| `gemm` (transpose flags `transa`/`transb`, no transpose buffer) | `mlrs-backend/src/prims/gemm.rs` | `Xw` residuals, gradients, predict | PRIM-01 / P2 |
| `DeviceArray<R, F: Pod>` (`from_host`/`from_raw`/`to_host`/`to_host_metered`/`release_into`) | `mlrs-backend/src/device_array.rs` | Device-resident state; **`F: Pod` admits `i32` for D-06** | FOUND-05 / P1 |
| `BufferPool` / `PoolStats` (byte-size keyed free-list; counters `allocations`/`reuses`/`peak_bytes`/`live_bytes`/`read_backs`) | `mlrs-backend/src/pool.rs` | Pool reuse + the memory gate; **element-type-agnostic (byte-keyed) → i32 works unchanged** | FOUND-05 / P1 |
| `Fit` / `Predict` / `Transform` traits | `mlrs-algos/src/traits.rs` | Surface to extend with D-05/D-07 traits | P4 |
| `AlgoError` (`thiserror`; `InvalidNComponents`/`InvalidAlpha`/`NotFitted`/`Unsupported`/`Prim(#[from] PrimError)`) | `mlrs-algos/src/error.rs` | Extend with new hyperparameter-guard variants | P4 |
| `label_perm` (greedy confusion-matrix best-permutation, `i64` labels) | `mlrs-core/src/label_perm.rs` | Clustering comparison up to permutation (D-09) | FOUND-08 / P1 |
| `assert_close` / tolerance policy (1e-5 abs+rel, near-zero floor, per-family looser bound escape hatch) | `mlrs-core/src/{compare.rs,tolerance.rs}` | Oracle assertions | P3 D-10 |
| `skip_f64_with_log` / `supports_type(FloatKind::F64)` | `mlrs-backend/src/capability.rs` | f64-on-rocm skip-with-log (D-07) | P3 |
| `cholesky_solve` (single-cube SPD solve) | `mlrs-backend/src/prims/cholesky.rs` | NOT needed by Phase-5 iterative solvers (they use CD/L-BFGS, not normal equations) | P4 |

### Supporting (the numerical contract — source-of-truth, read at research time)
| Source file (installed sklearn 1.9.0 / scipy 1.17.1) | What it pins |
|-------------------------------------------------------|--------------|
| `sklearn/linear_model/_cd_fast.pyx` | CD update + `gap_enet` duality gap; `tol *= dot(y,y)` |
| `sklearn/linear_model/_coordinate_descent.py:781-782` | `l1_reg = α·l1_ratio·n`, `l2_reg = α·(1−l1_ratio)·n` |
| `sklearn/linear_model/_logistic.py:580-597` | `l2_reg = 1/(C·n)`; L-BFGS-B options (`maxiter`,`maxls=50`,`gtol=tol`,`ftol=64·eps`) |
| `sklearn/linear_model/_linear_loss.py:48-56,226-229` | objective `(1/n)Σloss + ½·l2_reg·‖coef‖²`; intercept unpenalized |
| `sklearn/_loss/loss.py` (`HalfMultinomialLoss`) | symmetric over-parameterized softmax; `_logsumexp` stable form |
| `sklearn/cluster/_kmeans.py:285-293,585-594` | `_tolerance=mean(var(X,0))·tol`; strict-convergence (labels equal) OR `center_shift_tot ≤ tol` |
| `sklearn/cluster/_k_means_common.pyx:167-201` | empty-cluster relocation = farthest points by `argpartition` |
| `sklearn/cluster/_dbscan_inner.pyx` | DFS (LIFO stack) expansion; index-order seed scan; `label_num` increment |
| `sklearn/cluster/_dbscan.py` | core = eps-neighbor-count (incl. self) ≥ min_samples |
| `sklearn/neighbors/_base.py:743-749` | `argpartition(k-1)` then `argsort` of the k; return `sqrt(d²)` |
| `sklearn/neighbors/_classification.py` | predict = argmax(proba); proba = neighbor class fraction (uniform) |
| `sklearn/neighbors/_regression.py` | predict = mean(neighbor targets) (uniform) |
| `scipy/optimize/_lbfgsb_py.py:96,272-274` | L-BFGS-B `m=maxcor=10`; `ftol=factr·eps`, `gtol=pgtol`; Moré-Thuente strong-Wolfe line search (Fortran 3.0, Zhu/Byrd/Nocedal) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff / Why rejected |
|------------|-----------|-------------------------|
| Host-driven iteration + scalar readback (D-10) | In-kernel iteration (Phase-3 Jacobi precedent) | In-kernel rejected for CD/L-BFGS by D-10: far harder, and worse at reproducing sklearn's exact stopping rule. Jacobi worked because its convergence test is a simple off-diagonal norm; CD/L-BFGS need the host to own history + line search + the gap test. |
| Symmetric over-parameterized multinomial softmax | n_classes−1 reference-class parameterization | Rejected: sklearn `HalfMultinomialLoss` is symmetric (n_classes full weight vectors). Using the reduced form would not reproduce sklearn's `coef_` (which sklearn does NOT post-center for lbfgs in 1.9). Must match sklearn's parameterization to hit 1e-5. |
| Injected-init KMeans oracle (D-09) | RNG-matched k-means++ oracle | RNG can't be reproduced bit-for-bit across numpy/Rust; injected init tests the Lloyd math deterministically. k-means++ validated separately for validity+seed-reproducibility. |
| Plain cyclic CD (no screening) | sklearn's Gap-Safe screening | sklearn 1.9 adds Gap-Safe screening as a SPEEDUP that does not change the result. mlrs can run plain cyclic CD and still match `coef_` — the screening only prunes coordinates provably zero at the optimum. Document this: do NOT try to reproduce the screening, only the final `coef_` + gap. |

**Installation:** None. No new crates. (cubecl 0.10.0 + cubek-matmul/cubek-std 0.2.0 already pinned in `Cargo.toml`.)

**Version verification:** `cubecl = "0.10.0"` and `cubek-matmul/std = "0.2.0"` confirmed in `Cargo.toml` and the workspace already builds against them (Phases 1-4 green). `CubeElement for i32` confirmed present in `~/.cargo/registry/.../cubecl-core-0.10.0/src/pod.rs:136` [VERIFIED: cargo registry cache]. No registry calls needed — zero new dependencies.

## Package Legitimacy Audit

> This phase installs **no external packages**. No slopcheck run is required.

| Package | Registry | Disposition |
|---------|----------|-------------|
| (none) | — | No new dependencies; all compute is in-workspace over the already-pinned cubecl 0.10.0 / cubek 0.2.0 stack. |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

Oracle fixtures are generated by `scripts/gen_oracle.py` against the **already-installed** `/tmp/oracle-venv` (numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0) — these are build-time-only, never installed by the Rust crate, never in CI. [VERIFIED: `/tmp/oracle-venv/bin/python -c 'import sklearn...'`]

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────────────────┐
  Arrow / host f32,f64   │            mlrs-algos estimators          │
   (Phase 6 ingests;     │  cluster/  neighbors/  linear/            │
    Phase 5 takes        │  KMeans DBSCAN  NN KNNc KNNr  Lasso EN LR │
    DeviceArray<F>)      └──────┬──────────┬──────────┬─────────────┘
                                │          │          │
                  ┌─────────────┘          │          └──────────────┐
                  ▼                        ▼                         ▼
        ┌──────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
        │ DISTANCE FAMILY  │   │   NEIGHBORS         │   │ ITERATIVE SOLVERS     │
        │ (device-resident)│   │   (top-k + host vote)│   │ (host loop + device)  │
        └──────────────────┘   └────────────────────┘   └──────────────────────┘
                  │                        │                         │
   ┌──────────────┼───────────┐    ┌───────┴────────┐     ┌──────────┴───────────┐
   ▼              ▼           ▼     ▼                ▼     ▼                      ▼
distance     argmin_rows  centroid top-k(NEW)   host vote  CD step(NEW)      L-BFGS(NEW)
(PRIM-03)    (PRIM-02)    update    select       (i32)     + gemm/reduce     softmax loss/grad
   │              │       (NEW)        │            │      + host gap loop   + host 2-loop
   │              │          │         │            │       (1 scalar/iter)   (1 scalar/iter)
   └──────────────┴──────────┴─────────┴────────────┴──────────┴──────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────────┐
                       │  mlrs-backend prims (launch wrap)  │   each NEW prim:
                       │  + BufferPool / PoolStats          │   oracle + PoolStats gate
                       └───────────────┬───────────────────┘   BEFORE estimator (D-01)
                                       ▼
                       ┌───────────────────────────────────┐
                       │  mlrs-kernels  feature-free #[cube]│   generic <F: Float+CubeElement>
                       │  top_k, kpp_d2, lloyd_update,      │   (i32 outputs materialized
                       │  eps_region_core, cd_step, lbfgs_* │    host-side from u32 argmin)
                       └───────────────────────────────────┘

  DBSCAN ONLY: device computes n² distance + core mask → READBACK to host →
               host DFS (dbscan_inner equivalent) → labels_ (noise=-1) + core_sample_indices_
               (documented memory-gate exception: n² bound + buffer reuse, NOT no-readback)
```

### Recommended Project Structure (Claude's discretion — recommended)
```
crates/mlrs-algos/src/
├── traits.rs           # extend: ClusterFit/PredictLabels (D-05), KNeighbors + PredictProba (D-07)
├── error.rs            # extend AlgoError: InvalidK, InvalidEps, InvalidMinSamples, NotConverged, InvalidL1Ratio, InvalidC
├── cluster/
│   ├── mod.rs
│   ├── kmeans.rs       # Fit + Predict + fit_predict; k-means++ init; Lloyd loop
│   └── dbscan.rs       # Fit + fit_predict (NO predict); device core-mask → host DFS
├── neighbors/
│   ├── mod.rs
│   ├── nearest.rs      # NearestNeighbors: KNeighbors trait (distances + i32 indices)
│   ├── classifier.rs   # KNeighborsClassifier: PredictLabels + PredictProba
│   └── regressor.rs    # KNeighborsRegressor: Predict<F>
└── linear/
    ├── mod.rs          # (already has Phase-4 linear models)
    ├── coordinate_descent.rs  # shared CD host loop for Lasso + ElasticNet
    ├── lasso.rs        # Lasso = ElasticNet(l1_ratio=1) thin wrapper
    ├── elastic_net.rs  # ElasticNet
    └── logistic.rs     # LogisticRegression: L-BFGS host loop + softmax

crates/mlrs-kernels/src/   # feature-free #[cube], generic <F: Float + CubeElement>
├── topk.rs             # partial-select-k over distance rows (NEW)
├── kmeans.rs           # D² compute for k-means++; Lloyd centroid sum-by-label + inertia (NEW)
├── dbscan.rs           # eps-threshold + per-row core-count → core mask (NEW)
├── coordinate.rs       # CD coordinate update / residual update (NEW)
└── lbfgs.rs            # softmax loss+grad; (history two-loop is host-side) (NEW)

crates/mlrs-backend/src/prims/   # launch wrappers + host orchestration
├── topk.rs   kmeans.rs   dbscan.rs   coordinate_descent.rs   lbfgs.rs   (NEW, one per kernel group)
```

### Pattern 1: Primitive-first with the build-failing memory gate (Phase-4 Cholesky precedent)
**What:** Each NEW device compute lands as a feature-free `#[cube]` kernel in `mlrs-kernels` + a `mlrs-backend/prims/*.rs` wrapper that **validates geometry before any `unsafe` launch** (ASVS V5), threads an optional reused buffer through (D-11 reuse), and returns a device-resident `DeviceArray`. It passes its own oracle (f32+f64, cpu+rocm) AND a PoolStats memory-gate assertion BEFORE any estimator consumes it.
**When to use:** Every D-01 primitive.
**Example (the precedent — `cholesky.rs` wrapper shape):**
```rust
// Source: crates/mlrs-backend/src/prims/cholesky.rs (Phase-4 precedent)
pub fn cholesky_solve<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    b: &DeviceArray<ActiveRuntime, F>,
    n: usize, rhs: usize,
    out: Option<DeviceArray<ActiveRuntime, F>>,   // D-11 reuse: thread Gram through
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where F: Float + CubeElement + Pod {
    // 1. validate a.len()==n*n, n<=MAX_DIM, b.len()==n*rhs  BEFORE unsafe launch
    // 2. single in-kernel launch (no host round-trip between phases)
    // 3. read back tiny `info` scalar; map non-SPD pivot → typed PrimError
    // ...
}
```

### Pattern 2: Host-driven iteration loop, single scalar readback (D-10)
**What:** The HOST owns the convergence loop. Each iteration: launch device kernel(s) that update device-resident state (residual/gradient/coefficients), then read back **exactly one scalar** (duality gap for CD; max-|proj-grad| for L-BFGS). Solver buffers are acquired ONCE before the loop and reused every iteration (allocation flat after warmup).
**When to use:** CD (Lasso/EN) and L-BFGS (LogReg).
**Pseudocode (CD):**
```text
# Source: sklearn/linear_model/_cd_fast.pyx (enet_coordinate_descent), de-screened
norm2_cols = column SumSq of X            # device, once
R = y - X @ w                             # device residual, once (w=0 → R=y)
tol_scaled = tol * dot(y, y)              # scalar, once  (tol=1e-4 default)
for n_iter in 0..max_iter (=1000):
    for j in 0..n_features:               # CYCLIC (sklearn default, random=0)
        if norm2_cols[j] == 0: continue
        t = X[:,j] · R + w[j] * norm2_cols[j]                       # device dot + scalar
        w_j_old = w[j]
        w[j] = sign(t) * max(|t| - l1_reg, 0) / (norm2_cols[j] + l2_reg)   # soft-threshold
        if w[j] != w_j_old:
            R += (w_j_old - w[j]) * X[:,j]                          # device residual update
    if w_max==0 or d_w_max/w_max <= tol or last_iter:              # cheap host gate
        gap = gap_enet(...)               # device dots → 1 scalar readback
        if gap <= tol_scaled: break
return w
```
**Memory-gate encoding (D-10 exception):** assert `read_backs` grows by 1 per OUTER convergence check (not per coordinate, not per array), and that `allocations` is FLAT after warmup across iterations (residual/norm/coefficient buffers reused). This is the iterative-solver analogue of the Phase-3 `memory_gate_jacobi_scratch_bounded` gate.

### Pattern 3: i32 outputs materialized host-side from u32 argmin (D-06)
**What:** Integer labels/indices are produced by reading back `u32` row-argmin results (KMeans labels, KNN votes) or `u32` top-k indices, then materialized into a `DeviceArray<ActiveRuntime, i32>` via `DeviceArray::from_host`. The pool is byte-size-keyed and `DeviceArray<R, F: Pod>` is generic over `F`, so `i32` works with **zero changes** to pool/bridge. DBSCAN noise `-1` is naturally representable in `i32`.
**When to use:** All label/index outputs (D-05/D-06/D-07).
**Verification note:** `argmin_rows` already returns `Vec<u32>` (host), so KMeans labels and KNN votes are host-computed and re-uploaded as `i32` — no new compute kernel needs an `i32` math path. A pure `i32` device kernel is only needed if a label transform runs on-device (not required in v1).

### Anti-Patterns to Avoid
- **Reproducing sklearn's Gap-Safe screening / active-set logic.** It's a speedup that does not change `coef_`. Match the final coefficients + gap with plain cyclic CD; reproducing screening adds risk for zero numerical benefit.
- **n_classes−1 multinomial parameterization.** sklearn uses the symmetric over-parameterized form; the reduced form won't reproduce `coef_`.
- **Device-side RNG for k-means++.** Backend-divergent streams, not seed-reproducible (D-09c rejected it). Host-side seeded RNG only.
- **Per-iteration array readback in the solver loop.** Only ONE scalar per outer convergence check (D-10). Reading the whole `w`/`R`/gradient back each iteration breaks the memory-gate exception.
- **Mergesort vs quicksort confusion in KNN tie-break.** sklearn `kneighbors` re-sorts the k-subset with `np.argsort(dist[...])` (default quicksort) AFTER `argpartition`; on exact distance ties the index order is NOT guaranteed stable. Pin the tie-break as **lowest-index** (D-02) and generate oracle fixtures with distinct distances to avoid a sort-stability mismatch — flag exact-tie cases for an algebraic-invariant check rather than index-equality.
- **In-kernel iteration for CD/L-BFGS.** Rejected by D-10. Host loop only.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pairwise distance | A new distance kernel | `prims::distance` (PRIM-03) | Already validated, GEMM-expansion + `max(d²,0)` clamp; one prim for all 3 families |
| Per-row min/argmin | A custom reduction | `prims::reduce::argmin_rows` | Lowest-index tie-break already pinned (matches D-02) |
| Column means / norms | A loop | `column_reduce` / `row_reduce` (`ScalarOp::Mean`/`SumSq`) | Centering, centroid recompute, CD column norms |
| `Xw` / gradients | A matmul | `prims::gemm` | Transpose flags, no transpose buffer |
| Clustering comparison | A label matcher | `mlrs_core::label_perm` | Greedy confusion-matrix best-permutation (D-09) |
| 1e-5 comparison | An ad-hoc epsilon | `assert_close` (abs+rel, near-zero floor) | Per-family looser-bound escape hatch already built |
| f64-on-rocm skip | A manual cfg | `skip_f64_with_log` | D-07 contract, logs the skip reason |
| The CD/L-BFGS math constants | Guessed scalings | The pinned sklearn constants below | Source-of-truth read from installed 1.9.0; guessing breaks 1e-5 |
| Soft-threshold / duality gap | A textbook formula | sklearn's EXACT `_cd_fast.pyx` form | `tol·‖y‖²`, `l1_reg/l2_reg = α·{l1_ratio,1−l1_ratio}·n` — non-obvious |

**Key insight:** Almost all device compute for the distance family is ALREADY a validated primitive. The genuinely-new device work is small: top-k partial-select, k-means++ D², Lloyd centroid-update+inertia, DBSCAN eps-core-mask, the CD coordinate update, and the softmax loss/gradient. The hard part is NOT the kernels — it's reproducing sklearn's exact host-side objective/stopping math, which is now pinned verbatim.

## Runtime State Inventory

> Not a rename/refactor/migration phase. This section is the closest analogue: **what fitted state crosses device↔host, and what must be host-only.**

| Category | Items | Action |
|----------|-------|--------|
| Device-resident fitted state | `cluster_centers_` (F), `coef_`/`intercept_` (F) | Stored device-resident on `self` (D-03), lazy host-materialize at accessor/oracle boundary |
| Host-materialized integer state | `labels_` (i32), `core_sample_indices_` (i32), KNN neighbor `indices` (i32) | Produced host-side from u32 argmin/top-k readback; re-uploaded as `DeviceArray<i32>` (D-06) |
| Host-only sequential state | DBSCAN DFS stack + cluster expansion | Inherently host (D-04); device only computes the n² distance + core mask |
| Solver scratch (reused across iterations) | CD: `R` residual, `norm2_cols`, `w`; L-BFGS: gradient, history `(s,y)` pairs ×m=10, search direction | Acquired ONCE before the host loop, reused every iteration (D-10 bounded-allocation gate) |
| Seeded RNG state | k-means++ host RNG (numpy `default_rng(seed)` equivalent for fixtures; Rust-side seeded for the estimator default) | Host-side only (D-09c); device never RNGs |

**Nothing found in category — n² distance matrix:** DBSCAN's n² matrix IS the dominant allocation and is the documented memory-gate exception (D-04): the gate BOUNDS it and confirms buffer reuse, rather than asserting no-readback.

## Common Pitfalls

### Pitfall 1: Wrong penalty scaling in CD (the 1e-5 killer for Lasso/EN)
**What goes wrong:** Using `α` directly in the soft-threshold instead of the n-scaled `l1_reg = α·l1_ratio·n_samples`.
**Why:** The user-facing objective is `(1/2n)‖y−Xw‖² + α(l1_ratio‖w‖₁ + ½(1−l1_ratio)‖w‖₂²)`, but `_cd_fast.pyx` minimizes the **un-normalized** `½‖y−Xw‖² + l1_reg‖w‖₁ + ½l2_reg‖w‖₂²` after multiplying through by n. The mapping is `l1_reg=α·l1_ratio·n`, `l2_reg=α·(1−l1_ratio)·n` [`_coordinate_descent.py:781-782`].
**How to avoid:** Use the pinned un-normalized form internally; verify the sparsity pattern (exact zeros) matches sklearn, not just the magnitudes.
**Warning signs:** `coef_` magnitudes off by a factor of ~n; wrong number of nonzeros.

### Pitfall 2: Wrong duality-gap tolerance scaling (CD stops too early/late)
**What goes wrong:** Comparing the raw gap to `tol` instead of `tol·‖y‖²`.
**Why:** sklearn does `tol *= dot(y,y)` because `G(0,0)=½‖y‖²` [`_cd_fast.pyx:391-392, 294-295`].
**How to avoid:** Scale tol by `‖y‖²` once before the loop. Use `gap_enet` formulation A (`alpha>0`): `gap` from `R_norm2`, `Ry`, `dual_norm_XtA=‖XᵀR−βw‖_∞`, `w_l1_norm`, `w_l2_norm2`.
**Warning signs:** Iterate count differs from sklearn by 10s-100s; final `coef_` off in the last digits.

### Pitfall 3: LogReg penalty on the intercept / wrong C scaling (the highest-risk 1e-5 failure)
**What goes wrong:** Penalizing the intercept, or using `l2_reg = 1/C` instead of `1/(C·n_samples)`.
**Why:** `l2_reg_strength = 1.0/(C·sw_sum)` with `sw_sum=n_samples` [`_logistic.py:580`]; the penalty is `½·l2_reg·‖coef_no_intercept‖²` — `weight_intercept` splits off the intercept BEFORE the penalty [`_linear_loss.py:147-184, 226-229`].
**How to avoid:** Penalize only the feature weights; intercept gradient gets the loss term but NO `+l2_reg·intercept`. Pin `l2_reg = 1/(C·n)`.
**Warning signs:** `intercept_` systematically shrunk toward 0; `coef_` slightly too small.

### Pitfall 4: Unstable softmax / log-sum-exp (NaN at large logits)
**What goes wrong:** `exp(raw)` overflows before normalization.
**Why:** Multinomial loss needs `logsumexp(raw, axis=1)` with the max subtracted.
**How to avoid:** Compute `m = max_k raw_k`; `lse = m + log(Σ exp(raw_k − m))`; `softmax_k = exp(raw_k − lse)`. sklearn uses `_logsumexp` + `softmax` from `utils.extmath` [`_loss/loss.py:1612,1645`].
**Warning signs:** NaN/Inf in the gradient on well-separated classes.

### Pitfall 5: L-BFGS not reproducing scipy's stopping → wrong iterate count → off `coef_`
**What goes wrong:** A home-grown L-BFGS with a different line search / history size stops at a different point than scipy's Fortran L-BFGS-B.
**Why:** scipy uses `m=10`, Moré-Thuente strong-Wolfe line search, `gtol=1e-4` (max |proj grad|), `ftol=64·eps` (relative f decrease), `maxls=50`, `maxiter=100` [`scipy/optimize/_lbfgsb_py.py`, `_logistic.py:588-594`].
**How to avoid:** Implement the standard two-loop recursion with `m=10` and a strong-Wolfe line search; use the SAME stopping constants. **Validate the L-BFGS primitive standalone on a convex quadratic with a known minimizer (e.g. `½xᵀAx − bᵀx`, optimum `x*=A⁻¹b`) BEFORE LogReg consumes it.** Because the convex objective converges to a unique global minimum, the final iterate must match `A⁻¹b` within 1e-5 regardless of small line-search differences — this isolates "is my L-BFGS correct" from "does it match sklearn's path."
**Escape hatch:** If bit-stopping parity proves infeasible, the per-family looser bound (P3 D-10) applies — match `coef_`/`predict_proba` within a documented per-family tolerance rather than the exact iterate. Flag this explicitly: LogReg is the one place where a slightly looser bound (e.g. 1e-4 on `coef_`, 1e-5 on `predict_proba`) may be the right escape hatch, because the *predictions* are far more stable than the *coefficients* under the symmetric over-parameterization (the softmax is invariant to adding a constant to all class logits).
**Warning signs:** `coef_` differs but `predict_proba` matches → the over-parameterization gauge freedom, not a bug. Test `predict_proba`/`predict` as the primary gate, `coef_` as secondary.

### Pitfall 6: KMeans convergence semantics (strict vs tol)
**What goes wrong:** Only checking `center_shift_tot ≤ tol` and missing sklearn's strict-convergence early exit.
**Why:** sklearn breaks if `array_equal(labels, labels_old)` (strict convergence) BEFORE the tol check, and otherwise if `center_shift_tot = Σ‖Δcenter‖² ≤ tol` where `tol = mean(var(X, axis=0))·tol_param` [`_kmeans.py:585-594, 285-293`]. After the loop, if not strict-converged, it runs ONE more label-assignment pass.
**How to avoid:** Reproduce both exit conditions and the scaled tol. `max_iter=300`.
**Warning signs:** Off-by-one iteration → slightly different final centers (but injected-init oracle + label_perm + 1e-5 absorbs tiny differences if the loop logic matches).

### Pitfall 7: DBSCAN border-point determinism
**What goes wrong:** A border point (non-core, but in some core point's eps-neighborhood) gets assigned to a different cluster than sklearn.
**Why:** sklearn's `dbscan_inner` is a DFS over points in **index order**; a border point joins the cluster of the FIRST core point that reaches it in that traversal [`_dbscan_inner.pyx`]. The result is order-dependent but DETERMINISTIC given index order.
**How to avoid:** Reproduce the exact DFS: iterate seeds `i in 0..n` in index order, skip if `labels[i]!=-1 or not is_core[i]`, push neighbors onto a LIFO stack, label `label_num` incrementing. Use `<= eps` on Euclidean distance; neighborhoods include the point itself.
**Warning signs:** Border points in a different cluster than sklearn → check traversal is index-ordered DFS (LIFO), not BFS (FIFO).

### Pitfall 8: KNN distance sqrt timing + tie-break
**What goes wrong:** Comparing/sorting on squared distance but returning squared distance, or a tie-break mismatch.
**Why:** sklearn `kneighbors` selects k via `argpartition` on **squared** distance (cheaper), re-sorts the k by squared distance, then returns `sqrt(d²)` (true Euclidean) [`_base.py:743-749`]. Selection on d² and sqrt only at the boundary is exactly the Phase-2 `distance(sqrt=...)` design.
**How to avoid:** Top-k on squared distance; apply sqrt only to the returned k distances. Tie-break = lowest index (D-02). Generate oracle fixtures with distinct distances; flag exact-tie cases for invariant checks.

## Code Examples

### CD soft-threshold update (the core CD kernel math)
```text
// Source: sklearn/linear_model/_cd_fast.pyx (enet_coordinate_descent), de-screened
// l1_reg = alpha * l1_ratio * n_samples ; l2_reg = alpha * (1-l1_ratio) * n_samples
t   = dot(X[:,j], R) + w_j_old * norm2_cols_X[j]        // device dot + fused scalar
w_j = sign(t) * max(|t| - l1_reg, 0) / (norm2_cols_X[j] + l2_reg)
if w_j != w_j_old: R += (w_j_old - w_j) * X[:,j]        // device residual update (axpy)
```

### CD duality gap (formulation A, alpha>0)
```text
// Source: sklearn/linear_model/_cd_fast.pyx (gap_enet + dual_gap_formulation_A)
XtA          = X.T @ R - beta * w            // gemv
dual_norm    = max_j |XtA[j]|                // abs_max  (||X'R - beta w||_inf)
R_norm2      = R @ R ;  Ry = R @ y           // dots
// scale R into a dual-feasible point if dual_norm > alpha, then:
gap = R_norm2 + alpha*||w||_1 + 0.5*beta*||w||_2^2 - Ry + (gauge terms)   // 1 scalar
stop when gap <= tol * (y @ y)               // tol=1e-4, max_iter=1000
```

### LogReg objective + gradient (multinomial, symmetric)
```text
// Source: sklearn/linear_model/_linear_loss.py:48-64,226-229 + _loss/loss.py (HalfMultinomial)
raw[i, :]  = X[i] @ W + b              // W: (n_classes, n_features), b: (n_classes,)
lse[i]     = logsumexp(raw[i, :])      // stable: m + log sum exp(raw - m)
loss       = (1/n) * sum_i ( lse[i] - raw[i, y_i] )  +  0.5 * l2_reg * ||W||_F^2
                                                       // l2_reg = 1/(C*n); b UNPENALIZED
p[i, :]    = softmax(raw[i, :])
gradW      = (1/n) * (P - Y_onehot).T @ X  +  l2_reg * W      // (P - Y) has the symmetric form
gradb      = (1/n) * sum_i (p[i,:] - Y_onehot[i,:])           // no penalty term
// scipy L-BFGS-B: m=10, maxls=50, gtol=1e-4, ftol=64*eps, maxiter=100, jac=True
```

### KMeans Lloyd iteration + convergence
```text
// Source: sklearn/cluster/_kmeans.py (lloyd) + _k_means_common.pyx (relocate)
tol_scaled = mean(var(X, axis=0)) * tol_param            // _tolerance, tol_param=1e-4
for it in 0..max_iter (=300):
    labels  = argmin_rows( distance(X, centers, sqrt=false) )   // squared distance
    for k:  centers[k] = mean(X[labels==k])                     // sum-by-label / count
            if cluster k empty: relocate to farthest point (argpartition of point->center d²)
    if array_equal(labels, labels_old): break                   // strict convergence
    if sum_k ||centers[k]-centers_old[k]||^2 <= tol_scaled: break
inertia = sum_i d2(X[i], centers[labels[i]])
```

### DBSCAN core mask + host DFS
```text
// Source: sklearn/cluster/_dbscan.py + _dbscan_inner.pyx
// DEVICE: D = distance(X, X, sqrt=false); neighbor[i] = { j : D[i,j] <= eps^2 }  (incl. i)
//         is_core[i] = (|neighbor[i]| >= min_samples)        // readback core mask + adjacency
// HOST (DFS, index order):
labels[:] = -1 ; label_num = 0
for i in 0..n:
    if labels[i] != -1 or not is_core[i]: continue
    stack = [i]
    while stack:
        v = stack.pop()                      // LIFO
        if labels[v] == -1:
            labels[v] = label_num
            if is_core[v]:
                for u in neighbor[v]:
                    if labels[u] == -1: stack.push(u)
    label_num += 1
core_sample_indices_ = { i : is_core[i] }    // noise stays -1
```

### i32 label DeviceArray (D-06 confirmation)
```rust
// DeviceArray<R, F: Pod> is generic; BufferPool is byte-size-keyed → i32 needs NO new code.
// labels computed host-side from u32 argmin_rows, then re-uploaded as i32:
let labels_u32: Vec<u32> = argmin_rows::<f32>(pool, &dist, rows, k, ReducePath::Shared)??;
let labels_i32: Vec<i32> = labels_u32.iter().map(|&l| l as i32).collect();
let labels_dev: DeviceArray<ActiveRuntime, i32> = DeviceArray::from_host(pool, &labels_i32);
// DBSCAN noise = -1 is directly representable; no kernel-side i32 math required in v1.
```

## State of the Art

| Old Approach | Current Approach (sklearn 1.9.0) | Impact |
|--------------|----------------------------------|--------|
| Classic Friedman cyclic CD | Cyclic CD + Gap-Safe screening (a *speedup*; same result) | mlrs runs plain cyclic CD; do NOT reproduce screening — match final `coef_`+gap |
| `multi_class='ovr'`/`'multinomial'` param | `multi_class` deprecated; `lbfgs` is always multinomial (binary = 2-class multinomial) | D-12 is correct; one softmax path |
| KMeans `n_init=10` default | `n_init='auto'` = 1 for k-means++ | D-09b `n_init=1` matches current default |
| Separate binary logistic | `HalfBinomialLoss` for binary, but lbfgs multinomial path covers it | mlrs uses one symmetric multinomial path (D-12) |

**Deprecated/outdated:**
- `multi_class` argument in `LogisticRegression` (deprecated; lbfgs → multinomial always). Do not expose it.
- `algorithm='full'`/`'auto'` in KMeans (renamed to `'lloyd'`/`'elkan'`; default `'lloyd'`).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | scipy L-BFGS-B uses Moré-Thuente strong-Wolfe line search (Fortran 3.0) | L-BFGS pinning | LOW — confirmed by web search + scipy docs; line-search *details* may differ slightly but the convex-quadratic standalone validation + predict_proba gate absorb it |
| A2 | A custom Rust L-BFGS with m=10 + strong-Wolfe can match sklearn `coef_` to 1e-5 across all penalty types | LINEAR-05 | MEDIUM — the symmetric over-parameterization gives `coef_` gauge freedom; **mitigated** by gating on `predict_proba`/`predict` (gauge-invariant) with `coef_` as a looser secondary check (Pitfall 5 escape hatch) |
| A3 | Plain cyclic CD (no screening) reproduces sklearn's exact sparsity pattern | LINEAR-03/04 | LOW — screening only prunes provably-zero coordinates; the optimum is identical. Verified by reading the screening logic (it includes/excludes, never changes the soft-threshold) |
| A4 | Oracle fixtures generated against sklearn 1.9.0 are the binding contract (CLAUDE.md says ≥1.6) | All | LOW — 1.9.0 is installed; if CI pins a different minor the committed blobs are regenerated by `gen_oracle.py`. Document the exact version in the fixture metadata. |
| A5 | KNN exact-distance ties are rare in random-data oracles and can be sidestepped with distinct-distance fixtures | NEIGH-01/02 | LOW — sklearn's post-`argpartition` `argsort` is not stable on ties; distinct-distance fixtures avoid the ambiguity (flag tie cases for invariant-only checks) |

## Open Questions (RESOLVED)

1. **CD granularity (Claude's discretion under D-01): one "CD step" primitive or split dot/soft-threshold/residual-update?**
   - RESOLVED: a single `cd_coordinate_update` kernel driven by a host cyclic loop (plan 05-05).
   - What we know: the per-coordinate update is a fused `dot(X_j,R) + axpy residual update`. sklearn does this with BLAS `_dot`/`_axpy` per coordinate.
   - What's unclear: whether to expose ONE `cd_sweep` device primitive (a full cyclic pass) or compose from `gemm`/reduce per coordinate.
   - Recommendation: a single `cd_coordinate_update` kernel that, given column `j`, computes `t` and updates `R` and `w[j]` device-side, called in the host cyclic loop. This keeps the device residual update on-device (memory-gate friendly) while the host owns the loop + gap. Validate standalone against numpy soft-threshold on a fixed `(X, y, w, R, j)`.

2. **L-BFGS history storage: device or host?**
   - RESOLVED: host owns the m=10 (s,y) two-loop history; device computes only softmax loss/grad (plan 05-06).
   - What we know: the two-loop recursion touches all m=10 `(s,y)` pairs each iteration; gradients are length `n_classes·(n_features+1)`.
   - Recommendation: host-owns the history + two-loop recursion (it's tiny — m·dof scalars); device computes only the softmax loss+gradient (the n_samples-heavy part). This matches the D-10 "host loop, device kernel" split and keeps the gradient the only large device buffer (reused each iteration).

3. **Does the memory-gate exception need a NEW assertion form, or can it reuse the Phase-3 `allocations-flat-after-warmup` pattern?**
   - RESOLVED: reuse the Phase-3 allocations-flat-after-warmup pattern + a DBSCAN n²-bound gate (plan 05-11).
   - Recommendation: reuse the Phase-3 pattern. Add an iterative-solver gate that runs N solver iterations and asserts (a) `read_backs` grows by exactly 1 per outer convergence check, (b) per-iteration `allocations` delta == 0 after warmup, (c) `live_bytes`/`peak_bytes` conserve. For DBSCAN, a separate gate asserts the n² matrix is allocated once and reused, and that the core-mask readback is the documented single host round-trip.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust + cargo workspace | all | ✓ | (in-tree) | — |
| cubecl | kernels | ✓ | 0.10.0 | — |
| cubek-matmul / cubek-std | gemm | ✓ | 0.2.0 | — |
| `CubeElement for i32` | D-06 labels | ✓ | cubecl-core 0.10.0 (pod.rs:136) | — |
| cpu backend (f64 gate) | oracle f64 | ✓ | (Phase 1-4 green) | — |
| rocm/HIP backend (f32 gate, gfx1100) | oracle f32 | ✓ | (Phase 3 bring-up) | f64-on-rocm skip-with-log (D-07) |
| `/tmp/oracle-venv` (numpy/scipy/sklearn) | fixture regen only | ✓ | numpy 2.4.6 / scipy 1.17.1 / sklearn 1.9.0 | committed blobs (regen never in CI) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** f64-on-rocm (skip-with-log, D-07 — not a blocker).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` (integration tests in `crates/*/tests/`, AGENTS.md §2 — no in-source `#[cfg(test)]`) |
| Config file | none (cargo default); fixtures in `tests/fixtures/*.npz` via `mlrs_core::oracle::load_npz` |
| Quick run command | `cargo test --features cpu -p mlrs-backend <prim>_test` (targeted prim) |
| Full suite command | `cargo test --features cpu` then `cargo test --features rocm` (cpu f64 + rocm f32 gate) |

> **Note (MEMORY.md):** the `mlrs-backend` cpu suite is ~6 min (reduce_test 248s, svd_test 99s). Run TARGETED post-merge gates per prim; background the full run. Phase-5 adds top-k/k-means/dbscan/CD/L-BFGS tests — keep each prim's oracle tight and run the full sweep in the background.

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| (NEW prim) top-k select | k indices+distances, lowest-index tie | unit/oracle | `cargo test --features cpu -p mlrs-backend topk_test` | ❌ Wave 0 |
| (NEW prim) k-means++ D² | valid+seed-reproducible D² sampling | invariant | `cargo test --features cpu -p mlrs-backend kmeanspp_test` | ❌ Wave 0 |
| (NEW prim) Lloyd update+inertia | centroid sum-by-label, inertia | oracle | `cargo test --features cpu -p mlrs-backend lloyd_test` | ❌ Wave 0 |
| (NEW prim) DBSCAN eps-core-mask | core bit, eps-neighborhood incl self | oracle | `cargo test --features cpu -p mlrs-backend dbscan_mask_test` | ❌ Wave 0 |
| (NEW prim) CD coordinate update | soft-threshold + residual update | oracle | `cargo test --features cpu -p mlrs-backend cd_test` | ❌ Wave 0 |
| (NEW prim) L-BFGS direction + softmax loss/grad | convex-quadratic min; softmax grad | oracle/invariant | `cargo test --features cpu -p mlrs-backend lbfgs_test` | ❌ Wave 0 |
| CLUSTER-01 | KMeans centers/labels/inertia up to perm | oracle | `cargo test --features cpu -p mlrs-algos kmeans_test` | ❌ Wave 0 |
| CLUSTER-02 | DBSCAN labels(-1)+core_sample_indices_ | oracle | `cargo test --features cpu -p mlrs-algos dbscan_test` | ❌ Wave 0 |
| NEIGH-01 | NearestNeighbors k dist+idx 1e-5 | oracle | `cargo test --features cpu -p mlrs-algos nearest_neighbors_test` | ❌ Wave 0 |
| NEIGH-02 | KNeighborsClassifier predict/proba | oracle | `cargo test --features cpu -p mlrs-algos knn_classifier_test` | ❌ Wave 0 |
| NEIGH-03 | KNeighborsRegressor predict | oracle | `cargo test --features cpu -p mlrs-algos knn_regressor_test` | ❌ Wave 0 |
| LINEAR-03 | Lasso sparse coef_ | oracle | `cargo test --features cpu -p mlrs-algos lasso_test` | ❌ Wave 0 |
| LINEAR-04 | ElasticNet coef_ | oracle | `cargo test --features cpu -p mlrs-algos elastic_net_test` | ❌ Wave 0 |
| LINEAR-05 | LogReg predict/proba (binary+multiclass) | oracle | `cargo test --features cpu -p mlrs-algos logistic_test` | ❌ Wave 0 |
| Memory gate (D-10) | iterative-solver bounded alloc + 1-scalar/iter readback | hard gate | `cargo test --features cpu -p mlrs-backend memory_gate_test` | extends existing |
| Memory gate (D-04) | DBSCAN n² bound + core-mask readback | hard gate | `cargo test --features cpu -p mlrs-backend memory_gate_test` | extends existing |

### Sampling Rate
- **Per task commit:** the touched prim's targeted oracle test (`cargo test --features cpu -p mlrs-backend <prim>_test`).
- **Per wave merge:** that estimator's `-p mlrs-algos` oracle test + the prim oracle tests it depends on, on cpu(f64); spot-check rocm(f32).
- **Phase gate:** full `cargo test --features cpu` + `cargo test --features rocm` green (f64-on-rocm skips logged), including the extended `memory_gate_test`, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tests/topk_test.rs`, `kmeanspp_test.rs`, `lloyd_test.rs`, `dbscan_mask_test.rs`, `cd_test.rs`, `lbfgs_test.rs` (mlrs-backend prim oracles)
- [ ] `tests/{kmeans,dbscan,nearest_neighbors,knn_classifier,knn_regressor,lasso,elastic_net,logistic}_test.rs` (mlrs-algos estimator oracles)
- [ ] `scripts/gen_oracle.py` extensions: `gen_kmeans` (with INJECTED init centers — D-09), `gen_dbscan`, `gen_knn`, `gen_lasso`, `gen_elastic_net`, `gen_logistic` (binary + multiclass) — committed `.npz` blobs, regen in `/tmp/oracle-venv`
- [ ] `memory_gate_test.rs` extensions: iterative-solver bounded-allocation gate (CD + L-BFGS) + DBSCAN n²-bound gate (D-10/D-04 exceptions)
- [ ] `traits.rs` extensions: label-returning + KNeighbors + PredictProba traits (D-05/D-07)
- [ ] `error.rs` extensions: new hyperparameter-guard variants (InvalidK/InvalidEps/InvalidMinSamples/InvalidL1Ratio/InvalidC/NotConverged)

## Security Domain

> `security_enforcement: true`, `security_asvs_level: 1`. This is a Rust numeric library; the attack surface is **untrusted hyperparameters and geometry at the estimator/prim boundary** (the Phase-4 threat model T-04-01-01 carries forward).

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (library) |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No access control surface |
| V5 Input Validation | **yes** | Validate ALL hyperparameters + geometry BEFORE any `unsafe` kernel launch → typed `AlgoError`/`PrimError`, never an out-of-bounds device read. `k ≤ n_samples`, `eps ≥ 0`, `min_samples ≥ 1`, `0 ≤ l1_ratio ≤ 1`, `α ≥ 0`, `C > 0`, `n_clusters ≤ n_samples`, shape products match buffer lengths. |
| V6 Cryptography | no | No crypto; seeded RNG is for reproducibility only (not security-sensitive) — use a documented seeded PRNG, never `OsRng` |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device read from a bad `(rows, cols)` / `k` / `n_clusters` | Tampering / Info-disclosure | Validate geometry vs buffer `len` before `unsafe { ArrayArg::from_raw_parts }` (Phase-2/3/4 precedent: `distance`/`cholesky` validate-before-launch) |
| NaN poisoning from non-convergent solver / degenerate input (empty cluster, all-zero column, singular) | DoS | Guard: empty-cluster relocation (KMeans), `norm2_cols[j]==0 → skip` (CD), `max_iter` cap → `AlgoError::NotConverged` (never silent NaN). Mirrors Cholesky non-SPD `info` guard. |
| Integer overflow in n² distance allocation (DBSCAN large n) | DoS | Bound n² allocation; the memory gate asserts the bound (D-04). Validate `n_samples` against a sane cap or document the n² memory cost. |
| Division by zero in soft-threshold / centroid mean (zero column / empty cluster) | DoS | `norm2_cols[j]+l2_reg` denominator guarded by the `==0 → skip`; empty cluster relocated before the mean. |

## Sources

### Primary (HIGH confidence — read line-by-line this session)
- **Installed scikit-learn 1.9.0 source** (`/tmp/oracle-venv/lib/python3.12/site-packages/sklearn/`): `linear_model/_cd_fast.pyx`, `_coordinate_descent.py`, `_logistic.py`, `_linear_loss.py`, `_loss/loss.py`, `cluster/_kmeans.py`, `_k_means_common.pyx`, `_dbscan.py`, `_dbscan_inner.pyx`, `neighbors/_base.py`, `_classification.py`, `_regression.py` — the numerical contract.
- **Installed scipy 1.17.1 source** (`scipy/optimize/_lbfgsb_py.py`): L-BFGS-B defaults `m=10`, `ftol`/`gtol`/`maxls` mapping.
- **In-tree code:** `crates/mlrs-backend/src/{device_array.rs,pool.rs,prims/{distance,reduce,gemm,cholesky}.rs,capability.rs,runtime.rs}`, `crates/mlrs-algos/src/{traits,error}.rs`, `crates/mlrs-backend/tests/memory_gate_test.rs`, `crates/mlrs-core/src/label_perm.rs`, `scripts/gen_oracle.py`.
- **cubecl-core 0.10.0** (`~/.cargo/registry/.../pod.rs:136`): `impl CubeElement for i32`.
- CONTEXT.md D-01..D-13, ROADMAP §Phase 5, REQUIREMENTS.md, Phase-2/3/4 CONTEXT, AGENTS.md.

### Secondary (MEDIUM confidence)
- scipy docs `minimize(method='L-BFGS-B')` — confirmed maxcor/ftol/gtol semantics.

### Tertiary (LOW confidence — flagged, verified against primary where used)
- WebSearch: L-BFGS-B Moré-Thuente strong-Wolfe line search (Fortran 3.0, Zhu/Byrd/Nocedal) — corroborates the local scipy source; line-search micro-details not load-bearing given the convex-quadratic standalone validation + gauge-invariant predict_proba gate.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all reused prims inspected in-tree; no new deps; i32 support confirmed in the crate cache.
- Solver objectives/scalings (CD + LogReg): HIGH — pinned verbatim from installed sklearn 1.9.0 / scipy 1.17.1 source.
- Architecture/primitive split: HIGH — follows the proven Phase-4 Cholesky precedent + the existing memory-gate structure.
- L-BFGS exact path-matching: MEDIUM — gauge freedom in the symmetric multinomial means `coef_` may need the per-family looser bound; mitigated by gating on gauge-invariant `predict_proba`/`predict` + a standalone convex-quadratic validation.
- KNN exact-tie behavior: MEDIUM — sklearn's post-argpartition argsort is not tie-stable; sidestepped with distinct-distance fixtures.

**Research date:** 2026-06-12
**Valid until:** 2026-07-12 (stable — sklearn/scipy source pinned to installed versions; CubeCL stack already locked. Re-verify only if the oracle sklearn/scipy version changes.)
