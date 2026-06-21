# Phase 9: Spectral Family - Research

**Researched:** 2026-06-21
**Domain:** Spectral graph embedding & clustering (normalized graph Laplacian + symmetric eig + KMeans), sklearn-exact value/label matching
**Confidence:** HIGH (sklearn 1.9.0 source read line-by-line on this machine; full-spectrum-then-slice empirically reproduces sklearn ARPACK to 8.3e-7; every mlrs reuse seam read with file:line)

## Summary

Phase 9 is an **assembly-and-pin** phase, not a discovery phase. CONTEXT.md already locked every architectural decision (D-01..D-11); the job of this research was to (a) CONFIRM-AND-DOCUMENT the `[v2-P3]` smallest-eigenpair research flag, and (b) PIN the exact sklearn-source formulas the decisions defer to "during planning." Both are now done against the **actual sklearn 1.9.0 + scipy source installed on this machine** (`/home/user/.local/lib/python3.12/site-packages/sklearn/`), not from memory.

The phase ships ONE new device primitive (`prims/laplacian.rs`) plus two estimators (`SpectralEmbedding`, `SpectralClustering`). The rbf affinity IS Phase-8 `kernel_matrix(X, X, Kernel::Rbf{gamma})` (zero new affinity math); the `nearest_neighbors` affinity is a small new sklearn-exact binary-connectivity builder over v1 `distance`+`topk`. The eigensolver is v1 `eig` used as-is — it returns the FULL symmetric spectrum sorted DESCENDING with a `MAX_DIM=64` cap, which *is* the documented `n_samples ≤ 64` problem-size ceiling. KMeans (v1, `KMeans::new`, kmeans++, `n_init=1`) is reused unchanged for the clustering label-assignment.

**Primary recommendation:** Build `laplacian.rs` as a thin host orchestration mirroring `kernel_matrix.rs` (affinity base-op → in-place map), reproducing scipy's EXACT dense normalized-Laplacian (zero-diagonal → row-sum degree → `dd = where(w==0, 1, sqrt(w))` typed-zero guard → `L = I − D^-1/2 A D^-1/2` with isolated-node diagonal = 0). Then `SpectralEmbedding` = affinity → laplacian → eig (reverse to ascending) → `/dd` recovery → sign-flip → drop trivial; `SpectralClustering` = embedding (drop_first=False) → v1 KMeans. The `[v2-P3]` flag is **CLOSED** as confirm-and-document: full-spectrum-then-slice via dense Jacobi `eig` is exact and sufficient at n≤64 (empirically 8.3e-7 vs sklearn ARPACK); no Lanczos/shift-invert.

## User Constraints (from CONTEXT.md)

### Locked Decisions (verbatim from 09-CONTEXT.md `## Implementation Decisions`)

**Affinity scope, defaults & gamma**
- **D-01:** Mirror sklearn per-estimator affinity defaults. `SpectralEmbedding` default affinity = `'nearest_neighbors'`; `SpectralClustering` default affinity = `'rbf'`. Both affinity builders must work for both estimators. The oracle uses each estimator's own default constructor with NO override.
- **D-02:** rbf affinity = `kernel_matrix(Rbf)` from Phase 8. No re-derivation.
- **D-03:** nearest_neighbors affinity = sklearn-exact binary connectivity graph. Build a kNN graph with `mode='connectivity'` (weights 0/1), `n_neighbors` default `10`, symmetrized via `0.5·(A + Aᵀ)`. Reuse v1 neighbors top-k (`prims/topk.rs` + `prims/distance.rs`), then binarize + symmetrize.
- **D-04:** gamma mirrors each estimator exactly. `SpectralEmbedding`: `gamma=None → 1/n_features` (computed at fit). `SpectralClustering`: `gamma` default `1.0` (literal). Both gamma paths value-pinned in the oracle.

**Eigensolver path & problem-size cap ([v2-P3])**
- **D-05:** Full-spectrum-then-slice, hard cap `n_samples ≤ 64`. Use v1 `eig` as-is. Reverse to ascending and slice the smallest non-trivial eigenvectors. No Lanczos / shift-invert. `[v2-P3]` is pre-answered; the spike confirms-and-documents.
- **D-06:** Reject `n_samples > 64` as a typed `AlgoError` BEFORE any device work (ASVS-V5). Don't defer to `eig`'s internal `PrimError`; fail loud with a spectral-domain message naming the MAX_DIM cap.

**Embedding recovery**
- **D-07:** Reproduce the `D^-1/2` diffusion-map recovery exactly. Divide each recovered eigenvector by `dd = sqrt(degree)` BEFORE the deterministic sign flip. Pin the exact operation ORDER from sklearn `_spectral_embedding.py`.
- **D-08:** `n_components` default `2`; mirror sklearn drop_first. Compute the smallest `(n_components + 1)` eigenvectors, drop the trivial ≈0 one, keep `n_components`.
- **D-09:** Degenerate spectra → subspace test. For value-ambiguous (repeated) eigenvalues the per-vector value match is replaced by a subspace test; normal spectra use the value match after sign alignment.

**Exact-label reproduction**
- **D-10:** Well-separated oracle fixture → init-invariant labels. Design SpectralClustering oracle data so clusters are well-separated → unique partition → any KMeans converges to the same labels up to permutation. Reuse v1 KMeans as-is (default kmeans++, `n_init=1`). Mirrors the Phase-5 DBSCAN tuned-fixture design, NOT Phase-5 KMeans init-injection (rejected here).
- **D-11:** `n_components = n_clusters`, `assign_labels='kmeans'`-only. The embedding dimension defaults to `n_clusters`; kmeans is the ONLY label-assignment path. `'discretize'`/`'cluster_qr'` out of scope.

### Claude's Discretion (verbatim)
- Exact f32-on-rocm tolerance band for `SpectralEmbedding` `embedding_` (embedding band + sign, or the subspace test for degenerate spectra) — follow the v1 per-family documented-band precedent; f64 stays strict (gated by `skip_f64_with_log`). **Exact labels** is the hard gate for `SpectralClustering` (no band).
- The precise `laplacian.rs` degree-reduction kernel shape (single-owner row reduction; GATHER not scatter; typed-zero guard for zero-degree nodes — NO `F::INFINITY`, no atomics, SharedMemory-free).
- Whether `SpectralEmbedding`/`SpectralClustering` need a new trait or compose on existing `Fit` + `Transform` (`embedding_`) / `PredictLabels` (`labels_`) — likely NO new trait.
- Exact sklearn-source-pinned formulas: gamma `None→value` per estimator (D-04), the `_spectral_embedding` recovery/drop_first slice order (D-07/D-08), the `n_components` default (D-11). **← all pinned in this RESEARCH below.**

### Deferred Ideas (OUT OF SCOPE)
- `affinity='precomputed'` / `'precomputed_nearest_neighbors'`.
- `assign_labels='discretize'` / `'cluster_qr'`.
- Lanczos / shift-invert smallest-eigenpair solver (lifts the n≤64 cap).
- `n_samples > 64` (above v1 `eig` MAX_DIM) — hard-rejected as a typed error.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PRIM-09 | Graph-Laplacian primitive (normalized Laplacian with GATHER degree-normalization, no atomics) composes the kernel/affinity matrix, serving the spectral estimators | `laplacian.rs` kernel shape pinned below (scipy `_laplacian_dense` exact formula); reuses `reduce::row_reduce(SumSq→Sum)` for degree, `kernel_matrix.rs` in-place-map idiom for `d_inv_sqrt`/`L`. PoolStats gate per `memory_gate_test.rs` precedent. |
| SPECTRAL-01 | `SpectralEmbedding` → `embedding_` matching sklearn within tolerance after sign alignment | Full operation order pinned from `_spectral_embedding` (lines 353–477); empirically 8.3e-7 vs ARPACK at n=12. Eig reuse seam at `eig.rs:75`. Sign-flip exact (`_deterministic_vector_sign_flip`). |
| SPECTRAL-02 | `SpectralClustering` → `labels_` matching sklearn up to label permutation | `n_components=n_clusters`, `drop_first=False`, `assign_labels='kmeans'` pinned from `SpectralClustering.fit`. v1 `KMeans::new` at `kmeans.rs:112`. D-10 well-separated fixture makes `n_init` immaterial. |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| rbf affinity `K(X,X)` | Device prim (`kernel_matrix.rs`) | — | Phase-8 keystone; pure GEMM/distance + map, device-resident |
| kNN-connectivity affinity | Device prim (`distance`+`topk`) + host binarize/symmetrize | Algos estimator | top-k is device; the 0/1 binarize + `0.5(A+Aᵀ)` is small host math on the n×k indices |
| normalized Laplacian | Device prim (`laplacian.rs`) | — | New PRIM-09; degree via device `row_reduce`, `d_inv_sqrt`/`L` via device in-place map |
| symmetric eig (full spectrum) | Device prim (`eig.rs`) | Host sort/slice | Jacobi sweep on device; descending sort + column permute on host (already in eig) |
| `D^-1/2` recovery + sign-flip + drop-trivial | Algos estimator (host) | — | Small post-eig host math on the n×(k+1) eigenvector slice (sklearn does this in numpy too) |
| KMeans label assignment | Algos estimator (`kmeans.rs`) | — | v1 KMeans reused unchanged; device assign/update, host convergence |
| dtype dispatch / GIL release | PyO3 (`mlrs-py`) | — | `any_estimator!` Unfit/F32/F64; `py.detach`; `guard_f64()` |

## Standard Stack

No new external dependencies. v2 adds ZERO compute dependencies (REQUIREMENTS.md "Out of Scope": "no `cubek-random`, no pyo3 bump (stays 0.28)"). This phase is pure assembly over already-vendored crates.

### Core (all existing, validated)
| Component | Location | Purpose | Reuse note |
|-----------|----------|---------|------------|
| `kernel_matrix` | `crates/mlrs-backend/src/prims/kernel_matrix.rs:121` | rbf affinity `K=exp(-γ‖xᵢ−xⱼ‖²)` | `Kernel::Rbf{gamma}` arm at `:164`; pass `y=x`, `(n,d),(n,d)` |
| `eig` | `crates/mlrs-backend/src/prims/eig.rs:75` | full symmetric spectrum, DESCENDING, `MAX_DIM=64` | returns `(w,V)`; `V` column-major (`v[c*n+r]=V[r,c]`, see `:179`) |
| `distance` | `crates/mlrs-backend/src/prims/distance.rs:79` | pairwise sq-euclidean (`sqrt=false`) | base for kNN affinity |
| `top_k` | `crates/mlrs-backend/src/prims/topk.rs:61` | k smallest per row + indices (lowest-index tie-break) | for kNN-connectivity |
| `row_reduce` | `crates/mlrs-backend/src/prims/reduce.rs:180` | per-row reduction; `ScalarOp::Sum` | degree vector = row-sum of affinity (single-owner GATHER) |
| `KMeans` | `crates/mlrs-algos/src/cluster/kmeans.rs:112` | kmeans++ + Lloyd, `n_init=1` | `KMeans::new(n_clusters, seed)`; do NOT use `with_init` (D-10) |
| `NearestNeighbors` | `crates/mlrs-algos/src/neighbors/nearest.rs:77` | brute-force kNN | optional helper for kNN affinity; or call `distance`+`topk` directly |
| `any_estimator!` | `crates/mlrs-py/src/dispatch.rs:85` | Unfit/F32/F64 dtype dispatch | mirror `estimators/kernel.rs` |

**Installation:** none.

## Package Legitimacy Audit

Not applicable — this phase installs no external packages. All dependencies are first-party crates already in the workspace (`mlrs-backend`, `mlrs-algos`, `mlrs-py`, `mlrs-kernels`, `mlrs-core`) plus the already-pinned `cubecl`/`cubek-*`/`pyo3`/`bytemuck`/`thiserror` stack from v1. **Packages removed due to [SLOP]:** none. **Packages flagged [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```text
                    SpectralEmbedding.fit(X)                 SpectralClustering.fit(X)
                            │                                         │
              ┌─────────────┴─────────────┐                          │
   affinity='nearest_neighbors'      affinity='rbf'           affinity='rbf' (default, D-01)
   (SE default, D-01)                                                 │
        │                                  │                          │
        ▼                                  ▼                          ▼
   distance(X,X,sqrt=F)            kernel_matrix(X,X,                kernel_matrix(X,X,
   → top_k(k=10) → binarize 0/1      Rbf{γ=1/n_feat})  ◄── D-02 ──►  Rbf{γ=1.0 literal, D-04})
   → A = 0.5·(A+Aᵀ)  (D-03)         (γ=None→1/n_feat, D-04)              │
        └──────────────┬───────────────────┘                            │
                       ▼   affinity matrix A (n×n, n≤64)                 │
        ┌──────────────────────────────────────────────────────────────┘
        ▼
   laplacian.rs  (PRIM-09, NEW)
   ── fill_diagonal(A,0)                       [drop self-loops]
   ── w = rowsum(A)        via row_reduce(Sum) [degree, GATHER single-owner]
   ── dd = where(w==0, 1, sqrt(w))             [typed-zero guard, NO F::INFINITY]
   ── L = -A / dd / ddᵀ ; diag(L)=1-isolated   [L = I − D^-1/2 A D^-1/2]
        ▼   L (n×n symmetric, returns dd alongside)
   eig(L)  → (w_desc, V_desc)   [v1 eig, MAX_DIM=64 = the n≤64 cap]
        ▼   REVERSE descending→ascending; take smallest (n_components+drop_first)
   recovery:  emb = V_slice.T / dd            (D-07, /dd row-wise BEFORE sign-flip)
   ── _deterministic_vector_sign_flip(emb)    (D-07)
   ── drop trivial ≈0 (drop_first):  SE → emb[1:n_comp].T ; SC → emb[:n_comp].T
        │                                              │
        ▼ embedding_  (SE output, SPECTRAL-01)         ▼ maps (n × n_clusters)
                                                  KMeans::new(n_clusters).fit(maps)
                                                        ▼ labels_ (SC output, SPECTRAL-02)
```

### Recommended Project Structure
```text
crates/mlrs-backend/src/prims/
  laplacian.rs        # NEW PRIM-09 — register in prims/mod.rs
crates/mlrs-algos/src/cluster/
  spectral_clustering.rs   # NEW SPECTRAL-02 (cluster/ home; register in cluster/mod.rs)
crates/mlrs-algos/src/manifold/   # NEW module group OR put SE under cluster/
  spectral_embedding.rs    # NEW SPECTRAL-01
crates/mlrs-algos/src/error.rs    # extend: NSamplesExceedsMaxDim (D-06) + n_neighbors guard
crates/mlrs-py/src/estimators/
  spectral.rs         # NEW — mirror kernel.rs (any_estimator! ×2)
crates/mlrs-backend/tests/laplacian_test.rs       # standalone prim + PoolStats gate
crates/mlrs-algos/tests/spectral_embedding_test.rs
crates/mlrs-algos/tests/spectral_clustering_test.rs
scripts/gen_oracle.py # extend with gen_spectral_embedding / gen_spectral_clustering
```
**Planner's call (Discretion):** `SpectralEmbedding` likely needs `Fit` + a host `embedding_(&pool)` accessor (it is NOT `Transform` — sklearn has no out-of-sample `transform`, only `fit_transform` returning the stored `embedding_`). `SpectralClustering` = `Fit` + `PredictLabels` (reusing the i32 surface) OR just a `labels_(&pool)` accessor + `fit_predict`. **No new trait needed** — `Fit` + accessor + (for SC) `fit_predict` mirror KMeans. Whether to create a `manifold/` module or place `SpectralEmbedding` under `cluster/` is a file-organization choice; `cluster/` is the lower-friction home (no new top-level `pub mod` in lib.rs).

### Pattern 1: laplacian.rs as base-op → in-place-map (the kernel_matrix.rs idiom)
**What:** `laplacian.rs` is a thin host orchestration: affinity (input, or built here) → degree via `row_reduce` → `d_inv_sqrt`+`L` via an in-place per-element map kernel (added in `mlrs-kernels`), exactly the `kernel_matrix.rs:144-193` composition over `covariance.rs`'s scale-in-place idiom.
**When to use:** PRIM-09.
**Example (host orchestration shape, pinned to scipy `_laplacian_dense`):**
```rust
// Source: pinned to scipy.sparse.csgraph._laplacian._laplacian_dense (scipy installed here)
//   m = A.copy(); np.fill_diagonal(m, 0)
//   w = m.sum(axis=1)
//   isolated = (w == 0); w = np.where(isolated, 1, np.sqrt(w))   # this is `dd`
//   m /= w; m /= w[:, None]; m *= -1; setdiag(m, 1 - isolated)
//
// laplacian(pool, affinity A (n×n), n) -> (L: n×n device, dd: length-n device)
//   1. zero the diagonal of A          (in-place map: tid where row==col → 0)
//   2. w = row_reduce(A, n, n, Sum, Shared)            // degree, GATHER, single-owner
//   3. dd[i] = if w[i]==0 {1} else {sqrt(w[i])}        // typed-zero guard, NO F::INFINITY
//   4. L[i,j] = -A[i,j]/(dd[i]*dd[j]); then L[i,i] = if w[i]==0 {0} else {1}
//      (one SharedMemory-FREE elementwise kernel reading dd by gather)
```
**Critical:** step 4's diagonal is `1 - isolated` = `0` for a zero-degree node, NOT `1`. The typed-zero guard (`dd=1` for isolated) means off-diagonal terms `-A_ij/(1·dd_j)` are well-defined and the diagonal is forced to 0 — no `F::INFINITY`, no NaN. This is the success-criterion "no NaN/inf on zero-degree nodes" (and a zero-degree node has `A_ij=0` for all j anyway, so its whole row of L is 0).

### Pattern 2: Eig reuse — descending → ascending reversal + drop-trivial
**What:** v1 `eig` returns `w` DESCENDING and `V` column-major (`eig.rs:1-5, :179`). The Laplacian's smallest eigenvalues (closest to 0) are the LAST entries of the descending `w`. Reverse to ascending on the host, take the first `(n_components + 1)` columns (drop_first) or `n_components` (clustering, drop_first=False).
**When:** SPECTRAL-01/02 post-eig.
**Example:**
```rust
// Source: eig.rs returns (w descending, V col-major); _spectral_embedding wants smallest ascending.
// let (w_desc, v_desc) = eig(pool, &laplacian, n, None)?;   // w[0] largest .. w[n-1] smallest
// ascending order = reverse: smallest eigenvector is v_desc column (n-1), next (n-2), ...
// take m = n_components + (drop_first as usize) smallest columns into emb (m × n), row r = (n-1-r)th col of V.
// emb[r][i] = v_host[(n-1-r)*n + i]   // V col-major: column c starts at c*n
```
**Anti-pattern:** Do NOT pad/trust `n_samples > 64` to eig — reject FIRST (D-06) with a spectral-domain `AlgoError` BEFORE building the Laplacian or calling eig, so the message names the cap rather than surfacing eig's generic `PrimError::NotSquare`.

### Pattern 3: kNN-connectivity affinity (D-03)
**What:** `distance(X,X,sqrt=false)` → `top_k(.., k=n_neighbors, sqrt=false)` gives the n×k nearest indices per row. sklearn's `kneighbors_graph(include_self=True)` means **the self (distance 0) IS among the k** — so with `include_self=True`, point i's own row includes column i. Binarize to 1.0 at those (i, idx) positions, then symmetrize `A = 0.5·(A + Aᵀ)`.
**Verified behavior (run on this machine):** for 4 colinear points, `n_neighbors=2`, `include_self=True`, `mode='connectivity'`: the connectivity diagonal is 1, off-diagonal 1 for each of the k-1 other neighbors; `0.5(A+Aᵀ)` produces 0.5 for one-directional edges and 1.0 for mutual/self. (The self-loop on the diagonal is later zeroed by the Laplacian's `fill_diagonal(m,0)`, so it does not affect the embedding — but it MUST be present pre-symmetrization to match sklearn's intermediate `affinity_matrix_`.)
**Implementation note:** with `include_self=True`, the self is the nearest (distance 0, tie-break lowest index → itself when it is its own column). The simplest exact reproduction: build the n×n connectivity by setting `A[i, j]=1` for the k smallest-distance columns of row i (which includes j=i since d(i,i)=0 is the global min for that row), then `0.5(A+Aᵀ)`.

### Anti-Patterns to Avoid
- **`F::INFINITY` for `1/sqrt(0)`:** the typed-zero guard (`dd=1` for isolated nodes, diagonal forced to 0) replaces it — cpu-MLIR backend panics on `F::INFINITY` ([[cubecl-cpu-no-shared-memory]]).
- **SharedMemory in laplacian.rs:** the dense `n×n` Laplacian stays in GLOBAL memory; the degree reduction uses the existing `ReducePath::Shared` reduce kernel (which DOES use SharedMemory but is already cpu-validated — your NEW map kernel must be SharedMemory-free). No new SharedMemory kernel.
- **Edge-scatter / atomics for degree:** degree = `row_reduce(Sum)` is a GATHER (each output owns one row, reads its row's entries) — never a scatter accumulating into shared degree cells.
- **`KMeans::with_init` injection (D-10 rejects it):** sklearn `SpectralClustering` does not expose its inner KMeans init, so injection would compare against a hand-built pipeline, not the real estimator. Use `KMeans::new` and rely on fixture separation.
- **Treating `eig`'s `out`-buffer path as a Laplacian buffer-reuse opportunity:** `eig`'s `out` arg is the covariance/GEMM reuse hook (`eig.rs:96-105`); it copies `out` into the kernel's working `a_in` and the kernel only READS it. For Phase 9 you may pass the Laplacian buffer as `out` to thread it as the eig input (it is consumed/released after launch, `eig.rs:149-151`) OR simply pass `a=laplacian, out=None`. Either is correct; passing `out=Some(laplacian)` avoids one copy only if you no longer need the Laplacian afterward.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| rbf affinity | exp/distance kernel | `kernel_matrix(Rbf{gamma})` `:164` | Phase-8 keystone, already f32/f64-validated |
| pairwise distance | sq-euclidean kernel | `distance(sqrt=false)` `:79` | clamp + GEMM-expansion already correct |
| k-nearest select | argpartition kernel | `top_k` `:61` | lowest-index tie-break already sklearn-faithful |
| degree vector | scatter/atomic sum | `row_reduce(Sum)` `:180` | single-owner GATHER, no atomics |
| symmetric eig | any eigensolver | `eig` `:75` | full spectrum, MAX_DIM=64, NotConverged guard, descending sort |
| label assignment | KMeans reimpl | `KMeans::new` `:112` | kmeans++ + Lloyd + empty-cluster relocation already sklearn-matched |
| sign canonicalization | ad-hoc sign rule | reproduce `_deterministic_vector_sign_flip` (5 lines) | sklearn's exact rule (argmax abs per row → sign of that element) |
| dtype dispatch / GIL | new pyclass plumbing | `any_estimator!` `:85` + mirror `kernel.rs` | zero new binding infra |

**Key insight:** Every device kernel this phase needs already exists EXCEPT the single `d_inv_sqrt`/`L` elementwise map in `laplacian.rs`. The phase is ~90% wiring + ~10% one new SharedMemory-free map kernel.

## Pinned sklearn Formulas (the Discretion deferrals — all confirmed against sklearn 1.9.0 on this machine)

> Source files: `sklearn/manifold/_spectral_embedding.py`, `sklearn/cluster/_spectral.py`, `sklearn/metrics/pairwise.py` (`rbf_kernel`), `sklearn/utils/extmath.py` (`_deterministic_vector_sign_flip`), `scipy/sparse/csgraph/_laplacian.py` (`_laplacian_dense`). sklearn `__version__ == 1.9.0`.

### D-04 — gamma fork (CONFIRMED)
- **SpectralEmbedding rbf:** `_spectral_embedding.py:720` — `self.gamma_ = self.gamma if self.gamma is not None else 1.0 / X.shape[1]`. So `gamma=None → 1/n_features`, computed at fit. Default constructor `gamma=None` (`:643`). Empirically confirmed: at n_features=2, `gamma_ == 0.5`.
- **SpectralClustering rbf:** `_spectral.py` default `gamma=1.0` (literal, docstring `:439` + `__init__`). It passes `params["gamma"] = self.gamma` straight into `pairwise_kernels(X, metric='rbf', ...)`. NO None→1/n_features fallback for SC.
- **rbf_kernel itself** (`pairwise.py`): `K = exp(-gamma · ‖x−y‖²)`, full matrix INCLUDING the diagonal `exp(0)=1`. (The Laplacian zeroes the diagonal anyway.)
- **mlrs mapping:** SE builds `Kernel::Rbf{ gamma: gamma.unwrap_or(1.0/n_features as F) }`; SC builds `Kernel::Rbf{ gamma: 1.0 }`. `AlgoError::InvalidGamma` already exists (`error.rs:260`) for the non-finite guard.

### D-05/D-06 — eigensolver path + cap (CONFIRMED, [v2-P3] CLOSED)
- sklearn uses ARPACK shift-invert (`eigsh(L, k=n_components, sigma=-1e-5, which="LM")`, `_spectral_embedding.py:375`) to get the SMALLEST eigenpairs efficiently on LARGE sparse graphs. At v2 sizes (n≤64) this is overkill: a **dense full-spectrum Jacobi `eig` + host-slice is exact and sufficient**.
- **Empirical confirmation (run this session):** dense `scipy.linalg.eigh(L_sym)` (full spectrum) → take smallest `n_components+1` → `/dd` recovery → sign-flip → drop-first reproduces sklearn's ARPACK `SpectralEmbedding.fit_transform` to **max abs diff 8.32e-7** at n=12, n_features=2, gamma=0.5. This is within an f32-band and far inside any f64 tolerance. **[v2-P3] is confirm-and-documented: no Lanczos/shift-invert needed.**
- v1 `eig` enforces `n ≤ MAX_DIM = 64` at `eig.rs:221-229` (returns `PrimError::NotSquare` over-cap). Because the Laplacian is `n_samples × n_samples`, **`n_samples ≤ 64` is the documented v2 spectral problem-size ceiling.**
- **D-06 guard:** add `AlgoError::NSamplesExceedsMaxDim { estimator, n_samples, max: 64 }` (new variant), checked at `fit` BEFORE building affinity/Laplacian. Do NOT rely on eig's `PrimError::NotSquare` (its message says "not square", misleading for a spectral caller).

### D-07/D-08 — recovery + drop_first ORDER (PINNED, exact)
From `_spectral_embedding.py` lines 353–477. For `norm_laplacian=True`, `eigen_solver="arpack"` (the default, and the n≤64-equivalent dense path at `:447` for lobpcg short-circuit uses the SAME tail):
```
1. laplacian, dd = csgraph_laplacian(adjacency, normed=True, return_diag=True)   # :329
       dd = sqrt(degree), or 1 where degree==0  (scipy _laplacian_dense)
2. if drop_first: n_components = n_components + 1                                  # :321-322
3. (eig)  _, diffusion_map = eigsh(L, k=n_components, sigma=-1e-5, which="LM")     # :375  (smallest k)
4. embedding = diffusion_map.T[:n_components]                                      # :378  (shape k × n)
5. if norm_laplacian: embedding = embedding / dd                                   # :380-381  ← D-07 RECOVERY
6. embedding = _deterministic_vector_sign_flip(embedding)                          # :473  ← AFTER recovery
7. if drop_first: return embedding[1:n_components].T                               # :474-475  (drop trivial row 0)
   else:          return embedding[:n_components].T                                # :476-477  (keep all)
```
**Operation ORDER (load-bearing, D-07):** slice smallest → `/dd` recovery → sign-flip → drop-first. The `/dd` recovery happens BEFORE the sign-flip; the drop-first slice happens AFTER the sign-flip. `dd` is the SAME length-n vector the Laplacian returned (so `laplacian.rs` must return `dd` alongside `L`).
- **`_deterministic_vector_sign_flip(u)`** (`extmath.py`, exact 5 lines): `max_abs_rows = argmax(|u|, axis=1); signs = sign(u[row, max_abs_rows]); u *= signs[:,None]`. Each ROW (eigenvector) is flipped so its largest-magnitude element is positive. Note: this operates on the `k × n` array (eigenvectors as ROWS), before the final `.T`.
- **drop_first slice:** `embedding[1:n_components]` — row 0 (the trivial ≈0 eigenvector, constant for a connected graph) is dropped; rows `1..n_components` kept. After step 2's `n_components += 1`, this keeps exactly the user's `n_components` rows.

### D-11 — n_components default for SpectralClustering (PINNED)
From `_spectral.py SpectralClustering.fit`: `n_components = self.n_clusters if self.n_components is None else self.n_components` and `_spectral_embedding(..., n_components=n_components, drop_first=False)`. So:
- **SpectralClustering:** `n_components` default `None → n_clusters` (D-11 confirmed); `drop_first=False` → keeps ALL `n_components` eigenvectors (including the trivial one). `assign_labels='kmeans'` → `k_means(maps, n_clusters, n_init=10)`. The trivial eigenvector is KEPT for clustering (it is near-constant and contributes ~nothing to KMeans separation, but must be kept for an exact `maps` match).
- **SpectralEmbedding:** `n_components` default `2` (`:640`); `drop_first=True` (the `_spectral_embedding` default).

### Affinity diagonal handling (cross-cutting, CONFIRMED)
scipy `_laplacian_dense` does `np.fill_diagonal(m, 0)` BEFORE computing the degree `w = m.sum(axis)`. So:
- For rbf affinity (diagonal = `exp(0) = 1`): the self-similarity is excluded from the degree.
- For kNN-connectivity (diagonal = 1 from `include_self=True`): the self-loop is excluded from the degree.
`laplacian.rs` MUST zero the diagonal first, or degrees (and thus `dd`, and thus the whole embedding) will be wrong.

## Runtime State Inventory

> Greenfield additive phase (new files + new estimators). No rename/refactor/migration. **Section omitted per format rule** — but note: extending `gen_oracle.py` (`scripts/gen_oracle.py`) adds new fixtures (`.npz` blobs committed); regen needs a /tmp venv with numpy+scipy+sklearn (PEP 668, [[oracle-fixture-regen-needs-venv]]). No stored data / live-service config / OS-registered state / secrets affected.

## Common Pitfalls

### Pitfall 1: Forgetting to zero the affinity diagonal before degree
**What goes wrong:** degree includes the self-similarity (rbf: +1 per node; kNN: +1 per node), so `dd` is inflated and the embedding diverges from sklearn.
**Why:** scipy `_laplacian_dense` zeroes the diagonal first; easy to skip.
**Avoid:** make diagonal-zeroing step 1 of `laplacian.rs`. **Warning sign:** embedding off by a smooth scale factor that grows with self-weight.

### Pitfall 2: `/dd` recovery applied AFTER sign-flip, or omitted
**What goes wrong:** `embedding_` fails the value match (D-07 is "make-or-break").
**Avoid:** ORDER is slice → `/dd` → sign-flip → drop-first (pinned above). `dd` is the Laplacian's returned diagonal, NOT a fresh sqrt.

### Pitfall 3: descending-vs-ascending eigenvector confusion
**What goes wrong:** taking the LARGEST eigenvectors (top of descending `w`) instead of the SMALLEST → completely wrong embedding.
**Why:** v1 eig sorts DESCENDING; the Laplacian's informative vectors are the SMALLEST eigenvalues (closest to 0, the LAST entries of descending `w`).
**Avoid:** reverse the order; the trivial ≈0 eigenvector is `w[n-1]` (smallest), its eigenvector is `V` column `n-1`. **Warning sign:** the "trivial" vector you drop is not near-constant.

### Pitfall 4: degenerate (repeated) eigenvalues → per-vector value mismatch
**What goes wrong:** when eigenvalues are repeated, the eigenvectors within that eigenspace are only defined up to rotation (not just sign), so a per-element value match fails even with correct math.
**Avoid (D-09):** detect near-equal adjacent eigenvalues (gap below a tol); for the degenerate block, replace the value match with a **subspace test** — compare the column spaces via principal angles (e.g. `‖Q_mlrsᵀ Q_sklearn‖` singular values ≈ 1, or `subspace_distance = ‖P_mlrs − P_sklearn‖` where `P = QQᵀ` is the projector). Normal (well-separated) spectra use the sign-aligned value match. **Design fixtures with separated spectra for the primary gate; add ONE degenerate fixture to exercise the subspace path.**

### Pitfall 5: SpectralClustering exact-label flakiness from KMeans RNG
**What goes wrong:** mlrs KMeans (SplitMix64) and sklearn KMeans (MT19937, `n_init=10`) draw different inits → different labels on a borderline partition.
**Avoid (D-10):** design the oracle fixture so clusters are well-separated in the EMBEDDING space → the partition is unique → any init converges to the same labels up to permutation. Compare with `label_perm` (best-match accuracy). Mirror the Phase-5 tuned-DBSCAN fixture. **Do NOT** inject KMeans init. **Warning sign:** label match passes on some seeds and fails on others — the fixture is not separated enough.

### Pitfall 6: `n_samples > 64` reaching the device
**What goes wrong:** eig rejects with a generic `PrimError::NotSquare`, or (worse) a Laplacian larger than MAX_DIM is built first, wasting device work.
**Avoid (D-06):** reject at `fit` entry with `AlgoError::NSamplesExceedsMaxDim` BEFORE any affinity/Laplacian/eig call.

### Pitfall 7: f32 catastrophic cancellation in the Laplacian / embedding band
**What goes wrong:** f32-on-rocm `embedding_` exceeds a too-tight tolerance.
**Avoid (Discretion):** use a DOCUMENTED f32-on-rocm band (per-family, like KernelRidge/KernelDensity precedent); f64 stays strict via `skip_f64_with_log`. The 8.3e-7 empirical gap is already near f32 epsilon accumulation — set the f32 band accordingly (e.g. ~1e-4, matching the Phase-8 f32 bands). Exact labels remain the HARD gate for clustering (no band).

## Code Examples

### Reproducing the full SpectralEmbedding pipeline (verified host reference for the oracle)
```python
# Source: verified this session — reproduces sklearn 1.9.0 SpectralEmbedding(arpack) to 8.3e-7.
# This is the gen_oracle.py reference math (the oracle still stores sklearn's own embedding_).
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.extmath import _deterministic_vector_sign_flip

A = rbf_kernel(X, gamma=gamma)              # SE: gamma = 1/n_features
m = A.copy(); np.fill_diagonal(m, 0)
w = m.sum(axis=1)
mask = (w == 0); dd = np.where(mask, 1, np.sqrt(w))
L = m / dd; L = L / dd[:, None]; L *= -1
np.fill_diagonal(L, 1 - mask)               # L_sym
evals, evecs = eigh(L)                       # ASCENDING (v1 eig is DESCENDING → reverse)
emb = evecs[:, :n_components + 1].T          # smallest (n_components+1), rows = eigenvectors
emb = emb / dd                               # D^-1/2 recovery (D-07, BEFORE sign-flip)
emb = _deterministic_vector_sign_flip(emb)
embedding_ = emb[1:n_components + 1].T        # drop_first → drop trivial row 0
```

### Eig column extraction (mlrs, V is column-major)
```rust
// Source: eig.rs:179 — v_host[c*n + r] = V[r, c] (column c is contiguous, length n).
// Smallest eigenvector (ascending index 0) = descending column (n-1).
// Build emb rows for the m smallest = descending columns (n-1), (n-2), ... (n-m):
let m = n_components + drop_first as usize;
let mut emb = vec![F::from_int(0); m * n];        // m × n, row r = (r)th-smallest eigenvector
for r in 0..m {
    let col = n - 1 - r;                           // descending col index of the r-th smallest
    for i in 0..n { emb[r * n + i] = v_host[col * n + i]; }
}
// then: emb[r][i] /= dd[i]  (recovery), sign-flip per row, drop row 0 if drop_first, transpose → n × n_components.
```

## State of the Art

| Old Approach | Current Approach | When | Impact |
|--------------|------------------|------|--------|
| ARPACK shift-invert smallest-eigenpair | dense full-spectrum Jacobi `eig` + host slice | v2 (n≤64) | exact at small n; defers the iterative solver to v3 |
| `n_init=10` KMeans for robustness | `n_init=1` + separated fixture | D-10 | exact labels via fixture design, RNG-immaterial |

**Deprecated/outdated:** none for this phase. sklearn 1.9.0's `_spectral_embedding` API is stable (the `'auto'` eigen_tol was added 1.2; the formulas pinned here are unchanged across 1.6→1.9).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | sklearn 1.9.0 formulas match the project's `scikit-learn >= 1.6` oracle target (the pinned `_spectral_embedding`/`_spectral`/`_laplacian_dense` are byte-identical 1.6→1.9) | D-04/07/08/11 | LOW — these functions are stable; if the oracle pins a different patch, re-run the 3 source dumps. Tagged because pins were read from 1.9.0, not 1.6. |
| A2 | The dense `eigh`-vs-ARPACK 8.3e-7 gap generalizes to the f32 device path within the documented band | D-05, Pitfall 7 | LOW — Jacobi `eig` is already 1e-5-validated v1; the residual is ARPACK-vs-dense, not backend. Planner should still set the f32 band empirically from the first oracle run. |
| A3 | `include_self=True` self-loop being zeroed by the Laplacian means mlrs may skip materializing the diagonal pre-symmetrization without affecting `embedding_` | D-03, Pitfall 1 | MEDIUM — TRUE for the final embedding, but `affinity_matrix_` would differ if ever exposed. mlrs does not expose `affinity_matrix_`, so safe; flagged so the planner does not also try to value-match an intermediate affinity. |

**Everything else in this RESEARCH is `[VERIFIED]` (source read on-machine) or `[CITED]` (sklearn/scipy file:line).**

## Open Questions (RESOLVED)

> Both questions are resolved inline by recommendation; the Phase-9 plans implement
> these resolutions exactly (module home → `cluster/`; `laplacian.rs` RECEIVES affinity).

1. **Module home for `SpectralEmbedding` (`manifold/` vs `cluster/`)**
   - Known: lib.rs currently has no `manifold` module; adding one touches `lib.rs` (the file-disjoint Wave-0 scaffold owns lib.rs edits).
   - Unclear: whether the roadmap wants a `manifold/` group for future Isomap/TSNE/UMAP.
   - Recommendation: place `SpectralEmbedding` under `cluster/` (alongside SpectralClustering) for Phase 9 to keep the Wave-0 scaffold minimal; defer a `manifold/` split to whenever a second manifold estimator lands. Planner's call.

2. **Whether `laplacian.rs` builds the affinity or receives it**
   - Known: rbf affinity = `kernel_matrix`, kNN affinity = `distance`+`topk`+host. Both are estimator-level choices (D-01 per-estimator default).
   - Recommendation: `laplacian.rs` RECEIVES a ready affinity matrix `A (n×n)` and returns `(L, dd)`. The estimator builds the affinity (selecting rbf vs kNN) and passes it in — keeps `laplacian.rs` affinity-agnostic and standalone-testable (success criterion 1).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust + cargo + cubecl stack | all | ✓ | workspace-pinned | — |
| cpu backend (`--features cpu`) | f64 gate + primary correctness | ✓ | — | — |
| rocm backend (`--features rocm`) | f32 GPU gate | ✓ (gfx1100/ROCm 7.1.1) | — | f64 skips-with-log on rocm ([[rocm-is-runnable-gpu-gate]]) |
| python3 + numpy + scipy + sklearn (oracle regen) | `gen_oracle.py` fixture generation | ✓ (sklearn 1.9.0 at `~/.local`) | 1.9.0 | /tmp venv (PEP 668, [[oracle-fixture-regen-needs-venv]]) — but committed `.npz` blobs mean CI never regenerates |

**Missing dependencies with no fallback:** none. **Missing dependencies with fallback:** none blocking — oracle fixtures are committed blobs; regen is a build-time dev step only.

## Validation Architecture

> nyquist_validation is ENABLED (`config.json workflow.nyquist_validation: true`).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (integration tests under `crates/*/tests/`, AGENTS.md §2 — never in-source `mod tests`) |
| Config file | none (cargo standard) |
| Quick run command | `cargo test --features cpu -p mlrs-backend laplacian_test` (targeted) |
| Full suite command | `cargo test --features cpu` (BUT [[backend-test-suite-slow]] ~6min + [[full-cargo-test-exhausts-disk]] — prefer targeted gates, background the full run) |
| Oracle harness | committed `.npz` blobs via `mlrs_core::oracle::load_npz`; regen `scripts/gen_oracle.py` in /tmp venv |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRIM-09 | Laplacian standalone: `L = I − D^-1/2 A D^-1/2` vs host reference, f32+f64 | unit (oracle) | `cargo test --features cpu -p mlrs-backend laplacian_test` | ❌ Wave 0 |
| PRIM-09 | No NaN/inf on zero-degree node (isolated-node fixture) | unit | `cargo test --features cpu -p mlrs-backend laplacian_test::zero_degree` | ❌ Wave 0 |
| PRIM-09 | PoolStats memory gate (reuse bounded, no mid-pipeline readback) | unit | `cargo test --features cpu -p mlrs-backend laplacian_test::memory_gate` | ❌ Wave 0 (mirror `memory_gate_test.rs`) |
| SPECTRAL-01 | `embedding_` value-match (rbf) after sign align, f64 strict / f32 band | unit (oracle) | `cargo test --features cpu -p mlrs-algos spectral_embedding_test` | ❌ Wave 0 |
| SPECTRAL-01 | `embedding_` value-match (nearest_neighbors default) | unit (oracle) | `cargo test --features cpu -p mlrs-algos spectral_embedding_test::knn_affinity` | ❌ Wave 0 |
| SPECTRAL-01 | degenerate-spectrum subspace test (D-09) | unit | `cargo test --features cpu -p mlrs-algos spectral_embedding_test::subspace` | ❌ Wave 0 |
| SPECTRAL-01 | `n_samples > 64` → typed `AlgoError` BEFORE device (D-06) | unit | `cargo test --features cpu -p mlrs-algos spectral_embedding_test::reject_oversize` | ❌ Wave 0 |
| SPECTRAL-02 | `labels_` match up to permutation, well-separated fixture (D-10) | unit (oracle) | `cargo test --features cpu -p mlrs-algos spectral_clustering_test` | ❌ Wave 0 |
| PY-06 (share) | PyO3 smoke: fit + `embedding_`/`labels_` accessors, f32+f64 | smoke | `cargo test -p mlrs-py spectral` (or maturin smoke) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** targeted `cargo test --features cpu -p <crate> <test_name>` (sub-30s for laplacian/spectral oracle cases).
- **Per wave merge:** the phase's own test files green on cpu (f32+f64); rocm f32 opportunistic.
- **Phase gate:** all Phase-9 tests green on cpu(f64)+rocm(f32) before `/gsd-verify-work`; f64-on-rocm skips-with-log.

### Wave 0 Gaps
- [ ] `crates/mlrs-backend/tests/laplacian_test.rs` — covers PRIM-09 (value + zero-degree + memory gate)
- [ ] `crates/mlrs-algos/tests/spectral_embedding_test.rs` — covers SPECTRAL-01 (rbf + knn + subspace + reject-oversize)
- [ ] `crates/mlrs-algos/tests/spectral_clustering_test.rs` — covers SPECTRAL-02 (label_perm)
- [ ] `crates/mlrs-py/tests/` spectral smoke — covers PY-06 share
- [ ] `prims/laplacian.rs` compiling stub (signature + geometry validation real, compute `todo!()`) + `mlrs-kernels` map-kernel stub — mirror the 08-01 `kernel_matrix.rs` Wave-0 scaffold
- [ ] `cluster/spectral_*.rs` + `estimators/spectral.rs` module homes registered (empty compiling stubs)
- [ ] `AlgoError::NSamplesExceedsMaxDim` variant + (optional) `n_neighbors` guard added to `error.rs`
- [ ] `scripts/gen_oracle.py` extended with `gen_spectral_embedding` / `gen_spectral_clustering` (committed `.npz` blobs)

## Security Domain

> `security_enforcement: true`, `security_asvs_level: 1`.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (offline numeric library) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | **yes** | Validate-before-launch at the host→estimator boundary: reject `n_samples > 64` (D-06, new `AlgoError`), `n_clusters` 1..=n_samples (`InvalidK` exists), `n_neighbors ≥ 1` (new/reuse), non-finite gamma (`InvalidGamma` exists) — ALL before any device allocation/launch (the established Phase-5/7/8 pre-allocation discipline). |
| V6 Cryptography | no | — (no secrets; the only RNG is KMeans kmeans++ SplitMix64, already non-`OsRng` per ASVS V6 from PRIM-06) |

### Known Threat Patterns for {Rust/CubeCL device kernels, host→estimator hyperparameters}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Untrusted `n_samples`/`n_neighbors`/`n_clusters` → OOB device gather | Tampering / DoS | Typed `AlgoError` validation BEFORE any launch (D-06; mirrors `kmeans.rs:238`, `topk.rs:74`) |
| `1/sqrt(0)` on zero-degree node → `F::INFINITY`/NaN propagation | Tampering (silent wrong result) + cpu-MLIR panic | Typed-zero guard `dd = where(w==0,1,sqrt(w))`, diagonal=0 for isolated nodes; NO `F::INFINITY` |
| f32 catastrophic cancellation in Laplacian/eig → wrong embedding | Tampering | Documented f32-on-rocm band; f64 strict via `skip_f64_with_log`; exact labels hard gate for SC |
| `n×n` Laplacian overflowing LDS on a tiled kernel | DoS | Dense Laplacian stays in GLOBAL memory; no SharedMemory tile (gfx1100 LDS ≤ 65536 B); LDS-budget audit (there should be none) |

## Sources

### Primary (HIGH confidence — read on-machine this session)
- `sklearn/manifold/_spectral_embedding.py` (sklearn 1.9.0) — `_spectral_embedding` lines 295–477, `SpectralEmbedding` `__init__` 638–657, `_get_affinity_matrix` 668–724 (gamma, kNN, rbf)
- `sklearn/cluster/_spectral.py` (sklearn 1.9.0) — `SpectralClustering.fit` (n_components default, drop_first=False, k_means call), `__init__` defaults (gamma=1.0)
- `scipy/sparse/csgraph/_laplacian.py` — `_laplacian_dense` (exact normalized-Laplacian + `dd` formula)
- `sklearn/utils/extmath.py` — `_deterministic_vector_sign_flip` (exact 5-line rule)
- `sklearn/metrics/pairwise.py` — `rbf_kernel` (gamma=None→1/n_features, full-matrix incl. diagonal)
- mlrs codebase (file:line throughout): `eig.rs:75/179/221`, `kernel_matrix.rs:121/164`, `reduce.rs:180`, `distance.rs:79`, `topk.rs:61`, `kmeans.rs:112`, `nearest.rs:77`, `dispatch.rs:85`, `kernel.rs` (PyO3 mirror), `error.rs:260`, `memory_gate_test.rs`
- Empirical confirmation: dense full-spectrum eigh → recovery → sign-flip → drop-first reproduces sklearn ARPACK `SpectralEmbedding` to **8.32e-7** (script run this session)
- `kneighbors_graph(include_self=True, mode='connectivity')` + `0.5(A+Aᵀ)` behavior verified on a 4-point fixture this session

### Secondary (MEDIUM)
- none needed — all claims pinned to primary source.

### Tertiary (LOW)
- none.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all reuse seams are validated v1/Phase-8 code, cited file:line.
- Architecture (laplacian.rs shape, recovery order): HIGH — pinned to scipy `_laplacian_dense` + sklearn `_spectral_embedding` line-by-line and empirically reproduced to 8.3e-7.
- Pitfalls: HIGH — derived directly from the pinned source order + project memory ([[cubecl-cpu-no-shared-memory]], [[rocm-is-runnable-gpu-gate]], [[oracle-fixture-regen-needs-venv]]).
- `[v2-P3]` resolution: HIGH — confirm-and-documented with empirical evidence.

**Research date:** 2026-06-21
**Valid until:** 2026-07-21 (30 days — sklearn spectral API is stable; re-verify only if the oracle target sklearn version moves off the 1.6–1.9 line)

## RESEARCH COMPLETE
