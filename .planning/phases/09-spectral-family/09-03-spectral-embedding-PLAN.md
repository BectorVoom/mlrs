---
phase: 09-spectral-family
plan: 03
type: execute
wave: 2
depends_on: ["09-02"]
files_modified:
  - crates/mlrs-algos/src/cluster/spectral_embedding.rs
  - crates/mlrs-algos/tests/spectral_embedding_test.rs
autonomous: true
requirements: [SPECTRAL-01]
must_haves:
  truths:
    - "SpectralEmbedding.fit(X) builds affinity → laplacian → eig → /dd recovery → sign-flip → drop-trivial and stores embedding_ (n × n_components)"
    - "embedding_ value-matches sklearn within tolerance after sign alignment for the rbf affinity (f64 strict 1e-5, f32 documented band)"
    - "embedding_ value-matches sklearn for the DEFAULT nearest_neighbors affinity (D-01) built via distance+topk binarize+symmetrize"
    - "gamma=None resolves to 1/n_features at fit (D-04); an explicit gamma is used as-is"
    - "The /dd recovery happens BEFORE the deterministic sign flip; drop-first happens AFTER (exact sklearn order, D-07/D-08)"
    - "A degenerate (repeated-eigenvalue) spectrum passes a subspace test (principal angles), not a per-vector value match (D-09)"
    - "n_samples > 64 is rejected with AlgoError::NSamplesExceedsMaxDim BEFORE any affinity/Laplacian/eig device work (D-06)"
  artifacts:
    - path: "crates/mlrs-algos/src/cluster/spectral_embedding.rs"
      provides: "SpectralEmbedding Fit + embedding_ host accessor + kNN-connectivity affinity builder"
      contains: "fn fit"
    - path: "crates/mlrs-algos/tests/spectral_embedding_test.rs"
      provides: "rbf + knn_affinity + subspace + reject_oversize tests (un-ignored)"
      contains: "reject_oversize"
  key_links:
    - from: "crates/mlrs-algos/src/cluster/spectral_embedding.rs"
      to: "mlrs_backend::prims::laplacian::laplacian"
      via: "(L, dd) = laplacian(A, n)"
      pattern: "laplacian"
    - from: "crates/mlrs-algos/src/cluster/spectral_embedding.rs"
      to: "mlrs_backend::prims::eig::eig"
      via: "eig(L) → descending spectrum, reverse to ascending"
      pattern: "eig"
    - from: "crates/mlrs-algos/src/cluster/spectral_embedding.rs"
      to: "mlrs_backend::prims::kernel_matrix"
      via: "rbf affinity = kernel_matrix(X,X,Rbf)"
      pattern: "kernel_matrix"
---

<objective>
SPECTRAL-01: implement `SpectralEmbedding` on the validated laplacian prim + v1
eig, value-matching sklearn `embedding_`.

Pipeline (RESEARCH System Diagram + pinned _spectral_embedding order): affinity
(rbf via `kernel_matrix(Rbf)` D-02, OR the new sklearn-exact kNN-connectivity
builder D-03 — DEFAULT is nearest_neighbors per D-01) → `laplacian` (L, dd) → v1
`eig` (returns DESCENDING; reverse to ascending, take the smallest n_components+1
columns) → `/dd` recovery (D-07, BEFORE sign-flip) → `_deterministic_vector_sign_flip`
→ drop the trivial ≈0 row (drop_first, D-08) → transpose → `embedding_` (n × n_components).

The exact operation ORDER is load-bearing (RESEARCH §D-07/D-08, Pitfall 2):
slice smallest → `/dd` → sign-flip → drop-first. `dd` is the SAME vector the
laplacian returned.

Purpose: a sklearn-faithful spectral embedding gated by the committed oracle.
Output: SpectralEmbedding estimator + four green tests (rbf, knn, subspace, reject).
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/09-spectral-family/09-RESEARCH.md
@.planning/phases/09-spectral-family/09-PATTERNS.md
@.planning/phases/09-spectral-family/09-VALIDATION.md
@AGENTS.md

# Analogs (READ before editing):
@crates/mlrs-algos/src/cluster/kmeans.rs
@crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs
@crates/mlrs-backend/src/prims/eig.rs
@crates/mlrs-backend/src/prims/kernel_matrix.rs
@crates/mlrs-backend/src/prims/distance.rs
@crates/mlrs-backend/src/prims/topk.rs
@crates/mlrs-algos/tests/kernel_ridge_test.rs

# Phase 9 prim (READ — consumed here):
@crates/mlrs-backend/src/prims/laplacian.rs
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Affinity builders (rbf + kNN-connectivity) + validate-before-launch guards</name>
  <read_first>
    - crates/mlrs-backend/src/prims/kernel_matrix.rs (:164 Rbf arm; pass y=x, (n,d),(n,d))
    - crates/mlrs-backend/src/prims/distance.rs (:79 sqrt=false) + topk.rs (:61 k smallest + indices, lowest-index tie-break)
    - crates/mlrs-algos/src/cluster/kmeans.rs (:234-252 validate-before-launch block)
    - crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs (:64-66 gamma=None→1/n_features at fit)
    - RESEARCH Pattern 3 (kNN-connectivity: include_self=True, mode=connectivity, 0.5(A+Aᵀ)) + D-04 gamma fork
  </read_first>
  <behavior>
    - validate-before-launch: reject n_samples > 64 → AlgoError::NSamplesExceedsMaxDim{estimator:"SpectralEmbedding", n_samples, max:64} BEFORE any device work (D-06); reject n_neighbors < 1 (InvalidK reuse) and non-finite resolved gamma (InvalidGamma).
    - rbf affinity: A = kernel_matrix(X, X, Kernel::Rbf{ gamma: gamma.unwrap_or(1.0/n_features) }) (D-02/D-04). Full matrix incl. diagonal exp(0)=1 (laplacian zeroes it).
    - kNN-connectivity affinity (DEFAULT, D-03): distance(X,X,sqrt=false) → top_k(k=n_neighbors, sqrt=false) → set A[i,j]=1 for the k smallest-distance columns of row i (includes self, d(i,i)=0 is the row min) → A = 0.5·(A + Aᵀ). Binary weights 0/1, NOT distance weights.
  </behavior>
  <action>
    Implement the affinity selection + guards in spectral_embedding.rs (the fit prologue).
    Copy the kmeans.rs validate-before-launch block: reject n_samples > 64 with the NEW
    AlgoError::NSamplesExceedsMaxDim (D-06 — fail loud naming the cap, do NOT defer to eig's
    PrimError::NotSquare); reuse InvalidK for n_neighbors<1; resolve gamma=None→1/n_features
    at fit (copy kernel_ridge's at-fit resolution, D-04) and validate the resolved gamma is
    finite via InvalidGamma.

    Build the affinity by `affinity` string:
    - "rbf": kernel_matrix(X, X, Kernel::Rbf{gamma_resolved}) (D-02).
    - "nearest_neighbors" (DEFAULT): the new sklearn-exact binary connectivity builder
      (D-03, RESEARCH Pattern 3) — distance(sqrt=false) → top_k(n_neighbors) → binarize 0/1
      at the k smallest columns per row (self included) → symmetrize 0.5(A+Aᵀ). The binarize +
      symmetrize is small host math on the n×k indices.
    Any other affinity string → a typed error (out of scope per CONTEXT; precomputed deferred).
  </action>
  <verify>
    <automated>cargo test --features cpu -p mlrs-algos spectral_embedding_test::reject_oversize 2>&1 | tail -4</automated>
  </verify>
  <acceptance_criteria>
    - reject_oversize: n_samples > 64 raises AlgoError::NSamplesExceedsMaxDim BEFORE any device allocation (assert the error variant, no device work observed).
    - rbf builds Kernel::Rbf with gamma=1/n_features when gamma=None; kNN builds a 0/1 symmetric connectivity matrix.
  </acceptance_criteria>
  <done>Both affinity builders work; the n_samples>64 guard fires pre-launch with the spectral-domain error; gamma=None→1/n_features resolved at fit.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Post-eig recovery (slice→/dd→sign-flip→drop) + embedding_ accessor + oracle tests</name>
  <read_first>
    - crates/mlrs-backend/src/prims/eig.rs (:75 returns w DESCENDING, V col-major v_host[c*n+r]; :221 MAX_DIM=64)
    - RESEARCH "Eig column extraction" snippet + the pinned _spectral_embedding order (slice asc → /dd → _deterministic_vector_sign_flip → drop_first) + Code Examples host reference
    - crates/mlrs-algos/tests/kernel_ridge_test.rs (assert_close strict-1e-5-floor + *_F32_BAND + skip_f64_with_log)
    - The committed spectral_embedding (rbf + nn default) and degenerate .npz fixtures (Wave 0)
  </read_first>
  <behavior>
    - fit completes the pipeline: (L,dd)=laplacian(A,n); (w_desc,V_desc)=eig(L,n,None); reverse desc→asc; take m = n_components + (drop_first=1) smallest columns into a k×n host array (row r = descending column n-1-r); emb[r][i] /= dd[i] (recovery, D-07); _deterministic_vector_sign_flip per row (argmax|row|→sign); drop row 0; transpose → embedding_ (n × n_components). Store embedding_ device-resident; host accessor embedding_(&pool) → Vec<F>.
    - spectral_embedding (rbf) test: embedding_ matches sklearn within tolerance after sign alignment (f64 strict 1e-5, f32 band). knn_affinity test: same for the nearest_neighbors DEFAULT.
    - subspace test (D-09): a degenerate-spectrum fixture passes a principal-angles subspace test (column-space match) instead of per-vector value match.
  </behavior>
  <action>
    Implement the post-eig recovery host math (replace the Wave-0 todo!()), reproducing the
    pinned sklearn _spectral_embedding order EXACTLY (RESEARCH §D-07/D-08):
    1. (w_desc, V_desc) = eig(pool, &L, n, None) — DESCENDING; the Laplacian's smallest
       eigenvalues (informative) are the LAST descending entries (Pitfall 3).
    2. Reverse to ascending: the r-th smallest eigenvector is descending column (n-1-r).
       Extract m = n_components + 1 (drop_first=TRUE for SE) smallest columns into a k×n
       host array using the V col-major layout v_host[col*n + i] (RESEARCH Eig column snippet).
    3. emb[r][i] /= dd[i] — the D^-1/2 recovery, BEFORE the sign-flip (D-07, make-or-break).
    4. _deterministic_vector_sign_flip on the k×n array (per ROW: argmax|row| → sign of that
       element → multiply row) — reproduce sklearn extmath exactly (5 lines).
    5. Drop the trivial row 0 (drop_first): keep rows 1..n_components; transpose → embedding_
       (n × n_components). Store device-resident; add the embedding_(&pool) host accessor.

    You MAY pass the Laplacian buffer as eig's `out` to thread it as the eig input (it is
    consumed/released after launch) OR pass out=None — either is correct (RESEARCH Anti-Pattern).

    Then un-ignore + implement the four tests: spectral_embedding (rbf), knn_affinity
    (nearest_neighbors default), subspace (D-09 principal-angles for the degenerate fixture),
    reject_oversize (already covered by Task 1 — keep it green). Use assert_close with the
    strict 1e-5 absolute floor + a documented *_F32_BAND (~1e-4, Pitfall 7); f64 strict via
    skip_f64_with_log. For the subspace test, compute principal angles (e.g. singular values
    of Q_mlrsᵀ Q_sklearn ≈ 1, or ‖P_mlrs − P_sklearn‖ with P=QQᵀ) instead of element compare.
  </action>
  <verify>
    <automated>cargo test --features cpu -p mlrs-algos spectral_embedding_test 2>&1 | tail -8</automated>
  </verify>
  <acceptance_criteria>
    - All four spectral_embedding_test cases green on cpu (f32+f64; f64 strict 1e-5, f32 band).
    - The /dd recovery is applied BEFORE the sign-flip and the drop-first AFTER (order verified by the value match — a wrong order fails).
    - The degenerate fixture passes the subspace test, not the per-vector value match.
  </acceptance_criteria>
  <done>SpectralEmbedding embedding_ value-matches sklearn (rbf + nn default) after sign alignment, the degenerate spectrum passes the subspace test, and the pipeline order is pinned-exact.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host → SpectralEmbedding.fit | Untrusted n_samples / n_neighbors / gamma cross here |
| eig output → recovery host math | Eigenvector ordering (descending) must be reversed correctly |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-9-VAL | Tampering/DoS | SpectralEmbedding.fit entry | mitigate | Reject `n_samples > 64` with `AlgoError::NSamplesExceedsMaxDim` (D-06) BEFORE any affinity/Laplacian/eig device allocation; reuse `InvalidK` (n_neighbors≥1) and `InvalidGamma` (non-finite resolved gamma). Verified by `reject_oversize`. Mirrors `kmeans.rs:238` pre-allocation discipline. |
| T-9-F32 | Tampering (wrong embedding) | f32 catastrophic cancellation in Laplacian/eig | mitigate | Documented f32-on-rocm band (~1e-4, Pitfall 7); f64 strict via `skip_f64_with_log`. The 8.3e-7 empirical dense-vs-ARPACK gap is inside the band. |
| T-9-DEG | Tampering (false mismatch) | degenerate repeated eigenvalues | accept-with-control | Repeated eigenvalues are defined only up to rotation; a per-vector value match would false-fail correct math. Replaced by a subspace (principal-angles) test for the degenerate fixture (D-09); normal spectra keep the sign-aligned value match. |
</threat_model>

<verification>
- `cargo test --features cpu -p mlrs-algos spectral_embedding_test` green (rbf, knn,
  subspace, reject_oversize) on cpu f32+f64.
- reject_oversize asserts the typed AlgoError fires before device work.
- The recovery order (slice→/dd→sign-flip→drop) is implicitly verified by the value match.
</verification>

<success_criteria>
- SPECTRAL-01: embedding_ matches sklearn within tolerance after sign alignment for BOTH
  affinities (rbf + nearest_neighbors default) — f64 strict, f32 banded.
- Degenerate spectra pass the subspace test (D-09).
- n_samples > 64 rejected pre-launch with the spectral-domain typed error (D-06).
- gamma=None → 1/n_features resolved at fit (D-04).
</success_criteria>

<output>
Create `.planning/phases/09-spectral-family/09-03-SUMMARY.md` when done.
</output>
