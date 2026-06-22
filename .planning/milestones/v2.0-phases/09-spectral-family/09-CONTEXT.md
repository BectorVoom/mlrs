# Phase 9: Spectral Family - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver spectral embedding and clustering assembled on v1's hardest-won prim
(`eig`) plus v1 KMeans, with one new graph-Laplacian device primitive. The graph
affinity *is* `kernel_matrix(Rbf)` from Phase 8 (HARD DEPENDENCY — the phase
order is mandatory). Scope is fixed by ROADMAP.md Phase 9 success criteria and
REQUIREMENTS PRIM-09, SPECTRAL-01, SPECTRAL-02.

**In scope:**
- `prims/laplacian.rs` — normalized graph Laplacian: affinity → single-owner
  row-reduction degree → `d_inv_sqrt` with a typed-zero guard (**NO
  `F::INFINITY`**, no edge-scatter / no atomics, GATHER degree-normalization).
  Validated standalone with no NaN/inf on zero-degree nodes, with its
  build-failing PoolStats memory gate. (PRIM-09)
- `SpectralEmbedding` — affinity (rbf via `kernel_matrix(Rbf)` OR sklearn-exact
  kNN-connectivity graph) → normalized Laplacian → **smallest** non-trivial
  eigenvectors via v1 `eig`, sorted ascending, `D^-1/2` recovery, deterministic
  `_deterministic_vector_sign_flip`, dropping the trivial ≈0 eigenvector.
  `embedding_` matches scikit-learn within tolerance after sign alignment
  (subspace test for degenerate spectra). (SPECTRAL-01)
- `SpectralClustering` — spectral embedding → v1 KMeans; `labels_` matches
  scikit-learn up to label permutation (the **exact-labels** hard gate, sign-
  immune via `label_perm`). (SPECTRAL-02)
- A **new sklearn-exact kNN affinity builder** (binary connectivity graph,
  symmetrized `0.5·(A + Aᵀ)`) carried in to support `affinity='nearest_neighbors'`
  — see D-03 (scope note: this is a small new capability beyond the bare
  `kernel_matrix(Rbf)` seam, accepted deliberately for sklearn-default parity).

**Out of scope (deferred / other phases):**
- `affinity='precomputed'` / `'precomputed_nearest_neighbors'` — not selected.
- `assign_labels='discretize'` / `'cluster_qr'` — kmeans-only label assignment.
- Lanczos / shift-invert smallest-eigenpair solvers — the n≤64 eig cap makes
  full-spectrum-then-slice exact and sufficient ([v2-P3] closed, D-05).
- `n_samples > 64` (above v1 `eig` MAX_DIM) — hard-rejected as a typed error.

</domain>

<decisions>
## Implementation Decisions

### Affinity scope, defaults & gamma
- **D-01: Mirror sklearn per-estimator affinity defaults.** `SpectralEmbedding`
  default affinity = `'nearest_neighbors'`; `SpectralClustering` default
  affinity = `'rbf'`. Both affinity builders must work for both estimators. The
  oracle can use each estimator's own default constructor with NO override (the
  opposite of Phase-7's batch_size injection — here the defaults are honored).
- **D-02: rbf affinity = `kernel_matrix(Rbf)`** from Phase 8 (the keystone seam
  this phase was ordered to cash in). No re-derivation.
- **D-03: nearest_neighbors affinity = sklearn-exact binary connectivity graph.**
  Build a kNN graph with `mode='connectivity'` (weights 0/1, NOT distance
  weights), `n_neighbors` default `10`, symmetrized via `0.5·(A + Aᵀ)`. This
  reproduces sklearn's `nearest_neighbors` affinity so `embedding_` can value-
  match. Reuse v1 neighbors top-k (`prims/topk.rs` + `prims/distance.rs`) for
  the k nearest, then binarize + symmetrize. **Scope note:** this kNN-graph
  builder is a small new capability beyond the bare `kernel_matrix(Rbf)` seam
  ROADMAP scoped; included deliberately so SpectralEmbedding matches sklearn's
  default-constructor behavior.
- **D-04: gamma mirrors each estimator exactly.** `SpectralEmbedding`:
  `gamma=None → 1/n_features` (computed at fit, like KernelRidge D-05).
  `SpectralClustering`: `gamma` default `1.0` (literal). Pin the exact
  `None→value` formula and per-estimator defaults from sklearn source during
  planning; both gamma paths value-pinned in the oracle.

### Eigensolver path & problem-size cap ([v2-P3] research flag)
- **D-05: Full-spectrum-then-slice, hard cap `n_samples ≤ 64`.** Use v1 `eig`
  as-is — it computes the full symmetric spectrum (cyclic Jacobi, sorted
  descending on host) and caps `n ≤ MAX_DIM = 64`. The Laplacian is
  `n_samples × n_samples`, so **`n_samples` is forced ≤ 64**. Reverse to
  ascending and slice the smallest non-trivial eigenvectors. **No Lanczos /
  shift-invert** — pointless at n≤64 where dense Jacobi is exact and already
  validated. This is the documented v2 problem-size cap; [v2-P3] is effectively
  pre-answered, so the research spike confirms-and-documents rather than
  investigates a new solver.
- **D-06: Reject `n_samples > 64` as a typed `AlgoError` BEFORE any device
  work** (ASVS-V5, mirrors the Phase-7/8 pre-allocation guard discipline). Don't
  defer to `eig`'s internal `PrimError` — fail loud with a spectral-domain
  message naming the MAX_DIM cap.

### Embedding recovery (post-eig math → `embedding_` value match)
- **D-07: Reproduce the `D^-1/2` diffusion-map recovery exactly.** For the
  normalized Laplacian (`norm_laplacian=True` path), divide each recovered
  eigenvector by `dd = sqrt(degree)` BEFORE the deterministic sign flip —
  required for `embedding_` to value-match sklearn within tolerance. Pin the
  exact operation ORDER (slice ascending → `/dd` recovery → deterministic sign
  flip → drop trivial ≈0) from sklearn `_spectral_embedding.py` during planning.
- **D-08: `n_components` default `2`; mirror sklearn drop_first.** Compute the
  smallest `(n_components + 1)` eigenvectors, drop the trivial ≈0 one, keep
  `n_components`. Reproduces sklearn's dimensionality and drop_first slice
  indices (pin exact indices from source).
- **D-09: Degenerate spectra → subspace test.** For value-ambiguous (repeated)
  eigenvalues the per-vector value match is replaced by a subspace test (ROADMAP
  gate); normal spectra use the value match after sign alignment.

### Exact-label reproduction (SpectralClustering hard gate)
- **D-10: Well-separated oracle fixture → init-invariant labels.** Design the
  SpectralClustering oracle data so clusters are well-separated in the embedding
  → the partition is UNIQUE → any KMeans (any init/RNG) converges to the same
  labels up to permutation. Reuse v1 KMeans as-is (default kmeans++, `n_init=1`);
  the `label_perm` gate passes regardless of the SplitMix64-vs-MT19937 RNG
  difference. Mirrors the Phase-5 DBSCAN tuned-fixture design (cluster
  separation chosen so the answer is unambiguous), NOT the Phase-5 KMeans
  init-injection (rejected here — sklearn `SpectralClustering` doesn't expose its
  inner KMeans init, so injection would force comparing against a hand-built
  `spectral_embedding + KMeans(init=…)` pipeline instead of the actual
  `SpectralClustering` estimator, weakening "oracle = the estimator").
- **D-11: `n_components = n_clusters`, `assign_labels='kmeans'`-only.** The
  embedding dimension defaults to `n_clusters` (sklearn default); kmeans is the
  ONLY label-assignment path. `'discretize'`/`'cluster_qr'` are out of scope
  (deferred). Pin the `n_components`-vs-`n_clusters` default from sklearn source.

### Claude's Discretion
- Exact f32-on-rocm tolerance band for `SpectralEmbedding` `embedding_`
  (embedding band + sign, or the subspace test for degenerate spectra) — follow
  the v1 per-family documented-band precedent; f64 stays strict (gated by
  `skip_f64_with_log`). **Exact labels** is the hard gate for
  `SpectralClustering` (no band — labels match or they don't).
- The precise `laplacian.rs` degree-reduction kernel shape (single-owner row
  reduction; GATHER not scatter; typed-zero guard for zero-degree nodes — NO
  `F::INFINITY`, no atomics, SharedMemory-free per [[cubecl-cpu-no-shared-memory]]).
- Whether `SpectralEmbedding`/`SpectralClustering` need a new trait or compose
  on existing `Fit` + `Transform` (`embedding_`) / `PredictLabels` (`labels_`) —
  planner's call (likely NO new trait, unlike Phase 7/8).
- Exact sklearn-source-pinned formulas: gamma `None→value` per estimator (D-04),
  the `_spectral_embedding` recovery/drop_first slice order (D-07/D-08), the
  `n_components` default (D-11).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` — Phase 9 "Spectral Family" goal, success criteria,
  recurring gates, and the `[v2-P3]` research flag (smallest-eigenpair
  extraction + document the n_samples cap — pre-answered here as n≤64).
- `.planning/REQUIREMENTS.md` — PRIM-09, SPECTRAL-01, SPECTRAL-02 (exact wording,
  incl. "GATHER degree-normalization, no atomics" and "within tolerance after
  sign alignment" vs "up to label permutation").
- `.planning/PROJECT.md` — milestone v2.0 goal, constraints, Key Decisions
  (oracle = sklearn ≤1e-5, gate = cpu(f64)+rocm(f32), primitive-first).
- `.planning/research/questions.md` §[v2-P3] — the graph-Laplacian +
  smallest-eigenpairs research question this phase resolves.
- `.planning/seeds/v2-breadth-roadmap.md` — v2 family/prim mapping (Spectral
  family = graph-Laplacian prim → eig + KMeans).
- `.planning/phases/08-kernel-family/08-CONTEXT.md` — `kernel_matrix(Rbf)` seam
  (the affinity source, D-01/D-02 of Phase 8: typed `Kernel<F>` enum, general
  `K(X,Y)`, self-contained prim) + carried-forward precedents (PoolStats gate,
  f32 bands, PyO3 `any_estimator!` reuse, no-`F::INFINITY` discipline).
- `.planning/phases/07-covariance-projection/07-CONTEXT.md` — trait-addition
  shape + oracle-injection precedent (contrast: D-01 honors sklearn defaults
  rather than injecting).

### Reusable primitive & estimator code (v1 + Phase 8, validated)
- `crates/mlrs-backend/src/prims/kernel_matrix.rs` — Phase-8 keystone prim;
  `kernel_matrix(X, X, Kernel::Rbf { gamma })` IS the rbf graph affinity (D-02).
- `crates/mlrs-backend/src/prims/eig.rs` — v1 symmetric eig: full spectrum,
  TRUSTED-symmetric, sorted DESCENDING on host, **`MAX_DIM = 64` cap** (the
  n_samples cap, D-05/D-06). Has an `out`-buffer reuse path. f64 gated by
  `skip_f64_with_log`. NOTE: its `jacobi_eig` kernel uses SharedMemory but RUNS
  on cpu (eig_test.rs cpu f32+f64) — the new `laplacian.rs` must stay
  SharedMemory-free regardless (ROADMAP / [[cubecl-cpu-no-shared-memory]]).
- `crates/mlrs-backend/src/prims/distance.rs` — pairwise squared-euclidean; base
  for the rbf affinity AND the kNN-connectivity graph (D-03).
- `crates/mlrs-backend/src/prims/topk.rs` — top-k selection for the k nearest in
  the kNN-connectivity affinity builder (D-03).
- `crates/mlrs-backend/src/prims/reduce.rs` — single-owner row-reduction for the
  Laplacian degree vector (D-05 laplacian.rs, GATHER not scatter).
- `crates/mlrs-backend/src/prims/mod.rs` — register the new `laplacian` module.
- `crates/mlrs-algos/src/cluster/kmeans.rs` — v1 KMeans; `KMeans::new`
  (default kmeans++, `n_init=1`) reused as-is for SpectralClustering (D-10);
  `KMeans::with_init` exists but is NOT used here (D-10 rejects init-injection).
- `crates/mlrs-algos/src/neighbors/nearest.rs` — NearestNeighbors estimator (k
  nearest) feeding the kNN-connectivity affinity (D-03).
- `crates/mlrs-algos/src/traits.rs` — `Fit`/`Transform`/`PredictLabels`/
  `PartialFit`/`ScoreSamples` surface; likely NO new trait (planner's call).
- `crates/mlrs-algos/src/error.rs` — `AlgoError`; extend with Phase-9 guards
  (`n_samples > 64` MAX_DIM cap D-06, `n_neighbors ≥ 1`, `n_clusters ≥ 1`,
  gamma/affinity validation) — typed, pre-allocation.
- `crates/mlrs-algos/src/lib.rs` — register the new spectral estimator module
  group (e.g. under `cluster/` for SpectralClustering + a `manifold`/`cluster`
  home for SpectralEmbedding — planner's call; file-disjoint Wave-0 scaffold).
- `crates/mlrs-py/src/dispatch.rs` + `crates/mlrs-py/src/estimators/` — the
  `any_estimator!` Unfit/F32/F64 machinery; add a spectral estimator module
  (mirrors `kernel.rs`); `embedding_`/`labels_` accessors, `py.detach`,
  `guard_f64()`. PY-06's final cross-cutting sign-off remains Phase 11.
- `tests/` + `crates/*/tests/` + `gen_oracle.py` — committed-`.npz` oracle
  harness; well-separated SpectralClustering fixture per D-10
  ([[oracle-fixture-regen-needs-venv]]: regen needs a `/tmp` venv with
  numpy+scipy+sklearn, PEP 668). [[backend-test-suite-slow]] — run targeted
  `laplacian_test` / spectral gates, background the full run.

### Kernel / build guidance
- `/home/user/Documents/workspace/cubecl_manual/` — CubeCL manuals (for the new
  `laplacian.rs` degree/`d_inv_sqrt` map kernel).
- `/home/user/Documents/workspace/cintx/docs/cubecl_error_guideline.md` —
  CubeCL error guideline referenced by `AGENTS.md`.
- `AGENTS.md` — tests separated from source; consult the CubeCL error guideline
  on any build error; generics-over-float protocol.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase-8 `kernel_matrix(Rbf)`** (`prims/kernel_matrix.rs`): IS the rbf graph
  affinity — the seam this phase was ordered to cash in (D-02). Zero new affinity
  math for the rbf path.
- **v1 `eig`** (`prims/eig.rs`): full symmetric spectrum, already validated on
  cpu(f32+f64)/rocm(f32). Reused as-is; its `MAX_DIM = 64` is the n_samples cap
  (D-05). Returns DESCENDING — reverse to ascending and drop the trivial vector.
- **v1 KMeans** (`cluster/kmeans.rs`): `KMeans::new` (kmeans++, `n_init=1`,
  Lloyd) reused unchanged for SpectralClustering (D-10). Init-invariant on the
  well-separated fixture, so the SplitMix64-vs-MT19937 RNG gap is immaterial.
- **v1 neighbors top-k** (`prims/topk.rs`, `neighbors/nearest.rs`): the k
  nearest for the kNN-connectivity affinity (D-03).
- **v1 reduce** (`prims/reduce.rs`): single-owner row reduction for the degree
  vector in `laplacian.rs` (GATHER, no scatter/atomics).
- **PyO3 `any_estimator!`** (`mlrs-py/src/dispatch.rs` + `estimators/`): zero new
  binding infra; spectral estimators get the Unfit/F32/F64 enum + `embedding_`/
  `labels_` accessors (mirrors `kernel.rs`).

### Established Patterns
- **Primitive-first discipline:** land + standalone-validate `laplacian.rs`
  (no NaN/inf on zero-degree nodes, f32+f64) with its build-failing PoolStats
  memory gate BEFORE wiring the estimators (mirrors Phase 7/8 prim gating).
- **cpu-MLIR-safe kernels:** `laplacian.rs` feature-free, SharedMemory-free, no
  atomics, GATHER degree-normalization, **never `F::INFINITY`** — the typed-zero
  guard on `d_inv_sqrt` for zero-degree nodes exists specifically to avoid it
  ([[cubecl-cpu-no-shared-memory]]).
- **LDS budget:** dense `n×n` Laplacian stays in global memory; LDS-budget audit
  on any tile (there should be none; gfx1100 LDS ≤ 65536 B).
- **f64 gated by `skip_f64_with_log`** (cpu runs f64, rocm skips); documented
  f32-on-rocm band for `SpectralEmbedding` embedding ([[rocm-is-runnable-gpu-gate]]).
- **Pre-allocation typed guards:** reject `n_samples > 64`, bad `n_neighbors`/
  `n_clusters`/gamma as `AlgoError` BEFORE any device work (ASVS-V5).

### Integration Points
- `crates/mlrs-backend/src/prims/mod.rs` — register `laplacian`.
- `crates/mlrs-algos/src/lib.rs` + a spectral module group — register
  SpectralEmbedding / SpectralClustering (file-disjoint Wave-0 scaffold).
- `crates/mlrs-algos/src/error.rs` — Phase-9 hyperparameter guards.
- `crates/mlrs-py/src/estimators/` — new spectral estimator wrapper module.

</code_context>

<specifics>
## Specific Ideas

- The affinity defaults DISAGREE across the two estimators and we honor both:
  SpectralEmbedding→`nearest_neighbors`, SpectralClustering→`rbf` (D-01). This is
  the inverse of Phase-7's "inject the non-default" pattern — here sklearn's own
  default constructor is the oracle.
- gamma is a DOUBLE parity fork: `None→1/n_features` (SE) vs literal `1.0` (SC)
  (D-04) — pin both from sklearn source, both value-pinned in the oracle.
- The `D^-1/2` recovery (`/sqrt(degree)`) is the make-or-break step for the
  `embedding_` value match (D-07) — easy to omit, must reproduce exactly.
- SpectralClustering exact-label matching comes from FIXTURE DESIGN (unique
  well-separated partition, D-10), not from RNG/init matching — the v2 spectral
  analogue of Phase-5's tuned DBSCAN fixture.
- n_samples ≤ 64 is a real, documented v2 cap inherited from `eig` MAX_DIM —
  oracle fixtures must stay within it (D-05).

</specifics>

<deferred>
## Deferred Ideas

- **`affinity='precomputed'` / `'precomputed_nearest_neighbors'`** — user passes
  the n×n affinity directly; not selected for Phase 9 (rbf + nearest_neighbors
  only). Cheap future add (skip the affinity build).
- **`assign_labels='discretize'` / `'cluster_qr'`** — alternative spectral
  label-assignment methods; out of scope (kmeans-only per ROADMAP).
- **Lanczos / shift-invert smallest-eigenpair solver** — would lift the n≤64
  cap; not worth it until v2 needs n_samples ≫ 64 (a large new sparse-iterative
  kernel, contradicts the reuse-v1-eig framing). Revisit only if a future
  milestone raises the problem-size ceiling.
- None outside phase scope surfaced during discussion — stayed within Phase 9.

</deferred>

---

*Phase: 9-spectral-family*
*Context gathered: 2026-06-21*
