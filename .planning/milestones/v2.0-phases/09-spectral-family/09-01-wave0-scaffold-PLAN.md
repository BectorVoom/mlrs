---
phase: 09-spectral-family
plan: 01
type: execute
wave: 0
depends_on: []
files_modified:
  - crates/mlrs-algos/src/error.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/src/prims/laplacian.rs
  - crates/mlrs-kernels/src/elementwise.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-algos/src/cluster/mod.rs
  - crates/mlrs-algos/src/cluster/spectral_embedding.rs
  - crates/mlrs-algos/src/cluster/spectral_clustering.rs
  - crates/mlrs-py/src/estimators/mod.rs
  - crates/mlrs-py/src/estimators/spectral.rs
  - crates/mlrs-py/src/lib.rs
  - crates/mlrs-backend/tests/laplacian_test.rs
  - crates/mlrs-algos/tests/spectral_embedding_test.rs
  - crates/mlrs-algos/tests/spectral_clustering_test.rs
  - crates/mlrs-py/tests/spectral_smoke_test.rs
  - scripts/gen_oracle.py
autonomous: true
requirements: [PRIM-09, SPECTRAL-01, SPECTRAL-02]
must_haves:
  truths:
    - "Workspace compiles with --features cpu after the scaffold (all stubs in place)"
    - "AlgoError::NSamplesExceedsMaxDim exists and is constructible with a spectral-domain message naming MAX_DIM=64"
    - "laplacian prim has a real geometry-validating signature with a todo!() compute body"
    - "Five test files exist with #[ignore] Nyquist scaffolds that compile and assert fixture-load+shape only"
    - "gen_oracle.py emits committed spectral_embedding/spectral_clustering .npz fixtures using each estimator's own default constructor (D-01)"
  artifacts:
    - path: "crates/mlrs-algos/src/error.rs"
      provides: "AlgoError::NSamplesExceedsMaxDim variant (D-06)"
      contains: "NSamplesExceedsMaxDim"
    - path: "crates/mlrs-backend/src/prims/laplacian.rs"
      provides: "laplacian(pool, A, n) -> (L, dd) signature with geometry guard + todo!() compute"
      contains: "pub fn laplacian"
    - path: "crates/mlrs-kernels/src/elementwise.rs"
      provides: "laplacian_map #[cube(launch)] stub (SharedMemory-free)"
      contains: "laplacian_map"
    - path: "crates/mlrs-algos/src/cluster/spectral_embedding.rs"
      provides: "SpectralEmbedding struct + new() stub"
      contains: "pub struct SpectralEmbedding"
    - path: "crates/mlrs-algos/src/cluster/spectral_clustering.rs"
      provides: "SpectralClustering struct + new() stub"
      contains: "pub struct SpectralClustering"
    - path: "crates/mlrs-py/src/estimators/spectral.rs"
      provides: "PySpectralEmbedding/PySpectralClustering pyclass stubs"
      contains: "SpectralEmbedding"
    - path: "crates/mlrs-backend/tests/laplacian_test.rs"
      provides: "PRIM-09 #[ignore] scaffolds (value/zero_degree/memory_gate)"
      contains: "laplacian"
    - path: "crates/mlrs-algos/tests/spectral_embedding_test.rs"
      provides: "SPECTRAL-01 #[ignore] scaffolds"
      contains: "spectral_embedding"
    - path: "crates/mlrs-algos/tests/spectral_clustering_test.rs"
      provides: "SPECTRAL-02 #[ignore] scaffold"
      contains: "spectral_clustering"
    - path: "scripts/gen_oracle.py"
      provides: "gen_spectral_embedding + gen_spectral_clustering generators"
      contains: "gen_spectral_embedding"
  key_links:
    - from: "crates/mlrs-backend/src/prims/mod.rs"
      to: "crates/mlrs-backend/src/prims/laplacian.rs"
      via: "pub mod laplacian"
      pattern: "pub mod laplacian"
    - from: "crates/mlrs-algos/src/cluster/mod.rs"
      to: "crates/mlrs-algos/src/cluster/spectral_embedding.rs"
      via: "pub mod spectral_embedding"
      pattern: "pub mod spectral_embedding"
    - from: "crates/mlrs-py/src/lib.rs"
      to: "crates/mlrs-py/src/estimators/spectral.rs"
      via: "add_class::<PySpectralEmbedding>"
      pattern: "add_class::<PySpectralEmbedding>"
---

<objective>
Wave-0 scaffold for Phase 9 (Spectral Family), mirroring the 08-01 kernel_matrix
scaffold. Front-load ALL shared-file edits (error.rs, prims/mod.rs, cluster/mod.rs,
estimators/mod.rs, lib.rs, kernels lib.rs) plus every compiling stub and test file
into this one wave so Waves 1/2/3 are file-disjoint and parallel-safe.

This plan lands NO compute logic. It lands: the typed AlgoError guard
(NSamplesExceedsMaxDim, D-06), the laplacian prim signature with a REAL geometry
guard but a todo!() body, the laplacian_map kernel stub, the two estimator struct
homes + the PyO3 wrapper stubs, five #[ignore] Nyquist test scaffolds, and the two
new oracle generators (committed .npz blobs).

Purpose: every downstream wave edits only its own files; no shared-file contention.
Output: a compiling workspace (--features cpu) with all module homes registered and
all fixtures committed.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/09-spectral-family/09-CONTEXT.md
@.planning/phases/09-spectral-family/09-RESEARCH.md
@.planning/phases/09-spectral-family/09-PATTERNS.md
@.planning/phases/09-spectral-family/09-VALIDATION.md
@AGENTS.md

# Analog files to copy structure from (READ before editing):
@crates/mlrs-algos/src/error.rs
@crates/mlrs-backend/src/prims/kernel_matrix.rs
@crates/mlrs-kernels/src/elementwise.rs
@crates/mlrs-algos/src/cluster/kmeans.rs
@crates/mlrs-py/src/estimators/kernel.rs
@crates/mlrs-algos/tests/kernel_ridge_test.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Shared-file edits — error variant, module registrations, prim + kernel + estimator + pyclass stubs</name>
  <read_first>
    - crates/mlrs-algos/src/error.rs (InvalidK :99, InvalidGamma :260 — copy the struct-variant + #[error(...)] shape)
    - crates/mlrs-backend/src/prims/kernel_matrix.rs (:42-52 imports, :121 host signature, :133-142 geometry guard — copy for laplacian.rs)
    - crates/mlrs-backend/src/prims/mod.rs (:24 `pub mod kernel_matrix;`)
    - crates/mlrs-kernels/src/elementwise.rs (rbf_map :114 cube(launch) shape, div_by_row :301 gather-divisor) and lib.rs (:25 pub use block)
    - crates/mlrs-algos/src/cluster/mod.rs (:21-22 `pub mod dbscan; pub mod kmeans;`) and kmeans.rs (:76 struct, :112 new)
    - crates/mlrs-py/src/estimators/kernel.rs (any_estimator! invocations, #[pyclass] new/signature), estimators/mod.rs (:34 pub mod kernel), lib.rs (:139 use, :171-172 add_class)
  </read_first>
  <action>
    Land all shared-file edits and compiling stubs (NO compute):

    (a) error.rs — add the NSamplesExceedsMaxDim struct-variant per D-06, copying the
    InvalidK shape (struct variant + #[error(...)] message). Fields:
    `{ estimator: &'static str, n_samples: usize, max: usize }`. Message must name the
    dense eigensolver MAX_DIM cap (e.g. "n_samples = {n_samples} exceeds the dense
    eigensolver cap (must be <= {max} = MAX_DIM)"). Reuse existing InvalidK for
    n_clusters/n_neighbors and InvalidGamma for non-finite gamma — do NOT add new
    variants for those. Optionally add InvalidNNeighbors only if InvalidK's message
    does not fit n_neighbors semantics (planner discretion — prefer reuse).

    (b) prims/mod.rs — add `pub mod laplacian;` next to `pub mod kernel_matrix;`.

    (c) prims/laplacian.rs — create with the REAL host signature
    `pub fn laplacian<F, R>(pool, a: &DeviceArray<.., F>, n: usize) -> Result<(DeviceArray, DeviceArray), PrimError>`
    returning `(L, dd)` per RESEARCH Open Q2. The geometry guard is REAL (reject
    `a.len() != n*n`, `n == 0` with a typed PrimError, copying kernel_matrix.rs:133-142).
    The compute path is `todo!()` (filled in Wave 1 / Plan 09-02). Import
    `mlrs_kernels::laplacian_map`, `reduce::{row_reduce, ReducePath, ScalarOp}`. Do NOT
    add the n<=64 cap here — the cap is the estimator's job (D-06).

    (d) elementwise.rs — add the `laplacian_map` #[cube(launch)] stub copying the rbf_map
    signature shape with the gather-divisor pattern from div_by_row. Signature:
    `pub fn laplacian_map<F: Float + CubeElement>(a: &Array<F>, dd: &Array<F>, output: &mut Array<F>, n: u32)`.
    Stub body is bounds-checked but writes a placeholder (e.g. copies a through) — Wave 1
    fills the real -a/(dd_i*dd_j) + diagonal logic. MUST be SharedMemory-free, atomics-free,
    no infinity constant even in the stub (doc comment must avoid the literal tokens that
    trip grep gates per [08-02] Rule 3). Re-export via `pub use` in mlrs-kernels/src/lib.rs
    next to rbf_map.

    (e) cluster/mod.rs — add `pub mod spectral_embedding;` and `pub mod spectral_clustering;`
    plus `pub use` of each estimator type (mirror the dbscan/kmeans lines). Estimator plans
    edit this mod.rs only via their own files — but the module declarations land HERE in Wave 0.

    (f) cluster/spectral_embedding.rs — create with `pub struct SpectralEmbedding<F>` storing
    `n_components: usize` (default 2, D-08), `affinity: String` (default "nearest_neighbors",
    D-01), `gamma: Option<F>` (None→1/n_features at fit, D-04), `n_neighbors: usize` (default
    10, D-03), and a fitted `embedding_: Option<DeviceArray<.., F>>`. `new(...)` constructor.
    `fit`/`embedding_` accessor bodies are `todo!()` (Wave 2 / Plan 09-03). Mirror kmeans.rs
    struct/new shape and kernel_ridge gamma-Option storage.

    (g) cluster/spectral_clustering.rs — create with `pub struct SpectralClustering<F>` storing
    `n_clusters: usize` (default 8), `n_components: Option<usize>` (None→n_clusters, D-11),
    `affinity: String` (default "rbf", D-01), `gamma: F` (default 1.0 literal, D-04),
    `n_neighbors: usize` (default 10), `seed: u64`, fitted `labels_: Option<DeviceArray<.., i32>>`
    (the kmeans.rs i32 idiom). `new(...)` constructor; `fit`/`labels_` bodies `todo!()`
    (Wave 3 / Plan 09-04).

    (h) estimators/spectral.rs — create the PyO3 wrapper stub: two `crate::any_estimator!`
    invocations (AnySpectralEmbedding / AnySpectralClustering) per the 09-PATTERNS unfit
    field lists, two `#[pyclass(name="SpectralEmbedding")]` / `#[pyclass(name="SpectralClustering")]`
    with `#[new]` + `#[pyo3(signature=(...))]` carrying the sklearn defaults (SE
    n_components=2, affinity="nearest_neighbors", gamma=None, n_neighbors=10; SC n_clusters=8,
    affinity="rbf", gamma=1.0, n_neighbors=10, seed). fit/accessor method bodies are `todo!()`
    or return not_fitted (Wave 3 fills them). Copy kernel.rs structure verbatim.

    (i) estimators/mod.rs — add `pub mod spectral;` (copy the `pub mod kernel;` line).

    (j) lib.rs — add `use estimators::spectral::{PySpectralEmbedding, PySpectralClustering};`
    and the two `m.add_class::<...>()?;` registrations next to the kernel ones (:171-172).

    Do NOT place fenced code blocks in source comments that contain the literal grep-gate
    tokens; reword doc comments per [08-02] Rule 3 if needed.
  </action>
  <verify>
    <automated>cargo build --features cpu 2>&1 | tail -5 && grep -q "NSamplesExceedsMaxDim" crates/mlrs-algos/src/error.rs && grep -q "pub mod laplacian" crates/mlrs-backend/src/prims/mod.rs && grep -q "laplacian_map" crates/mlrs-kernels/src/lib.rs && grep -q "add_class::<PySpectralEmbedding>" crates/mlrs-py/src/lib.rs && echo SCAFFOLD_OK</automated>
  </verify>
  <acceptance_criteria>
    - `cargo build --features cpu` succeeds (stubs compile; todo!() bodies allowed).
    - AlgoError::NSamplesExceedsMaxDim exists with a MAX_DIM-naming message.
    - laplacian prim, laplacian_map kernel, both estimator structs, both pyclasses registered.
    - No literal SharedMemory/F::INFINITY tokens in the new kernel source (grep-clean).
  </acceptance_criteria>
  <done>Workspace compiles with --features cpu; all module homes and the error variant land; the new kernel stub is SharedMemory-free and infinity-free.</done>
</task>

<task type="auto">
  <name>Task 2: Five #[ignore] Nyquist test scaffolds + two oracle generators (committed .npz)</name>
  <read_first>
    - crates/mlrs-algos/tests/kernel_ridge_test.rs (load_npz :35, fixture() :49-56, host_to_f64/f64_to :58-72, assert_close :78, *_F32_BAND :47, skip_f64_with_log)
    - crates/mlrs-backend/tests/memory_gate_test.rs (PoolStats build-failing gate shape: BufferPool, live_bytes/peak_bytes/reuses assertions)
    - scripts/gen_oracle.py (gen_kernel_ridge :1384 region, the __main__ dtype loop near :1643-1648 — copy the np.random.default_rng/np.ascontiguousarray/np.savez/register pattern)
  </read_first>
  <action>
    Create the five test files as compiling #[ignore] Nyquist scaffolds and extend the
    oracle generator. Each test scaffold loads its fixture and asserts shape/finite ONLY
    (no compute symbols — they compile today against the Wave-0 stubs and are un-ignored by
    their owning wave):

    (a) crates/mlrs-backend/tests/laplacian_test.rs — three #[ignore] tests:
    `laplacian_value` (load fixture, assert L is n×n), `zero_degree` (load isolated-node
    fixture, assert no NaN/inf placeholder), `memory_gate` (BufferPool counters scaffold
    mirroring memory_gate_test.rs). Carry the skip_f64_with_log gate verbatim on the f64 case.

    (b) crates/mlrs-algos/tests/spectral_embedding_test.rs — four #[ignore] tests mapping to
    9-SE-01..04: `spectral_embedding` (rbf value-match), `knn_affinity` (nearest_neighbors
    default), `subspace` (degenerate-spectrum subspace test, D-09), `reject_oversize`
    (n>64 → AlgoError::NSamplesExceedsMaxDim). Include the *_F32_BAND const and
    skip_f64_with_log gate from kernel_ridge_test.rs.

    (c) crates/mlrs-algos/tests/spectral_clustering_test.rs — one #[ignore] test
    `spectral_clustering` (labels_ up to permutation; carry a best_match_accuracy/label_perm
    helper stub copied from kmeans_test.rs/dbscan_test.rs). EXACT labels — no band.

    (d) crates/mlrs-py/tests/spectral_smoke_test.rs — one #[ignore] smoke scaffold for
    fit + embedding_/labels_ accessors (f32+f64), f64 gated by backend_supports_f64().

    (e) scripts/gen_oracle.py — add `gen_spectral_embedding(seed, dtype)` and
    `gen_spectral_clustering(seed, dtype)`. CRITICAL (D-01): fit each sklearn estimator with
    its OWN DEFAULT CONSTRUCTOR — NO affinity/gamma override (the inverse of kernel_ridge's
    explicit kwargs). SE stores X + embedding_ (n_samples<=64, D-05; n_features such that the
    1/n_features gamma default is exercised). SC stores X + labels_ on a WELL-SEPARATED fixture
    (D-10) so the partition is unique up to permutation. Also add a degenerate-spectrum SE
    fixture (e.g. a symmetric block structure producing repeated eigenvalues) for the subspace
    test, and an isolated-node laplacian fixture for zero_degree. Register all in the __main__
    dtype loop. Run the generator in a /tmp venv (numpy+scipy+sklearn, PEP 668) and COMMIT the
    resulting .npz blobs to tests/fixtures/. Run the generator in isolation so other phases'
    blobs do not churn.
  </action>
  <verify>
    <automated>cargo test --features cpu -p mlrs-backend laplacian_test 2>&1 | tail -3 && cargo test --features cpu -p mlrs-algos spectral_embedding_test spectral_clustering_test 2>&1 | tail -3 && ls tests/fixtures/ | grep -E "spectral_(embedding|clustering)" && echo SCAFFOLD_TESTS_OK</automated>
  </verify>
  <acceptance_criteria>
    - All five test files compile; their #[ignore] tests are collected (0 run, N ignored).
    - Spectral .npz fixtures exist in tests/fixtures/ for both estimators (f32+f64) and were generated with each estimator's DEFAULT constructor (no override).
    - A degenerate-spectrum SE fixture and an isolated-node laplacian fixture are committed.
  </acceptance_criteria>
  <done>Five #[ignore] scaffolds compile and are collected; committed spectral .npz fixtures land using default constructors (D-01); degenerate + isolated-node fixtures present.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Python/host → estimator constructor & fit | Untrusted hyperparameters (n_samples, n_clusters, n_neighbors, gamma) cross here |
| host orchestration → device kernel launch | Buffer sizes / loop bounds cross into the CubeCL kernel |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-9-VAL | Tampering/DoS | error.rs + estimator fit entry | mitigate | Land the typed `AlgoError::NSamplesExceedsMaxDim` variant (D-06) in this wave so Waves 2/3 can reject `n_samples > 64` BEFORE any device allocation; reuse `InvalidK`/`InvalidGamma` for the other guards. |
| T-9-LAP | Tampering | laplacian_map kernel stub | mitigate | The stub is SharedMemory-free, atomics-free, and contains no infinity constant — the cpu-MLIR-safe profile is established at scaffold time so Wave 1 inherits it. |
| T-9-SC | Tampering | (none — no package installs this phase) | accept | No npm/pip/cargo installs; all deps are first-party workspace crates (RESEARCH Package Legitimacy Audit: N/A). |
</threat_model>

<verification>
- `cargo build --features cpu` green (whole workspace; todo!() bodies allowed in stubs).
- `cargo test --features cpu -p mlrs-backend laplacian_test` and the two mlrs-algos
  spectral test files compile and collect their #[ignore] scaffolds.
- error.rs grep shows NSamplesExceedsMaxDim; prims/mod.rs, cluster/mod.rs, estimators/mod.rs,
  lib.rs, kernels lib.rs all show the new registrations.
- tests/fixtures/ contains the spectral_embedding / spectral_clustering / degenerate /
  isolated-node .npz blobs.
</verification>

<success_criteria>
- Workspace compiles with --features cpu after the full scaffold.
- All shared-file edits (error.rs, both mod.rs files, lib.rs ×2, kernels lib.rs) are
  complete so Waves 1/2/3 are file-disjoint.
- Five #[ignore] Nyquist test scaffolds compile and are collected.
- Committed spectral .npz fixtures use each estimator's default constructor (D-01).
</success_criteria>

<output>
Create `.planning/phases/09-spectral-family/09-01-SUMMARY.md` when done.
</output>
