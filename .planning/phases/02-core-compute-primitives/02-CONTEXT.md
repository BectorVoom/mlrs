# Phase 2: Core Compute Primitives - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 2 delivers four backend-portable CubeCL compute primitives — **GEMM**,
**reductions** (sum/mean/min/max/argmin/L2-norm), **pairwise squared-Euclidean
distance**, and **covariance / XᵀX (Gram)** — each generic over `<F: Float>`
and over the CubeCL runtime, validated *standalone* (f32 **and** f64; cpu **and**
wgpu) against a reference within the Phase 1 tolerance policy. Covers PRIM-01..04.

**Scope anchor:** This phase builds *trusted primitives*, not estimators. No
LinearRegression/PCA/KMeans/etc. (those are Phases 4–5), and no SVD/eig (that's
the Phase 3 hard gate). The goal is that downstream estimators reuse these
kernels rather than debugging linear algebra inside estimator code. The
primitives must compose on-device (the memory gate below proves it).

</domain>

<decisions>
## Implementation Decisions

### Reduction primitives — axis scope (PRIM-02)
- **D-01:** Reductions expose **full-array (1D total) AND axis-wise (2D
  row-reduce + column-reduce)** for sum/mean/min/max/L2-norm. Rationale: PCA
  needs column-means (centering), distance needs per-row L2-norms — building and
  validating the full axis surface here is exactly what "primitive-first" exists
  to do, so estimators inherit validated axis reductions.
- **D-02:** **argmin/argmax** deliver **full-array AND per-row argmin** over a 2D
  matrix (the op KMeans uses to assign each point to its nearest centroid in
  Phase 5). **Tie-break = lowest index** (matches numpy/sklearn `argmin`). The
  index-reduction kernel is built and validated here, not deferred into KMeans.
- **D-03 (carried, reaffirmed):** Each reduction must pass on wgpu via **both** a
  plane/subgroup path **and** a shared-memory fallback, with **no hardcoded plane
  width** (use `PLANE_DIM`), numerically stable on large inputs (roadmap
  criterion 2). The dual-path requirement implies hand-written kernels per the
  CubeCL plane/shared-memory manuals — a library that hides one path can't
  satisfy "both paths pass."

### Primitive API contract (consumed by Phase 4/5 estimators)
- **D-04:** **Matrix shape is passed explicitly as `(rows, cols)` per call.**
  `DeviceArray<R,F>` stays the flat 1D buffer from Phase 1 — its carried `len`
  remains the single source of truth for read-back size (preserves the T-04-01
  mitigation). Caller-side `rows*cols == len` is asserted. DeviceArray is **not**
  extended with 2D shape state.
- **D-05:** **Primitives take and return `DeviceArray` (device-resident
  in/out).** Chained calls (GEMM→reduce→distance) never round-trip to host. Thin
  host-slice helpers exist **only in tests** for oracle comparison, never on the
  primary API.
- **D-06:** **GEMM exposes BLAS-style `transa`/`transb` flags** so XᵀX (Gram /
  covariance = Aᵀ·A) reuses GEMM directly without materializing a transpose
  buffer. Depends on `cubecl-matmul` supporting transposed/strided operands —
  **researcher must confirm**; if unsupported, fallback is a dedicated transpose
  kernel + row-major multiply (see Open Questions).

### Pairwise distance primitive (PRIM-03)
- **D-07:** Distance is computed via **GEMM-expansion**: `‖x‖² + ‖y‖² − 2·XYᵀ`,
  reusing the GEMM (D-06) and the row-L2-norm reduction (D-01) this phase builds,
  then the **`max(d², 0)` clamp** (roadmap criterion 3 — the clamp exists because
  this method can yield small negatives under f32 cancellation). Direct
  difference-accumulation was rejected as the default (slower, reuses neither
  GEMM nor reductions).
- **D-08:** Primitive returns **squared distance as its core output** (matches
  the criterion + clamp; KMeans/DBSCAN compare in squared space), with an
  **optional sqrt** at the boundary for KNN's reported Euclidean distances
  (NEIGH-01 needs true sqrt distances within 1e-5). One validated kernel serves
  all three downstream consumers.

### Covariance / XᵀX (Gram) primitive (PRIM-04)
- **D-09:** Built **on GEMM** (roadmap), realized as `Aᵀ·A` via D-06's transpose
  flags. Convention (population vs sample normalization) is pinned by a committed
  numpy fixture — see D-12. Must reuse the GEMM output buffer per the memory gate
  (D-10).

### Memory-efficiency gate — HARD assertions (activates D-05 deferred from Phase 1)
- **D-10:** Phase 1 logged pool counters only and **deferred hard buffer-reuse
  assertions to Phase 2** (Phase 1 D-05). They now become a **build-failing
  gate** with three assertions:
  1. **Reuse > 0** — repeated same-shape primitive calls drive the pool's reuse
     counter up; allocation count stays **bounded** (not linear in call count).
  2. **No mid-pipeline host round-trip** — a chained pipeline (GEMM→reduce→
     distance) performs **zero** host read-backs between stages.
  3. **Gram reuses GEMM buffer** — covariance/XᵀX reuses the GEMM output buffer
     rather than allocating a parallel one.
  This satisfies PROJECT.md "memory efficiency is verified per phase, not
  deferred."
- **D-11:** To make reuse deterministic and testable, **primitives accept an
  optional caller-provided output `DeviceArray`** (reused across iterations —
  e.g. KMeans' per-iteration distance matrix) and **draw internal scratch from
  the `BufferPool`** (so scratch is metered and recycled). When no out-buffer is
  supplied, a fresh array is allocated and returned.

### Validation / oracle source for primitives
- **D-12:** **Hybrid reference.** Primary = a **live Rust host reference** (naive
  seeded-random CPU loops: triple-loop matmul, Σ reductions, direct distance) —
  hermetic, no Python, broad random-shape coverage, and literally the "host
  reference" the success criteria name. **Supplemented** by a **small set of
  committed numpy `.npz` convention fixtures** (reusing Phase 1's `gen_oracle.py`
  + npz-loader infra) that pin the exact conventions estimators inherit:
  **covariance normalization (`np.cov` ddof=1 vs population ddof=0), distance
  squared-vs-sqrt semantics, and GEMM**. The host reference catches arithmetic
  errors; the fixtures catch convention mismatches before an estimator phase
  does.

### Carried forward from Phase 1 (reaffirmed, not re-decided)
- **D-13:** Tolerance = `assert_close` with `F32_TOL`/`F64_TOL` = 1e-5 (abs
  **AND** rel, with the near-zero guard / `NEAR_ZERO_FLOOR`). f64 paths are
  **capability-gated via `skip_f64_with_log`** (skip-with-log, never fail) for
  wgpu backend portability. `mlrs-kernels` stays **feature-free** (`#[cube]`
  kernels generic over `<F: Float + CubeElement>`); launch wrappers + host
  orchestration live in **`mlrs-backend`** (owns the runtime). `thiserror` in
  libraries / `anyhow` at boundaries; all deps track latest (D-10 from Phase 1).

### Claude's Discretion
- Exact module/file layout within `mlrs-kernels` (kernel bodies) and
  `mlrs-backend` (launch wrappers + primitive host API) — e.g. a `prims` module
  vs per-primitive files. Honor AGENTS.md source/test separation (no in-source
  `mod tests`).
- Internal kernel tiling/block sizes, launch-config helpers, shared-memory tile
  dimensions (subject to the `PLANE_DIM` / no-hardcoded-width constraint, D-03).
- Specific `cubecl-matmul` API surface used for GEMM and the precise
  transpose-flag plumbing (subject to D-06 + the Open Question).
- Naming of new primitive error variants (extend the `thiserror` enums).
- The set of random shapes/seeds for the host-reference sweep, and which exact
  cases get committed numpy convention fixtures (subject to D-12 covering cov
  ddof, distance squared, GEMM).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning context
- `.planning/PROJECT.md` — core value, constraints, out-of-scope, key decisions
- `.planning/REQUIREMENTS.md` — PRIM-01…PRIM-04 requirement text + traceability
- `.planning/ROADMAP.md` §"Phase 2: Core Compute Primitives" — goal + 4 success
  criteria (the gate for this phase)
- `.planning/phases/01-foundation-oracle-backend-abstraction-arrow-bridge/01-CONTEXT.md`
  — Phase 1 decisions this phase builds on (oracle harness, DeviceArray/pool,
  tolerance policy, f64 gating); D-05 there explicitly deferred hard buffer-reuse
  assertions to **this** phase (now D-10 here)

### Build / kernel protocol (MANDATORY before writing any CubeCL code)
- `AGENTS.md` — source/test separation (no `mod tests` in source files; use
  `tests/` or `*_test.rs`); CubeCL generics-over-float requirement; build-error
  protocol
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — CubeCL
  manual index; read before writing kernels
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` — specifically the
  matmul/gemm, reduce, plane, shared-memory, generics, and dynamic-vectorization
  manuals (GEMM = cubecl-matmul; reductions = plane + shared-mem dual path)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md`
  — MANDATORY troubleshooting reference on ANY CubeCL build/compile/feature/
  toolchain error

### Memory-efficiency guidance (informs the D-10 memory gate + D-11 buffers)
- `/home/user/Documents/workspace/optimisor/manual/` — zero-copy Arrow↔CubeCL,
  zero-copy transmutation, buffer-reuse patterns

### Reference implementation (read-only — behavior/convention reference)
- `cuml-main/` — RAPIDS cuML v26.08.00; primitive/algorithm behavior reference
  (NOT code to port verbatim; numerical agreement is with scikit-learn/numpy)
- `.planning/codebase/*.md` — codebase maps (ARCHITECTURE, CONVENTIONS, STACK,
  TESTING, STRUCTURE, INTEGRATIONS, CONCERNS)

### Existing source this phase extends
- `crates/mlrs-kernels/src/` (`smoke.rs` saxpy = the `#[cube(launch)]` generic
  pattern to follow) — feature-free kernels
- `crates/mlrs-backend/src/` (`runtime.rs` = active client/runtime facade;
  `device_array.rs` = DeviceArray; `pool.rs` = BufferPool + counters;
  `capability.rs` = `skip_f64_with_log` / `feature_enabled`)
- `crates/mlrs-core/src/` (`compare.rs` `assert_close`; `tolerance.rs` TOL
  constants; `oracle.rs` npz loader; `sign_flip.rs`/`label_perm.rs` helpers)
- `crates/mlrs-backend/tests/spike_test.rs` — the proven launch/read-back idiom
  (`launch::<F, ActiveRuntime>`, `ArrayArg::from_raw_parts`, `read_one`)
- `crates/mlrs-core/examples/gen_fixture.rs` + `scripts/gen_oracle.py` — oracle
  fixture generation path for the D-12 convention fixtures (regen needs a /tmp
  venv with numpy per PEP 668)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`DeviceArray<R,F>` + `BufferPool` (Phase 1):** the device-resident in/out
  contract (D-05) and the memory gate (D-10/D-11) build directly on these. Pool
  already exposes allocation/reuse/peak-bytes counters — the gate asserts on
  them instead of just logging.
- **`saxpy_kernel` (smoke.rs):** the canonical `#[cube(launch)]` generic-over-`F`
  kernel + `launch::<F, ActiveRuntime>` idiom every primitive kernel follows.
- **Oracle harness (`mlrs-core`):** `assert_close`, TOL constants, npz loader —
  reused for D-12's convention fixtures; the host-reference path (D-12 primary)
  is new in-test Rust code compared against device output.
- **Capability gating (`skip_f64_with_log`):** the f64 skip-with-log pattern
  (D-13) wraps every f64 primitive oracle test.

### Established Patterns
- Feature-free kernels in `mlrs-kernels`; runtime-bound launch wrappers in
  `mlrs-backend` — the same split GEMM/reduce/distance/covariance must honor.
- sklearn/numpy conventions are the contract (covariance ddof, distance squared
  semantics), NOT cuML's — pinned by D-12 fixtures.
- A3 honest single-upload semantics (one host copy on ingest); CubeCL 0.10 has
  no in-place write into an `empty` handle — relevant to D-11 scratch handling.

### Integration Points
- GEMM is the shared substrate: distance (D-07) and covariance (D-09) both build
  on it — a correctness/perf issue in GEMM propagates, so GEMM is validated first.
- Reductions feed distance (row-L2-norm) and downstream PCA (column-mean).
- The memory gate (D-10) is the new per-phase verification surface that proves
  the device-resident composition contract (D-05) actually holds end-to-end.

</code_context>

<specifics>
## Specific Ideas

- The `max(d², 0)` clamp (D-07) is the visible signature of the GEMM-expansion
  distance method — keep it even where f64 makes negatives unlikely, so the f32
  path is correct by construction.
- argmin tie-break = lowest index (D-02) is a sklearn/numpy convention, not a
  free choice — KMeans label parity depends on it.
- Convention fixtures (D-12) should be minimal and self-describing (encode op +
  dtype + seed, mirroring Phase 1's `linreg_f64_seed42.npz` naming) — they pin
  conventions, not coverage; the host-reference sweep provides coverage.

</specifics>

<deferred>
## Deferred Ideas

- **Direct difference-accumulation distance kernel** — rejected as default
  (D-07); revisit only if f32 GEMM-expansion can't hold 1e-5 for some shape (then
  it becomes a fallback, not the primary path).
- **Extending `DeviceArray` to carry 2D shape** — deferred (D-04 keeps it flat);
  reconsider only if explicit `(rows, cols)` threading becomes error-prone across
  many primitives.
- **GEMM library fallback (transpose kernel + row-major multiply)** — only build
  if `cubecl-matmul` lacks transposed-operand support (the D-06 Open Question);
  not built speculatively.
- **Per-estimator-family tolerance tables** — still deferred from Phase 1 D-08 to
  Phase 3/4/5; the global 1e-5 TOL governs Phase 2 primitives.

## Open Questions for Research
- **Does `cubecl-matmul` (0.10) support transposed/strided operands** so D-06's
  `transa`/`transb` flags map onto it directly, or is a transpose kernel needed?
  (Highest-leverage research item — gates GEMM API + covariance reuse.)
- **Does `cubecl-matmul` support f64**, or only f32, on the wgpu backend? If f64
  is unsupported on wgpu, the f64 GEMM/distance/covariance paths fall under the
  existing `skip_f64_with_log` capability gate (D-13).
- **Reduction dual-path mechanics in CubeCL 0.10** — how to express the
  plane/subgroup path and the shared-memory fallback as selectable paths that can
  *both* be exercised in tests on wgpu (D-03).

</deferred>

---

*Phase: 2-Core Compute Primitives*
*Context gathered: 2026-06-12*
