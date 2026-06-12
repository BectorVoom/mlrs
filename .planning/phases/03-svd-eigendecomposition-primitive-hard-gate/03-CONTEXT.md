# Phase 3: SVD / Eigendecomposition Primitive (Hard Gate) - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 3 delivers a validated **SVD / symmetric-eigendecomposition** compute
primitive (PRIM-05): **two distinct Jacobi routines** — a general one-sided
Jacobi **SVD** and a separate classic Jacobi **symmetric-eigendecomposition** —
each generic over `<F: Float>` and over the CubeCL runtime, validated *standalone*
(f32 **and** f64; **cpu and rocm**) against a numpy reference + reference-free
algebraic invariants within the Phase-1 tolerance policy. The `svd_flip` sign
convention (already implemented in `mlrs-core/src/sign_flip.rs`) is applied at
oracle-comparison time.

**Scope anchor:** This phase builds *trusted primitives*, not estimators. No
PCA / TruncatedSVD / LinearRegression / Ridge (those are Phase 4). The goal is
that those four estimators reuse this validated SVD/eig rather than debugging
iterative linear algebra inside estimator code. This is the project's single
hardest, highest-leverage primitive and the gate for Phase 4.

**⚠ Gate-backend change (cross-cutting, decided this phase):** the GPU
correctness gate moves from **wgpu → rocm, project-wide**, as of Phase 3 (see
D-07). ROADMAP.md/PROJECT.md still document the old cpu+wgpu gate and need
reconciling.

</domain>

<decisions>
## Implementation Decisions

### Primitive surface & outputs
- **D-01:** **Two distinct routines**, not one unified core. (1) A general
  one-sided **Jacobi SVD** for arbitrary matrices. (2) A separate classic
  (two-sided) **Jacobi symmetric-eigendecomposition** for the covariance/Gram
  PCA-`full` path. The covariance matrix is PSD so its eig coincides with its
  SVD, but the user chose two routines so the eig path returns true signed
  eigenvalues directly and is reusable for any symmetric matrix. Two kernels to
  validate, each on its own correctness surface.
- **D-02:** **Thin / economy extent.** SVD returns U (m×k), S (k), Vᵀ (k×n) with
  `k = min(m, n)` — matching `numpy.linalg.svd(full_matrices=False)` / sklearn.
  Covers all v1 consumers: PCA components (rows of Vᵀ) + singular values +
  transform (U·S), TruncatedSVD, and the OLS/Ridge pseudo-inverse (U, S, V). Full
  square U (m×m) / V (n×n) is **not** built — the extra columns are unused and
  m×m is large for many-sample inputs.
- **D-03:** **Raw output; sign-align only at comparison.** The primitive returns
  vectors in whatever sign Jacobi produces. The existing `mlrs-core` `sign_flip`
  helper (`align_rows` / `align_sign`, largest-magnitude element made positive)
  canonicalizes **only at oracle-comparison time**, exactly as Phase 1 designed.
  The kernel stays pure; PCA in Phase 4 applies `svd_flip` itself to match
  sklearn. No device-side flip kernel.
- **D-04:** **Descending LAPACK/numpy ordering.** Singular values sorted
  descending (`S[0]` largest); eigenvalues/eigenvectors sorted by descending
  eigenvalue. Matches `np.linalg.svd` (always descending) and the sklearn
  PCA/TruncatedSVD contract. NB: `np.linalg.eigh` returns *ascending* — the eig
  primitive sorts descending on-device so estimators inherit the right order.

### Input-shape coverage
- **D-05:** **SVD handles tall AND wide.** Support `m ≥ n` and `m < n`. Standard
  trick: when `m < n`, run Jacobi on `Aᵀ` and swap U ↔ V. Makes the primitive
  shape-agnostic so PCA/TruncatedSVD can see wide inputs (n_features >
  n_samples). Shape passed explicitly as `(rows, cols)` per call (carries D-04
  from Phase 2 — `DeviceArray` stays flat 1D).
- **D-06:** **Eig assumes square symmetric; caller guarantees.** The symmetric-eig
  routine validates squareness but **trusts symmetry** — no defensive
  symmetrization step ((A+Aᵀ)/2). Its only v1 feeder is the Phase-2 covariance
  primitive, which is symmetric by construction.
- **D-08 (validation sweep):** Validation = **random sweep + degenerate cases**.
  Random well-conditioned tall/wide/square shapes (f32+f64, cpu+rocm) PLUS
  explicit hard cases that break iterative Jacobi/sign: **rank-deficient**
  (repeated / zero singular values), **near-identity**, and a **clustered-
  eigenvalue** matrix. Sizing = **mostly small + one moderate ~256×64 case** so
  the convergence loop and reduction reuse get exercised on the rocm GPU beyond
  toy sizes.

### Gate backend (cross-cutting — supersedes the project-wide wgpu gate)
- **D-07:** **GPU correctness gate = cpu + rocm, project-wide, from Phase 3
  onward.** wgpu drops to opportunistic. Rationale: this environment has a
  genuinely runnable ROCm stack — ROCm 7.1.1, `hipcc`, AMD `gfx1100` (RDNA3),
  `/dev/kfd` + `/dev/dri/renderD128`. gfx1100 supports **f64 natively**, so f64
  oracle paths that *skipped* on wgpu (no `SHADER_F64`) now actually **run** on
  rocm — a stronger gate. **Consequence:** `rocm` has been **compile-only /
  never executed** through Phase 2, so **ROCm bring-up is the first task** of
  this phase — confirm `ActiveRuntime` resolves to CubeCL's HIP runtime on
  gfx1100 and a trivial kernel runs end-to-end before any SVD work.
  **Reconciliation action:** ROADMAP.md + PROJECT.md document the cpu+wgpu gate
  and must be updated (orchestrator/planner action). The f64 capability gate
  (`skip_f64_with_log`) stays in place as the portable mechanism, but on rocm
  f64 is expected to RUN, not skip.

### Oracle & tolerance policy
- **D-09:** **Primary reference = numpy fixtures + reference-free invariants.**
  Committed `.npz` fixtures (`np.linalg.svd(full_matrices=False)` /
  `np.linalg.eigh`), reusing the Phase-1 `gen_oracle.py` + npz-loader infra,
  compared after `svd_flip` sign-alignment. PLUS hermetic algebraic invariants
  that need **no** reference: reconstruction `‖U·diag(S)·Vᵀ − A‖`, orthonormality
  `‖UᵀU − I‖` / `‖VᵀV − I‖`, and eig residual `‖A·v − λ·v‖`. A hand-written host
  Jacobi was **rejected** as primary (itself iterative/error-prone, would need
  its own validation); the invariants catch bugs the fixture's sign/order can't.
- **D-10:** **Hold global 1e-5; per-family looser bound only if forced.** Keep
  the global 1e-5 abs+rel (with near-zero floor) and the abs-OR-rel numpy-allclose
  style already used for f32 large-magnitude reductions. Introduce a documented
  per-family looser bound for singular vectors/values (the per-family tolerance
  table deferred from Phase 1 D-08 / Phase 2 D-13) **only if a real case can't
  hold 1e-5** — and record exactly which case forced it (ill-conditioned
  singular vectors of clustered/repeated eigenvalues are the likely candidate).
  Do not pre-loosen.

### Memory-efficiency gate — extends the D-10 build-failing gate to the iterative primitive
- **D-11:** **Extend the build-failing memory gate** (the Phase-2 `D-10`
  PoolStats gate) to SVD/eig with HARD assertions:
  1. **Bounded Jacobi scratch** — allocation count does **not** grow with
     sweep/iteration count; sweep scratch is drawn from `BufferPool` and recycled
     (carries Phase-2 D-11 optional-out + pooled-scratch).
  2. **Eig reuses the covariance/GEMM output buffer** rather than allocating a
     parallel matrix (mirrors Phase-2 D-10 gate 3).
  3. **No host round-trip between sweeps** — the convergence loop stays
     device-resident; only the final result is read back (mirrors Phase-2 gate 2
     `read_backs == 1`).
  Satisfies PROJECT.md "memory efficiency verified per phase, not deferred."
- **D-12:** **Convergence policy = fixed internal constants** (Claude/researcher
  discretion), NOT public API. The off-diagonal-norm threshold and max-sweep cap
  are internal constants chosen to hold the 1e-5 gate. A primitive is not an
  estimator — no hyperparameters. The only public contract is "matches reference
  within tolerance."

### Carried forward from Phases 1–2 (reaffirmed, not re-decided)
- **D-13:** Device-resident in/out (`DeviceArray`, Phase-2 D-05); explicit
  `(rows, cols)` per call (D-04); optional caller-out buffer + pooled scratch
  (D-11). Feature-free `#[cube]` kernels generic over `<F: Float + CubeElement>`
  in `mlrs-kernels`; launch wrappers + host orchestration in `mlrs-backend`
  (D-13). `assert_close` with 1e-5 abs+rel; `thiserror` in libs / `anyhow` at
  boundaries; deps track latest. Source/test separation per AGENTS.md (no
  in-source `mod tests`).

### Claude's Discretion
- The entire **Jacobi rotation-kernel design** — one-sided vs two-sided
  mechanics, rotation-pair scheduling (e.g. round-robin / chess-tournament
  parallel ordering), plane/subgroup vs shared-memory expression — is **deferred
  to the researcher** (this phase is roadmap-flagged NEEDS DEEPER RESEARCH).
- Module/file layout within `mlrs-kernels` (kernel bodies) and `mlrs-backend`
  (`prims/svd.rs`, `prims/eig.rs` or similar) — honor source/test separation.
- Internal convergence constants (D-12), sweep ordering, block/tile sizes
  (subject to no-hardcoded-plane-width carried from Phase 2 D-03).
- Exact random shapes/seeds for the sweep, and which cases get committed `.npz`
  fixtures vs invariant-only checks (subject to D-08 + D-09 coverage).
- Naming of new primitive error variants (extend the `thiserror` enums).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning context
- `.planning/PROJECT.md` — core value, constraints, out-of-scope, key decisions
  (NB: documents the cpu+wgpu gate — D-07 supersedes it with cpu+rocm; needs update)
- `.planning/REQUIREMENTS.md` — PRIM-05 requirement text + traceability
- `.planning/ROADMAP.md` §"Phase 3: SVD / Eigendecomposition Primitive (Hard
  Gate)" — goal + 3 success criteria (the gate for this phase) + the **NEEDS
  DEEPER RESEARCH** flag (run `/gsd-plan-phase --research-phase 3`)
- `.planning/phases/02-core-compute-primitives/02-CONTEXT.md` — Phase 2 decisions
  this phase builds on (GEMM transpose flags D-06, covariance D-09, device-
  resident D-05, the D-10/D-11 memory gate this phase extends, tolerance D-13)
- `.planning/phases/01-foundation-oracle-backend-abstraction-arrow-bridge/01-CONTEXT.md`
  — oracle harness, sign_flip helper (FOUND-08), capability gating, tolerance policy

### Build / kernel protocol (MANDATORY before writing any CubeCL code)
- `AGENTS.md` — source/test separation (no `mod tests` in source files; use
  `tests/` or `*_test.rs`); CubeCL generics-over-float requirement; build-error protocol
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — CubeCL
  manual index; read before writing kernels
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` — specifically the
  generics, plane/subgroup, shared-memory, and dynamic-vectorization manuals (the
  Jacobi sweep kernel needs plane/shared-memory patterns; no pre-built SVD primitive)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md`
  — MANDATORY troubleshooting reference on ANY CubeCL build/compile/feature/toolchain error
  (especially relevant: first real **HIP/ROCm** runtime bring-up, D-07)

### Memory-efficiency guidance (informs the D-11 memory gate)
- `/home/user/Documents/workspace/optimisor/manual/` — zero-copy Arrow↔CubeCL,
  buffer-reuse patterns (bounded iterative scratch, D-11)

### Reference implementation (read-only — behavior/convention reference)
- `cuml-main/` — RAPIDS cuML v26.08.00; SVD/eig solver behavior reference (NOT
  code to port verbatim; numerical agreement is with scikit-learn/numpy/LAPACK)
- `.planning/codebase/*.md` — codebase maps (ARCHITECTURE, CONVENTIONS, STACK,
  TESTING, STRUCTURE, INTEGRATIONS, CONCERNS)

### Existing source this phase extends
- `crates/mlrs-core/src/sign_flip.rs` — `canonical_sign` / `align_sign` /
  `align_rows` (svd_flip convention, applied at comparison per D-03)
- `crates/mlrs-backend/src/prims/` — `gemm.rs` (transpose flags, reused by SVD/eig
  scratch math), `reduce.rs` (plane+shared dual-path, off-diagonal norms),
  `covariance.rs` (the eig path's only feeder, D-06)
- `crates/mlrs-backend/src/{device_array.rs, pool.rs}` — DeviceArray + BufferPool
  + PoolStats counters (the D-11 memory gate asserts on these)
- `crates/mlrs-backend/src/{runtime.rs, capability.rs}` — `ActiveRuntime` facade
  (D-07: must resolve to the HIP runtime under `--features rocm`),
  `skip_f64_with_log` (expected to RUN, not skip, f64 on gfx1100)
- `crates/mlrs-backend/tests/memory_gate_test.rs` — the Phase-2 hard PoolStats gate
  the D-11 SVD/eig assertions extend
- `crates/mlrs-core/examples/gen_fixture.rs` + `scripts/gen_oracle.py` — oracle
  fixture generation path for D-09 numpy fixtures (regen needs a /tmp venv with
  numpy per PEP 668 — fixtures are committed blobs, not test-time)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`sign_flip.rs` (Phase 1):** `align_rows` canonicalizes each component's sign
  at comparison time — exactly the D-03 mechanism; no new flip code needed.
- **GEMM + reductions (Phase 2):** the SVD/eig sweep math (off-diagonal norms,
  rotation application, reconstruction-invariant `U·diag(S)·Vᵀ`) reuses the
  validated GEMM (transpose flags, D-06 P2) and dual-path reductions.
- **covariance primitive (Phase 2):** the symmetric-eig path's only feeder
  (square symmetric Gram, D-06).
- **DeviceArray + BufferPool + PoolStats:** the D-11 memory gate asserts on the
  existing counters; the bounded-scratch / optional-out contract (Phase-2 D-11)
  is the iterative-scratch discipline.
- **Oracle harness + npz loader:** reused for D-09 numpy fixtures; the
  reference-free invariants are new in-test Rust compared against device output.

### Established Patterns
- Feature-free kernels in `mlrs-kernels`; runtime-bound launch wrappers in
  `mlrs-backend/prims/` — SVD/eig honor the same split.
- numpy/LAPACK conventions are the contract (descending order D-04, thin
  extent D-02, svd_flip sign D-03), NOT cuML's.
- The per-phase build-failing memory gate (Phase-2 D-10) is the verification
  surface; this phase extends it to the iterative primitive (D-11).

### Integration Points
- **First real rocm execution (D-07):** `ActiveRuntime`/`capability` must resolve
  the HIP runtime on gfx1100 — bring-up precedes SVD work. This is the highest
  bring-up risk (the rocm feature has only ever compiled, never run).
- SVD/eig is the Phase-4 gate: PCA (eig + svd_flip), TruncatedSVD (thin SVD),
  LinearRegression/Ridge (SVD pseudo-inverse) all consume it. A
  correctness/convention bug here propagates to four estimators — hence the
  invariant + fixture double-check (D-09).

</code_context>

<specifics>
## Specific Ideas

- The Jacobi convergence loop must stay **device-resident** (D-11 gate 3) — the
  visible signature that the iterative primitive composes on-device like the
  Phase-2 pipeline; only the final U/S/Vᵀ (or eigenpairs) is read back.
- Reference-free invariants (D-09) are the strongest hermetic check for an
  iterative SVD: reconstruction + orthonormality + eig-residual catch arithmetic
  bugs independent of sign/order ambiguity, which the fixture comparison can't.
- f64 is expected to genuinely RUN on rocm/gfx1100 (D-07) — do not assume the
  wgpu skip-path; assert f64 results, don't silently skip them.

</specifics>

<deferred>
## Deferred Ideas

- **Per-estimator-family tolerance tables** — still deferred (Phase 1 D-08 →
  Phase 2 D-13 → here D-10); activates only if a real SVD/eig case can't hold
  1e-5, and only for the case that forces it.
- **Unified single Jacobi core (eig derived from SVD)** — rejected in favor of
  two distinct routines (D-01); revisit only if maintaining two kernels proves
  redundant.
- **Full U (m×m) / V (n×n) SVD** — deferred (D-02 thin only); build only if a
  future consumer needs the full orthonormal basis (no v1 estimator does).
- **Device-side svd_flip kernel** — deferred (D-03 aligns at comparison); only if
  an estimator needs already-canonicalized vectors on-device.
- **Defensive eig symmetrization ((A+Aᵀ)/2)** — deferred (D-06 trusts the
  covariance feeder); add only if a non-symmetric feeder appears.
- **wgpu as a gate** — dropped to opportunistic project-wide (D-07); rocm + cpu
  are the gate. Reconsider only if a target without ROCm becomes primary CI.

## Open Questions for Research (run `/gsd-plan-phase --research-phase 3`)
- **Jacobi rotation-kernel design in CubeCL 0.10** — one-sided SVD vs two-sided
  symmetric-eig sweep mechanics; parallel rotation-pair ordering (chess-
  tournament / round-robin) expressible in `#[cube]`; plane/subgroup vs
  shared-memory path for the off-diagonal sweep. Highest-leverage research item.
- **ROCm/HIP runtime bring-up (D-07):** does CubeCL's HIP runtime build+run on
  ROCm 7.1.1 / gfx1100 from `--features rocm`? Does `ActiveRuntime` resolve, does
  a trivial kernel + read-back work, does f64 run? Gates the entire phase gate.
- **Convergence constants (D-12):** off-diagonal-norm threshold + max-sweep cap
  that hold 1e-5 across the D-08 sweep (incl. degenerate cases) for f32 and f64.
- **Wide-matrix path (D-05):** confirm the Aᵀ-and-swap approach holds tolerance,
  or whether a dedicated wide kernel is needed.
- **Thin-SVD extraction (D-02):** how to recover thin U from the one-sided Jacobi
  (column normalization of A·V) without forming the full square factor.

</deferred>

---

*Phase: 3-SVD / Eigendecomposition Primitive (Hard Gate)*
*Context gathered: 2026-06-12*
