# Phase 1: Foundation — Oracle, Backend Abstraction, Arrow Bridge - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 delivers the testable skeleton every later primitive and estimator hangs off:
the five-crate Rust workspace (`mlrs-core`, `mlrs-kernels`, `mlrs-backend`, `mlrs-algos`,
`mlrs-py`), a generic `<F: Float>` compute spine over CubeCL runtimes, the scikit-learn
oracle harness, the Arrow zero-copy bridge with validation, the f64 capability gate, and
the mimalloc global allocator. A trivial end-to-end `#[cube]` kernel proves the whole
pipeline (Arrow ingest → device → kernel → read-back → oracle compare) on cpu and wgpu.

**Scope anchor:** This phase builds *infrastructure*, not algorithms. No estimators, no
real compute primitives (those are Phase 2+). The only kernel is a trivial smoke test.

</domain>

<decisions>
## Implementation Decisions

### Oracle harness mechanism
- **D-01:** scikit-learn reference values are **pre-generated as committed fixtures**, not
  computed live at test time. A checked-in `scripts/gen_oracle.py` produces reference
  outputs from seeded inputs; Rust tests load the committed files and compare.
- **D-02:** Fixture format is **NumPy `.npz`** (bundled named arrays per case, e.g.
  `linreg_f64_seed42.npz` carrying `X`, `y`, `coef_`, `intercept_`). Read in Rust via an
  npy/npz reader crate (latest).
- **D-03:** Fixtures are **committed binary blobs**; `scripts/gen_oracle.py` regenerates on
  demand. **CI runs Rust tests against the committed files — no Python/sklearn needed in the
  test job.** This keeps the oracle hermetic and reproducible (the project's "CPU oracle runs
  in CI without a GPU" goal extends to "without a Python env at test time").

### Device-array abstraction (FOUND-05)
- **D-04:** Build the **buffer-reuse/pool layer in Phase 1** (free-list/arena over CubeCL
  buffers), not just a thin wrapper — consistent with PROJECT.md's "memory efficiency is
  first-class, verified per phase, not deferred." `DeviceArray<R,F>` over the pool;
  zero-copy ingest from the validated Arrow bridge; host read-back.
- **D-05:** The pool exposes a **stats/counters API** (allocations, reuses, peak bytes).
  In Phase 1 these counters are **logged only** — hard reuse assertions are **deferred to
  Phase 2**, when real primitive workloads expose realistic allocation patterns. (Phase 1
  has only the trivial smoke kernel, so a hard reuse gate now would be testing an artifact.)

### Arrow bridge reject behavior (FOUND-06)
- **D-06:** **Hard-reject only.** Zero-copy ingest is the *only* path. Non-conforming input
  (non-zero offset / slice, set null bits, misaligned buffer) returns a typed `Err` **before
  any unsafe transmute**. No compacting-copy escape hatch in Phase 1 — keeps memory behavior
  obvious and the zero-copy contract unambiguous. Compaction is the caller's responsibility
  (Python/app layer).
- **D-07:** Bridge errors are a typed **`thiserror`** enum (`BridgeError`) with variants for
  each violation class (e.g. `HasNulls`, `Offset`, `Misaligned`).

### f32 tolerance policy (FOUND-08)
- **D-08:** Start with a **single global tolerance** (`F32_TOL` and `F64_TOL`, both
  `abs = 1e-5, rel = 1e-5`) rather than a per-family table. Split into per-estimator-family
  tolerances later (Phase 3/4/5) only if a family needs looser bounds. FOUND-08's
  "per-family policy" is satisfied by a policy *structure* that can grow rows; it does not
  require populated per-family values before any estimator exists.
- **D-09:** `assert_close` requires **both abs AND rel error to pass** (the stricter form),
  not numpy-style abs-OR-rel.
  - ⚠ **Implementation consideration for researcher/planner (not a re-opened decision):**
    "both must pass" is brittle for near-zero reference values — when `|expected| ≈ 0`, the
    relative term explodes even when the value is effectively correct. The harness should
    include a **near-zero guard** (e.g. fall back to abs-only when `|expected|` is below a
    small floor) so genuinely-correct near-zero results don't spuriously fail. Design this
    into `assert_close` from the start.

### Project-wide error handling (cross-cutting)
- **D-10:** Use **`thiserror`** for typed error enums in library crates
  (`mlrs-core`/`mlrs-kernels`/`mlrs-backend`/`mlrs-algos`) and **`anyhow`** at application /
  PyO3-binding boundaries (`mlrs-py`, binaries, `scripts`-driven Rust). **All Cargo
  dependencies track latest versions** — do not pin old versions.

### Claude's Discretion
- Choice of the trivial smoke-test kernel (e.g. SAXPY / elementwise add) — any minimal
  `#[cube]` kernel generic over `<F: Float>` that exercises the full pipeline.
- Specific npy/npz reader crate and Arrow crate versions (latest of each).
- Exact `BridgeError` variant names and the pool's internal data structure (free-list vs
  arena).
- f64 capability-gate skip-vs-xfail mechanics on wgpu adapters lacking `SHADER_F64`
  (roadmap requires "skip/xfail with a logged reason"; exact mechanism is Claude's).
- Near-zero guard floor value in `assert_close`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning context
- `.planning/PROJECT.md` — core value, constraints, out-of-scope, key decisions
- `.planning/REQUIREMENTS.md` — FOUND-01…FOUND-09 requirement text and traceability
- `.planning/ROADMAP.md` §"Phase 1" — goal + 5 success criteria (the gate for this phase)

### Build / kernel protocol (MANDATORY before writing any CubeCL code)
- `AGENTS.md` — source/test separation rule (no `mod tests` in source files; use `tests/`
  or `*_test.rs`); CubeCL generics-over-float requirement; build-error protocol
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — CubeCL manual
  index; read before writing kernels (generics, plane, shared memory, etc.)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` —
  MANDATORY troubleshooting reference on ANY CubeCL build/compile/feature/toolchain error

### Memory-efficiency guidance (informs device-array + pool + Arrow bridge)
- `/home/user/Documents/workspace/optimisor/manual/` — zero-copy Arrow↔CubeCL, zero-copy
  transmutation, jemalloc/mimalloc, smallvec/compact_str, Arrow dictionary/numeric handling

### Reference implementation (read-only — behavior/API reference, not code to port verbatim)
- `cuml-main/` — RAPIDS cuML v26.08.00; algorithm behavior + sklearn-compatible API surface
- `.planning/codebase/*.md` — codebase maps (ARCHITECTURE, CONVENTIONS, STACK, TESTING, etc.)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **None — greenfield.** No Rust workspace exists yet; this phase creates it from scratch.

### Established Patterns
- `cuml-main/` shows the cuML pattern being collapsed: C++/CUDA `libcuml++` kernels → thin
  Cython bindings → sklearn-compatible estimators (`Base`, `CumlArray`, `@reflect`
  output-type mirroring). mlrs collapses this into Rust core + CubeCL kernels + PyO3.
- sklearn-matching defaults are the contract (OLS=svd, KMeans=k-means++, TSVD=arpack,
  PCA with svd_flip) — NOT cuML defaults.

### Integration Points
- Arrow bridge is the single ingress for data into device buffers (zero-copy, validated).
- Capability layer gates f64 paths so f32 stays the portable wgpu baseline.
- Oracle harness is the universal verification gate consumed by every downstream phase.

</code_context>

<specifics>
## Specific Ideas

- Oracle fixture naming should encode case + dtype + seed (e.g. `linreg_f64_seed42.npz`)
  so cases are self-describing and reproducible.
- `assert_close` is the shared comparison entry point; sign-flip and label-permutation
  helpers (FOUND-08) wrap/feed into it for SVD/PCA and clustering respectively.

</specifics>

<deferred>
## Deferred Ideas

- **Per-estimator-family tolerance tables** — deferred from D-08; introduce in Phase 3/4/5
  when a family demonstrates it needs looser bounds than the global default.
- **Compacting-copy Arrow ingest path** — deferred from D-06; only add if a real caller
  cannot produce conforming (contiguous, null-free, aligned) arrays. Revisit in Phase 6
  (Python surface) if the PyCapsule path surfaces non-conforming inputs.
- **Hard buffer-reuse assertions** — deferred from D-05 to Phase 2, gated on realistic
  primitive allocation patterns.

</deferred>

---

*Phase: 1-Foundation — Oracle, Backend Abstraction, Arrow Bridge*
*Context gathered: 2026-06-11*
